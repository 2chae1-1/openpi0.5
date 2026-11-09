import dataclasses
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaModel

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel


class _TinyGemmaBackbone(PaliGemmaWithExpertModel):
    """Keep the production Gemma prefix/cache and expert paths with tiny CPU weights."""

    width = 8

    def __init__(self):
        nn.Module.__init__(self)
        self.paligemma = nn.Module()
        self.paligemma.language_model = self._gemma(use_adarms=False)
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = self._gemma(use_adarms=True)
        for layer in self.gemma_expert.model.layers:
            # Ensure denoising uses the attention output with the tiny initialized weights.
            nn.init.constant_(layer.input_layernorm.dense.bias[-self.width :], 0.5)
        self.prefix_call = None
        self.denoise_calls = []

    def _gemma(self, *, use_adarms):
        return GemmaModel(
            GemmaConfig(
                vocab_size=32,
                hidden_size=self.width,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                use_adarms=use_adarms,
                adarms_cond_dim=self.width if use_adarms else None,
                attn_implementation="eager",
            )
        )

    def embed_image(self, image):
        # Supply projected tokens without loading the production vision tower.
        return image[:, 0, 0, :1].reshape(-1, 1, 1).expand(-1, 256, self.width)

    def forward(self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, adarms_cond=None):
        if past_key_values is not None:
            self.denoise_calls.append((attention_mask, position_ids, past_key_values, past_key_values.get_seq_length()))
        outputs, cache = super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            adarms_cond=adarms_cond,
        )
        if use_cache:
            self.prefix_call = (attention_mask, position_ids, cache)
        return outputs, cache


@torch.no_grad()
def _run_split_style_inference(model, observation, noise):
    images, masks, tokens, lang_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    kept = [(image, mask) for image, mask in zip(images, masks, strict=True) if mask.any()]
    images, masks = zip(*kept, strict=True)
    assert len(images) == 2
    embs, pad, att = model.embed_prefix(images, masks, tokens, lang_masks, inference=True)
    attention_mask = model._prepare_attention_masks_4d(pi0_pytorch.make_att_2d_masks(pad, att))  # noqa: SLF001
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=attention_mask,
        position_ids=pad.cumsum(dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[embs, None],
        use_cache=True,
    )
    for timestep in (1.0, 0.5):
        noise = noise - 0.5 * model.denoise_step(state, pad, cache, noise, noise.new_full((noise.shape[0],), timestep))
    return noise


def _make_model(mode):
    model = pi0_pytorch.PI0Pytorch.__new__(pi0_pytorch.PI0Pytorch)
    nn.Module.__init__(model)
    model.config = Pi0Config(pi05=True, action_dim=2, action_horizon=3, pytorch_compile_mode=None)
    model.pi05 = True
    model.vision_token_pooling = mode
    model.gradient_checkpointing_enabled = False
    backbone = _TinyGemmaBackbone()
    model.paligemma_with_expert = backbone
    model.action_in_proj = nn.Linear(2, backbone.width)
    model.action_out_proj = nn.Linear(backbone.width, 2)
    model.time_mlp_in = nn.Linear(backbone.width, backbone.width)
    model.time_mlp_out = nn.Linear(backbone.width, backbone.width)
    model.eval()
    return model


def _make_observation(prompt_mask):
    batch_size, prompt_length = prompt_mask.shape
    return SimpleNamespace(
        images={
            "base_0_rgb": torch.ones(batch_size, 3, 224, 224),
            "left_wrist_0_rgb": -torch.ones(batch_size, 3, 224, 224),
            "right_wrist_0_rgb": torch.zeros(batch_size, 3, 224, 224),
        },
        image_masks={
            "base_0_rgb": torch.ones(batch_size, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool),
        },
        state=torch.zeros(batch_size, 2),
        tokenized_prompt=(torch.arange(prompt_length) % 32).expand(batch_size, -1),
        tokenized_prompt_mask=prompt_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )


def _run_inference(model, observation, noise, path):
    if path == "split-style":
        return _run_split_style_inference(model, observation, noise)
    return model.sample_actions(
        torch.device("cpu"),
        observation,
        noise=noise,
        num_steps=2,
        skip_masked_views=path == "sample-actions-skip-empty",
    )


@pytest.mark.parametrize(("mode", "image_tokens"), [("none", 512), ("avg", 384), ("max", 384)])
@pytest.mark.parametrize("path", ["split-style", "sample-actions", "sample-actions-skip-empty"])
def test_libero_prefix_cache_and_denoising_positions(mode, image_tokens, path):
    model = _make_model(mode)
    backbone = model.paligemma_with_expert
    # Even when enabled, the option is a no-op for batch size two.
    model.config = dataclasses.replace(model.config, trim_language_padding=True)
    valid_prompt_tokens = torch.tensor([17, 9])
    prompt_mask = torch.arange(200)[None, :] < valid_prompt_tokens[:, None]
    actions = _run_inference(model, _make_observation(prompt_mask), torch.zeros(2, 3, 2), path)
    assert actions.shape == (2, 3, 2)
    assert torch.isfinite(actions).all()

    inactive_image_tokens = 256 if path == "sample-actions" else 0
    prefix_len = image_tokens + inactive_image_tokens + 200
    prefix_pad = torch.cat(
        (
            torch.ones(2, image_tokens, dtype=torch.bool),
            torch.zeros(2, inactive_image_tokens, dtype=torch.bool),
            prompt_mask,
        ),
        dim=1,
    )
    prefix_mask, prefix_positions, cache = backbone.prefix_call
    assert prefix_mask.shape == (2, 1, prefix_len, prefix_len)
    torch.testing.assert_close(prefix_mask[:, 0] == 0, prefix_pad[:, :, None] & prefix_pad[:, None, :])
    torch.testing.assert_close(prefix_positions, prefix_pad.cumsum(dim=1) - 1)
    assert cache.get_seq_length() == prefix_len
    assert cache[0][0].shape == cache[0][1].shape == (2, 1, prefix_len, 4)

    assert len(backbone.denoise_calls) == 2
    expected_positions = image_tokens + valid_prompt_tokens[:, None] + torch.arange(3)
    expected_suffix_mask = torch.cat((prefix_pad, torch.ones(2, 3, dtype=torch.bool)), dim=1)
    for mask, positions, incoming_cache, cache_length in backbone.denoise_calls:
        assert incoming_cache is cache
        assert cache_length == prefix_len
        assert mask.shape == (2, 1, 3, prefix_len + 3)
        torch.testing.assert_close(mask[:, 0] == 0, expected_suffix_mask[:, None, :].expand(-1, 3, -1))
        torch.testing.assert_close(positions, expected_positions)


@pytest.mark.parametrize("path", ["split-style", "sample-actions-skip-empty"])
@pytest.mark.parametrize("prompt_end", [0, 17, 200])
def test_trim_language_padding_preserves_cache_and_actions(path, prompt_end):
    model = _make_model("max")
    backbone = model.paligemma_with_expert
    prompt_mask = torch.arange(200)[None, :] < prompt_end
    observation = _make_observation(prompt_mask)
    noise = torch.linspace(-1, 1, 6).reshape(1, 3, 2)

    assert not model.config.trim_language_padding
    baseline_actions = _run_inference(model, observation, noise, path)
    baseline_mask, baseline_positions, baseline_cache = backbone.prefix_call
    baseline_denoise_calls = backbone.denoise_calls.copy()
    backbone.denoise_calls.clear()

    model.config = dataclasses.replace(model.config, trim_language_padding=True)
    actions = _run_inference(model, observation, noise, path)
    mask, positions, cache = backbone.prefix_call
    prefix_len = 384 + prompt_end
    assert baseline_cache.get_seq_length() == 584
    assert cache.get_seq_length() == prefix_len
    assert mask.shape == (1, 1, prefix_len, prefix_len)
    torch.testing.assert_close(mask, baseline_mask[:, :, :prefix_len, :prefix_len])
    torch.testing.assert_close(positions, baseline_positions[:, :prefix_len])
    for baseline_layer, layer in zip(baseline_cache, cache, strict=True):
        for baseline_tensor, tensor in zip(baseline_layer, layer, strict=True):
            assert tensor.shape == (1, 1, prefix_len, 4)
            torch.testing.assert_close(tensor, baseline_tensor[:, :, :prefix_len, :])

    assert len(backbone.denoise_calls) == 2
    for baseline_call, trimmed_call in zip(baseline_denoise_calls, backbone.denoise_calls, strict=True):
        baseline_suffix_mask, baseline_suffix_positions, _, _ = baseline_call
        suffix_mask, suffix_positions, incoming_cache, cache_length = trimmed_call
        assert incoming_cache is cache
        assert cache_length == prefix_len
        expected_mask = torch.cat((baseline_suffix_mask[..., :prefix_len], baseline_suffix_mask[..., 584:]), dim=-1)
        torch.testing.assert_close(suffix_mask, expected_mask)
        torch.testing.assert_close(suffix_positions, baseline_suffix_positions)
        torch.testing.assert_close(suffix_positions, prefix_len + torch.arange(3)[None, :])
    torch.testing.assert_close(actions, baseline_actions, rtol=1e-5, atol=1e-6)


def test_trim_language_padding_keeps_interior_masked_slots_and_training_shape(monkeypatch):
    model = _make_model("max")
    model.config = dataclasses.replace(model.config, trim_language_padding=True)
    observation = _make_observation(torch.tensor([[True, False, True, False]]))
    images, masks, tokens, lang_masks, _ = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    full_embs, _, _ = model.embed_prefix(images, masks, tokens, lang_masks)
    embs, pad, _ = model.embed_prefix(images, masks, tokens, lang_masks, inference=True)
    assert pad.shape[1] == 640 + 3
    torch.testing.assert_close(pad[:, -3:], lang_masks[:, :3])
    torch.testing.assert_close(embs, full_embs[:, :-1])

    # Inspect the prefix emitted by the training entry point before joint attention.
    def training_backbone(*, inputs_embeds, **kwargs):
        assert inputs_embeds[0].shape[1] == 640 + 4
        return inputs_embeds, None

    model.train()
    monkeypatch.setattr(model.paligemma_with_expert, "forward", training_backbone)
    actions = torch.zeros(1, 3, 2)
    loss = model(observation, actions, noise=torch.ones_like(actions), time=torch.tensor([0.5]))
    assert loss.shape == actions.shape
