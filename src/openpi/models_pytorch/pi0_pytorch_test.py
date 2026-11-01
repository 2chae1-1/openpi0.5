"""CPU checks for optional vision pooling; no checkpoint download is needed."""

import dataclasses
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import openpi.models.gemma as _gemma
from openpi.models.pi0_config import Pi0Config
import openpi.models_pytorch.gemma_pytorch as _gemma_pytorch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


class _EmbeddingStub(nn.Module):
    """Accept precomputed image embeddings to check exact pooling values."""

    def embed_image(self, image):
        return image

    def embed_language_tokens(self, tokens):
        return tokens[..., None].expand(-1, -1, 4).to(torch.float32)


def _embedding_model(config):
    # Avoid constructing billion-parameter backbones for the arithmetic tests.
    model = PI0Pytorch.__new__(PI0Pytorch)
    nn.Module.__init__(model)
    model.config = config
    model.pi05 = config.pi05
    model.gradient_checkpointing_enabled = False
    model.paligemma_with_expert = _EmbeddingStub()
    return model.eval()


def _prefix_inputs(dtype=torch.float32):
    # Distinct rows, columns, features, batches and cameras expose accidental
    # vertical pooling, crossing row boundaries, or mixing camera streams.
    row = torch.arange(16).reshape(1, 16, 1, 1) * 100
    column = torch.arange(16).reshape(1, 1, 16, 1)
    feature = torch.arange(4).reshape(1, 1, 1, 4) * 10
    batch = torch.arange(2).reshape(2, 1, 1, 1) * 10000
    image = (row + column + feature + batch).reshape(2, 256, 4).to(dtype)
    images = [image, image + 20000, image + 40000]
    image_masks = [torch.tensor([True, True]), torch.tensor([True, False]), torch.tensor([False, False])]
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    token_masks = torch.tensor([[True, True, False], [True, False, False]])
    return images, image_masks, tokens, token_masks


def test_vision_pooling_defaults_to_disabled():
    assert Pi0Config().vision_token_pooling is False


@pytest.mark.parametrize("explicit_disabled", [False, True])
def test_disabled_pooling_matches_original_prefix(explicit_disabled):
    config = Pi0Config(pi05=True, pytorch_compile_mode=None)
    if explicit_disabled:
        config = dataclasses.replace(config, vision_token_pooling=False)
    model = _embedding_model(config)
    images, image_masks, tokens, token_masks = _prefix_inputs()

    embeddings, pad_masks, att_masks = model.embed_prefix(images, image_masks, tokens, token_masks)

    expected_embeddings = torch.cat([*images, tokens[..., None].expand(-1, -1, 4) * math.sqrt(4)], dim=1)
    expected_pad_masks = torch.cat([*(mask[:, None].expand(-1, 256) for mask in image_masks), token_masks], dim=1)
    torch.testing.assert_close(embeddings, expected_embeddings, rtol=0, atol=0)
    torch.testing.assert_close(pad_masks, expected_pad_masks)
    assert embeddings.shape == (2, 3 * 256 + 3, 4)
    assert att_masks.shape == pad_masks.shape
    assert not att_masks.any()


def test_pooling_horizontal_means_are_independent_per_camera_and_batch():
    model = _embedding_model(Pi0Config(pi05=True, vision_token_pooling=True, pytorch_compile_mode=None))
    images, image_masks, tokens, token_masks = _prefix_inputs()

    embeddings, pad_masks, att_masks = model.embed_prefix(images, image_masks, tokens, token_masks)

    assert embeddings.shape == (2, 3 * 128 + 3, 4)
    for camera, image in enumerate(images):
        expected = (image[:, 0::2] + image[:, 1::2]) / 2
        torch.testing.assert_close(embeddings[:, camera * 128 : (camera + 1) * 128], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            pad_masks[:, camera * 128 : (camera + 1) * 128], image_masks[camera][:, None].expand(-1, 128)
        )
    # An entirely masked camera still contributes 128 physical tokens.
    assert not pad_masks[:, 256:384].any()
    torch.testing.assert_close(embeddings[:, -3:], tokens[..., None].expand(-1, -1, 4).to(embeddings.dtype) * 2)
    torch.testing.assert_close(pad_masks[:, -3:], token_masks)
    assert att_masks.shape == pad_masks.shape
    assert not att_masks.any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pooling_preserves_image_embedding_dtype(dtype):
    model = _embedding_model(Pi0Config(pi05=True, vision_token_pooling=True, pytorch_compile_mode=None))
    # Keep both modalities in the requested dtype to avoid torch.cat promotion.
    model.paligemma_with_expert.embed_language_tokens = lambda tokens: tokens[..., None].expand(-1, -1, 4).to(dtype)
    images, image_masks, tokens, token_masks = _prefix_inputs(dtype)
    embeddings, _, _ = model.embed_prefix(images, image_masks, tokens, token_masks)
    assert embeddings.dtype == dtype
    assert embeddings.device.type == "cpu"


@pytest.mark.parametrize("token_count", [128, 255, 257])
def test_pooling_rejects_non_16_by_16_inputs(token_count):
    model = _embedding_model(Pi0Config(pi05=True, vision_token_pooling=True, pytorch_compile_mode=None))
    with pytest.raises(ValueError, match="256"):
        model.embed_prefix(
            [torch.zeros(1, token_count, 4)],
            [torch.ones(1, dtype=torch.bool)],
            torch.ones(1, 3, dtype=torch.long),
            torch.ones(1, 3, dtype=torch.bool),
        )


@pytest.mark.parametrize("pooling", [False, True])
def test_prefix_attention_and_position_ids_follow_physical_length(pooling):
    model = _embedding_model(Pi0Config(pi05=True, vision_token_pooling=pooling, pytorch_compile_mode=None))
    embeddings, pad_masks, att_masks = model.embed_prefix(*_prefix_inputs())
    tokens_per_camera = 128 if pooling else 256
    length = 3 * tokens_per_camera + 3

    attention = make_att_2d_masks(pad_masks, att_masks)
    positions = pad_masks.cumsum(dim=1) - 1
    assert attention.shape == (2, length, length)
    assert embeddings.shape[1] == length
    for batch, valid_images, valid_language in [(0, 2, 2), (1, 1, 1)]:
        valid_count = valid_images * tokens_per_camera + valid_language
        assert pad_masks[batch].sum() == valid_count
        torch.testing.assert_close(positions[batch, pad_masks[batch]], torch.arange(valid_count))
        assert attention[batch][pad_masks[batch]][:, pad_masks[batch]].all()
        assert not attention[batch][~pad_masks[batch]].any()
        assert not attention[batch][:, ~pad_masks[batch]].any()


@pytest.mark.parametrize("pooling", [False, True])
def test_embed_prefix_compiles_as_one_cpu_graph(pooling):
    model = _embedding_model(Pi0Config(pi05=True, vision_token_pooling=pooling, pytorch_compile_mode=None))
    inputs = _prefix_inputs()
    expected = model.embed_prefix(*inputs)
    # fullgraph rejects graph breaks; eager backend keeps this test CPU-only
    # and independent of a host C++ compiler or CUDA/Triton installation.
    compiled = torch.compile(model.embed_prefix, backend="eager", fullgraph=True)
    actual = compiled(*inputs)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.fixture
def tiny_model(monkeypatch):
    """Use the real patched Transformers path with small random backbones."""
    original_paligemma = _gemma_pytorch.PaliGemmaForConditionalGeneration
    original_gemma = _gemma_pytorch.GemmaForCausalLM

    def small_paligemma(config):
        config.text_config.vocab_size = 32
        config.image_token_index = 31
        config.projection_dim = 32
        config.hidden_size = 32
        config.vision_config.hidden_size = 16
        config.vision_config.intermediate_size = 32
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 2
        config.vision_config.projection_dim = 32
        return original_paligemma(config)

    def small_gemma(config):
        config.vocab_size = 32
        return original_gemma(config)

    monkeypatch.setattr(_gemma_pytorch, "PaliGemmaForConditionalGeneration", small_paligemma)
    monkeypatch.setattr(_gemma_pytorch, "GemmaForCausalLM", small_gemma)
    monkeypatch.setattr(
        _gemma,
        "get_config",
        lambda variant: _gemma.Config(width=32, depth=1, mlp_dim=64, num_heads=8, num_kv_heads=1, head_dim=4),
    )
    return PI0Pytorch(
        Pi0Config(pi05=True, dtype="float32", action_dim=4, action_horizon=3, pytorch_compile_mode=None)
    ).eval()


@pytest.mark.parametrize("pooling", [False, True])
def test_real_backbone_cache_masks_positions_and_denoising(tiny_model, monkeypatch, pooling):
    model = tiny_model
    model.config = dataclasses.replace(model.config, vision_token_pooling=pooling)
    tokens_per_camera = 128 if pooling else 256
    prefix_length = 3 * tokens_per_camera + 3
    horizon = model.config.action_horizon
    _, image_masks, tokens, token_masks = _prefix_inputs()
    observation = SimpleNamespace(
        images={name: torch.zeros(2, 3, 224, 224) for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        image_masks=dict(zip(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), image_masks, strict=True)),
        tokenized_prompt=tokens,
        tokenized_prompt_mask=token_masks,
        state=torch.zeros(2, 4),
        token_ar_mask=None,
        token_loss_mask=None,
    )
    weights_before = {name: value.clone() for name, value in model.state_dict().items()}
    original_forward = model.paligemma_with_expert.forward
    calls = []
    cached_tensors_before_denoising = []

    def record_forward(**kwargs):
        output, cache = original_forward(**kwargs)
        if cache is not None:
            cached_tensors_before_denoising.extend(tensor.clone() for tensor in [*cache.key_cache, *cache.value_cache])
        calls.append((kwargs, output, cache))
        return output, cache

    monkeypatch.setattr(model.paligemma_with_expert, "forward", record_forward)
    actions = model.sample_actions(torch.device("cpu"), observation, noise=torch.ones(2, horizon, 4), num_steps=2)

    assert actions.shape == (2, horizon, 4)
    assert torch.isfinite(actions).all()
    assert len(calls) == 3
    prefix, _, cache = calls[0]
    assert prefix["inputs_embeds"][0].shape == (2, prefix_length, 32)
    assert prefix["attention_mask"].shape == (2, 1, prefix_length, prefix_length)
    assert prefix["position_ids"].shape == (2, prefix_length)
    expected_pad_masks = torch.cat(
        [*(mask[:, None].expand(-1, tokens_per_camera) for mask in image_masks), token_masks], dim=1
    )
    torch.testing.assert_close(prefix["position_ids"], expected_pad_masks.cumsum(dim=1) - 1)
    expected_prefix_attention = expected_pad_masks[:, None, :, None] & expected_pad_masks[:, None, None, :]
    torch.testing.assert_close(prefix["attention_mask"] == 0, expected_prefix_attention)
    assert cache.get_seq_length() == prefix_length
    for key, value in zip(cache.key_cache, cache.value_cache, strict=True):
        assert key.shape[-2] == value.shape[-2] == prefix_length

    valid_prefix_counts = torch.tensor([2 * tokens_per_camera + 2, tokens_per_camera + 1])
    for kwargs, outputs, _ in calls[1:]:
        assert kwargs["past_key_values"] is cache
        assert kwargs["use_cache"] is False
        assert kwargs["attention_mask"].shape == (2, 1, horizon, prefix_length + horizon)
        expected_positions = valid_prefix_counts[:, None] + torch.arange(horizon)
        torch.testing.assert_close(kwargs["position_ids"], expected_positions)
        assert outputs[1].shape == (2, horizon, 32)
        prefix_visibility = kwargs["attention_mask"][:, 0, :, :prefix_length] == 0
        torch.testing.assert_close(prefix_visibility, expected_pad_masks[:, None].expand(-1, horizon, -1))
        assert (kwargs["attention_mask"][:, :, :, prefix_length:] == 0).all()

    # Denoising reuses the compressed prefix cache without appending suffixes,
    # and inference does not update any pretrained or projection parameters.
    assert cache.get_seq_length() == prefix_length
    for tensor, original in zip([*cache.key_cache, *cache.value_cache], cached_tensors_before_denoising, strict=True):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights_before[name], rtol=0, atol=0)
