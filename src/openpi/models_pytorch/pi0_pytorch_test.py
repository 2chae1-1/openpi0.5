"""CPU coverage for camera selection and the existing prefix/cache plumbing.

The tiny deterministic backbone replaces model weights, not PI0's embedding,
attention-mask, position-ID, sampling, or denoising methods.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch import preprocessing_pytorch


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_calls = []
        self.forward_calls = []
        self.paligemma = SimpleNamespace(
            language_model=SimpleNamespace(
                config=SimpleNamespace(),
                layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=nn.Linear(4, 4)))],
            )
        )
        self.gemma_expert = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace()))

    def embed_image(self, image):
        self.image_calls.append(image)
        intensity = image.mean(dim=tuple(range(1, image.ndim)))[:, None, None]
        return intensity + torch.arange(8, dtype=image.dtype).reshape(1, 2, 4) / 20

    def embed_language_tokens(self, tokens):
        return tokens[:, :, None].float() / 10 + torch.arange(4).float() / 20

    @torch.compiler.disable
    def _record_forward(self, attention_mask, position_ids, past_key_values, use_cache, prefix_length):
        # Keep fixture-only Python instrumentation outside tensor graphs. Model
        # arithmetic before and after this recording boundary is still compiled.
        self.forward_calls.append(
            SimpleNamespace(
                mask=attention_mask.clone(),
                positions=position_ids.clone(),
                cache=past_key_values,
                use_cache=use_cache,
                prefix_length=prefix_length,
            )
        )

    def forward(self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, adarms_cond=None):
        prefix, suffix = inputs_embeds
        hidden = (
            torch.cat([prefix, suffix], dim=1)
            if prefix is not None and suffix is not None
            else (prefix if prefix is not None else suffix)
        )
        queries = (hidden + position_ids[:, :, None] / 100)[:, None]
        keys = queries / 2
        values = hidden[:, None]
        if past_key_values is not None:
            keys = torch.cat([past_key_values[0], keys], dim=2)
            values = torch.cat([past_key_values[1], values], dim=2)
        self._record_forward(
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            None if prefix is None else prefix.shape[1],
        )
        probabilities = torch.softmax(queries @ keys.transpose(-1, -2) / 2 + attention_mask, dim=-1)
        output = (probabilities @ values)[:, 0] + hidden
        if prefix is not None and suffix is not None:
            outputs = [output[:, : prefix.shape[1]], output[:, prefix.shape[1] :]]
        else:
            outputs = [output, None] if prefix is not None else [None, output]
        return outputs, (keys, values) if use_cache else None


def _make_model(*, enabled=False):
    # Avoid allocating production SigLIP/Gemma weights or downloading a checkpoint.
    model = pi0_pytorch.PI0Pytorch.__new__(pi0_pytorch.PI0Pytorch)
    nn.Module.__init__(model)
    model.config = Pi0Config(pi05=True, action_dim=2, action_horizon=3, pytorch_compile_mode=None)
    model.config = dataclasses.replace(model.config, pytorch_skip_inactive_images=enabled)
    model.pi05 = True
    model.gradient_checkpointing_enabled = False
    model.paligemma_with_expert = _TinyBackbone()
    model.action_in_proj = nn.Linear(2, 4)
    model.action_out_proj = nn.Linear(4, 2)
    model.time_mlp_in = nn.Linear(4, 4)
    model.time_mlp_out = nn.Linear(4, 4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.05)
    return model.eval()


def _prefix_inputs(camera_masks):
    batch_size = len(camera_masks[0]) if camera_masks else 2
    images = [torch.full((batch_size, 3, 2, 2), float(index)) for index in range(len(camera_masks))]
    masks = [torch.tensor(mask, dtype=torch.bool) for mask in camera_masks]
    tokens = torch.tensor([[1, 2, 0], [3, 0, 0]])[:batch_size]
    language_masks = torch.tensor([[True, True, False], [True, False, False]])[:batch_size]
    return images, masks, tokens, language_masks


def _observation(camera_masks):
    images, masks, tokens, language_masks = _prefix_inputs(camera_masks)
    keys = preprocessing_pytorch.IMAGE_KEYS
    return SimpleNamespace(
        images={key: image[:, :, :1, :1].expand(-1, -1, 224, 224) for key, image in zip(keys, images, strict=True)},
        image_masks=dict(zip(keys, masks, strict=True)),
        state=torch.zeros(2, 2),
        tokenized_prompt=tokens,
        tokenized_prompt_mask=language_masks,
        token_ar_mask=None,
        token_loss_mask=None,
    )


def test_camera_skipping_is_disabled_by_default():
    assert Pi0Config(pi05=True).pytorch_skip_inactive_images is False
    model = _make_model()
    inputs = _prefix_inputs([[True, True], [True, True], [False, False]])
    embeddings, masks, attention = model.embed_prefix(*inputs)
    assert len(model.paligemma_with_expert.image_calls) == 3
    assert embeddings.shape == (2, 9, 4)
    assert masks.shape == attention.shape == (2, 9)
    assert not masks[:, 4:6].any()


def test_camera_skipping_rejects_non_pi05_pytorch_models():
    config = Pi0Config(pytorch_skip_inactive_images=True, pytorch_compile_mode=None)
    with pytest.raises(ValueError, match="pi05"):
        pi0_pytorch.PI0Pytorch(config)


@pytest.mark.parametrize(
    ("camera_masks", "expected_calls"),
    [
        ([[True, True], [True, True], [True, True]], 3),
        ([[True, True], [True, True], [False, False]], 2),
        ([[False, False], [True, True], [False, True]], 2),
        ([[False, False], [False, False], [False, False]], 0),
        ([], 0),
    ],
)
def test_only_completely_inactive_cameras_are_skipped(camera_masks, expected_calls):
    baseline = _make_model()
    optimized = _make_model(enabled=True)
    inputs = _prefix_inputs(camera_masks)
    original = baseline.embed_prefix(*inputs)
    actual = optimized.embed_prefix(*inputs)
    # Camera zero has zero-valued pixels but must stay whenever its mask is valid.
    retained = [index for index, mask in enumerate(camera_masks) if any(mask)]
    kept_tokens = [token for index in retained for token in (2 * index, 2 * index + 1)]
    kept_tokens += list(range(2 * len(camera_masks), 2 * len(camera_masks) + 3))
    assert len(optimized.paligemma_with_expert.image_calls) == expected_calls
    for observed, expected in zip(actual, original, strict=True):
        torch.testing.assert_close(observed, expected[:, kept_tokens])


@pytest.mark.parametrize("enabled", [False, True])
def test_mismatched_image_and_mask_lists_raise(enabled):
    model = _make_model(enabled=enabled)
    images, masks, tokens, language_masks = _prefix_inputs([[False, False]])
    with pytest.raises(ValueError, match="zip"):
        model.embed_prefix(images, masks + masks, tokens, language_masks)
    with pytest.raises(ValueError, match="zip"):
        model.embed_prefix(images + images, masks, tokens, language_masks)


@pytest.mark.parametrize("inactive_camera", [0, 1, 2])
def test_attention_and_position_ids_match_retained_baseline_tokens(inactive_camera):
    baseline = _make_model()
    optimized = _make_model(enabled=True)
    camera_masks = [[True, True], [True, False], [False, True]]
    camera_masks[inactive_camera] = [False, False]
    inputs = _prefix_inputs(camera_masks)
    _, original_pad, original_ar = baseline.embed_prefix(*inputs)
    _, actual_pad, actual_ar = optimized.embed_prefix(*inputs)
    suffix = baseline.embed_suffix(torch.zeros(2, 2), torch.zeros(2, 3, 2), torch.ones(2))
    original_pad = torch.cat([original_pad, suffix[1]], dim=1)
    actual_pad = torch.cat([actual_pad, suffix[1]], dim=1)
    original_ar = torch.cat([original_ar, suffix[2]], dim=1)
    actual_ar = torch.cat([actual_ar, suffix[2]], dim=1)
    kept = [index for index in range(12) if index not in (2 * inactive_camera, 2 * inactive_camera + 1)]
    original_attention = pi0_pytorch.make_att_2d_masks(original_pad, original_ar)
    actual_attention = pi0_pytorch.make_att_2d_masks(actual_pad, actual_ar)
    expected_attention = original_attention[:, kept][:, :, kept]
    torch.testing.assert_close(actual_attention, expected_attention)
    torch.testing.assert_close(
        optimized._prepare_attention_masks_4d(actual_attention),  # noqa: SLF001
        baseline._prepare_attention_masks_4d(expected_attention),  # noqa: SLF001
    )
    torch.testing.assert_close(actual_pad.cumsum(1) - 1, (original_pad.cumsum(1) - 1)[:, kept])


@pytest.mark.parametrize(
    "camera_masks",
    [
        [[True, True], [True, True], [False, False]],
        [[True, False], [False, False], [False, True]],
        [[False, False], [False, False], [False, False]],
    ],
)
def test_sampling_preserves_actions_and_cache_mask_position_alignment(camera_masks):
    baseline = _make_model()
    optimized = _make_model(enabled=True)
    observation = _observation(camera_masks)
    noise = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 12
    expected = baseline.sample_actions(torch.device("cpu"), observation, noise=noise.clone(), num_steps=2)
    actual = optimized.sample_actions(torch.device("cpu"), observation, noise=noise.clone(), num_steps=2)
    torch.testing.assert_close(actual, expected)
    original_calls = baseline.paligemma_with_expert.forward_calls
    actual_calls = optimized.paligemma_with_expert.forward_calls
    retained = [index for index, mask in enumerate(camera_masks) if any(mask)]
    kept = [token for index in retained for token in (2 * index, 2 * index + 1)] + [6, 7, 8]
    prefix_length = len(kept)
    assert len(actual_calls) == len(original_calls) == 3
    assert actual_calls[0].use_cache
    assert actual_calls[0].prefix_length == prefix_length
    torch.testing.assert_close(actual_calls[0].positions, original_calls[0].positions[:, kept])
    torch.testing.assert_close(actual_calls[0].mask, original_calls[0].mask[:, :, kept][:, :, :, kept])
    suffix_columns = [*kept, 9, 10, 11]
    for call, original in zip(actual_calls[1:], original_calls[1:], strict=True):
        assert not call.use_cache
        assert call.cache[0].shape == call.cache[1].shape == (2, 1, prefix_length, 4)
        assert call.mask.shape == (2, 1, 3, prefix_length + 3)
        torch.testing.assert_close(call.positions, original.positions)
        torch.testing.assert_close(call.mask, original.mask[:, :, :, suffix_columns])
        torch.testing.assert_close(call.cache[0], original.cache[0][:, :, kept])
        torch.testing.assert_close(call.cache[1], original.cache[1][:, :, kept])
    # Denoising reuses the same prefix cache without growing it between steps.
    assert actual_calls[1].cache is actual_calls[2].cache


def test_training_forward_preserves_loss_and_active_image_gradients(monkeypatch):
    baseline = _make_model()
    optimized = _make_model(enabled=True)
    images, masks, tokens, language_masks = _prefix_inputs([[True, True], [False, False], [False, True]])
    images = [image.requires_grad_() for image in images]
    state = torch.zeros(2, 2)
    # Isolate model arithmetic from stochastic image augmentation.
    for model in (baseline, optimized):
        monkeypatch.setattr(
            model,
            "_preprocess_observation",
            lambda observation, **kwargs: (images, masks, tokens, language_masks, state),
        )
    actions = torch.zeros(2, 3, 2)
    noise = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 12
    time = torch.tensor([0.3, 0.7])
    expected = baseline(None, actions, noise=noise, time=time)
    actual = optimized(None, actions, noise=noise, time=time)
    torch.testing.assert_close(actual, expected)
    expected.sum().backward()
    expected_image_gradients = [image.grad.clone() for image in images]
    for image in images:
        image.grad = None
    actual.sum().backward()
    assert images[0].grad is not None
    assert images[1].grad is None
    assert images[2].grad is not None
    torch.testing.assert_close(images[0].grad, expected_image_gradients[0])
    torch.testing.assert_close(expected_image_gradients[1], torch.zeros_like(images[1]))
    torch.testing.assert_close(images[2].grad, expected_image_gradients[2])
    assert optimized.action_in_proj.weight.grad is not None
    assert optimized.action_out_proj.weight.grad is not None
    for name, parameter in optimized.named_parameters():
        reference = dict(baseline.named_parameters())[name]
        if reference.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, reference.grad)


def test_compiled_prefix_rechecks_masks_on_every_call():
    model = _make_model(enabled=True)
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    compiled = torch.compile(model.embed_prefix, backend=backend, fullgraph=False)
    layouts = [
        [[True, True], [True, True], [False, False]],
        [[True, True], [True, True], [False, True]],
        [[True, True], [True, True], [False, False]],
    ]
    for camera_masks, expected_length in zip(layouts, [7, 9, 7], strict=True):
        inputs = _prefix_inputs(camera_masks)
        actual = compiled(*inputs)
        expected = _make_model(enabled=True).embed_prefix(*inputs)
        assert actual[0].shape[1] == expected_length
        for observed, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(observed, reference)
    assert graphs


def test_compiled_sampling_rechecks_masks_and_preserves_actions():
    model = _make_model(enabled=True)
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    # Returning gm.forward exercises Dynamo capture and resumption like the eager
    # backend; this test makes no assertion about Inductor or GPU performance.
    compiled = torch.compile(model.sample_actions, backend=backend, fullgraph=False)
    layouts = [
        [[True, True], [True, True], [False, False]],
        [[True, True], [True, True], [False, True]],
        [[True, True], [True, True], [False, False]],
    ]
    noise = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 12
    for camera_masks, prefix_length in zip(layouts, [7, 9, 7], strict=True):
        observation = _observation(camera_masks)
        model.paligemma_with_expert.image_calls.clear()
        model.paligemma_with_expert.forward_calls.clear()
        actual = compiled(torch.device("cpu"), observation, noise=noise.clone(), num_steps=2)
        expected = _make_model(enabled=True).sample_actions(
            torch.device("cpu"), observation, noise=noise.clone(), num_steps=2
        )
        torch.testing.assert_close(actual, expected)
        calls = model.paligemma_with_expert.forward_calls
        assert len(calls) == 3
        assert calls[0].prefix_length == prefix_length
        assert calls[0].mask.shape == (2, 1, prefix_length, prefix_length)
        for call in calls[1:]:
            assert call.cache[0].shape[2] == prefix_length
            assert call.mask.shape == (2, 1, 3, prefix_length + 3)
    assert graphs
