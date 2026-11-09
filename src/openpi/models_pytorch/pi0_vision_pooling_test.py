from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import pi0_pytorch


class _ProjectedTokens(nn.Module):
    def embed_image(self, image):
        patches = torch.arange(256, dtype=image.dtype, device=image.device)
        # Opposite channels ensure max selects values from both horizontal neighbors.
        patches = torch.stack((patches, -patches), dim=-1)
        return patches[None] + image[:, :1, :1, 0]

    def embed_language_tokens(self, tokens):
        return torch.stack((tokens, -tokens), dim=-1).float()


@pytest.fixture
def make_model(monkeypatch):
    monkeypatch.delenv("OPENPI_VISION_TOKEN_POOLING", raising=False)
    monkeypatch.setattr(pi0_pytorch, "PaliGemmaWithExpertModel", lambda *args, **kwargs: _ProjectedTokens())

    def create(**kwargs):
        return pi0_pytorch.PI0Pytorch(Pi0Config(pytorch_compile_mode=None, **kwargs))

    return create


def _split_inputs(model, *, keep_base=True, wrist_first=False):
    base = torch.zeros(1, 224, 224, 3)
    observation = SimpleNamespace(
        images={"base_0_rgb": base, "left_wrist_0_rgb": base, "right_wrist_0_rgb": base},
        image_masks={
            "base_0_rgb": torch.tensor([keep_base]),
            "left_wrist_0_rgb": torch.tensor([True]),
            "right_wrist_0_rgb": torch.tensor([False]),
        },
        state=torch.zeros(1, 32),
        tokenized_prompt=torch.arange(200)[None],
        tokenized_prompt_mask=torch.arange(200)[None] < 37,
        token_ar_mask=None,
        token_loss_mask=None,
    )
    images, masks, tokens, token_mask, _ = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    active = [(image, mask) for image, mask in zip(images, masks, strict=True) if bool(mask.any())]
    if wrist_first:
        active.reverse()
    return [image for image, _ in active], [mask for _, mask in active], tokens, token_mask


@pytest.mark.parametrize("mode", ["none", "avg", "max"])
def test_split_prefix_values_and_masks(make_model, mode):
    model = make_model(pi05=True, vision_token_pooling=mode)
    images, masks, tokens, token_mask = _split_inputs(model)
    assert len(images) == 2

    embeddings, padding, attention = model.embed_prefix(images, masks, tokens, token_mask)
    base, wrist = [model.paligemma_with_expert.embed_image(image) for image in images]
    if mode == "avg":
        base = (base[:, ::2] + base[:, 1::2]) / 2
    elif mode == "max":
        base = torch.maximum(base[:, ::2], base[:, 1::2])
    lang = model.paligemma_with_expert.embed_language_tokens(tokens) * (2**0.5)
    torch.testing.assert_close(embeddings, torch.cat((base, wrist, lang), dim=1), rtol=0, atol=0)

    prefix_len = 712 if mode == "none" else 584
    assert embeddings.shape == (1, prefix_len, 2)
    assert padding.shape == attention.shape == (1, prefix_len)
    torch.testing.assert_close(padding[:, :-200], torch.ones_like(padding[:, :-200]))
    torch.testing.assert_close(padding[:, -200:], token_mask)
    assert not attention.any()


@pytest.mark.parametrize("mode", ["avg", "max"])
def test_split_preserves_wrist_when_base_is_reordered_or_removed(make_model, mode):
    model = make_model(pi05=True, vision_token_pooling=mode)
    images, masks, tokens, token_mask = _split_inputs(model, wrist_first=True)
    embeddings, _, _ = model.embed_prefix(images, masks, tokens, token_mask)
    wrist = model.paligemma_with_expert.embed_image(images[0])
    assert embeddings.shape[1] == 584
    torch.testing.assert_close(embeddings[:, :256], wrist, rtol=0, atol=0)

    images, masks, tokens, token_mask = _split_inputs(model, keep_base=False)
    embeddings, _, _ = model.embed_prefix(images, masks, tokens, token_mask)
    assert embeddings.shape[1] == 456
    torch.testing.assert_close(embeddings[:, :256], wrist, rtol=0, atol=0)


def test_untagged_inputs_require_explicit_camera_names_only_when_pooling(make_model):
    model = make_model(pi05=True)
    assert model.vision_token_pooling == "none"
    images, masks, tokens, token_mask = _split_inputs(model, keep_base=False)
    assert model.embed_prefix(images, masks, tokens, token_mask)[0].shape[1] == 456

    model = make_model(pi05=True, vision_token_pooling="avg")
    with pytest.raises(ValueError, match="requires camera names"):
        model.embed_prefix(images, masks, tokens, token_mask)
    prefix, _, _ = model.embed_prefix(images, masks, tokens, token_mask, image_keys=("left_wrist_0_rgb",))
    assert prefix.shape[1] == 456


@pytest.mark.parametrize("mode", ["none", "avg", "max"])
def test_environment_override_is_read_when_model_is_created(make_model, monkeypatch, mode):
    monkeypatch.setenv("OPENPI_VISION_TOKEN_POOLING", mode)
    model = make_model(pi05=True, vision_token_pooling="max" if mode == "none" else "none")
    assert model.vision_token_pooling == mode


def test_invalid_environment_value_fails_at_model_creation(make_model, monkeypatch):
    monkeypatch.setenv("OPENPI_VISION_TOKEN_POOLING", "mean")
    with pytest.raises(ValueError, match="none, avg, or max"):
        make_model(pi05=True)


@pytest.mark.parametrize("mode", ["avg", "max"])
def test_compiled_prefix_consumes_tagged_split_inputs(make_model, mode):
    model = make_model(pi05=True, vision_token_pooling=mode)
    inputs = _split_inputs(model, wrist_first=True)
    compiled = torch.compile(model.embed_prefix, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(*inputs), model.embed_prefix(*inputs))
    inputs = _split_inputs(model)
    torch.testing.assert_close(compiled(*inputs), model.embed_prefix(*inputs))
    inputs = _split_inputs(model, keep_base=False)
    torch.testing.assert_close(compiled(*inputs), model.embed_prefix(*inputs))
