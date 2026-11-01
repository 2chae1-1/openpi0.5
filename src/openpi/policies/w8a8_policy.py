# SPDX-License-Identifier: Apache-2.0
# Adapted from openpi_duquant at 7d4d1582297ced3131a7b3f9403f7ce403fd59f9.
# See quantization/w8a8/NOTICE.md for sources and intentional changes.
# ruff: noqa: PLC0415 -- Defer heavyweight model dependencies until loading.
"""Explicit local-only W8A8 loading; official policy/model entry points are unchanged."""

import dataclasses
import json
import logging
import pathlib

import numpy as np
import torch

from openpi.models_pytorch.bitblas_w8a8_layers import BitBLASW8A8Linear
from openpi.models_pytorch.bitblas_w8a8_layers import replace_w8a8_layers


def read_manifest(checkpoint_dir: pathlib.Path | str) -> list[str]:
    """Require the complete-model manifest emitted by the upstream exporter."""
    path = pathlib.Path(checkpoint_dir)
    manifest = json.loads((path / "w8a8_config.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("w8a8_config.json must contain a JSON object")
    if manifest.get("quantization") != "w8a8_bitblas" or manifest.get("complete_model") is not True:
        raise ValueError("Expected complete_model=true and quantization=w8a8_bitblas")
    names = manifest.get("w8a8_layers")
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("w8a8_layers must be a nonempty list of explicit module names")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate W8A8 target names")
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(path / "model.safetensors")
    return names


def prepare_checkpoint_state(model, state: dict) -> None:
    """Normalize only known nonpersistent buffers and the model's verified tied pair.

    safetensors.save_model may omit either alias. Other missing keys remain errors.
    Mutates the supplied dictionary without copying weight tensors.
    """
    expected = model.state_dict()
    optional_buffers = (
        "paligemma_with_expert.gemma_expert.model.rotary_emb.inv_freq",
        "paligemma_with_expert.paligemma.model.language_model.rotary_emb.inv_freq",
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_ids",
    )
    for key in optional_buffers:
        if key not in expected:
            state.pop(key, None)
    head = "paligemma_with_expert.paligemma.lm_head.weight"
    embedding = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    if head not in expected or embedding not in expected:
        return
    if model.get_parameter(head) is not model.get_parameter(embedding):
        return  # No alias restoration unless the actual architecture ties these weights.
    if head in state and embedding not in state:
        state[embedding] = state[head]
    elif embedding in state and head not in state:
        state[head] = state[embedding]
    elif (
        head in state
        and embedding in state
        and (state[head].dtype != state[embedding].dtype or not torch.equal(state[head], state[embedding]))
    ):
        raise ValueError("Conflicting checkpoint values for tied lm_head/embedding")


def validate_state_dict(model, state: dict, names: list[str]) -> None:
    """Check every persistent weight/buffer before load or GPU initialization."""
    expected = model.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(f"Checkpoint keys differ: missing={missing}, unexpected={unexpected}")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(f"{name}: shape {tuple(tensor.shape)} != {tuple(expected[name].shape)}")
        if expected[name].is_floating_point():
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f"{name}: expected finite floating point tensor")
        elif tensor.dtype != expected[name].dtype:
            raise ValueError(f"{name}: expected {expected[name].dtype}, got {tensor.dtype}")
    for name in names:
        if state[f"{name}.qweight"].dtype != torch.int8:
            raise ValueError(f"{name}: qweight must be raw INT8 [out_features, in_features]")
        scale_key = f"{name}.weight_scale"
        scale = state[scale_key].to(dtype=expected[scale_key].dtype)
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError(
                f"{scale_key}: all values must be finite and greater than zero "
                f"after conversion to {expected[scale_key].dtype}"
            )


def load_norm_stats(checkpoint_dir, data_config):
    """LIBERO state/actions stats are mandatory; never use an empty fallback."""
    from openpi.shared import normalize

    if not data_config.asset_id:
        raise ValueError("An asset_id is required for checkpoint normalization statistics")
    stats = normalize.load(pathlib.Path(checkpoint_dir) / "assets" / data_config.asset_id)
    for key in ("state", "actions"):
        if key not in stats:
            raise ValueError(f"Missing normalization statistics: {key}")
        item = stats[key]
        fields = ("mean", "std", "q01", "q99") if data_config.use_quantile_norm else ("mean", "std")
        arrays = [np.asarray(getattr(item, field)) for field in fields]
        if any(a.ndim != 1 or a.size == 0 or not np.isfinite(a).all() for a in arrays):
            raise ValueError(f"{key}: statistics must be finite nonempty vectors ({fields})")
        if any(a.shape != arrays[0].shape for a in arrays):
            raise ValueError(f"{key}: normalization statistics have inconsistent shapes")
        if (item.std < 0).any():
            raise ValueError(f"{key}: negative standard deviation")
        if data_config.use_quantile_norm and (item.q99 < item.q01).any():
            raise ValueError(f"{key}: q99 is smaller than q01")
    # This entry point is deliberately limited to the official pi05_libero config.
    if not 8 <= stats["state"].mean.size <= 32 or not 7 <= stats["actions"].mean.size <= 32:
        raise ValueError("pi05_libero statistics must cover 8 state/7 action dimensions, with at most 32 padded dims")
    return stats


def load_w8a8_policy(
    checkpoint_dir: pathlib.Path | str,
    *,
    device: str = "cuda",
    disable_compile_for_validation: bool = False,
    enable_tuning: bool = False,
    opt_m: tuple[int, ...] = (1, 16, 32, 64),
    sample_kwargs: dict | None = None,
    default_prompt: str | None = None,
):
    """Load an already downloaded pi05_libero checkpoint and return (policy, model).

    Only the opt-in validation flag sets a local config copy's compile mode to None.
    No checkpoint downloads, environment mutation, pickle loading or FP fallback.
    The official transform factory still needs the PaliGemma tokenizer cache.
    """
    from safetensors.torch import load_file

    from openpi import transforms
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies.policy import Policy
    from openpi.training import config

    names = read_manifest(checkpoint_dir)
    logging.info("W8A8 manifest accepted: %d explicit layers", len(names))
    train_config = config.get_config("pi05_libero")
    if disable_compile_for_validation:
        train_config = dataclasses.replace(
            train_config, model=dataclasses.replace(train_config.model, pytorch_compile_mode=None)
        )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = load_norm_stats(checkpoint_dir, data_config)
    logging.info("W8A8 normalization statistics validated")
    model = PI0Pytorch(train_config.model)
    # Convert before replacement so checkpoint scales never pass through BF16.
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    replace_w8a8_layers(model, names)
    state = load_file(str(pathlib.Path(checkpoint_dir) / "model.safetensors"), device="cpu")

    prepare_checkpoint_state(model, state)
    validate_state_dict(model, state, names)
    model.load_state_dict(state, strict=True)
    logging.info("W8A8 strict checkpoint load passed: %d layers; scales loaded as FP32", len(names))
    del state
    model.to(device).eval()
    initialized = 0
    for module in model.modules():
        if isinstance(module, BitBLASW8A8Linear):
            # Scales remain FP32 at runtime; packing is separate from checkpoint qweight.
            module.weight_scale = module.weight_scale.float()
            module.initialize_kernel(opt_m=opt_m, enable_tuning=enable_tuning)
            initialized += 1
            logging.info("W8A8 INT8 probe passed [%d/%d]: %s", initialized, len(names), module.name)
    if initialized != len(names):
        raise RuntimeError("Incomplete W8A8 initialization")
    policy = Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=True,
        pytorch_device=device,
    )
    return policy, model
