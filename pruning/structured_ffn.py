"""Training-free *physical* FFN channel pruning for OpenPI pi0.5 PyTorch.

Read this before use:
- Call on a *loaded full-precision* PI0Pytorch instance; NOT on W8A8 BitBLAS layers.
- This is a simple weight-magnitude heuristic, not a reproduced SPAR/Wanda method.
- Modifies Gemma MLP's gate/up/down projections together, preserving I/O size.
- Reconstruct shapes from saved manifest BEFORE loading a pruned state_dict.
- No inference accuracy guarantee. Assess LIBERO before deployment.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn

Target = Literal["expert", "language", "both"]


def _check_mlp(mlp: nn.Module) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if not all(isinstance(layer, nn.Linear) for layer in (gate, up, down)):
        raise TypeError("Expected unquantized nn.Linear gate_proj/up_proj/down_proj. Load BF16/FP32 BEFORE pruning or quantizing.")
    width, hidden = gate.in_features, gate.out_features
    if up.in_features != width or up.out_features != hidden or down.in_features != hidden or down.out_features != width:
        raise ValueError("Incompatible gated FFN dimensions.")
    if gate.weight.device.type == "meta" or up.weight.device.type == "meta" or down.weight.device.type == "meta":
        raise ValueError("Model parameters must be materialized before ranking channels.")
    return gate, up, down


def _dense_linear_with_weights(src: nn.Linear, new_weight: torch.Tensor, new_bias: torch.Tensor | None) -> nn.Linear:
    """Avoid allocating/initializing another large weight matrix before replacing it."""
    out_features, in_features = new_weight.shape
    dst = nn.Linear(in_features, out_features, bias=new_bias is not None, device="meta", dtype=src.weight.dtype)
    dst.weight = nn.Parameter(new_weight.detach().contiguous(), requires_grad=src.weight.requires_grad)
    if new_bias is not None:
        dst.bias = nn.Parameter(new_bias.detach().contiguous(), requires_grad=src.bias.requires_grad)
    return dst


def _aligned_kept(width: int, fraction_to_keep: float, multiple_of: int) -> int:
    if not 0 < fraction_to_keep <= 1:
        raise ValueError("keep_fraction must be in (0,1].")
    if multiple_of < 1:
        raise ValueError("multiple_of must be positive.")
    proposed = max(1, round(width * fraction_to_keep))
    if multiple_of > 1 and width >= multiple_of:
        proposed = max(multiple_of, round(proposed / multiple_of) * multiple_of)
    return min(width, proposed)


@torch.no_grad()
def prune_gemma_mlp(mlp: nn.Module, keep_fraction: float = 0.9, multiple_of: int = 64) -> dict:
    """Remove low-magnitude *intermediate channels* and shrink all 3 matrices.

    Score[j] = L2(gate[j,:]) * L2(up[j,:]) * L2(down[:,j]);
    weight-only heuristic, NO gradient, NO calibration data, NO fine-tuning.
    """
    gate, up, down = _check_mlp(mlp)
    old_width = gate.out_features
    new_width = _aligned_kept(old_width, keep_fraction, multiple_of)
    score = (torch.linalg.vector_norm(gate.weight.float(), dim=1)
             * torch.linalg.vector_norm(up.weight.float(), dim=1)
             * torch.linalg.vector_norm(down.weight.float(), dim=0))
    if not bool(torch.isfinite(score).all()):
        raise ValueError("Non-finite magnitude score.")
    kept = torch.topk(score, k=new_width, sorted=False).indices.sort().values
    removed = old_width - new_width
    # A common index across gate/up output channels and down input channels.
    mlp.gate_proj = _dense_linear_with_weights(gate, gate.weight.index_select(0, kept).clone(),
                         gate.bias.index_select(0, kept).clone() if gate.bias is not None else None)
    mlp.up_proj = _dense_linear_with_weights(up, up.weight.index_select(0, kept).clone(),
                         up.bias.index_select(0, kept).clone() if up.bias is not None else None)
    mlp.down_proj = _dense_linear_with_weights(down, down.weight.index_select(1, kept).clone(),
                         down.bias.clone() if down.bias is not None else None)
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = new_width
    return {"original_intermediate": old_width,
            "new_intermediate": new_width,
            "removed_channels": removed,
            "kept_indices": kept.cpu().tolist(),
            "score": "l2_gate_times_l2_up_times_l2_down"}


def _get_layers(model: nn.Module, part: str):
    root = model.paligemma_with_expert
    if part == "expert":
        return root.gemma_expert.model.layers
    if part == "language":
        return root.paligemma.language_model.layers
    raise ValueError(part)


def prune_pi05_ffns(model: nn.Module, *, target: Target = "expert",
                    keep_fraction: float = 0.9, multiple_of: int = 64) -> dict:
    """Prune all layers in selected Pi0Pytorch component(s), *after* checkpoint load."""
    if target not in ("expert", "language", "both"):
        raise ValueError("target must be expert/language/both")
    parts = ["expert", "language"] if target == "both" else [target]
    before = sum(p.numel() for p in model.parameters())
    specs = []
    for part in parts:
        for index, layer in enumerate(_get_layers(model, part)):
            info = prune_gemma_mlp(layer.mlp, keep_fraction, multiple_of)
            specs.append({"part": part, "layer": index, **info})
    after = sum(p.numel() for p in model.parameters())
    return {"method": "training_free_magnitude_structured_ffn",
            "target": target, "requested_keep_fraction": keep_fraction,
            "multiple_of": multiple_of, "model_parameters_before": before,
            "model_parameters_after": after, "parameters_removed": before-after,
            "model_parameter_reduction_pct": 100 * (before-after) / before,
            "layers": specs}


def _resize_ffn(mlp: nn.Module, intermediate_size: int):
    """Reshape FFN before load_state_dict(pruned_state) using *fresh* untrained buffers."""
    gate, up, down = _check_mlp(mlp)
    if intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    def empty_linear(src: nn.Linear, in_f: int, out_f: int):
        # Allocate real buffers; state_dict will replace their contents.
        return nn.Linear(in_f, out_f, bias=src.bias is not None,
                         device=src.weight.device, dtype=src.weight.dtype)
    mlp.gate_proj = empty_linear(gate, gate.in_features, intermediate_size)
    mlp.up_proj = empty_linear(up, up.in_features, intermediate_size)
    mlp.down_proj = empty_linear(down, intermediate_size, down.out_features)
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = intermediate_size


def prepare_pi05_pruned_shapes(model: nn.Module, manifest: dict) -> None:
    """Call on freshly instantiated model BEFORE loading saved pruned state_dict.

    Only changes module shapes. The values remain random until state_dict load.
    """
    if manifest.get("method") != "training_free_magnitude_structured_ffn":
        raise ValueError("Unknown pruning manifest")
    for item in manifest["layers"]:
        layers = _get_layers(model, item["part"])
        _resize_ffn(layers[item["layer"]].mlp, item["new_intermediate"])
