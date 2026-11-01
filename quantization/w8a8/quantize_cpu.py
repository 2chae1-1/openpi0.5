"""CPU weight conversion only; no model download or GPU initialization."""

import torch

def quantize_linear_weights(state, layer_names):
    """Convert explicit Linear weights to raw INT8 and FP32 channel scales.

    The caller supplies a full state_dict and explicit target module names.
    Other tensors, including bias, are preserved. Input state is unchanged.
    This function does not verify the complete pi0.5 model architecture.
    """
    names = list(layer_names)
    if (
        not names
        or any(not isinstance(n, str) or not n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Target names must be nonempty, valid and unique")

    # Validate every target before creating the converted dictionary.
    for name in names:
        key = name + ".weight"
        if key not in state:
            raise ValueError(f"Missing weight: {key}")
        if name + ".qweight" in state or name + ".weight_scale" in state:
            raise ValueError(f"Already quantized or conflicting keys: {name}")

        weight = state[key]
        if not isinstance(weight, torch.Tensor):
            raise ValueError(f"Expected tensor: {key}")
        if weight.device.type != "cpu":
            raise ValueError(f"CPU tensor required: {key}")
        if weight.ndim != 2 or min(weight.shape) == 0:
            raise ValueError(f"Expected nonempty 2D Linear weight: {key}")
        if not weight.is_floating_point():
            raise ValueError(f"Floating point weight required: {key}")
        if not torch.isfinite(weight.float()).all():
            raise ValueError(f"Weight must remain finite in FP32: {key}")

    result = dict(state)
    for name in names:
        weight = state[name + ".weight"].detach().float()
        scale = (weight.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        qweight = (
            (weight / scale[:, None]).round().clamp(-127, 127)
            .to(torch.int8).contiguous()
        )
        del result[name + ".weight"]
        result[name + ".qweight"] = qweight
        result[name + ".weight_scale"] = scale.contiguous()

    return result
