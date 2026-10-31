# SPDX-License-Identifier: Apache-2.0
# Adapted from JingxuanZhang77/openpi_duquant, commit
# 7d4d1582297ced3131a7b3f9403f7ce403fd59f9. See quantization/w8a8/NOTICE.md.
"""Inference-only W8A8 Linear. No floating point matmul fallback."""

import torch
from torch import nn


class BitBLASW8A8Linear(nn.Module):
    """Raw INT8 checkpoint weights, separately packed runtime weights, dynamic A8."""

    def __init__(self, linear: nn.Linear, name: str):
        super().__init__()
        self.name = name
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.register_buffer("qweight", torch.empty_like(linear.weight, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(self.out_features, dtype=torch.float32))
        self.register_buffer("bias", torch.empty_like(linear.bias) if linear.bias is not None else None)
        # Official Gemma accesses .weight.dtype. Never dequantize a packed matrix for this.
        self.register_buffer("_dtype_marker", torch.empty(0, dtype=linear.weight.dtype), persistent=False)
        self.register_buffer("_packed_weight", None, persistent=False)
        self.bitblas_matmul = None
        self.kernel_verified = False
        self.kernel_target = None
        self.probe_m = None
        self.forward_calls = 0

    @property
    def weight(self) -> torch.Tensor:
        """Dtype/device marker only; this layer has no floating point weight matrix."""
        return self._dtype_marker

    def initialize_kernel(self, *, opt_m: tuple[int, ...] = (1, 16, 32, 64), enable_tuning: bool = False):
        """Build, pack and verify a real CUDA INT8 call against an integer reference."""
        self.kernel_verified = False
        self.bitblas_matmul = None
        self.kernel_target = None
        self.probe_m = None
        self._packed_weight = None
        if self.qweight.device.type != "cuda":
            raise RuntimeError(f"{self.name}: W8A8 requires CUDA weights; CPU fallback is disabled")
        if not opt_m or any(type(m) is not int or m <= 0 for m in opt_m):
            raise ValueError("opt_m must contain positive integers")
        try:
            from bitblas.ops import Matmul
            from bitblas.ops import MatmulConfig
            from bitblas.utils import auto_detect_nvidia_target

            with torch.cuda.device(self.qweight.device):
                target = auto_detect_nvidia_target()
                config = MatmulConfig(
                    M=list(opt_m), N=self.out_features, K=self.in_features,
                    A_dtype="int8", W_dtype="int8", out_dtype="int32", accum_dtype="int32",
                    layout="nt", with_scaling=False, with_bias=False,
                )
                operator = Matmul(config, target=target, backend="tir", enable_tuning=enable_tuning)
                packed = operator.transform_weight(self.qweight.contiguous())
                if isinstance(packed, (list, tuple)):
                    if len(packed) != 1:
                        raise RuntimeError("Unsupported multi-tensor BitBLAS weight layout")
                    packed = packed[0]
                if not isinstance(packed, torch.Tensor) or packed.device != self.qweight.device:
                    raise RuntimeError("BitBLAS returned an invalid packed weight")
                # Check all output channels, with signed nonconstant activations.
                probe = (torch.arange(self.in_features, device=self.qweight.device) % 7 - 3).to(torch.int8)
                probe = probe.expand(opt_m[0], -1).contiguous()
                output = torch.empty((opt_m[0], self.out_features), dtype=torch.int32, device=probe.device)
                operator(probe, packed, output=output)
                torch.cuda.synchronize(self.qweight.device)
                reference = (self.qweight.cpu().to(torch.int64) * probe[0].cpu().to(torch.int64)).sum(dim=1)
                if not torch.equal(output.cpu().to(torch.int64), reference.expand(opt_m[0], -1)):
                    raise RuntimeError("INT8 kernel probe disagrees with integer reference")
                self._packed_weight = packed
                self.bitblas_matmul = operator
                self.kernel_verified = True
                self.kernel_target = str(target)
                self.probe_m = opt_m[0]
        except Exception as exc:
            raise RuntimeError(f"{self.name}: BitBLAS INT8 initialization/packing/probe failed") from exc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.kernel_verified or self.bitblas_matmul is None or self._packed_weight is None:
            raise RuntimeError(f"{self.name}: INT8 kernel has not passed initialization")
        if x.device != self.qweight.device or x.shape[-1] != self.in_features or not x.is_floating_point():
            raise ValueError(f"{self.name}: incompatible W8A8 input device, shape or dtype")
        x_2d = x.reshape(-1, self.in_features).float()
        # Per-tensor dynamic activation quantization; FP32 avoids BF16 reciprocal overflow/rounding.
        scale = x_2d.abs().amax().clamp(min=1e-8) / 127.0
        quantized = (x_2d / scale).round().clamp(-127, 127).to(torch.int8).contiguous()
        output = torch.empty((x_2d.shape[0], self.out_features), dtype=torch.int32, device=x.device)
        self.bitblas_matmul(quantized, self._packed_weight, output=output)
        self.forward_calls += 1
        result = output.float() * (scale * self.weight_scale.float())
        if self.bias is not None:
            result = result + self.bias.float()
        return result.reshape(*x.shape[:-1], self.out_features).to(x.dtype)


def replace_w8a8_layers(model: nn.Module, layer_names: list[str]) -> None:
    """Replace an explicit, validated manifest of Linear layers; never silently skip."""
    if not layer_names or len(set(layer_names)) != len(layer_names):
        raise ValueError("W8A8 targets must be nonempty and unique")
    originals = {name: model.get_submodule(name) for name in layer_names}
    for name, linear in originals.items():
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{name}: expected nn.Linear, got {type(linear).__name__}")
    for name, linear in originals.items():
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent), attribute, BitBLASW8A8Linear(linear, name))


def w8a8_diagnostics(model: nn.Module) -> dict:
    """Report initialization evidence separately from inference calls and profiler evidence."""
    layers = [
        {
            "name": name, "in_features": layer.in_features, "out_features": layer.out_features,
            "qweight_dtype": str(layer.qweight.dtype), "device": str(layer.qweight.device),
            "operator_created": layer.bitblas_matmul is not None,
            "weight_packed": layer._packed_weight is not None,  # noqa: SLF001
            "int8_probe_passed": layer.kernel_verified, "forward_calls": layer.forward_calls,
            "target": layer.kernel_target, "probe_m": layer.probe_m,
            "activation_dtype": "int8", "accum_dtype": "int32",
        }
        for name, layer in model.named_modules() if isinstance(layer, BitBLASW8A8Linear)
    ]
    return {
        "layer_count": len(layers), "all_int8_probes_passed": bool(layers)
        and all(layer["int8_probe_passed"] for layer in layers),
        "profiler_verified": False, "layers": layers,
    }
