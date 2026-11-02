# SPDX-License-Identifier: Apache-2.0
# Adapted from JingxuanZhang77/openpi_duquant, commit
# 7d4d1582297ced3131a7b3f9403f7ce403fd59f9. See quantization/w8a8/NOTICE.md.
# ruff: noqa: PLC0415 -- BitBLAS is imported only when initializing CUDA kernels.
"""Inference-only W8A8 Linear. No floating point matmul fallback."""

import contextlib
import importlib.metadata
import logging
import os
import sys

import torch
from torch import nn

_LOGGER = logging.getLogger(__name__)
_BITBLAS_LOGGERS = (
    "bitblas.ops.operator",
    "bitblas.base.tuner",
    "bitblas.base.utils",
    "bitblas.relax.transform.apply_fast_tuning",
    "bitblas.tl.tuner",
    "bitblas.builder.lib_generator",
    "bitblas.utils.target_detector",
)


class _BitBLASErrorCapture(logging.Handler):
    """Recover exceptions while BitBLAS is still inside its swallowing except block."""

    def __init__(self, layer_name: str):
        super().__init__(logging.DEBUG)
        self.layer_name = layer_name
        self.first_error = None
        self.error_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        # Some 0.1.0.post1 tuning logs pass an exception without a %s placeholder.
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            message = f"{record.msg!s} args={record.args!r}"
        exception = record.exc_info if record.exc_info and record.exc_info[0] else sys.exc_info()
        if exception[0] is not None:
            self.error_count += 1
            if self.first_error is None:
                self.first_error = f"{exception[0].__name__}: {exception[1]}"
            _LOGGER.error(
                "BitBLAS internal exception #%d layer=%s source=%s:%d: %s",
                self.error_count,
                self.layer_name,
                record.name,
                record.lineno,
                message,
                exc_info=exception,
            )
        else:
            # Keep non-exception DEBUG evidence (including subprocess build errors)
            # visible through the benchmark worker's INFO handler, too.
            _LOGGER.log(
                max(record.levelno, logging.INFO),
                "BitBLAS %s layer=%s source=%s:%d: %s",
                record.levelname,
                self.layer_name,
                record.name,
                record.lineno,
                message,
            )


@contextlib.contextmanager
def _capture_bitblas_errors(layer_name: str, *, enabled: bool):
    """Temporarily expose post1's hidden build tracebacks, without patching its code.

    Loading initializes layers sequentially. Logging changes apply only for this
    initialization scope and are restored even if construction/packing fails.
    """
    capture = _BitBLASErrorCapture(layer_name)
    states = []
    try:
        if enabled:
            for name in _BITBLAS_LOGGERS:
                logger = logging.getLogger(name)
                states.append((logger, logger.level, logger.propagate, logger.disabled, logger.handlers[:]))
                logger.setLevel(logging.DEBUG)
                logger.propagate = False
                logger.disabled = False
                logger.handlers = [capture]
        yield capture
    finally:
        for logger, level, propagate, disabled, handlers in reversed(states):
            logger.handlers = handlers
            logger.disabled = disabled
            logger.propagate = propagate
            logger.setLevel(level)
        capture.close()


def _log_bitblas_environment(bitblas, device) -> None:
    try:
        version = importlib.metadata.version("bitblas")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(bitblas, "__version__", "unknown")
    tvm = sys.modules.get("tvm")
    _LOGGER.info(
        "BitBLAS diagnostic environment: bitblas=%s bitblas_file=%s tvm=%s tvm_file=%s "
        "torch=%s torch_cuda=%s device=%s gpu=%s capability=%s CUDA_HOME=%s CUDA_PATH=%s",
        version,
        getattr(bitblas, "__file__", None),
        getattr(tvm, "__version__", None),
        getattr(tvm, "__file__", None),
        torch.__version__,
        torch.version.cuda,
        device,
        torch.cuda.get_device_name(device),
        torch.cuda.get_device_capability(device),
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
    )


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
        self.last_activation_dtype = None
        self.last_output_dtype = None
        self.kernel_stage = "not_initialized"
        self.kernel_error = None

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
        self.forward_calls = 0
        self.last_activation_dtype = None
        self.last_output_dtype = None
        self._packed_weight = None
        self.kernel_stage = "preflight"
        self.kernel_error = None
        if self.qweight.device.type != "cuda":
            raise RuntimeError(f"{self.name}: W8A8 requires CUDA weights; CPU fallback is disabled")
        if not opt_m or any(type(m) is not int or m <= 0 for m in opt_m):
            raise ValueError("opt_m must contain positive integers")
        diagnostics = os.environ.get("OPENPI_BITBLAS_DIAGNOSTICS") == "1"
        target = None
        config = None
        capture = None
        try:
            self.kernel_stage = "bitblas_import"
            import bitblas
            from bitblas.ops import Matmul
            from bitblas.ops import MatmulConfig
            from bitblas.utils import auto_detect_nvidia_target

            with (
                torch.cuda.device(self.qweight.device),
                _capture_bitblas_errors(self.name, enabled=diagnostics) as capture,
            ):
                if diagnostics:
                    _log_bitblas_environment(bitblas, self.qweight.device)
                self.kernel_stage = "target_detection"
                target = auto_detect_nvidia_target()
                self.kernel_target = str(target)
                self.kernel_stage = "matmul_config"
                config = MatmulConfig(
                    M=list(opt_m),
                    N=self.out_features,
                    K=self.in_features,
                    A_dtype="int8",
                    W_dtype="int8",
                    out_dtype="int32",
                    accum_dtype="int32",
                    layout="nt",
                    with_scaling=False,
                    with_bias=False,
                )
                self.kernel_stage = "matmul_build"
                _LOGGER.info(
                    "W8A8 kernel start: layer=%s stage=%s device=%s target=%s backend=tir "
                    "enable_tuning=%s config=%s diagnostics=%s",
                    self.name,
                    self.kernel_stage,
                    self.qweight.device,
                    target,
                    enable_tuning,
                    config,
                    diagnostics,
                )
                operator = Matmul(config, target=target, backend="tir", enable_tuning=enable_tuning)
                self.kernel_stage = "runtime_validation"
                runtime_ready = getattr(operator, "rt_mod", None) is not None
                torch_ready = callable(getattr(operator, "torch_func", None))
                _LOGGER.info(
                    "W8A8 runtime state: layer=%s rt_mod=%s torch_func=%s lib=%s scheduled_ir_module=%s",
                    self.name,
                    runtime_ready,
                    torch_ready,
                    getattr(operator, "lib", None) is not None,
                    getattr(operator, "scheduled_ir_module", None) is not None,
                )
                if not runtime_ready or not torch_ready:
                    raise RuntimeError(
                        "BitBLAS returned an incomplete TIR runtime "
                        f"(rt_mod={runtime_ready}, torch_func={torch_ready}); "
                        "scheduling/build may have failed internally. "
                        "Rerun with OPENPI_BITBLAS_DIAGNOSTICS=1 and inspect the first "
                        "'BitBLAS internal exception' in benchmark.log."
                    )
                self.kernel_stage = "weight_packing"
                packed = operator.transform_weight(self.qweight.contiguous())
                if isinstance(packed, list | tuple):
                    if len(packed) != 1:
                        raise RuntimeError("Unsupported multi-tensor BitBLAS weight layout")
                    packed = packed[0]
                if not isinstance(packed, torch.Tensor) or packed.device != self.qweight.device:
                    raise RuntimeError("BitBLAS returned an invalid packed weight")
                self.kernel_stage = "int8_probe"
                _LOGGER.info("W8A8 weight packing passed: layer=%s; starting INT8 probe", self.name)
                # Check all output channels, with signed nonconstant activations.
                probe = (torch.arange(self.in_features, device=self.qweight.device) % 7 - 3).to(torch.int8)
                probe = probe.expand(opt_m[0], -1).contiguous()
                output = torch.empty((opt_m[0], self.out_features), dtype=torch.int32, device=probe.device)
                operator(probe, packed, output=output)
                self.kernel_stage = "probe_synchronize"
                torch.cuda.synchronize(self.qweight.device)
                self.kernel_stage = "integer_reference"
                reference = (self.qweight.cpu().to(torch.int64) * probe[0].cpu().to(torch.int64)).sum(dim=1)
                if not torch.equal(output.cpu().to(torch.int64), reference.expand(opt_m[0], -1)):
                    raise RuntimeError("INT8 kernel probe disagrees with integer reference")
                self._packed_weight = packed
                self.bitblas_matmul = operator
                self.kernel_verified = True
                self.kernel_target = str(target)
                self.probe_m = opt_m[0]
                self.kernel_stage = "verified"
        except Exception as exc:
            first_error = capture.first_error if capture is not None else None
            self.kernel_error = f"{type(exc).__name__}: {exc}"
            _LOGGER.exception(
                "W8A8 kernel failed: layer=%s stage=%s device=%s M=%s N=%d K=%d target=%s "
                "backend=tir enable_tuning=%s config=%s first_internal_error=%s",
                self.name,
                self.kernel_stage,
                self.qweight.device,
                opt_m,
                self.out_features,
                self.in_features,
                target,
                enable_tuning,
                config,
                first_error,
            )
            detail = f"; first_internal_error={first_error}" if first_error else ""
            raise RuntimeError(
                f"{self.name}: BitBLAS INT8 initialization/packing/probe failed "
                f"at stage={self.kernel_stage}: {self.kernel_error}{detail}"
            ) from exc

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
        # Record only returned operator calls, separately from initialization probes.
        self.forward_calls += 1
        self.last_activation_dtype = str(quantized.dtype)
        self.last_output_dtype = str(output.dtype)
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
    """Separate real integer probes and inference operator calls from profiler evidence."""
    layers = [
        {
            "name": name,
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "qweight_dtype": str(layer.qweight.dtype),
            "qweight_shape": list(layer.qweight.shape),
            "qweight_bytes": layer.qweight.numel() * layer.qweight.element_size(),
            "device": str(layer.qweight.device),
            "operator_created": layer.bitblas_matmul is not None,
            "weight_packed": layer._packed_weight is not None,  # noqa: SLF001
            "int8_probe_passed": layer.kernel_verified,
            "forward_calls": layer.forward_calls,
            "target": layer.kernel_target,
            "probe_m": layer.probe_m,
            "initialization_stage": layer.kernel_stage,
            "initialization_error": layer.kernel_error,
            "activation_dtype": "int8",
            "accum_dtype": "int32",
            "last_activation_dtype": layer.last_activation_dtype,
            "last_output_dtype": layer.last_output_dtype,
        }
        for name, layer in model.named_modules()
        if isinstance(layer, BitBLASW8A8Linear)
    ]
    return {
        "layer_count": len(layers),
        "all_int8_probes_passed": bool(layers) and all(layer["int8_probe_passed"] for layer in layers),
        "profiler_verified": False,
        "layers": layers,
    }
