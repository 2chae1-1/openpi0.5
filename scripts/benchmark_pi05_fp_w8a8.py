"""Local pi05_libero W8A8 smoke benchmark, with optional FP comparison (CUDA only).

The coordinator and --help need only the standard library. No checkpoint downloads,
JAX fallback, compilation of the policy, or model implementation live here.
"""

import argparse
import contextlib
import copy
import csv
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import pathlib
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRICS = ("mean", "p50", "p95")
DTYPE_SCHEME = "official selected bfloat16 with remaining float32; W8A8 scales float32"


def latency_statistics(values):
    """Milliseconds, with linearly interpolated percentiles; absent data stays null."""
    values = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Latencies must be finite and nonnegative")

    def percentile(fraction):
        index = (len(values) - 1) * fraction
        low = math.floor(index)
        high = math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)

    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "p50_ms": percentile(0.50) if values else None,
        "p95_ms": percentile(0.95) if values else None,
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
    }


def write_json(path, data):
    """Replace a report atomically, retaining the previous valid report on failure."""
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def append_latency(path, model, phase, iteration, latency_ms):
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow([model, phase, iteration, latency_ms])


def build_comparison(models):
    fp, w8a8 = models["fp"], models["w8a8"]
    result = {
        "status": "not_measured",
        "reasons": [],
        "speedup": dict.fromkeys(METRICS),
        "checkpoint_size": {"reduction_fraction": None, "compression_ratio": None},
        "memory": {},
    }
    if fp.get("status") == "not_measured":
        result["reasons"].append("FP checkpoint not provided; FP improvements were not measured")
        return result
    if any(model.get("status") != "passed" for model in (fp, w8a8)):
        result["reasons"].append("Both complete model runs must pass")
    if not fp.get("signature") or fp.get("signature") != w8a8.get("signature"):
        result["reasons"].append("Runtime, model, normalization, observation or noise signatures differ or are missing")
    if any(model.get("checks", {}).get("policy_infer") != "passed" for model in (fp, w8a8)):
        result["reasons"].append("Both policy.infer checks must pass")
    if any(w8a8.get("checks", {}).get(key) != "passed" for key in ("int8_kernels", "all_target_layers_called")):
        result["reasons"].append("W8A8 INT8 probes and inference calls must be verified")
    for key in METRICS:
        a = fp.get("latency_statistics", {}).get(f"{key}_ms")
        b = w8a8.get("latency_statistics", {}).get(f"{key}_ms")
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or a <= 0 or b <= 0:
            result["reasons"].append(f"Missing or invalid {key} latency")
    if result["reasons"]:
        return result
    result["status"] = "passed"
    fp_size = (fp.get("checkpoint_weights") or {}).get("bytes")
    w8_size = (w8a8.get("checkpoint_weights") or {}).get("bytes")
    if fp_size and w8_size:
        result["checkpoint_size"] = {
            "reduction_fraction": 1 - w8_size / fp_size,
            "compression_ratio": fp_size / w8_size,
        }
    result["speedup"] = {
        key: fp["latency_statistics"][f"{key}_ms"] / w8a8["latency_statistics"][f"{key}_ms"] for key in METRICS
    }
    for phase in ("loaded", "loading", "inference"):
        result["memory"][phase] = {}
        for metric in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"):
            a = fp.get("memory", {}).get(phase, {}).get(metric)
            b = w8a8.get("memory", {}).get(phase, {}).get(metric)
            result["memory"][phase][metric] = {
                "fp_minus_w8a8_bytes": a - b if a is not None and b is not None else None,
                "reduction_fraction": 1 - b / a if a and b is not None else None,
            }
    return result


def write_summary_csv(path, summary):
    """One row per model; unavailable numeric measurements are empty CSV cells."""
    rows = []
    for name, result in summary["models"].items():
        memory = result.get("memory", {})
        loaded = memory.get("loaded") or {}
        nvml = memory.get("nvml") or {}
        peaks = [memory.get(phase) or {} for phase in ("loading", "inference")]
        row = {
            "model": name,
            "status": result["status"],
            "checkpoint_bytes": (result.get("checkpoint_weights") or {}).get("bytes"),
            **{f"{key}_ms": result["latency_statistics"][f"{key}_ms"] for key in METRICS},
            "samples": result["latency_statistics"]["samples"],
            "int8_weights": result["checks"].get("int8_weights", "not_run"),
            "int8_kernels": result["checks"]["int8_kernels"],
            "all_target_layers_called": result["checks"]["all_target_layers_called"],
            "loaded_allocated_bytes": loaded.get("allocated_bytes"),
            "loaded_reserved_bytes": loaded.get("reserved_bytes"),
            "peak_allocated_bytes": max((p["peak_allocated_bytes"] for p in peaks if p), default=None),
            "peak_reserved_bytes": max((p["peak_reserved_bytes"] for p in peaks if p), default=None),
            "nvml_status": nvml.get("status", "not_measured"),
            "process_peak_used_bytes": nvml.get("process_peak_used_bytes"),
            "device_peak_used_bytes": nvml.get("device_peak_used_bytes"),
            "comparison_status": summary["comparison"]["status"],
            **{f"speedup_{key}": summary["comparison"]["speedup"][key] if name == "w8a8" else None for key in METRICS},
            "checkpoint_reduction_fraction": summary["comparison"]["checkpoint_size"]["reduction_fraction"]
            if name == "w8a8"
            else None,
            "failure_stage": result.get("failure_stage"),
            "error": result.get("error"),
        }
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_kernel_execution(before, after, expected_names):
    """Use existing integer-reference probes plus actual inference call evidence."""
    expected = set(expected_names)
    if not expected or len(expected) != len(expected_names):
        raise RuntimeError("W8A8 targets must be nonempty and unique")
    for diagnostics in (before, after):
        layers = diagnostics["layers"]
        if (
            diagnostics["layer_count"] != len(expected)
            or len(layers) != len(expected)
            or {layer["name"] for layer in layers} != expected
            or not diagnostics["all_int8_probes_passed"]
            or any(
                layer["qweight_dtype"] != "torch.int8" or not layer["device"].startswith("cuda:") for layer in layers
            )
            or any(
                not all(layer[key] for key in ("operator_created", "weight_packed", "int8_probe_passed"))
                for layer in layers
            )
        ):
            raise RuntimeError("W8A8 target set or real INT8 initialization evidence is incomplete")
    counts = {layer["name"]: layer["forward_calls"] for layer in before["layers"]}
    uncalled = [layer["name"] for layer in after["layers"] if layer["forward_calls"] <= counts[layer["name"]]]
    if uncalled:
        raise RuntimeError(f"W8A8 targets were not called by policy.infer: {uncalled}")
    if any(
        layer.get("last_activation_dtype") != "torch.int8" or layer.get("last_output_dtype") != "torch.int32"
        for layer in after["layers"]
    ):
        raise RuntimeError("W8A8 inference did not record INT8 activations and INT32 output for every target")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fp-checkpoint",
        type=pathlib.Path,
        help="Optional local FP directory; omit to measure W8A8 only (no improvement estimates)",
    )
    parser.add_argument(
        "--w8a8-checkpoint", required=True, type=pathlib.Path, help="Local public W8A8 checkpoint directory"
    )
    parser.add_argument(
        "--output-dir", required=True, type=pathlib.Path, help="New or empty directory; never overwrite a run"
    )
    parser.add_argument("--fp-id", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--w8a8-id", default="fatdove/pi05-libero-w8a8")
    parser.add_argument("--fp-revision", help="Operator-declared source revision; unknown by default")
    parser.add_argument("--w8a8-revision", help="Operator-declared source revision; unknown by default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--observation", type=pathlib.Path, help="Recorded LIBERO NPZ, loaded with allow_pickle=False")
    parser.add_argument("--prompt", default="pick up the book")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmup", type=int, default=10, help="Additional warmup calls after the separately timed first call"
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--opt-m", type=int, nargs="+", default=[1, 16, 32, 64])
    parser.add_argument("--enable-tuning", action="store_true")
    parser.add_argument("--_worker", choices=("fp", "w8a8"), help=argparse.SUPPRESS)
    return parser


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _empty_result(checkpoint, identifier, revision):
    return {
        "status": "not_run",
        "stage": "not_run",
        "failure_stage": None,
        "error": None,
        "checkpoint": {
            "directory": str(checkpoint.resolve()) if checkpoint else None,
            "identifier": identifier,
            "revision": revision,
            "revision_provenance": "operator_declared",
        },
        "checkpoint_weights": None,
        "environment": None,
        "signature": None,
        "checks": {
            "policy_infer": "not_run",
            "int8_weights": "not_run",
            "int8_kernels": "not_run",
            "all_target_layers_called": "not_run",
        },
        "timing_seconds": dict.fromkeys(
            ("load_total", "load_without_kernel", "kernel_prepare", "first_infer", "warmup_total")
        ),
        "latency_statistics": latency_statistics([]),
        "memory": {phase: None for phase in ("baseline", "loaded", "loading", "inference")},
        "actions_shape": None,
        "actions_all_finite": None,
        "completed_calls": {"first": 0, "warmup": 0, "measurement": 0},
        "profiler_verified": False,
    }


def _memory(torch, device):
    torch.cuda.synchronize(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


class VramSampler:
    """Sample NVML outside PyTorch's allocator; peaks are observed, not exact maxima."""

    def __init__(self, gpu_uuid, interval=0.05):
        self.interval = interval
        self.phase = "loading"
        self.nvml = None
        self.handle = None
        self.thread = None
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.data = {
            "status": "not_measured",
            "sample_interval_ms": interval * 1000,
            "gpu_uuid": gpu_uuid,
            "pid": os.getpid(),
            "errors": [],
            "phases": {},
            **self._empty_stats(),
        }
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            if gpu_uuid:
                self.handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
            else:
                # Never assume the CUDA ordinal equals the physical NVML ordinal.
                matches = []
                for index in range(pynvml.nvmlDeviceGetCount()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    if any(p.pid == os.getpid() for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)):
                        matches.append(handle)
                if len(matches) != 1:
                    raise RuntimeError("Cannot unambiguously map this CUDA process to an NVML GPU")
                self.handle = matches[0]
            self._sample()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        except Exception as exc:
            self.data["errors"].append(f"NVML unavailable: {type(exc).__name__}: {exc}")

    @staticmethod
    def _empty_stats():
        return {
            "samples": 0,
            "process_samples": 0,
            "device_samples": 0,
            "process_used_bytes": None,
            "process_peak_used_bytes": None,
            "device_used_bytes": None,
            "device_peak_used_bytes": None,
        }

    def _sample(self):
        if self.handle is None:
            return
        with self.lock:
            process_used = device_used = None
            try:
                device_used = int(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)
            except Exception as exc:
                error = f"Device VRAM unavailable: {type(exc).__name__}: {exc}"
                if error not in self.data["errors"]:
                    self.data["errors"].append(error)
            try:
                processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                values = [p.usedGpuMemory for p in processes if p.pid == os.getpid()]
                # NVML_VALUE_NOT_AVAILABLE (e.g. WDDM) must never become a huge peak or zero.
                if values and all(isinstance(v, int) and 0 <= v < 2**63 for v in values):
                    process_used = sum(values)
                else:
                    raise RuntimeError("Current PID VRAM is unavailable (PID namespace, MIG/MPS or driver support)")
            except Exception as exc:
                error = f"Process VRAM unavailable: {type(exc).__name__}: {exc}"
                if error not in self.data["errors"]:
                    self.data["errors"].append(error)
            phase = self.data["phases"].setdefault(self.phase, self._empty_stats())
            for stats in (self.data, phase):
                stats["samples"] += 1
                for prefix, value in (("process", process_used), ("device", device_used)):
                    stats[f"{prefix}_used_bytes"] = value
                    if value is not None:
                        stats[f"{prefix}_samples"] += 1
                        previous = stats[f"{prefix}_peak_used_bytes"]
                        stats[f"{prefix}_peak_used_bytes"] = max(value, previous if previous is not None else value)
            self.data["status"] = (
                "sampled"
                if self.data["process_samples"] and self.data["device_samples"] and not self.data["errors"]
                else "partial"
                if self.data["process_samples"] or self.data["device_samples"]
                else "not_measured"
            )

    def _run(self):
        while not self.done.wait(self.interval):
            self._sample()

    def snapshot(self, phase=None):
        if phase:
            with self.lock:
                self.phase = phase
        self._sample()
        with self.lock:
            return copy.deepcopy(self.data)

    def stop(self):
        self.done.set()
        if self.thread:
            self.thread.join()
        result = self.snapshot()
        if self.nvml:
            with contextlib.suppress(Exception):
                self.nvml.nvmlShutdown()
        return result


def _array_hash(value):
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _parameter_signature(model, quantized_names):
    """Compare actual FP dtypes/shapes, including the FP interface of replaced Linear layers."""
    parameters = {
        name: {"dtype": str(parameter.dtype), "shape": list(parameter.shape)}
        for name, parameter in model.named_parameters()
    }
    for name in quantized_names:
        layer = model.get_submodule(name)
        parameters[f"{name}.weight"] = {"dtype": str(layer.weight.dtype), "shape": list(layer.qweight.shape)}
        if layer.bias is not None:
            parameters[f"{name}.bias"] = {"dtype": str(layer.bias.dtype), "shape": list(layer.bias.shape)}
    return parameters


def _inputs(args, config):
    """Same observation contract and RNG order as validate_w8a8_forward.py."""
    import numpy as np

    rng = np.random.default_rng(args.seed)
    keys = ("observation/state", "observation/image", "observation/wrist_image")
    if args.observation:
        with np.load(args.observation, allow_pickle=False) as data:
            observation = {key: data[key] for key in keys}
    else:
        observation = {
            keys[0]: np.zeros(8, dtype=np.float32),
            keys[1]: rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            keys[2]: rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        }
    state = observation[keys[0]]
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Observation state must be a finite (8,) vector")
    for key in keys[1:]:
        image = observation[key]
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) <= 0:
            raise ValueError(f"{key}: expected nonempty HWC uint8 RGB image")
    observation["prompt"] = args.prompt
    noise = rng.standard_normal((config.action_horizon, config.action_dim)).astype(np.float32)
    return observation, noise, {key: _array_hash(observation[key]) for key in keys}


def _runtime_environment(torch, device):
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "bitblas": _version("bitblas"),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_index": device.index,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_uuid": str(properties.uuid) if hasattr(properties, "uuid") else None,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
    }


def _worker(args):
    """One model lifetime per process, including BitBLAS/TVM CUDA allocations."""
    output = args.output_dir
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    name = args._worker  # noqa: SLF001 -- Internal argparse worker flag.
    result = summary["models"][name]
    latencies = []
    torch = None
    device = None
    loading_started = None
    loading_finished = False
    inference_started = False
    kernel_seconds = []
    vram = None

    def save(stage=None):
        if stage:
            result["stage"] = stage
        if vram is not None:
            result["memory"]["nvml"] = vram.snapshot()
        write_json(summary_path, summary)

    try:
        save("checkpoint_preflight")
        checkpoint = args.fp_checkpoint if name == "fp" else args.w8a8_checkpoint
        if checkpoint is None or not (checkpoint / "model.safetensors").is_file():
            result["status"] = "failed"
            result["failure_stage"] = result["stage"]
            result["error"] = (
                "Local model.safetensors is required; no JAX comparison, downloads or automatic conversion"
            )
            logging.error("%s: %s", name, result["error"])
            save()
            return 1
        weights = checkpoint / "model.safetensors"
        size = weights.stat().st_size
        if size <= 0:
            raise ValueError("Empty checkpoint weight file")
        result["checkpoint_weights"] = {
            "file": str(weights.resolve()),
            "bytes": size,
            "gib": size / 2**30,
            "basis": "model.safetensors only; assets, metadata and duplicate cache files excluded",
        }
        save("environment")
        # Must happen before imports of OpenPI/JAX. CPU transforms must not reserve CUDA memory.
        os.environ["JAX_PLATFORMS"] = "cpu"
        sys.path.insert(0, str(ROOT / "src"))
        import numpy as np
        import torch

        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Run on a CUDA GPU server; CPU model execution is not supported")
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        result["environment"] = _runtime_environment(torch, device)
        from openpi.policies.w8a8_policy import load_norm_stats
        from openpi.training import config

        train_config = config.get_config("pi05_libero")
        train_config = dataclasses.replace(
            train_config, model=dataclasses.replace(train_config.model, pytorch_compile_mode=None)
        )
        save("input_and_normalization")
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        norm_stats = load_norm_stats(checkpoint, data_config)
        norm_signature = {
            key: {
                field: _array_hash(np.asarray(getattr(item, field)))
                for field in (("mean", "std", "q01", "q99") if data_config.use_quantile_norm else ("mean", "std"))
            }
            for key, item in norm_stats.items()
            if key in ("state", "actions")
        }
        observation, noise, observation_hashes = _inputs(args, train_config.model)
        expected_names = []
        if name == "w8a8":
            from openpi.policies.w8a8_policy import read_manifest

            expected_names = read_manifest(checkpoint)

        torch.cuda.synchronize(device)
        vram = VramSampler(result["environment"]["gpu_uuid"])
        result["memory"]["baseline"] = _memory(torch, device)
        torch.cuda.reset_peak_memory_stats(device)
        save("model_loading")
        loading_started = time.perf_counter()
        with torch.inference_mode():
            if name == "fp":
                from openpi.policies import policy_config

                policy = policy_config.create_trained_policy(
                    train_config,
                    checkpoint,
                    pytorch_device=str(device),
                    sample_kwargs={"num_steps": args.num_steps},
                    norm_stats=norm_stats,
                )
                model = policy._model  # noqa: SLF001 - official Policy has no public model accessor.
            else:
                from openpi.models_pytorch.bitblas_w8a8_layers import BitBLASW8A8Linear
                from openpi.models_pytorch.bitblas_w8a8_layers import w8a8_diagnostics
                from openpi.policies.w8a8_policy import load_w8a8_policy

                original_initialize = BitBLASW8A8Linear.initialize_kernel

                def timed_initialize(module, *positional, **keywords):
                    save("kernel_preparation")
                    started = time.perf_counter()
                    try:
                        torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        value = original_initialize(module, *positional, **keywords)
                        torch.cuda.synchronize(device)
                        return value
                    except Exception:
                        result["kernel_failure"] = {
                            "layer": module.name,
                            "stage": getattr(module, "kernel_stage", None),
                            "error": getattr(module, "kernel_error", None),
                            "in_features": module.in_features,
                            "out_features": module.out_features,
                            "opt_m": list(args.opt_m),
                            "enable_tuning": args.enable_tuning,
                        }
                        try:
                            torch.cuda.synchronize(device)
                        except Exception:
                            logging.exception(
                                "Secondary CUDA synchronize failure for %s; preserving initialization exception",
                                module.name,
                            )
                        raise
                    finally:
                        kernel_seconds.append(time.perf_counter() - started)

                with mock.patch.object(BitBLASW8A8Linear, "initialize_kernel", timed_initialize):
                    policy, model = load_w8a8_policy(
                        checkpoint,
                        device=str(device),
                        disable_compile_for_validation=True,
                        enable_tuning=args.enable_tuning,
                        opt_m=tuple(args.opt_m),
                        sample_kwargs={"num_steps": args.num_steps},
                    )
        torch.cuda.synchronize(device)
        result["timing_seconds"]["load_total"] = time.perf_counter() - loading_started
        loading_finished = True
        kernel_total = sum(kernel_seconds) if name == "w8a8" else None
        result["timing_seconds"]["kernel_prepare"] = kernel_total
        result["timing_seconds"]["load_without_kernel"] = result["timing_seconds"]["load_total"] - (kernel_total or 0)
        result["memory"]["loaded"] = _memory(torch, device)
        result["memory"]["loading"] = result["memory"]["loaded"].copy()
        save("post_load_validation")
        if model.config != train_config.model or model.config.pytorch_compile_mode is not None:
            raise RuntimeError("Loaded model config/compile setting differs from the fixed FP/W8A8 experiment")
        result["signature"] = {
            "environment": result["environment"],
            "model_config": dataclasses.asdict(model.config),
            "dtype_scheme": DTYPE_SCHEME,
            "batch_size": 1,
            "compile": False,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "prompt": args.prompt,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "observation_hashes": observation_hashes,
            "noise_hash": _array_hash(noise),
            "normalization_hashes": norm_signature,
            "parameter_dtypes_and_shapes": _parameter_signature(model, expected_names),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        }
        result["parameter_dtypes"] = sorted({str(parameter.dtype) for parameter in model.parameters()})
        before = w8a8_diagnostics(model) if name == "w8a8" else None
        if before is not None:
            result["diagnostics_before"] = before
            if any(layer["qweight_dtype"] != "torch.int8" for layer in before["layers"]):
                raise RuntimeError("All manifest qweights must be actual INT8 tensors")
            result["checks"]["int8_weights"] = "passed"
        save("first_infer")
        vram.snapshot("inference")
        torch.cuda.reset_peak_memory_stats(device)
        inference_started = True
        # Kernel initialization may consume RNG state; start inference from the same seed.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        def infer(phase, iteration):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                prediction = policy.infer(observation, noise=noise)
            torch.cuda.synchronize(device)
            latency = (time.perf_counter() - started) * 1000
            # Preserve every completed infer even when its action validation fails.
            append_latency(output / "latency.csv", name, phase, iteration, latency)
            actions = np.asarray(prediction["actions"])
            result["actions_shape"] = list(actions.shape)
            result["actions_all_finite"] = bool(np.isfinite(actions).all())
            if actions.shape != (model.config.action_horizon, 7) or not result["actions_all_finite"]:
                raise ValueError(f"Expected finite actions {(model.config.action_horizon, 7)}; got {actions.shape}")
            result["completed_calls"][phase] += 1
            logging.info("%s %s[%d]: %.3f ms; actions=%s finite", name, phase, iteration, latency, actions.shape)
            return latency

        result["timing_seconds"]["first_infer"] = infer("first", 0) / 1000
        result["checks"]["policy_infer"] = "passed"
        if name == "w8a8":
            save("int8_inference_verification")
            after = w8a8_diagnostics(model)
            result["diagnostics_after_first"] = after
            check_kernel_execution(before, after, expected_names)
            result["checks"]["int8_kernels"] = "passed"
            result["checks"]["all_target_layers_called"] = "passed"
        save("warmup")
        warmups = []
        for index in range(args.warmup):
            warmups.append(infer("warmup", index))
            result["timing_seconds"]["warmup_total"] = sum(warmups) / 1000
            save()
        if not warmups:
            result["timing_seconds"]["warmup_total"] = 0.0  # explicitly requested zero calls
        save("measurement")
        measurement_before = w8a8_diagnostics(model) if name == "w8a8" else None
        for index in range(args.iterations):
            latencies.append(infer("measurement", index))
            result["latency_statistics"] = latency_statistics(latencies)
            save()
        if name == "w8a8":
            result["diagnostics_after"] = w8a8_diagnostics(model)
            check_kernel_execution(measurement_before, result["diagnostics_after"], expected_names)
        result["memory"]["inference"] = _memory(torch, device)
        result["status"] = "passed"
        save("complete")
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["failure_stage"] = result["stage"]
        result["error"] = f"{type(exc).__name__}: {exc}"
        logging.exception("%s failed during %s", name, result["stage"])
        if kernel_seconds:
            result["timing_seconds"]["kernel_prepare"] = sum(kernel_seconds)
        if loading_started is not None and not loading_finished:
            result["timing_seconds"]["load_elapsed_before_failure"] = time.perf_counter() - loading_started
        if torch is not None and device is not None and device.type == "cuda":
            with contextlib.suppress(Exception):
                result["memory"]["inference" if inference_started else "loading"] = _memory(torch, device)
        save()
        return 1
    finally:
        if vram is not None:
            result["memory"]["nvml"] = vram.stop()
            write_json(summary_path, summary)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.num_steps <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("--num-steps and --iterations must be positive; --warmup must be nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if any(value <= 0 for value in args.opt_m):
        parser.error("--opt-m values must be positive")
    if args._worker:  # noqa: SLF001 -- Internal argparse worker flag.
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
        return _worker(args)
    output = args.output_dir
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output-dir must be new or empty; existing results are preserved")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    with (output / "latency.csv").open("x", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(["model", "phase", "iteration", "latency_ms"])
    summary = {
        "schema_version": 2,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),  # noqa: UP017 -- Keep Python 3.10 CPU tooling compatible.
        "finished_at": None,
        "git_commit": _git_commit(),
        "status": "not_run",
        "environment": {
            "python": platform.python_version(),
            "pytorch": _version("torch"),
            "bitblas": _version("bitblas"),
            "cuda": None,
            "gpu": None,
        },
        "experiment": {
            "config": "pi05_libero",
            "framework": "pytorch",
            "dtype_scheme": DTYPE_SCHEME,
            "compile": False,
            "batch_size": 1,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "prompt": args.prompt,
            "device": args.device,
            "opt_m": args.opt_m,
            "enable_tuning": args.enable_tuning,
            "input_kind": "recorded_observation" if args.observation else "synthetic_smoke",
            "observation": str(args.observation.resolve()) if args.observation else None,
            "isolation": "sequential fresh subprocess per model",
            "jax_platforms": "cpu",
        },
        "limitations": [
            "Synthetic input measures smoke latency, not LIBERO task performance"
            if not args.observation
            else "One fixed recorded observation is replayed; no simulator episodes evaluated",
            "LIBERO success rate and accuracy are not evaluated",
            "Kernel timing combines BitBLAS construction/compilation, packing and integer-reference probes",
            "Loading excludes input/norm preflight; load_total includes kernel_prepare; do not add them",
            "Inference memory peak covers first infer, additional warmup and all measured calls",
            "PyTorch allocated/reserved peaks exclude external CUDA allocations",
            "NVML process VRAM includes external allocations; its 50 ms sampled peak can miss shorter spikes",
            "NVML device VRAM includes other processes; unsupported NVML measurements stay null with reasons",
            "Fresh processes isolate live allocations; disk/kernel caches, thermals and other GPU users are uncontrolled",
            "Checkpoint identifiers and revisions are operator declarations, not verified remote provenance",
            "Existing internal policy_timing is asynchronous and is not used for end-to-end speedup",
        ],
        "libero_success_rate": None,
        "accuracy_evaluated": False,
        "models": {
            "fp": _empty_result(args.fp_checkpoint, args.fp_id, args.fp_revision),
            "w8a8": _empty_result(args.w8a8_checkpoint, args.w8a8_id, args.w8a8_revision),
        },
        "comparison": None,
    }
    if args.fp_checkpoint is None or not (args.fp_checkpoint / "model.safetensors").is_file():
        summary["models"]["fp"].update(
            status="not_measured",
            stage="skipped",
            error="FP model.safetensors not provided or unavailable; FP comparison not measured",
        )
    write_json(summary_path, summary)
    worker_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        with (output / "benchmark.log").open("x", encoding="utf-8") as log:
            log.write("Experiment: " + json.dumps(summary["experiment"], ensure_ascii=False) + "\n")
            if args.observation is None:
                log.write("WARNING: synthetic smoke input; LIBERO task performance/success rate is not evaluated\n")
            for name in ("fp", "w8a8"):
                if summary["models"][name]["status"] == "not_measured":
                    log.write(f"{name}: not_measured; {summary['models'][name]['error']}\n")
                    continue
                print(f"Running {name} in an isolated process; log: {output / 'benchmark.log'}", flush=True)
                log.write(f"\n=== {name}: isolated process ===\n")
                log.flush()
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-u",
                        str(pathlib.Path(__file__).resolve()),
                        *worker_argv,
                        "--_worker",
                        name,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    exit_code = process.wait()
                except KeyboardInterrupt:
                    # Stop the worker before finalizing the shared report; it must not
                    # continue using CUDA or overwrite the coordinator's partial result.
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    model = summary["models"][name]
                    model.update(
                        status="failed",
                        failure_stage=model["stage"],
                        error="Coordinator interrupted this worker",
                        process_exit_code=process.returncode,
                    )
                    write_json(summary_path, summary)
                    raise
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                model = summary["models"][name]
                model["process_exit_code"] = exit_code
                if model.get("environment") and summary["environment"]["cuda"] is None:
                    summary["environment"] = model["environment"]
                if exit_code != 0 and model["error"] is None:
                    model.update(
                        status="failed", failure_stage=model["stage"], error="Worker terminated; inspect benchmark.log"
                    )
                write_json(summary_path, summary)
    except (OSError, KeyboardInterrupt) as exc:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["coordinator_error"] = f"{type(exc).__name__}: {exc}"
        with (output / "benchmark.log").open("a", encoding="utf-8") as log:
            log.write(summary["coordinator_error"] + "\n")
    summary["comparison"] = build_comparison(summary["models"])
    requested = [model for model in summary["models"].values() if model["status"] != "not_measured"]
    summary["status"] = (
        "passed"
        if all(model["status"] == "passed" for model in requested) and not summary.get("coordinator_error")
        else "incomplete"
    )
    summary["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()  # noqa: UP017 -- Python 3.10 CPU tooling.
    write_json(summary_path, summary)
    write_summary_csv(output / "summary.csv", summary)
    with (output / "benchmark.log").open("a", encoding="utf-8") as log:
        log.write("Comparison: " + json.dumps(summary["comparison"], ensure_ascii=False) + "\n")
    print(f"Benchmark {summary['status']}: {summary_path}", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
