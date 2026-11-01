"""Load official pi05_libero PyTorch weights, then prune selected gated FFNs.

Run from the repository root: python -m pruning.run_structured_ffn --help
Only the original dense model is accepted; no quantization or training is used.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
from datetime import timezone
import gc
import importlib.util
import json
import logging
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import torch

from pruning.lm_head_audit import lm_head_metrics
from pruning.lm_head_audit import observe_lm_head_calls
from pruning.structured_ffn import estimate_pi05_ffn_reduction
from pruning.structured_ffn import prepare_pi05_pruned_shapes
from pruning.structured_ffn import prune_pi05_ffns

CONFIG_NAME = "pi05_libero"
STATE_DICT_NAME = "pruned_state_dict.pt"
MANIFEST_NAME = "pruning_manifest.json"
REPORT_NAME = "report.json"
RESTORE_RTOL = 1e-5
RESTORE_ATOL = 1e-6


def write_json(path: Path, data: dict) -> None:
    """Publish complete JSON files, including reports from unsuccessful runs."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def preflight(checkpoint_dir: str, device: str) -> dict:
    """Check local prerequisites without importing OpenPI or downloading weights."""
    issues = []
    if sys.version_info < (3, 11):
        issues.append("OpenPI requires Python >=3.11.")
    modules = {name: importlib.util.find_spec(name) is not None for name in (
        "openpi", "safetensors", "transformers", "jax", "flax",
    )}
    for name, available in modules.items():
        if not available:
            issues.append(f"Missing dependency: {name} (install the repository environment).")
    requested = torch.device(device)
    if requested.type not in ("cpu", "cuda"):
        issues.append("Only cpu and cuda devices are supported by this runner.")
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            issues.append("CUDA is unavailable; GPU inference has not been verified.")
        elif requested.index is not None and requested.index >= torch.cuda.device_count():
            issues.append(f"CUDA device index {requested.index} is unavailable.")
        else:
            with torch.cuda.device(requested):
                if not torch.cuda.is_bf16_supported():
                    issues.append("CUDA device does not support BF16.")
    remote = "://" in checkpoint_dir
    local_weights = None if remote else Path(checkpoint_dir).expanduser() / "model.safetensors"
    if local_weights is not None and not local_weights.is_file():
        issues.append(
            f"Missing PyTorch checkpoint: {local_weights}. JAX params are not accepted; "
            "convert with examples/convert_jax_model_to_pytorch.py --config-name pi05_libero first."
        )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "modules": modules,
        "remote_checkpoint_not_inspected": remote,
        "local_pytorch_weights_found": None if remote else local_weights.is_file(),
        "issues": issues,
    }


def get_train_config(model_config: dict | None = None):
    # Lazy imports allow CPU tests and prerequisite reports without the full stack.
    from openpi.training import config as training_config

    config = training_config.get_config(CONFIG_NAME)
    overrides = dict(model_config or {})
    # Mutate dense eager modules, never a compiled sampling graph.
    overrides.update(dtype="bfloat16", pytorch_compile_mode=None)
    return dataclasses.replace(config, model=dataclasses.replace(config.model, **overrides))


def load_original_bf16(checkpoint_dir: str):
    """Use the same loader and selective BF16 conversion as create_trained_policy."""
    from openpi.shared import download

    config = get_train_config()
    resolved = (
        Path(download.maybe_download(checkpoint_dir))
        if "://" in checkpoint_dir
        else Path(checkpoint_dir).expanduser().resolve()
    )
    weight_path = resolved / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"{weight_path} is required. Convert the official pi05_libero JAX checkpoint to PyTorch first."
        )
    model = config.model.load_pytorch(config, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()
    # Protect the requested source precision and avoid accidentally pruning W8A8.
    root = model.paligemma_with_expert
    for layers in (root.gemma_expert.model.layers, root.paligemma.language_model.layers):
        for layer in layers:
            for projection in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                if not isinstance(projection, torch.nn.Linear) or projection.weight.dtype != torch.bfloat16:
                    raise TypeError("Expected original dense BF16 Action Expert and Language Backbone FFN weights.")
    return model, config, resolved


def parameter_metrics(model: torch.nn.Module) -> dict:
    """Count unique parameters and their actual dtype-dependent tensor storage."""
    def count(parameters):
        result = {"parameters": 0, "weight_bytes": 0, "by_dtype": {}}
        seen = set()
        for parameter in parameters:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            elements = parameter.numel()
            size = elements * parameter.element_size()
            result["parameters"] += elements
            result["weight_bytes"] += size
            dtype = result["by_dtype"].setdefault(str(parameter.dtype), {"parameters": 0, "weight_bytes": 0})
            dtype["parameters"] += elements
            dtype["weight_bytes"] += size
        result["weight_mib"] = result["weight_bytes"] / 1024**2
        return result

    root = model.paligemma_with_expert
    expert_layers = root.gemma_expert.model.layers
    language_layers = root.paligemma.language_model.layers
    return {
        "model": count(model.parameters()),
        "action_expert_ffn": count(p for layer in expert_layers for p in layer.mlp.parameters()),
        "language_ffn": count(p for layer in language_layers for p in layer.mlp.parameters()),
        "action_expert_lm_head": lm_head_metrics(model),
        "expert_intermediate_sizes": [layer.mlp.gate_proj.out_features for layer in expert_layers],
        "language_intermediate_sizes": [layer.mlp.gate_proj.out_features for layer in language_layers],
    }


def make_smoke_inputs(config, device: torch.device, batch_size: int, seed: int):
    """Synthetic, already preprocessed observations; no tokenizer or dataset needed."""
    image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    tokens = torch.zeros(batch_size, config.max_token_len, dtype=torch.long, device=device)
    tokens[:, 0] = 2  # Gemma BOS; remaining tokens are padding.
    token_mask = torch.zeros_like(tokens, dtype=torch.bool)
    token_mask[:, 0] = True
    observation = SimpleNamespace(
        images={key: torch.zeros(batch_size, 224, 224, 3, device=device) for key in image_keys},
        image_masks={key: torch.full((batch_size,), key != "right_wrist_0_rgb", device=device) for key in image_keys},
        state=torch.zeros(batch_size, config.action_dim, device=device),
        tokenized_prompt=tokens,
        tokenized_prompt_mask=token_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(batch_size, config.action_horizon, config.action_dim, generator=generator, device=device)
    return observation, noise


@torch.inference_mode()
def measure_output(
    model, device: torch.device, observation, noise: torch.Tensor, num_steps: int,
) -> tuple[dict, torch.Tensor]:
    """Validate actions and keep the tensor for the saved-model comparison."""
    with observe_lm_head_calls(model) as lm_head_usage:
        output = model.sample_actions(device, observation, noise=noise.clone(), num_steps=num_steps)
    if not isinstance(output, torch.Tensor):
        raise TypeError("sample_actions did not return a tensor.")
    result = {
        "status": "passed", "shape": list(output.shape), "dtype": str(output.dtype), "lm_head_usage": lm_head_usage,
    }
    if list(output.shape) != list(noise.shape):
        raise ValueError(f"Unexpected action shape: {list(output.shape)}; expected {list(noise.shape)}")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("sample_actions produced non-finite values.")
    result["all_finite"] = True
    return result, output.detach().clone()


def save_pruned_artifacts(model, config, manifest: dict, output_dir: Path, *, save_state_dict: bool) -> dict:
    manifest.update({
        "format_version": 1,
        "train_config_name": CONFIG_NAME,
        "model_config": dataclasses.asdict(config.model),
        "precision_policy": "official_selective_bfloat16_with_float32_stability_parameters",
        "state_dict_file": STATE_DICT_NAME if save_state_dict else None,
        "restore": "pruning.run_structured_ffn.load_pruned_model",
    })
    artifacts = {"manifest": {"status": "pending"}, "state_dict": {"status": "skipped"}}
    if save_state_dict:
        destination = output_dir / STATE_DICT_NAME
        temporary = destination.with_suffix(".pt.tmp")
        # Keep shared storages intact and avoid a second full CPU copy of the model.
        torch.save(model.state_dict(), temporary)
        temporary.replace(destination)
        artifacts["state_dict"] = {"status": "saved", "path": str(destination), "file_bytes": destination.stat().st_size}
    write_json(output_dir / MANIFEST_NAME, manifest)
    artifacts["manifest"] = {"status": "saved", "path": str(output_dir / MANIFEST_NAME)}
    return artifacts


def load_pruned_model(artifact_dir: str | Path, device: str = "cpu"):
    """Restore a compact state_dict strictly, without the original checkpoint.

    The manifest reconstructs the compact architecture, not the removed weights.
    This does not create a Policy or load its tokenizer/normalization assets.
    """
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("train_config_name") != CONFIG_NAME:
        raise ValueError("Unsupported pi05_libero pruning manifest.")
    if manifest.get("state_dict_file") != STATE_DICT_NAME:
        raise ValueError("Manifest does not reference a saved pruned state_dict.")
    config = get_train_config(manifest["model_config"])
    model = PI0Pytorch(config.model)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    prepare_pi05_pruned_shapes(model, manifest)
    state = torch.load(artifact_dir / STATE_DICT_NAME, map_location="cpu", weights_only=True, mmap=True)
    # Copy into existing parameters to preserve HuggingFace tied-weight identities.
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def run(args: argparse.Namespace) -> tuple[dict, int]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Do not leave an old successful report/weights associated with a failed rerun.
    if any((output_dir / name).exists() for name in (REPORT_NAME, MANIFEST_NAME, STATE_DICT_NAME)):
        raise FileExistsError(f"Use a new output directory; existing artifacts found in {output_dir}")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "unverified",
        "config_name": CONFIG_NAME,
        "checkpoint_dir": args.checkpoint_dir,
        "device": args.device,
        "target": args.target,
        "requested_prune_fraction": args.prune_fraction,
        "multiple_of": args.multiple_of,
        "before": None,
        "after": None,
        "ten_percent_projection": None,
        "per_target": None,
        "outputs": {
            "before": {"status": "unverified", "shape": None},
            "after": {"status": "unverified", "shape": None},
            "restored": {"status": "unverified", "shape": None},
            "input_kind": "synthetic_preprocessed_observation_fixed_noise",
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "artifacts": {"manifest": {"status": "not_saved"}, "state_dict": {"status": "not_saved"}},
        "verification": {
            "original_checkpoint_loaded": False,
            "physical_pruning": False,
            "output_shapes_match": None,
            "restore_roundtrip": {
                "status": "unverified", "reason": "No saved-model action comparison has run.",
                "rtol": RESTORE_RTOL, "atol": RESTORE_ATOL,
            },
            "libero_accuracy": {"status": "unverified", "reason": "No LIBERO evaluation is performed."},
        },
        "memory_definition": "Unique parameter numel * element_size; excludes buffers, activations, KV cache and allocator overhead.",
        "notes": ["Training-free weight-magnitude heuristic; no calibration, gradients or fine-tuning.",
                  "Magnitude pruning is the baseline; only selected expert/language gated FFN intermediate channels are removed.",
                  "All lm_head modules are retained; Action Expert head usage observations apply only to each sampled forward.",
                  "Output shapes are raw model actions, before LIBERO policy postprocessing."],
    }
    exit_code = 2
    stage = "preflight"
    try:
        report["preflight"] = preflight(args.checkpoint_dir, args.device)
        if report["preflight"]["issues"] or args.preflight_only:
            report["unverified_reasons"] = report["preflight"]["issues"] or ["Preflight only; model execution was not requested."]
            return report, exit_code
        stage = "load_original_bf16"
        model, config, resolved = load_original_bf16(args.checkpoint_dir)
        report["resolved_checkpoint_dir"] = str(resolved)
        report["verification"]["original_checkpoint_loaded"] = True
        report["before"] = parameter_metrics(model)
        report["ten_percent_projection"] = {
            "basis": "shape_based_projection_not_compression_measurement",
            "requested_prune_fraction": 0.10,
            "multiple_of": args.multiple_of,
            "model_parameters_before": report["before"]["model"]["parameters"],
            **estimate_pi05_ffn_reduction(model, keep_fraction=0.9, multiple_of=args.multiple_of),
        }
        device = torch.device(args.device)
        inputs = None
        if not args.skip_forward:
            stage = "baseline_forward"
            try:
                model.to(device)
                inputs = make_smoke_inputs(config.model, device, args.batch_size, args.seed)
                report["outputs"]["before"], _ = measure_output(model, device, *inputs, args.num_steps)
            except Exception as exc:
                report["outputs"]["before"] = {"status": "failed", "shape": None, "error": f"{type(exc).__name__}: {exc}"}
                inputs = None
                # Prune/save on CPU even when the device cannot run a smoke test.
                model.cpu()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        stage = "physical_pruning"
        manifest = prune_pi05_ffns(model, target=args.target, keep_fraction=1 - args.prune_fraction, multiple_of=args.multiple_of)
        report["verification"]["physical_pruning"] = True
        report["after"] = parameter_metrics(model)
        report["parameters_removed"] = manifest["parameters_removed"]
        report["model_parameter_reduction_pct"] = manifest["model_parameter_reduction_pct"]
        report["weight_bytes_removed"] = report["before"]["model"]["weight_bytes"] - report["after"]["model"]["weight_bytes"]
        report["per_target"] = manifest["per_target"]
        for part, metrics in report["per_target"].items():
            metric_key = "action_expert_ffn" if part == "expert" else "language_ffn"
            metrics["weight_bytes_removed"] = (
                report["before"][metric_key]["weight_bytes"] - report["after"][metric_key]["weight_bytes"]
            )
            metrics["weight_mib_removed"] = metrics["weight_bytes_removed"] / 1024**2
        for layer in manifest["layers"]:
            layer["actual_prune_fraction"] = layer["removed_channels"] / layer["original_intermediate"]
        report["layers"] = manifest["layers"]
        stage = "save_artifacts"
        manifest["source_checkpoint"] = str(resolved)
        report["artifacts"] = save_pruned_artifacts(model, config, manifest, output_dir, save_state_dict=not args.no_save_state_dict)
        if inputs is not None:
            stage = "pruned_forward"
            report["outputs"]["after"], pruned_actions = measure_output(model, device, *inputs, args.num_steps)
            report["verification"]["output_shapes_match"] = report["outputs"]["before"]["shape"] == report["outputs"]["after"]["shape"]
            if not args.no_save_state_dict:
                # Release the compact model before constructing another full model,
                # so the roundtrip does not require two resident GPU models.
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                stage = "restore_load"
                roundtrip = report["verification"]["restore_roundtrip"]
                roundtrip.pop("reason", None)
                restored_model = load_pruned_model(output_dir, device=str(device))
                stage = "restored_forward"
                report["outputs"]["restored"], restored_actions = measure_output(
                    restored_model, device, *inputs, args.num_steps,
                )
                stage = "restore_compare"
                roundtrip.update({
                    "actions_close": False,
                    "max_abs_diff": (restored_actions.double() - pruned_actions.double()).abs().max().item(),
                })
                torch.testing.assert_close(
                    restored_actions, pruned_actions, rtol=RESTORE_RTOL, atol=RESTORE_ATOL,
                    check_dtype=True, check_device=True,
                )
                roundtrip.update(status="passed", actions_close=True)
            else:
                report["verification"]["restore_roundtrip"]["reason"] = "No state_dict was saved (--no-save-state-dict)."
                report["outputs"]["restored"]["reason"] = "No state_dict was saved (--no-save-state-dict)."
        else:
            report["outputs"]["after"]["reason"] = "Forward checks were skipped or baseline forward failed."
            report["outputs"]["restored"]["reason"] = "Forward checks were skipped or baseline forward failed."
            report["verification"]["restore_roundtrip"]["reason"] = "Forward checks were skipped or baseline forward failed."
        verified = (
            report["verification"]["output_shapes_match"]
            and report["verification"]["restore_roundtrip"]["status"] == "passed"
        )
        report["status"] = "completed" if verified else "partial"
        exit_code = 0 if report["status"] == "completed" else 2
    except Exception as exc:
        missing_prerequisite = stage in ("preflight", "load_original_bf16") and isinstance(exc, (ImportError, FileNotFoundError))
        report["status"] = "unverified" if missing_prerequisite else "failed"
        report["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        if stage in ("restore_load", "restored_forward", "restore_compare"):
            report["verification"]["restore_roundtrip"].update(status="failed", error=report["error"])
        phase = {"pruned_forward": "after", "restored_forward": "restored"}.get(stage)
        if phase:
            report["outputs"][phase].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        exit_code = 2 if missing_prerequisite else 1
        if not missing_prerequisite:
            logging.exception("Structured pruning failed during %s", stage)
    finally:
        write_json(output_dir / REPORT_NAME, report)
    return report, exit_code


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="Local or remote directory containing original dense model.safetensors.")
    parser.add_argument("--output-dir", required=True, help="New directory for report, manifest and compact state_dict.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu; BF16 weights are retained on CPU.")
    parser.add_argument("--target", choices=("expert", "language", "both"), default="expert",
                        help="FFNs to prune: Action Expert (default), Language Backbone, or both.")
    parser.add_argument("--prune-fraction", type=float, default=0.10)
    parser.add_argument("--multiple-of", type=int, default=1, help="Retained FFN width alignment; actual ratio is reported.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--preflight-only", action="store_true", help="Inspect prerequisites without downloading or loading weights.")
    parser.add_argument("--skip-forward", action="store_true", help="Prune/save only; outputs and restore roundtrip remain unverified.")
    parser.add_argument("--no-save-state-dict", action="store_true", help="Save report/manifest only; restore roundtrip remains unverified.")
    args = parser.parse_args(argv)
    if not 0 < args.prune_fraction < 1:
        parser.error("--prune-fraction must be in (0, 1).")
    if min(args.multiple_of, args.batch_size, args.num_steps) < 1:
        parser.error("--multiple-of, --batch-size and --num-steps must be positive.")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    report, exit_code = run(args)
    print(json.dumps({"status": report["status"], "output_dir": str(Path(args.output_dir).resolve())}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
