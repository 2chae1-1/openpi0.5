"""GPU-server smoke test of a local W8A8 pi05_libero checkpoint and policy.infer."""

import argparse
import json
import logging
import os
import pathlib
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--observation", type=pathlib.Path, help="LIBERO observation NPZ; allow_pickle=False")
    parser.add_argument("--prompt", default="pick up the book")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opt-m", type=int, nargs="+", default=[1, 16, 32, 64])
    parser.add_argument("--enable-tuning", action="store_true")
    parser.add_argument("--report", required=True, type=pathlib.Path, help="New JSON report; no overwrite")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.num_steps <= 0:
        parser.error("--num-steps must be positive")
    # Reserve the report before expensive work; failures also leave a report.
    with args.report.open("x", encoding="utf-8") as report_file:
        report = {
            "checkpoint": str(args.checkpoint_dir.resolve()),
            "input_kind": "recorded_observation" if args.observation else "synthetic_smoke",
            "observation": str(args.observation.resolve()) if args.observation else None,
            "prompt": args.prompt,
            "seed": args.seed, "num_steps": args.num_steps,
            "strict_model_load_and_kernel_probes": "not_run",
            "policy_infer": "not_run", "all_target_layers_called": "not_run",
            "accuracy_evaluated": False, "profiler_verified": False,
            "performance_evaluated": False, "pytorch_compile_mode": None,
        }
        stage = None
        model = None
        try:
            # Keep JAX transforms from reserving GPU memory used by PyTorch.
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
            import numpy as np
            import torch

            from openpi.models_pytorch.bitblas_w8a8_layers import w8a8_diagnostics
            from openpi.policies.w8a8_policy import load_w8a8_policy

            device = torch.device(args.device)
            if device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("Run on a CUDA GPU server; CPU execution is not supported")
            torch.manual_seed(args.seed)
            rng = np.random.default_rng(args.seed)
            report["environment"] = {
                "python": platform.python_version(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
                "opt_m": args.opt_m, "enable_tuning": args.enable_tuning,
            }
            if args.observation:
                with np.load(args.observation, allow_pickle=False) as data:
                    observation = {key: data[key] for key in (
                        "observation/state", "observation/image", "observation/wrist_image"
                    )}
            else:
                observation = {
                    "observation/state": np.zeros(8, dtype=np.float32),
                    "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
                    "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
                }
            observation["prompt"] = args.prompt
            state = observation["observation/state"]
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError("Observation state must be a finite (8,) vector")
            for key in ("observation/image", "observation/wrist_image"):
                image = observation[key]
                if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError(f"{key}: expected HWC uint8 RGB image")
            stage = "strict_model_load_and_kernel_probes"
            policy, model = load_w8a8_policy(
                args.checkpoint_dir, device=args.device, disable_compile_for_validation=True,
                enable_tuning=args.enable_tuning, opt_m=tuple(args.opt_m),
                sample_kwargs={"num_steps": args.num_steps},
            )
            report["strict_model_load_and_kernel_probes"] = "passed"
            report["diagnostics_before"] = w8a8_diagnostics(model)
            noise = rng.standard_normal((model.config.action_horizon, model.config.action_dim)).astype(np.float32)
            stage = "policy_infer"
            with torch.inference_mode():
                result = policy.infer(observation, noise=noise)
            torch.cuda.synchronize(device)
            actions = np.asarray(result["actions"])
            expected_shape = (model.config.action_horizon, 7)
            if actions.shape != expected_shape or not np.isfinite(actions).all():
                raise ValueError(f"Expected finite actions {expected_shape}; got {actions.shape}")
            report["policy_infer"] = "passed"
            report["actions_shape"] = list(actions.shape)
            report["diagnostics_after"] = w8a8_diagnostics(model)
            stage = "all_target_layers_called"
            before = {layer["name"]: layer["forward_calls"] for layer in report["diagnostics_before"]["layers"]}
            uncalled = [layer["name"] for layer in report["diagnostics_after"]["layers"]
                        if layer["forward_calls"] <= before[layer["name"]]]
            if uncalled:
                report["all_target_layers_called"] = "failed"
                raise RuntimeError(f"Inference did not call these W8A8 targets: {uncalled}")
            report["all_target_layers_called"] = "passed"
        except Exception as exc:
            if stage is not None:
                report[stage] = "failed"
            if model is not None:
                report["diagnostics_after"] = w8a8_diagnostics(model)
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            json.dump(report, report_file, indent=2)
            report_file.write("\n")
    print(f"GPU forward smoke test passed. Report: {args.report}")


if __name__ == "__main__":
    main()
