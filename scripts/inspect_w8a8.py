"""Inspect local checkpoint headers; GPU initialization requires an explicit flag."""

import argparse
import json
import logging
import pathlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=pathlib.Path)
    parser.add_argument("--initialize", action="store_true", help="Build and probe all INT8 kernels on GPU")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-compile-for-validation", action="store_true")
    parser.add_argument("--enable-tuning", action="store_true")
    parser.add_argument("--opt-m", type=int, nargs="+", default=[1, 16, 32, 64])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.initialize:
        from openpi.models_pytorch.bitblas_w8a8_layers import w8a8_diagnostics
        from openpi.policies.w8a8_policy import load_w8a8_policy

        _, model = load_w8a8_policy(
            args.checkpoint_dir, device=args.device,
            disable_compile_for_validation=args.disable_compile_for_validation,
            enable_tuning=args.enable_tuning, opt_m=tuple(args.opt_m),
        )
        report = w8a8_diagnostics(model)
        report["pytorch_compile_mode"] = model.config.pytorch_compile_mode
    else:
        from safetensors import safe_open

        from openpi.policies.w8a8_policy import read_manifest

        names = read_manifest(args.checkpoint_dir)
        layers = []
        with safe_open(str(args.checkpoint_dir / "model.safetensors"), framework="pt", device="cpu") as tensors:
            quantized_names = {key.removesuffix(".qweight") for key in tensors.keys() if key.endswith(".qweight")}
            if quantized_names != set(names):
                raise ValueError("Manifest targets and checkpoint qweight keys disagree")
            for name in names:
                weight = tensors.get_slice(f"{name}.qweight")
                scale = tensors.get_slice(f"{name}.weight_scale")
                shape = weight.get_shape()
                if weight.get_dtype() != "I8" or len(shape) != 2 or scale.get_shape() != [shape[0]]:
                    raise ValueError(f"{name}: unsupported raw INT8 weight/scale shape")
                if scale.get_dtype() not in {"F16", "BF16", "F32", "F64"}:
                    raise ValueError(
                        f"{name}.weight_scale: unsupported dtype {scale.get_dtype()}; "
                        "expected F16, BF16, F32 or F64"
                    )
                layers.append({"name": name, "qweight_shape": shape, "scale_dtype": scale.get_dtype()})
        report = {
            "inspection": "headers_only", "layer_count": len(layers), "layers": layers,
            "kernel_initialization": "not_run", "full_weight_and_norm_validation": "not_run",
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
