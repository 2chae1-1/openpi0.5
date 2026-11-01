"""Offline pi05_libero CPU export. No GPU kernels or automatic downloads."""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Local FP checkpoint directory containing model.safetensors and assets")
    parser.add_argument("--targets", required=True, type=Path,
                        help="Candidate JSON containing explicit w8a8_layers")
    parser.add_argument("--output", required=True, type=Path,
                        help="New output directory; existing paths are rejected")
    args = parser.parse_args()

    # Set offline behavior before importing model dependencies.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from safetensors.torch import load_file, save_file

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from quantize_cpu import quantize_linear_weights
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models_pytorch.bitblas_w8a8_layers import replace_w8a8_layers
    from openpi.policies.w8a8_policy import load_norm_stats, validate_state_dict
    from openpi.training import config

    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    if not output.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    weights_path = source / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    selection = json.loads(args.targets.read_text(encoding="utf-8-sig"))
    if not isinstance(selection, dict) or selection.get("model_config") != "pi05_libero":
        raise ValueError("Targets must specify model_config=pi05_libero")
    names = selection.get("w8a8_layers")
    if (not isinstance(names, list) or not names
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("Targets must be a nonempty list of unique names")

    # This first export recipe explicitly covers both transformer stacks only.
    prefixes = (
        "paligemma_with_expert.paligemma.model.language_model.layers",
        "paligemma_with_expert.gemma_expert.model.layers",
    )
    parts = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    recipe = {f"{prefix}.{index}.{part}"
              for prefix in prefixes for index in range(18) for part in parts}
    if set(names) != recipe:
        raise ValueError("Targets differ from the explicit 252-layer candidate recipe")

    # Meta tensors describe the full architecture without allocating model weights.
    # Model construction uses local config constructors, never from_pretrained.
    train_config = config.get_config("pi05_libero")
    model_config = dataclasses.replace(train_config.model, pytorch_compile_mode=None)
    with torch.device("meta"):
        model = PI0Pytorch(model_config)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    if any(t.device.type != "meta" for t in model.state_dict().values()):
        raise RuntimeError("Architecture validation must use meta tensors only")
    for name in names:
        if not isinstance(model.get_submodule(name), torch.nn.Linear):
            raise TypeError(f"Target is not an actual Linear: {name}")

    # Match the existing pi05_libero loader's documented assets contract.
    asset_id = "physical-intelligence/libero"
    stats_dir = source / "assets" / asset_id
    load_norm_stats(source, SimpleNamespace(asset_id=asset_id, use_quantile_norm=True))
    print(f"Architecture targets and normalization statistics checked: {len(names)} layers", flush=True)
    print("Loading local weights on CPU; full checkpoint RAM is required.", flush=True)
    state = load_file(str(weights_path), device="cpu")
    optional = (
        "paligemma_with_expert.gemma_expert.model.rotary_emb.inv_freq",
        "paligemma_with_expert.paligemma.model.language_model.rotary_emb.inv_freq",
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_ids",
    )
    expected_keys = set(model.state_dict())
    for key in optional:
        if key not in expected_keys:
            state.pop(key, None)
    head = "paligemma_with_expert.paligemma.lm_head.weight"
    embedding = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    restored_head = head not in state
    if restored_head:
        if embedding not in state:
            raise ValueError("Missing tied embedding for lm_head restoration")
        state[head] = state[embedding]
    validate_state_dict(model, state, [])
    converted = quantize_linear_weights(state, names)
    del state
    with torch.device("meta"):
        replace_w8a8_layers(model, names)
    validate_state_dict(model, converted, names)
    del model
    # Retain the source's omitted alias convention; the loader restores it.
    if restored_head:
        del converted[head]

    staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        save_file(converted, str(staging / "model.safetensors"))
        del converted
        destination_stats = staging / "assets" / asset_id
        destination_stats.mkdir(parents=True)
        shutil.copy2(stats_dir / "norm_stats.json", destination_stats / "norm_stats.json")
        manifest = {
            "quantization": "w8a8_bitblas",
            "complete_model": True,
            "model_type": "pi05_libero",
            "w8a8_layers": names,
            "weight_format": "raw_int8_out_in",
            "weight_scale_dtype": "float32",
            "source_checkpoint": str(source),
            "validation": {
                "architecture_keys_shapes_and_values": True,
                "normalization_statistics": True,
                "full_model_load_executed": False,
                "gpu_kernel_executed": False,
                "accuracy_evaluated": False,
            },
        }
        (staging / "w8a8_config.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Windows rename rejects an existing destination, including a race.
        if output.exists():
            raise ValueError("Output was created during export; refusing to overwrite")
        staging.rename(output)
    except BaseException:
        # Keep partial files for diagnosis; never remove an unknown directory.
        print(f"Export interrupted. Partial files, if any: {staging}", file=sys.stderr)
        raise
    print(f"CPU export saved: {output}")
    print("Full-model loading, GPU execution and accuracy evaluation remain unverified.")


if __name__ == "__main__":
    main()
