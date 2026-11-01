"""CPU-only proof of real tensor shape and parameter count reduction.
Does NOT load a pi05_libero trained checkpoint or evaluate LIBERO.
Run: python -m pruning.demo_cpu
"""
import json
from pathlib import Path
import torch

from pruning.structured_ffn import prune_pi05_ffns, prune_gemma_mlp
from pruning.test_structured_ffn import TinyPi05, TinyGemmaMLP


def main():
    torch.manual_seed(123)
    toy = TinyPi05()
    before_b = sum(p.numel() * p.element_size() for p in toy.parameters())
    report = prune_pi05_ffns(toy, target="expert", keep_fraction=.75, multiple_of=4)
    after_b = sum(p.numel() * p.element_size() for p in toy.parameters())
    report["tensor_bytes_before"] = before_b
    report["tensor_bytes_after"] = after_b
    report["test_model"] = "tiny_randomly_initialized_gemma_style_mlp_only_NOT_pi05_checkpoint"
    report["gpu_measured"] = False
    report["libero_measured"] = False
    # Math-based extrapolation for a FULL 18-layer gemma_300m-like expert FFN.
    width, inner, n_layers = 1024, 4096, 18
    kept = 3072
    # Real tensor compaction of ONE full-size 1024 -> 4096 -> 1024 FFN,
    # using random weights; this is still NOT a trained pi05 checkpoint.
    fullsize_ffn = TinyGemmaMLP(width=width, intermediate=inner).to(dtype=torch.bfloat16)
    true_before = sum(p.numel() for p in fullsize_ffn.parameters())
    true_bytes_before = sum(p.numel() * p.element_size() for p in fullsize_ffn.parameters())
    one_layer_info = prune_gemma_mlp(fullsize_ffn, keep_fraction=.75, multiple_of=64)
    true_after = sum(p.numel() for p in fullsize_ffn.parameters())
    true_bytes_after = sum(p.numel() * p.element_size() for p in fullsize_ffn.parameters())
    assert true_before == 12_582_912 and true_after == 9_437_184
    assert true_bytes_before == 25_165_824 and true_bytes_after == 18_874_368
    report["real_tensor_compaction_single_fullsize_gemma_expert_ffn_random_weights"] = {
      "shape_before": [1024, 4096, 1024], "shape_after": [1024, 3072, 1024],
      "parameters_before": true_before, "parameters_after": true_after,
      "bf16_tensor_bytes_before": true_bytes_before, "bf16_tensor_bytes_after": true_bytes_after,
      "note": "ACTUAL tensor sizes after real slicing, but random weights not pi05_libero weights."
    }
    report["analytical_pi05_expert_mlp_if_25pct_pruned"] = {
      "hidden_size": width, "intermediate_before": inner, "intermediate_after": kept,
      "layers": n_layers, "ffn_weights_before": n_layers * 3 * width * inner,
      "ffn_weights_after": n_layers * 3 * width * kept,
      "ffn_weights_removed": n_layers * 3 * width * (inner-kept),
      "estimated_bf16_weight_bytes_reduced": n_layers * 3 * width * (inner-kept) * 2,
      "note": "Analytical (NOT actual checkpoint compression, accuracy, VRAM or runtime benchmark)."
    }
    lang_width, lang_inner, lang_layers = 2048, 16384, 18
    report["analytical_pi05_language_ffn_if_25pct_pruned"] = {
      "hidden_size": lang_width, "intermediate_before": lang_inner,
      "intermediate_after": 12288, "layers": lang_layers,
      "ffn_weights_before": lang_layers * 3 * lang_width * lang_inner,
      "ffn_weights_after": lang_layers * 3 * lang_width * 12288,
      "ffn_weights_removed": lang_layers * 3 * lang_width * (lang_inner - 12288),
      "estimated_bf16_weight_bytes_reduced": lang_layers * 3 * lang_width * (lang_inner - 12288) * 2,
      "note": "Analytical (NOT actual trained pi05 checkpoint, accuracy, VRAM or runtime measurement)."
    }
    output = Path("pruning/cpu_proof_result.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TOY TEST:", report["model_parameters_before"], "->", report["model_parameters_after"], "parameters")
    print("TOY TENSORS:", before_b, "->", after_b, "bytes")
    print("FULL-SIZE 1 FFN BF16 bytes:", true_bytes_before, "->", true_bytes_after)
    print("EXPERT FFN theoretical removed:", report["analytical_pi05_expert_mlp_if_25pct_pruned"]["ffn_weights_removed"])
    print("Written:", output)


if __name__ == "__main__":
    main()
