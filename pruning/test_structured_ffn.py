import copy
import unittest

import torch
from torch import nn

from pruning.structured_ffn import estimate_pi05_ffn_reduction
from pruning.structured_ffn import prepare_pi05_pruned_shapes
from pruning.structured_ffn import prune_gemma_mlp
from pruning.structured_ffn import prune_pi05_ffns


class TinyGemmaMLP(nn.Module):
    """Matches the 3-projection gate/up/down arrangement in OpenPI modeling_gemma.py."""
    def __init__(self, width=8, intermediate=16, *, bias=False):
        super().__init__()
        self.intermediate_size = intermediate
        self.gate_proj = nn.Linear(width, intermediate, bias=bias)
        self.up_proj = nn.Linear(width, intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, width, bias=bias)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class TinyPi05(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = nn.Module()
        self.paligemma_with_expert.gemma_expert.model.layers = nn.ModuleList([nn.Module() for _ in range(2)])
        for layer in self.paligemma_with_expert.gemma_expert.model.layers:
            layer.mlp = TinyGemmaMLP()


class TinyDualPi05(TinyPi05):
    """Different backbone/expert widths plus untouched model components."""

    def __init__(self, *, bias=False):
        super().__init__()
        root = self.paligemma_with_expert
        for layer in root.gemma_expert.model.layers:
            layer.mlp = TinyGemmaMLP(8, 20, bias=bias)
            layer.self_attn = nn.Linear(8, 8)
        root.gemma_expert.lm_head = nn.Linear(8, 23, bias=False)
        root.paligemma = nn.Module()
        language = root.paligemma.language_model = nn.Module()
        language.layers = nn.ModuleList([nn.Module() for _ in range(3)])
        for layer in language.layers:
            layer.mlp = TinyGemmaMLP(12, 30, bias=bias)
            layer.self_attn = nn.Linear(12, 12)
        language.embed_tokens = nn.Embedding(23, 12)
        root.paligemma.lm_head = nn.Linear(12, 23, bias=False)
        root.paligemma.lm_head.weight = language.embed_tokens.weight
        root.paligemma.vision_tower = nn.Linear(3, 12)
        self.action_out_proj = nn.Linear(8, 4)


class TestStructuredPruning(unittest.TestCase):
    def test_physical_compaction_preserves_masked_output(self):
        torch.manual_seed(123)
        original = TinyGemmaMLP()
        masked = copy.deepcopy(original)
        pruned = copy.deepcopy(original)
        x = torch.randn(2, 5, 8)
        manifest = prune_gemma_mlp(pruned, keep_fraction=.75, multiple_of=4)
        assert manifest["new_intermediate"] == 12
        indices = manifest["kept_indices"]
        remove_mask = torch.ones(16, dtype=torch.bool)
        remove_mask[indices] = False
        with torch.no_grad():
            masked.down_proj.weight[:, remove_mask] = 0
        self.assertEqual(pruned.gate_proj.weight.shape, (12, 8))
        self.assertEqual(pruned.up_proj.weight.shape, (12, 8))
        self.assertEqual(pruned.down_proj.weight.shape, (8, 12))
        self.assertEqual(sum(p.numel() for p in original.parameters()), 384)
        self.assertEqual(sum(p.numel() for p in pruned.parameters()), 288)
        self.assertTrue(torch.allclose(pruned(x), masked(x), atol=1e-6))

    def test_pi05_target_manifest_and_state_roundtrip(self):
        torch.manual_seed(123)
        model = TinyPi05()
        manifest = prune_pi05_ffns(model, target="expert", keep_fraction=.75, multiple_of=4)
        self.assertEqual(manifest["parameters_removed"], 192)
        self.assertEqual(manifest["model_parameter_reduction_pct"], 25.0)
        fresh = TinyPi05()
        # Older expert-only artifacts have no additive per-target metrics.
        manifest.pop("per_target")
        prepare_pi05_pruned_shapes(fresh, manifest)
        fresh.load_state_dict(model.state_dict(), strict=True)
        x = torch.randn(2, 5, 8)
        for l1, l2 in zip(model.paligemma_with_expert.gemma_expert.model.layers,
                          fresh.paligemma_with_expert.gemma_expert.model.layers):
            self.assertTrue(torch.allclose(l1.mlp(x), l2.mlp(x), atol=0))

    def test_rejects_non_linear_quantized_layers(self):
        class QuantizedStub(nn.Module):
            pass
        toy = TinyGemmaMLP()
        toy.gate_proj = QuantizedStub()
        with self.assertRaises(TypeError):
            prune_gemma_mlp(toy)

    def test_language_and_both_keep_other_weights_and_roundtrip(self):
        for target in ("expert", "language", "both"):
            with self.subTest(target=target):
                torch.manual_seed(123)
                model = TinyDualPi05().eval()
                original = {key: value.clone() for key, value in model.state_dict().items()}
                before = sum(p.numel() for p in model.parameters())
                expert_head = model.paligemma_with_expert.gemma_expert.lm_head
                language_head = model.paligemma_with_expert.paligemma.lm_head
                manifest = prune_pi05_ffns(model, target=target, keep_fraction=.9, multiple_of=1)
                selected = ("expert", "language") if target == "both" else (target,)
                expected_removed = {"expert": 96, "language": 324}
                self.assertEqual(set(manifest["per_target"]), set(selected))
                self.assertEqual({item["part"] for item in manifest["layers"]}, set(selected))
                self.assertEqual(manifest["parameters_removed"], sum(expected_removed[part] for part in selected))
                self.assertEqual(manifest["model_parameters_before"], before)
                self.assertEqual(manifest["model_parameters_after"], sum(p.numel() for p in model.parameters()))
                for part in selected:
                    metrics = manifest["per_target"][part]
                    self.assertEqual(metrics["parameters_removed"], expected_removed[part])
                    self.assertEqual(metrics["ffn_parameter_reduction_pct"], 10.0)
                    self.assertAlmostEqual(metrics["model_parameter_reduction_pct"], 100 * expected_removed[part] / before)
                self.assertAlmostEqual(
                    manifest["model_parameter_reduction_pct"],
                    sum(item["model_parameter_reduction_pct"] for item in manifest["per_target"].values()),
                )
                selected_paths = {
                    "expert": "paligemma_with_expert.gemma_expert.model.layers.",
                    "language": "paligemma_with_expert.paligemma.language_model.layers.",
                }
                for key, value in model.state_dict().items():
                    if not any(key.startswith(selected_paths[part]) and ".mlp." in key for part in selected):
                        self.assertTrue(torch.equal(value, original[key]), key)
                self.assertIs(model.paligemma_with_expert.gemma_expert.lm_head, expert_head)
                self.assertIs(model.paligemma_with_expert.paligemma.lm_head, language_head)
                self.assertIs(language_head.weight, model.paligemma_with_expert.paligemma.language_model.embed_tokens.weight)
                fresh = TinyDualPi05().eval()
                prepare_pi05_pruned_shapes(fresh, manifest)
                fresh.load_state_dict(model.state_dict(), strict=True)
                for part in ("expert", "language"):
                    if part == "expert":
                        layers = model.paligemma_with_expert.gemma_expert.model.layers
                        restored = fresh.paligemma_with_expert.gemma_expert.model.layers
                        width = 8
                    else:
                        layers = model.paligemma_with_expert.paligemma.language_model.layers
                        restored = fresh.paligemma_with_expert.paligemma.language_model.layers
                        width = 12
                    x = torch.randn(2, 5, width)
                    for layer, other in zip(layers, restored):
                        torch.testing.assert_close(layer.mlp(x), other.mlp(x), rtol=0, atol=0)

    def test_estimate_matches_real_removal_without_mutating_model(self):
        for bias in (False, True):
            for multiple in (1, 4):
                with self.subTest(bias=bias, multiple_of=multiple):
                    torch.manual_seed(123)
                    model = TinyDualPi05(bias=bias).eval()
                    original = {key: value.clone() for key, value in model.state_dict().items()}
                    module_ids = {key: id(module) for key, module in model.named_modules()}
                    forecast = estimate_pi05_ffn_reduction(model, keep_fraction=.9, multiple_of=multiple)
                    self.assertEqual(module_ids, {key: id(module) for key, module in model.named_modules()})
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, original[key]), key)
                    for target in ("expert", "language", "both"):
                        compact = copy.deepcopy(model)
                        measured = prune_pi05_ffns(compact, target=target, keep_fraction=.9, multiple_of=multiple)
                        self.assertEqual(forecast[target]["parameters_removed"], measured["parameters_removed"])
                        self.assertAlmostEqual(
                            forecast[target]["model_parameter_reduction_pct"], measured["model_parameter_reduction_pct"],
                        )
                        for item, measured_item in zip(forecast[target]["layers"], measured["layers"]):
                            self.assertEqual(item["new_intermediate"], measured_item["new_intermediate"])
                            self.assertAlmostEqual(
                                item["actual_prune_fraction"], item["removed_channels"] / item["original_intermediate"],
                            )

    def test_projection_replacement_preserves_mode_and_gradient_flags(self):
        model = TinyDualPi05(bias=True).eval()
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.requires_grad_(False)
                if module.bias is not None:
                    module.bias.requires_grad_(False)
        manifest = prune_pi05_ffns(model, target="both", keep_fraction=.9, multiple_of=1)
        prepare_pi05_pruned_shapes(model, manifest)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.assertFalse(module.training)
                self.assertFalse(module.weight.requires_grad)
                if module.bias is not None:
                    self.assertFalse(module.bias.requires_grad)

    def test_rejects_invalid_target_and_non_linear_language_ffn(self):
        model = TinyDualPi05()
        with self.assertRaises(ValueError):
            prune_pi05_ffns(model, target="unknown")
        model.paligemma_with_expert.paligemma.language_model.layers[0].mlp.gate_proj = nn.Identity()
        with self.assertRaises(TypeError):
            prune_pi05_ffns(model, target="language")


if __name__ == "__main__":
    unittest.main()
