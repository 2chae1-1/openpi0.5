import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from pruning.structured_ffn import prune_gemma_mlp, prune_pi05_ffns, prepare_pi05_pruned_shapes


class TinyGemmaMLP(nn.Module):
    """Matches the 3-projection gate/up/down arrangement in OpenPI modeling_gemma.py."""
    def __init__(self, width=8, intermediate=16):
        super().__init__()
        self.intermediate_size = intermediate
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

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


if __name__ == "__main__":
    unittest.main()
