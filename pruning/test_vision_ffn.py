from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from pruning.structured_ffn import prepare_pi05_pruned_shapes
from pruning.structured_ffn import prune_pi05_ffns


class TinySiglipMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, patches):
        return self.fc2(torch.nn.functional.gelu(self.fc1(patches)))


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        root = self.paligemma_with_expert
        root.gemma_expert = nn.Linear(4, 4)
        root.paligemma = nn.Module()
        root.paligemma.language_model = nn.Linear(4, 4)
        root.paligemma.vision_tower = nn.Module()
        root.paligemma.vision_tower.vision_model = nn.Module()
        vision = root.paligemma.vision_tower.vision_model
        vision.encoder = nn.Module()
        vision.encoder.layers = nn.ModuleList([nn.ModuleDict({"mlp": TinySiglipMLP()})])
        root.paligemma.multi_modal_projector = nn.Linear(4, 2048)

    def forward(self, patches):
        paligemma = self.paligemma_with_expert.paligemma
        patches = paligemma.vision_tower.vision_model.encoder.layers[0].mlp(patches)
        return paligemma.multi_modal_projector(patches)


class TestVisionFFN(unittest.TestCase):
    @torch.no_grad()
    def test_compact_siglip_slices_and_strict_roundtrip(self):
        model = TinyPolicy().eval()
        patches = torch.randn(2, 256, 4, generator=torch.Generator().manual_seed(123))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        dense = model(patches)
        manifest = prune_pi05_ffns(model, target="vision", keep_fraction=0.75, multiple_of=1)
        compact = model(patches)
        self.assertEqual(dense.shape, (2, 256, 2048))
        self.assertEqual(compact.shape, dense.shape)
        self.assertTrue(bool(torch.isfinite(compact).all()))
        self.assertEqual(manifest["parameters_removed"], 2 * (4 + 1 + 4))
        kept = manifest["layers"][0]["kept_indices"]
        for name, value in model.state_dict().items():
            expected = before[name]
            if name.endswith(("fc1.weight", "fc1.bias")):
                expected = expected[kept]
            elif name.endswith("fc2.weight"):
                expected = expected[:, kept]
            torch.testing.assert_close(value, expected, rtol=0, atol=0)

        restored = TinyPolicy().eval()
        prepare_pi05_pruned_shapes(restored, manifest)
        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp) / "pruned_state_dict.pt"
            torch.save(model.state_dict(), saved)
            state = torch.load(saved, map_location="cpu", weights_only=True, mmap=True)
            restored.load_state_dict(state, strict=True)
            del state
        torch.testing.assert_close(restored(patches), compact, rtol=0, atol=0)
