"""Offline public metadata and small CPU loading regressions; no model payload/GPU."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import inspect_w8a8
from openpi.models_pytorch.bitblas_w8a8_layers import replace_w8a8_layers
from openpi.policies.w8a8_policy import read_manifest
from openpi.policies.w8a8_policy import validate_state_dict

REFERENCE = pathlib.Path(__file__).with_name("reference")


class PublicCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((REFERENCE / "w8a8_config.json").read_text())
        self.header = json.loads((REFERENCE / "model.safetensors.header.json").read_text())
        self.candidate = json.loads(
            (REFERENCE.parent / "targets_pi05_libero_candidate.json").read_text(encoding="utf-8-sig")
        )["w8a8_layers"]

    def test_public_manifest_and_real_header_via_existing_inspector(self):
        slices = {
            key: types.SimpleNamespace(
                get_shape=lambda value=value: value["shape"],
                get_dtype=lambda value=value: value["dtype"],
            )
            for key, value in self.header.items()
        }
        reader = types.SimpleNamespace(keys=lambda: slices.keys(), get_slice=slices.__getitem__)
        module = types.ModuleType("safetensors")
        module.safe_open = lambda *args, **kwargs: contextlib.nullcontext(reader)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            (path / "w8a8_config.json").write_text(json.dumps(self.manifest))
            (path / "model.safetensors").touch()  # Header reader only is substituted.
            self.assertEqual(read_manifest(path), self.manifest["w8a8_layers"])
            output = io.StringIO()
            with mock.patch.dict(sys.modules, {"safetensors": module}), mock.patch.object(
                sys, "argv", ["inspect_w8a8.py", directory]
            ), contextlib.redirect_stdout(output):
                inspect_w8a8.main()
            report = json.loads(output.getvalue())
        self.assertEqual(report["layer_count"], 180)
        self.assertTrue(all(layer["scale_dtype"] == "F16" for layer in report["layers"]))
        self.assertEqual(report["kernel_initialization"], "not_run")

    def test_difference_is_exactly_72_expert_attention_projections(self):
        public = set(self.manifest["w8a8_layers"])
        expected = {
            f"paligemma_with_expert.gemma_expert.model.layers.{index}.self_attn.{projection}_proj"
            for index in range(18) for projection in ("q", "k", "v", "o")
        }
        self.assertEqual(len(self.candidate), 252)
        self.assertEqual(set(self.candidate) - public, expected)
        self.assertFalse(public - set(self.candidate))
        for name in expected:
            self.assertEqual(self.header[name + ".weight"]["dtype"], "BF16")
            self.assertNotIn(name + ".qweight", self.header)

    def test_public_selection_and_fp16_scales_strict_load_on_small_cpu_model(self):
        # Preserve real 252 module paths and public 180 selection, but use tiny
        # dimensions. This does not construct or validate the full PI0 model.
        model = torch.nn.Module()
        for name in self.candidate:
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, torch.nn.Module())
                parent = getattr(parent, part)
            parent.add_module(parts[-1], torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16))
        names = self.manifest["w8a8_layers"]
        untouched = {name: model.get_submodule(name) for name in set(self.candidate) - set(names)}
        replace_w8a8_layers(model, names)
        state = dict(model.state_dict())
        for name in names:
            state[name + ".qweight"] = torch.tensor([[1, -2], [-3, 4]], dtype=torch.int8)
            state[name + ".weight_scale"] = torch.tensor([0.125, 0.25], dtype=torch.float16)
        validate_state_dict(model, state, names)
        model.load_state_dict(state, strict=True)
        for name in names:
            self.assertEqual(model.get_submodule(name).weight_scale.dtype, torch.float32)
            self.assertTrue(torch.equal(model.get_submodule(name).weight_scale, state[name + ".weight_scale"].float()))
        for name, original in untouched.items():
            self.assertIs(model.get_submodule(name), original)
            self.assertEqual(original.weight.dtype, torch.bfloat16)
        bad_state = dict(state)
        del bad_state[names[0] + ".weight_scale"]
        with self.assertRaisesRegex(ValueError, "missing="):
            validate_state_dict(model, bad_state, names)

    def test_public_normalization_json_contract(self):
        stats = json.loads((REFERENCE / "norm_stats.json").read_text())["norm_stats"]
        for name, length in (("state", 8), ("actions", 7)):
            for field in ("mean", "std", "q01", "q99"):
                array = np.asarray(stats[name][field])
                self.assertEqual(array.shape, (length,))
                self.assertTrue(np.isfinite(array).all())
            self.assertTrue((np.asarray(stats[name]["std"]) >= 0).all())
            self.assertTrue((np.asarray(stats[name]["q99"]) >= np.asarray(stats[name]["q01"])).all())


if __name__ == "__main__":
    unittest.main()
