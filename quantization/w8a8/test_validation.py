"""CPU regressions; header tests substitute safe_open, without installing safetensors.

Run from the repository root:
    python -B -m unittest discover -s quantization/w8a8 -p test_validation.py -v
"""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import inspect_w8a8
from openpi.models_pytorch.bitblas_w8a8_layers import replace_w8a8_layers
from openpi.policies.w8a8_policy import prepare_checkpoint_state
from openpi.policies.w8a8_policy import validate_state_dict


class CheckpointAliasTest(unittest.TestCase):
    head = "paligemma_with_expert.paligemma.lm_head.weight"
    embedding = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"

    def make_model(self, *, tied=True):
        model = torch.nn.Module()
        for name in (self.head, self.embedding):
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, torch.nn.Module())
                parent = getattr(parent, part)
            parent.register_parameter(parts[-1], torch.nn.Parameter(torch.ones(4, 3)))
        if tied:
            model.get_submodule(self.head.rsplit(".", 1)[0]).weight = model.get_parameter(self.embedding)
        return model

    def test_restores_either_verified_alias(self):
        for omitted, retained in ((self.head, self.embedding), (self.embedding, self.head)):
            with self.subTest(omitted=omitted):
                model = self.make_model()
                state = {retained: torch.ones(4, 3)}
                prepare_checkpoint_state(model, state)
                self.assertIs(state[omitted], state[retained])
                validate_state_dict(model, state, [])
                model.load_state_dict(state, strict=True)

    def test_does_not_restore_untied_weights_or_both_missing(self):
        for tied, state in ((False, {self.head: torch.ones(4, 3)}), (True, {})):
            with self.subTest(tied=tied):
                model = self.make_model(tied=tied)
                prepare_checkpoint_state(model, state)
                with self.assertRaisesRegex(ValueError, "missing="):
                    validate_state_dict(model, state, [])

    def test_rejects_conflicting_tied_values(self):
        model = self.make_model()
        state = {self.head: torch.ones(4, 3), self.embedding: torch.zeros(4, 3)}
        with self.assertRaisesRegex(ValueError, "Conflicting checkpoint values"):
            prepare_checkpoint_state(model, state)


class ScaleValidationTest(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Sequential(torch.nn.Linear(8, 4))
        replace_w8a8_layers(self.model, ["0"])
        self.state = {
            "0.qweight": torch.zeros((4, 8), dtype=torch.int8),
            "0.weight_scale": torch.ones(4),
            "0.bias": torch.zeros(4),
        }

    def test_rejects_values_invalid_after_destination_conversion(self):
        for value in (1e100, 1e-100, 0.0, -1.0):
            with self.subTest(value=value):
                scale = torch.ones(4, dtype=torch.float64)
                scale[2] = value
                self.state["0.weight_scale"] = scale
                with self.assertRaisesRegex(
                    ValueError, r"0\.weight_scale:.*finite and greater than zero.*torch\.float32"
                ):
                    validate_state_dict(self.model, self.state, ["0"])

    def test_rejects_nonfinite_source_values(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.state["0.weight_scale"] = torch.tensor([1.0, value, 1.0, 1.0])
                with self.assertRaisesRegex(ValueError, r"0\.weight_scale: expected finite"):
                    validate_state_dict(self.model, self.state, ["0"])

    def test_valid_scales_load_as_positive_finite_fp32(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                scale = torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=dtype)
                self.state["0.weight_scale"] = scale
                validate_state_dict(self.model, self.state, ["0"])
                self.model.load_state_dict(self.state, strict=True)
                loaded = self.model[0].weight_scale
                self.assertEqual(loaded.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loaded).all().item())
                self.assertTrue((loaded > 0).all().item())
                self.assertTrue(torch.equal(loaded, scale.float()))

    def test_missing_alias_is_still_rejected(self):
        shared = torch.nn.Parameter(torch.ones(1))
        self.model.register_parameter("alias_source", shared)
        self.model.register_parameter("alias", shared)
        self.state["alias_source"] = shared.detach().clone()
        with self.assertRaisesRegex(ValueError, r"missing=\['alias'\]"):
            validate_state_dict(self.model, self.state, ["0"])


class HeaderValidationTest(unittest.TestCase):
    def inspect(self, dtype):
        # Exercise the real CLI branch and manifest reader. Only the unavailable
        # safetensors reader is substituted; no tensor payloads or GPU are used.
        weight = types.SimpleNamespace(get_shape=lambda: [4, 8], get_dtype=lambda: "I8")
        scale = types.SimpleNamespace(get_shape=lambda: [4], get_dtype=lambda: dtype)
        slices = {"0.qweight": weight, "0.weight_scale": scale}
        reader = types.SimpleNamespace(keys=lambda: slices.keys(), get_slice=slices.__getitem__)
        safe_open = mock.Mock(side_effect=lambda *args, **kwargs: contextlib.nullcontext(reader))
        module = types.ModuleType("safetensors")
        module.safe_open = safe_open
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            (path / "w8a8_config.json").write_text(
                json.dumps({"quantization": "w8a8_bitblas", "complete_model": True, "w8a8_layers": ["0"]}),
                encoding="utf-8",
            )
            (path / "model.safetensors").touch()
            output = io.StringIO()
            with mock.patch.dict(sys.modules, {"safetensors": module}), mock.patch.object(
                sys, "argv", ["inspect_w8a8.py", directory]
            ), contextlib.redirect_stdout(output):
                inspect_w8a8.main()
            safe_open.assert_called_once_with(str(path / "model.safetensors"), framework="pt", device="cpu")
            return json.loads(output.getvalue())

    def test_supported_float_headers_pass(self):
        for dtype in ("F16", "BF16", "F32", "F64"):
            with self.subTest(dtype=dtype):
                report = self.inspect(dtype)
                self.assertEqual(report["layers"][0]["scale_dtype"], dtype)
                self.assertEqual(report["layer_count"], 1)
                self.assertEqual(report["kernel_initialization"], "not_run")
                self.assertEqual(report["full_weight_and_norm_validation"], "not_run")

    def test_unsupported_headers_fail(self):
        for dtype in ("I8", "I32", "U8", "BOOL", "F8_E4M3", "UNKNOWN"):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, rf"0\.weight_scale: unsupported dtype {dtype}"):
                    self.inspect(dtype)


if __name__ == "__main__":
    unittest.main()
