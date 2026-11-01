"""CPU regressions; header tests substitute safe_open, without installing safetensors.

Run from the repository root:
    python -B -m unittest discover -s quantization/w8a8 -p test_validation.py -v
"""

import contextlib
import copy
import csv
import importlib
import io
import json
import pathlib
import subprocess
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


class BenchmarkTest(unittest.TestCase):
    """Exercise reporting and comparison gates without loading any full model."""

    @classmethod
    def setUpClass(cls):
        cls.benchmark = importlib.import_module("benchmark_pi05_fp_w8a8")

    def make_results(self):
        signature = {
            "architecture": "pi05_libero", "framework": "pytorch", "batch_size": 1,
            "pytorch_compile_mode": None, "num_steps": 10, "seed": 42,
            "observation_sha256": "same-observation", "noise_sha256": "same-noise",
        }
        results = {}
        for name, multiplier, size in (("fp", 2, 800), ("w8a8", 1, 400)):
            results[name] = {
                "status": "passed", "signature": dict(signature),
                "latency_statistics": self.benchmark.latency_statistics([10 * multiplier, 20 * multiplier]),
                "checks": {
                    "policy_infer": "passed", "all_target_layers_called": "passed",
                    "int8_kernels": "passed",
                },
                "checkpoint_weights": {"bytes": size, "gib": size / 1024**3},
                "memory": {
                    phase: {
                        "allocated_bytes": size, "reserved_bytes": size * 2,
                        "peak_allocated_bytes": size * 2, "peak_reserved_bytes": size * 3,
                    }
                    for phase in ("loaded", "loading", "inference")
                },
            }
        return results

    def test_latency_statistics_interpolation_and_empty_values(self):
        stats = self.benchmark.latency_statistics([40, 10, 30, 20])
        self.assertEqual(stats["samples"], 4)
        for key, expected in (
            ("mean_ms", 25), ("p50_ms", 25), ("p95_ms", 38.5), ("min_ms", 10), ("max_ms", 40)
        ):
            self.assertAlmostEqual(stats[key], expected)
        empty = self.benchmark.latency_statistics([])
        self.assertEqual(empty["samples"], 0)
        self.assertTrue(all(empty[key] is None for key in ("mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms")))

    def test_latency_statistics_reject_invalid_measurements(self):
        for value in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.benchmark.latency_statistics([10, value])

    def test_speedup_uses_matching_statistics(self):
        comparison = self.benchmark.build_comparison(self.make_results())
        self.assertEqual(comparison["status"], "passed")
        for statistic in ("mean", "p50", "p95"):
            self.assertAlmostEqual(comparison["speedup"][statistic], 2)

    def test_failed_or_unmatched_runs_never_produce_speedup(self):
        variants = []
        for model in ("fp", "w8a8"):
            results = self.make_results()
            results[model]["status"] = "failed"
            variants.append(results)
        results = self.make_results()
        results["w8a8"]["signature"]["noise_sha256"] = "different-noise"
        variants.append(results)
        for key in ("policy_infer", "int8_kernels", "all_target_layers_called"):
            results = self.make_results()
            results["w8a8"]["checks"][key] = "not_run"
            variants.append(results)
        results = self.make_results()
        results["w8a8"]["latency_statistics"] = self.benchmark.latency_statistics([])
        variants.append(results)
        results = self.make_results()
        results["w8a8"]["latency_statistics"] = self.benchmark.latency_statistics([0, 0])
        variants.append(results)
        for index, results in enumerate(variants):
            with self.subTest(case=index):
                comparison = self.benchmark.build_comparison(results)
                self.assertEqual(comparison["status"], "not_run")
                self.assertTrue(all(value is None for value in comparison["speedup"].values()))

    def test_strict_json_and_csv_preserve_null_and_partial_rows(self):
        # Windows sandbox Temp can deny atomic rename; use the writable workspace.
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = pathlib.Path(directory)
            summary = path / "summary.json"
            data = {"status": "not_run", "latency_ms": None, "models": {"fp": {"failure_stage": "loading"}}}
            self.benchmark.write_json(summary, data)
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), data)
            with self.assertRaises(ValueError):
                self.benchmark.write_json(summary, {"invalid": float("nan")})
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), data)
            latency = path / "latency.csv"
            with latency.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(["model", "phase", "iteration", "latency_ms"])
            self.benchmark.append_latency(latency, "fp", "first_inference", 0, 42.5)
            self.benchmark.append_latency(latency, "w8a8", "measurement", 1, 7.25)
            with latency.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["model"], "fp")
            self.assertEqual(rows[0]["phase"], "first_inference")
            self.assertEqual(rows[0]["iteration"], "0")
            self.assertEqual(float(rows[0]["latency_ms"]), 42.5)
            self.assertEqual(float(rows[1]["latency_ms"]), 7.25)

    def test_kernel_evidence_requires_each_exact_target_to_run(self):
        names = ["language.layer", "expert.mlp"]
        before = {
            "layer_count": 2, "all_int8_probes_passed": True,
            "layers": [
                {
                    "name": name, "int8_probe_passed": True, "forward_calls": 0,
                    "operator_created": True, "weight_packed": True,
                    "qweight_dtype": "torch.int8", "device": "cuda:0",
                }
                for name in names
            ],
        }
        after = copy.deepcopy(before)
        for layer in after["layers"]:
            layer["forward_calls"] = 1
        self.benchmark.check_kernel_execution(before, after, names)
        variants = []
        uncalled = copy.deepcopy(after)
        uncalled["layers"][1]["forward_calls"] = 0
        variants.append(uncalled)
        substituted = copy.deepcopy(after)
        substituted["layers"][1]["name"] = "expert.attention"
        variants.append(substituted)
        failed_probe = copy.deepcopy(after)
        failed_probe["all_int8_probes_passed"] = False
        failed_probe["layers"][0]["int8_probe_passed"] = False
        variants.append(failed_probe)
        for index, invalid in enumerate(variants):
            with self.subTest(case=index), self.assertRaises(RuntimeError):
                self.benchmark.check_kernel_execution(before, invalid, names)

    def test_help_needs_no_model_or_gpu_dependencies(self):
        # Guard imports as well as checking exit status: installed CPU torch must
        # not hide an accidental heavyweight import in the help path.
        code = """import builtins
import runpy
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'openpi', 'numpy', 'bitblas'}:
        raise AssertionError('GPU/model dependency imported for --help: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
script = sys.argv[1]
sys.argv = [script, '--help']
runpy.run_path(script, run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, str(ROOT / "scripts/benchmark_pi05_fp_w8a8.py")],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)

    def test_missing_checkpoints_save_partial_results_and_reject_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = pathlib.Path(directory)
            output = path / "results"
            command = [
                sys.executable, "-B", str(ROOT / "scripts/benchmark_pi05_fp_w8a8.py"),
                "--w8a8-checkpoint", str(path / "missing-w8a8"), "--output-dir", str(output),
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual({item.name for item in output.iterdir()}, {"summary.json", "latency.csv", "benchmark.log"})
            original = (output / "summary.json").read_bytes()
            summary = json.loads(original)
            self.assertEqual(summary["status"], "incomplete")
            self.assertTrue(all(value is None for value in summary["comparison"]["speedup"].values()))
            for name in ("fp", "w8a8"):
                self.assertEqual(summary["models"][name]["failure_stage"], "checkpoint_preflight")
                self.assertIsNone(summary["models"][name]["latency_statistics"]["mean_ms"])
                self.assertTrue(summary["models"][name]["error"])
            with (output / "latency.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            self.assertIn("model.safetensors", (output / "benchmark.log").read_text(encoding="utf-8"))
            repeated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(repeated.returncode, 2)
            self.assertEqual((output / "summary.json").read_bytes(), original)

    def test_synthetic_inputs_and_noise_are_identical_for_repeated_runs(self):
        args = types.SimpleNamespace(seed=42, observation=None, prompt="pick up the book")
        config = types.SimpleNamespace(action_horizon=10, action_dim=32)
        observation, noise, hashes = self.benchmark._inputs(args, config)
        repeated, repeated_noise, repeated_hashes = self.benchmark._inputs(args, config)
        self.assertEqual(hashes, repeated_hashes)
        self.assertEqual(observation["prompt"], args.prompt)
        self.assertEqual(noise.shape, (10, 32))
        self.assertEqual(noise.dtype, np.float32)
        np.testing.assert_array_equal(noise, repeated_noise)
        for key in hashes:
            np.testing.assert_array_equal(observation[key], repeated[key])
        different = types.SimpleNamespace(seed=43, observation=None, prompt=args.prompt)
        _, different_noise, different_hashes = self.benchmark._inputs(different, config)
        self.assertNotEqual(hashes, different_hashes)
        self.assertFalse(np.array_equal(noise, different_noise))

    def test_recorded_observation_npz_preserves_arrays_and_shared_noise(self):
        config = types.SimpleNamespace(action_horizon=10, action_dim=32)
        recorded = {
            "observation/state": np.arange(8, dtype=np.float32),
            "observation/image": np.full((3, 4, 3), 17, dtype=np.uint8),
            "observation/wrist_image": np.full((4, 3, 3), 29, dtype=np.uint8),
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = pathlib.Path(directory) / "observation.npz"
            np.savez(path, **recorded)
            args = types.SimpleNamespace(seed=42, observation=path, prompt="recorded task")
            observation, noise, hashes = self.benchmark._inputs(args, config)
            repeated, repeated_noise, repeated_hashes = self.benchmark._inputs(args, config)
            self.assertEqual(hashes, repeated_hashes)
            self.assertEqual(observation["prompt"], "recorded task")
            np.testing.assert_array_equal(noise, repeated_noise)
            with np.load(path, allow_pickle=False) as unchanged:
                for key, expected in recorded.items():
                    np.testing.assert_array_equal(observation[key], expected)
                    np.testing.assert_array_equal(repeated[key], expected)
                    np.testing.assert_array_equal(unchanged[key], expected)

    def test_recorded_inputs_reject_invalid_state_images_and_pickle(self):
        config = types.SimpleNamespace(action_horizon=10, action_dim=32)
        recorded = {
            "observation/state": np.zeros(8, dtype=np.float32),
            "observation/image": np.zeros((3, 4, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((4, 3, 3), dtype=np.uint8),
        }
        invalid = (
            ("observation/state", np.zeros(7, dtype=np.float32)),
            ("observation/state", np.full(8, np.nan, dtype=np.float32)),
            ("observation/image", np.zeros((3, 4, 3), dtype=np.float32)),
            ("observation/wrist_image", np.zeros((3, 4, 4), dtype=np.uint8)),
            ("observation/image", np.zeros((0, 4, 3), dtype=np.uint8)),
            ("observation/state", np.array([{"object": True}], dtype=object)),
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = pathlib.Path(directory) / "invalid.npz"
            args = types.SimpleNamespace(seed=42, observation=path, prompt="task")
            for index, (key, value) in enumerate(invalid):
                arrays = {**recorded, key: value}
                np.savez(path, **arrays)
                with self.subTest(case=index), self.assertRaises(ValueError):
                    self.benchmark._inputs(args, config)

    def test_parameter_signature_compares_actual_fp_interfaces_of_replaced_layers(self):
        fp = torch.nn.Sequential(
            torch.nn.Linear(4, 3), torch.nn.Linear(3, 2, bias=False),
        ).to(torch.bfloat16)
        w8a8 = copy.deepcopy(fp)
        replace_w8a8_layers(w8a8, ["0"])
        signature = self.benchmark._parameter_signature(fp, [])
        self.assertEqual(signature, self.benchmark._parameter_signature(w8a8, ["0"]))
        self.assertEqual(signature["0.weight"], {"dtype": "torch.bfloat16", "shape": [3, 4]})
        self.assertEqual(signature["0.bias"], {"dtype": "torch.bfloat16", "shape": [3]})
        w8a8[1].to(torch.float32)
        self.assertNotEqual(signature, self.benchmark._parameter_signature(w8a8, ["0"]))


if __name__ == "__main__":
    unittest.main()
