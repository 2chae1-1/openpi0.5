# ruff: noqa: PT009, PT027 -- Keep stdlib unittest assertions and their existing tolerances.
"""Windows CPU regressions; no checkpoint download, model load or CUDA execution.

Run: python -B -m unittest discover -s quantization/w8a8 -p test_benchmark.py -v
Reporting/input tests reuse the development branch's BenchmarkTest cases.
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

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
benchmark = importlib.import_module("benchmark_pi05_fp_w8a8")


class BenchmarkTest(unittest.TestCase):
    def make_results(self):
        signature = {"architecture": "pi05_libero", "noise_hash": "same-noise"}
        results = {}
        for name, multiplier, size in (("fp", 2, 800), ("w8a8", 1, 400)):
            results[name] = {
                "status": "passed",
                "signature": dict(signature),
                "latency_statistics": benchmark.latency_statistics([10 * multiplier, 20 * multiplier]),
                "checks": {
                    "policy_infer": "passed",
                    "all_target_layers_called": "passed",
                    "int8_kernels": "passed",
                },
                "checkpoint_weights": {"bytes": size, "gib": size / 1024**3},
                "memory": {
                    phase: {
                        "allocated_bytes": size,
                        "reserved_bytes": size * 2,
                        "peak_allocated_bytes": size * 2,
                        "peak_reserved_bytes": size * 3,
                    }
                    for phase in ("loaded", "loading", "inference")
                },
            }
        return results

    def assert_unmeasured(self, comparison):
        self.assertEqual(comparison["status"], "not_measured")
        self.assertTrue(comparison["reasons"])
        self.assertTrue(all(value is None for value in comparison["speedup"].values()))
        self.assertTrue(all(value is None for value in comparison["checkpoint_size"].values()))
        for phase in comparison["memory"].values():
            for metric in phase.values():
                self.assertTrue(all(value is None for value in metric.values()))

    def test_latency_statistics_interpolate_and_preserve_empty(self):
        stats = benchmark.latency_statistics([40, 10, 30, 20])
        self.assertEqual(stats["samples"], 4)
        for key, expected in (("mean_ms", 25), ("p50_ms", 25), ("p95_ms", 38.5)):
            self.assertAlmostEqual(stats[key], expected)
        empty = benchmark.latency_statistics([])
        self.assertEqual(empty["samples"], 0)
        self.assertTrue(all(value is None for key, value in empty.items() if key != "samples"))

    def test_latency_statistics_reject_invalid_measurements(self):
        for value in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark.latency_statistics([10, value])

    def test_matching_successful_runs_allow_comparison(self):
        comparison = benchmark.build_comparison(self.make_results())
        self.assertEqual(comparison["status"], "passed")
        for statistic in ("mean", "p50", "p95"):
            self.assertAlmostEqual(comparison["speedup"][statistic], 2)
        self.assertEqual(comparison["checkpoint_size"]["reduction_fraction"], 0.5)
        self.assertEqual(comparison["checkpoint_size"]["compression_ratio"], 2)

    def test_missing_fp_never_estimates_any_improvement(self):
        results = self.make_results()
        results["fp"] = benchmark._empty_result(None, None, None)  # noqa: SLF001 -- Exercise report fixtures.
        results["fp"]["status"] = "not_measured"
        self.assert_unmeasured(benchmark.build_comparison(results))

    def test_failed_or_unmatched_runs_never_produce_improvement(self):
        variants = []
        for name in ("fp", "w8a8"):
            results = self.make_results()
            results[name]["status"] = "failed"
            variants.append(results)
        results = self.make_results()
        results["w8a8"]["signature"]["noise_hash"] = "different-noise"
        variants.append(results)
        for key in ("policy_infer", "int8_kernels", "all_target_layers_called"):
            results = self.make_results()
            results["w8a8"]["checks"][key] = "not_run"
            variants.append(results)
        for values in ([], [0, 0]):
            results = self.make_results()
            results["w8a8"]["latency_statistics"] = benchmark.latency_statistics(values)
            variants.append(results)
        for index, results in enumerate(variants):
            with self.subTest(case=index):
                self.assert_unmeasured(benchmark.build_comparison(results))

    def test_strict_json_preserves_last_valid_report(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = pathlib.Path(directory) / "summary.json"
            data = {"status": "not_measured", "latency_ms": None}
            benchmark.write_json(path, data)
            with self.assertRaises(ValueError):
                benchmark.write_json(path, {"invalid": float("nan")})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), data)

    def test_summary_csv_preserves_unmeasured_as_blank(self):
        models = self.make_results()
        models["fp"] = benchmark._empty_result(None, None, None)  # noqa: SLF001 -- Exercise report fixtures.
        models["fp"]["status"] = "not_measured"
        summary = {"models": models, "comparison": benchmark.build_comparison(models)}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = pathlib.Path(directory) / "summary.csv"
            benchmark.write_summary_csv(path, summary)
            with path.open(encoding="utf-8", newline="") as stream:
                rows = {row["model"]: row for row in csv.DictReader(stream)}
        self.assertEqual(rows["fp"]["status"], "not_measured")
        for field in ("checkpoint_bytes", "mean_ms", "p50_ms", "p95_ms"):
            self.assertEqual(rows["fp"][field], "")
        self.assertEqual(int(rows["w8a8"]["checkpoint_bytes"]), 400)
        self.assertEqual(float(rows["w8a8"]["mean_ms"]), 15)

    def test_latency_csv_retains_distinct_phases(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = pathlib.Path(directory) / "latency.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(["model", "phase", "iteration", "latency_ms"])
            for phase, latency in (("first", 42.5), ("warmup", 20.0), ("measurement", 7.25)):
                benchmark.append_latency(path, "w8a8", phase, 0, latency)
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual([row["phase"] for row in rows], ["first", "warmup", "measurement"])
        self.assertEqual(float(rows[-1]["latency_ms"]), 7.25)

    def test_kernel_evidence_requires_int8_cuda_and_each_exact_target_to_run(self):
        names = ["language.layer", "expert.mlp"]
        before = {
            "layer_count": 2,
            "all_int8_probes_passed": True,
            "layers": [
                {
                    "name": name,
                    "int8_probe_passed": True,
                    "forward_calls": 0,
                    "operator_created": True,
                    "weight_packed": True,
                    "qweight_dtype": "torch.int8",
                    "device": "cuda:0",
                    "activation_dtype": "int8",
                    "accum_dtype": "int32",
                }
                for name in names
            ],
        }
        after = copy.deepcopy(before)
        for layer in after["layers"]:
            layer["forward_calls"] = 1
            layer["last_activation_dtype"] = "torch.int8"
            layer["last_output_dtype"] = "torch.int32"
        benchmark.check_kernel_execution(before, after, names)
        invalid_values = {
            "forward_calls": 0,
            "name": "expert.attention",
            "int8_probe_passed": False,
            "operator_created": False,
            "weight_packed": False,
            "qweight_dtype": "torch.float32",
            "device": "cpu",
            "last_activation_dtype": "torch.float32",
            "last_output_dtype": "torch.float32",
        }
        for key, value in invalid_values.items():
            invalid = copy.deepcopy(after)
            invalid["layers"][1][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                benchmark.check_kernel_execution(before, invalid, names)

    def test_help_needs_no_model_gpu_or_nvml_dependencies(self):
        code = """import builtins
import runpy
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'openpi', 'numpy', 'bitblas', 'pynvml'}:
        raise AssertionError('Heavy dependency imported for --help: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
script = sys.argv[1]
sys.argv = [script, '--help']
runpy.run_path(script, run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, str(ROOT / "scripts/benchmark_pi05_fp_w8a8.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)

    def test_forward_records_actual_int8_operator_arguments_on_cpu(self):
        import torch

        sys.path.insert(0, str(ROOT / "src"))
        from openpi.models_pytorch.bitblas_w8a8_layers import BitBLASW8A8Linear
        from openpi.models_pytorch.bitblas_w8a8_layers import w8a8_diagnostics

        layer = BitBLASW8A8Linear(torch.nn.Linear(4, 2, bias=False), "0")
        layer.qweight.copy_(torch.tensor([[1, 0, -1, 2], [0, 1, 2, -1]], dtype=torch.int8))
        layer.weight_scale.fill_(0.5)
        layer._packed_weight = layer.qweight  # noqa: SLF001 -- CPU fake operator fixture.
        layer.kernel_verified = True
        calls = []

        def fake_bitblas(activation, packed, *, output):
            self.assertEqual(activation.dtype, torch.int8)
            self.assertEqual(output.dtype, torch.int32)
            calls.append(activation.clone())
            output.copy_(activation.to(torch.int32) @ packed.to(torch.int32).T)

        layer.bitblas_matmul = fake_bitblas
        inputs = torch.tensor([[1.0, -2.0, 3.0, 0.0]])
        actual = layer(inputs)
        self.assertEqual(len(calls), 1)
        expected = (calls[0].float() @ layer.qweight.float().T) * (3.0 / 127.0 * 0.5)
        torch.testing.assert_close(actual, expected)
        evidence = w8a8_diagnostics(torch.nn.Sequential(layer))["layers"][0]
        self.assertEqual(evidence["forward_calls"], 1)
        self.assertEqual(evidence["last_activation_dtype"], "torch.int8")
        self.assertEqual(evidence["last_output_dtype"], "torch.int32")
        self.assertFalse(w8a8_diagnostics(torch.nn.Sequential(layer))["profiler_verified"])

    def test_missing_w8a8_saves_failure_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = pathlib.Path(directory)
            output = path / "results"
            command = [
                sys.executable,
                "-B",
                str(ROOT / "scripts/benchmark_pi05_fp_w8a8.py"),
                "--w8a8-checkpoint",
                str(path / "missing-w8a8"),
                "--output-dir",
                str(output),
                "--fp-checkpoint",
                str(path / "missing-fp"),
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue(
                {"summary.json", "summary.csv", "latency.csv", "benchmark.log"}
                <= {item.name for item in output.iterdir()}
            )
            original = (output / "summary.json").read_bytes()
            summary = json.loads(original)
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["models"]["fp"]["status"], "not_measured")
            w8a8 = summary["models"]["w8a8"]
            self.assertEqual(w8a8["status"], "failed")
            self.assertEqual(w8a8["failure_stage"], "checkpoint_preflight")
            self.assertIsNone(w8a8["latency_statistics"]["mean_ms"])
            self.assert_unmeasured(summary["comparison"])
            with (output / "latency.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            repeated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(repeated.returncode, 2)
            self.assertEqual((output / "summary.json").read_bytes(), original)

    def test_cpu_target_saves_environment_error_without_cuda_calls(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = pathlib.Path(directory)
            checkpoint = path / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"preflight-only")
            output = path / "results"
            output.mkdir()
            summary_path = output / "summary.json"
            benchmark.write_json(
                summary_path,
                {
                    "models": {
                        "w8a8": benchmark._empty_result(checkpoint, None, None),  # noqa: SLF001 -- Exercise report fixtures.
                    }
                },
            )
            args = benchmark._parser().parse_args(  # noqa: SLF001 -- Exercise worker preflight without a subprocess.
                [
                    "--w8a8-checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                    "--_worker",
                    "w8a8",
                ]
            )
            fake_torch = types.ModuleType("torch")
            fake_torch.device = lambda value: types.SimpleNamespace(type="cpu")
            fake_torch.cuda = mock.Mock()
            with mock.patch.dict(sys.modules, {"torch": fake_torch}), self.assertLogs(level="ERROR"):
                self.assertEqual(benchmark._worker(args), 1)  # noqa: SLF001 -- Exercise worker preflight.
            self.assertEqual(fake_torch.cuda.mock_calls, [])
            result = json.loads(summary_path.read_text(encoding="utf-8"))["models"]["w8a8"]
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure_stage"], "environment")
            self.assertEqual(result["checkpoint_weights"]["bytes"], len(b"preflight-only"))
            self.assertIsNone(result["latency_statistics"]["mean_ms"])

    def test_w8a8_only_coordinator_passes_without_launching_fp(self):
        launched = []
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = pathlib.Path(directory)
            output = path / "results"

            def fake_process(command, **kwargs):
                name = command[command.index("--_worker") + 1]
                launched.append(name)

                def wait():
                    summary_path = output / "summary.json"
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    summary["models"][name].update(self.make_results()[name])
                    benchmark.write_json(summary_path, summary)
                    return 0

                return types.SimpleNamespace(wait=wait, returncode=0)

            with (
                mock.patch.object(benchmark.subprocess, "Popen", side_effect=fake_process),
                mock.patch.object(benchmark, "_git_commit", return_value="cpu-test"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = benchmark.main(
                    [
                        "--w8a8-checkpoint",
                        str(path / "checkpoint"),
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(launched, ["w8a8"])
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["models"]["fp"]["status"], "not_measured")
            self.assert_unmeasured(summary["comparison"])

    def test_synthetic_inputs_and_noise_are_repeatable(self):
        args = types.SimpleNamespace(seed=42, observation=None, prompt="pick up the book")
        config = types.SimpleNamespace(action_horizon=10, action_dim=32)
        observation, noise, hashes = benchmark._inputs(args, config)  # noqa: SLF001 -- Verify deterministic inputs.
        repeated, repeated_noise, repeated_hashes = benchmark._inputs(args, config)  # noqa: SLF001
        self.assertEqual(hashes, repeated_hashes)
        self.assertEqual(observation["prompt"], args.prompt)
        self.assertEqual(noise.shape, (10, 32))
        self.assertEqual(noise.dtype, np.float32)
        np.testing.assert_array_equal(noise, repeated_noise)
        for key in hashes:
            np.testing.assert_array_equal(observation[key], repeated[key])

    def test_worker_crash_before_json_update_preserves_failed_json_and_csv(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = pathlib.Path(directory)
            output = path / "results"
            process = types.SimpleNamespace(wait=lambda: 7, returncode=7)
            with (
                mock.patch.object(benchmark.subprocess, "Popen", return_value=process),
                mock.patch.object(benchmark, "_git_commit", return_value="cpu-test"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = benchmark.main(
                    [
                        "--w8a8-checkpoint",
                        str(path / "checkpoint"),
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(code, 1)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["models"]["fp"]["status"], "not_measured")
            w8a8 = summary["models"]["w8a8"]
            self.assertEqual(w8a8["status"], "failed")
            self.assertEqual(w8a8["process_exit_code"], 7)
            self.assertTrue(w8a8["failure_stage"])
            self.assertTrue(w8a8["error"])
            self.assertIsNone(w8a8["latency_statistics"]["mean_ms"])
            self.assert_unmeasured(summary["comparison"])
            with (output / "summary.csv").open(encoding="utf-8", newline="") as stream:
                rows = {row["model"]: row for row in csv.DictReader(stream)}
            self.assertEqual(rows["w8a8"]["status"], "failed")
            self.assertEqual(rows["w8a8"]["mean_ms"], "")


class VramSamplerTest(unittest.TestCase):
    def fake_nvml(self, process_bytes=100, device_bytes=1000):
        state = {"process_bytes": process_bytes, "device_bytes": device_bytes}
        nvml = types.ModuleType("pynvml")
        nvml.nvmlInit = mock.Mock()
        nvml.nvmlShutdown = mock.Mock()
        nvml.nvmlDeviceGetHandleByUUID = mock.Mock(return_value="uuid-handle")
        nvml.nvmlDeviceGetCount = mock.Mock(return_value=2)
        nvml.nvmlDeviceGetHandleByIndex = mock.Mock(side_effect=lambda index: f"physical-{index}")
        nvml.nvmlDeviceGetMemoryInfo = mock.Mock(
            side_effect=lambda handle: types.SimpleNamespace(used=state["device_bytes"]),
        )
        nvml.nvmlDeviceGetComputeRunningProcesses = mock.Mock(
            side_effect=lambda handle: [
                types.SimpleNamespace(pid=benchmark.os.getpid(), usedGpuMemory=state["process_bytes"]),
                types.SimpleNamespace(pid=-1, usedGpuMemory=999999),
            ]
        )
        return nvml, state

    def test_uuid_selection_and_process_device_peaks_are_separate(self):
        nvml, state = self.fake_nvml()
        with mock.patch.dict(sys.modules, {"pynvml": nvml}), mock.patch.object(benchmark.threading, "Thread"):
            sampler = benchmark.VramSampler("GPU-test-uuid")
            state.update(process_bytes=400, device_bytes=4000)
            snapshot = sampler.snapshot(phase="inference")
            state.update(process_bytes=200, device_bytes=2000)
            final = sampler.stop()
        nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-test-uuid")
        nvml.nvmlDeviceGetCount.assert_not_called()
        nvml.nvmlDeviceGetHandleByIndex.assert_not_called()
        nvml.nvmlShutdown.assert_called_once()
        self.assertEqual(final["status"], "sampled")
        self.assertEqual(final["sample_interval_ms"], 50)
        self.assertEqual(final["process_used_bytes"], 200)
        self.assertEqual(final["device_used_bytes"], 2000)
        self.assertEqual(final["process_peak_used_bytes"], 400)
        self.assertEqual(final["device_peak_used_bytes"], 4000)
        self.assertEqual(final["phases"]["loading"]["process_peak_used_bytes"], 100)
        self.assertEqual(final["phases"]["inference"]["process_peak_used_bytes"], 400)
        self.assertEqual(snapshot["process_used_bytes"], 400)

    def test_unavailable_process_memory_never_becomes_huge_or_zero(self):
        nvml, _ = self.fake_nvml(process_bytes=2**64 - 1)
        with mock.patch.dict(sys.modules, {"pynvml": nvml}), mock.patch.object(benchmark.threading, "Thread"):
            final = benchmark.VramSampler("GPU-test-uuid").stop()
        self.assertEqual(final["status"], "partial")
        self.assertIsNone(final["process_used_bytes"])
        self.assertIsNone(final["process_peak_used_bytes"])
        self.assertEqual(final["device_peak_used_bytes"], 1000)
        self.assertEqual(final["process_samples"], 0)
        self.assertTrue(final["errors"])

    def test_missing_nvml_is_reported_as_unmeasured(self):
        with mock.patch.dict(sys.modules, {"pynvml": None}):
            final = benchmark.VramSampler("GPU-test-uuid").stop()
        self.assertEqual(final["status"], "not_measured")
        self.assertIsNone(final["process_peak_used_bytes"])
        self.assertIsNone(final["device_peak_used_bytes"])
        self.assertTrue(final["errors"])

    def test_ambiguous_pid_mapping_does_not_guess_a_cuda_ordinal(self):
        nvml, _ = self.fake_nvml()
        with mock.patch.dict(sys.modules, {"pynvml": nvml}), mock.patch.object(benchmark.threading, "Thread") as thread:
            final = benchmark.VramSampler(None).stop()
        self.assertEqual(final["status"], "not_measured")
        self.assertIsNone(final["process_peak_used_bytes"])
        self.assertIsNone(final["device_peak_used_bytes"])
        self.assertIn("unambiguously", final["errors"][0])
        thread.assert_not_called()
        nvml.nvmlDeviceGetHandleByUUID.assert_not_called()
        nvml.nvmlDeviceGetMemoryInfo.assert_not_called()

    def test_unique_pid_mapping_uses_the_matching_physical_gpu(self):
        nvml, _ = self.fake_nvml()
        nvml.nvmlDeviceGetComputeRunningProcesses.side_effect = lambda handle: [
            types.SimpleNamespace(
                pid=benchmark.os.getpid() if handle == "physical-1" else -1,
                usedGpuMemory=100,
            ),
        ]
        with mock.patch.dict(sys.modules, {"pynvml": nvml}), mock.patch.object(benchmark.threading, "Thread"):
            final = benchmark.VramSampler(None).stop()
        self.assertEqual(final["status"], "sampled")
        nvml.nvmlDeviceGetMemoryInfo.assert_called_with("physical-1")


if __name__ == "__main__":
    unittest.main()
