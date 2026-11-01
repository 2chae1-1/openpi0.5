"""CPU integration tests with tiny random fixtures, never a real pi05 checkpoint."""

import dataclasses
import json
from pathlib import Path
import tempfile
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from pruning import run_structured_ffn as runner
from pruning.structured_ffn import prepare_pi05_pruned_shapes
from pruning.test_structured_ffn import TinyPi05


@dataclasses.dataclass
class TinyModelConfig:
    action_dim: int = 8
    action_horizon: int = 3
    max_token_len: int = 4
    dtype: str = "bfloat16"
    pytorch_compile_mode: str | None = None


class MockCheckpointModel(TinyPi05):
    """Random CPU fixture with real dense BF16 FFNs and untouched FP32 sentinels."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or TinyModelConfig()
        self.paligemma_with_expert.gemma_expert.to(torch.bfloat16)
        self.paligemma_with_expert.paligemma = nn.Linear(8, 8)
        self.action_out_proj = nn.Linear(8, 8)
        self.vision_sentinel = nn.Parameter(torch.randn(3))
        self.calls = 0

    def sample_actions(self, device, observation, noise=None, num_steps=10):
        self.calls += 1
        if torch.is_grad_enabled():
            raise AssertionError("Training-free pipeline must disable gradients during inference.")
        self.last_noise = noise.detach().clone()
        output = noise.to(torch.bfloat16)
        for layer in self.paligemma_with_expert.gemma_expert.model.layers:
            output = layer.mlp(output)
        return self.action_out_proj(output.float())


class FailingForwardModel(MockCheckpointModel):
    def sample_actions(self, *args, **kwargs):
        raise RuntimeError("Intentional synthetic baseline failure")


class FailingPrunedForwardModel(MockCheckpointModel):
    def sample_actions(self, *args, **kwargs):
        if self.calls:
            raise RuntimeError("Intentional synthetic compact forward failure")
        return super().sample_actions(*args, **kwargs)


class TestMockedCheckpointPipeline(unittest.TestCase):
    """These tests exercise orchestration, not official pi05_libero inference."""

    def setUp(self):
        torch.manual_seed(123)
        self.directory = tempfile.TemporaryDirectory(prefix="structured-ffn-cpu-test-", dir=Path(__file__).resolve().parent)
        self.addCleanup(self.directory.cleanup)
        self.output_dir = Path(self.directory.name) / "artifacts"
        self.args = runner.parse_args([
            "--checkpoint-dir", "MOCK_ONLY_NOT_PI05_CHECKPOINT",
            "--output-dir", str(self.output_dir),
            "--device", "cpu",
        ])
        self.config = SimpleNamespace(model=TinyModelConfig())

    def run_mock_checkpoint(self, model):
        with mock.patch.object(runner, "preflight", return_value={"issues": []}), mock.patch.object(
            runner, "load_original_bf16", return_value=(model, self.config, Path("MOCK_ONLY_NOT_PI05_CHECKPOINT"))
        ):
            return runner.run(self.args)

    def test_default_ten_percent_compaction_metrics_output_and_disk_roundtrip(self):
        model = MockCheckpointModel()
        original_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        before_count = sum(parameter.numel() for parameter in model.parameters())
        before_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

        report, exit_code = self.run_mock_checkpoint(model)
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["requested_prune_fraction"], 0.10)
        self.assertEqual(report["multiple_of"], 1)
        self.assertEqual(report["before"]["model"]["parameters"], before_count)
        self.assertEqual(report["before"]["model"]["weight_bytes"], before_bytes)
        self.assertEqual(report["before"]["expert_intermediate_sizes"], [16, 16])
        self.assertEqual(report["after"]["expert_intermediate_sizes"], [14, 14])
        # Tiny width rounding removes 2/16 channels; the requested ratio stays 10%.
        self.assertEqual(report["layers"][0]["actual_prune_fraction"], 0.125)
        self.assertEqual(report["parameters_removed"], 2 * 2 * 3 * 8)
        self.assertEqual(report["weight_bytes_removed"], report["parameters_removed"] * 2)
        self.assertEqual(report["after"]["model"]["parameters"], before_count - 96)
        self.assertEqual(report["after"]["model"]["weight_bytes"], before_bytes - 192)
        for phase in ("before", "after"):
            self.assertEqual(report["outputs"][phase]["status"], "passed")
            self.assertEqual(report["outputs"][phase]["shape"], [1, 3, 8])
            self.assertTrue(report["outputs"][phase]["all_finite"])
        self.assertTrue(report["verification"]["output_shapes_match"])
        self.assertEqual(report["verification"]["libero_accuracy"]["status"], "unverified")
        # Non-expert weights must remain bit-exact, including their original dtype.
        for name, value in model.state_dict().items():
            if "gemma_expert.model.layers" not in name:
                self.assertTrue(torch.equal(value, original_state[name]), name)
                self.assertEqual(value.dtype, original_state[name].dtype)
        self.assertIn("torch.bfloat16", report["after"]["model"]["by_dtype"])
        self.assertIn("torch.float32", report["after"]["model"]["by_dtype"])

        self.assertEqual(json.loads((self.output_dir / runner.REPORT_NAME).read_text(encoding="utf-8")), report)
        manifest = json.loads((self.output_dir / runner.MANIFEST_NAME).read_text(encoding="utf-8"))
        state = torch.load(self.output_dir / runner.STATE_DICT_NAME, map_location="cpu", weights_only=True)
        self.assertEqual(manifest["model_config"], dataclasses.asdict(self.config.model))
        self.assertEqual(manifest["state_dict_file"], runner.STATE_DICT_NAME)
        self.assertEqual(manifest["target"], "expert")
        fresh = MockCheckpointModel()
        prepare_pi05_pruned_shapes(fresh, manifest)
        fresh.load_state_dict(state, strict=True)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, fresh.state_dict()[name]), name)
        observation, noise = runner.make_smoke_inputs(self.config.model, torch.device("cpu"), 1, 123)
        with torch.inference_mode():
            expected = model.sample_actions(torch.device("cpu"), observation, noise=noise)
            restored = fresh.sample_actions(torch.device("cpu"), observation, noise=noise)
        self.assertTrue(torch.equal(expected, restored))
        # Check the public restore helper too, replacing only the unavailable model class.
        fake_module = ModuleType("openpi.models_pytorch.pi0_pytorch")
        fake_module.PI0Pytorch = MockCheckpointModel
        with mock.patch.dict("sys.modules", {"openpi.models_pytorch.pi0_pytorch": fake_module}), mock.patch.object(
            runner, "get_train_config", return_value=self.config
        ), mock.patch.object(nn.Module, "to_bfloat16_for_selected_params", create=True):
            restored_model = runner.load_pruned_model(self.output_dir)
        self.assertFalse(restored_model.training)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored_model.state_dict()[name]), name)
            self.assertEqual(value.dtype, restored_model.state_dict()[name].dtype)

    def test_missing_prerequisites_are_unverified_and_do_not_load_model(self):
        with mock.patch.object(runner, "preflight", return_value={"issues": ["Missing synthetic checkpoint"]}), mock.patch.object(
            runner, "load_original_bf16"
        ) as load:
            report, exit_code = runner.run(self.args)
        load.assert_not_called()
        self.assertEqual(exit_code, 2)
        self.assertEqual(report["status"], "unverified")
        self.assertIsNone(report["before"])
        self.assertIsNone(report["after"])
        self.assertFalse(report["verification"]["original_checkpoint_loaded"])
        for phase in ("before", "after"):
            self.assertIsNone(report["outputs"][phase]["shape"])
            self.assertEqual(report["outputs"][phase]["status"], "unverified")
        self.assertEqual(list(self.output_dir.iterdir()), [self.output_dir / runner.REPORT_NAME])

    def test_failed_baseline_saves_pruning_without_claiming_forward_success(self):
        report, exit_code = self.run_mock_checkpoint(FailingForwardModel())
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["outputs"]["before"]["status"], "failed")
        self.assertIn("Intentional synthetic baseline failure", report["outputs"]["before"]["error"])
        self.assertIsNone(report["outputs"]["after"]["shape"])
        self.assertIsNone(report["verification"]["output_shapes_match"])
        self.assertTrue(report["verification"]["physical_pruning"])
        self.assertTrue((self.output_dir / runner.STATE_DICT_NAME).is_file())

    def test_failed_compact_forward_keeps_saved_artifacts_and_failure_report(self):
        with self.assertLogs(level="ERROR"):
            report, exit_code = self.run_mock_checkpoint(FailingPrunedForwardModel())
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["stage"], "pruned_forward")
        self.assertEqual(report["outputs"]["before"]["status"], "passed")
        self.assertIsNone(report["outputs"]["after"]["shape"])
        self.assertTrue((self.output_dir / runner.STATE_DICT_NAME).is_file())

    def test_skip_forward_reports_unverified_outputs_and_optional_state_skip(self):
        self.args.skip_forward = True
        self.args.no_save_state_dict = True
        model = MockCheckpointModel()
        report, exit_code = self.run_mock_checkpoint(model)
        self.assertEqual(exit_code, 2)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(model.calls, 0)
        self.assertEqual(report["outputs"]["before"]["status"], "unverified")
        self.assertEqual(report["outputs"]["after"]["status"], "unverified")
        self.assertEqual(report["artifacts"]["manifest"]["status"], "saved")
        self.assertEqual(report["artifacts"]["state_dict"]["status"], "skipped")
        manifest = json.loads((self.output_dir / runner.MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertIsNone(manifest["state_dict_file"])
        self.assertFalse((self.output_dir / runner.STATE_DICT_NAME).exists())

    def test_existing_report_prevents_stale_artifact_reuse(self):
        self.output_dir.mkdir()
        old_report = self.output_dir / runner.REPORT_NAME
        old_report.write_text('{"status":"old_report_sentinel"}', encoding="utf-8")
        with self.assertRaises(FileExistsError), mock.patch.object(runner, "load_original_bf16") as load:
            runner.run(self.args)
        load.assert_not_called()
        self.assertEqual(json.loads(old_report.read_text(encoding="utf-8"))["status"], "old_report_sentinel")

    def test_cli_defaults_and_invalid_pruning_arguments(self):
        args = runner.parse_args(["--checkpoint-dir", "unused", "--output-dir", "unused"])
        self.assertEqual(args.prune_fraction, 0.1)
        self.assertEqual(args.multiple_of, 1)
        self.assertEqual(args.device, "cuda")
        for option, value in (("--prune-fraction", "0"), ("--prune-fraction", "1"), ("--multiple-of", "0"), ("--num-steps", "0")):
            with self.subTest(option=option, value=value), mock.patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                runner.parse_args(["--checkpoint-dir", "unused", "--output-dir", "unused", option, value])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
