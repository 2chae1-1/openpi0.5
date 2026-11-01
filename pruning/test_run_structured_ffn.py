"""CPU integration tests with tiny random fixtures, never a real pi05 checkpoint."""

import dataclasses
from contextlib import ExitStack
import json
from pathlib import Path
import sys
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
        self.paligemma_with_expert.gemma_expert.lm_head = nn.Linear(8, 12, bias=False)
        self.paligemma_with_expert.gemma_expert.to(torch.bfloat16)
        self.paligemma_with_expert.paligemma = nn.Linear(8, 8)
        self.action_out_proj = nn.Linear(8, 8)
        self.vision_sentinel = nn.Parameter(torch.randn(3))
        self.calls = 0
        self.inference_inputs = []

    def sample_actions(self, device, observation, noise=None, num_steps=10):
        self.calls += 1
        if torch.is_grad_enabled():
            raise AssertionError("Training-free pipeline must disable gradients during inference.")
        self.last_noise = noise.detach().clone()
        self.inference_inputs.append((observation, self.last_noise, num_steps))
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


class MismatchedRestoredModel(MockCheckpointModel):
    def sample_actions(self, *args, **kwargs):
        return super().sample_actions(*args, **kwargs) + 1


class NonfiniteRestoredModel(MockCheckpointModel):
    def sample_actions(self, *args, **kwargs):
        return super().sample_actions(*args, **kwargs) * float("nan")


class WrongShapeRestoredModel(MockCheckpointModel):
    def sample_actions(self, *args, **kwargs):
        return super().sample_actions(*args, **kwargs)[..., :-1]


class WrongDtypeRestoredModel(MockCheckpointModel):
    def sample_actions(self, *args, **kwargs):
        return super().sample_actions(*args, **kwargs).double()


class FailingRestoredModel(MockCheckpointModel):
    def sample_actions(self, *args, **kwargs):
        raise RuntimeError("Intentional synthetic restored forward failure")


class TiedHeadModel(MockCheckpointModel):
    def __init__(self, config=None):
        super().__init__(config)
        expert = self.paligemma_with_expert.gemma_expert
        expert.embed_tokens = nn.Embedding(12, 8, dtype=torch.bfloat16)
        expert.lm_head.weight = expert.embed_tokens.weight


class TestMockedCheckpointPipeline(unittest.TestCase):
    """These tests exercise orchestration, not official pi05_libero inference."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Thread-pool overhead dominates these tiny CPU-only fixtures.
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)

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

    def run_mock_checkpoint(self, model, *, restore_class=MockCheckpointModel, artifact_edit=None):
        """Use real artifact save/load; replace only unavailable OpenPI dependencies."""
        self.restored_models = []
        fake_module = ModuleType("openpi.models_pytorch.pi0_pytorch")

        def create_restore_model(config):
            restored = restore_class(config)
            self.restored_models.append(restored)
            return restored

        fake_module.PI0Pytorch = create_restore_model
        save = runner.save_pruned_artifacts

        def save_and_edit(*args, **kwargs):
            artifacts = save(*args, **kwargs)
            artifact_edit(self.output_dir)
            return artifacts

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "preflight", return_value={"issues": []}))
            stack.enter_context(mock.patch.object(
                runner, "load_original_bf16", return_value=(model.eval(), self.config, Path("MOCK_ONLY_NOT_PI05_CHECKPOINT"))
            ))
            # Replace only this key: patch.dict would remove unrelated lazily
            # imported Torch modules and can duplicate registered CPU kernels.
            module_name = "openpi.models_pytorch.pi0_pytorch"
            missing = object()
            previous_module = sys.modules.get(module_name, missing)
            sys.modules[module_name] = fake_module

            def restore_module():
                if previous_module is missing:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = previous_module

            stack.callback(restore_module)
            stack.enter_context(mock.patch.object(runner, "get_train_config", return_value=self.config))
            stack.enter_context(mock.patch.object(nn.Module, "to_bfloat16_for_selected_params", create=True))
            self.restore_loader = stack.enter_context(mock.patch.object(
                runner, "load_pruned_model", wraps=runner.load_pruned_model
            ))
            if artifact_edit is not None:
                stack.enter_context(mock.patch.object(runner, "save_pruned_artifacts", side_effect=save_and_edit))
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
        for phase in ("before", "after", "restored"):
            self.assertEqual(report["outputs"][phase]["status"], "passed")
            self.assertEqual(report["outputs"][phase]["shape"], [1, 3, 8])
            self.assertTrue(report["outputs"][phase]["all_finite"])
            self.assertEqual(report["outputs"][phase]["lm_head_usage"]["status"], "not_called_on_observed_sample_actions")
            self.assertEqual(report["outputs"][phase]["lm_head_usage"]["forward_calls"], 0)
        self.assertTrue(report["verification"]["output_shapes_match"])
        roundtrip = report["verification"]["restore_roundtrip"]
        self.assertEqual(roundtrip["status"], "passed")
        self.assertTrue(roundtrip["actions_close"])
        self.assertEqual(roundtrip["max_abs_diff"], 0)
        self.assertEqual(roundtrip["rtol"], 1e-5)
        self.assertEqual(roundtrip["atol"], 1e-6)
        self.restore_loader.assert_called_once()
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(self.restored_models), 1)
        restored_model = self.restored_models[0]
        self.assertFalse(restored_model.training)
        self.assertEqual(restored_model.calls, 1)
        baseline_input, compact_input = model.inference_inputs
        restored_input = restored_model.inference_inputs[0]
        self.assertIs(baseline_input[0], compact_input[0])
        self.assertIs(compact_input[0], restored_input[0])
        self.assertTrue(torch.equal(baseline_input[1], compact_input[1]))
        self.assertTrue(torch.equal(compact_input[1], restored_input[1]))
        self.assertEqual(baseline_input[2], self.args.num_steps)
        self.assertEqual(compact_input[2], restored_input[2])
        for phase in ("before", "after"):
            head = report[phase]["action_expert_lm_head"]
            self.assertEqual(head["parameters"], 96)
            self.assertEqual(head["weight_bytes"], 192)
            self.assertEqual(head["weight_mib"], 192 / 1024**2)
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
        for phase in ("before", "after", "restored"):
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
        self.restore_loader.assert_not_called()
        self.assertEqual(report["verification"]["restore_roundtrip"]["status"], "unverified")

    def test_failed_compact_forward_keeps_saved_artifacts_and_failure_report(self):
        with self.assertLogs(level="ERROR"):
            report, exit_code = self.run_mock_checkpoint(FailingPrunedForwardModel())
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["stage"], "pruned_forward")
        self.assertEqual(report["outputs"]["before"]["status"], "passed")
        self.assertEqual(report["outputs"]["after"]["status"], "failed")
        self.assertIsNone(report["outputs"]["after"]["shape"])
        self.assertTrue((self.output_dir / runner.STATE_DICT_NAME).is_file())
        self.restore_loader.assert_not_called()

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
        self.restore_loader.assert_not_called()
        self.assertEqual(report["outputs"]["restored"]["status"], "unverified")
        self.assertEqual(report["verification"]["restore_roundtrip"]["status"], "unverified")

    def test_saved_weights_without_forward_are_partial_and_do_not_restore(self):
        self.args.skip_forward = True
        report, exit_code = self.run_mock_checkpoint(MockCheckpointModel())
        self.assertEqual(exit_code, 2)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["artifacts"]["state_dict"]["status"], "saved")
        self.assertEqual(report["verification"]["restore_roundtrip"]["status"], "unverified")
        self.restore_loader.assert_not_called()

    def test_forward_without_saved_weights_cannot_claim_completed(self):
        self.args.no_save_state_dict = True
        report, exit_code = self.run_mock_checkpoint(MockCheckpointModel())
        self.assertEqual(exit_code, 2)
        self.assertEqual(report["status"], "partial")
        self.assertTrue(report["verification"]["output_shapes_match"])
        self.assertEqual(report["verification"]["restore_roundtrip"]["status"], "unverified")
        self.assertIn("reason", report["verification"]["restore_roundtrip"])
        self.restore_loader.assert_not_called()

    def test_restore_preserves_tied_head_parameters(self):
        report, exit_code = self.run_mock_checkpoint(TiedHeadModel(), restore_class=TiedHeadModel)
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["verification"]["restore_roundtrip"]["status"], "passed")
        expert = self.restored_models[0].paligemma_with_expert.gemma_expert
        self.assertIs(expert.lm_head.weight, expert.embed_tokens.weight)

    def test_restore_constructor_errors_are_failures_including_missing_dependencies_and_files(self):
        for index, error in enumerate((RuntimeError("restore construction failed"), ImportError("missing restore module"),
                                       FileNotFoundError("missing restore asset"))):
            with self.subTest(error=type(error).__name__), self.assertLogs(level="ERROR"):
                self.output_dir = Path(self.directory.name) / f"constructor-{index}"
                self.args.output_dir = str(self.output_dir)
                report, exit_code = self.run_mock_checkpoint(MockCheckpointModel(), restore_class=mock.Mock(side_effect=error))
                self.assertEqual(exit_code, 1)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["error"]["stage"], "restore_load")
                self.assertEqual(report["verification"]["restore_roundtrip"]["status"], "failed")
                self.assertEqual(json.loads((self.output_dir / runner.REPORT_NAME).read_text(encoding="utf-8")), report)

    def test_missing_corrupt_and_incompatible_saved_artifacts_fail_restore(self):
        def missing_state(directory):
            (directory / runner.STATE_DICT_NAME).unlink()

        def corrupt_state(directory):
            (directory / runner.STATE_DICT_NAME).write_bytes(b"not a PyTorch checkpoint")

        def missing_state_key(directory):
            path = directory / runner.STATE_DICT_NAME
            state = torch.load(path, map_location="cpu", weights_only=True)
            del state["action_out_proj.weight"]
            torch.save(state, path)

        def invalid_manifest(directory):
            path = directory / runner.MANIFEST_NAME
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["format_version"] = -1
            path.write_text(json.dumps(manifest), encoding="utf-8")

        for edit in (missing_state, corrupt_state, missing_state_key, invalid_manifest):
            with self.subTest(edit=edit.__name__), self.assertLogs(level="ERROR"):
                self.output_dir = Path(self.directory.name) / edit.__name__
                self.args.output_dir = str(self.output_dir)
                report, exit_code = self.run_mock_checkpoint(MockCheckpointModel(), artifact_edit=edit)
                self.assertEqual(exit_code, 1)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["error"]["stage"], "restore_load")
                self.assertEqual(report["verification"]["restore_roundtrip"]["status"], "failed")

    def test_invalid_restored_actions_cannot_claim_roundtrip_success(self):
        cases = ((MismatchedRestoredModel, "restore_compare"), (WrongDtypeRestoredModel, "restore_compare"),
                 (NonfiniteRestoredModel, "restored_forward"), (WrongShapeRestoredModel, "restored_forward"),
                 (FailingRestoredModel, "restored_forward"))
        for restore_class, stage in cases:
            with self.subTest(model=restore_class.__name__), self.assertLogs(level="ERROR"):
                self.output_dir = Path(self.directory.name) / restore_class.__name__
                self.args.output_dir = str(self.output_dir)
                report, exit_code = self.run_mock_checkpoint(MockCheckpointModel(), restore_class=restore_class)
                self.assertEqual(exit_code, 1)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["error"]["stage"], stage)
                self.assertEqual(report["verification"]["restore_roundtrip"]["status"], "failed")
                self.assertTrue((self.output_dir / runner.STATE_DICT_NAME).is_file())

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
