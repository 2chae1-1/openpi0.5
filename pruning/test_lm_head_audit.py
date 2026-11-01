"""CPU fixtures for head accounting and instrumentation, not OpenPI inference."""

import unittest

import torch
from torch import nn

from pruning.lm_head_audit import lm_head_metrics
from pruning.lm_head_audit import observe_lm_head_calls


def tiny_model(*, tied=False):
    model = nn.Module()
    model.paligemma_with_expert = nn.Module()
    expert = nn.Module()
    model.paligemma_with_expert.gemma_expert = expert
    expert.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
    if tied:
        expert.embedding = nn.Embedding(8, 4, dtype=torch.bfloat16)
        expert.embedding.weight = expert.lm_head.weight
    return model


class TestLmHeadAudit(unittest.TestCase):
    def test_actual_bf16_tensor_bytes_and_retention(self):
        model = tiny_model()
        head = model.paligemma_with_expert.gemma_expert.lm_head
        original = head.weight.detach().clone()
        metrics = lm_head_metrics(model)
        self.assertEqual(metrics["status"], "measured")
        self.assertTrue(metrics["retained"])
        self.assertEqual(metrics["parameters"], 32)
        self.assertEqual(metrics["weight_bytes"], 64)
        self.assertEqual(metrics["weight_mib"], 64 / 1024**2)
        self.assertEqual(metrics["by_dtype"], {"torch.bfloat16": {"parameters": 32, "weight_bytes": 64}})
        self.assertEqual(metrics["exclusive"]["weight_bytes"], 64)
        self.assertEqual(metrics["shared_with_rest"]["weight_bytes"], 0)
        self.assertIs(model.paligemma_with_expert.gemma_expert.lm_head, head)
        self.assertTrue(torch.equal(head.weight, original))

    def test_shared_weight_is_not_claimed_as_exclusive_memory(self):
        model = tiny_model(tied=True)
        metrics = lm_head_metrics(model)
        self.assertEqual(metrics["parameters"], 32)
        self.assertEqual(metrics["shared_with_rest"]["parameters"], 32)
        self.assertEqual(metrics["exclusive"]["parameters"], 0)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 32)

    def test_mixed_dtype_and_duplicate_head_parameters(self):
        model = tiny_model()
        head = model.paligemma_with_expert.gemma_expert.lm_head
        head.alias = head.weight
        head.extra = nn.Parameter(torch.ones(2, dtype=torch.float32))
        metrics = lm_head_metrics(model)
        self.assertEqual(metrics["parameters"], 34)
        self.assertEqual(metrics["weight_bytes"], 72)
        self.assertEqual(metrics["by_dtype"]["torch.float32"]["weight_bytes"], 8)

    def test_observes_calls_then_removes_hook(self):
        model = tiny_model()
        head = model.paligemma_with_expert.gemma_expert.lm_head
        with observe_lm_head_calls(model) as observed:
            self.assertEqual(observed["status"], "unverified")
            head(torch.zeros(1, 4, dtype=torch.bfloat16))
            head(torch.zeros(1, 4, dtype=torch.bfloat16))
        self.assertEqual(observed, {"status": "called_on_observed_sample_actions", "forward_calls": 2})
        head(torch.zeros(1, 4, dtype=torch.bfloat16))
        self.assertEqual(observed["forward_calls"], 2)
        self.assertEqual(len(head._forward_pre_hooks), 0)

    def test_successful_sampling_without_head_is_scoped_to_observed_path(self):
        model = tiny_model()
        with observe_lm_head_calls(model) as observed:
            torch.zeros(1)
        self.assertEqual(observed, {"status": "not_called_on_observed_sample_actions", "forward_calls": 0})

    def test_failure_keeps_unverified_and_removes_hook(self):
        model = tiny_model()
        head = model.paligemma_with_expert.gemma_expert.lm_head
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"), observe_lm_head_calls(model) as observed:
            head(torch.zeros(1, 4, dtype=torch.bfloat16))
            raise RuntimeError("synthetic failure")
        self.assertEqual(observed["status"], "unverified")
        self.assertEqual(observed["forward_calls"], 1)
        self.assertIn("synthetic failure", observed["error"])
        self.assertEqual(len(head._forward_pre_hooks), 0)

    def test_missing_head_does_not_invent_measurements(self):
        model = nn.Module()
        self.assertEqual(lm_head_metrics(model)["status"], "not_present")
        self.assertNotIn("parameters", lm_head_metrics(model))
        with observe_lm_head_calls(model) as observed:
            pass
        self.assertEqual(observed, {"status": "not_present", "forward_calls": None})


if __name__ == "__main__":
    unittest.main()
