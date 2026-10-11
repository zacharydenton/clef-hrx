#!/usr/bin/env python3
"""Check throughput accounting without requiring a model or a GPU."""
import sys
import types
import unittest
from unittest.mock import patch

from eval_jevbench_llama import speed_summary


class SpeedAccountingTests(unittest.TestCase):
    def setUp(self):
        metrics = types.ModuleType("jevbench.metrics")
        metrics.latency_summary = lambda values: {"n": len(values)}
        self.modules = patch.dict(sys.modules, {"jevbench": types.ModuleType("jevbench"),
                                              "jevbench.metrics": metrics})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_aggregate_is_token_sum_over_time_sum(self):
        result = speed_summary([{"ok": True, "input_tokens": 100, "latency_s": 1},
                                {"ok": True, "input_tokens": 900, "latency_s": 3}])
        self.assertEqual(result["input_tokens_per_second"], 250)
        self.assertEqual(result["decisions_per_second"], 0.5)

    def test_failure_keeps_time_but_suppresses_throughput_claim(self):
        result = speed_summary([{"ok": True, "input_tokens": 100, "latency_s": 1},
                                {"ok": False, "input_tokens": None, "latency_s": 3}])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["wall_seconds"], 4)
        self.assertIsNone(result["input_tokens_per_second"])

    def test_missing_usage_suppresses_throughput_claim(self):
        result = speed_summary([{"ok": True, "input_tokens": None, "latency_s": 1}])
        self.assertIsNone(result["input_tokens_per_second"])


if __name__ == "__main__":
    unittest.main()
