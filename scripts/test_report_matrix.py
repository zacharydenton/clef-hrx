import json
from pathlib import Path
import tempfile
import unittest

from report_matrix import records, report, workflow_progress


class MatrixReporting(unittest.TestCase):
    def test_live_log_tail_and_committed_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calls.jsonl"
            path.write_text('{"ok": true}\n{"inflight":')
            self.assertEqual(list(records(path)), [{"ok": True}])
            path.write_text('{"ok": true}\nbroken\n')
            with self.assertRaises(json.JSONDecodeError):
                list(records(path))

    def test_case_failures_and_resumed_loads_are_accounted_for(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "native-manifest.json").write_text(json.dumps({
                "datasets": {"invoice": {"case_ids": ["a", "b", "c"]}}}))
            cases = root / "invoice/cases"
            cases.mkdir(parents=True)
            for case_id, status in [("a", "ok"), ("b", "error")]:
                (cases / f"{case_id}.json").write_text(json.dumps({
                    "case_id": case_id, "execution": {"status": status}}))
            calls = []
            for first, tokens, seconds in [(True, 999, 100), (False, 100, 1),
                                           (True, 999, 100), (False, 900, 3)]:
                calls.append({"first_request_including_load": first,
                              "input_tokens": tokens, "wall_seconds": seconds,
                              "prediction": {"timings": {"allocated_bytes": 2**30}}})
            (root / "native-calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
            result = workflow_progress(root)
            self.assertEqual(result["checkpointed_cases"], 2)
            self.assertEqual(result["planned_cases"], 3)
            self.assertEqual(result["failed_cases"], 1)
            self.assertEqual(result["resident_calls"], 2)
            self.assertEqual(result["resident_input_tok_s"], 250)
            self.assertEqual(result["peak_engine_gib"], 1)
            (cases / "duplicate.json").write_text((cases / "a.json").read_text())
            with self.assertRaises(ValueError):
                workflow_progress(root)

    def test_not_started_is_pending_not_passed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(json.dumps({
                "suites": ["context", "workflowevals"], "models": {"clef-flash": "model"},
                "precisions": ["bf16"], "binary_sha256": "fixture"}))
            (root / "status.json").write_text(json.dumps({"complete": False, "jobs": {}}))
            result = report(root)
            self.assertFalse(result["complete"])
            self.assertTrue(all(j["state"] == "pending" for j in result["jobs"].values()))
            self.assertIsNone(result["jobs"]["context-clef-flash-bf16"]["replay_checks_pass"])


if __name__ == "__main__":
    unittest.main()
