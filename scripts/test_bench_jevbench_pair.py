"""Exercise paired request ordering and child cleanup with tiny CPU-only CLIs."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import bench_jevbench_pair as bench


class PairedRunTests(unittest.TestCase):
    def fixture(self, root, name, fail_at=None):
        binary = root / f"{name}.py"
        binary.write_text(
            "#!/usr/bin/env python3\nimport json, sys\n"
            "for line in sys.stdin:\n"
            " request = json.loads(line)\n"
            f" if request['index'] == {fail_at!r}: sys.exit(3)\n"
            " print(json.dumps({'input_tokens': request['tokens'], 'questions': [], "
            "'timings': {'allocated_bytes': 1024}}), flush=True)\n"
        )
        binary.chmod(0o755)
        run = root / name
        run.mkdir()
        rows = [{"task_id": f"case-{i}", "correct": True, "ok": True,
                 "strict_valid": True, "renormalized": False} for i in range(231)]
        metadata = {"count": 231, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                    "command": [str(binary), "--precision", "int4"],
                    "dataset_sha256": "fixture", "benchmark_commit": "fixture",
                    "checkpoint_revision": "fixture"}
        summary = {"complete": True, "n_planned": 231, "n_attempted": 231, "n_correct": 231}
        for name, data in [("metadata", metadata), ("summary", summary),
                           ("input_lengths", {f"case-{i}": 128 + i for i in range(231)})]:
            (run / f"{name}.json").write_text(json.dumps(data))
        (run / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (run / "requests.jsonl").write_text("".join(
            json.dumps({"index": i, "tokens": 128 + i}) + "\n" for i in range(231)))
        (run / "predictions.jsonl").write_text("".join(json.dumps({
            "task_id": f"case-{i}", "prediction": {"questions": []}}) + "\n" for i in range(231)))
        return run

    def exercise(self, fail_at=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self.fixture(root, "baseline")
            candidate = self.fixture(root, "candidate", fail_at)
            output = root / "output"
            children = []
            popen = subprocess.Popen

            def spawn(*args, **kwargs):
                child = popen(*args, **kwargs)
                children.append(child)
                return child

            with patch("sys.argv", ["bench", str(baseline), str(candidate), "--output", str(output)]), \
                 patch.object(bench, "hardware", return_value={}), \
                 patch.object(bench, "kernel_cache_snapshot", return_value={}), \
                 patch.object(bench.subprocess, "Popen", side_effect=spawn), \
                 contextlib.redirect_stdout(io.StringIO()):
                if fail_at is None:
                    bench.main()
                else:
                    with self.assertRaisesRegex(RuntimeError, f"variant 1 stopped at case-{fail_at}"):
                        bench.main()
            self.assertEqual(len(children), 2)
            self.assertTrue(all(p.poll() is not None for p in children))
            self.assertTrue(all(p.stdin.closed and p.stdout.closed for p in children))
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["complete"], fail_at is None)
            if fail_at is not None:
                self.assertFalse((output / "summary.json").exists())
                return
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["resident_input_tokens_per_variant"], sum(range(129, 359)))
            self.assertEqual(len(summary["measurement_samples"]), 230)
            self.assertEqual(summary["unchanged_question_arrays_per_variant"], [231, 231])
            for variant, name in enumerate(("baseline", "candidate")):
                rows = list(map(json.loads, (output / f"{name}.jsonl").read_text().splitlines()))
                self.assertEqual(len(rows), 231)
                self.assertEqual([r["position_in_pair"] for r in rows],
                                 [int(variant != i % 2) for i in range(231)])

    def test_full_pair_excludes_first_request_and_alternates_order(self):
        self.exercise()

    def test_failed_child_reaps_both_processes_without_success_summary(self):
        self.exercise(fail_at=3)


if __name__ == "__main__":
    unittest.main()
