#!/usr/bin/env python3
"""Alternate two qualified INT4 binaries over the complete resident JevBench suite.

Both models stay loaded, but only one request executes at a time. This is a
comparison under shared memory pressure, not the single-model throughput gate.
Inputs and exact expected outputs come from complete eval_jevbench.py runs.
"""
import argparse
import hashlib
import json
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

from bench_workload import hardware, paired_summary
from compare_jevbench import load
from eval_jevbench import kernel_cache_snapshot


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.baseline, args.candidate]
    qualified = [load(path) for path in paths]
    assert all(r["ok"] and r["strict_valid"] and not r["renormalized"]
               for _, _, records in qualified for r in records.values()), "qualification is not strict-valid"
    metadata = [q[0] for q in qualified]
    requests = [(p / "requests.jsonl").read_text().splitlines() for p in paths]
    assert requests[0] == requests[1] and len(requests[0]) == 231
    lengths = [json.loads((p / "input_lengths.json").read_text()) for p in paths]
    assert lengths[0] == lengths[1]
    for key in ("dataset_sha256", "benchmark_commit", "checkpoint_revision"):
        assert metadata[0][key] == metadata[1][key], f"{key} differs"
    expected = [list(map(json.loads, (p / "predictions.jsonl").read_text().splitlines()))
                for p in paths]
    ids = [r["task_id"] for r in expected[0]]
    assert ids == [r["task_id"] for r in expected[1]]
    assert len(ids) == len(set(ids)) == 231
    assert set(ids) == set(qualified[0][2]) == set(qualified[1][2])
    commands = [m["command"] for m in metadata]
    for command, meta in zip(commands, metadata):
        assert digest(command[0]) == meta["binary_sha256"], "binary changed"
        assert "--precision" in command
        assert command[command.index("--precision") + 1] == "int4", \
            "This harness budgets for two full INT4 models, not two BF16 models"
    assert commands[0][1:] == commands[1][1:], "CLI configurations differ"
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"complete": False, "qualified_runs": [str(p.resolve()) for p in paths],
              "commands": commands, "binary_sha256": [m["binary_sha256"] for m in metadata],
              "script_sha256": digest(__file__), "requests_sha256": digest(paths[0] / "requests.jsonl"),
              "dataset_sha256": metadata[0]["dataset_sha256"],
              "checkpoint_revision": metadata[0]["checkpoint_revision"],
              "hardware_before": hardware(), "cache_before": kernel_cache_snapshot(),
              "timing": "Alternating per-case order; two resident INT4 models; serialized requests. "
                        "Wall includes preparation and compilation. Excludes each model's first request. "
                        "Paired-case bootstrap describes this case mix, not repeated full-suite trials."}
    meta_path = args.output / "metadata.json"
    meta_path.write_text(json.dumps(record, indent=2) + "\n")
    processes = [None, None]
    samples, peak = [], [0, 0]
    with ExitStack() as stack:
        logs = [stack.enter_context((args.output / f"{name}.stderr.log").open("w"))
                for name in ("baseline", "candidate")]
        outputs = [stack.enter_context((args.output / f"{name}.jsonl").open("w"))
                   for name in ("baseline", "candidate")]
        telemetry = stack.enter_context((args.output / "telemetry.jsonl").open("w"))
        try:
            for index, (task_id, request) in enumerate(zip(ids, requests[0])):
                elapsed = [0, 0]
                for variant in (index % 2, 1 - index % 2):
                    started = time.perf_counter_ns()
                    if processes[variant] is None:
                        processes[variant] = subprocess.Popen(commands[variant], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=logs[variant], text=True, bufsize=1)
                    process = processes[variant]
                    process.stdin.write(request + "\n")
                    process.stdin.flush()
                    line = process.stdout.readline()
                    elapsed[variant] = time.perf_counter_ns() - started
                    if not line:
                        raise RuntimeError(f"variant {variant} stopped at {task_id}; inspect stderr")
                    prediction = json.loads(line)
                    assert prediction["input_tokens"] == lengths[variant][task_id]
                    assert prediction["questions"] == expected[variant][index]["prediction"]["questions"], \
                        f"variant {variant} output changed at {task_id}"
                    peak[variant] = max(peak[variant], prediction["timings"]["allocated_bytes"])
                    outputs[variant].write(json.dumps({"task_id": task_id, "prediction": prediction,
                        "latency_ns": elapsed[variant], "position_in_pair": int(variant != index % 2)}) + "\n")
                    outputs[variant].flush()
                if index:
                    samples.append({"iterations": 1, "reference_ns": elapsed[0], "optimized_ns": elapsed[1]})
                telemetry.write(json.dumps(hardware()) + "\n")
                telemetry.flush()
                print(f"{index + 1}/231 {task_id}: baseline={elapsed[0]/1e9:.3f}s "
                      f"candidate={elapsed[1]/1e9:.3f}s", flush=True)
            for process in processes:
                process.stdin.close()
                assert process.wait() == 0
        finally:
            for process in processes:
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait()
                if process is not None:
                    process.stdin.close()
                    process.stdout.close()
    tokens = sum(lengths[0][task_id] for task_id in ids[1:])
    seconds = [sum(s[key] for s in samples) / 1e9 for key in ("reference_ns", "optimized_ns")]
    summary = {"complete": True, "cases_per_variant": 231, "resident_input_tokens_per_variant": tokens,
               "unchanged_question_arrays_per_variant": [231, 231],
               "qualified_correct_per_variant": [q[1]["n_correct"] for q in qualified],
               "resident_seconds": seconds, "resident_tok_s": [tokens / s for s in seconds],
               "peak_engine_bytes_per_variant": peak, "paired": paired_summary(samples),
               "measurement_samples": samples, "timing_note": record["timing"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    record.update(complete=True, hardware_after=hardware(), cache_after=kernel_cache_snapshot())
    meta_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "measurement_samples"}, indent=2))


if __name__ == "__main__":
    main()
