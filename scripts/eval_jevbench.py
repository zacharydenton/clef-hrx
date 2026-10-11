#!/usr/bin/env python3
"""Run the pinned JevBench public suite through the resident CLEF CLI.

Clone https://github.com/fstandhartinger/jevbench into artifacts/jevbench,
checkout 7625ecb1b075cfe4d09d34141d353325825eea1d, and build CLEF first.
Uses only Python's standard library and the benchmark's own scorer.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import threading
from bench_workload import hardware
from pathlib import Path


def kernel_cache_snapshot():
    """Record HRX's content-addressed cache without changing shared artifacts."""
    xdg = Path(os.environ.get("XDG_CACHE_HOME", ""))
    root = xdg if xdg.is_absolute() else Path.home() / ".cache"
    path = root / "hrx" / "kernels"
    return {"path": str(path),
            "entries": sorted(p.name for p in path.iterdir() if p.is_dir()) if path.is_dir() else []}


def speed_summary(records):
    from jevbench.metrics import percentile
    elapsed = sum(r["latency_s"] for r in records)
    result = {
        "n": len(records),
        "input_tokens": sum(r["input_tokens"] for r in records),
        "wall_seconds": elapsed,
        "decisions_per_second": len(records) / elapsed,
        "input_tokens_per_second": sum(r["input_tokens"] for r in records) / elapsed,
    }
    values = {"wall": [r["latency_s"] for r in records]}
    for key in ("encode_ms", "prepare_ms", "inference_ms", "readback_ms"):
        values[key.removesuffix("_ms")] = [r["timings"][key] / 1000 for r in records]
    result["seconds"] = {
        key: {"mean": sum(v) / len(v), "p50": percentile(v, 0.5),
              "p95": percentile(v, 0.95), "min": min(v), "max": max(v)}
        for key, v in values.items()
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("artifacts/jevbench"))
    parser.add_argument("--binary", type=Path, default=Path("target/release/clef"))
    parser.add_argument("--output", type=Path, default=Path("tests/results/jevbench"))
    parser.add_argument("--precision", choices=("bf16", "int4-mlp", "int4"), default="bf16")
    parser.add_argument("--memory-gib", type=int, default=80)
    parser.add_argument("--model", choices=("clef", "clef-flash"), default="clef")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared", type=Path, help="Reuse a preflight from the same binary, data, and model source")
    args = parser.parse_args()
    sys.path.insert(0, str(args.benchmark.resolve()))
    from jevbench.adapters.typesafe import TypeSafeAdapter
    from jevbench.tasks import load_jsonl, dataset_hash
    from jevbench.scoring import score_task
    from jevbench.summarize import summarize

    args.output.mkdir(parents=True, exist_ok=False)
    splits = {name: load_jsonl(str(args.benchmark / "datasets/public" / f"{name}.jsonl"))
              for name in ("easy", "original", "hard")}
    tasks = [task for group in splits.values() for task in group]
    assert len({t.id for t in tasks}) == len(tasks)
    adapter = TypeSafeAdapter(model=args.model, key_env=None)
    requests = [adapter.build_request(task) for task in tasks]
    binary = str(args.binary.resolve())
    source = (["--model-dir", str(args.model_dir.resolve())] if args.model_dir else ["--offline"])
    source += ["--model", args.model, "--strict-context"]
    inspection = json.loads(subprocess.check_output([str(args.binary.resolve()), *source, "inspect"], text=True))
    lengths = []
    # Preflight every request at the largest supported context so no input is
    # silently truncated by a smaller runtime allocation.
    binary_hash = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    if args.prepared:
        previous = json.loads((args.prepared / "metadata.json").read_text())
        assert previous["binary_sha256"] == binary_hash, "Preflight binary differs"
        assert previous["dataset_sha256"] == dataset_hash(tasks), "Preflight dataset differs"
        assert previous["command"][1:1 + len(source)] == source, "Preflight model source differs"
        length_map = json.loads((args.prepared / "input_lengths.json").read_text())
        lengths = [length_map[t.id] for t in tasks]
    else:
        for request in requests:
            result = subprocess.run([binary, *source, "--max-length", "65536", "encode"],
                                    input=json.dumps(request), text=True, capture_output=True, check=True)
            encoded = json.loads(result.stdout)
            lengths.append(len(encoded["input_ids"]))
    assert max(lengths) <= 65536, "Input exceeds supported context"
    max_length = max(lengths)
    command = [binary, *source, "--memory-gib", str(args.memory_gib),
               "--max-length", str(max_length), "decide", "-", "--jsonl", "--raw"]
    if args.precision != "bf16":
        command[-4:-4] = ["--precision", args.precision]
    metadata = {
        "benchmark_url": "https://github.com/fstandhartinger/jevbench",
        "benchmark_commit": subprocess.check_output(
            ["git", "-C", str(args.benchmark), "rev-parse", "HEAD"], text=True).strip(),
        "clef_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "binary_sha256": binary_hash,
        "dataset_sha256": dataset_hash(tasks), "command": command,
        "count": len(tasks), "max_input_tokens": max_length,
        "checkpoint_revision": inspection["revision"],
        "model": args.model,
        "scoring": "Upstream score_task and summarize; raw native probabilities; score uses argmax",
        "latency_note": "Sequential local CLI; first call includes model loading. No warmup. Compilation included.",
        "cache_before": kernel_cache_snapshot(),
        "environment": {k: v for k, v in os.environ.items() if k.startswith("CLEF_EXPERIMENT_") or k in ("XDG_CACHE_HOME", "HRX_RUNTIME_DIR", "HRX_BUNDLE_MANIFEST", "HRX_OFFLINE")},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "working-tree.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    (args.output / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    (args.output / "input_lengths.json").write_text(json.dumps(dict(zip([t.id for t in tasks], lengths)), indent=2))
    if args.prepare_only:
        print(f"Prepared {len(tasks)} requests; {min(lengths)}–{max_length} tokens.")
        return
    print(f"Preflight: {len(tasks)} requests; {min(lengths)}–{max_length} tokens. Loading model.", flush=True)
    records = []
    telemetry_stop = threading.Event()
    def telemetry():
        with (args.output / "hardware.jsonl").open("w") as log:
            while not telemetry_stop.is_set():
                log.write(json.dumps(hardware()) + "\n")
                log.flush()
                telemetry_stop.wait(1)
    telemetry_thread = threading.Thread(target=telemetry, daemon=True)
    telemetry_thread.start()
    started = time.perf_counter()
    with (args.output / "stderr.log").open("w") as err, \
         (args.output / "predictions.jsonl").open("w") as raw_file, \
         (args.output / "scored.jsonl").open("w") as scored_file:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=err, text=True, bufsize=1)
        try:
            for index, (task, request, length) in enumerate(zip(tasks, requests, lengths)):
                before = time.perf_counter()
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError(f"CLEF stopped at {task.id}; see {args.output}/stderr.log")
                elapsed = time.perf_counter() - before
                prediction = json.loads(line)
                raw_file.write(json.dumps({"task_id": task.id, "prediction": prediction}) + "\n")
                raw_file.flush()
                assert prediction["input_tokens"] == length, f"Truncation at {task.id}"
                question, = prediction["questions"]
                assert question["question_id"] == "decision"
                probs = dict(zip(question["option_ids"], question["probabilities"], strict=True))
                if task.question["type"] == "noul":
                    assert set(probs) == {"true", "false"}
                    probs = {"yes": probs["true"], "no": probs["false"]}
                record = {"task_id": task.id, "ok": True, "model": args.model + "-hrx",
                          "probs_source": "native", "latency_s": elapsed,
                          "input_tokens": length, "timings": prediction["timings"],
                          **score_task(probs, task)}
                records.append(record)
                scored_file.write(json.dumps(record) + "\n")
                scored_file.flush()
                print(f"{index + 1}/{len(tasks)} {task.id}: {record['predicted']} "
                      f"expected={task.expected} correct={record['correct']} ({elapsed:.2f}s)", flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"CLEF exit status {process.returncode}")
        finally:
            telemetry_stop.set()
            telemetry_thread.join()
            if process.poll() is None:
                process.terminate()
                process.wait()
    summary = summarize(tasks, records)
    metadata["cache_after"] = kernel_cache_snapshot()
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary["tiers"] = {name: summarize(group, [r for r in records if r["task_id"] in {t.id for t in group}])
                        for name, group in splits.items()}
    summary["wall_seconds"] = time.perf_counter() - started
    summary["first_request_including_load_seconds"] = records[0]["latency_s"]
    from jevbench.metrics import latency_summary
    summary["resident_latency"] = latency_summary([r["latency_s"] for r in records[1:]])
    resident = records[1:]
    resident_seconds = sum(r["latency_s"] for r in resident)
    summary["resident_decisions_per_second"] = len(resident) / resident_seconds
    summary["resident_input_tokens_per_second"] = sum(r["input_tokens"] for r in resident) / resident_seconds
    summary["gpu_inference_latency"] = latency_summary([r["timings"]["inference_ms"] / 1000 for r in records])
    reference = json.loads((args.benchmark / "results/v1.2/jevbench-v1.2-per-task.json").read_text())
    reference_outcomes = reference["systems"]["jev-1.13.0"]["public_tasks"]
    summary["published_jev_1_13_0"] = {
        "note": "Published outcomes on the same public task IDs; no fresh Jev API calls.",
        "n_correct": sum(reference_outcomes[t.id][0] == "c" for t in tasks),
        "n_scorable": len(tasks),
        "tiers": {name: {"n_correct": sum(reference_outcomes[t.id][0] == "c" for t in group),
                         "n_scorable": len(group)} for name, group in splits.items()},
    }
    summary["speed"] = {
        "resident": speed_summary(resident),
        "tiers_excluding_first_request": {
            name: speed_summary([r for r in resident if r["task_id"] in {t.id for t in group}])
            for name, group in splits.items()
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_correct", "n_scorable", "accuracy", "brier_mean", "resident_latency")}))


if __name__ == "__main__":
    main()
