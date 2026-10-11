#!/usr/bin/env python3
"""Run pinned public JevBench through llama.cpp's native CLEF decision API.

Starts a private loopback server, submits the same requests as eval_jevbench.py,
and saves responses, upstream scores, request latency, and hardware telemetry.
Requires a built llama-server, the pinned JevBench checkout, and a local GGUF.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from bench_workload import hardware

BENCHMARK_COMMIT = "7625ecb1b075cfe4d09d34141d353325825eea1d"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def speed_summary(records):
    from jevbench.metrics import latency_summary
    elapsed = sum(r["latency_s"] for r in records)
    tokens = [r["input_tokens"] for r in records]
    # Failed requests remain in accuracy and wall time. Missing usage prevents
    # a throughput claim; a failed call must never inflate the reported rate.
    complete = bool(records) and all(r["ok"] and n is not None for r, n in zip(records, tokens))
    return {"n": len(records), "wall_seconds": elapsed,
            "input_tokens": sum(tokens) if complete else None,
            "input_tokens_per_second": sum(tokens) / elapsed if complete and elapsed else None,
            "decisions_per_second": len(records) / elapsed if complete and elapsed else None,
            "latency": latency_summary([r["latency_s"] for r in records])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("artifacts/jevbench"))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Hugging Face model API response with blobs=true")
    parser.add_argument("--reference", type=Path, help="Optional HRX run for exact request and input-token comparisons")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Only first and longest cases; no full-suite claim")
    args = parser.parse_args()
    sys.path.insert(0, str(args.benchmark.resolve()))
    from jevbench.adapters.typesafe import TypeSafeAdapter
    from jevbench.tasks import load_jsonl, dataset_hash
    from jevbench.scoring import score_task
    from jevbench.summarize import summarize

    revision = subprocess.check_output(["git", "-C", str(args.benchmark), "rev-parse", "HEAD"], text=True).strip()
    if revision != BENCHMARK_COMMIT:
        raise ValueError(f"JevBench must be checked out at {BENCHMARK_COMMIT}")
    splits = {name: load_jsonl(str(args.benchmark / "datasets/public" / f"{name}.jsonl"))
              for name in ("easy", "original", "hard")}
    tasks = [task for group in splits.values() for task in group]
    reference_lengths = {}
    reference_metadata = {}
    if args.reference:
        reference_lengths = json.loads((args.reference / "input_lengths.json").read_text())
        reference_metadata = json.loads((args.reference / "metadata.json").read_text())
        if dataset_hash(tasks) != reference_metadata["dataset_sha256"]:
            raise ValueError("Reference dataset differs")
    if args.smoke:
        if not args.reference:
            parser.error("--smoke requires --reference to identify the longest case")
        tasks = [tasks[0], max(tasks, key=lambda t: reference_lengths[t.id])]
    if reference_lengths and max(reference_lengths[t.id] for t in tasks) > 4096:
        raise ValueError("Increase context and ubatch size for this dataset")
    manifest = json.loads(args.manifest.read_text())
    entry, = [f for f in manifest["siblings"] if f["rfilename"] == args.model.name]
    digest = sha256(args.model)
    if digest != entry["lfs"]["sha256"] or args.model.stat().st_size != entry["size"]:
        raise ValueError("GGUF does not match the pinned Hub manifest")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(args.binary.resolve()), "--model", str(args.model.resolve()),
               "--host", "127.0.0.1", "--port", str(port), "-ngl", "999", "-np", "1",
               "-c", "4096", "-b", "4096", "-ub", "4096", "-t", "16", "-tb", "16",
               "-fa", "on", "--no-warmup", "--cache-ram", "0", "--no-context-shift", "--fit", "off", "--log-verbosity", "4"]
    adapter = TypeSafeAdapter(endpoint=f"http://127.0.0.1:{port}", model="clef", key_env=None, timeout_s=600)
    requests = [adapter.build_request(t) for t in tasks]
    if args.reference and not args.smoke:
        reference_requests = [json.loads(line) for line in (args.reference / "requests.jsonl").read_text().splitlines()]
        if requests != reference_requests:
            raise ValueError("Requests differ from the HRX reference")
    if hardware()["kfd_pids"]:
        raise RuntimeError("GPU compute clients are already running; use an idle GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"benchmark_commit": revision, "dataset_sha256": dataset_hash(tasks),
                "smoke": args.smoke, "count": len(tasks), "command": command,
                "model_repository": manifest["id"], "model_revision": manifest["sha"],
                "model_file": args.model.name, "model_sha256": digest,
                "source_model_revision": (args.manifest.parent / ".src_sha").read_text().strip(),
                "llama_commit": subprocess.check_output(["git", "-C", str(args.llama_source), "rev-parse", "HEAD"], text=True).strip(),
                "binary_sha256": sha256(args.binary), "harness_sha256": sha256(__file__),
                "server_version": subprocess.check_output([str(args.binary.resolve()), "--version"], stderr=subprocess.STDOUT, text=True),
                "reference": str(args.reference) if args.reference else None, "reference_checkpoint": reference_metadata.get("checkpoint_revision"),
                "latency_note": "Sequential loopback HTTP; adapter.run wall time includes request construction, serialization, HTTP, native decision execution and response parsing. Server startup is separate. Headline excludes the first request, matching HRX's task subset. No warmup; no text generation.",
                "environment": {k: v for k, v in os.environ.items() if k.startswith(("GGML_", "HIP_", "ROCR_", "HSA_"))}}
    write_json(args.output / "metadata.json", metadata)
    write_json(args.output / "model-manifest.json", manifest)
    (args.output / "eval_jevbench_llama.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "CMakeCache.txt").write_bytes((args.binary.resolve().parent.parent / "CMakeCache.txt").read_bytes())
    (args.output / "llama-working-tree.diff").write_bytes(subprocess.check_output(["git", "-C", str(args.llama_source), "diff", "HEAD"]))
    (args.output / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    ldd = subprocess.check_output(["ldd", str(args.binary.resolve())], text=True)
    (args.output / "ldd.txt").write_text(ldd)
    libraries = [Path(line.split("=>", 1)[1].split()[0]) for line in ldd.splitlines() if "=> /" in line]
    write_json(args.output / "library-sha256.json", {str(p): sha256(p) for p in libraries})
    stop = threading.Event()

    def monitor():
        with (args.output / "hardware.jsonl").open("w") as out:
            while not stop.is_set():
                out.write(json.dumps(hardware()) + "\n")
                out.flush()
                stop.wait(1)

    records = []
    started = time.perf_counter()
    with (args.output / "server.log").open("w") as log, \
         (args.output / "predictions.jsonl").open("w") as raw, \
         (args.output / "scored.jsonl").open("w") as scored:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        metadata["server_pid"] = process.pid
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}; see {args.output}/server.log")
                try:
                    with urllib.request.urlopen(adapter.endpoint + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server readiness timeout")
                time.sleep(0.25)
            metadata["startup_seconds"] = time.perf_counter() - started
            write_json(args.output / "metadata.json", metadata)
            for index, task in enumerate(tasks):
                request_time = time.time()
                before = time.perf_counter()
                result = adapter.run(task)
                elapsed = time.perf_counter() - before
                length = (result.usage or {}).get("input_tokens")
                record = {"task_id": task.id, "ok": result.ok, "model": "llama.cpp/" + args.model.stem,
                          "probs_source": "native", "latency_s": elapsed,
                          "http_latency_s": result.latency_s, "input_tokens": length, "started_at": request_time,
                          "reference_input_tokens": reference_lengths.get(task.id),
                          **score_task(result.probs if result.ok else None, task)}
                if not result.ok:
                    record["error"] = result.error
                records.append(record)
                raw.write(json.dumps({"task_id": task.id, "response": result.raw, "error": result.error}) + "\n")
                raw.flush()
                scored.write(json.dumps(record) + "\n")
                scored.flush()
                print(f"{index + 1}/{len(tasks)} {task.id}: correct={record['correct']} tokens={length} ({elapsed:.3f}s)", flush=True)
        finally:
            stop.set()
            thread.join()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    summary = summarize(tasks, records)
    summary["tiers"] = {name: summarize(group, [r for r in records if r["task_id"] in {t.id for t in group}])
                        for name, group in splits.items()}
    samples = [json.loads(line) for line in (args.output / "hardware.jsonl").read_text().splitlines()]
    summary["other_kfd_pids"] = sorted({pid for sample in samples for pid in sample["kfd_pids"]
                                         if pid != str(process.pid)})
    summary["startup_seconds"] = metadata["startup_seconds"]
    summary["first_request_seconds"] = records[0]["latency_s"]
    summary["speed"] = {"all_requests": speed_summary(records), "resident": speed_summary(records[1:])}
    summary["token_count_mismatches"] = [{"task_id": r["task_id"], "llama": r["input_tokens"], "hrx": r["reference_input_tokens"]}
                                         for r in records if r["input_tokens"] != r["reference_input_tokens"]] if args.reference else None
    write_json(args.output / "input_lengths.json", {r["task_id"]: r["input_tokens"] for r in records})
    write_json(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("n_correct", "n_scorable", "speed", "token_count_mismatches")}))


if __name__ == "__main__":
    main()
