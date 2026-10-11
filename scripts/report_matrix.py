#!/usr/bin/env python3
"""Read qualification checkpoints and throughput without touching the GPU.

The scheduler state is recorded state, not a process-liveness check. Partial
throughput describes only completed requests, not an estimate of the full suite.
"""

import argparse
import json
from pathlib import Path


def records(path):
    """Ignore an unfinished final write; reject malformed committed records."""
    if not path.exists():
        return
    with path.open() as source:
        for line in source:
            if not line.endswith("\n"):
                return
            yield json.loads(line)


def workflow_progress(root):
    manifest = json.loads((root / "native-manifest.json").read_text())
    workflows = {}
    for name, dataset in manifest["datasets"].items():
        planned = set(dataset["case_ids"])
        seen, failed = set(), 0
        for path in (root / name / "cases").glob("*.json"):
            case = json.loads(path.read_text())
            case_id = case["case_id"]
            if case_id not in planned or case_id in seen:
                raise ValueError(f"unexpected or duplicate checkpoint: {path}")
            seen.add(case_id)
            failed += case["execution"]["status"] != "ok"
        workflows[name] = {
            "checkpointed_cases": len(seen),
            "planned_cases": len(planned),
            "failed_cases": failed,
        }
    calls, resident_calls, tokens, seconds, peak = 0, 0, 0, 0.0, 0
    for call in records(root / "native-calls.jsonl"):
        calls += 1
        peak = max(peak, call["prediction"]["timings"]["allocated_bytes"])
        if not call["first_request_including_load"]:
            resident_calls += 1
            tokens += call["input_tokens"]
            seconds += call["wall_seconds"]
    return {
        "workflows": workflows,
        "checkpointed_cases": sum(w["checkpointed_cases"] for w in workflows.values()),
        "planned_cases": sum(w["planned_cases"] for w in workflows.values()),
        "failed_cases": sum(w["failed_cases"] for w in workflows.values()),
        "native_calls": calls,
        "resident_calls": resident_calls,
        "resident_input_tokens": tokens,
        "resident_wall_seconds": seconds,
        "resident_input_tok_s": tokens / seconds if seconds else None,
        "peak_engine_gib": peak / 2**30 if calls else None,
    }


def report(root):
    manifest = json.loads((root / "manifest.json").read_text())
    status = json.loads((root / "status.json").read_text())
    jobs = {}
    for suite in manifest["suites"]:
        for model in manifest["models"]:
            for precision in manifest["precisions"]:
                name = f"{suite}-{model}-{precision}"
                job = status["jobs"].get(name, {})
                # Failed context attempts may use a suffixed directory. Resolve
                # relative to the matrix so reports also work outside repo cwd.
                output = root / Path(job.get("output", name)).name
                row = {"state": job.get("state", "pending"), "attempt": job.get("attempt", 0)}
                if suite == "workflowevals" and (output / "native-manifest.json").exists():
                    row.update(workflow_progress(output))
                elif suite == "context":
                    cases = list(records(output / "results.jsonl"))
                    row.update(
                        completed_requests=len(cases),
                        correct_retrieval_answers=sum(c["correct"] for c in cases),
                        replay_checks_pass=all(c["replay_exact"] for c in cases) if cases else None,
                        input_64k_tok_s=[c["input_tokens_per_second"] for c in cases if c["tokens"] == 65536],
                        peak_engine_gib=max((c["prediction"]["timings"]["allocated_bytes"] / 2**30
                                             for c in cases), default=None),
                    )
                jobs[name] = row
    return {
        "complete": status["complete"],
        "binary_sha256": manifest["binary_sha256"],
        "jobs": jobs,
        "timing_note": "Completed native requests only; exclude the first request of every process. Includes encoding, preparation, compilation, GPU execution and readback; separate CPU preflight is excluded. Partial rates are not full-suite results or directly comparable across workflow branches.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.matrix), indent=2))


if __name__ == "__main__":
    main()
