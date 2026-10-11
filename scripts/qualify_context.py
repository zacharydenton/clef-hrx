#!/usr/bin/env python3
"""Measure actual strict-context execution with facts at the beginning, middle and end."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from bench_workload import hardware


def make_request(filler):
    left = filler // 2
    return {
        "model": "clef",
        "state": "The first access code is AMBER.\n"
        + " x" * left
        + "\nThe middle flag is enabled.\n"
        + " x" * (filler - left)
        + "\nThe final severity is high.",
        "questions": {
            "first": {
                "type": "choice",
                "instructions": "What is the first access code?",
                "criteria": {
                    "amber": "AMBER",
                    "blue": "BLUE",
                    "red": "RED",
                    "green": "GREEN",
                },
            },
            "middle": {"type": "noul", "instructions": "Is the middle flag enabled?"},
            "last": {
                "type": "score",
                "instructions": "What is the final severity?",
                "criteria": ["low", "medium", "high"],
            },
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=Path("target/release/clef"))
    p.add_argument("--model", choices=["clef", "clef-flash"], default="clef")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--precision", choices=["bf16", "int4-mlp", "int4"], default="bf16")
    p.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32768, 49152, 65536],
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if any(n < 1024 or n > 65536 for n in args.lengths):
        p.error("lengths must be 1024..65536")
    args.output.mkdir(parents=True, exist_ok=False)
    common = [
        str(args.binary.resolve()),
        "--model",
        args.model,
        "--model-dir",
        str(args.model_dir.resolve()),
        "--precision",
        args.precision,
        "--strict-context",
        "--memory-gib",
        "80",
    ]

    def encode(request):
        r = subprocess.run(
            common + ["encode"],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=True,
        )
        return len(json.loads(r.stdout)["input_ids"])

    fixed = encode(make_request(0))
    requests = []
    for target in args.lengths:
        filler = target - fixed - 16
        # Token boundaries around the middle sentence can change by one token.
        for attempt in range(5):
            request = make_request(filler)
            request["model"] = args.model
            # Construct below the boundary first; the fixed/filler pattern is checked, never truncated.
            n = encode(request)
            if n == target:
                break
            filler += target - n
        else:
            raise RuntimeError(f"cannot construct exact {target}-token request")
        requests.append((target, request))
    # Exercise graph and recurrent-state reset with a long -> short -> long tail.
    requests += [requests[0], requests[-1]]
    (args.output / "requests.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for _, r in requests)
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "command": common,
                "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "inspection": json.loads(
                    subprocess.check_output(common + ["inspect"], text=True)
                ),
                "lengths": [n for n, _ in requests],
                "expected_options": {"first": "amber", "middle": "true", "last": "2"},
                "hardware_before": hardware(),
            },
            indent=2,
        )
        + "\n"
    )
    if args.prepare_only:
        return
    with (
        (args.output / "stderr.log").open("w") as err,
        (args.output / "results.jsonl").open("w") as out,
    ):
        process = subprocess.Popen(
            common + ["decide", "-", "--jsonl", "--raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
        )
        seen = {}
        try:
            for index, (tokens, request) in enumerate(requests):
                before = hardware()
                start = time.perf_counter()
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
                elapsed = time.perf_counter() - start
                if not line:
                    raise RuntimeError("native process ended; see stderr.log")
                raw = json.loads(line)
                assert raw["input_tokens"] == tokens
                winners = {
                    q["question_id"]: q["option_ids"][
                        max(
                            range(len(q["probabilities"])),
                            key=q["probabilities"].__getitem__,
                        )
                    ]
                    for q in raw["questions"]
                }
                exact = tokens not in seen or seen[tokens] == raw["questions"]
                seen[tokens] = raw["questions"]
                record = {
                    "tokens": tokens,
                    "wall_seconds": elapsed,
                    "input_tokens_per_second": tokens / elapsed,
                    "first_request_including_load": index == 0,
                    "winners": winners,
                    "correct": sum(
                        winners.get(k) == v
                        for k, v in {
                            "first": "amber",
                            "middle": "true",
                            "last": "2",
                        }.items()
                    ),
                    "replay_exact": exact,
                    "prediction": raw,
                    "hardware_before": before,
                    "hardware_after": hardware(),
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(tokens, record["correct"], elapsed, flush=True)
                assert exact, "long/short replay changed predictions"
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=15)


if __name__ == "__main__":
    main()
