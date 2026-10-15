#!/usr/bin/env python3
"""Compare native --raw JSONL with the streaming Omni oracle's saved results.

Timing is native resident input-token throughput. Reference time includes weight
streaming and is deliberately not used for a speed comparison.
"""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--write-requests", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or not (args.native or args.write_requests):
        parser.error("provide --native or --write-requests, and a positive repeat count")
    references = [json.loads(path.read_text()) for path in args.reference]
    if args.write_requests:
        with args.write_requests.open("x") as output:
            for reference in references:
                for _ in range(args.repeats):
                    output.write(json.dumps(reference["request"]) + "\n")
    if not args.native:
        return
    # memory_guard combines diagnostic stderr and JSON stdout in run.log.
    native = [json.loads(line) for line in args.native.read_text().splitlines()
              if line.startswith('{')]
    if len(native) != len(references) * args.repeats:
        raise ValueError("native result count does not match reference/repeat count")
    report = []
    for i, (path, reference) in enumerate(zip(args.reference, references)):
        runs = native[i * args.repeats:(i + 1) * args.repeats]
        differences = []
        max_error = 0.0
        for run in runs:
            if run["input_tokens"] != len(reference["input_ids"]):
                raise ValueError(f"{path}: token counts differ")
            if len(run["questions"]) != len(reference["questions"]):
                raise ValueError(f"{path}: question counts differ")
            for actual, expected in zip(run["questions"], reference["questions"]):
                for key in ("question_id", "option_ids"):
                    if actual[key] != expected[key]:
                        raise ValueError(f"{path}: {key} differs")
                a, b = actual["probabilities"], expected["probabilities"]
                if (len(a) != len(b) or not a or
                        any(not math.isfinite(p) or not 0 <= p <= 1 for p in a + b) or
                        abs(sum(a) - 1) > 1e-5 or abs(sum(b) - 1) > 1e-5):
                    raise ValueError(f"{path}: invalid probabilities")
                max_error = max(max_error, max(abs(x - y) for x, y in zip(a, b)))
                if max(range(len(a)), key=a.__getitem__) != max(range(len(b)), key=b.__getitem__):
                    differences.append(actual["question_id"])
        timings = []
        for run in runs:
            t = run["timings"]
            resident_ms = sum(t[k] for k in ("encode_ms", "prepare_ms", "inference_ms", "readback_ms"))
            timings.append({"resident_tok_s": run["input_tokens"] * 1000 / resident_ms,
                            "gpu_tok_s": run["input_tokens"] * 1000 / t["inference_ms"],
                            **t})
        report.append({"reference": str(path), "input_tokens": runs[0]["input_tokens"],
                       "questions": len(reference["questions"]),
                       "argmax_differences": sorted(set(differences)),
                       "max_probability_error": max_error,
                       "repeat_exact": all(r["questions"] == runs[0]["questions"] for r in runs),
                       "runs": timings})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
