#!/usr/bin/env python3
"""Check the text/image/video corpus and changed-input replay in every precision.

BF16 must match the independent reference within 0.003. Initial INT4 differences
are reported without an accuracy cutoff; response validity and replay still gate.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=Path("target/release/clef"))
    p.add_argument("--model", choices=["clef", "clef-flash"], required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [0, 1, 2, 3, 4, 5, 6, 7, 0, 0]
    requests = [Path(f"tests/fixtures/corpus/{i}.json").read_text() for i in cases]
    references = [
        json.loads((args.reference / f"{i}.reference.json").read_text()) for i in cases
    ]
    command = [
        str(args.binary.resolve()),
        "--model",
        args.model,
        "--model-dir",
        str(args.model_dir.resolve()),
        "--max-length",
        "1536",
        "--memory-gib",
        "80",
    ]
    manifest = {
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inspection": json.loads(
            subprocess.check_output(command + ["inspect"], text=True)
        ),
        "commands": [],
        "cases": cases,
        "fixtures": {},
    }
    for i in set(cases):
        for path in [
            Path(f"tests/fixtures/corpus/{i}.json"),
            args.reference / f"{i}.reference.json",
        ]:
            manifest["fixtures"][str(path)] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    results = {}
    for precision in ["bf16", "int4-mlp", "int4"]:
        invocation = command + [
            "--precision",
            precision,
            "decide",
            "-",
            "--jsonl",
            "--raw",
        ]
        manifest["commands"].append(invocation)
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        with (
            (args.output / f"{precision}.jsonl").open("w") as out,
            (args.output / f"{precision}.stderr").open("w") as err,
        ):
            subprocess.run(
                invocation,
                input="".join(json.dumps(json.loads(r)) + "\n" for r in requests),
                text=True,
                stdout=out,
                stderr=err,
                check=True,
            )
        predictions = [
            json.loads(line)
            for line in (args.output / f"{precision}.jsonl").read_text().splitlines()
        ]
        assert len(predictions) == len(cases)
        maximum, changed = 0.0, []
        for case, actual, reference in zip(cases, predictions, references):
            assert actual["input_tokens"] == len(reference["input_ids"])
            assert len(actual["questions"]) == len(reference["questions"])
            for a, b in zip(actual["questions"], reference["questions"]):
                assert (
                    a["question_id"] == b["question_id"]
                    and a["option_ids"] == b["option_ids"]
                )
                probs = a["probabilities"]
                assert len(probs) == len(b["probabilities"])
                assert all(math.isfinite(x) and 0 <= x <= 1 for x in probs)
                assert abs(sum(probs) - 1) < 1e-5
                maximum = max(
                    maximum, max(abs(x - y) for x, y in zip(probs, b["probabilities"]))
                )
                if max(range(len(probs)), key=probs.__getitem__) != max(
                    range(len(probs)), key=b["probabilities"].__getitem__
                ):
                    changed.append({"case": case, "question_id": a["question_id"]})
        assert (
            predictions[0]["questions"]
            == predictions[-2]["questions"]
            == predictions[-1]["questions"]
        )
        results[precision] = {
            "maximum_probability_error": maximum,
            "changed_winners": changed,
            "valid": True,
            "replay_exact": True,
        }
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(precision, results[precision], flush=True)
        if precision == "bf16":
            assert maximum <= 0.003 and not changed, "BF16 reference regression"


if __name__ == "__main__":
    main()
