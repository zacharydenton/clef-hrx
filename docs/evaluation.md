# JevBench public evaluation

CLEF-HRX answered **201 of 231 decisions correctly (87.0%)** on 10 October 2026.
Jev 1.13.0's published outcomes on the same public cases total **200/231 (86.6%)**.
This is a one-decision difference on the public subset, not a full JevBench
leaderboard result or a fresh Jev API run.

## Results

| Public tier | CLEF-HRX correct | Accuracy | Published Jev correct | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy | 48/48 | 100.0% | 48/48 | 1.58 s | 1.79 s |
| Standard (`original`) | 71/72 | 98.6% | 71/72 | 1.10 s | 3.17 s |
| Hard | 82/111 | 73.9% | 81/111 | 3.02 s | 12.09 s |
| All | 201/231 | 87.0% | 200/231 | 1.78 s | 9.90 s |

All 231 requests completed with strictly valid probability distributions and
no input truncation. Inputs ranged from 158 to 4,032 tokens, totaling 166,629.
The multiclass Brier score was 0.1667; top-label ECE was 0.0216 using 10 bins.
Both systems were correct on 190 cases; CLEF alone was correct on 11, Jev alone
on 10, and both were wrong on 20. CLEF's weakest family was temporal/numeric
reasoning, with 5/15 correct.

## Speed and measurement conditions

The run used AMD Strix Halo (`gfx1151`), the BF16 checkpoint, and a release build
of the native HRX implementation. One CLI process handled requests sequentially,
with the model resident between requests and a 60 GiB allocation budget. Another
application also used the GPU; these are shared-machine measurements from one
run. Use an idle GPU for isolated speed measurements.

The latency table excludes the first request from the easy and overall rows.
The remaining **230 requests took 713.1 s** in total measured request time:

| Resident measurement | Result |
| --- | ---: |
| Mean wall latency | 3.10 s |
| Throughput | 0.323 decisions/s (19.4/min) |
| Input-token throughput | 233.4 tokens/s |
| GPU execution median / p95 | 1.06 s / 7.80 s |
| Graph preparation median / p95 | 0.70 s / 2.09 s |

The first request, including model loading, took **97.2 s**. The complete
inference run took **810.8 s (13.5 minutes)**. These measurements exclude model
download, release compilation and token-length preflight. Request wall time
includes encoding, kernel compilation, graph preparation, GPU execution and
readback. There was no separate warmup. GPU execution timings include launch and
synchronization; they are not isolated per-kernel profiler measurements.

## Pinned inputs and scoring

- Inference source: `afa40e389de0336b0bfa61bd05a390f85009e437`.
- [JevBench source](https://github.com/fstandhartinger/jevbench/tree/7625ecb1b075cfe4d09d34141d353325825eea1d): `7625ecb1b075cfe4d09d34141d353325825eea1d`.
- [Cloudflare/CLEF checkpoint](https://huggingface.co/Cloudflare/clef/tree/2f3de3dd85f379784083b0814d997ab627200f0c): `2f3de3dd85f379784083b0814d997ab627200f0c`.
- Canonical dataset SHA-256: `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.
- Jev reference: the `jev-1.13.0.public_tasks` outcomes in the pinned
  [per-task results](https://github.com/fstandhartinger/jevbench/blob/7625ecb1b075cfe4d09d34141d353325825eea1d/results/v1.2/jevbench-v1.2-per-task.json).

The runner uses JevBench's request builder, `score_task` and `summarize` directly.
It evaluates every item in `easy.jsonl`, `original.jsonl` and `hard.jsonl`, in
that order. Requests contain the state and question, without answer keys.
Unrounded native probabilities are scored; Noul's `true`/`false` options map to
`yes`/`no`. Accuracy uses argmax, including for ordinal Score questions; ordinal
expected-value MAE is also reported. Binary Brier uses the two-class sum.

## Reproduce

Run from the repository root. In addition to the project's build and GPU
requirements, use Python 3.10+ and the Hugging Face `hf` CLI for downloading the
checkpoint. The evaluation runner and benchmark core need no Python packages.
Allow roughly 52 GiB of disk for weights and at least 51 GiB of available RAM
plus headroom for other processes.

```sh
git clone https://github.com/fstandhartinger/jevbench artifacts/jevbench
git -C artifacts/jevbench checkout 7625ecb1b075cfe4d09d34141d353325825eea1d

hf download Cloudflare/clef \
  --revision 2f3de3dd85f379784083b0814d997ab627200f0c \
  --local-dir artifacts/model \
  --include '*.json' '*.safetensors'

cargo build --release --locked
python scripts/eval_jevbench.py \
  --model-dir artifacts/model \
  --memory-gib 60 \
  --output tests/results/jevbench-run
```

These commands evaluate the currently checked-out inference source. The revision
above identifies the source used for the recorded result. If the benchmark is
already cloned, skip the clone command. Use a fresh output directory for every
run. If the pinned checkpoint is complete in the default Hugging Face cache,
omit `--model-dir`; the runner uses offline resolution in that case.

Preflight encodes all requests at the maximum supported context, then sets the
runtime limit to the longest input (4,032 tokens for this suite). Each inference
result must match its preflight token count. To prepare inputs separately, add
`--prepare-only`; reuse that directory with `--prepared PATH` on the actual run.
Reuse requires the same binary, dataset and model source.

The output directory contains:

- `summary.json`: accuracy, calibration, family/tier breakdowns, published Jev
  comparison and resident speed statistics.
- `predictions.jsonl` and `scored.jsonl`: raw logits/probabilities, per-case
  correctness, token counts, wall latency and internal phase timings.
- `requests.jsonl` and `input_lengths.json`: exact inputs and preflight lengths.
- `metadata.json` and `working-tree.diff`: revisions, binary/dataset hashes,
  inference command and tracked changes at run time.
- `stderr.log`: CLI diagnostics.

Generated outputs and downloaded artifacts are Git-ignored. The recorded run
is stored locally in `tests/results/jevbench-20261010-full/`; a fresh checkout
must run the commands above to produce its own artifacts.
