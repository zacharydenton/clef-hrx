# JevBench public evaluation

CLEF-HRX answered **202 of 231 decisions correctly (87.4%)** on 10 October 2026.
Jev 1.13.0's published outcomes on the same public cases total **200/231 (86.6%)**.
This is a two-decision difference on the public subset, not a full JevBench
leaderboard result or a fresh Jev API run.

## Results

| Public tier | CLEF-HRX correct | Accuracy | Published Jev correct | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy | 48/48 | 100.0% | 48/48 | 0.81 s | 0.88 s |
| Standard (`original`) | 71/72 | 98.6% | 71/72 | 0.83 s | 0.89 s |
| Hard | 83/111 | 74.8% | 81/111 | 1.55 s | 7.10 s |
| All | 202/231 | 87.4% | 200/231 | 0.90 s | 5.54 s |

All 231 requests completed with strictly valid probability distributions and
no input truncation. Inputs ranged from 158 to 4,032 tokens, totaling 166,629.
The multiclass Brier score was 0.1669; top-label ECE was 0.0282 using 10 bins.
CLEF's weakest family was temporal/numeric reasoning, with 5/15 correct.

The earlier chunked DeltaNet run scored 201/231 at 233.4 input tokens/s. Keeping
recurrent state in registers increased resident throughput by 1.48×, gained
`hard-sol-c-judge_hard-05`, and lost no correct decisions. Maximum probability
change was 0.04375. The numerical comparison and full-suite gate are described in
[Performance experiments](optimization.md).

The preceding run added tiled attention for inputs of at least 1,024 tokens and
uses runtime sequence lengths to share compiled attention code. It retained
202/231 with no changed decisions; maximum probability change was 0.01727 against
the register-resident baseline. Brier changed from 0.1668 to 0.1669 and ECE from
0.0228 to 0.0282. The observed throughput increased from 345.3 to 401.9 tokens/s,
but these runs did not control compiler-cache warmth. Preparation also improved
on unchanged short requests, so the full 16.4% difference cannot be attributed
to the attention kernel. A repeat of baseline source `0f79e8b` with the shared
cache populated scored the same 202/231 at **381.5 tokens/s**. Against that run,
the candidate's observed gain is **5.3%**: resident wall time fell from 436.1 to
414.1 s, GPU time from 420.4 to 397.2 s, while preparation rose from 15.1 to
16.3 s. This remains a sequential shared-machine comparison rather than a paired
statistical estimate. The repeat's probabilities exactly match the earlier
register-resident baseline.

The current run groups DeltaNet's eight leader-lane stores at the end of each
token iteration. Arithmetic and storage precision are unchanged. All 231
probability distributions exactly match the attention run, retaining 202/231.
Resident throughput increased from 401.9 to **407.8 tokens/s** (1.5%); GPU time
fell from 397.2 to 388.9 s while preparation rose from 16.3 to 18.5 s. This is
another sequential shared-machine comparison. Separate alternating-order
Criterion runs measured 1.15–1.32× kernel speedups, with bitwise-identical outputs.
See [the kernel investigation](llama-kernel-notes.md) for those results and repro.

## Speed and measurement conditions

The run used AMD Strix Halo (`gfx1151`), the BF16 checkpoint, and a release build
of the native HRX implementation. One CLI process handled requests sequentially,
with the model resident between requests and a 60 GiB allocation budget. Another
application also used the GPU; these are shared-machine measurements from one
run. The shared compiler cache already contained artifacts from earlier runs;
this is not a cold-cache measurement. Use an idle GPU and matching cache
conditions for isolated speed comparisons.

The latency table excludes the first request from the easy and overall rows.
The remaining **230 requests took 408.0 s** in total measured request time:

| Resident measurement | Result |
| --- | ---: |
| Mean wall latency | 1.77 s |
| Throughput | 0.564 decisions/s (33.8/min) |
| Input-token throughput | 407.8 tokens/s |
| GPU execution median / p95 | 0.84 s / 5.41 s |
| Graph preparation median / p95 | 0.067 s / 0.159 s |

The first request, including model loading, took **34.1 s**. The complete
inference run took **453.3 s (7.6 minutes)**. These measurements exclude model
download, release compilation and token-length preflight. Request wall time
includes encoding, kernel compilation, graph preparation, GPU execution and
readback. There was no separate warmup. GPU execution timings include launch and
synchronization; they are not isolated per-kernel profiler measurements.

## Pinned inputs and scoring

- Inference: register-resident DeltaNet with grouped stores and runtime-length tiled long-sequence attention shipped with this document.
- Evaluated CLI SHA-256: `c530e47d460081cc18128ecc51e92c1c2df044ce773b480d63eeaf582e785a15`.
- Earlier chunked baseline source: `afa40e389de0336b0bfa61bd05a390f85009e437`.
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

These commands evaluate the currently checked-out inference source. The binary
hash above identifies the recorded executable. If the benchmark is
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
  inference command and tracked changes at run time. New runs also record the
  HRX kernel-cache path and entry names before/after inference, plus cache/runtime
  environment overrides; older runs did not capture cache state.
- `stderr.log`: CLI diagnostics.

Generated outputs and downloaded artifacts are Git-ignored. The recorded run
is stored locally in `tests/results/jevbench-20261010-delta-stores/`; its
`comparison.json` records the complete accuracy and speed gate. The archived
executable is `artifacts/throughput-1000/clef-delta-stores`. Prior baselines are in
`tests/results/jevbench-20261010-attention-long/`,
`tests/results/jevbench-20261010-persistent/` and
`tests/results/jevbench-20261010-full/`. The warm baseline repeat is in
`tests/results/jevbench-20261010-persistent-warm/`; its binary SHA-256 is
`0b45482678f225acbfc11e14e7c4a5082aa36f1956171ada23b3f44382312405`.
A fresh checkout
must run the commands above to produce its own artifacts.

## 64k context regression

The `5aa94a6` context implementation was rechecked over all 231 cases in
`tests/results/jevbench-20261010-context-64k/`: **202/231**, all strictly valid,
with input lengths unchanged. Every logit and probability matches the grouped
DeltaNet-store baseline exactly. Resident throughput was **412.67 input tok/s**,
median **0.887 s**, p95 **5.496 s**. This single run is a regression check, not
paired evidence for a speed improvement. The requests themselves are 158–4,032
tokens; this does not measure 64k-token model accuracy.

Archived binary: `artifacts/throughput-1000/clef-context-64k`, SHA-256
`672781517ce7d08ab4ab0fcfa601d226a3be83149c0d168cc150e8049dfce8fd`.
