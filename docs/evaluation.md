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

## INT4 scratch reuse

The ordinary release CLI with `--precision int4` was checked on 11 October 2026
in `tests/results/jevbench-20261011-int4-scratch-reuse/`. It scored **194/231**,
with all 231 outputs strictly valid. All logits, probabilities, task IDs and
input token counts match `jevbench-20261010-int4-backbone` exactly. BF16 remains
the default; this check preserves the existing optional INT4 accuracy tradeoff.

Resident throughput was **775.79 input tok/s**: 166,394 tokens in 214.485 seconds,
excluding only the first request/model load. Median request latency was 0.398 s,
p95 3.286 s. The previous INT4 run measured 747.71 tok/s; these separate runs
do not establish a paired speed improvement. Peak engine allocation decreased
from **38.430 to 14.725 GiB** by releasing packed activation scratch for reuse
after its last recorded read, while retaining immutable packed weights.

Reproduce with the pinned checkpoint and JevBench checkout described above:

```sh
cargo build --locked --release
python3 scripts/eval_jevbench.py \
  --binary target/release/clef --precision int4 \
  --model-dir artifacts/model \
  --output tests/results/jevbench-int4-scratch-repro
```

Archived binary: `artifacts/throughput-1000/clef-int4-scratch-reuse`, SHA-256
`ee36cd7d229aba89603f108f87665b83ca27684be92bb7f23895db1df2cbf85e`.
The run's `working-tree.diff` records the change over `fb1f8c8`; `comparison.json`
records output equality and peak allocations. GPU tests additionally cover a
32-projection graph with bounded scratch and exact identity outputs on two
replays, packed-weight lifetime across plan resets, and an independent CPU
packing oracle.

## Fused SwiGLU

The full runs in `jevbench-20261011-bf16-swiglu` and
`jevbench-20261011-int4-swiglu` preserve all 231 logit and probability arrays
against their respective 64k-context BF16 and shared-packing INT4 baselines.
Both runs are strictly valid, with identical task IDs, choices and token counts.

| Mode | Correct | Resident input tok/s | Resident seconds | Median / p95 request seconds | Peak engine GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 default | 202/231 | 411.94 | 403.930 | 0.904 / 5.470 | 47.773 |
| INT4 eligible backbone | 194/231 | 814.21 | 204.364 | 0.382 / 3.130 | 14.508 |

Both resident measurements cover 166,394 input tokens, excluding only the first
request/model load. The previous respective rates were 412.67 and 780.17 tok/s;
these full runs are not paired timing measurements. The operator's eleven-shape
paired Criterion sweep independently showed 1.31–1.84× speedups with bitwise
equality. BF16 remains the default, with no accuracy regression; INT4 retains
its explicit accuracy tradeoff.

Reproduce using the pinned inputs above and a fresh result directory:

```sh
cargo build --locked --release
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision int4 --output tests/results/jevbench-swiglu-int4-repro
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --output tests/results/jevbench-swiglu-bf16-repro
python3 scripts/bench_workload.py swiglu-paired-repro --filter paired_swiglu
```

Archived binary: `artifacts/throughput-1000/clef-swiglu`, SHA-256
`b95f8406b8334fa68b1b842ba0be370f21116211f63e5e6bbc0a4cb3fb2f3f66`.
`artifacts/throughput-1000/clef-swiglu-source.json` records the complete source
snapshot from `benches/results/swiglu-paired-full/source`, including new files
absent from `git diff`. Each eval's `comparison.json` and `output-equality.json`
record the complete gates and exact-output checks.

## Single-pass INT4 packing

`jevbench-20261011-int4-single-packing` scored **194/231**, with all 231 requests
strictly valid and all logits and probabilities identical to the fused-SwiGLU
INT4 baseline. The checkpoint, task IDs, choices and input lengths are unchanged.
BF16 remains the default; this change affects integer packing only.

The resident run processed **166,394 input tokens in 200.085 seconds**:
**831.62 input tok/s**, with **0.329 s median / 3.068 s p95** request latency and
**14.508 GiB** peak engine allocation. Timing includes encoding, preparation,
compilation, GPU execution and readback, excluding only the first request/model
load. The preceding run measured 814.21 tok/s; these are separate full-suite
measurements. The 1,000 tok/s target has not been reached.

Packing keeps transformed FP32 values in local memory until the row scale is
known, avoiding a second transform for widths up to 6,144. Wider operands retain
the original implementation. The 22-pair Criterion sweep over 5,120/6,144 columns
and all eleven workload lengths preserved exact codes/scales and measured
1.30–1.95× operator speedups, with all 99% paired intervals above 1.0. An 8,192
column candidate lacked a consistent win and was excluded from selection.

Reproduce with the pinned inputs above, running GPU jobs sequentially:

```sh
cargo build --locked --release
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision int4 --output tests/results/jevbench-single-packing-repro
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261011-int4-swiglu \
  tests/results/jevbench-single-packing-repro
python3 scripts/bench_workload.py packing-paired-repro \
  --filter 'paired_packing/pair/.*x(5120|6144)$'
```

Archived binary: `artifacts/throughput-1000/clef-int4-single-packing`, SHA-256
`b28bccfcb1be76a544eb641ad330e49fb579674d92cb3381bc5ec280d4aa0dad`.
`artifacts/throughput-1000/clef-int4-single-packing-source.json` records every
source hash in `benches/results/single-packing-paired-full/source`, including new
kernel files. The run's `comparison.json` and `output-equality.json` record the
full accuracy gate, exact-output check and peak allocation. GPU oracle tests
also cover INT4/INT8, zero/constant/random rows, packing fallback widths,
scratch reuse and packed-weight lifetime.

## Prepared DeltaNet experiment

The eight-value prepared recurrence computes normalized Q/K and gates once per
token in FP32, then reuses them across recurrent waves. Seven paired operator
shapes from 128 through 2,048 tokens matched output bits and measured
1.04–1.19× speedups. Full JevBench qualification selected it only from 128 to
2,048 tokens, retaining the existing implementation outside that range.

| Mode | Correct / strict valid | Prior resident tok/s | Candidate resident tok/s | Resident seconds | Peak engine GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 202 / 231 | 411.94 | 406.22 | 409.614 | 47.774 |
| INT4 eligible backbone | 194 / 231 | 831.62 | 832.16 | 199.954 | 14.509 |

Both runs processed 166,394 resident input tokens and preserved all 231 logit
and probability arrays against their respective baselines. These are separate
sequential full runs, not paired model timings. INT4 mean GPU time fell from
0.7968 to 0.7940 seconds per request, while mean preparation rose from 0.0701 to
0.0727 seconds. There was no useful end-to-end gain, so **this path is not selected
by default**. It remains a benchmark experiment for future overhead work.

Results are saved in `jevbench-20261011-bf16-delta-prepared-v8` and
`jevbench-20261011-int4-delta-prepared-v8`, including `comparison.json` and
`output-equality.json`. Archived normal-build candidate:
`artifacts/throughput-1000/clef-delta-prepared-v8`, SHA-256
`4c6d791c462e0a65f2fcdb178ba123f2c4a769e39efe0c4cee139ab917f6d95b`.
Its `-source.json` records all files in
`benches/results/delta-prepared-v8-boundary/source`; that snapshot includes the
temporary default selection used for these measurements.

To reproduce the algorithm using the current benchmark-only switch:

```sh
cargo build --locked --release --features bench-internals --bin clef
CLEF_EXPERIMENT_DELTA_PREPARED_V8_MAX=2048 \
  python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision int4 --output tests/results/jevbench-prepared-v8-int4-repro
CLEF_EXPERIMENT_DELTA_PREPARED_V8_MAX=2048 \
  python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --output tests/results/jevbench-prepared-v8-bf16-repro
```

Benchmark builds also include diagnostic labels; use the archived normal binary
or build its source snapshot to repeat the exact recorded configuration. The
older `CLEF_EXPERIMENT_DELTA_PREPARED_MAX` switch retains its four-value kernel.

## Vectorized DeltaNet qualification

The recurrence can express its eight independent value channels as vectors while
retaining each channel's FP32 operation order. This lets the compiler schedule
reductions together without adding a preparation dispatch. The selected range
is 128–4,032 tokens; shorter and longer requests retain the scalar recurrence.

Eight paired operator cases (128, 158, 235, 260, 620, 1,024, 2,048 and 4,032)
matched output bits and measured 1.16–1.43× speedups, with all 99% lower bounds
above 1.0. Artifacts are `delta-vector8-paired` and `delta-vector8-long-paired`
under `benches/results`. A separate GPU fixture checks input slices, distinct
and varying gates, repeated execution and both selection boundaries.

Both full JevBench runs preserved every logit and probability array:

| Mode | Correct / strict valid | Resident input tok/s | Resident seconds | Peak engine GiB |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 202 / 231 | 277.05 | 600.588 | 47.773 |
| INT4 eligible backbone | 194 / 231 | 612.33 | 271.741 | 14.508 |

Each run processed the same 166,394 resident tokens. These rates include
preparation and compilation. They were recorded while load, clocks and
temperature changed, and are slower than the historical 411.94/831.62 tok/s
runs. The first INT4 run also compiled previously unseen vector specializations.
These results qualify accuracy but do **not** establish an end-to-end speedup or
achieve the 1,000 tok/s target. Retain them when comparing later runs.

Full records are `tests/results/jevbench-20261011-{bf16,int4}-delta-vector8`, with
`comparison.json` and `output-equality.json`. Telemetry from part of the INT4 run
through the end of BF16 is saved in
`artifacts/throughput-1000/delta-vector8-eval-telemetry.jsonl`; it is not a complete
first-run trace. The normal-build binary is
`artifacts/throughput-1000/clef-delta-vector8`, SHA-256
`e827fdf92a4e23223f286856180a4de7209936025ada6fded49b11a0408b8d8d`.
Its `-source.json` lists all 99 files in
`artifacts/throughput-1000/delta-vector8-source`.

Reproduce operator and full-model checks sequentially:

```sh
CLEF_BENCH_DELTA_SOURCE=kernels/delta_persistent_vector.loom \
  python3 scripts/bench_workload.py delta-vector-repro --suite kernels \
  --filter paired_delta_prefill/candidate
cargo test --locked --release --lib delta_variants_preserve_state_arithmetic \
  -- --ignored
cargo build --locked --release --bin clef
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision int4 --output tests/results/jevbench-vector-int4-repro
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --prepared tests/results/jevbench-vector-int4-repro \
  --output tests/results/jevbench-vector-bf16-repro
```

Use `compare_jevbench.py` against the respective qualified precision baseline,
as in the earlier sections. Keep the single-model runs separate from the
[alternating two-model comparison](optimization.md#recording-and-comparing-runs),
which retains additional weights in memory.

After a host restart, a fresh single-model INT4 run completed at **883.33 input
tok/s**: 166,394 resident tokens in 188.370 seconds, with 194/231 correct, all
231 strictly valid and all outputs unchanged. Median/p95 resident latency was
0.326/2.797 seconds; peak engine allocation remained 14.508 GiB. The binary and
inputs were unchanged. This is a new complete resident measurement, not an
isolated estimate of the kernel's end-to-end gain over the earlier 831.62 run.
The 1,000 tok/s gate still fails.

Records are in `tests/results/jevbench-20261011-int4-delta-vector8-postboot`;
`comparison.json` includes `required_tok_s: 1000` and correctly reports failure
of that throughput gate despite passing accuracy. The compiler cache was
populated. Telemetry beginning during the run is in
`artifacts/throughput-1000/delta-vector8-postboot-telemetry.jsonl`.

An intervening sequential old-binary control was interrupted by the restart at
178/231 cases and has no valid full-suite summary. Its partial files and
`interrupted.json` remain in `jevbench-20261011-int4-vector-control`. The earlier
two-model attempt also produced no timing result: its second loader failed the
available-memory preflight. Neither incomplete attempt is used as a speed result.

## Kernel-source fingerprint cache

Kernel dispatch previously copied and hashed the full Loom source with every
cache lookup. The engine now memoizes a SHA-256 fingerprint, verifies source
content on memo hits, and keys compiled kernels by fingerprint, export symbol
and configuration. The bounded memo safely handles reused source allocations;
eviction does not remove compiled kernels. This changes CPU preparation only.

Both complete sequential JevBench runs preserved all 231 logit and probability
arrays against their respective qualified baselines:

| Mode | Correct / strict valid | Resident input tok/s | Resident seconds | Median / p95 seconds | Peak engine GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 default | 202 / 231 | 524.73 | 317.107 | 0.658 / 4.416 | 47.773 |
| INT4 eligible backbone | 194 / 231 | 906.55 | 183.546 | 0.294 / 2.799 | 14.508 |

Each rate includes encoding, preparation, compilation, GPU execution and readback
for 166,394 resident tokens, excluding only the first request including model
loading. The compiler cache was populated. INT4 mean preparation fell from
60.57 to 30.74 ms against the preceding 883.33 tok/s run; mean GPU execution
changed from 755.70 to 764.58 ms. The measured end-to-end rate rose 2.6%.
These are sequential runs, not a paired causal estimate. The older BF16
411.94 tok/s result also differs in kernel selection and host load, so its
27.4% rate difference cannot be assigned to this CPU change.

Results live in `tests/results/jevbench-20261011-{int4,bf16}-source-fingerprint`,
with `comparison.json` and `output-equality.json`. The INT4 `throughput-gate.json`
explicitly fails the still-unmet **1,000 tok/s** target while passing accuracy.
The archived normal binary is `artifacts/throughput-1000/clef-source-fingerprint`,
SHA-256 `0ef451de8cc33806b80b8097df8b8aa4ac6bf011c9fcb314ba8dcf9f4c3e7e67`.
Its `-source.json` hashes all 102 files in
`benches/results/source-fingerprint-cpu/source`.

The CPU Criterion run `source-fingerprint-cpu` measures warm key generation and
two map lookups using three real source lengths and a fixed eight-entry
configuration. Mean time changed from 1,550 to 450 ns for pointwise, 9,713 to
736 ns for integer GEMM, and 4,694 to 514 ns for DeltaNet. These are CPU
microbenchmarks, not model throughput. Source-allocation reuse and export/config
identity tests, the CPU test suite and all-target/all-feature Clippy passed.

Reproduce with fresh artifact names, running GPU jobs sequentially:

```sh
python3 scripts/bench_workload.py source-fingerprint-cpu-repro \
  --suite preprocessing --filter kernel_cache_key
cargo build --locked --release --bin clef
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision int4 --output tests/results/jevbench-fingerprint-int4-repro
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --prepared tests/results/jevbench-fingerprint-int4-repro \
  --output tests/results/jevbench-fingerprint-bf16-repro
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261011-int4-delta-vector8-postboot \
  tests/results/jevbench-fingerprint-int4-repro --min-tps 1000
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261011-bf16-swiglu \
  tests/results/jevbench-fingerprint-bf16-repro
```

The INT4 comparison returns nonzero when throughput is below the target, even
when accuracy passes. Use the archived binary or source snapshot to reproduce
the exact measured implementation after subsequent changes.

## Register-based integer packing

The selected packing kernel keeps transformed FP32 values in registers until
the row scale is known. Wave shuffles implement the same radix-4 Hadamard
arithmetic; only the maximum reduction crosses waves. It replaces LDS packing
at backbone widths 5,120, 6,144 and 17,408. The BF16 path is unchanged.

The complete INT4 JevBench run scored **194/231**, with all 231 requests strictly
valid and every logit/probability array identical to the source-fingerprint
baseline. It processed 166,394 resident input tokens in **176.344 seconds**:
**943.58 input tok/s**, compared with 906.55 previously. Median/p95 resident
latency was 0.282/2.697 seconds, and peak engine allocation stayed 14.508 GiB.
The **1,000 tok/s target remains unmet**.

The timing includes encoding, preparation, compilation, GPU execution and
readback, excluding only the first request including model loading. The compiler
cache was populated. Mean GPU time fell from 764.58 to 732.84 ms; mean preparation
was 31.08 ms versus 30.74 previously. This is a 4.1% increase between sequential
full runs, not a paired full-model estimate. Telemetry covers preflight through
completion and reports only the native evaluation PID in KFD.

Records are in `tests/results/jevbench-20261011-int4-register-packing`.
`comparison.json` includes the failed 1,000 tok/s gate alongside unchanged
accuracy; `output-equality.json` records all 231 exact arrays and peak allocation.
Telemetry is `artifacts/throughput-1000/register-packing-eval-telemetry.jsonl`.
Archived normal binary: `artifacts/throughput-1000/clef-register-packing`, SHA-256
`3f8bfc3605c46757ab73d5afbc3b0155fef92441cad1ca8cb650a56214ec1483`.
Its `-source.json` hashes all 103 files in
`benches/results/register-packing-selected-smoke/source`.

All 44 paired operator cases matched code and scale bits, with 1.32–4.30×
speedups and every 99% lower bound above 1.0. CPU tests, all-target/all-feature
Clippy, the independent packing CPU oracle, exceptional-value/slice/replay GPU
checks and the 12-case selected-source smoke sweep passed. BF16's unchanged path
retains the preceding 202/231, 524.73 tok/s measurement; it was not remeasured for
this INT4-only change.

Reproduce with fresh result directories:

```sh
CLEF_BENCH_PACKING_SOURCE=kernels/prepare_integer_register.loom \
  python3 scripts/bench_workload.py register-packing-repro --filter paired_packing
cargo test --locked --release --lib packing_ -- --ignored
cargo build --locked --release --bin clef
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision int4 --output tests/results/jevbench-register-packing-repro
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261011-int4-source-fingerprint \
  tests/results/jevbench-register-packing-repro --min-tps 1000
```

The last command returns nonzero below 1,000 tok/s even if accuracy passes.
Use the archived binary/source snapshot for the exact recorded implementation.

## Fused SwiGLU and packing

For quantized MLP down projections with at least 158 rows, SwiGLU now feeds the
register-based Hadamard transform directly. The kernel preserves BF16 rounding
after SiLU and after multiplication, while avoiding the intermediate activation
buffer. Shorter inputs and BF16 projections retain the separate path.

The complete INT4 JevBench run scored **194/231**, with all 231 requests strictly
valid and all logit/probability arrays identical to the register-packing baseline.
It processed 166,394 resident input tokens in **172.051 seconds**, measuring
**967.12 input tok/s** versus 943.58 previously. Median/p95 resident latency was
0.276/2.598 seconds. Peak engine allocation fell from 14.508 to **14.278 GiB**.
The **1,000 tok/s target remains unmet**.

The rate includes encoding, preparation, compilation, GPU execution and readback,
excluding only the first request including model loading. The compiler cache was
populated, and full-run telemetry reports only the evaluation PID in KFD. Mean
GPU time fell from 732.84 to 715.21 ms and mean preparation from 31.08 to 30.02 ms.
The 2.5% increase compares separate sequential full runs, not paired model timing.

Records are in `tests/results/jevbench-20261011-int4-swiglu-packing`, including
`comparison.json` with the failed throughput gate and `output-equality.json`.
Telemetry is `artifacts/throughput-1000/swiglu-packing-eval-telemetry.jsonl`.
Archived normal binary: `artifacts/throughput-1000/clef-swiglu-packing`, SHA-256
`239c9c922c61be483b06fad05d9cb04f01c9302c87e4f2d4b8e1f8589fc0e6b7`.
Its `-source.json` hashes all 103 files in
`benches/results/swiglu-packing-selected-smoke/source`.

The paired operator experiment preserved exact packed codes and scale bits at
all eleven lengths. The seven measured lengths from 158 through 4,032 improved
1.32–1.59×, with all 99% lower bounds above 1.0. Short-length regressions motivated
the fallback. CPU tests, Clippy, the independent packing oracle and GPU checks
for INT4/INT8, input slices, exceptional values and replay passed. BF16's existing
202/231 and 524.73 tok/s measurement remains historical; this run qualifies INT4.

Reproduce sequentially, with fresh names:

```sh
CLEF_BENCH_PACKING_SWIGLU=1 \
  CLEF_BENCH_PACKING_SOURCE=kernels/prepare_integer_register.loom \
  python3 scripts/bench_workload.py swiglu-packing-repro \
  --filter 'paired_packing/pair/.*x17408'
cargo test --locked --release --lib packing_ -- --ignored
cargo build --locked --release --bin clef
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision int4 --output tests/results/jevbench-swiglu-packing-repro
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261011-int4-register-packing \
  tests/results/jevbench-swiglu-packing-repro --min-tps 1000
```

The final command returns nonzero below the target even when accuracy passes.

## Adaptive INT4 MLP tiles

INT4 MLP projections now select M128/N128/K256 when its padded row count is at
least 12.5% smaller than M256's: rows 1–128, 257–384, 513–640 and 769–896.
Other MLP down projections use M256/N128/K256; up/gate outside those bands
retains M256/N128/K128. Host row grouping and the packed weight format are
unchanged. BF16, INT8 and other projection families retain their existing paths.

The full JevBench run scored **194/231**, with all 231 requests strictly valid,
identical input token counts and every logit/probability array exactly matching
the fused-packing baseline. It processed 166,394 resident input tokens in
**170.407 seconds**, or **976.45 input tok/s**, versus 967.12 previously.
Median/p95 resident latency was 0.274/2.565 seconds. Peak engine allocation
remained **14.278 GiB**. The **1,000 tok/s target remains unmet**.

Timing includes encoding, graph preparation, compilation, GPU execution and
readback, excluding only the first request including model loading. The compiler
cache was populated. Mean GPU time was 708.72 ms and preparation 29.48 ms.
The 0.96% increase compares separate sequential full runs; it does not establish
a paired full-model speedup. All 254 telemetry samples reported no KFD client
other than the native evaluation process when a client was present.

Records are in `tests/results/jevbench-20261011-int4-adaptive-gemm`, including
`comparison.json` with the failed throughput gate and `output-equality.json`.
Telemetry is `artifacts/throughput-1000/adaptive-gemm-eval-telemetry.jsonl`.
Archived normal binary: `artifacts/throughput-1000/clef-adaptive-gemm`, SHA-256
`3c6abeb09e45a48938270715b8f4a9274022906323a71af78e944db7d2d381d9`.
Its `-source.json` hashes all 105 files in
`benches/results/adaptive-wide-k-selected-smoke/source`.

The selected sources passed 44 M128 and 22 wider-K exact-output smoke cases
with independent, nonuniform row scales and zero-scale rows. Full and boundary
paired sweeps support the conditional choices; the direct M128/wider-K boundary
comparison also had every 99% speedup lower bound above 1.0. CPU tests and
all-target/all-feature Clippy passed. BF16's 202/231 and 524.73 tok/s measurement
is from the earlier source-fingerprint build; BF16 was not rerun for this change.

Reproduce sequentially, with fresh result names:

```sh
CLEF_BENCH_INTEGER_BITS=4 CLEF_BENCH_INTEGER_PADDED=1 \
  CLEF_BENCH_INTEGER_ROW_SCALES=1 \
  CLEF_BENCH_INTEGER_SOURCE=kernels/gemm_integer_m128.loom \
  python3 scripts/bench_workload.py adaptive-m128-repro \
  --filter 'paired_integer_gemm/mlp_'
CLEF_BENCH_INTEGER_BITS=4 CLEF_BENCH_INTEGER_PADDED=1 \
  CLEF_BENCH_INTEGER_ROW_SCALES=1 \
  CLEF_BENCH_INTEGER_SOURCE=kernels/gemm_integer_wide_k.loom \
  python3 scripts/bench_workload.py adaptive-wide-k-repro \
  --filter 'paired_integer_gemm/mlp_down'
CLEF_BENCH_INTEGER_BITS=4 CLEF_BENCH_INTEGER_PADDED=1 \
  CLEF_BENCH_INTEGER_ROW_SCALES=1 CLEF_BENCH_LENGTHS=257,384,513,640,769,896 \
  CLEF_BENCH_INTEGER_REFERENCE_SOURCE=kernels/gemm_integer_wide_k.loom \
  CLEF_BENCH_INTEGER_SOURCE=kernels/gemm_integer_m128.loom \
  python3 scripts/bench_workload.py adaptive-boundary-repro \
  --filter 'paired_integer_gemm/mlp_down'
cargo build --locked --release --bin clef
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision int4 --output tests/results/jevbench-adaptive-gemm-repro
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261011-int4-swiglu-packing \
  tests/results/jevbench-adaptive-gemm-repro --min-tps 1000
```

The final command returns nonzero below the target even when accuracy passes.
Use the archived binary/source snapshot for the exact recorded implementation.

## Pipelined narrow BF16 projections

The 5,120→1,024 BF16 projection now uses the existing pipelined GEMM at
257–4,032 rows. This covers full-attention K/V projections in both BF16 and
INT4 modes. Shorter inputs and lengths beyond the measured range keep their
previous dispatch. No projection changes precision.

Paired operator comparisons cover 128, 158, 235, 256, 257, 384, 512, 513, 620,
768, 1,024, 2,048 and 4,032 rows, with replay and rotating weights. All outputs
match bits. All measured cases in the selected range improve 1.09–2.67×, with
every 99% lower bound above 1.0. Short shapes include regressions and are excluded.
Artifacts: `benches/results/bf16-kv-pipelined-paired` and
`benches/results/bf16-kv-pipelined-boundaries`.

Full INT4 JevBench preserved every output array and input token count, scoring
194/231 with all 231 requests strictly valid. It processed 166,394 resident
tokens in 170.193 seconds: **977.68 input tok/s**, versus 976.45 previously.
Median/p95 latency was 0.274/2.619 seconds and peak engine allocation 14.278 GiB.
The 0.13% difference between sequential full runs is not evidence of a reliable
model-level speedup. Mean GPU time fell from 708.72 to 706.10 ms while preparation
rose from 29.48 to 31.24 ms. The 1,000 tok/s target remains unmet.

Full BF16 also preserved all 231 output arrays and token counts, scoring
202/231 with all requests strictly valid. It processed the same 166,394 resident
tokens in 321.067 seconds: **518.25 input tok/s**, below the earlier 524.73 run.
Median/p95 latency was 0.653/4.535 seconds and peak engine allocation 47.773 GiB.
Mean GPU time was 1,362.34 ms and preparation 30.86 ms. These full measurements
qualify arithmetic and report actual throughput; they do not establish an
overall speedup from the operator improvement.

The normal binary is `artifacts/throughput-1000/clef-kv-pipelined`, SHA-256
`eb88b1aa04c026c280426419fa03f28d757a93ad7de2f0261d9c7fb7e4d9ab84`.
Its `-source.json` hashes all 105 files in
`artifacts/throughput-1000/kv-pipelined-source`. Records are in
`tests/results/jevbench-20261011-int4-kv-pipelined`; telemetry is
`artifacts/throughput-1000/kv-pipelined-int4-eval-telemetry.jsonl`.
BF16 records use `tests/results/jevbench-20261011-bf16-kv-pipelined` and
`artifacts/throughput-1000/kv-pipelined-bf16-eval-telemetry.jsonl`.
Both result directories contain `comparison.json` and `output-equality.json`.
The INT4/BF16 telemetry has 254/335 samples, respectively; each reports only
its native evaluation process in KFD when a client is present.
Timing includes encoding, preparation, compilation, GPU work and readback,
excluding only the first request including model loading. The compiler cache
was populated. CPU tests, Clippy and GPU parity with independent sliced operands,
bias, ragged rows and replay passed.

Reproduce with fresh names, one GPU process at a time:

```sh
CLEF_BENCH_GEMM_REFERENCE=wide \
  CLEF_BENCH_GEMM_SOURCE=kernels/gemm_bf16_pipelined.loom \
  CLEF_BENCH_LENGTHS=128,158,235,256,257,384,512,513,620,768,1024,2048,4032 \
  python3 scripts/bench_workload.py bf16-kv-pipelined-repro \
  --filter paired_bf16_gemm/kv_and_head
cargo test --locked --release --lib pipelined_gemm_parity -- --ignored
cargo build --locked --release --bin clef
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision int4 --output tests/results/jevbench-int4-kv-pipelined-repro
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261011-int4-adaptive-gemm \
  tests/results/jevbench-int4-kv-pipelined-repro --min-tps 1000
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision bf16 --output tests/results/jevbench-bf16-kv-pipelined-repro
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261011-bf16-source-fingerprint \
  tests/results/jevbench-bf16-kv-pipelined-repro
```

The INT4 comparison returns nonzero below 1,000 tok/s even when accuracy passes.

## Attention-gate packing and bounded tile changes

Three operator changes qualified together: sixteen DeltaNet value channels per
wave at 128–1,024 tokens; M128 INT4 tiles for attention output, Q/gate, DeltaNet
QKV and Z in the existing 257–384, 513–640 and 769–896 padding bands; and fused
attention gate extraction, sigmoid, multiplication and integer packing from
128 rows. Both BF16 rounding boundaries in gating remain explicit. BF16 uses
only the DeltaNet change; integer projections remain opt-in.

Full sequential JevBench results, excluding only the first request including
model loading:

| Mode | Correct / strictly valid | Resident tokens / seconds | Input tok/s | Median / p95 seconds | Peak engine GiB |
| --- | --- | --- | ---: | --- | ---: |
| INT4 | 194/231 / 231 | 166,394 / 169.689 | 980.58 | 0.289 / 2.528 | 14.078 |
| BF16 | 202/231 / 231 | 166,394 / 320.028 | 519.94 | 0.660 / 4.495 | 47.773 |

Every question output array and input token count matches the corresponding
pipelined K/V baseline exactly. INT4 mean GPU time fell from 706.10 to 695.16 ms,
while preparation rose from 31.24 to 39.99 ms. BF16 mean GPU time was 1,358.08 ms
and preparation 30.62 ms. Compilation remains inside the timing boundary.
These separate full runs do not establish a paired model-level speedup, and
INT4 still fails the 1,000 tok/s gate.

Records: `tests/results/jevbench-20261011-{int4,bf16}-attention-gate-packing`,
including `comparison.json` and `output-equality.json`. Telemetry:
`artifacts/throughput-1000/attention-gate-packing-{int4,bf16}-eval-telemetry.jsonl`;
the 253/334 samples report only each native evaluation process in KFD when a
client is present. Normal binary `artifacts/throughput-1000/clef-attention-gate-packing`
has SHA-256 `e467185c788e341786c767ca00d7441f7304bcda65d90696e56ae798d9978ebb`.
Its `-source.json` hashes all 106 files in `attention-gate-packing-source`.

An unchanged-binary repeat, `tests/results/jevbench-20261011-int4-attention-gate-packing-warm`,
measured **992.76 tok/s** (166,394 tokens / 167.607 seconds), again preserving all
231 output arrays exactly. Mean preparation fell to 29.25 ms while GPU time
was 696.73 ms. Median/p95 latency was 0.274/2.528 seconds. This retained repeat
shows why the first run alone cannot isolate preparation costs from cache
warmup; it still misses 1,000 tok/s. Its telemetry has 181 samples and only the
native evaluation KFD client. Both measurements use the same binary SHA above.

Paired operator evidence is in `delta-vector16-paired`,
`int4-m128-other-boundaries-paired` and `attention-gate-packing-paired-v2` under
`benches/results`. All selected measured cases match bits and have 99% speedup
lower bounds above 1.0. Longer DeltaNet shapes and short attention gating retain
their earlier implementations where comparisons regressed. GPU tests cover
selection boundaries, sliced operands, distinct recurrence gates, INT4/INT8
packing, exceptional values and deterministic replay. Unused Q halves are
poisoned with NaNs in the gate-packing fixture. CPU tests and Clippy passed.

Reproduce sequentially with fresh names (use the archived source for the exact
implementation measured above):

```sh
CLEF_BENCH_DELTA_REFERENCE_SOURCE=kernels/delta_persistent_vector.loom \
  CLEF_BENCH_DELTA_SOURCE=kernels/delta_persistent_vector16.loom \
  CLEF_BENCH_DELTA_VALUES=16 CLEF_BENCH_PAIRED_SECONDS=3 \
  python3 scripts/bench_workload.py delta-vector16-repro --suite kernels \
  --filter paired_delta_prefill/candidate
CLEF_BENCH_PACKING_ATTENTION_GATE=1 \
  CLEF_BENCH_PACKING_SOURCE=kernels/prepare_integer_register.loom \
  python3 scripts/bench_workload.py attention-gate-repro \
  --filter 'paired_packing/pair/.*x6144'
CLEF_BENCH_INTEGER_BITS=4 CLEF_BENCH_INTEGER_PADDED=1 \
  CLEF_BENCH_INTEGER_ROW_SCALES=1 CLEF_BENCH_LENGTHS=257,384,513,640,769,896 \
  CLEF_BENCH_INTEGER_SOURCE=kernels/gemm_integer_m128.loom \
  python3 scripts/bench_workload.py attention-m128-repro \
  --filter 'paired_integer_gemm/(attention_output|delta_qkv|delta_z|attention_q_gate)'
cargo test --locked --release --lib packing_ -- --ignored
cargo test --locked --release --lib delta_variants_preserve_state_arithmetic -- --ignored
cargo build --locked --release --bin clef
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision int4 --output tests/results/jevbench-int4-attention-gate-repro
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261011-int4-kv-pipelined \
  tests/results/jevbench-int4-attention-gate-repro --min-tps 1000
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision bf16 --output tests/results/jevbench-bf16-attention-gate-repro
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261011-bf16-kv-pipelined \
  tests/results/jevbench-bf16-attention-gate-repro
```

The INT4 comparison returns nonzero below 1,000 tok/s even when accuracy passes.

## Projection residual fusion

The candidate adds the residual in the INT4 projection epilogue at 128–4,032
rows for attention/DeltaNet output and MLP down projections. It preserves the
projection's BF16 rounding before adding the residual, followed by the original
BF16 result rounding. Other shapes and precisions retain separate addition.
Matched operator results and reproduction are in
[the optimization log](optimization.md#fused-projection-residuals).

The first full INT4 run shared the GPU with an unrelated server. It preserved
all 231 output arrays (194 correct, all strictly valid) but measured only
629.70 input tok/s. It is retained as
`tests/results/jevbench-20261011-int4-residual-projection`; this is not a clean
throughput comparison. Its telemetry records both KFD clients.

After the other GPU workload stopped, the complete repeat measured **969.78
input tok/s**: 166,394 resident tokens in 171.579 seconds, median/p95 latency
0.275/2.597 seconds, and 14.078 GiB peak engine allocation. All 231 output arrays
and token counts again matched exactly. Its 186 telemetry samples contain only
the evaluator's KFD client when a client is present. Records:
`tests/results/jevbench-20261011-int4-residual-projection-repeat`.
The 1,000 tok/s gate failed; the preceding warm binary measured 992.76 tok/s
in an earlier run, so kernel-level savings alone do not establish an
end-to-end improvement.

BF16 measured **511.20 input tok/s** (166,394 / 325.498 seconds), 202/231 correct,
all strictly valid and all output arrays exact. Median/p95 latency was
0.660/4.642 seconds and peak engine allocation was 47.816 GiB. Records:
`tests/results/jevbench-20261011-bf16-residual-projection`. Mean preparation and
GPU time were 31.36 and 1,381.08 ms, respectively. This was also an exclusive-GPU
run. BF16 projection arithmetic is unchanged; host temporary lifetimes differ.

These runs use the normal binary
`artifacts/throughput-1000/clef-residual-projection`, SHA-256
`1fd2083d29b3013139cc6482e21eb3b7ee54f59cf45f2c0cdb6040356503542b`.
Its `-source.json` hashes all 106 files in `residual-projection-source`.
Telemetry is recorded alongside it as
`residual-projection-{int4,bf16,int4-mlp}-eval-telemetry.jsonl` and
`residual-projection-repeat-int4-eval-telemetry.jsonl`.

MLP-only INT4 measured **740.23 input tok/s** (166,394 / 224.786 seconds),
199/231 correct, all strictly valid and all 231 arrays and token counts exact
against its earlier qualified run. Median/p95 latency was 0.419/3.244 seconds;
peak engine allocation was 23.962 GiB. Mean preparation/GPU time was
28.67/945.97 ms. Records:
`tests/results/jevbench-20261011-int4-mlp-residual-projection`.
This updates the older 625.29 tok/s, 32.532 GiB observation after several
intervening optimizations; the difference cannot be attributed solely to
residual fusion.

Reproduce with an idle GPU and fresh output names. Use the archived candidate
source above if later changes supersede this experiment:

```sh
CLEF_BENCH_INTEGER_BITS=4 CLEF_BENCH_INTEGER_PADDED=1 \
  CLEF_BENCH_INTEGER_ROW_SCALES=1 CLEF_BENCH_INTEGER_RESIDUAL=1 \
  CLEF_BENCH_INTEGER_RESIDUAL_EXPORT=h3_gemm_i4_256_residual \
  CLEF_BENCH_INTEGER_REFERENCE_SOURCE=kernels/gemm_integer_wide_k.loom \
  CLEF_BENCH_INTEGER_SOURCE=kernels/gemm_integer_wide_k.loom \
  python3 scripts/bench_workload.py residual-wide-k-repro --profile quick \
  --filter paired_integer_gemm/mlp_down
cargo test --locked --release --lib residual_projection_matches_separate_addition -- --ignored
cargo build --locked --release --bin clef
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision int4 --output tests/results/jevbench-int4-residual-repro
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261011-int4-attention-gate-packing-warm \
  tests/results/jevbench-int4-residual-repro --min-tps 1000
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision bf16 --output tests/results/jevbench-bf16-residual-repro
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261011-bf16-attention-gate-packing \
  tests/results/jevbench-bf16-residual-repro
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision int4-mlp --output tests/results/jevbench-int4-mlp-residual-repro
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261010-int4-mlp \
  tests/results/jevbench-int4-mlp-residual-repro
```

The INT4 comparison returns nonzero below 1,000 tok/s even when accuracy passes.
For attention-output operators, use `gemm_integer.loom` for both source flags
and filter `paired_integer_gemm/attention_output`. For M128 boundary cases,
use `gemm_integer_m128.loom` for both flags, set
`CLEF_BENCH_LENGTHS=257,384,620,896`, and include both projection filters.

A fresh unchanged-binary control measured **958.48 tok/s**, followed immediately
by a second candidate run at **967.61 tok/s**, an observed 0.95% increase.
Both scored 194/231, all strictly valid, with every question array and token
count exact. This is a sequential comparison, not a paired model-level
confidence interval. It supports retaining the fusion alongside the matched
operator gains, while the 1,000 tok/s target remains unmet.

The control processed 166,394 tokens in 173.601 seconds; the candidate took
171.963 seconds. Candidate median/p95 latency was 0.273/2.616 seconds, mean
preparation/GPU time 30.06/714.87 ms, and peak engine allocation 14.078 GiB.
The preceding control's preparation/GPU time was 31.65/720.43 ms. Records:
`tests/results/jevbench-20261011-int4-residual-control` and
`tests/results/jevbench-20261011-int4-residual-projection-clean`; telemetry:
`residual-control-int4-eval-telemetry.jsonl` and
`residual-projection-clean-int4-eval-telemetry.jsonl`. Both used exclusive GPU
access. The control uses the archived attention-gate-packing binary above;
the candidate uses the residual-projection binary. Both clean candidate runs
are retained, giving the **967.61–969.78 tok/s** range in the README.
