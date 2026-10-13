# Long-context attention tuning

These October 12, 2026 experiments build on the
[attention precision comparison](attention-precision.md) on shared gfx1151
hardware. The reference is the optimized 64-query, 32-key, single-probability
component kernel. BF16 and FP16 below describe attention operands; the attempted
model check uses original CLEF with broader INT4 projections.

## Candidates

Increasing occupancy did not improve throughput. A 32-query tile with eight
waves reduced LDS from 64,768 to 40,064 bytes and VGPRs from 180 to 152. Reported
occupancy rose from 25% to 37.5%, but the extra workgroups and reduced data reuse
made it slower. A 64-query tile with sixteen waves retained data reuse and
reached 50% reported occupancy. It also regressed, including after removing
duplicated softmax work. These variants preserve output bits on the paired
fixtures but are not selected.

Two or four independent QK accumulation chains changed FP32 summation order.
Neither established a long-context gain: their 99% paired intervals include 1.0.

The useful changes were:

- **64-key softmax tiles.** Each workgroup retains 64 queries and eight waves.
  Keys and values stage in two 32-key halves so the kernel needs only 63,744 LDS
  bytes. It updates online softmax once per 64 keys, with a different reduction
  order. It uses 195 VGPRs without spills.
- **Value prefetch.** Load V before QK computation and defer its LDS stores until
  after softmax. The 32-key version uses 196 VGPRs and preserves arithmetic.
  The combined 64-key version also prefetches the second V half during the first
  PV pass; it uses 212 VGPRs. Neither spills, and both retain 25% reported occupancy.

## Paired operator results

Each measurement alternates candidate/reference execution order and retains 20
measured batches. Confidence intervals use 10,000 paired bootstrap resamples at
99%. Value packing and completion fences are included; model preparation and
loading are excluded. These ratios are operator speedups, not request throughput.

| BF16 candidate vs optimized 32-key reference | 8,192 tokens | 36,916 tokens |
| --- | ---: | ---: |
| 32 queries, eight waves | 0.608× | 0.501× |
| Two QK chains | 1.006× | 1.022× |
| Four QK chains | 1.003× | 1.031× |
| 64 queries, sixteen waves | 0.777× | 0.820× |
| Sixteen waves, four softmax rows per wave | 0.865× | 0.840× |
| 64-key tile | 1.093× (1.083–1.101) | 1.158× (1.145–1.168) |
| Value prefetch, 32-key tile | 1.052× (1.043–1.061) | 1.103× (1.057–1.148) |
| 64-key tile + value prefetch, direct final comparison | 1.147× (1.100–1.221) | 1.136× (1.088–1.181) |

Comparing the combined 64-key/prefetch candidate directly with the 64-key
candidate gives **1.065× at 36,916 tokens** (99% interval 1.039–1.096), with
bit-for-bit identical outputs. Its 8,192-token result is inconclusive:
1.018× (0.978–1.047). Do not multiply estimates from separate runs to claim a
combined speedup.

The final direct comparison measures the combined candidate against the original
optimized 32-key reference. Both intervals exclude 1.0, but shared-GPU variation
is substantial; this does not establish that the combined candidate is better
than the plain 64-key tile in every workload.

FP16 value prefetch preserves output bits but does not establish a speed gain:
0.983× (0.933–1.030) at 8,192 tokens and 1.046× (0.999–1.099) at 36,916. Its
independent 64k oracle also passes. The FP16 result is kept separate from the
BF16 comparisons.

Combining FP16 with the 64-key tile and prefetch does establish a gain at
36,916 tokens: **1.103× (1.099–1.107)**, from 1,029.73 to 933.28 ms. At 8,192,
**1.016× (0.999–1.036)** remains inconclusive. Both compare with the optimized
32-key FP16 reference on original CLEF's head geometry. Maximum output error
is 0.001953125, replay is exact, and the independent 64k oracle passes. This
does not establish that FP16 is faster than BF16 or preserves model accuracy.

The 64-key tile differs from the 32-key fixture by at most 0.001953125 at both
lengths, within the existing 0.008 bound. Prefetch alone preserves output bits
in both paired cases. All variants retain exact replay. The independent
uniform-softmax oracle passes the 64-key tile and combined candidate through
65,536 tokens, including nonzero buffer slices, causal boundaries and replay,
without changing its 0.002 tolerance.

### Flash geometry

The combined BF16 candidate was also measured with Flash's 16 query heads and
four KV heads, against the same optimized 32-key reference:

| Tokens | Reference | Candidate | Speedup | 99% paired interval |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 67.18 ms | 79.19 ms | 0.848× | 0.507–1.330× |
| 16,384 | 144.57 ms | 121.85 ms | 1.186× | 1.170–1.205× |
| 30,342 | 430.55 ms | 353.18 ms | 1.219× | 1.162–1.255× |
| 65,536 | 3,798.83 ms | 3,552.07 ms | 1.069× | 1.025–1.115× |

The 8k result is inconclusive under substantial shared-GPU variation. The longer
three contexts establish operator gains, with a smaller improvement at 64k.
Maximum absolute error is 0.001953125 and replay is exact at every length.
These measurements do not establish an end-to-end Flash speedup or qualify a
new default.

## Model qualification held for memory

The seven-invoice check used a 32 GiB engine cap with the existing 20 GiB host
reserve and 1 GiB swap-growth stop. The guard stopped its owned process group
before any invoice completed: swap grew by 1.034 GiB, with 38.47 GiB still
available. System and task-cgroup OOM-kill counters were zero. The run was not
retried, and the guard was not relaxed.

There is no new original-CLEF full-model throughput or accuracy result for these candidates.
The larger key tile requires the same-precision WorkflowEvals accuracy gate;
passing an operator error bound does not establish model accuracy. These larger
key tiles are not selected in production. The earlier FP16 seven-invoice result remains the
latest completed original-CLEF check described in
[attention precision experiments](attention-precision.md#accuracy-investigation).

A separate Flash BF16 attempt used a 24 GiB cap, compared with the earlier
19 GiB measured engine allocation. It also stopped during loading, before any
request completed: swap grew by 1.002 GiB with 48.00 GiB still available.
The same guard remained in force; system and task-cgroup OOM-kill counts stayed
zero. The loader already calls `Checkpoint::done_with` after consuming each
tensor, which releases its complete mapped pages with `MADV_DONTNEED`; no
loader or host-memory settings were changed for these attempts.

## Flash INT4 pilot

The smaller Flash INT4 configuration completed the first seven invoices with
the combined BF16 64-key/prefetch candidate, a 14 GiB engine cap and the same
memory guard. Peak engine allocation was 10.23 GiB; no case failed or truncated.
Long vector DeltaNet was not enabled.

The six resident requests contain 164,064 input tokens and take 73.97 seconds:
**2,218.02 input tokens/s**. The historical production INT4 run took 298.15 seconds
on these same requests, or 550.28 tokens/s. This is a separate-run comparison on
shared hardware, not a paired model speedup estimate or a full-suite result.

The pilot **fails the accuracy screen** against production Flash INT4:

| Consensus metric | Production | Candidate | Gains | Losses |
| --- | ---: | ---: | ---: | ---: |
| Exact action sets | 9/21 | 10/21 | 6 | 5 |
| Primary actions | 17/21 | 15/21 | 2 | 4 |

The primary-action regression prevents promotion. These seven cases do not
qualify the remaining workflows or any other precision. Partial comparison
records are in `flash-int4-pilot-comparison.json` under the guard-record directory;
the frozen full-suite gate still requires all 705 cases.

The optimized 32-key BF16 control also fails this pilot: 10/21 action sets and
16/21 primary actions, at 2,072.31 resident input tokens/s. Against that control,
the 64-key candidate gains five and loses five action sets, and gains four and
loses five primary actions. Thus the regression is not confined to the new
64-key tile. All seven requests and token counts match the production baseline.

The subsequent 32-key FP16 model check stopped during loading after 1.069 GiB
of swap growth and completed no requests. It provides no Flash accuracy result.

## Preserving production's update order

The accuracy losses motivated a separate candidate generated by
`scripts/attention_key16_candidate.py`. It keeps shared Q/K/V staging and 64
queries per workgroup, but advances online softmax by 16 keys, as production
does. It retains both BF16 probability components. The unused second score
half is masked, and its V load and PV pass are removed, including the load that
would otherwise exceed the packed buffer on an aligned final tile.

Unlike the 32- and 64-key candidates, this version matches production output
bits on both paired Flash fixtures:

| Tokens | Production | Shared 16-key candidate | Speedup | 99% paired interval |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 227.56 ms | 55.37 ms | 4.110× | 4.079–4.142× |
| 30,342 | 5,880.79 ms | 832.18 ms | 7.067× | 6.939–7.182× |

Replay is exact, the independent boundary oracle passes through 65,536 tokens,
and a separate original-CLEF 8k probe also has zero error against production.
A direct Flash 65,536-token production comparison likewise has zero output
error and exact replay. Its single timing pair is 28.80 seconds versus 6.02
seconds; this is a correctness check with a timing observation, not a repeated
speedup estimate.
It gives up some of the larger tiles' speed to retain the original arithmetic
cadence. After full qualification, it is now selected by default for Flash INT4
causal text attention at 8,192–65,536 tokens. Original CLEF's quantized modes and
Flash MLP-only INT4 retain their previous paths; BF16 remains the default precision.

### Model qualification

After host memory became available, the Flash INT4 pilot completed seven cases
from each of the four workflows. All 51 native requests match production's
question logits and probabilities exactly; all 63 policy decisions are unchanged.
The invoice scores remain 9/21 action sets and 17/21 primary actions, recovering
the production result that the 32- and 64-key candidates failed to preserve.

The 50 resident requests contain 245,700 tokens and take 120.77 seconds,
or **2,034.48 input tokens/s**. The same saved production requests took 328.85
seconds, or 747.15 tokens/s. Peak engine allocation is 10.23 GiB. The current
pilot had no other KFD client; these timings are separate from the earlier
shared-GPU runs. This pilot is not the complete-suite accuracy gate.

Qualification records are under
`tests/results/attention-k16-qualification-20261012/`. The frozen executable,
source hashes and environment are in `manifest.json`; progress is recorded in
`flash-int4-status.json`. JevBench, context checks and the complete workflow
suite use the same candidate source with no DeltaNet override.

JevBench retains **165/231**, with zero probability changes and strictly valid
responses. This run measured 2,317.23 resident input tokens/s versus 2,518.98 in
the saved baseline; the new attention path is not selected for those short
requests. The context gate passes **15/15**, including exact 8k/64k replays,
with every probability matching production. Peak context-test allocation is
16.50 GiB.

The full **705-case WorkflowEvals run passed**, with no execution failures and
all 15 reported accuracy metrics unchanged. All **1,294 native predictions**
(including logits and probabilities) and **1,524 policy decisions** match the
former production implementation exactly.

Resident throughput is **1,847.81 input tokens/s**, a **3.10× speedup**. Comparing
the same 1,292 resident requests and 6,404,177 tokens gives 3,465.81 seconds for
the candidate versus 10,746.11 seconds for the saved baseline (595.95 tokens/s).
The run resumed after its pilot, so both model-loading requests are excluded
from this matched comparison. Encoding, preparation, compilation, GPU execution
and readback are included; CPU preflight and Python workflow execution are not.
These are separate complete runs, not an alternating-order model benchmark.

Peak engine allocation was **30.10 GiB** under the 40 GiB cap. Available host
memory stayed above 80 GiB, swap usage did not grow, and telemetry observed only
the owned GPU process. The full gate is in
`workflowevals-clef-flash-int4-comparison.json`; `flash-int4-prediction-audit.json`
records exact native predictions and matched timings, and
`promotion-evidence.json` records decision equality and memory observations.

The production kernel contains the measured instructions with updated comments.
Normal release builds select it without `bench-internals` or an environment
override. A release build without experiment features reproduced the qualified
predictions exactly for a 36,916-token invoice, a short JevBench request, the
8,192-token boundary and the invoice replayed afterward, in both BF16 and INT4.
The records are `default-bf16-validation.json` and `default-int4-validation.json`
in the qualification directory.
See [the full-suite instructions](workflowevals.md#flash-full-suite-results)
to reproduce with `--model clef-flash --precision int4`.

### Original CLEF INT4 screen

On October 13, the same 16-key kernel completed the first seven original-CLEF
invoices with broader INT4 projections and the original DeltaNet path. All seven
native predictions, including logits and probabilities, match production
exactly. All 21 policy decisions are unchanged: consensus agreement remains
14/21 exact action sets and 18/21 primary actions.

The six resident requests contain **164,064 tokens** and take **257.82 seconds**,
or **636.35 input tokens/s**. The saved production run took 901.04 seconds on
the same requests, or 182.08 tokens/s: a **3.49×** ratio. These are separate runs;
the baseline shared the GPU, while this candidate recorded only its owned
compute process. This seven-invoice result does not qualify the full suite or
change original CLEF's default kernel.

Peak engine allocation was **20.27 GiB** under a 40 GiB cap. Available host memory
stayed above 90 GiB and swap usage did not grow. Records and the exact-output
audit are in `tests/results/clef-k16-qualification-20261013/`; the audit is
`pilot-audit.json`, with the frozen binary, source and runner hashes in
`manifest.json`.

To reproduce the screen with an experiment-enabled CLI and a fresh output:

```sh
cargo build --release --locked --features bench-internals --bin clef
env -u CLEF_EXPERIMENT_DELTA_VECTOR_LONG \
  CLEF_EXPERIMENT_ATTENTION_SOURCE="$PWD/kernels/attention_shared_k16.loom" \
  artifacts/WorkflowEvals/.venv/bin/python scripts/eval_workflowevals.py \
    --binary target/release/clef --model clef --model-dir artifacts/model \
    --precision int4 --memory-gib 40 --workflows invoice_processing \
    --new-cases-per-workflow 7 --output tests/results/clef-k16-invoices-repro
```

Use the same live host-memory watchdog described below. The allocation cap
alone does not enforce the host reserve or swap-growth stop.

## Reproduce

Generate the reference and the two useful candidate changes:

```sh
python3 scripts/attention_probability_candidate.py \
  --output artifacts/attention-reference.loom
python3 scripts/attention_key_tile_candidate.py \
  --source artifacts/attention-reference.loom \
  --output artifacts/attention-k64.loom
python3 scripts/attention_value_prefetch_candidate.py \
  --source artifacts/attention-k64.loom \
  --output artifacts/attention-k64-prefetch.loom
cargo build --release --locked --features bench-internals --example probe_attention
target/release/examples/probe_attention --compile-only artifacts/attention-k64-prefetch.loom
target/release/examples/probe_attention --tokens 256 --rounds 3 artifacts/attention-k64-prefetch.loom
CLEF_BENCH_ATTENTION_REFERENCE_SOURCE=artifacts/attention-reference.loom \
CLEF_BENCH_ATTENTION_SOURCE=artifacts/attention-k64-prefetch.loom \
CLEF_BENCH_LENGTHS=8192,36916 \
  python3 scripts/bench_workload.py attention-k64-prefetch-repro \
    --suite workload --filter paired_attention
CLEF_BENCH_ATTENTION_ORACLE_SOURCE=artifacts/attention-k64-prefetch.loom \
  cargo test --release --locked --lib --features bench-internals \
    shared_attention_context_oracle -- --ignored --nocapture
```

To reproduce the candidate that retains production's 16-key update order:

```sh
python3 scripts/attention_key16_candidate.py \
  --source benches/kernels/attention_shared.loom \
  --output artifacts/attention-k16.loom
target/release/examples/probe_attention --model clef-flash --tokens 8192 \
  --rounds 3 --reference-source kernels/attention_tiled.loom \
  --max-error 0 artifacts/attention-k16.loom
target/release/examples/probe_attention --model clef-flash --tokens 65536 \
  --rounds 1 --reference-source kernels/attention_tiled.loom \
  --max-error 0 artifacts/attention-k16.loom
CLEF_BENCH_MODEL=clef-flash CLEF_BENCH_LENGTHS=8192,30342 \
CLEF_BENCH_ATTENTION_REFERENCE_SOURCE=kernels/attention_tiled.loom \
CLEF_BENCH_ATTENTION_SOURCE=artifacts/attention-k16.loom \
  python3 scripts/bench_workload.py attention-k16-repro --filter paired_attention
```

The explicit reference reproduces the former production kernel used in these
measurements; Flash BF16's default has since changed. The measured candidate
source is reproduced byte-for-byte by the generator.

To isolate prefetch, feed the 32-key reference directly to
`attention_value_prefetch_candidate.py`. For FP16, use
`benches/kernels/attention_shared_fp16.loom` as the starting source. Both
generators produce the measured source bytes exactly for these inputs and
refuse to overwrite an existing output. They do not change production dispatch.

Original-CLEF model qualification uses the existing frozen WorkflowEvals runner with the
candidate selected by `CLEF_EXPERIMENT_ATTENTION_SOURCE` and long DeltaNet held
fixed with `CLEF_EXPERIMENT_DELTA_VECTOR_LONG=1`. See the seven-invoice command
in [the precision reproduction instructions](attention-precision.md#reproduce).
For the Flash INT4 pilot, leave the DeltaNet override unset and run:

```sh
CLEF_EXPERIMENT_ATTENTION_SOURCE="$PWD/artifacts/attention-k64-prefetch.loom" \
  artifacts/WorkflowEvals/.venv/bin/python scripts/eval_workflowevals.py \
    --binary target/release/clef --model clef-flash \
    --model-dir artifacts/model-flash --precision int4 --memory-gib 14 \
    --workflows invoice_processing --new-cases-per-workflow 7 \
    --output tests/results/flash-int4-attention-pilot
```

Build the CLI with `bench-internals` for source overrides. The recorded pilot
wraps this command in a watchdog that requires 34 GiB available at admission,
then stops its owned process group below 20 GiB available or above 1 GiB of
additional swap usage. An engine allocation cap alone does not enforce those
host-memory limits.
Keep a live memory watchdog around model runs; the direct commands above only
set experiment options and do not supply a host-memory watchdog.

Sources, generators for rejected variants, compiler reports and manifests are
under `artifacts/attention-tiles-20261012/`. Guard records and the held model
attempt are under `tests/results/attention-tiles-20261012/`. Raw paired records,
source snapshots and telemetry are under `benches/results/attention-*-20261012/`;
the run names identify `q32-bf16`, `chains2-bf16`, `chains4-bf16`, `q64w16-bf16`,
`q64w16-soft4-bf16`, `k64-bf16`, `prefetch-v-bf16` and `k64-prefetch-bf16`.
The direct final comparison is `attention-k64-prefetch-final-20261012`; the FP16
comparison is `attention-prefetch-fp16-20261012`.
The combined FP16 extension is `attention-k64-prefetch-fp16-20261012`, using
the FP16 source as input to both candidate generators.
The Flash geometry extension is `attention-k64-prefetch-flash-20261012`, using
`CLEF_BENCH_MODEL=clef-flash` and
`CLEF_BENCH_LENGTHS=8192,16384,30342,65536` with the same BF16 sources.
The production-order candidate's paired run is `attention-k16-flash-20261012`.
