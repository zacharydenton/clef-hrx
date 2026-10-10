# Performance experiments

Optimize completed resident requests on the public JevBench suite. The target is
1,000 **input tokens/second**, including encoding, graph preparation, compilation,
GPU execution, and readback. Exclude only the first request, which includes model
loading. Kernel throughput and exact-input graph replay are diagnostic metrics;
neither substitutes for this end-to-end target.

Accept small numerical differences if the complete evaluation does not regress:
at least the accepted baseline's number correct (currently 202/231), all 231 requests strictly valid, the same
checkpoint and dataset, and identical input token counts. Report changed decisions,
probability drift, calibration, and tier scores. Keep the existing operator and
reference tests; do not loosen their tolerances to qualify a candidate.

## Criterion coverage

| Suite | Coverage | Timing boundary |
| --- | --- | --- |
| `workload` / `workload_gemm` | Nine projection families; 1, 65, 127, 128, 158, 235, 260, 620, 1,024, 2,048, 4,032 rows; repeated weights and a rotating working set of at least 64 MiB | Completed dispatch, excluding allocation, compilation, upload and readback |
| `workload` / `workload_delta` | Chunked and register-resident recurrence at every listed length; deterministic reset and allocation checks | Completed graph replay |
| `kernels` / `paired_delta_prefill/persistent` | Alternating-order chunked/register comparisons at 235, 620 and 1,024 tokens; raw paired timing totals | Completed graph replay, with both variants resident |
| `workload` / `workload_attention` | Causal 24-query/4-KV-head attention, 256 dimensions; 16 through 4,032 tokens | Completed graph, including GPU value transpose/padding and the selected attention kernel |
| `normalization`, `preprocessing` | Normalization shapes and CPU request processing | Existing suite-specific boundaries |
| `workload_inference` | Seven representative requests selected by nearest token length from an actual request corpus; exact-input replay and graph rebuilding | Complete `infer` calls, including preparation in rebuild cases |

Kernel groups use 20 samples, a 1-second warmup, 3-second measurement windows,
99% confidence intervals, and a 3% noise threshold. Full-model groups use 10 samples,
3-second warmups, and 30-second measurement windows. Criterion may extend the run
to collect its samples. GEMM throughput counts multiply/add operations (floating
point for BF16, integer for INT4/INT8); other groups count tokens. Never report a
GEMM's operation rate as model token throughput.
Each kernel case constructs and validates its graph once, outside warmup and timed
samples. Repeated samples reuse the same allocations; rotating cases advance to
the next resident weight buffer after each completed dispatch.

The `quick` profile limits GEMM and DeltaNet to 235, 620 and 4,032 tokens. Attention
retains its tile-aligned sweep. Use filters to investigate a specific bottleneck;
run the full shape sweep before selecting an implementation. Attention uses stable
tile-aligned benchmark IDs; operator tests additionally cover odd lengths, noncausal
cross-attention, exact value packing, FP64 spot checks, and deterministic replay.

## Recording and comparing runs

Run GPU jobs sequentially. Record background GPU activity; repeat comparisons if
clocks, temperature or competing work differ. Do not kill unrelated GPU workloads.

```sh
# One iteration per case to check construction and execution, without a speed claim:
python3 scripts/bench_workload.py smoke --profile quick --test

# Establish a named baseline (start narrowly, then run the full suite):
python3 scripts/bench_workload.py baseline-delta --filter workload_delta

# Paired confirmation alternates reference/candidate order to reduce clock drift:
python3 scripts/bench_workload.py paired-delta --suite kernels \
  --filter paired_delta_prefill/persistent

# After an implementation change, compare the same shapes and save the candidate:
python3 scripts/bench_workload.py candidate-delta --filter workload_delta \
  --compare baseline-delta

# Complete projection coverage, including a rotating weight working set:
python3 scripts/bench_workload.py baseline-gemm --filter workload_gemm

# Full-model replay and preparation using requests from an evaluation preflight:
CLEF_BENCH_FULL_MODEL=1 CLEF_MODEL_DIR=artifacts/model \
  CLEF_BENCH_REQUESTS=tests/results/jevbench-20261010-full/requests.jsonl \
  python3 scripts/bench_workload.py baseline-model --suite workload_inference
```

Names must be unique. Each run writes `benches/results/<name>/`: exact source and
candidate files, hashes, compiler version, git diff, command, logs, half-second GPU
telemetry, copied Criterion estimates/samples, and `summary.json`. Criterion named
baselines remain in `target/criterion`; retain that directory for later comparisons.
`--compare` requires a baseline for every selected case. A full-model run also saves
phase observations for encoding, preparation, GPU work and readback. These include
warmup and explain wall time; use Criterion's wall-time confidence intervals for
acceptance. Invalidating a graph occurs inside the rebuild timing boundary.
The full-model request corpus is copied and hashed in the run record as well.

The runner labels a case an improvement only if the entire confidence interval
for relative mean time lies below -3%, a regression if it lies above +3%, and
otherwise inconclusive. This is a screening rule, not an automatic shipping gate.
Paired DeltaNet results additionally include a 99% speedup interval from 10,000
paired-batch bootstrap resamples, excluding warmup and preserving each batch's
reference/candidate pairing. Raw measurement batches remain in `paired.json`.
Repeat promising changes in reversed baseline/candidate order and inspect the
worst affected shapes. Multiple comparisons and thermal drift can produce false
winners. Do not select a change from one minimum-latency sample.

For kernel experiments without rebuilding Rust, set `CLEF_BENCH_GEMM_SOURCE`,
`CLEF_BENCH_DELTA_SOURCE`, or `CLEF_BENCH_ATTENTION_SOURCE` to a Loom file matching
the corresponding harness contract. DeltaNet also accepts
`CLEF_BENCH_DELTA_VALUES=4|8|16`. Candidate sources are copied into the run record.
The GEMM source override applies only to compatible pipelined shapes; the remaining
shapes still exercise automatic selection. Attention sources use the current native
ABI with runtime query/key lengths and padded V capacity, as in
`kernels/attention_tiled.loom`; the benchmark includes GPU value packing. Set
`CLEF_BENCH_ATTENTION_REFERENCE=1` to measure the original attention kernel. The
default measures production selection; a source override forces that source at
every measured length. These overrides are benchmark-only.
Experimental sources must produce finite, deterministic outputs; numerical drift
against the current kernel is logged for diagnosis. Bitwise agreement is not an
acceptance requirement. The complete evaluation remains the accuracy gate.

`CLEF_BENCH_PAD_GEMM=1` tests the h3-hrx approach of adding 64 elements to operand
row strides when K is a multiple of 1,024. This preserves logical matrix values
and checks exact output equality. Choose it separately from a GEMM source override.
Padding happens outside the timed region: a winning result still needs a production
weight layout and activation-packing implementation, with those costs included in
the full-model evaluation.

`CLEF_BENCH_INTEGER_SOURCE` accepts an experimental h3-style `h3_gemm_i4_256`
or `h3_gemm_i8_256` kernel with BF16 output and per-row FP32 scales. Select
`CLEF_BENCH_INTEGER_BITS=4|8` (default 8); `CLEF_BENCH_INTEGER_PADDED=1` adds
64 bytes of stride padding when K is a multiple of 1,024 (128 INT4 or 64 INT8
elements, preserving each kernel's required alignment). Synthetic inputs are exactly representable at the
selected precision, with scales of 1/8 or 1/128. Outputs are checked against
BF16 multiplication of those same values, within one BF16 step. This isolates
dispatch performance; it does not measure model quantization error or
activation-packing cost. Candidate sources are archived, and the complete
evaluation is required before using integer kernels in inference. The older
`CLEF_BENCH_INT8_SOURCE` and `CLEF_BENCH_INT8_PADDED` names remain aliases.

## Evaluation gate

Build the intended production binary, then follow [the evaluation protocol](evaluation.md).
Keep the original baseline and write each candidate into a fresh directory:

```sh
cargo build --release --locked --bin clef
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --output tests/results/jevbench-candidate
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261010-persistent tests/results/jevbench-candidate \
  --output tests/results/jevbench-candidate/comparison.json

# This additionally fails until the actual throughput target is reached:
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261010-persistent tests/results/jevbench-candidate --min-tps 1000
```

The comparison rejects incomplete or incompatible runs, changed token counts,
invalid outputs, and a lower total number correct. Individual gained/lost decisions
and maximum probability change are reported rather than required to match exactly.
The full-model `compare_delta` example similarly reports probability drift; it
checks finiteness but leaves the accuracy gate to the complete evaluation.

Use kernel measurements to choose the next experiment, representative full-model
cases to establish its effect, and the full evaluation to decide whether to keep it.
Preserve each raw result and change one major factor at a time.

## Recorded baseline, 10 October 2026

All 228 full-profile kernel smoke cases passed. The quick measured baseline
completed 68 cases and is stored locally in `benches/results/workload-baseline-v2/`.
The `v2` run initializes each graph once per case. Earlier exploratory runs
recreated graphs between samples and should not be used for new comparisons.
The alternating-order DeltaNet confirmation is in `benches/results/delta-paired/`:

| Tokens | Chunked mean | Register-resident mean | Paired speedup | 99% interval |
| ---: | ---: | ---: | ---: | ---: |
| 235 | 3.750 ms | 0.680 ms | 5.51× | 5.508–5.519× |
| 620 | 10.229 ms | 1.813 ms | 5.64× | 5.633–5.652× |
| 1,024 | 16.489 ms | 3.060 ms | 5.39× | 5.380–5.397× |

These are single-layer kernel results. The complete model's measured improvement
was 233.4 to 345.3 input tokens/s, with accuracy increasing from 201 to 202 correct.
See [evaluation results](evaluation.md) for the full timing boundary and shared-GPU
conditions. The 1,000 tok/s target remains unmet.

The h3-inspired stride-padding experiment is recorded in
`benches/results/gemm-padding-v2/`, compared against `workload-baseline-v2`.
All 12 MLP cases retained exact outputs, but the timing results were mixed:

| Projection, replay | Tokens | Mean latency change | 99% interval |
| --- | ---: | ---: | ---: |
| MLP up/gate | 235 | -20.0% | -20.3% to -19.6% |
| MLP down | 235 | -20.3% | -21.9% to -18.8% |
| MLP down | 620 | +9.0% | +7.7% to +10.0% |
| MLP down | 4,032 | +5.7% | +4.5% to +7.2% |

Padding remains experimental. Reproduce this screening comparison with fresh names:

```sh
python3 scripts/bench_workload.py mlp-base --profile quick \
  --filter 'workload_gemm/mlp_(up_gate|down)'
CLEF_BENCH_PAD_GEMM=1 python3 scripts/bench_workload.py mlp-pad --profile quick \
  --filter 'workload_gemm/mlp_(up_gate|down)' --compare mlp-base
```

Validation also passed all 30 GPU/fixture tests and both full-model benchmark
smoke modes. The strict checkpoint-reference test retains its existing failure
on the `outage` probability: 0.004716 absolute difference against a 0.003 tolerance
(the earlier recorded run differed by 0.005436). All fixture winners matched;
the tolerance remains unchanged. The complete JevBench accuracy gate passed.

## Tiled attention and integer probes

The current BF16 attention implementation batches 16 queries per wave, transposes
V on the GPU, and accumulates online softmax in FP32. Splitting probabilities into
high/residual BF16 components reduces the precision loss of the PV multiplication.
Runtime query/key lengths let different requests reuse compiled code. The pinned
compiler needs a loop-dependent zero for each QK accumulator; an invariant zero
produced stale tile offsets during testing. The source records this workaround.

Production selects tiled attention for 256-dimensional BF16 heads at 1,024 or more
query tokens; shorter requests retain the original kernel. Criterion's
`attention-dynamic` run includes value packing and compares against
`workload-baseline-v2`:

| Tokens | Original mean | Tiled mean | Latency change |
| ---: | ---: | ---: | ---: |
| 1,024 | 5.73 ms | 1.94 ms | -66.1% |
| 2,048 | 28.76 ms | 8.45 ms | -70.6% |
| 4,032 | 134.21 ms | 30.46 ms | -77.3% |

All three met the runner's 99% interval screening rule. An unchanged 624-token
control also moved by 4.4%, illustrating why between-run noise requires care.
All 31 GPU/fixture tests passed, including odd lengths, cross-attention, exact
transpose/padding, FP64 sampled comparisons and deterministic replay.

The full evaluation retained 202/231, no changed decisions, at 401.9 input tokens/s
(`jevbench-20261010-attention-long`). Its compiler cache was populated. The earlier
345.3-token/s run had substantially higher preparation time even on unchanged
short requests, so the recorded 16.4% end-to-end difference is not a controlled
kernel speedup. See [measurement details](evaluation.md). An earlier candidate
using tiled attention from 64 tokens scored 201/231 at 351.4 tokens/s and was
rejected (`jevbench-20261010-attention`).

Packed integer kernels remain benchmark-only. The h3-derived implementation is
checked in at `benches/kernels/gemm_integer.loom`; it supports both INT4 and INT8
with BF16 outputs. Each precision passed all 12 padded MLP smoke cases at 235,
620 and 4,032 tokens, including repeated and rotating weights. Reproduce a
correctness sweep or measured experiment with
fresh run names:

```sh
CLEF_BENCH_INTEGER_SOURCE=benches/kernels/gemm_integer.loom \
  CLEF_BENCH_INTEGER_BITS=4 CLEF_BENCH_INTEGER_PADDED=1 \
  python3 scripts/bench_workload.py int4-smoke --profile quick \
  --filter 'workload_gemm/mlp_(up_gate|down)' --test

# Use bits=8 to compare INT8; omit --test and supply a baseline for measurements.
CLEF_BENCH_INTEGER_SOURCE=benches/kernels/gemm_integer.loom \
  CLEF_BENCH_INTEGER_BITS=8 CLEF_BENCH_INTEGER_PADDED=1 \
  python3 scripts/bench_workload.py int8-measured --profile quick \
  --filter 'workload_gemm/mlp_(up_gate|down)' --compare workload-baseline-v2
```

The earlier 12-case INT8 measurements (`int8-mlp`, `int8-mlp-padded`) were mixed:
padded replay at 235 tokens reduced up/gate latency by 32.4% and down by 55.4%,
but at 620 tokens increased them by 13.2% and 8.9%. At 4,032 tokens both changes
were inconclusive. These synthetic dispatch results do not qualify model
quantization; neither integer precision is enabled in inference.
