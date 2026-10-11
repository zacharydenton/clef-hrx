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
reference tests; do not loosen their tolerances to qualify a candidate. The
202/231 floor applies to the BF16 default. For the explicitly optional INT4
speed/accuracy modes, gate kernel optimizations against the qualified baseline
of the same precision mode and report the difference from BF16 separately.

The latest qualified full runs measure **943.58 tok/s INT4** and **524.73 tok/s
BF16**, with every output unchanged. See [register packing and reproduction
instructions](evaluation.md#register-based-integer-packing). The 1,000 tok/s
target remains unmet.

## Criterion coverage

| Suite | Coverage | Timing boundary |
| --- | --- | --- |
| `workload` / `workload_gemm` | Nine projection families; 1, 65, 127, 128, 158, 235, 260, 620, 1,024, 2,048, 4,032 rows; repeated weights and a rotating working set of at least 64 MiB | Completed dispatch, excluding allocation, compilation, upload and readback |
| `workload` / `workload_delta` | Chunked and register-resident recurrence at every listed length; deterministic reset and allocation checks | Completed graph replay |
| `kernels` / `paired_delta_prefill/persistent` | Alternating-order chunked/register comparisons at 235, 620 and 1,024 tokens; raw paired timing totals | Completed graph replay, with both variants resident |
| `kernels` / `paired_delta_prefill/prepared` and `/candidate` | Prepared or external DeltaNet kernels at 158, 235, 620, 1,024 and 2,048 tokens; external candidates additionally cover 128 and 260, plus 4,032 without preparation; alternating order, maximum error and bitwise equality | Completed graph replay, including preparation when selected |
| `workload` / `workload_attention` | Causal 24-query/4-KV-head attention, 256 dimensions; 16 through 4,032 tokens | Completed graph, including GPU value transpose/padding and the selected attention kernel |
| `workload` / `paired_swiglu` | MLP width 17,408 at all eleven workload lengths; separate SiLU/multiply versus fused activation, alternating execution order | Completed graph; raw paired batches and 99% bootstrap speedup intervals |
| `workload` / `paired_packing` | INT4 packing at widths 5,120, 6,144 and 8,192; an external candidate also adds 17,408; exact logical codes and scales | Completed graph with alternating order and paired confidence intervals; default compares two-pass/one-pass LDS, external candidates use the pre-register production baseline |
| `workload` / `paired_integer_gemm` | Current packed GEMM versus `CLEF_BENCH_INTEGER_SOURCE`, six projection families, repeated and rotating weights, all workload lengths | Alternating-order completed graphs; prepacked identical operands; exact output bits checked before and after timing |
| `preprocessing` / `kernel_cache_key` (with `bench-internals`) | CPU cache-hit key generation and two map lookups; three real kernel source lengths and a fixed eight-entry configuration | Full-source keys versus content-verified cached SHA-256 fingerprints; no GPU execution |
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

To compare qualified INT4 binaries while background load changes, run the full
231-case suite with alternating per-case order:

```sh
python3 scripts/bench_jevbench_pair.py \
  tests/results/jevbench-baseline tests/results/jevbench-candidate \
  --output tests/results/jevbench-paired
```

Both input directories must be complete, strictly valid `eval_jevbench.py` runs
with identical inputs, pins and CLI settings. The harness verifies binary hashes
and checks every output against its qualified run. Two INT4 models remain
resident, but requests execute serially; both processes must also pass the loader's
conservative available-memory preflight. The first attempt on the shared host rejected the
second process (48.5 GiB required, 44.8 GiB available), so no paired full-model
timing has yet been qualified. The failed run cleaned up both processes. It records
raw predictions, per-case order/timing, engine allocations, telemetry and paired
case-resampling intervals. Each rate excludes only that model's first request.
The intervals describe variation across this case mix, not repeated full-suite
runs. This comparison has additional memory pressure and does not replace the
single-model 1,000 tok/s gate.

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
The paired `kernels` suite accepts the same DeltaNet overrides under
`paired_delta_prefill/candidate`. Its optional `CLEF_BENCH_DELTA_REFERENCE_SOURCE`
selects an archived eight-values-per-wave reference, allowing comparisons after
the production kernel changes. Both sources are copied into the run record.
`CLEF_BENCH_DELTA_PREPARE_SOURCE` adds a preparation dispatch inside the timed
DeltaNet graph. Its output is FP32 `[tokens, 4192]`: normalized/scaled Q (2,048),
normalized K (2,048), then interleaved decay/beta (96). The recurrent source
receives that buffer in its `a` binding; the other bindings keep their original
layout. It requires `CLEF_BENCH_DELTA_SOURCE` and preserves the 0.002 maximum
output-error check against the chunked reference for experimental sources too.
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

The fused SwiGLU activation retains the original intermediate BF16 rounding
between SiLU and multiplication. It removes one temporary tensor/pass and uses
256-thread workgroups. GPU tests cover row offsets, partial workgroups and two
replays against the original two-dispatch path. The `swiglu-paired-full` sweep
matched output bits before and after timing at every length and measured
1.31–1.84× operator speedups, with all 99% paired intervals above 1.0. These are
operator measurements. Full JevBench qualification preserves all logits and
probabilities in both modes: BF16 scores 202/231 at 411.94 tok/s, and broader
INT4 scores 194/231 at 814.21 tok/s. All outputs are strictly valid and token
counts are unchanged. See [full results and reproduction](evaluation.md#fused-swiglu).

```sh
python3 scripts/bench_workload.py swiglu-paired-repro --filter paired_swiglu
```

The earlier single-pass packing candidate keeps transformed FP32 values in local
memory until the scale is known, eliminating the repeated transform. Its selected
cutoff was 6,144 columns: quick paired measurements at 8,192 columns showed no
benefit at 4,032 rows. Wider operands retained the two-pass implementation.
The independent CPU packing oracle covers INT4 and INT8, zero/constant/random
rows, widths 256/768/1,280/5,120/6,144/8,192/17,408 and replay. Scratch-boundedness
and weight lifetime tests also pass.

`single-packing-paired-full` covers all eleven workload lengths at both selected
model widths (5,120 and 6,144): all 22 pairs have exact codes/scales and measured
1.30–1.95× speedups, with every 99% paired lower bound above 1.0. Full INT4
JevBench qualification scored 194/231 at 831.62 resident input tok/s, preserving
all 231 logit/probability arrays and the 14.508 GiB peak allocation. The previous
run measured 814.21 tok/s; these full runs are separate sequential measurements,
not paired timing evidence. See [results and reproduction](evaluation.md#single-pass-int4-packing).

```sh
python3 scripts/bench_workload.py packing-paired-repro \
  --filter 'paired_packing/pair/.*x(5120|6144)$'
```

The register-packing candidate retains FP32 transformed values in registers and
uses wave shuffles for the four radix-4 stages. Only the row maximum crosses
wave boundaries. It preserves the transform's pair sums/differences, quantization
rounding and row scales. `packing-register-generic-paired` covers all eleven row
lengths at four widths; all 44 pairs have identical logical codes and scale bits,
and every 99% speedup lower bound exceeds 1.0:

| Width | Reference | Paired speedup range | Candidate VGPRs / LDS |
| ---: | --- | ---: | ---: |
| 5,120 | One-pass LDS | 1.32–2.57× | 47 / 32 bytes |
| 6,144 | One-pass LDS | 1.46–2.69× | 48 / 32 bytes |
| 8,192 | Two-pass LDS | 1.97–3.69× | 55 / 32 bytes |
| 17,408 | Two-pass LDS | 2.14–4.30× | 96 / 32 bytes |

All four compiler reports have zero materialized spill bytes. External packing
candidates use the previous production LDS selection as a fixed reference,
so reproduction remains meaningful after the new implementation is selected:

```sh
CLEF_BENCH_PACKING_SOURCE=kernels/prepare_integer_register.loom \
  python3 scripts/bench_workload.py register-packing-repro --filter paired_packing
cargo test --locked --release --lib packing_ -- --ignored
```

The external source is copied and hashed with the other benchmark inputs. The
GPU checks compare against the independent CPU oracle and the original packing
kernel, including nonzero input slices, INT4/INT8, zero/constant/random rows,
NaN/Inf, overflow, subnormals and repeated execution. Operator speedups alone
do not establish the 1,000 tok/s model target.

Full INT4 qualification retained all 231 logit/probability arrays and scored
194/231 at **943.58 resident input tok/s**, up from 906.55 in the preceding
sequential run. Register packing is selected for the backbone widths 5,120,
6,144 and 17,408. BF16 does not use this path. See
[complete measurement conditions](evaluation.md#register-based-integer-packing).

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

For alternating-order integer comparisons, select `paired_integer_gemm` and set
`CLEF_BENCH_INTEGER_SOURCE`, `CLEF_BENCH_INTEGER_BITS` and
`CLEF_BENCH_INTEGER_PADDED`. It keeps the current and candidate kernels resident
with separate identical operands, checks exact output bits, and records raw
paired batches and 99% bootstrap intervals. Rotating cases allocate at least
64 MiB of weights per variant; combined engine allocation is capped at 1 GiB.
This group times prepacked operands only. Set
`CLEF_BENCH_INTEGER_ROW_SCALES=1` for an additional correctness sweep with distinct,
nonuniform FP32 input/weight row scales and zero-scale rows. Use that flag with
`paired_integer_gemm`; the regular raw-integer/BF16 oracle assumes uniform scales.
The paired JSON records which scale fixture was used. For example:

```sh
CLEF_BENCH_INTEGER_SOURCE=artifacts/throughput-1000/gemm-integer-epilogue-unroll.loom \
CLEF_BENCH_INTEGER_BITS=4 CLEF_BENCH_INTEGER_PADDED=1 \
  python3 scripts/bench_workload.py integer-paired-repro --profile quick \
  --filter 'paired_integer_gemm/mlp_'
```

`CLEF_BENCH_INTEGER_PREPARE=1` additionally rotates and packs BF16 activations
on the GPU inside each replay, using 256-element orthogonal Hadamard groups and
per-row scales. Weights are transformed once outside timing. This mode compares
outputs with the original BF16 inputs and reports relative L2 error; it does not
use the exactly representable input check from the raw integer probe. A separate
GPU test compares packed codes and scales with a CPU oracle. Model accuracy
still requires the full evaluation.

For checkpoint qualification only, a binary built with `--features bench-internals`
accepts `CLEF_EXPERIMENT_MLP_BITS=4|8`. Selected language-model MLP weights are
rotated and packed once during loading; activations are prepared inside each
inference graph. `CLEF_EXPERIMENT_MLP_PROJECTIONS=up,gate,down` chooses the
projections (all three by default). Dense selected weights are replaced, and packed
weights remain leased across graph resets. Ordinary builds expose `--precision int4-mlp` and `--precision int4`; the environment switches remain benchmark-only.
`CLEF_EXPERIMENT_BACKBONE_BITS=4|8` also selects other large text-backbone
projections (at least 4,096 outputs and 1,024 inputs, both divisible by 256).
An explicit MLP precision overrides the backbone precision for selected MLP
projections. The full INT4 backbone trial missed the BF16 accuracy floor below; it remains available as an explicit speed/accuracy tradeoff.
`CLEF_EXPERIMENT_SHARE_PACKING=1` reuses the most recent packed activation when
the next quantized projection reads the exact same tensor view with matching
precision and stride. The source lease prevents allocation reuse from producing
a false cache hit; completing or resetting a graph clears the cache. It targets
the shared inputs of MLP gate/up and DeltaNet QKV/Z projections. It is enabled
in the public INT4 modes after the full `jevbench-20261011-int4-shared-packing`
qualification: all 231 logits/probability arrays match the scratch-reuse run,
194/231 correct, all strictly valid, unchanged tokens. Resident throughput was
780.17 tok/s versus 775.79 in the preceding scratch-reuse run; that 0.6% difference
does not establish a reliable speed gain. Peak engine allocation was 14.738 GiB.
The binary SHA-256 is
`b4e0595172103401d1d08ecb17fafc64b6b3c76acc6cd0f0bcffe080b08aee5a`.
The evaluation recorder saves benchmark-only environment variables.

The profiling example accepts `--precision int4` alongside the ordinary precision
choices. Its labels distinguish integer GEMMs and packing exports from BF16
GEMMs. A one-round serialized diagnostic on the 4,032-token JevBench request
recorded approximately 404 ms in packing, 341 ms in SiLU/multiply, 579 ms in
DeltaNet, and 488 ms in attention, with integer GEMMs accounting for roughly
half the GPU work. These intervals include profiling barriers and are not the
end-to-end speed metric. The log is
`artifacts/throughput-1000/int4-shared-profile.log`; ordinary runs before and
after profiling are checked for exact output agreement.

With `CLEF_EXPERIMENT_BACKBONE_BITS=4`, all 368 eligible projections were packed.
The full `jevbench-20261010-int4-backbone` run reached **747.7 input tokens/s**
(222.54 s resident), but only **194/231** correct: 48 easy, 71 standard and 75
hard. All 231 requests were strictly valid with unchanged token counts. Against
the grouped-store BF16 baseline it gained one decision and lost nine; maximum
probability drift was 0.7170, Brier 0.2021 and ECE 0.0576. This configuration is
**not selected as the default**. The 1.83× observed speed gain remains below
the 1,000-token/s target. It is available through `--precision int4` for users
who accept the measured accuracy tradeoff. The archived binary is
`artifacts/throughput-1000/clef-int4-backbone`, SHA-256
`842cc995af321a1363de3df1029b6e40127316f23467e4ff416e15191e6c0fa5`.
This run includes grouped DeltaNet stores but does not enable packing reuse.
Its `comparison.json` records every changed decision. Reproduce using the
commands below with `CLEF_EXPERIMENT_BACKBONE_BITS=4` instead of the MLP flag,
and use `tests/results/jevbench-20261010-delta-stores` as the comparison baseline.

```sh
cargo build --release --locked --features bench-internals --bin clef
CLEF_EXPERIMENT_MLP_BITS=4 python3 scripts/eval_jevbench.py \
  --model-dir artifacts/model --output tests/results/jevbench-int4-mlp
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261010-persistent tests/results/jevbench-int4-mlp
```

The first complete INT4 MLP run (`jevbench-20261010-int4-mlp`) reached
**625.3 resident input tokens/s**, but scored **199/231**, below the 202/231
floor. All requests were strictly valid with identical token counts. It gained
two decisions and lost five versus the accepted attention baseline; maximum
probability drift was 0.5063, Brier 0.1653 and ECE 0.0499. This configuration is
**not selected as the default**, but is available as `--precision int4-mlp`.
The 256-element rotation and per-row max scaling trade three net correct
decisions for the observed speed gain.
The archived binary is `artifacts/throughput-1000/clef-int4-mlp`, SHA-256
`e8e1fa0a004659008d513e3fddb25c45fe63d468158004ae8eb4a974e51e8b79`;
the exact source diff and GPU telemetry are retained alongside it.

See [llama.cpp kernel findings](llama-kernel-notes.md) for the pinned source
observations behind the next GEMM and DeltaNet experiments.

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

An experimental single-BF16 probability component preserved finite, deterministic
outputs in the long-shape smoke checks but had mixed timing results. Against a
fresh current-kernel screen, it was 1.8% slower at 1,024 tokens, 6.5% slower at
2,048, and 8.3% faster at 4,032 (`attention-probability-bf16-screen`). Maximum
absolute difference against the original scalar attention fixture was 0.0078125.
These are separate Criterion runs, not paired timings. It was not selected and
has no full-model accuracy qualification.

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
kernel speedup. Repeating baseline source `0f79e8b` with the cache populated
produced 381.5 tokens/s and exactly reproduced the baseline probabilities;
the candidate is 5.3% faster against that run. See [measurement details](evaluation.md). An earlier candidate
using tiled attention from 64 tokens scored 201/231 at 351.4 tokens/s and was
rejected (`jevbench-20261010-attention`).

Grouping DeltaNet's eight output stores at the end of each token iteration then
improved paired kernel throughput by 1.15–1.32× at 158 through 2,048 tokens,
with bitwise-identical outputs. The complete JevBench run retained 202/231 and
identical probabilities, reaching **407.8 input tokens/s**, up 1.5% from 401.9.
This is the current production path; the end-to-end comparison is sequential
on a shared machine. The target of 1,000 tokens/s remains unmet. See
[the llama.cpp investigation](llama-kernel-notes.md) for the scheduling sweep,
paired intervals, rejected candidates, and reproduction commands.

Packed integer kernels remain benchmark-only. The h3-derived implementation is
checked in at `kernels/gemm_integer.loom`; it supports both INT4 and INT8
with BF16 outputs. Each precision passed all 12 padded MLP smoke cases at 235,
620 and 4,032 tokens, including repeated and rotating weights. Reproduce a
correctness sweep or measured experiment with
fresh run names:

```sh
CLEF_BENCH_INTEGER_SOURCE=kernels/gemm_integer.loom \
  CLEF_BENCH_INTEGER_BITS=4 CLEF_BENCH_INTEGER_PADDED=1 \
  python3 scripts/bench_workload.py int4-smoke --profile quick \
  --filter 'workload_gemm/mlp_(up_gate|down)' --test

# Use bits=8 to compare INT8; omit --test and supply a baseline for measurements.
CLEF_BENCH_INTEGER_SOURCE=kernels/gemm_integer.loom \
  CLEF_BENCH_INTEGER_BITS=8 CLEF_BENCH_INTEGER_PADDED=1 \
  python3 scripts/bench_workload.py int8-measured --profile quick \
  --filter 'workload_gemm/mlp_(up_gate|down)' --compare workload-baseline-v2
```

The earlier 12-case INT8 measurements (`int8-mlp`, `int8-mlp-padded`) were mixed:
padded replay at 235 tokens reduced up/gate latency by 32.4% and down by 55.4%,
but at 620 tokens increased them by 13.2% and 8.9%. At 4,032 tokens both changes
were inconclusive. These synthetic dispatch results do not qualify model
quantization; neither integer precision is enabled in inference.
