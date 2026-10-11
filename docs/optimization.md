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

The latest qualified full runs measure **1,032.14 tok/s INT4**, **846.83 tok/s
MLP-only INT4** and **615.78 tok/s BF16** (unchanged-binary repeat; the first
BF16 run with a cold kernel cache measured 565.23), with every output unchanged.
See [qualification and reproduction](evaluation.md#tiled-weight-layout).
The **1,000 tok/s target remains unmet for BF16**, and the
[arithmetic ceiling](#tiled-weight-layout) shows BF16 WMMA cannot reach it on
this GPU.

## Original MLP INT4: current bottleneck

The Flash-support regression run retained every original CLEF probability:
202/231 BF16, 204/231 MLP INT4 and 194/231 broader INT4. Resident input rates were
623.67, 850.75 and 1,004.39 tok/s respectively. These changes are within the 3%
benchmark noise threshold of the accepted runs. MLP INT4 still misses 1,000 tok/s;
no kernel optimization is claimed from this support work.

A fresh three-round serialized profile of actual JevBench requests gives:

| Input tokens | INT4 GEMM share | BF16 GEMM share | Total serialized GPU ms |
| --- | ---: | ---: | ---: |
| 235 | 38.2% | 44.1% | 276.85 |
| 620 | 41.0% | 36.6% | 662.35 |
| 2,729 | 39.9% | 33.9% | 2,876.55 |

Profiling inserts barriers; these are diagnostic shares, not end-to-end speed
claims. Outputs before and after profiling matched exactly. Reaching 1,000 tok/s
from 850.75 needs about a 15% reduction in complete request time. The next useful
candidates therefore need to improve the large matrix multiplies; small head or
normalization changes alone cannot close that gap. Existing rejected tiling
experiments remain documented below.

Raw log: `artifacts/flash-qualification/profile-mlp-int4.log`. The three requests
are cases 0, 137 and 121 from the frozen JevBench `requests.jsonl`, respectively.
Reproduce with:

```sh
cargo build --release --features bench-internals --example profile_inference
CLEF_MODEL_DIR=artifacts/model target/release/examples/profile_inference \
  --model clef --precision int4-mlp --max-length 4032 --rounds 3 \
  REQUEST_235.json REQUEST_620.json REQUEST_2729.json
```

## Flash vectorized DeltaNet candidates

The original CLEF vector kernels can share recurrence work across eight or sixteen
value channels per wave. Two benchmark-only adaptations use Flash's 32 value
heads, 16 key heads and 8,192-wide QKV input. Production Flash still selects the
scalar kernel; these measurements do not establish a full-model speedup.

Both candidates matched the scalar output bit-for-bit at all eleven tested
lengths. Before/after timing checks found no replay changes. Measurements alternate
reference/candidate order, use 40 samples and 10,000 paired bootstrap resamples,
and exclude warmup. Telemetry recorded at most one KFD client in each run.

| Tokens | Eight-channel speedup | 99% interval | Sixteen-channel speedup |
| ---: | ---: | ---: | ---: |
| 1 | 1.062× | 1.061–1.063× | 0.884× |
| 65 | 1.192× | 1.190–1.193× | 1.101× |
| 127 | 1.208× | 1.206–1.210× | 1.140× |
| 128 | 1.207× | 1.206–1.208× | 1.129× |
| 158 | 1.210× | 1.208–1.212× | 1.136× |
| 235 | 1.169× | 1.168–1.171× | 1.137× |
| 260 | 1.222× | 1.221–1.223× | 1.142× |
| 620 | 1.208× | 1.208–1.209× | 1.151× |
| 1,024 | 1.270× | 1.269–1.271× | 1.192× |
| 2,048 | 1.246× | 1.246–1.247× | 1.176× |
| 4,032 | 1.219× | 1.211–1.229× | 1.083× |

Artifacts: `benches/results/flash-delta-{vector,vector16}-{smoke,paired}`.
Each includes source snapshots, executable/compiler hashes and raw timing pairs.
Reproduce the eight-channel measurements with:

```sh
CLEF_BENCH_MODEL=clef-flash CLEF_BENCH_PAIRED_SECONDS=3 \
  CLEF_BENCH_DELTA_VALUES=8 \
  CLEF_BENCH_DELTA_SOURCE=benches/kernels/delta_flash_vector.loom \
  python3 scripts/bench_workload.py flash-vector-repro \
    --suite kernels --filter paired_flash_delta
```

For sixteen channels, set `CLEF_BENCH_DELTA_VALUES=16`, use
`benches/kernels/delta_flash_vector16.loom`, and choose a fresh result name.
The next gate is full-model accuracy and paired resident request timing. A
17–27% improvement in this recurrence alone is not a 17–27% model improvement.

## Criterion coverage

Flash has its own `flash` suite covering its projection, packing, attention and
32-head DeltaNet shapes. See [Flash benchmarks](clef-flash.md#benchmarks).
Full-model benchmarks accept `CLEF_BENCH_MODEL` and `CLEF_BENCH_PRECISION`; a
long-context request corpus also adds targets through 65,536 tokens.

| Suite | Coverage | Timing boundary |
| --- | --- | --- |
| `workload` / `workload_gemm` | Nine projection families; 1, 65, 127, 128, 158, 235, 260, 620, 1,024, 2,048, 4,032 rows; repeated weights and a rotating working set of at least 64 MiB | Completed dispatch, excluding allocation, compilation, upload and readback |
| `workload` / `workload_delta` | Chunked and register-resident recurrence at every listed length; deterministic reset and allocation checks | Completed graph replay |
| `kernels` / `paired_delta_prefill/persistent` | Alternating-order chunked/register comparisons at 235, 620 and 1,024 tokens; raw paired timing totals | Completed graph replay, with both variants resident |
| `kernels` / `paired_delta_prefill/prepared` and `/candidate` | Prepared or external DeltaNet kernels at 158, 235, 620, 1,024 and 2,048 tokens; external candidates additionally cover 128 and 260, plus 4,032 without preparation; alternating order, maximum error and bitwise equality | Completed graph replay, including preparation when selected |
| `workload` / `workload_attention` | Causal 24-query/4-KV-head attention, 256 dimensions; 16 through 4,032 tokens | Completed graph, including GPU value transpose/padding and the selected attention kernel |
| `workload` / `paired_attention` | Current native attention versus `CLEF_BENCH_ATTENTION_SOURCE`, causal 24-query/4-KV-head attention at selected lengths ≥1,024 | Alternating completed graphs including value packing; paired 99% intervals, numerical differences and exact per-variant replay checks |
| `workload` / `paired_swiglu` | MLP width 17,408 at all eleven workload lengths; separate SiLU/multiply versus fused activation, alternating execution order | Completed graph; raw paired batches and 99% bootstrap speedup intervals |
| `workload` / `paired_packing` | INT4 packing at widths 5,120, 6,144 and 8,192; an external candidate also adds 17,408; exact logical codes and scales | Completed graph with alternating order and paired confidence intervals; default compares two-pass/one-pass LDS, external candidates use the pre-register production baseline |
| `workload` / `paired_integer_gemm` | Original packed GEMM versus `CLEF_BENCH_INTEGER_SOURCE`, eight projection families including K/V and joint head, repeated and rotating weights, all workload lengths | Alternating-order completed graphs; prepacked identical operands; exact output bits checked before and after timing |
| `workload` / `paired_bf16_gemm` | BF16 GEMM versus `CLEF_BENCH_GEMM_SOURCE`, the same eight projection families at 128 or more rows, repeated and rotating weights | Alternating-order completed graphs; exact output bits before and after timing, combined allocation capped at 1 GiB |
| `workload` / `paired_convolution` | Scalar versus adjacent-row causal convolution and SiLU, replay and at least 64 MiB rotating inputs | Alternating completed graphs; exact bits and replay checked; separate identical inputs |
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

Hardware telemetry also records active KFD process IDs so shared-GPU runs can
be identified. Exclusive access is checked separately from output correctness.

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

`CLEF_BENCH_PACKING_SWIGLU=1`, together with an external packing source, compares
the existing SwiGLU plus register-packing graph with a fused graph. Both paths
include activation and packing in the timed region, and both BF16 rounding
steps remain explicit. `swiglu-register-packing-paired` preserved exact codes
and scale bits at all eleven row lengths. The seven lengths from 158 through
4,032 measured 1.32–1.59× speedups, with every 99% lower bound above 1.0.
One row regressed to 0.60× and 128 rows to 0.984×, so the native selection keeps
the separate path below 158 rows. Fusion uses the same register-packing template
with a distinct export and a second input buffer.

Full INT4 JevBench qualification preserved every logit/probability array and
scored 194/231 at **967.12 resident input tok/s**. Peak engine allocation fell
from 14.508 to 14.278 GiB. See [the complete result](evaluation.md#fused-swiglu-and-packing);
the 1,000 tok/s gate still fails.

```sh
CLEF_BENCH_PACKING_SWIGLU=1 \
  CLEF_BENCH_PACKING_SOURCE=kernels/prepare_integer_register.loom \
  python3 scripts/bench_workload.py swiglu-packing-repro \
  --filter 'paired_packing/pair/.*x17408'
```

`CLEF_BENCH_PACKING_ATTENTION_GATE=1` instead compares gate-column extraction,
sigmoid, multiplication and register packing with their fused implementation.
It is mutually exclusive with `CLEF_BENCH_PACKING_SWIGLU`; filter packing cases
to width 6,144. Q/gate input has twice that width, interleaved as adjacent
256-element Q and gate halves per head. Both BF16 rounding steps are preserved.
The harness retains reference intermediates across both resident graphs so
either replay order leaves the outputs intact.

`attention-gate-packing-paired-v2` preserved packed codes and scale bits at all
eleven lengths. Speedups from 128 through 4,032 were 1.97–3.68×, with every 99%
lower bound above 1.0. One row regressed to 0.968×; the candidate uses fusion
from 128 rows. GPU checks cover INT4/INT8, slices, exceptional values, replay,
and NaNs in every unused Q half to catch incorrect gate indexing.

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
The paired JSON records which scale fixture was used.
`CLEF_BENCH_INTEGER_REFERENCE_SOURCE` replaces the original integer GEMM reference
with another source file; the runner archives and hashes both sources.
For `paired_bf16_gemm`, `CLEF_BENCH_GEMM_REFERENCE=auto|wide|reference|pipelined`
chooses the reference dispatch (`auto` by default). Use `wide` to reproduce the
older K/V path after a production dispatch change. Both paired GEMM groups
retain raw timing batches and 99% bootstrap speedup intervals.
`CLEF_BENCH_LENGTHS=257,384,513,640,769,896` overrides the quick/full row sweep
in the workload suite with up to 64 comma-separated lengths in 1–4,032.
Values are sorted and deduplicated. For example:

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

## Compiled-kernel eviction experiment

Replacing the 512-entry compiled-kernel cache's full flush with least-recently
used eviction did not improve complete INT4 JevBench: **982.11 tok/s**, versus
**992.76 tok/s** for the preceding unchanged-binary warm control. All 231 output
arrays and token counts matched exactly (194 correct; all strictly valid).
Mean preparation was 29.57 versus 29.25 ms; GPU time was 704.34 versus 696.73 ms.
This sequential comparison does not isolate the GPU-time difference, but it
provides no evidence of the intended preparation gain. The change was rejected.

The candidate binary is `artifacts/throughput-1000/clef-kernel-lru`, SHA-256
`5c7ae7c854ddb0297f0d10bfc2dbeea0f094f9c808760dfb6b248d7dc43ac6c7`, with a hashed
106-file source snapshot and the rejected patch alongside it. Full INT4 records
are in `tests/results/jevbench-20261011-int4-kernel-lru`. The subsequent BF16 run
was deliberately stopped after rejecting the candidate; its `interrupted.json`
records the stop, and its partial output is not a qualification result.

## Allocation lookup experiment

A capacity/slot index replaced full-buffer scans while preserving the existing
best-fit rule, tie ordering, host-input isolation and immutable weight lifetime.
Full INT4 JevBench retained every output and every per-request engine allocation
byte count (194/231 correct, all strictly valid), but measured **991.36 tok/s**,
versus 992.76 for the warm control. Preparation fell from 29.25 to 27.98 ms;
GPU time rose from 696.73 to 699.19 ms. The full run did not establish an
end-to-end gain, so the index was not selected. CPU policy checks, GPU scratch
reuse and weight lifetime tests, and Clippy passed.

Records: `tests/results/jevbench-20261011-int4-allocation-index`; normal binary
`artifacts/throughput-1000/clef-allocation-index`, SHA-256
`b9077f06f0dddb05722450fca81fd1116faa6535059db09a0ea023a1d8f89f62`, with its
107-file source snapshot. The run processed 166,394 resident tokens in 167.845
seconds, with 0.271/2.558-second median/p95 latency and 14.078 GiB peak engine
allocation. The 1,000 tok/s gate failed.

## Single-component attention probability experiment

A single BF16 probability component removes the second probability/value WMMA
from tiled attention. The new `paired_attention` harness compares it with the
current native graph using alternating order, including value packing in both
variants. Each graph must replay its own output exactly; maximum numerical
change is recorded separately from timing and is not an accuracy qualification.
Use `CLEF_BENCH_LENGTHS` to include ragged lengths, since aligned cases alone
missed large regressions here.

`attention-single-probability-paired` measured speedups of 0.992/1.044/1.097/
1.004/1.115/1.023× at 1,024/2,048/2,560/3,072/3,584/4,032 tokens. The boundary
sweep (`attention-single-probability-boundaries`) exposed 0.916× at 2,049,
0.768× at 3,073 and 0.977× at 3,583, all with 99% upper bounds below 1.0.
Maximum output change was 0.0078125. GPU tests passed the unchanged 0.008 bound,
FP64 spot checks, tails and replay, but inconsistent performance rejected the
candidate before full JevBench evaluation. Native attention precision is unchanged.

The rejected native trial, including its GPU test, is archived as
`artifacts/throughput-1000/clef-attention-single-probability-rejected`, SHA-256
`b3cea3cc245022d819e6e2185819994b6c1ad502712bbea9fbe546ff4e137e87`, with a
108-file source snapshot. The standalone benchmark candidate remains
`artifacts/throughput-1000/attention-probability-bf16.loom`. Reproduce its
boundary timing separately from any full-model run:

```sh
CLEF_BENCH_ATTENTION_SOURCE=artifacts/throughput-1000/attention-probability-bf16.loom \
  CLEF_BENCH_LENGTHS=2047,2048,2049,2559,2561,3071,3073,3583,3585,4031 \
  python3 scripts/bench_workload.py attention-probability-boundary-repro \
  --filter paired_attention
```

## Fused projection residuals

`CLEF_BENCH_INTEGER_RESIDUAL=1` includes a separate BF16 residual addition in
the reference graph and fuses that addition into the candidate GEMM epilogue.
Use INT4, varied row scales and a matching reference tile. Set
`CLEF_BENCH_INTEGER_RESIDUAL_EXPORT=h3_gemm_i4_256_residual` for the native
optional export; omit it for external prototypes using the original export name
with a residual binding. Both the projection rounding and final addition
rounding remain explicit. Reference intermediates remain leased across all
rotating graphs, and output bits are checked before and after timing.

The external wide-K prototype passed six smoke cases and measured 1.034–1.050×
in the quick sweep. The expanded sweep at 158, 235, 1,024, 2,048 and 4,032 rows
measured 1.024–1.048× (`residual-wide-k-full`). The original attention-output
tile measured 1.028–1.164× over 128, 158, 235, 1,024, 2,048 and 4,032 rows
(`residual-original-paired`). All outputs matched bits and all 99% lower bounds
exceeded 1.0. Compiler reports retained 256/200 VGPRs and 54/30 KiB LDS,
respectively, without spills.

M128 boundary measurements covered 128, 257, 384, 513, 640, 769 and 896 rows for
both projection roles (`residual-m128-boundaries`). All 28 pairs matched bits;
two MLP-down comparisons were initially inconclusive. Retesting with the native
optional export at 257, 384, 620 and 896 rows resolved those cases: all 16 pairs
improved 1.009–1.119×, with every 99% lower bound above 1.0
(`residual-native-m128-paired`). The original observations are retained.
The M128 compiler reports used 218 VGPRs and 36 KiB LDS without spills.

The final native original-tile and wide-K export checks at 235, 1,024 and
4,032 rows also preserve exact outputs. Their six pairs each measured
1.071–1.163× and 1.025–1.116×, respectively, with every 99% lower bound above
1.0 (`residual-native-original-paired`, `residual-native-wide-k-paired`).

The GPU fixture checks both roles at 13 lengths, including selection boundaries
and 4,033-row fallback, independently sliced codes/scales/residuals, zeros,
subnormals, large values, infinities, NaNs and two replays. Non-NaN outputs must
match bits; NaNs must remain NaNs. Full JevBench qualification preserves every
output in BF16 (202/231), MLP-only INT4 (199/231) and broader INT4 (194/231).
Two clean broader INT4 runs measured 967.61–969.78 tok/s; a fresh control measured
958.48 tok/s. The fusion is retained, although this sequential comparison does
not establish a paired model-level confidence interval and remains below
1,000 tok/s. The contended first run is retained separately.
See [all full-run records and reproduction](evaluation.md#projection-residual-fusion).

For native BF16 tile comparisons, `CLEF_BENCH_GEMM_CANDIDATE` accepts `auto`,
`reference`, `wide`, `pipelined` or `tiled` instead of `CLEF_BENCH_GEMM_SOURCE`.
The paired harness also covers `delta_gate` (5,120 inputs, 48 outputs), including
short lengths, with `auto`, `reference` or `wide` candidates. The pipelined
kernel does not support this narrow shape. For example:

```sh
CLEF_BENCH_GEMM_CANDIDATE=wide CLEF_BENCH_GEMM_REFERENCE=reference \
  python3 scripts/bench_workload.py delta-gate-wide-repro --profile quick \
  --filter paired_bf16_gemm/delta_gate
```

The initial native wide-K DeltaNet gate sweep (`delta-gate-wide-paired`) matched
output bits at 235, 620 and 4,032 rows in both cache modes. Six paired speedups
were 1.023–1.044×, each with a 99% lower bound above 1.0. The expanded GPU GEMM
fixture also passed sliced 5,120-to-48 projections from 1 through 4,033 rows
with replay. This tile is not selected for narrow gates: boundary performance
and complete-model qualification remain pending. No model-throughput gain is
claimed from this probe.

## Tiled weight layout

Per-kernel profiles of the BF16 model (`profile_inference`, 235 and 2,729
tokens) showed that short requests are 95% projection time while streaming
weights at only 45–92 GB/s, and that the two N=5,120 projections (MLP down,
attention output) ran at 28 TFLOPS against 41 for the other families. The
pipelined kernel reads each weight K-step as 256 separate 128-byte row segments
at the row stride, so every workgroup touches 256 DRAM pages per step. Storing
the weight as `[N / 256][K / 64][256][64]` makes each K-step one contiguous
32 KiB stream. `probe_gemm` compared `kernels/gemm_bf16_tiled.loom` against the
row-major pipelined kernel (alternating order, median of 15 rounds, same
accumulation order so outputs are identical):

| Shape (M×K×N) | Pipelined | Tiled | Speedup | Tiled TFLOPS |
| --- | ---: | ---: | ---: | ---: |
| 235×5,120×17,408 | 1.937 ms | 1.142 ms | 1.70× | 36.7 |
| 235×17,408×5,120 | 2.396 ms | 1.127 ms | 2.13× | 37.2 |
| 620×5,120×17,408 | 2.993 ms | 2.730 ms | 1.10× | 40.5 |
| 2,729×5,120×17,408 | 12.064 ms | 11.784 ms | 1.02× | 41.3 |
| 2,729×17,408×5,120 | 14.800 ms | 11.244 ms | 1.32× | 43.3 |
| 2,729×6,144×5,120 | 5.421 ms | 3.981 ms | 1.36× | 43.1 |

The layout is the standard storage for the 64 layers' projection weights: the
pristine checkpoint shards now remain intact and first load writes a derived
tiled cache keyed by model, revision, layout and source hash. Previously converted
original CLEF snapshots remain readable. The loader uses tiled bytes directly. The residual-add epilogue (`gemm_bf16_tiled_residual.loom`)
rounds the projection to BF16 before adding the residual, so attention-output
and MLP-down projections match the former separate `addf` bit for bit.
INT4 modes restore logical rows on the CPU before packing, then permute the
packed codes on the GPU into 128-row by 32-word blocks (`kernels/retile_codes.loom`);
all three integer GEMM kernels read that block order, which replaces the earlier
stride-padding workaround for weights (activations keep their padded stride). Two rejected probes are recorded for contrast: a K-step-32 variant of
the pipelined kernel (30 KiB LDS, three workgroups per WGP) ran 2.1–2.6× slower
at every shape and 1.36× slower even when padded back to 55 KiB LDS, so higher
occupancy does not help this kernel family.

Arithmetic ceiling: the 166,394 resident input tokens perform about
8.1×10¹⁵ BF16 multiply-adds in the projections alone. Reaching 1,000 tok/s
would need 48.7 TFLOPS sustained across the whole run, including the
bandwidth-bound short requests, against a 59.4 TFLOPS theoretical peak at the
2.9 GHz boost clock and 43.5 TFLOPS measured for the best kernel. INT4 reaches
the target only because its 4-bit WMMA path delivers roughly twice the
arithmetic rate (84 TOPS measured). BF16 arithmetic therefore cannot reach
1,000 tok/s on gfx1151; the tiled layout removes the memory-side waste instead.

A mixed-precision probe tested whether integer arithmetic could reach the
target without falling below the 202/231 BF16 floor: MLP plus all 48 DeltaNet
layers' projections in INT4 with the 16 full-attention layers kept in BF16
(`tests/results/jevbench-20261011-int4-delta-tiled`, same binary source as the
tiled runs). It scored 198/231 at 995.33 tok/s, losing six decisions and
gaining two against the BF16 repeat, with a maximum probability change of
0.668. It fails both gates and was not kept; MLP-only INT4 (204/231, 846.83
tok/s) remains the fastest qualified mode above the BF16 floor.

## Causal convolution row reuse

`paired_convolution` compares the original four-tap causal convolution plus
SiLU with `CLEF_BENCH_CONV_SOURCE`. `CLEF_BENCH_CONV_ROWS` sets output rows per
thread (1, 2, 4 or 8). Both paths retain the original sequential FP32 FMA order,
BF16 convolution rounding, SiLU and final BF16 store. Each candidate thread
loads an overlapping input window and four weights once, then produces several
adjacent time rows. Missing history skips its FMA, which matters for nonfinite
weights. Ragged final rows are guarded separately.

The harness alternates execution order with 20 measured batches, records raw
paired times and 99% bootstrap intervals, and retains each variant's separate
inputs and outputs. Rotating cases use at least 64 MiB of input per variant;
shorter-than-128-row cases use replay only. Outputs must match the reference
bits before timing and the saved initial reference after timing. Inputs and
weights use nonzero row slices. Compilation and upload are excluded from this
operator measurement; full-model qualification includes request preparation.

The initial two/four/eight-row quick sweeps all preserved output bits. Two rows
measured 1.308–1.561×; four rows 1.744–3.132×; eight rows 1.654–3.051× over
235, 620 and 4,032 tokens in both cache modes. Eight rows were strongest for
long inputs. The compact eight-row implementation then passed 27 smoke cases
covering 1–9-row causal startup, 65, 127–129, 158, 235, 620, 1,024, 2,048 and
4,031–4,032 rows. Its expanded 20-pair timing sweep measured **1.522–3.135×**;
every 99% lower bound exceeded 1.0. Records under `benches/results`:
`conv-rows{2,4,8}-paired`, `conv-rows8-compact-smoke` and
`conv-rows8-compact-full`. All variants and source snapshots are retained.

The final native source's six quick pairs measured **2.188–3.055×** with exact
outputs and every 99% lower bound above 1.0 (`conv-rows8-native-paired`).
Compiler reports for the compact kernel used 23–29 VGPRs at the measured
128–4,032-row shapes versus 8 for the scalar reference, with no LDS or spills
in either. Reports: `artifacts/throughput-1000/conv-rows8-compact-reports`.

Native dispatch selects eight rows per thread only for BF16 10,240-channel
convolution at 128–4,032 rows. Other shapes retain the scalar path. The GPU
fixture compares the candidate and native dispatch to the scalar reference at
19 lengths through 4,033, using independent input/weight slices, missing causal
history, partial row tiles, zeros, subnormals, large values, infinities, NaNs and
two replays. Non-NaN results match bits; NaNs remain NaNs. Selected finite
outputs also pass an independent FP64 convolution/SiLU oracle. CPU tests and
Clippy passed. Full JevBench qualification retained all 231 output arrays in
each mode: 202 correct for BF16, 199 for MLP-only INT4, and 194 for broader INT4.
Broader INT4 measured 1,002.98 tok/s initially and 986.28 on an unchanged-binary
repeat. Both are retained; the repeat misses the goal. BF16/MLP-only measured
516.08/752.25 tok/s. See [full results and reproduction](evaluation.md#eight-row-causal-convolution).

```sh
CLEF_BENCH_CONV_SOURCE=kernels/conv_delta_rows8.loom CLEF_BENCH_CONV_ROWS=8 \
  CLEF_BENCH_LENGTHS=128,129,158,235,260,620,1024,2048,4031,4032 \
  python3 scripts/bench_workload.py convolution-rows8-repro \
  --filter paired_convolution
cargo test --locked --release --lib convolution_row_reuse_preserves_outputs -- --ignored
```

`CLEF_BENCH_INTEGER_M_GROUP` lets `paired_integer_gemm` change only the candidate's
maximum row-tile group (1–16); the reference remains at four. The effective group
is capped by `ceil(rows / 256)`, matching the native rule. Use a source whose
`m_group` configuration accepts the chosen limit and a matching reference tile.
This is a benchmark-only probe of weight reuse; native dispatch remains unchanged
until paired operator and full-model results qualify a different ordering.

The initial row-group sweeps retained exact outputs in all 48 pairs across MLP
up/gate and down at 1,025, 2,048 and 4,032 rows, in replay and rotating modes.
Increasing the group cap to eight or sixteen improved the 1,025-row boundary
but regressed both longer shapes, reaching 0.892× for sixteen-row-group
MLP up/gate at 4,032. Caps of one or two were mostly unchanged on aligned long
shapes, with boundary gains up to 1.089×. These sweeps do not justify changing
the general four-tile rule. Records: `int4-group{1,2,8,16}-mlp-{up-gate,down}`
under `benches/results`; candidate sources are archived with each run.

A separate prototype keeps four-tile raster groups for the interior but removes
padding from the last group. Gfx11 lowering rejected a dynamic divisor; the
working version selects constant division/remainder cases for group sizes one
through four. The two compiler-failed smoke attempts are retained separately;
`unpadded-groups-smoke-v3` passed six exact cases. Its MLP-down report retained
256 VGPRs, 54 KiB LDS and zero spills, matching the original wide-K kernel.

Quick paired probes (`unpadded-groups-mlp-{up-gate,down}`) found boundary gains
at 1,025 rows but little change at aligned controls. The expanded sweeps cover
both endpoints of each partial-group band at 1,025–3,840 rows, in both cache
modes (`unpadded-groups-mlp-{up-gate,down}-boundaries`). All 72 pairs matched
output bits. Up/gate ranged 0.994–1.022× and is not selected. MLP down improved
**1.017–1.097× in all 36 pairs**, with every 99% lower bound above 1.0.
The rejected native experiment is described below.

## Partial MLP-down raster groups

**Rejected for native dispatch:** paired kernel gains did not establish a
full-model improvement. The candidate source and results remain in the
experiment archives; the production kernel and dispatch are restored.

The experimental wide-K INT4 kernel exposes separate compact-grid entry points,
with and without fused residual addition. A compile-time flag selects the final
partial group's mapping; full four-tile groups keep the original ordering.
The candidate selected this mapping only for 17,408-to-5,120 MLP-down
projections at 1,025–4,032 rows when `ceil(rows / 256) % 4 != 0`.
Complete groups, MLP up/gate and other shapes retained their prior dispatch.
This changes no dot product, scale or BF16 rounding operation.

`CLEF_BENCH_INTEGER_EXPORT` selects only the plain integer candidate export;
the reference retains its original export and the chosen reference source.
The native six-pair check at 1,025, 2,305 and 3,585 rows measured
**1.020–1.090×**, with all 99% lower bounds above 1.0 and bit-identical outputs
(`benches/results/compact-raster-native-paired`). These final pairs use uniform
scales; the expanded GPU fixture independently varies activation and weight
scales, includes zero and extreme scales, nonfinite residuals, sliced buffers,
all partial-group boundaries and fallback shapes, and checks two replays
against the original padded K128 projection plus separate residual addition.
CPU tests and Clippy passed.

```sh
CLEF_BENCH_INTEGER_SOURCE=artifacts/throughput-1000/compact-raster-source/kernels/gemm_integer_wide_k.loom \
  CLEF_BENCH_INTEGER_REFERENCE_SOURCE=artifacts/throughput-1000/compact-raster-source/kernels/gemm_integer_wide_k.loom \
  CLEF_BENCH_INTEGER_EXPORT=h3_gemm_i4_256_compact \
  CLEF_BENCH_INTEGER_BITS=4 CLEF_BENCH_INTEGER_PADDED=1 \
  CLEF_BENCH_LENGTHS=1025,2305,3585 \
  python3 scripts/bench_workload.py compact-raster-repro --profile quick \
  --filter paired_integer_gemm/mlp_down
cargo test --locked --release --lib residual_projection_matches_separate_addition -- --ignored
```

Add `CLEF_BENCH_INTEGER_ROW_SCALES=1` for independently varied operand scales.
Use fresh run names; the harness retains raw timing pairs, source snapshots,
binary/compiler identities and GPU telemetry.

For a comparison of two fused residual entry points, set
`CLEF_BENCH_INTEGER_RESIDUAL=1`, choose the candidate with
`CLEF_BENCH_INTEGER_RESIDUAL_EXPORT`, and choose the fused reference with
`CLEF_BENCH_INTEGER_REFERENCE_RESIDUAL_EXPORT`. Without the latter, the
reference remains the original projection plus separate addition. Both paths
receive identical independent residual tensors; paired records identify the
reference export so the timed operations are explicit.

The matched fused-residual sweep uses actual JevBench lengths 1,088, 1,290,
2,163, 2,390, 3,338 and 3,683, with independently varied row scales and both
cache modes. All **12 pairs improved by 1.007–1.078×**, with every 99% lower
bound above 1.0 and exact outputs (`compact-raster-fused-native-paired`). Both
compiled variants retain 256 VGPRs, 55,296 bytes LDS and zero spills; reports
are in `artifacts/throughput-1000/compact-raster-fused-reports`.

```sh
CLEF_BENCH_INTEGER_SOURCE=artifacts/throughput-1000/compact-raster-source/kernels/gemm_integer_wide_k.loom \
  CLEF_BENCH_INTEGER_REFERENCE_SOURCE=artifacts/throughput-1000/conv-rows8-source/kernels/gemm_integer_wide_k.loom \
  CLEF_BENCH_INTEGER_RESIDUAL=1 \
  CLEF_BENCH_INTEGER_RESIDUAL_EXPORT=h3_gemm_i4_256_residual_compact \
  CLEF_BENCH_INTEGER_REFERENCE_RESIDUAL_EXPORT=h3_gemm_i4_256_residual \
  CLEF_BENCH_INTEGER_BITS=4 CLEF_BENCH_INTEGER_PADDED=1 \
  CLEF_BENCH_INTEGER_ROW_SCALES=1 CLEF_BENCH_LENGTHS=1088,1290,2163,2390,3338,3683 \
  python3 scripts/bench_workload.py compact-raster-fused-repro --profile quick \
  --filter paired_integer_gemm/mlp_down
```

Full JevBench retained all outputs and accuracy floors in every mode, but the
initial 1,018.33 tok/s INT4 result fell to 985.73 on repeat. A fresh sequential
control/candidate pair measured 1,007.05/976.79 tok/s. Device clocks and power
varied between runs, so these are not a paired model-level confidence interval
or proof that the kernel itself causes a regression. They do not qualify a
model-level gain. The compact production entry points and dispatch were removed;
the benchmark export controls and expanded independent GPU boundary fixture
are retained. See [all full-run results](evaluation.md#partial-mlp-down-raster-groups).


## RMSNorm and integer packing fusion

The first candidate combined the named post-attention RMSNorm with Hadamard rotation
and activation packing for the MLP gate/up projections. Each wave preserves the
original 32-lane sequential FMA sums and subgroup reduction. Normalization uses
weight offset 1 and epsilon 1e-6, rounds to BF16, then applies the existing
Hadamard arithmetic and quantization. The two projections share the packed
codes and scales. Native selection covers INT4 5,120-to-17,408 gate/up pairs
at 128–4,032 rows; BF16 and other shapes retain their prior path.

The first 256-thread designs either shared one wave's RMS statistic or repeated
it independently in every wave. Both lost on longer inputs. Reducing the wave
count changed that result. Three-shape paired speedups versus separate native
normalization and register packing, with exact outputs in every timed case:

| Threads / statistic | 235 rows | 620 rows | 4,032 rows |
| --- | ---: | ---: | ---: |
| 256 / shared | 1.109× | 0.848× | 0.730× |
| 256 / repeated | 1.126× | 0.857× | 0.782× |
| 32 / one wave | 1.281× | 1.442× | 1.165× |
| 64 / shared | 1.627× | 1.290× | 1.461× |
| 64 / repeated | 1.654× | 1.308× | 1.440× |
| 128 / shared | 1.582× | 1.243× | 1.091× |
| 128 / repeated | 1.604× | 1.190× | 1.156× |

The first full-model candidate used a 64-thread variant that repeats the exact RMS statistic in two waves,
avoiding an extra cross-wave synchronization. Its initial report used 108 VGPRs,
8 bytes LDS and zero spills; the shared 64-thread variant used the same VGPRs
but 16 bytes LDS and more SGPRs. The single-wave variant used 188 VGPRs.
Reports and generated sources are under `artifacts/throughput-1000/norm-pack-*`.
The failed 256-thread and other experimental routes are retained only as local
experiment sources; the exact two-wave path is archived in `norm-packing-source`.

`paired_norm_packing` checks separate identical sliced inputs for each variant.
Rotating cases retain at least 64 MiB of input per variant. Each variant is
checked against its own saved initial output before and after timing, and the
candidate is separately compared with the reference. Padding words are excluded
because no projection reads them. The
reference includes the BF16 normalization intermediate, which stays leased
across every replay. Timing alternates variants and records raw pairs and
99% bootstrap intervals. Upload and compilation are excluded here and included
in full-model qualification where applicable.

The expanded 23-pair sweep (`norm-pack-redundant64-rotating`) covers 1, 65,
127–129, 158, 235, 260, 620, 1,024, 2,048 and 4,031–4,032 rows, with rotating
cases from 128 rows onward. It measured **1.034–1.641×**, with every 99% lower
bound above 1.0. The final native source passed eight pairs at 128, 235, 620 and
4,032 rows in both cache modes at **1.194–1.635×**, again with every lower bound
above 1.0 (`norm-pack-native-paired`).

The normalized-packing GPU fixture covers 14 lengths through 4,033, INT4 and
INT8, finite and exceptional weights, independently sliced inputs/weights,
signed zero, subnormals, small values dominated by epsilon, large values,
infinities and NaNs, with two replays. For this exact two-wave candidate, all
logical codes and scale bits matched the separate native normalization/packing
reference. The existing register,
SwiGLU and attention-gate packing fixtures also pass after extending the shared
template. CPU tests and Clippy passed.

The exact candidate reached 1,023.87 tok/s initially but 990.15 on repeat,
with unchanged full-suite outputs. It did not meet the repeated 1,000 tok/s
threshold. A separate profile confirmed 64 fused MLP dispatches; the next
experiment reduced the RMS dependency chain and fused input normalization too.

### Parallel RMS reduction and retained BF16 output

The qualified implementation distributes the 5,120 squares across 128 threads instead
of repeating the original 32-lane sum. Each lane accumulates 40 FP32 FMAs,
followed by a workgroup sum. This changes summation order. Weight offset 1,
epsilon 1e-6, BF16 rounding before the Hadamard transform, and the transform's
arithmetic remain unchanged. The four-wave packing loop retains 40 FP32 values
per thread. Separate 64-, 128- and 256-thread probes led to choosing 128 as a
balance across representative shapes and both cache modes. Final compiler reports
show 70 VGPRs without the BF16 store and 67 with it; both use 32 bytes LDS and
zero spills (`artifacts/throughput-1000/norm-pack-parallel-native-reports`).

An optional BF16 output shares the codes/scales allocation, aligned to 128 bytes.
It supplies small BF16 attention/DeltaNet projections while the larger INT4
projections reuse the prepared codes. Both input and post-attention fusion are
restricted to BF16 width 5,120, compatible INT4 weights and 128–4,032 rows.
BF16 mode and longer inputs keep their prior path, including the 64k default.

The benchmark's exact comparison is still the default. Explicit
`CLEF_BENCH_NORM_PACK_NUMERIC=1` permits at most one integer code level with at
most 1% of codes changed, scale relative error at most 0.004, and at most one
BF16 ULP with at most 1% of retained normalized values changed. These bounds
were established before the parallel candidate's measurements. They do not
replace the full-model accuracy floors. Reference and candidate outputs must
each remain bitwise stable across every timed replay.

Final native sweeps cover 128, 129, 158, 235, 260, 620, 1,024, 2,048, 4,031 and
4,032 rows with replay and independently rotating inputs: **40 paired cases**.
Every 99% lower confidence bound exceeds 1.0:

| Output | Paired speedup range | Lowest 99% lower bound | Result directory under `benches/results/` |
| --- | ---: | ---: | --- |
| Codes and scales | 1.363–3.423× | 1.361× | `norm-pack-parallel-native` |
| Codes, scales and BF16 values | 1.308–2.486× | 1.307× | `norm-pack-parallel-store-native` |

The largest changed-code fraction was 0.000000776 (less than one per million),
with maximum code change 1 and scale relative error 0.000154088. Retained BF16
values differed by at most one ULP, with at most 0.000006104 changed (6.11 per
million). All outputs were stable on replay. Raw pairs, numerical differences,
source snapshots and compiler reports are retained with the benchmark records.

The GPU fixture checks both output modes against separate native normalization
and packing over 14 lengths, both integer precisions, sliced inputs/weights,
exceptional values and two exact candidate replays. A separate FP64 RMS oracle
checks representative finite rows within one BF16 ULP. Store and no-store modes
must produce exactly identical packed codes and scales. All four packing GPU
fixtures, the CPU suite and Clippy pass.

```sh
CLEF_BENCH_PACKING_NORM=1 CLEF_BENCH_NORM_PACK_NUMERIC=1 \
  CLEF_BENCH_NORM_PACK_STORE=0 \
  CLEF_BENCH_PACKING_SOURCE=kernels/prepare_integer_register.loom \
  CLEF_BENCH_LENGTHS=128,129,158,235,260,620,1024,2048,4031,4032 \
  python3 scripts/bench_workload.py norm-parallel-repro --profile full \
  --filter paired_norm_packing
# Repeat with CLEF_BENCH_NORM_PACK_STORE=1 and a fresh result name.
cargo test --locked --release --lib packing_matches_reference -- --ignored
```

`CLEF_BENCH_PACKING_NORM=1` also enables the simpler `paired_packing` combined
reference for focused exact probes; filter that group to width 5,120.
`CLEF_TEST_NORM_PACK_SOURCE` can point the GPU fixture at an experimental source;
without it the fixture tests the native kernel. See
[full-model qualification and reproduction](evaluation.md#rmsnorm-and-packing-fusion).
