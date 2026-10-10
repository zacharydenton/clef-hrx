# llama.cpp kernel findings

Inspected the local `~/code/llama.cpp` checkout at
`23b0202a189c44a54625aadcb37a946dd1d6278d` on 10 October 2026. These are
observations from that source revision, not speed claims for CLEF. Paths below
are relative to that checkout; no llama.cpp files were changed.

| Finding | Source | Consequence for CLEF |
| --- | --- | --- |
| Dense BF16 prefill falls through to hipBLAS. The custom dense matrix kernel generally requires at most 16 input columns on RDNA3.5. | `ggml/src/ggml-cuda/mmf.cu:169`, `ggml/src/ggml-cuda/ggml-cuda.cu:1946`, `vendors/hip.h:40` | Measure hipBLAS on the exact 235/620/4032-token projection shapes as a vendor reference before assuming our BF16 kernels are near the attainable ceiling. |
| Q4_0 weights stay compressed in global memory, then expand to signed INT8 in LDS; activations use Q8_1 packing. | `ggml/src/ggml-cuda/mmq-load-tiles.cuh:187`, `ggml/src/ggml-cuda/mmq.cu:254`, `ggml/src/ggml-cuda/mma.cuh:1464` | W4A8 is an accuracy-oriented alternative to native W4A4. Four-bit storage does not imply four-bit matrix instructions or twice the arithmetic peak. Block scales add work compared with our per-row synthetic probe. |
| RDNA3.5 has its own MMQ table: 128/256 threads, target occupancy 2, output widths 64/128, input-column tiles from 16 to 128. | `ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`, config field definitions in `mmq.cuh:164` | Sweep tile shape and occupancy by workload shape. Our current packed probe uses a fixed 256-by-128 output tile, which can waste work on short inputs. |
| Its RDNA3.5 table disables Stream-K, despite a generic Stream-K implementation. | `ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`, `mmq.cuh:1430` | Do not assume Stream-K is the next universal optimization; compare simple tiling first on this GPU. |
| Quantized LDS strides deliberately satisfy `stride % 8 == 4` in 32-bit elements. | `ggml/src/ggml-cuda/mmq.cuh:153` | LDS bank padding is a separate variable from our global operand-stride padding experiment. Benchmark both independently. |
| DeltaNet splits each wave into four independent state-column reductions. For dimension 128, each 8-lane segment owns 16 state rows per lane. State stays in registers across tokens. | `ggml/src/ggml-cuda/gated_delta_net.cu:4` | Compare against our eight columns processed serially per wave. Segment reductions can reduce shuffle/reduction instruction overhead while changing reduction order; run the full accuracy gate if faster. |
| DeltaNet chooses 1/2/4 waves per block according to available parallelism. It still has a TODO for a chunked prefill kernel. | `ggml/src/ggml-cuda/gated_delta_net.cu:189` | Tune block size alongside column layout. This checkout does not supply a ready-made faster chunked prefill implementation. |
| Q/K normalization runs before the recurrent DeltaNet kernel. | `src/models/qwen35.cpp:443`, `build_gdn_l2_norm` | Compare one preparation pass against repeating normalization across all state-column waves. Include the preparation dispatch in timing. |
| Fusion checks shape, shared input identity, tensor lifetimes and overlapping ranges. Matmul/GLU dispatch is restricted to supported vector cases; RMSNorm/multiply/RoPE fusion requires supported F32 layouts and RoPE modes. | `ggml/src/ggml-cuda/ggml-cuda.cu:1703`, `:2786`, `:3130`, `:3976` | Fusion is worth testing, but these paths are not direct implementations for CLEF's BF16 multi-axis prefill. Preserve BF16 rounding points when isolating memory-traffic savings. |

The first two concrete probes are a standalone BF16 `hipblasGemmEx` comparison
and a four-column segmented DeltaNet source. Both are experiments; production
remains gated on the complete JevBench suite. The integer preparation benchmark
separately measures rotation, scale selection and packing on the GPU, with
weight preparation outside replay and activation preparation inside replay.

## Vendor GEMM screening

A standalone `hipblasGemmEx` probe used BF16 inputs/outputs, FP32 accumulation,
the same deterministic input values as the Criterion GEMM harness, ten warmups,
and twenty samples of ten completed calls. Allocation, upload and readback were
excluded; sampled outputs passed an independent FP64 dot-product check. Both
default selection and `ROCBLAS_USE_HIPBLASLT=1` were recorded. The latter requests
that backend; this probe does not prove which internal kernel was selected.

| MLP projection | Tokens | Recorded HRX replay | hipBLAS default | hipBLASLt requested |
| --- | ---: | ---: | ---: | ---: |
| Up/gate | 235 | 1.76 ms | 2.67 ms | 2.65 ms |
| Down | 235 | 2.23 ms | 2.66 ms | 2.62 ms |
| Up/gate | 620 | 2.74 ms | 3.33 ms | 3.32 ms |
| Down | 620 | 3.25 ms | 5.02 ms | 4.90 ms |
| Up/gate | 4,032 | 16.51 ms | 20.39 ms | 20.24 ms |
| Down | 4,032 | 22.93 ms | 29.41 ms | 29.09 ms |

This is diagnostic screening across different host dispatch paths, not a paired
Criterion comparison. Median active GPU clocks were 2.83/2.79 GHz for the two
vendor runs. Neither selected a promising replacement. Raw samples, source,
binary hash, environment and half-second telemetry are retained locally under
`artifacts/throughput-1000/hipblas-default/` and `hipblas-lt/`; HRX numbers come
from `benches/results/workload-baseline-v2/`.

## Segmented DeltaNet screening

The direct four-column-per-wave experiment passed finite-output and replay
checks, with maximum output drift of 0.00003052 against the chunked reference.
It was nevertheless slower than the existing eight-column persistent kernel:
0.672 ms at 235 tokens (+12.4%), 1.824 ms at 620 (+13.7%), and 18.188 ms at
4,032 (+32.1%). All three 99% confidence intervals excluded zero. Results are
in `benches/results/delta-segmented4/`; this candidate is not enabled.

Moving Q/K normalization and decay/beta calculation to one shared FP32 preparation
pass changes the result. The existing BF16 beta rounding point is preserved and
the preparation dispatch is inside the timed graph:

| Tokens | Existing persistent | Prepared, eight columns | Prepared, four segmented columns |
| ---: | ---: | ---: | ---: |
| 235 | 0.598 ms | 0.526 ms (−12.0%) | 0.371 ms (−38.0%) |
| 620 | 1.604 ms | 1.596 ms (−0.5%) | 1.291 ms (−19.5%) |
| 4,032 | 13.767 ms | 13.528 ms (−1.7%) | 17.250 ms (+25.3%) |

The eight-column candidate fails the 3% improvement threshold at the two longer
lengths. The segmented candidate is promising for shorter requests but regresses
the longest one. These are screening comparisons against a recorded baseline,
not alternating-order paired confirmation or full-model qualification. Results
are `benches/results/delta-prepared-v8/` and `delta-prepared-seg4/`; maximum output
drift against the chunked reference was 0.00003052 in both runs.

The segmented candidate and shared preparation source are checked into
`benches/kernels/`. Reproduce with a fresh name:

```sh
CLEF_BENCH_DELTA_PREPARE_SOURCE=benches/kernels/delta_prepare.loom \
CLEF_BENCH_DELTA_SOURCE=benches/kernels/delta_segmented_prepared.loom \
CLEF_BENCH_DELTA_VALUES=4 \
python3 scripts/bench_workload.py delta-prepared-repeat --profile quick \
  --filter workload_delta/persistent --compare workload-baseline-v2
```

Omit `--profile quick` for the eleven-length sweep. Use `--test` without
`--compare` for construction/correctness smoke checks. A production selector
would need the full sweep, alternating-order confirmation and the complete
JevBench accuracy/throughput gate before acceptance.
The eleven-length smoke sweep (1 through 4,032 tokens, including odd lengths)
passed output-error, finite-output, deterministic replay and allocation checks;
its record is `benches/results/delta-prepared-seg4-full-smoke/`.

Alternating-order confirmation is now in
`benches/results/delta-prepared-paired/` (40 measured batches, 10,000 paired
bootstrap resamples, 99% intervals). Preparation is included in candidate time:

| Tokens | Reference | Candidate | Paired speedup | 99% interval |
| ---: | ---: | ---: | ---: | ---: |
| 158 | 0.407 ms | 0.256 ms | 1.589× | 1.587–1.591× |
| 235 | 0.613 ms | 0.388 ms | 1.579× | 1.566–1.587× |
| 620 | 1.816 ms | 1.423 ms | 1.276× | 1.274–1.278× |
| 1,024 | 3.079 ms | 3.014 ms | 1.022× | 1.001–1.073× |
| 2,048 | 6.566 ms | 7.202 ms | 0.912× | 0.911–0.913× |

Only the first three meet the 3% improvement threshold across the interval.
Reproduce with `CLEF_BENCH_PAIRED_SECONDS=4 python3 scripts/bench_workload.py
delta-prepared-paired-repeat --suite kernels --filter paired_delta_prefill/prepared`.
`CLEF_EXPERIMENT_DELTA_PREPARED_MAX` enables a benchmark-feature-only full-model
trial for lengths 128 through that maximum. With the maximum set to 620, the
complete JevBench run passed at **202/231** with no changed decisions and
**405.2 resident input tokens/s**, compared with 401.9 for the accepted baseline.
All 231 requests remained strictly valid with identical token counts. Maximum
probability drift was 0.0810, Brier 0.1657 and ECE 0.0230. The observed 0.8%
end-to-end gain is a sequential shared-machine comparison, not a paired
confidence estimate. This remains an explicit experimental switch; the default
inference path is unchanged.

The complete record is `tests/results/jevbench-20261010-delta-prepared/`, including
`comparison.json`. The archived binary is
`artifacts/throughput-1000/clef-delta-prepared`, SHA-256
`fb4247f6e06cf8c0dd0cd9aa480796854ca6ace9e9b4c97860e98092c9a93e1e`.
The operator check also passed the unchanged 0.002 maximum-error bound with
distinct a/b gates, nonzero dt, and varied log-decay at seven lengths.

```sh
cargo build --release --locked --features bench-internals --bin clef
CLEF_EXPERIMENT_DELTA_PREPARED_MAX=620 python3 scripts/eval_jevbench.py \
  --model-dir artifacts/model --output tests/results/jevbench-delta-prepared-repeat
python3 scripts/compare_jevbench.py \
  tests/results/jevbench-20261010-attention-long \
  tests/results/jevbench-delta-prepared-repeat
```

## DeltaNet preparation and scheduling sweep

A follow-up sweep stored normalized Q/K as FP16, retaining FP32 recurrent state
and gates. Preparation is included in every timed replay. These are synthetic
Criterion means in milliseconds; full checkpoint accuracy has not been checked
for the FP16 candidates.

| Candidate | 235 tokens | 620 tokens | 4,032 tokens |
| --- | ---: | ---: | ---: |
| FP16, four values/wave, 64 threads | 0.354 | 1.105 | 14.678 |
| FP16, four values/wave, 128 threads | 0.347 | 1.007 | 14.535 |
| FP16, four values/wave, 256 threads | 0.348 | 0.963 | 14.858 |
| FP16, four values/wave, vector Q/K loads | 0.455 | 1.350 | 19.572 |
| FP16, eight values/wave | 0.519 | 1.451 | 12.702 |
| FP16, eight values/wave, 128 threads | 0.515 | 1.543 | 12.839 |
| FP16, sixteen values/wave | 0.792 | 2.112 | 17.174 |
| FP16, eight values/wave, grouped stores | 0.453 | 1.226 | 10.913 |
| FP16, sixteen values/wave, grouped stores | 0.507 | 1.381 | 13.000 |
| Original BF16 input, eight values/wave, grouped stores | 0.518 | 1.374 | 11.496 |

Grouping all leader-lane stores at the end of each token iteration improved
both precision variants. The BF16-input candidate changes only store placement;
it keeps the existing in-kernel normalization, FP32 state, and arithmetic.
Vectorizing Q/K loads reduced static load instructions but made runtime worse.
The compiler reported no spills and the same occupancy for the scalar and
vector segmented kernels, so instruction counts alone were not predictive.

Raw results and exact candidate sources are archived under
`benches/results/delta-f16-*/` and `benches/results/delta-v8-stores/`.
The paired suite also accepts `CLEF_BENCH_DELTA_SOURCE`, optional
`CLEF_BENCH_DELTA_PREPARE_SOURCE`, and `CLEF_BENCH_DELTA_VALUES`. Filter
`paired_delta_prefill/candidate` to compare an external candidate against the
resident production kernel in alternating order; its `paired.json` records
maximum error and bitwise equality as well as timing samples.

The original-input grouped-store candidate then passed paired comparisons with
bitwise-identical outputs at every measured length:

| Tokens | Reference | Candidate | Speedup, 99% paired interval |
| ---: | ---: | ---: | ---: |
| 158 | 0.405 ms | 0.344 ms | 1.176× [1.175, 1.177] |
| 235 | 0.601 ms | 0.501 ms | 1.200× [1.199, 1.201] |
| 620 | 1.728 ms | 1.503 ms | 1.150× [1.149, 1.151] |
| 1,024 | 3.128 ms | 2.373 ms | 1.318× [1.315, 1.321] |
| 2,048 | 6.705 ms | 5.333 ms | 1.257× [1.252, 1.262] |

The complete run is `benches/results/delta-v8-stores-paired/`. To compare with
the same reference after changing the production source, use
`CLEF_BENCH_DELTA_REFERENCE_SOURCE` (an eight-values-per-wave kernel):

```sh
git show 0aaf258:kernels/delta_persistent.loom > /tmp/clef-delta-reference.loom
CLEF_BENCH_DELTA_REFERENCE_SOURCE=/tmp/clef-delta-reference.loom \
CLEF_BENCH_DELTA_SOURCE=kernels/delta_persistent.loom \
CLEF_BENCH_DELTA_VALUES=8 CLEF_BENCH_PAIRED_SECONDS=4 \
python3 scripts/bench_workload.py delta-stores-paired-repeat --suite kernels \
  --filter paired_delta_prefill/candidate
```

The grouped-store kernel passed the complete checkpoint evaluation at
**202/231 and 407.8 resident input tokens/s**, with identical probabilities to
the 401.9-token/s attention baseline. All requests were strictly valid with
identical token counts. It is now the production kernel. The full record is
`tests/results/jevbench-20261010-delta-stores/`; the archived executable is
`artifacts/throughput-1000/clef-delta-stores`, SHA-256
`c530e47d460081cc18128ecc51e92c1c2df044ce773b480d63eeaf582e785a15`.
The observed 1.5% end-to-end gain is a sequential shared-machine comparison.
The full 11-length Criterion sweep passed (`delta-stores-full`, 1–4,032 tokens),
as did the five-case archived-reference smoke run
(`delta-stores-reference-smoke`), the seven-length operator comparison at its
unchanged tolerance, and all-target/all-feature Clippy with warnings denied.
The normal build and [evaluation reproduction commands](evaluation.md#reproduce)
exercise this path without experimental flags.

## Native INT4 follow-up

The h3-derived native INT4 probe measures a different arithmetic path from
llama.cpp's Q4_0-to-INT8 expansion. Including GPU activation rotation and packing
on every replay, with weights prepared once before timing, it reduced MLP kernel
time by 35–68% across replay and rotating-weight cases:

| MLP projection | Tokens | BF16 replay | INT4 GEMM only | INT4 with activation preparation |
| --- | ---: | ---: | ---: | ---: |
| Up/gate | 235 | 1.757 ms | 0.558 ms | 0.631 ms |
| Down | 235 | 2.231 ms | 0.514 ms | 0.708 ms |
| Up/gate | 620 | 2.740 ms | 1.608 ms | 1.789 ms |
| Down | 620 | 3.248 ms | 1.497 ms | 1.953 ms |
| Up/gate | 4,032 | 16.509 ms | 8.605 ms | 9.645 ms |
| Down | 4,032 | 22.929 ms | 8.436 ms | 11.125 ms |

Raw runs are `benches/results/int4-mlp-padded/` and `int4-mlp-prepared/`, compared
with `workload-baseline-v2`. These use deterministic synthetic inputs; prepared
INT4 relative L2 error was approximately 0.73–0.98%. This is not evidence of
checkpoint accuracy. Ordinary inference still uses BF16. The GPU
packing test independently checks codes/scales against a CPU oracle for both
INT4 and INT8 at widths 256, 5,120 and 17,408.

A subsequent complete checkpoint trial with INT4 MLP projections reached
625.3 tok/s but only 199/231 correct, and was rejected. See
[the optimization record](optimization.md) for the result and exact binary hash.

Reproduce the prepared run with a fresh result name (omit `--compare` if the
named baseline is unavailable):

```sh
CLEF_BENCH_INTEGER_SOURCE=kernels/gemm_integer.loom \
CLEF_BENCH_INTEGER_BITS=4 CLEF_BENCH_INTEGER_PADDED=1 \
CLEF_BENCH_INTEGER_PREPARE=1 \
python3 scripts/bench_workload.py int4-mlp-prepared-repeat --profile quick \
  --filter 'workload_gemm/mlp_(up_gate|down)' --compare workload-baseline-v2

cargo test --release --features bench-internals \
  benchmarking::quantized::tests -- --include-ignored --test-threads=1
```

## Priority after screening

1. Keep the qualified grouped DeltaNet stores. Compare prepared variants against
   this new baseline before further promotion: prepared FP32 DeltaNet passed
   accuracy but delivered only a 0.8% end-to-end gain against the previous
   baseline; FP16 preparation remains unqualified on checkpoint accuracy.
2. Improve calibrated quantization quality: the first real-checkpoint W4A4 trial
   missed the accuracy floor by three cases despite its throughput gain. W4A8
   is another candidate if activation quantization costs too much accuracy.
3. Sweep integer tile sizes and LDS padding using the RDNA3.5-specific choices
   above. Deprioritize hipBLAS replacement and a blanket Stream-K rewrite based
   on the evidence collected here.
