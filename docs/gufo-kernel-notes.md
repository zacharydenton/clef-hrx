# Gufo kernel findings

Inspected `~/code/gufo` at `d0f7c3a023a11af81161264eeb24e9a97aa7498a`
on 11 October 2026. The checkout was clean and was not modified. Source paths
below are relative to that checkout. These findings are hypotheses for CLEF,
not measured CLEF improvements.

The useful local guidance is Gufo's `.agents/skills/optimize-kernel/SKILL.md`
and `~/code/skills/skills/magpie-kernel-evaluator/SKILL.md`: preserve the binary
and workload, validate independent operator results, compare complete requests,
and separate profiling from headline timing. Our existing Criterion/HRX harness
provides the native Loom measurements; no Magpie workload was executed.
Hyperloom's Instinct/Python serving setup and TraceLens's PyTorch trace pipeline
do not match this runtime. No system memory settings were changed.

| Finding | Source | CLEF experiment |
| --- | --- | --- |
| W8A8 GEMM blocks both dimensions at 128, reducing repeated activation-panel loads; comments report 58% of a measured 55.07 TOPS INT8 ceiling. | `src/models/qwen/hip/kernels/prefill_quant_gemm.hip:399` | Compare the pending M128/N128 h3-derived INT4 tile against M256/N128, using rotating weights beyond the 32 MiB cache. Keep packing and arithmetic fixed. |
| Aggressive K-block unrolling caused Clang 23 register spills. Restricting unroll and adding a scheduling fence recovered substantial prefill speed. | same file, K-block loop near line 561 and fence at 613; `docs/models/qwen3.8-27b/EXPERIMENTS.md` | Inspect compiled Loom VGPR/scratch use before increasing unroll or prefetch. A Clang-specific fix is not evidence that Loom has the same problem. |
| The up-projection epilogue performs SwiGLU and emits the next GEMM's Q8 activation layout directly. One wave owns each 32-element scale group. | same file, `opt-c192-swiglu-epilogue` near line 272 | Fuse compatible intermediates only after preserving CLEF's BF16 rounding points. Our 256-element rotation and whole-row scale need a different reduction design; copying this epilogue directly would change quantization. |
| Residual/RMSNorm and activation packing can be fused; register and contraction changes still need full-model validation. | same file, `opt-c173-norm-quant` near line 39; Gufo optimization skill | Measure packing and normalization separately, then evaluate a fused pass against the original output bytes and independent packing oracle. |
| Persistent DeltaNet keeps state in registers and uses wave reductions; its standalone harness includes alternate recurrence layouts and an explicit arithmetic model. | `src/models/qwen/hip/kernels/ssm_recurrence.hip:63`; `tools/qwen27b/prefill_deltanet_bench.hip` | Compare the recurrence's actual instruction/reduction cost with our grouped-store kernel. Keep the existing BF16/FP32 boundaries and full JevBench gate. |
| Allocation type, cache residency and complete graph execution changed which microbenchmark wins survived. | `.agents/skills/optimize-kernel/SKILL.md`, bandwidth, memory and graphs sections | Retain rotating-weight controls and resident end-to-end timing; test graph replay and leases before accepting fusion. |

Gufo's published Q4 Qwen27B prefill is 656.33 tok/s at depth zero for an
approximately 2,048-token workload; its Q8 figure is 485.26 tok/s. Those use
GGUF K-quant formats and a generation runtime, not CLEF's decision head or our
W4A4 algorithm. They are useful context, not an apples-to-apples speed ranking.

The M128/N128 candidate passed all 36 shape and replay smoke checks. Its first
12-case Criterion comparison (`benches/results/int4-m128-mlp`, against the
previously recorded `int4-mlp-padded`) was mixed:

| Input rows | MLP up/gate time change, replay / rotating | MLP down time change, replay / rotating |
| --- | --- | --- |
| 235 | +5.67% / +4.23% | +6.09% / +5.18% |
| 620 | −11.92% / −12.16% | −4.68% / −4.59% |
| 4,032 | +2.83% / +2.33% | +1.14% / +1.48% |

Negative means faster. The harness classified the 235-row results as regressions,
620-row results as improvements, and 4,032-row results as inconclusive. These
are comparisons with an earlier run, not an interleaved paired experiment or
full-model speed claims. The default tile remains unchanged.

The scratch-reuse change releases packed activations after their last recorded
read, while preserving leases on immutable packed weights. Previously both were
retained for the whole graph. A 32-projection identity test verifies bounded
allocation and exact outputs on two replays; the independent packing oracle and
weight-lifetime tests also pass. The full 231-case JevBench run preserves every
logit and probability of the previous INT4 backbone run, scoring 194/231 at
775.8 resident input tok/s. Peak engine allocation falls from 38.43 to 14.73 GiB.
See [qualification and reproduction](evaluation.md#int4-scratch-reuse).
INT4 remains an explicit speed/accuracy option; BF16 remains the default.

Removing forced unrolling from the INT4 GEMM reduced compiler-reported vector
register use from 200 to 186 and raised the register residency tier from seven
to eight. Neither version spilled. The four long-input Criterion cases
(`int4-gemm-unroll-long` / `int4-gemm-rolled-long`) showed unchanged up/gate
timings and a 3.6% slowdown for replayed down-projection; the rotating down case
was inconclusive. The rolled loop was not selected. Register counts alone did
not predict a speed improvement.

The M128/N128 tile with a smaller K block (128 INT4 elements) passed all 36
projection smoke cases. It used 152 vector registers and 20 KiB LDS, versus
200 registers and 30 KiB for the production M256 tile. Its four 4,032-row MLP
cases were still 4.6–7.9% slower (`int4-m128-k128-long`); it was not selected.

The prepared eight-value DeltaNet experiment retains current grouped stores and
computes normalized Q/K and gates once per token in FP32. The five paired shapes
in `delta-current-prepared-paired` (158/235/620/1,024/2,048 tokens) matched output
bits and measured 1.12/1.16/1.19/1.04/1.04× speedups, with every 99% lower bound
above 1.0. Preparation is included in each timed graph. The additional 128/260
cases in `delta-prepared-v8-boundary` matched bits and measured 1.12× at both
lengths, again with 99% lower bounds above 1.0. Full JevBench preserved all outputs
but measured 832.16 tok/s in INT4 versus 831.62, and 406.22 in BF16 versus 411.94.
The extra preparation offset the INT4 GPU saving. It remains benchmark-only;
see [complete results and reproduction](evaluation.md#prepared-deltanet-experiment).

A second candidate applied Gufo's output identity
`dot(S_decay, Q) + delta * dot(K, Q)` while retaining the same state update.
`delta-prepared-algebra-paired` measured 1.07/1.08/1.03/0.98/0.97× at the same
lengths, with small BF16 output differences (maximum absolute change 0.00001526).
The two long shapes regressed against the current kernel, so this rearrangement
was rejected.

Unrolling the 16-fragment integer GEMM output loop removes its runtime accumulator
selection chain. All 36 paired smoke cases preserved output bits, but the twelve
MLP timing cases in `int4-epilogue-unroll-paired-mlp` ranged from 0.982× to 1.014×.
The rotating 4,032-row down projection regressed. Matched compiler reports kept
200 VGPRs and 30 KiB LDS with no spills; code grew from about 5.5 to 19.9 KiB.
The change was not selected. The new paired harness retains both variants,
alternates execution order and measures repeated and rotating weight sets.

Halving the production M256/N128 integer K block to 64 INT4 elements reduced
staging from 30 to 18 KiB and VGPR use from 200 to 184, raising the reported
register residency tier from seven to eight without spills. All 36 paired smoke
cases matched bits, but all twelve paired MLP cases slowed down: speedups ranged
from 0.641× to 0.993× (`int4-m256-k64-paired-mlp`). The extra staging iterations
did not pay off; the production K block remains unchanged.

Two 512-thread integer GEMM tiles also passed all 36 paired smoke cases. The
M256/N256 tile used 190 VGPRs and 40 KiB LDS; its twelve paired MLP speedups ranged
from 0.737× to 1.037× (`int4-n256-paired-mlp`). Only the longest down projection
improved. The M512/N128 tile used 194 VGPRs and 50 KiB LDS and slowed every MLP
case, with speedups from 0.482× to 0.966× (`int4-m512-paired-mlp`). Neither tile
was selected. Both retained the per-wave accumulator count while increasing the
workgroup from eight to sixteen waves; more operand reuse did not give a general
speedup.

Vectorizing the eight independent DeltaNet value channels keeps each channel's
FP32 recurrence order while allowing the compiler to schedule reductions together.
`delta-vector8-paired` and `delta-vector8-long-paired` cover eight lengths from
128 through 4,032 tokens. Outputs matched bits throughout, with 1.16–1.43× paired
operator speedups and every 99% lower bound above 1.0. The GPU fixture additionally
checks nonzero input slices, distinct gates, varying decay/bias, selection
boundaries and replay. Full BF16 and INT4 qualification preserved every output;
a fresh INT4 run after a host restart measured 883.33 tok/s. The selected range
is 128–4,032 tokens. See [measurement conditions and reproduction](evaluation.md#vectorized-deltanet-qualification).

Gufo's `src/models/minimax_h3/dense.hip` also suggested swapping WMMA operands
and packing the transposed accumulator directly into BF16 output. The prototype
preserved every output bit in 36 uniform-scale and 36 varied-scale cases, including
zero scales. Paired MLP timings with a rolled epilogue ranged from 0.995× to
1.010× (`int4-direct-store-paired-mlp`). Unrolling the direct epilogue passed the
36 varied-scale cases but produced 0.980–1.017× speedups, including short down
projection regressions (`int4-direct-store-unroll-paired-mlp`). Neither version
was selected. The source inspection used the same pinned Gufo revision above.

A second Gufo-inspired prototype loaded WMMA operands directly from global
memory and refilled registers after their last use, retaining our row-major
packed layout. It passed all 36 varied-scale correctness cases
(`int4-global-scales-smoke`). It reduced the compiler's register count from 200
to 168 VGPRs and LDS from 30 KiB to zero, without spills. Nevertheless, all twelve
paired MLP cases were much slower: candidate speedups were only 0.079–0.255×,
with every 99% upper bound below 1.0 (`int4-global-paired-mlp`). It was rejected.
The results do not isolate the cause; Gufo's operand layout and memory access
pattern differ, so removing staging alone does not reproduce its performance.

## Hosted comparison

[Cloudflare's October 9 post](https://blog.cloudflare.com/clef-faster-cheaper-multimodal/)
reports median latencies of 152/305/1,635 ms for roughly 800/3,400/16,000 input
tokens, implying 5,263/11,148/9,786 input tok/s. These fixed-size median rates
are not the same measurement as our mixed-workload aggregate resident rate.
The blog describes serving optimizations including SGLang, without identifying
its production hardware.

The linked [SGLang PR](https://github.com/sgl-project/sglang/pull/42721) separately
reports BF16 tests on one H200: 67 ms at 511 tokens and 572 ms at about 6.5k
tokens, with concurrency one. Its head batches token-wise work, uses variable-length
FlashAttention, and pools schema spans through FP32 prefix sums restricted to the
schema region. These are concrete source-inspection leads; the PR's hardware
should not be presented as confirmed hardware for the hosted blog measurements.
