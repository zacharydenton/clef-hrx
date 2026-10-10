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

The first follow-up is the M128 tile comparison, then register/scratch inspection
and activation-packing reuse. INT4 remains an explicit speed/accuracy option;
BF16 remains the default.

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
