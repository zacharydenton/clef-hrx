# Omni expert loads and attention

These experiments follow the [HRX 0.11 upgrade](hrx-011.md). The qualification
workloads are **full JevBench and all four WorkflowEvals datasets**, with separate
multimodal checks. The targets are 1,000 BF16 and 4,000 INT4 resident input tok/s.
Those full-suite targets have not been demonstrated.

Resident timing includes encoding, preparation, compilation, GPU execution and
readback, excluding model loading. Warm diagnostic measurements below exclude
the first request for each configuration and are not full-suite qualification.
All measurements used the shared Radeon 8060S (gfx1151).

## Changes

The default expert kernels now load 128 bytes of each operand row at once and
consume the original ascending K16 matrix fragments. BF16 uses K64 packets;
INT4 uses K256 packets. INT4 scaling, activation packing, output rounding and
weighted expert accumulation retain their existing order.

The router distributes each token's 128 logits across one wave. Subgroup
reductions select the top eight experts, breaking ties by the lowest expert ID.
Selected-probability normalization retains the serial rank order and BF16
rounding. Paired checks cover BF16/FP32 inputs, ties, signed zeros, large logits,
partial waves, expert counts below 128, and changed-input replay.

Expert chunks up to 2,048 tokens are available to benchmarks. The default stays
256 pending full-suite qualification. Experimental M32/N32 expert tiles reuse
weights across twice as many routed rows. At 1,024 tokens, they improve balanced
BF16 up/down projections by 1.89×/2.08× over the wider-load M16/N64 kernels.
A subsequent M64/N16 BF16 candidate also passes exact comparisons to M16/N64
across 1, 17, 256, 1,024 and 2,048 tokens, balanced/concentrated routes and
three changed-input replays. At 1,024 tokens it improves balanced up/down
projections by 3.46×/3.52×, and concentrated up/down by 2.04×/1.64×. It remains
experimental; smaller chunks retain M16/N64 in the adaptive diagnostic.

INT4 does not consistently benefit: concentrated down projection regresses by
13%, so M32 is not the default.

An M16/N32 INT4 candidate also preserves all 20 exact operator cases, but all
eight paired 1,024/2,048-token timings regress by roughly 4–22%. It is rejected.
`scripts/omni_expert_narrow_candidate.py OUTPUT` reproduces the source from the
current M16/N64 kernel; use its directory with `CLEF_BENCH_OMNI_KERNEL_DIR`.

Two experimental BF16 attention implementations support Omni's 128-wide heads:
16-query tiled attention with two probability components, and 64-query/32-key
shared-memory attention with one probability component. The latter reuses query
row padding for softmax statistics and occupies exactly 32 KiB of LDS. Both are
opt-in benchmark paths; the default attention remains unchanged.

## Ordered expert accumulation

An opt-in path computes each token's ascending expert order once during route
grouping, then accumulates eight output columns per thread. It preserves the
original BF16 rounding after every multiply and addition. All 12 operator cases
(1–2,048 tokens, balanced/concentrated routes) match exactly across three changed
inputs. Eight full-model requests, including the seven multimodal smoke cases,
also match exactly across two replays.

Paired Criterion timings include both grouping and accumulation. At 2,048
tokens, balanced routes improve from 2.692 to 0.695 ms (3.88×), and concentrated
routes from 2.392 to 0.679 ms (3.52×). The warm seven-request model diagnostic
improves from 940.0 to 968.7 input tok/s; its longest JevBench request is effectively
unchanged (667.1 versus 664.6 tok/s). These shared-GPU measurements do not establish
a full-suite speedup. Enable with `CLEF_BENCH_OMNI_COMBINE=1` in a
`bench-internals` build; it remains off by default.

Run `CLEF_BENCH_OMNI_FILTER=combine` with the Omni Criterion benchmark to reproduce
the operator checks and paired timing. `compare_omni --variants
packets,packets_sorted --sorted-combine` selects the exact-output model comparison.
The benchmark recorder also accepts `--suite omni` and saves hashes and copies
of explicitly selected expert/attention sources:

```sh
RUSTUP_TOOLCHAIN=stable CLEF_BENCH_OMNI_FILTER=combine \
  python3 scripts/memory_guard.py --output artifacts/omni-combine-guard \
  --memory-gib 2 -- python3 scripts/bench_workload.py omni-combine --suite omni
```

## Shared MLP packing and fused activation

`CLEF_BENCH_OMNI_FUSED_MLP=1` reuses the existing packed-input cache for the
expert gate/up pair and replaces separate SiLU/multiply dispatches with the
existing fused SwiGLU operator. Matching includes the input allocation lease,
view, bit width and packing stride; plan resets discard cached activations.
The default remains unchanged while full-suite qualification is pending.

GPU tests pass exact BF16 comparisons through 16,384 routed rows, sliced-input
and packed-weight lifetime checks, and bounded activation scratch checks.
Paired SwiGLU timing improves 0.628 to 0.344 ms at 2,048 tokens (1.83×).
`CLEF_BENCH_OMNI_FILTER=swiglu` selects the five Omni Criterion shapes;
`compare_omni --variants packets,packets_fused --fused-mlp` compares model outputs.
The combined shared-packing, fused-SwiGLU and ordered-accumulation path also
reproduces all eight diagnostic requests exactly, including both replays.

## Compact expert scheduling experiment

`CLEF_BENCH_OMNI_COMPACT_EXPERTS=1` builds a GPU task list of populated
16-row expert tiles using a workgroup prefix scan. It skips the empty row
partitions in the original launch geometry, keeping each expert's tasks
together. INT4 arithmetic and rounding remain unchanged. This is available
only in `bench-internals` and remains off by default.

All 20 operator cases pass exact output checks across three changed inputs.
Independent CPU checks verify the grouped routes and compact task list for
1, 17, 256, 1,024 and 2,048 tokens, including balanced and concentrated routing.
Paired Criterion timing includes grouping, task-list generation, packing and
the projection:

| Tokens | Projection | Balanced speedup | Concentrated speedup |
|---:|---|---:|---:|
| 1,024 | Up | 0.88× | 1.12× |
| 1,024 | Down | 0.98× | 1.78× |
| 2,048 | Up | 0.96× | 1.35× |
| 2,048 | Down | 1.11× | 2.62× |

The improvement depends on routing, so these results do not establish a
whole-model speedup. An alternative preserving the original channel/row
ordering also passes exact checks, but loses most of the concentrated-route
benefit. Reproduce the selected candidate with
`python3 scripts/omni_expert_compact_candidate.py kernels/omni_expert_i4_compact.loom`;
add `--row-first` to generate the alternative.

Run the Omni Criterion benchmark with `CLEF_BENCH_OMNI_FILTER=Int4`,
`CLEF_BENCH_OMNI_REFERENCE_DIR=kernels`, `CLEF_BENCH_OMNI_EXACT=1` and
`CLEF_BENCH_OMNI_TOKENS=1,17,256,1024,2048`, together with the compact flag.
Use `--test` for the exact checks and `--bench` for paired timing on the printed
benchmark executable, under a fresh 4 GiB memory guard.

The eight-request model check at 2,048-token chunks matches all outputs exactly
across both replays. With unchanged scalar attention, warm media throughput is
1,019.3 versus 1,007.3 input tok/s, and the longest JevBench request is 726.4
versus 701.7 tok/s (ordinary versus compact scheduling). This diagnostic does
not show an end-to-end improvement. With the faster transposed attention and
existing MLP fusions, full JevBench completes at **2,344.7 input tok/s**, versus
2,488.5 without compaction. All 231 outputs are identical: **187/231 correct**
and Brier score 0.25089. Resident time is 69.85 seconds, including 12.49 seconds
of preparation and 56.81 seconds on the GPU. Compaction is rejected for promotion;
the operator gains do not carry through to this workload.

## Text backbone graph reuse

`CLEF_BENCH_OMNI_REUSE_BACKBONE=1` is an experimental `bench-internals` path
that retains one padded text backbone graph. Requests with the same padded
positions upload new embeddings and rebuild only the decision head. Shape or
position changes rebuild the backbone; media requests use the ordinary path.
Recorded scratch/input leases stay pinned while the head is built, preventing
the two graphs from reusing each other's buffers. Benchmark configuration
changes invalidate both graphs.

The current JevBench ordering has 163 consecutive requests with the same padded
length, out of 230 transitions. CPU checks and the GPU buffer-isolation test
pass, including changed inputs and repeated head rebuilds. Full JevBench
reproduces all 231 outputs exactly, and the eight-request model diagnostic
matches both replays, including the seven media checks.

The initial full JevBench run is slower: **2,359.1 input tok/s**, with 19.88
seconds of preparation and 49.15 seconds of GPU execution. Many short requests
prepare faster, but larger spikes erase the saving. The current compiled-kernel
cache clears all 512 entries when full. `CLEF_BENCH_OMNI_KERNEL_LRU=1` instead
evicts the least recently used entry. Its full JevBench run reproduces all 231
outputs at **2,494.0 input tok/s**, essentially tied with 2,488.5 without graph
reuse. Preparation takes 15.10 seconds and GPU execution 50.19 seconds. The cache
records 142,202 lookups, 2,217 loads and 1,705 evictions, retaining 512 entries.
This does not establish a throughput improvement. `CLEF_BENCH_OMNI_CACHE_STATS=1`
logs these counters when the engine closes. Graph reuse and LRU remain off by
default.

The first LRU qualification attempt stopped during model loading because system
swap grew by more than 1 GiB. It produced no evaluation requests. The completed
retry used the same guard limits after memory pressure settled.

`CLEF_BENCH_OMNI_KERNEL_CACHE_CAPACITY` permits a bounded capacity experiment
from 1 to 8,192 entries; its default remains 512. The completed 4,096-entry
full JevBench run reproduces all 231 outputs: 187/231 correct and **2,505.9
resident input tok/s**. Preparation takes 16.62 seconds and GPU execution
48.43 seconds. It records 142,202 lookups, 2,053 loads, no evictions and 2,053
retained entries. This is only 0.7% faster than the selected 2,488.5 result and
does not establish a useful overall improvement; graph reuse and larger caches
remain experimental.

A further cache-only INT4 run, without backbone graph reuse, reproduces all 231
outputs at 2,133.5 input tok/s. An unrelated GPU client was active again; this
run does not establish a speed improvement or isolate the cache's timing effect.
Artifacts are in `artifacts/omni-memory-opt/jev-cache-only`.

## SwiGLU packing experiment

`CLEF_BENCH_OMNI_PACK_SWIGLU=1` sends the expert gate/up outputs directly to the
existing register packing kernel. It preserves BF16 SiLU and multiply rounding,
then rotates and packs without materializing the intermediate activation. BF16
experts retain their existing path. This is off by default and independent of
`CLEF_BENCH_OMNI_FUSED_MLP`.

The GPU packing test passes exact comparisons for both 4/8-bit dense operands
and Omni's 768-wide routed operands, including sliced inputs, exceptional values,
replay, and 8 through 16,384 routed rows. The isolated eight-request model
comparison reproduces all 32 outputs exactly, including seven media requests.
An earlier comparison also changed padding and differed on one video request;
its timings cannot isolate the fusion.

Paired operation timings improve by 1.29×, 1.52×, 1.32×, 1.77× and 2.28× at
1, 17, 256, 1,024 and 2,048 input tokens respectively. At 2,048 tokens the
operation falls from 0.657 to 0.289 ms. Full JevBench reproduces all 231
predictions exactly (187 correct), but measures **2,192.9 input tok/s**, below
2,488.5 without this fusion. Preparation takes 19.75 seconds and GPU execution
54.47 seconds. This shared-GPU run does not demonstrate a whole-model gain, so
the fusion is not selected. The Criterion cases compare the production
materialized path against the fusion under alternating execution order:

```sh
cargo +stable bench --locked --features bench-internals --bench omni --no-run
# Run the printed executable under the memory guard:
CLEF_BENCH_OMNI_FILTER=packing python3 scripts/memory_guard.py \
  --output artifacts/omni-packing-guard --memory-gib 4 -- \
  target/release/deps/omni-<hash> --bench omni_swiglu_packing
```

## Resident BF16 results

The M64 BF16 expert candidate completes all 231 JevBench cases at **981.7 resident
input tok/s**, versus 414.2 for the default (2.37×). It scores **185/231 correct
(80.09%)**, versus 184/231: two gains, one loss and four changed selected answers.
The Brier score worsens from 0.24948 to 0.25481. This clears the primary-accuracy
floor. This original run remains below 1,000 tok/s; the later 128-token
threshold run below crosses that target on JevBench. Neither qualifies
WorkflowEvals by itself.

The configuration uses M64 BF16 experts for chunks of at least 512 tokens,
2,048-token chunks, 128-token padding, transposed BF16 attention, and the exact
MLP/ordered-combine fusions. Resident preparation takes 23.55 seconds, GPU
execution 142.85 seconds, and peak engine allocation is 58.57 GiB. The separate
eight-request comparison produces 32 samples with exact replay and unchanged
selected answers, covering the seven media fixtures and the longest JevBench
request. Artifacts are `jev-bf16-m64` and `bf16-m64-model.jsonl` under
`artifacts/omni-memory-opt/`.

A cache-only follow-up stopped during loading with no predictions after an
unrelated `beam.smp` GPU client appeared and available RAM crossed the unchanged
24 GiB reserve. It does not supersede the completed result. The 62 GiB BF16
engine cap still requires 94 GiB available for admission; full BF16 workflows
need their own memory qualification.

## INT8 attention experiment

`scripts/omni_attention_int8_candidate.py` generates a Sage-inspired candidate
with signed INT8 query/key products, BF16 probabilities/values and FP32 softmax
and accumulation. Each 128-value query/key row uses its own symmetric scale.
Quantization happens inside the attention kernel; the experiment does not use
SageAttention's smoothing or H3's rotation. It accepts the existing BF16 bindings
and changes no default selector.

All six 64/128/256-query by 32/64-key variants compile without spills. The
128-query/32-key variant uses 184 VGPRs and 14,976 bytes of LDS, versus 184 VGPRs
and 18,944 bytes for the BF16 candidate. Compilation is not evidence of numerical
correctness or a speedup. The completed GPU checks reject all six variants
under the current 0.02 operator-error limit: the 17-token hashed fixture reaches
0.024658203 maximum absolute error. This is an operator check, not a measured
model accuracy regression; no INT8-attention variant is qualified for promotion.
The probe now supports independent `patterned`, `hashed` and `zero-queries`
fixtures; the latter exercises uniform softmax with zero quantized query rows.

```sh
python3 scripts/omni_attention_int8_candidate.py --queries 128 --keys 32 \
  artifacts/omni-int8-attention.loom
cargo +stable build --locked --release --features bench-internals \
  --example probe_attention
target/release/examples/probe_attention --model clef-omni --compile-only \
  --assembly artifacts/omni-int8-attention.asm \
  artifacts/omni-int8-attention.loom
```

AMD's [RDNA 3 WMMA throughput table](https://gpuopen.com/learn/wmma_on_rdna3/)
lists the same dense operation rate for BF16 and IU8, with twice that rate for
IU4. This architecture-family reference explains why INT8 QK should not be
assumed to double matrix arithmetic throughput: any gain must offset packing,
scaling and memory costs. The local paired results above remain the evidence
for this gfx1151 implementation.

### Prepacked INT8 Q/K

The generator's `--prepacked` mode moves row quantization into
`benches/kernels/omni_prepare_attention_i8.loom`. Each head stores 32 packed
INT8 words, one FP32 scale and three padding words. The attention kernel reuses
these values across query tiles. The separate packer compiles to 16 VGPRs,
8 SGPRs and no LDS or private scratch. All six attention variants compile
without spills; the 128-query/32-key variant uses 152 VGPRs and 14,976 bytes of
LDS. The independent GPU packing oracle passes. All six prepacked variants
match their inline counterparts exactly in 192 checks across the three fixtures,
tail lengths and contexts through 65,536. They retain the same 0.024658203
hashed-fixture error against BF16 and fail the 0.02 operator gate. Separate
long-context comparisons against the selected BF16 source are being collected;
prepacking alone does not establish an end-to-end speedup.

`CLEF_BENCH_OMNI_PREPACK_QK=1` enables packing only in benchmark builds. Native
model selection restricts this experiment to causal 128-wide text attention;
noncausal media encoder attention retains its ordinary path. Use a matching
prepacked source. `compare_omni --prepack-qk` applies packing to candidate
variants while preserving its reference. No default changes.

The independent packing test checks exact signed byte codes, row scales,
zero rows, outliers, sliced inputs and graph replay with changed inputs. The
operator queue then requires exact agreement with the corresponding inline
INT8 source across three fixtures, tail lengths and contexts through 65,536.
Separate comparisons against transposed BF16 attention check numerical error
and alternate reference/candidate timings. Every timed candidate replay includes
both Q/K packing passes and the existing V transpose. A successful operator
check is not a model accuracy qualification.

```sh
python3 scripts/omni_attention_int8_candidate.py --prepacked \
  --queries 128 --keys 32 artifacts/omni-prepacked-attention.loom
# Compilation only; does not open the GPU.
target/release/examples/probe_attention --model clef-omni --compile-only \
  --compile-packing benches/kernels/omni_prepare_attention_i8.loom
# Run this GPU probe under the memory guard, after other GPU jobs finish.
python3 scripts/memory_guard.py --output artifacts/omni-prepacked-probe-guard \
  --memory-gib 4 -- target/release/examples/probe_attention \
  --model clef-omni --tokens 32768 --rounds 5 --fixture hashed \
  --prepacked-qk --max-error 0.02 \
  --reference-source benches/kernels/omni_attention_transposed.loom \
  artifacts/omni-prepacked-attention.loom
```

The paired Criterion attention group also supports this mode. Set
`CLEF_BENCH_OMNI_FILTER=attention`, `CLEF_BENCH_OMNI_PREPACK_QK=1`, and
`CLEF_BENCH_OMNI_ATTENTION_SOURCE` to the generated source. Select the BF16
comparison with `CLEF_BENCH_OMNI_ATTENTION_REFERENCE`; without it the scalar
oracle is used. `CLEF_BENCH_OMNI_ATTENTION_TOKENS=1024,4096,32768,65536` selects
long-context measurements and `CLEF_BENCH_OMNI_ATTENTION_FIXTURE` accepts
`patterned`, `hashed`, or `zero-queries`. The existing 0.015625 maximum-error gate
and exact replay check still apply. Use a fresh memory-guard output for each run.

### Register-prefetched attention

`scripts/omni_attention_prefetch_candidate.py` carries the next key/value tile
in registers while the current tile computes. It preserves the original QK,
softmax and PV arithmetic, one LDS tile and both workgroup barriers. The final
unused prefetch is clamped to the last valid tile; no extra input padding is
required. This experiment follows the loading approach in krea2-hrx's Sage
kernel, with BF16 and prepacked INT8 source variants.

Four BF16 and four INT8 variants compile without private scratch. The BF16
64-query/64-key variant exceeds the register budget; the INT8 version spills
120 bytes and is excluded from GPU timing. For 128 queries/32 keys, BF16 uses
200 VGPRs and 18,944 LDS bytes; prepacked INT8 uses 184 VGPRs and 14,976 LDS
bytes. GPU checks require exact equality to the corresponding unprefetched
source, including hashed/zero-query fixtures and 64K contexts. INT8 comparisons
include packing on both sides; they do not waive the earlier BF16 error gate.

```sh
python3 scripts/omni_attention_prefetch_candidate.py --queries 128 --keys 32 \
  artifacts/omni-prefetch-bf16.loom
# Add --prepacked for the INT8 source. The probe's --prepacked-qk selects
# candidate packing; --reference-prepacked-qk includes reference packing too.
```

`CLEF_BENCH_OMNI_ATTENTION_REFERENCE_PREPACKED=1` enables the corresponding
paired Criterion reference, with an explicit `CLEF_BENCH_OMNI_ATTENTION_REFERENCE`
source. This separates the effect of prefetching from the effect of packing.
All eight compiled, spill-free variants pass 248 exact GPU comparisons. In the
paired operator samples, BF16 128-query/32-key prefetch improves its reference
by 2.0% at 32K and 2.4% at 64K. BF16 256-query/64-key prefetch improves its own
slower reference by 16.9% and 19.9%, reaching 267.0 ms and 1,048.4 ms. Those
are comparisons against the matching tile configuration, not a paired comparison
against the selected 128-query/32-key kernel. The best prepacked INT8 prefetch
variant reaches 258.4 ms and 1,002.3 ms, while retaining the failed BF16 error
gate. None establishes the full-model target; no prefetch candidate is promoted.

## Full JevBench results

The default wider-load INT4 kernels and parallel router complete all 231 cases:
**186/231 correct (80.52%)**, Brier score 0.25417, **619.8 resident input tok/s**.
The 230 resident requests contain 163,770 tokens and take 264.22 seconds. The
first request, including model loading, takes 102.82 seconds and is excluded
from resident throughput. No request is truncated.

Resident requests spend 105.92 seconds preparing graphs and kernels, 157.82
seconds executing on the GPU, and 0.43 seconds encoding. This includes compilation
and exposes costs that warm repeated-request measurements omit. These results
establish the INT4 reference for experimental attention and larger chunks; they
do not meet the 4,000 tok/s target.

The BF16 reference completes the same full suite at **184/231 correct (79.65%)**,
Brier score 0.24948, and **414.2 resident input tok/s**. Its peak engine allocation
is below 59 GiB. INT4 gains eight correct cases and loses six, changing 17 selected answers
in total; BF16 has the
lower Brier score. Both runs use the same 164,002-token request corpus and exclude
only the first request from resident timing. Neither throughput target is met.

## Experimental JevBench result

Shared 64-query/32-key attention, 2,048-token expert chunks and 128-token right
padding complete full INT4 JevBench at **188/231 correct (81.39%)** and
**1,741.2 resident input tok/s**. The reference is 186/231 at 619.8 tok/s. Four
cases improve and two regress; Brier score changes from 0.25417 to 0.25732.
Resident wall time is 94.06 seconds: 34.61 preparation, 59.02 GPU execution and
0.39 encoding. No padded tokens are added to the throughput numerator.

This is an opt-in result, not a new default. WorkflowEvals qualification is still
pending. The experimental flags are `CLEF_BENCH_OMNI_CHUNK=2048`,
`CLEF_BENCH_OMNI_ALIGNMENT=128`, `CLEF_BENCH_OMNI_ATTENTION=1` and
`CLEF_EXPERIMENT_ATTENTION_SOURCE=benches/kernels/omni_attention_shared.loom`.
They require a build with `bench-internals`.

The prior same-model INT4 diagnostic retains all ten choices with shared
attention. Chunk sizes 1,024 and 2,048, and right-padding to 128 tokens, reproduce
the same shared-attention probabilities on those eight requests. Shared
attention itself changes probabilities by up to 0.13421 versus scalar attention.

## WorkflowEvals qualification

The [pinned Cloudflare Omni model card](https://huggingface.co/Cloudflare/clef-omni/blob/0db1cd2607d76a7bdb2a382f659e7b313079f84b/README.md#workflow-evals)
publishes these consensus scores for the same four workflows:

| Metric | Published Omni |
|---|---:|
| Invoice exact actions | 60.2% |
| Invoice primary action | 82.0% |
| Customer service exact actions | 71.6% |
| Security exact actions | 61.7% |
| Agent trace primary action | 65.8% |

These are accuracy references, not local results. The sampled first invoice has
36,883 Omni input tokens and 55 questions. At that length, the ideal BF16
attention-only ceiling from 59.392 TFLOPS is approximately 4,095 input tok/s:
`peak / (2 * 48 layers * 32 heads * 128 dimensions * input_tokens)`.
This excludes experts, projections, softmax, data movement and runtime overhead;
INT4 weight packing alone does not remove this attention constraint.

The experimental INT4 configuration completes this invoice in 41.00 seconds
including first-shape preparation (899.5 input tok/s), and 47.78 seconds on the
ordinary post-profile repeat (772.0 tok/s). Both produce identical probabilities.
Peak engine allocation is 20.54 GiB under a 28 GiB cap. Shared-GPU contention
affects these measurements; neither is a full-suite throughput result.

The serialized profile attributes 68.2% of GPU time to attention, 12.7% to
experts and approximately 4.3% to expert accumulation. Its barriers add overhead;
use the ordinary request timings above for throughput. Further expert-only
improvements cannot close the long-context gap to 4,000 tok/s by themselves.

The full 705-case run was started at
`artifacts/omni-memory-opt/workflows-shared-int4`, using the frozen
`clef-candidate` binary and the same experimental flags as full JevBench.
It runs customer service, agent traces, security and invoices, with no case limit
or context truncation. The memory guard stopped it when additional system swap
exceeded 1 GiB, after the first three workflows completed and invoices had just
started. Their checkpoints are retained; there is no full-suite score yet.

Customer service has completed all 204 cases with no failed executions:
consensus exact actions **67.65%**, below the published 71.6% BF16 reference.
Its 506 resident native requests contain 594,335 tokens and take 282.95 seconds
(**2,100.5 input tok/s**), including preparation. This does not establish the
full-suite speed or prove nonregression against default INT4.

The same-precision default comparison completes all 204 cases without execution
failures at **68.63%** consensus exact actions (420/612 policy decisions), versus
414/612 for shared attention. The shared candidate gains 23 decisions and loses
29, a net loss of 0.98 percentage points. It therefore fails the customer-service
nonregression check.

The newer transposed attention plus exact MLP/ordered-combine fusions completes
all 204 customer-service cases without failures at **68.95%** (422/612), gaining
28 decisions and losing 26 versus default INT4. This passes the customer-service
nonregression check. Its 507 resident calls contain 595,675 input tokens and take
192.69 seconds: **3,091.4 input tok/s**, including 38.10 seconds of preparation
and 153.14 seconds on the GPU. It still needs the other three workflows; this
is not a full-suite qualification or a demonstrated 4,000 tok/s result.

The remaining agent-trace, security and invoice workflows are running at
`artifacts/omni-memory-opt/workflows-transposed-remaining`, using the same frozen
`clef-transposed-fused` binary and kernel flags as customer service, with a 28 GiB
engine cap. Graph reuse, LRU, compaction and direct SwiGLU packing are disabled.
The 24 GiB live system reserve and 1 GiB additional-swap limit remain in force.
There is no case limit or accepted-input truncation.

This transposed run has completed all 111 agent-trace cases without execution
failures: **63.06%** consensus primary-action accuracy, versus 64.41% for the
older shared-attention experiment. Its 371 resident calls process
791,135 input tokens in 250.18 seconds (**3,162.3 input tok/s**).
This does not establish same-precision nonregression against default INT4;
security has also finished and invoices remain in progress.

Security completes all 240 cases without execution failures at **56.67%**
consensus exact actions, versus 55.83% for the older shared-attention experiment.
Its 270 resident calls process 702,900 tokens in 194.83 seconds
(**3,607.8 input tok/s**). The default INT4 comparison is still missing;
these partial-suite results cannot establish overall nonregression.

The default reference resumed from 72 saved cases after available RAM fell below
the guard's 24 GiB reserve. Its combined resident timing is 912.9 input tok/s,
excluding both model loads; this is a resumed measurement, not an uninterrupted
throughput run. Both segments retain the same binary and guard settings.

Agent traces have also completed all 111 cases without execution failures:
consensus primary-action accuracy **64.41%**, versus the published 65.8%.
Resident throughput is **2,332.9 input tok/s**. Security completed all 240 cases
without execution failures at **2,166.5 input tok/s** and **55.83%** consensus
exact-action accuracy, versus the published 61.7%. Invoices are incomplete;
these completed workflows do not substitute for the full-suite result.

## Transposed BF16 attention candidate

`scripts/omni_attention_transposed_candidate.py` generates 64/128/256-query,
32/64-key variants using h3-hrx's transposed WMMA layout. Each lane carries one
query's softmax statistics, reducing repeated cross-lane exchanges. Queries,
keys, probabilities and values remain BF16; softmax and accumulation remain
FP32. Causal/GQA handling and bounded tail loads match Omni's interface.

The four 64/128-query variants compile with HRX 0.11. The 128-query/32-key
version uses 184 VGPRs and 18,944 bytes of LDS. The 64-query/32-key variant uses 240 VGPRs;
64-key variants use 240 VGPRs and 35,840 bytes of LDS. These are compiler
resource reports, not throughput measurements.

All four variants pass guarded operator checks from 1 through 65,536 tokens.
For 1–4,096 tokens they compare against scalar attention with maximum error
0.0078125; long cases compare against the existing shared kernel. The selected
128-query/32-key variant takes 282.6 ms versus 508.5 ms at 32K tokens (1.80×),
and 1,139.6 ms versus 2,204.3 ms at 64K (1.93×), with maximum long-case error
0.0009765625. These are shared-GPU attention-dispatch timings, not whole-model
throughput. The selected source is `benches/kernels/omni_attention_transposed.loom`.
The combined INT4 model diagnostic retains all ten choices across eight
requests, with exact repeats. Probabilities change by up to 0.15031 versus the
previous shared-attention path, so operator error alone does not establish model
parity. The warm seven-request multimodal aggregate improves from 1,030.2 to
1,145.6 input tok/s; the longest JevBench request improves from 3,124.5 to
4,022.6 tok/s. These are diagnostics, not the full-suite qualification metric.
Full JevBench completes at **187/231 (80.95%)** and **2,488.5 resident input
tok/s**, versus 188/231 and 1,741.2 tok/s for the previous experiment, and
186/231 and 619.8 tok/s for default INT4. Brier score improves to 0.25089 from
0.25732 and 0.25417, respectively. Relative to the previous experiment, three
cases improve and four regress; relative to default INT4, two improve and one
regresses. Resident time is 65.81 seconds: 16.22 preparation, 49.16 GPU and
0.40 encoding. This is a 1.43× experimental-throughput improvement, still below
the 4,000 tok/s goal. The run uses 2,048-token chunks, 128-token padding, shared
MLP packing/fused SwiGLU, ordered accumulation and transposed attention.

The prior WorkflowEvals run retained its frozen shared-attention kernel;
the new configuration still requires full WorkflowEvals qualification.

Two further 256-query candidates compile without spills: 32 keys uses 184
VGPRs and 18,944 bytes of LDS; 64 keys uses 208 VGPRs and 35,840 bytes of LDS.
Both launch 512 threads per workgroup. Their GPU correctness and performance
checks pass at 12 lengths from 1 through 65,536 tokens. The 32-key variant is
exact against the selected 128-query/32-key kernel, but only 1.02× faster at
64K and slower at shorter lengths. The 64-key variant has maximum error
0.001953125 and is about 11% slower at 32K/64K. Neither is selected.
The existing 64/128-query generated sources remain byte-for-byte unchanged.

Optional `--lane-controls` and `--paired-accumulators` generator switches
adapt H3's probability packing and accumulator layout. All six tested variants
pass exact comparisons against their corresponding tiles at 12 lengths each.
Neither change produces a convincing long-context gain; both remain experimental.

`probe_attention --compile-only --assembly PATH` saves the new runtime's target
assembly without opening the GPU. Guarded operator checks cover 1–65,536 tokens.

## Measurements

One resident INT4 model ran seven multimodal requests plus JevBench's longest
request, with two replays per configuration. All replays reproduced themselves
exactly. The reference reproduced the saved HRX 0.11 predictions exactly.

| Configuration | Warm multimodal input tok/s | Longest JevBench request, input tok/s | Exact reference probabilities |
|---|---:|---:|---|
| Previous expert kernels, 256-token chunks | 546 | 475 | Yes |
| Wider loads, 256-token chunks | 944 | 665 | Yes |
| Wider loads, 1,024-token chunks | 991 | 714 | Yes |
| M32 expert tiles, 1,024-token chunks | 998 | 713 | Yes |
| M32 plus tiled attention, 1,024-token chunks | 1,078 | 2,082 | No |

All ten question decisions across those eight requests remain unchanged. The
attention variant changes probabilities, so unchanged smoke answers are not
sufficient to qualify it. The longest request contains 3,968 tokens. Full
JevBench contains 231 requests and 164,002 input tokens with the Omni tokenizer.
WorkflowEvals planning validates all 705 cases (150 invoices, 204 customer
service, 111 agent traces and 240 security incidents).

The wider INT4 expert loads improve balanced 256-token up/down projections by
2.94×/2.71× against the previous grouped kernels. The parallel router improves
the 1,024-token top-eight selection by about 12×. These are operator gains,
not whole-model speedups.

The shared attention candidate passes 1, 17, 65, 127, 256, 1,024, 2,048 and 4,096
tokens against scalar attention, with maximum output error 0.0078125. A timed
20-pair test of the initial 33-KiB variant measured 107.93 → 8.63 ms at 4,096
tokens (12.50×). The revised 32-KiB candidate passes the same shape checks; its
single-pair timings are diagnostic only. The 32-KiB candidate also passes a
32,768-token comparison to the two-component tiled reference (maximum error
0.0078125). At 65,536 tokens, comparison with the separately staged 64-key
candidate gives maximum error 0.001953125. These bounded operator checks do not
establish full-model quality at long context.

The default wider-load BF16 kernels also reproduce the prior HRX 0.11 streamed
text and audio/video predictions exactly. They pass the full 48-layer Cloudflare
oracle checks, with maximum probability errors 0.000622 and 0.005157. Streaming
keeps allocation below 2.3 GiB and does not measure resident throughput.

Raw predictions, hashes, memory guards and operator results are in
[the result record](results/omni-memory-20261015.json). The first attempted
multimodal load was stopped by the unchanged swap-growth guard; the successful
comparison used a fresh guard. An initial JevBench attempt used an intermediate
binary with a router compilation error and produced no scored cases.

## Reproduce

Use the pinned, tiled Omni checkpoint described in [Omni setup](clef-omni.md).

```sh
cargo +stable build --release --locked --features bench-internals \
  --bin clef --example compare_omni --example probe_attention
cargo +stable bench --locked --features bench-internals --bench omni --no-run
```

Run the printed benchmark executable under `scripts/memory_guard.py`, using a
fresh output directory and a 4-GiB budget. `--test` checks correctness without a
full timing run; `--bench` collects paired Criterion timings.

- `CLEF_BENCH_OMNI_FILTER=topk` selects the exact serial/parallel router checks.
- `CLEF_BENCH_OMNI_FILTER=attention` selects the 128-wide attention matrix.
- `CLEF_BENCH_OMNI_ATTENTION_SOURCE=benches/kernels/omni_attention_shared.loom`
  selects the shared attention candidate for that matrix.
- `CLEF_BENCH_OMNI_TOKENS=512,1024,2048` selects larger expert batches.
- `CLEF_BENCH_OMNI_KERNEL_DIR` selects a directory containing
  `omni_expert_bf16.loom` and `omni_expert_i4.loom`; the corresponding `_m32`
  candidates live in `benches/kernels/` and must be copied under those names.
- `CLEF_BENCH_OMNI_EXACT=1` requires bitwise BF16 equality for expert comparisons.
- `CLEF_BENCH_OMNI_REFERENCE_DIR` selects an earlier grouped-kernel directory
  instead of the scalar reference. Reference kernels must support the chosen
  token counts.

The previous grouped K16 kernels are saved as `omni_expert_bf16_k16.loom` and
`omni_expert_i4_k16.loom` in `benches/kernels/`; they support chunks up to 256.
Create a reference directory with those files renamed to the ordinary kernel
names. Create the candidate directory the same way using the `_m32` files.
The seven smoke requests can be extracted from the `request` fields in
`tests/fixtures/omni/decisions-{0..6}.json`.

`compare_omni --help` describes the one-model diagnostic. It records every raw
prediction and wall time, checks repeated execution, and permits probability
changes for subsequent accuracy analysis. Its warm rates exclude compilation;
use the standard suite runners for the qualification metric.

```sh
python3 scripts/eval_jevbench.py --model clef-omni \
  --model-dir artifacts/model-omni-tiled --precision int4 --memory-gib 24 \
  --output artifacts/omni-jev-preflight --prepare-only

python3 scripts/memory_guard.py --output artifacts/omni-jev-guard \
  --memory-gib 24 -- python3 scripts/eval_jevbench.py --model clef-omni \
  --model-dir artifacts/model-omni-tiled --precision int4 --memory-gib 24 \
  --prepared artifacts/omni-jev-preflight --output artifacts/omni-jev

python3 scripts/memory_guard.py --output artifacts/omni-workflows-guard \
  --memory-gib 28 -- artifacts/WorkflowEvals/.venv/bin/python \
  scripts/eval_workflowevals.py --model clef-omni \
  --model-dir artifacts/model-omni-tiled --precision int4 --memory-gib 28 \
  --output artifacts/omni-workflows
```

Keep the binary unchanged between preflight and evaluation. BF16 needs a larger
resident allocation and corresponding system headroom; streaming BF16 oracle
checks validate quality but cannot establish resident throughput.

## Invoice residency interruption

The selected transposed INT4 run completed agent traces and security, then 78
invoice cases before the 28 GiB engine residency budget was exhausted preparing
`ap_00149`. The allocation pool retained 29,995,734,916 bytes after the 56,091-token
`ap_00141` case. Host memory reserves remained healthy; the native process
reported `resource busy: residency budget is pinned or loading`. The next queued
case, `ap_00151`, encountered the already-exited process. Neither error is an
accuracy result.

Both failed checkpoints are preserved under the run's `attempts/` directory.
An inspected resume used the same binary, runner, settings and 28 GiB cap. It
stopped during loading when system swap grew by more than 1 GiB; no additional
invoice checkpoint completed. The original 78 successful cases remain intact,
and further full-model loads are held pending inspection of memory pressure.
This second stop is a host swap-guard event, distinct from the original engine
residency exhaustion. Event details are in `workflows-transposed-remaining-resume1-guard/status.json`.
Future resumed timings must be labeled segmented and exclude each process's
loading request. The recovery record is
`artifacts/omni-memory-opt/invoice-residency-recovery.json`.

## Bounded scratch retention

After checkpoint loading, the native engine caps idle scratch at half the
configured budget left after live model allocations. At synchronized plan-reset
boundaries it discards unused buffers above that limit, keeping larger reusable
capacities first. Allocation slots remain stable for live tensors, packed
weights and retained graph pins; empty slots are reused for subsequent buffers.
HRX's separate upload-staging cache remains charged to the same engine budget.
This prevents unused capacities from accumulating across changing request shapes;
it does not increase the engine cap or alter the host memory guard.

The GPU regression executes 21 increasing/decreasing-shape requests with replay
under a shared 16 MiB budget, checking exact outputs, retained weights, bounded
idle scratch and reusable slot metadata. Separate tests confirm retained backbone
graphs survive aggressive eviction and initialized weights still share the context
budget. All three pass. The first accounting assertion omitted HRX's upload
staging; the corrected test includes the staging envelope implied by its seven
input sizes. Both attempts remain recorded.

The six saved invoices spanning 14,032–56,091 tokens pass both sweeps: all 12
predictions, including raw probabilities, exactly match the saved results. Peak
engine allocation is 21.85 GiB under the unchanged 28 GiB cap. This short sequence
is not a replay of the entire interrupted suite. The separate 64-sample text/media
comparison also passes with every output exactly matching the saved results.
Its first attempt stopped at the unchanged 1 GiB additional-swap guard after 29
exact samples. After inspecting the stop and renewed memory availability, a fresh
attempt completed under the same 24 GiB engine cap. Both attempts are retained;
this is validation of output preservation, not a full-suite throughput claim.
The successful media artifact is `scratch-cache-media-retry1.jsonl`. Its completed
queue is recorded in `after-media-retry-status.json`; the subsequent invoice
attempt and its terminal stop are recorded in `after-prefetch-selected-status.json`.

## Probability scheduling experiment

`scripts/omni_attention_probability_candidate.py` consumes each BF16 probability
fragment across all eight output-channel fragments before preparing the next
one. It preserves every operation, each accumulator's K-order and all rounding;
only independent scheduling changes. Six combinations of tile size and register
prefetch compile without spills. Most increase register use. The prefetched
128-query/64-key variant decreases from 256 to 240 VGPRs, but no GPU correctness
or performance claim has been established for this scheduling change. Sources
and resource reports are under `artifacts/omni-memory-opt/attention-probability-*`.


## Prefetch compared with the selected kernel

The direct paired comparison uses the selected 128-query/32-key transposed
kernel as its reference. Both prefetched candidates pass 25 operator cases each,
covering patterned, hashed and zero-query fixtures, tails, and lengths through
65,536. The 128-query/32-key candidate is exact; the 256-query/64-key candidate
has a maximum difference of 0.001953125 in these fixtures, below the existing
0.02 operator gate. This does not establish model-level accuracy.

| Candidate | 1,024 tokens | 32,768 tokens | 65,536 tokens |
| --- | ---: | ---: | ---: |
| Prefetch 128/32 | 0.983× | 1.021× | 1.055× |
| Prefetch 256/64 | 0.943× | 1.067× | 1.096× |

These seven-round, alternating-order operator measurements used the shared GPU.
The larger tile is a candidate for long-context dispatch, not a blanket
replacement. It remains experimental. Full commands, source hashes and outputs
are in `artifacts/omni-memory-opt/attention-prefetch-selected-results.json`.

## K128 integer expert packets

`scripts/omni_expert_k128_candidate.py OUTPUT --rows 16|32` generates N64
integer experts that load half of each packed K256 row at a time. Checkpoint
layout, ascending K16 accumulation, output scaling and BF16 rounding remain
unchanged. M32 reuses the four weight fragments across two routed-row fragments;
M16 isolates the packet-size change. Both up/down specializations compile without
spills: M16 uses 144 VGPRs and M32 uses 172, with 20 SGPRs and 2 KiB LDS each.
Both pass 20 exact operator cases across 1–2,048 tokens, up/down projections,
balanced/concentrated routes and three changed-input replays. M16 shows no
consistent timing benefit. M32 improves 2,048-token down projections by 1.23×
with balanced routes and 1.47× with concentrated routes, but balanced up
projections regress to 0.93×. At 1,024 tokens the corresponding down gains are
1.02×/1.19×. These are shared-GPU operator measurements, not model throughput.

The generator's `--down-only` variant therefore retains the original K256
up/gate kernel and uses M32/K128 only for down projections. Projection direction
is a specialization constant, so the compiler removes the unused branch. Both
branches compile without spills (224 VGPRs for the original up kernel, 172 for
the candidate down kernel). All 24 exact cases pass, including 512 tokens. Paired up/gate performance stays
at 0.98–1.01× for 512–2,048 tokens. Down-projection gains at 1,024 tokens are
1.16×/1.23× for balanced/concentrated routes, and 1.08×/1.31× at 2,048 tokens.
At 512, balanced down is 0.98× and concentrated down is 1.15×. Variation from the
first paired sweep illustrates the shared-GPU limits; these are not model-speed
claims. The cached-loader model run below now qualifies this candidate on JevBench
and the saved text/media checks; full WorkflowEvals remains incomplete.

`examples/compile_kernel.rs` exposes HRX's compile report and optional assembly
without opening the GPU. Supply each specialization with `-D clef.NAME=VALUE`.
The complete up/down configurations and raw reports are saved in
`artifacts/omni-memory-opt/expert-k128-compile-summary.json`. Run the existing
Omni Criterion benchmark with `CLEF_BENCH_OMNI_REFERENCE_DIR=kernels`,
`CLEF_BENCH_OMNI_KERNEL_DIR` pointing at the generated kernel directory,
`CLEF_BENCH_OMNI_FILTER=Int4`, `CLEF_BENCH_OMNI_EXACT=1` and
`CLEF_BENCH_OMNI_TOKENS=1,17,256,1024,2048`. Use `--test` for exact checks and
`--bench` for alternating paired timing, under the unchanged 4 GiB memory guard.


The bounded continuation `bounded-k128-bf16-status.json` records the down-only
checks followed by paired BF16 M64 measurements at 128, 256, 384 and 512 tokens.
All 16 cases pass exact changed-input and replay checks. At 128–256 tokens,
balanced routes remain within 2% of the reference while concentrated routes
improve by 2.12–2.86×. At 384 tokens, balanced up/down improve by 1.94×/1.92×;
concentrated up/down improve by 2.82×/2.24×. These results support testing a
128-token M64 threshold instead of 512. The later full JevBench check is
recorded below; full WorkflowEvals measurements remain necessary.


## Consumed checkpoint file-cache experiment

The existing loader calls HRX `FileView::done_with` after consuming each tensor,
which advises away its mapped pages. The opt-in Linux benchmark setting
`CLEF_BENCH_CHECKPOINT_DROP_CACHE=1` additionally gives `POSIX_FADV_DONTNEED`
advice for complete file pages inside that consumed tensor. It uses HRX 0.11's
original mapped-file descriptor and checked tensor file range, avoiding reopening
a possibly replaced pathname. It does not alter tensor bytes, model precision,
engine budgets, or host-memory guards. The intent is to reduce checkpoint
file-cache pressure during loading; a reduction in swap growth is not yet proven.

A CPU regression consumes and rereads tensors spanning multiple pages, checks a
small adjacent tensor, and replaces the pathname while retaining the original
mapped inode. All bytes remain exact. This setting remains off by default and
requires `bench-internals`; suite metadata now records all `CLEF_BENCH_` settings.


The full-model attempt with `compare-checkpoint-cache` stopped during loading
at the unchanged 1 GiB additional-swap guard, before producing any predictions.
JevBench did not start. File-cache advice alone has therefore not solved the
loading pressure. `checkpoint-cache-model-status.json` records that failed
attempt. The original invoice baseline remains held separately.

## Exact packed-weight cache

`CLEF_BENCH_PACKED_WEIGHT_CACHE=artifacts/omni-packed-weight-cache` enables an
experimental persistent cache of the GPU-produced integer codes and scales.
It requires `bench-internals` and remains off by default. Cache keys include
the source bytes, tensor shape and precision, packing kernels, compiler identity
and GPU target. Reads validate the header, extent and payload checksums; corrupt
entries fail explicitly. Writes use an atomic rename. Quantization arithmetic,
model precision and memory guards are unchanged.

The GPU regression covers INT4 and INT8, three matrix shapes, one and 17 input
rows, fresh engines and graph replays. Codes, scales and projection outputs
match exactly after reload. All 29 CPU tests pass, as do the stable/default and
Rust 1.91 Clippy checks. Validation logs are named `packed-cache-*` under
`artifacts/omni-memory-opt`.

To prepare weights without retaining the full model, build and run:

```sh
cargo build --release --locked --features bench-internals --example prepack_omni
CLEF_BENCH_CHECKPOINT_DROP_CACHE=1 python3 scripts/memory_guard.py \
  --output artifacts/omni-prepack-guard --memory-gib 4 -- \
  target/release/examples/prepack_omni \
  --model-dir artifacts/model-omni-tiled \
  --cache-dir artifacts/omni-packed-weight-cache
```

Preparation completed all 48 layers, producing 192 entries totaling
14,785,813,784 bytes. It retained one layer at a time, with 818,036,736 bytes of
engine allocation. A warm two-layer check had eight hits and no misses, with
308,035,584 bytes allocated. The loader still reads and hashes source BF16
weights; these checks establish exact cache reuse and lower temporary GPU
allocation, not a full-model loading or inference speedup.

The fresh `packed-cache-model-status.json` queue uses frozen
`compare-packed-cache-v2` and `clef-packed-cache-v2` binaries. It checks all 64
saved text/media outputs before full JevBench, using the down-only K128 candidate
at a 512-token minimum. It retains the 24 GiB cap, 32 GiB admission headroom,
24 GiB live reserve and 1 GiB additional-swap stop. The media comparison completed: all 64 saved outputs are exactly preserved,
with 192 cache hits and no misses. Model loading took 137.05 seconds under the
unchanged guard. Full JevBench then completed at **2,660.0 resident input tok/s**, with all
231 raw outputs exactly matching `jev-transposed-int4`: **187/231 correct**,
Brier 0.25089. This is 6.1% above the previous best 2,505.9 run and 6.9% above
2,488.5 without backbone reuse. The shared GPU and combined loader/scratch/kernel
changes prevent attributing that entire difference to K128 alone. Resident
preparation takes 11.68 seconds and GPU execution 49.42 seconds; peak engine
allocation is 17.55 GiB. Cache preparation and model loading are excluded from
the resident throughput metric. The 4,000 tok/s target remains unmet.

To reproduce the selected INT4 Jev configuration after preparing the cache:

```sh
cargo build --release --locked --features bench-internals --bin clef
mkdir -p artifacts/omni-k128-down
python3 scripts/omni_expert_k128_candidate.py \
  artifacts/omni-k128-down/omni_expert_i4.loom --rows 32 --down-only
cp kernels/omni_expert_bf16.loom artifacts/omni-k128-down/
CLEF_BENCH_PACKED_WEIGHT_CACHE=artifacts/omni-packed-weight-cache \
CLEF_BENCH_CHECKPOINT_DROP_CACHE=1 \
CLEF_BENCH_OMNI_KERNEL_DIR=artifacts/omni-k128-down \
CLEF_BENCH_OMNI_MIN_TOKENS=512 CLEF_BENCH_OMNI_CHUNK=2048 \
CLEF_BENCH_OMNI_ALIGNMENT=128 CLEF_BENCH_OMNI_ATTENTION=1 \
CLEF_BENCH_OMNI_COMBINE=1 CLEF_BENCH_OMNI_FUSED_MLP=1 \
CLEF_EXPERIMENT_ATTENTION_SOURCE=benches/kernels/omni_attention_transposed.loom \
python3 scripts/memory_guard.py --output artifacts/omni-jev-k128-guard \
  --memory-gib 24 -- python3 scripts/eval_jevbench.py \
  --model clef-omni --model-dir artifacts/model-omni-tiled \
  --binary target/release/clef --precision int4 --memory-gib 24 \
  --output artifacts/omni-jev-k128
```


## Capped row-grid experiment

`scripts/omni_expert_row_loop_candidate.py OUTPUT --row-waves 4|8 --source SOURCE`
caps the number of row waves launched for each expert/channel tile. Each wave
loops over further populated tiles with a stride equal to the capped grid width.
This avoids launching the worst-case number of empty row partitions without
building a separate compact task list. Every matrix product, route gather,
output scatter and rounding step is retained. Concentrated routing may lose
parallelism, so both balanced and concentrated cases must be measured.

Both variants compile without spills when derived from the down-only K128
candidate. Up/gate uses 224 VGPRs; down uses 192, versus 172 before adding the
row loop. LDS stays 2 KiB. Both grids pass 24 exact operator cases, covering
1, 17, 256, 512, 1,024 and 2,048 tokens, both projections and both route patterns.
Against the down-only K128 source, eight row waves at 2,048 tokens improve
balanced up/down by 1.04×/1.08× and concentrated up/down by 1.16×/1.59×.
At 512 tokens, concentrated up regresses to 0.92×. Four row waves also regress
several small cases. All timings alternate reference/candidate on the shared GPU.

The generator's `--min-tokens 1024` option retains the original specialized body
below 1,024 tokens. Compilation confirms unchanged 224/172 up/down VGPR counts
at 512 tokens and the capped-grid resources above the threshold, without spills.
The adaptive variant passes all 24 exact operator cases. Its subsequent full
JevBench attempt stopped during loading at the unchanged additional-swap guard,
with zero predictions. Available RAM was still 59.2 GiB, but global swap had
grown by more than 1 GiB. `row-loop-model-status.json` records this terminal
attempt; it is not automatically retried. Packed caching has enabled completed
loads, but does not eliminate global memory-pressure stops. No model speedup is
established for row-grid capping; defaults are unchanged.


## Current profile priorities

The cached K128 build completed a fresh profile under the 28 GiB guard. Ordinary
outputs before and after profiling match exactly for the 495-, 3,968- and
36,883-token requests. Peak engine allocation on the invoice is 20.20 GiB.
The serialized intervals include profiling barriers and are diagnostic only:

| Input tokens | Grouped experts | Causal attention |
| ---: | ---: | ---: |
| 495 | 52.1% | 4.2% |
| 3,968 | 31.1% | 21.0% |
| 36,883 | 12.7% | 67.3% |

Expert scheduling addresses the short-request bottleneck. Long invoices require
attention improvements as well; the observed expert-kernel gains alone cannot
establish the full WorkflowEvals target. The raw profile is
`artifacts/omni-memory-opt/packed-cache-profile.json`.


## Attention layout and mask experiments

The transposed-attention generator's experimental `--swizzle` option permutes
whole 16-element LDS packets within rows instead of adding padding. Writer and
reader permutations agree, leaving the attention arithmetic intact. K32/K64
layouts use 16/32 KiB rather than 18.5/35 KiB. Seven variants pass 25 operator
cases each, but all are slower at 65,536 tokens: 0.63–0.88× the selected kernel.
The layout change is rejected for promotion despite its smaller LDS footprint.

K128 fits in 64 KiB with this layout. Its first two candidates failed small-tail
checks (nonfinite output and a device execution error): the kernel assumed V's
64-token allocation padding covered a 128-key tile. The generator now clamps
out-of-capacity V packets to an existing packet; their probabilities are masked
to zero. Both corrected candidates pass 31 checks, including tails on either
side of 64/128-token boundaries and hashed inputs through 65,536. Maximum
output difference from the selected K32 reference is 0.0078125. They remain
slower (0.79×/0.70× at 64k), so neither is selected. The original failed sources
and corrected results are retained separately.

All six probability-block scheduling candidates also pass 31 operator cases.
K32 variants are exact; K64 differs by at most 0.0078125 on the long hashed
fixture. Three-round 64k timing ranges from 0.90× to 1.37×, with visible shared-GPU
variation; this does not establish an end-to-end gain. Raw evidence is in
`attention-probability-k128-results.json`.

`scripts/omni_attention_mask_candidate.py SOURCE OUTPUT` adds an interior-tile
branch to the existing transposed or prefetched kernel. A tile fully within the
key input and preceding every query needs no per-element mask. Boundary tiles
retain the original masks. No dot products, softmax operations, rescaling or
rounding change. All three variants compile without spills, retaining their
original VGPR/LDS counts. Both K32 variants complete 31 exact checks against
their matching sources plus seven-pair timings against the selected kernel.
The unprefetched variant is 1.02–1.03× at 32k–64k. The prefetched variant is
1.08× at 1k, 1.05× at 8k and 1.06× at 32k, but 0.93× at 64k. These shared-GPU
measurements do not support a blanket replacement. The K64 variant completed
30 exact checks before the global additional-swap guard stopped the run; its
selected-kernel timing is incomplete. `attention-mask-results.json` records
the interruption rather than treating it as a numerical failure.

The prefetched K32 candidate completed a paired model test on the 36,883-token
invoice: **1,463.2 → 1,521.5 resident input tok/s**, a **4.0%** whole-request
improvement over three alternating pairs. Both warmups and all six timed raw
predictions match exactly. One resident INT4 model retained the selected K128
experts and fusion settings; loading and first compilation are excluded.
`attention-mask-invoice.json` contains the samples, and
`attention-mask-invoice-launch.json` and `attention-mask-invoice-guard` record
the command and unchanged guards. An unrelated GPU client overlapped the timed
requests, so this remains a shared-GPU diagnostic. The 64k operator regression
still argues against a blanket replacement; no defaults change.

## Interleaved QK scheduling

`scripts/omni_attention_qk_candidate.py SOURCE OUTPUT` visits each dimension
fragment across the independent 16-key QK accumulators before advancing the
reduction dimension. Each accumulator's summation order stays unchanged.
Masking, softmax and PV retain the same operations. The generator verifies that
the reordered region contains exactly the original operation lines.

All three mask-based candidates compile without spills and retain their original
VGPR/LDS counts. Assembly confirms that independent destination accumulators
alternate, replacing the original eight-instruction dependent chains. The first
K32 candidate passes 31 exact checks. Seven-pair comparisons show 1.08× at 1k,
1.03× at 8k and 1.02× at 32k; the global swap guard stopped the run before its
64k comparison and the remaining candidates. Available memory increased during
that stop, from 63.2 to 66.3 GiB, while swap grew by slightly over 1 GiB. This
is incomplete performance evidence, not a qualified kernel. The artifacts are
`attention-qk-compile-summary.json` and `attention-qk-results.json`.

## JevBench after memory became available

The unrelated process retaining roughly 44 GiB of GPU memory exited. With
111.6 GiB available before admission, the 62 GiB BF16 job could run under the
unchanged guards. Lowering the M64 expert threshold from 512 to 128 tokens
completed full JevBench at **1,145.2 resident input tok/s**, **185/231 correct**,
with every raw prediction identical to `jev-bf16-m64` (Brier 0.25481). This
crosses the 1,000 tok/s target on JevBench only. Relative to the previous
981.7 tok/s run, the gain is 16.6%; resource availability also changed, so it
cannot all be attributed to the threshold. An additional GPU client appeared
for two telemetry samples; its ownership was not captured. Minimum available
RAM was 52.0 GiB.

`free-gpu-qualification-status.json` records this result and the following
fresh INT4 K128/adaptive-row comparisons. These are new runs after a verified
resource change; the earlier memory-stopped attempts remain separate.

The fresh K128 INT4 run completed at **3,031.4 resident input tok/s**, with
**187/231 correct** and all 231 raw predictions identical to `jev-packed-k128`.
The selected kernel/settings are unchanged from the 2,660.0 tok/s run. This
14.0% timing difference reinforces the need to record resource conditions.
The adaptive row-grid comparison completed at **2,979.1 tok/s**, with all raw
predictions unchanged. It is 1.7% slower, so the full qualification queue keeps
K128 without row-grid capping. The INT4 target needs a further 32.0% throughput
improvement over the selected completed run.

The memory guard now records owned/client process RSS and swap, global reclaim
counters, and its own source hash. These fields aid diagnosis only: admission
headroom, the live reserve and the additional-swap limit are unchanged. The
five existing guard tests pass.

To reproduce the BF16 threshold configuration:

```sh
cargo build --release --locked --features bench-internals --bin clef
mkdir -p artifacts/omni-m64-128
cp benches/kernels/omni_expert_bf16_m64.loom artifacts/omni-m64-128/omni_expert_bf16.loom
cp kernels/omni_expert_i4.loom artifacts/omni-m64-128/
export CLEF_BENCH_OMNI_KERNEL_DIR=artifacts/omni-m64-128
export CLEF_BENCH_OMNI_MIN_TOKENS=128 CLEF_BENCH_OMNI_CHUNK=2048
export CLEF_BENCH_OMNI_ALIGNMENT=128 CLEF_BENCH_OMNI_ATTENTION=1
export CLEF_BENCH_OMNI_COMBINE=1 CLEF_BENCH_OMNI_FUSED_MLP=1
export CLEF_EXPERIMENT_ATTENTION_SOURCE=benches/kernels/omni_attention_transposed.loom
python3 scripts/memory_guard.py --output artifacts/omni-jev-m64-128-guard \
  --memory-gib 62 -- python3 scripts/eval_jevbench.py \
  --model clef-omni --model-dir artifacts/model-omni-tiled \
  --binary target/release/clef --precision bf16 --memory-gib 62 \
  --output artifacts/omni-jev-m64-128
```

Use the same environment for full WorkflowEvals, with a 70 GiB engine budget
and fresh output directory:

```sh
python3 scripts/memory_guard.py --output artifacts/omni-workflows-bf16-guard \
  --memory-gib 70 -- artifacts/WorkflowEvals/.venv/bin/python \
  scripts/eval_workflowevals.py --model clef-omni \
  --model-dir artifacts/model-omni-tiled --binary target/release/clef \
  --precision bf16 --memory-gib 70 --max-length 65536 \
  --output artifacts/omni-workflows-bf16 \
  --workflows customer_service agent_trace_observability security_incidents invoice_processing
```

The pinned dry run validates all 705 cases (204 customer, 111 agent, 240 security,
150 invoice). `full-workflows-qualification-status.json` tracks the queued
multimodal checks and full BF16/INT4 runs, using frozen binaries. The INT4
configuration is chosen from the exact Jev comparisons; adaptive row scheduling
must exceed K128 by more than 1% to be selected. INT4 uses the persistent packed
cache and a 28 GiB engine cap for long inputs. This queue stops on any failed job
or changed preliminary raw prediction; it does not retry automatically. Full
suite performance and accuracy remain pending until those runs complete.

## Workgroup scheduling candidate

`scripts/omni_expert_workgroup_candidate.py OUTPUT --source SOURCE --waves N`
regroups the existing independent INT4 expert waves into workgroups of one,
four or eight waves. Two waves reproduce the input source unchanged. Global
wave IDs, private LDS slices and arithmetic stay the same; a final partial
workgroup masks excess waves before tensor access. All six K128 up/down
specializations compile successfully. GPU correctness and timing have not yet
run; these sources are experimental and do not affect defaults or the queued
suite configurations.
