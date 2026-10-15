# Omni grouped experts

The latest work is documented in [expert loads and attention](omni-memory-tuning.md).

The original Omni expert kernel used a separate matrix-vector product for every
selected token/expert pair. A serialized profile of the 258-token text fixture
at `3db62de` attributed 3,145 ms of 3,251 ms to expert multiplication. The ordinary
GPU forward pass took 3,240 ms.

The grouped path builds one stable route list per expert and reuses it for gate,
up and down projections. Each wave multiplies 16 routed rows by 64 output
channels, gathering inputs and scattering outputs in their original route order.
It reuses the WMMA fragments, tiled weight addressing and output publishing from
our BF16 GEMM, plus the packed operand encoding from the h3-hrx integer GEMM.
Weight conversion, activation quantization and weighted expert accumulation are
unchanged. No checkpoint reconversion is needed.

INT4 assigns each 16-row partition a separate wave. The earlier version that
looped through multiple partitions within a wave failed the concentrated-routing
check and was rejected. Empty partitions skip matrix operations. BF16 retains
its per-wave loop; both balanced and concentrated routing pass its checks.

## Results

The measurements below use HRX 0.10.1. The subsequent
[HRX 0.11 upgrade](hrx-011.md#validation-and-performance) retains grouped
experts, adds parallel route compaction, and requalifies the compiler's numerical
changes. Exact agreement with the initial INT4 run applies to 0.10.1 only.

Across the seven text/image/audio/video requests, warm resident throughput rises
from **75.7 to 520.3 input tok/s (6.88×)**. All 14 outputs, including repeats,
match the initial INT4 logits and probabilities exactly. Peak engine allocation
stays at 17.7 GiB. These rates include encoding, preparation, GPU execution and
readback, excluding model load. Both runs of each pair together give 461.8 tok/s.
[Raw outputs, profiles, operator timings and guard records](results/omni-grouped-experts-20261015.json).

On the shared Strix Halo GPU, the same text fixture's ordinary GPU forward pass
fell to 355 ms: **9.13× faster, 727 input tok/s**, with exactly unchanged INT4
logits and probabilities. Serialized diagnostic expert time fell to 245 ms.
Profiling inserts barriers; throughput uses ordinary inference timings.

The Criterion matrix covers BF16 and INT4, gate/up (`2048 → 768`) and down
(`768 → 2048`), 1/17/128/256 tokens, and balanced/concentrated expert selection.
All 128 expert weights are resident. Every replay includes activation packing
and route grouping. Reference and candidate runs alternate order; construction,
weight upload and correctness reads are outside timing. Production amortizes
route grouping across all three projections.

At 256 tokens, balanced routing gave **10.9× / 12.2×** BF16 gate-up/down gains
and **15.8× / 15.9×** INT4 gains. Concentrated routing is also checked, including empty
experts and partial 16-row tiles. Tiny balanced batches have little weight reuse;
the 17-token BF16 gate/up case was 4% slower in this shared-GPU measurement.

## Correctness and memory

Before timing, each operator graph runs three different sets of routes and
activations without rebuilding. INT4 must match the reference bit for bit; BF16
must stay within the checked tolerance (0.78125% relative, with a 0.000078125 absolute floor
near zero). A separate sparse CPU oracle verifies routing, selected weights and
weighted combination. Full BF16 text and synchronized audio/video checks pass
with unchanged top options and maximum probability error 0.00901 against the
Cloudflare oracle (maximum change from previous native BF16: 0.00625). Full-model
qualification is described in
[Omni results](clef-omni.md#qualification-october-15-2026).

The benchmark has a 4 GiB engine budget, checks host memory before constructing
fixtures, and holds one projection's weights at a time. Full-model INT4 uses a
24 GiB budget. Full BF16 validation streams layers through an 8 GiB budget and
releases checkpoint page cache between layers; its elapsed time includes loading
and is not resident throughput. All GPU runs use the existing memory guard and
leave unrelated GPU clients running.

## Reproduce

Build once, then run the Criterion matrix under the memory guard. The output
directories must be new:

```sh
cargo bench --locked --features bench-internals --bench omni --no-run
python3 scripts/memory_guard.py \
  --output artifacts/omni-expert-bench --memory-gib 4 -- \
  cargo bench --locked --features bench-internals --bench omni
```

To select cases before allocating fixtures, set `CLEF_BENCH_OMNI_FILTER`, for
example `Int4/256` or `Bf16/17/up/false`. The final component is `true` for
concentrated routing. This is separate from Criterion's timing filter.

For kernel experiments, `CLEF_BENCH_OMNI_KERNEL_DIR=PATH` reads
`omni_expert_bf16.loom` and `omni_expert_i4.loom` from that directory without
rebuilding Rust. The interface and tile geometry must remain compatible; the
same correctness checks run before timing. These overrides only exist with
`bench-internals`.

Use the [converted checkpoint view](clef-omni.md#verification-and-reproduction) to
avoid rehashing the pristine shards during repeated experiments:

```sh
cargo build --release --locked --features bench-internals --example profile_inference
CLEF_MODEL_DIR=artifacts/model-omni-tiled \
python3 scripts/memory_guard.py \
  --output artifacts/omni-profile --memory-gib 24 -- \
  target/release/examples/profile_inference \
    --model clef-omni --precision int4 --memory-gib 24 --rounds 2 \
    --output artifacts/omni-profile.json request.json
```

Set `CLEF_BENCH_OMNI_REFERENCE=1` for the old expert path in the same build.
Compare identical requests and include encoding, preparation and readback when
reporting resident throughput. Use the seven saved oracle requests and the JSONL
comparison commands in [Omni reproduction](clef-omni.md#verification-and-reproduction)
for full-model validation.
