# clef-hrx

Cloudflare CLEF typed decisions in Rust and Loom, using local `../hrx-rs` on
AMD Strix Halo (`gfx1151`). Implements the 64-layer hybrid text backbone,
27-layer vision encoder, and trained joint schema head. Kernel algorithms live
in `kernels/*.loom`; Rust supplies numeric compiler specializations, bindings,
and graph scheduling. No Python runtime, exporters, or kernel generator scripts.

The checkpoint is pinned to `Cloudflare/clef` revision
`2f3de3dd85f379784083b0814d997ab627200f0c`. Despite its Qwen3.8 branding, its
architecture is `Qwen3_5ForConditionalGeneration`. Weights stay BF16; reductions
and DeltaNet state use FP32. The head uses the separate output embedding matrix.

## Build and use

Requires Rust 1.91+, Linux, AMD kernel drivers, and local `../hrx-rs` (0.8.15).
HRX provisions its pinned native runtime/compiler. CPU tests do not open a GPU.

```sh
cargo build --release --locked
# Metadata only; no GPU or checkpoint weight download:
./target/release/clef inspect
./target/release/clef encode examples/invoice.json
./target/release/clef --max-length 512 decide examples/invoice.json
# Keep one model loaded for newline-delimited requests:
./target/release/clef --max-length 1536 decide requests.jsonl --jsonl
# Include logits, probabilities, and timings:
./target/release/clef --max-length 512 decide examples/invoice.json --raw
```

`--offline` uses cached files; `--model-dir /path/to/snapshot` uses a local
release. The loader checks all 1,184 backbone tensors and 122 head tensors,
including shape, dtype, and shard membership. Local files must remain unchanged
while the model is alive. Transfers use bounded 16 MiB staging chunks.

The checkpoint occupies about 51.2 GiB; resident weights use 46.5 GiB, plus
workspace and staging. Rust reads only the requested BF16 rows from the input
and lexical embedding tables, saving 4.7 GiB without changing precision.
Changed requests incur row reads and uploads; identical-input graph replay
reuses the uploaded rows. `inspect` reports both total and resident weight bytes.
Allow at least 50 GiB **available** RAM for short requests. The loader checks `MemAvailable` before
loading and rechecks during large allocations. `--memory-gib` is an allocation
budget (default 80), not a reservation. The default input limit is 16,384 tokens;
use `--max-length` to limit workspace. `--max-state-tokens` limits state text.
Oversized fixed schemas or media are rejected. Run full-model jobs sequentially.

SystemOne responses contain `model`, `answers`, and `usage`. `choice` returns
winner/confidence/probabilities; `score` returns expected score/legend/probabilities;
`noul` returns the probability of true. Responses round to four decimals.

## Media and library

Requests accept `"images": ["receipt.png"]` and `"videos": ["clip.mp4"]`.
Paths are relative to the working directory. PNG/JPEG/WebP decoding is Rust;
local video files require `ffmpeg` and `ffprobe` on PATH. Sampling, RGB bicubic
resize, normalization, patchification, timestamps, and MRoPE are implemented in
Rust. Decoded video storage is capped at 1 GiB; reduce source resolution or
`num_frames` for larger clips.

`media_kwargs` supports `min_pixels`, `max_pixels`, `fps`, `num_frames`, and
`do_sample_frames`. Unknown keys are errors; `fps` and `num_frames` are mutually
exclusive. Pixel defaults match the pinned processor.

`ClefModel::load` creates an HRX context; `load_in` accepts an existing
`hrx::inference::ModelContext`. `infer`, `infer_batch`, and `systemone` return
owned host results. `media::prepare` accepts RGB images and `VideoFrames`;
`infer_with_media` accepts prepared input. `Encoder` works without GPU/weights.

Transformer and decision-head weights remain resident. Scratch leases and
byte-range graph dependencies allow allocation reuse. Requests execute
sequentially through HRX `NativeSession`.
One exact-input graph is cached; input changes rebuild it while reusing kernels
and allocation capacity. Failed native execution requires reloading the model.
DeltaNet uses 64-token chunkwise prefill above 64 tokens and an independent
recurrent path for shorter sequences. Attention uses bounded-memory online softmax.

## Unit tests

```sh
cargo fmt --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
# GPU operators/head, cached tokenizer, and ffmpeg; excludes the full-model run:
cargo test --locked --lib -- --include-ignored --skip full_checkpoint_corpus --test-threads=1
# Full checkpoint corpus, run alone when RAM is available:
cargo test --release --locked --lib full_checkpoint_corpus -- --ignored --nocapture --test-threads=1
```

Tests cover canonical JSON, schema ordering, response semantics, invalid inputs,
checkpoint validation, exact token/position fixtures, bicubic patch fixtures,
nonzero GEMM/attention/DeltaNet references, state reset, scratch reuse, and the
trained head. Embedding tests check exact BF16 row bytes, duplicate and reordered
IDs, file bounds, and both real checkpoint tables after mappings are released.
Tests read captured reference data directly in Rust. GPU and cache
requirements are explicit `#[ignore]` annotations; no silent test skipping.

The full-model corpus covers all question types, Unicode/JSON, longer text,
images, video, mixed media, and changed/identical-input replay. It checks exact
tokens, winners, a 0.003 maximum probability-error gate, and deterministic replay.
Fixtures record upstream Torch 2.14.0 / Transformers 5.10.2 provenance; they are
not a claim of qualification against the release's Torch 2.11 environment.

**Experimental:** operator/head and preprocessing parity have passed locally.
Earlier full-model text/image runs agreed within 0.0002 probability error.
The full corpus still needs rerunning after the Loom migration, chunked-prefill
switch, video-marker fix, and embedding-row loading, when sufficient RAM is
available. The latest attempt stopped at preflight: 49.1 GiB required versus
24.9 GiB available. Hardware support is currently limited to gfx1151.

## Criterion benchmarks

```sh
cargo bench --locked --bench preprocessing
# Small synthetic graphs; no full checkpoint:
cargo bench --locked --features bench-internals --bench kernels
# Paired old/new comparison, followed by acceptance checks:
cargo bench --locked --features bench-internals --bench kernels -- paired_delta_prefill --save-baseline optimized
cargo run --locked --example check_benches -- target/criterion paired
# Diagnostic per-kernel GPU timestamps (serialized, not throughput evidence):
cargo run --release --locked --features bench-internals --example profile_delta -- 260
# Opt-in, cached checkpoint, at least 50 GiB available RAM:
CLEF_BENCH_FULL_MODEL=1 cargo bench --locked --bench inference
```

Criterion separates warmup and sampling. Kernel benches exclude compilation,
upload, and readback; each iteration includes dispatch and completion fencing.
The full-model bench loads one model and warms its graph before sampling the
complete `infer` call. `CLEF_MODEL_DIR` can select a local checkpoint for the
full-model test/benchmark. Per-request diagnostics separately report encoding,
graph preparation, inference, readback, and allocated/uploaded bytes.

Earlier gfx1151 Criterion estimates before the latest optimization (10 samples;
synthetic DeltaNet only):

| Tokens | Recurrent | Chunked |
| --- | ---: | ---: |
| 65 | 18.13 ms | 2.02 ms |
| 260 | 67.55 ms | 8.43 ms |
| 1,024 | 286.32 ms | 32.77 ms |

These measure a single DeltaNet operation, not complete-model latency.

Acceptance criteria live in `benches/criteria.json`: at least 20% faster chunked
prefill at 260 and 1,024 tokens, at most 5% regression at 65 tokens, and at most
256 MiB of device allocations for each comparison. The paired benchmark keeps
the old and new graphs resident, verifies numerical agreement before measuring,
alternates execution order, and checks that replay does not grow allocations.
The original kernels are retained for tests/benchmarks; normal inference selects
the optimized kernels.

The optimized path uses shared memory for the triangular inverse, computes four
matrix rows per lane to reuse loads, and prepares each token/head cooperatively
across a wave. Matrix reduction order and gate-prefix order are preserved;
normalization uses an FP32 wave reduction. Matrix-tail and inverse-residual tests
pass, and the existing DeltaNet error tolerance remains 0.002.

The checker uses the upper endpoint of a 95% paired bootstrap confidence
interval across 40 measurement batches. Missing data, different run identities,
or changed kernel sources fail the check. Calibration/warmup calls are excluded.
Results are written under `target/criterion`; run all three paired sizes together
from the repository root. Use an idle GPU for reproducible absolute timings.
Set `CLEF_BENCH_PAIRED_SECONDS=30` for a longer measurement under contention
(default 8 seconds per size, allowed range 1–300). This changes sampling time,
not the sample count or acceptance thresholds.
Named before/after Criterion baselines can also be compared with
`check_benches target/criterion before after`; paired evidence is preferable on
a shared machine. Benchmark checks supplement the unit tests and the separate
full-checkpoint qualification; they do not replace either.

Saved paired results are under `benches/results/gfx1151`. The 30-second-target run shared a GPU
that was observed at 99–100% utilization between our runs; absolute latency is
not representative of an idle device. The paired acceptance checks passed:

| Tokens | New / old time | Upper 95% ratio | Required maximum | Combined allocations |
| --- | ---: | ---: | ---: | ---: |
| 65 | 0.715 | 0.775 | 1.050 | 77.6 MiB |
| 260 | 0.641 | 0.680 | 0.800 | 89.8 MiB |
| 1,024 | 0.586 | 0.675 | 0.800 | 137.7 MiB |

The preceding 8-second-target run is retained under `benches/results/gfx1151/contention`.
It failed the 260-token gate (ratio 0.714, upper bound 0.902). Longer sampling
resolved the uncertainty without changing kernels or thresholds. Both runs
passed numerical and allocation checks; neither measures full-model performance.

Verify the saved evidence against the current source:

```sh
cargo run --locked --example check_benches -- benches/results/gfx1151 paired
cargo test --locked --example check_benches
```

`CLEF_TRACE_DIR=/path` retains and dumps FP32 intermediates for debugging;
it increases memory and readback costs. Disable tracing for benchmarks.
