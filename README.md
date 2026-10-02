# clef-hrx

Run [Cloudflare CLEF](https://huggingface.co/Cloudflare/clef) locally on AMD
Strix Halo (`gfx1151`). Rust handles schema encoding and media preprocessing;
[Loom](https://github.com/zacharydenton/hrx-rs) kernels run the text backbone,
vision encoder, and decision head. No Python runtime is required.

CLEF answers structured questions about text, images, and video:

| Type | Result |
| --- | --- |
| `choice` | Selected option, confidence, and probabilities |
| `score` | Expected score, legend, confidence, and probabilities |
| `noul` | Probability that a proposition is true |

**Experimental.** The full-model corpus has a known probability deviation from
the BF16 reference on the incident/outage fixture. The 0.003 error threshold is
kept in the tests. Only `gfx1151` is supported.

## Build

Requires Rust 1.91+, Linux, AMD kernel drivers, and at least 50 GiB of available
RAM for short requests. HRX downloads its pinned native runtime and compiler
when needed. Video input also requires `ffmpeg` and `ffprobe` on `PATH`.

```sh
git clone https://github.com/zacharydenton/clef-hrx
cd clef-hrx
cargo build --release --locked
```

The first inference downloads about 51.2 GiB from `Cloudflare/clef`, pinned to
revision `2f3de3dd85f379784083b0814d997ab627200f0c`. Weights use BF16; resident
weights need about 46.5 GiB, plus workspace. Embedding rows are read on demand.
Use `--offline` for cached checkpoint files or `--model-dir /path/to/snapshot`
for a local copy. Checkpoint files must remain unchanged while the model is loaded.

## CLI

```sh
# Inspect checkpoint metadata or encode a request without loading GPU weights.
./target/release/clef inspect
./target/release/clef encode examples/invoice.json

./target/release/clef --max-length 512 decide examples/invoice.json
./target/release/clef --max-length 512 decide examples/invoice.json --raw

# Keep one model loaded for newline-delimited requests. Use - for stdin.
./target/release/clef --max-length 1536 decide requests.jsonl --jsonl
```

Responses contain `model`, `answers`, and `usage`, with probabilities rounded to
four decimals. `--raw` includes logits, unrounded probabilities, and timings.

A minimal request:

```json
{
  "model": "clef",
  "state": "The invoice is 30 days overdue.",
  "questions": {
    "needs_followup": {
      "type": "noul",
      "instructions": "Does this invoice need a payment reminder?"
    }
  }
}
```

For `choice`, supply a `criteria` object mapping option IDs to descriptions.
For `score`, supply an ordered `criteria` array. See [examples/invoice.json](examples/invoice.json).

Requests can include `"images": ["receipt.png"]` and `"videos": ["clip.mp4"]`.
Paths are relative to the working directory; images support PNG, JPEG, and WebP.
`media_kwargs` accepts `min_pixels`, `max_pixels`, `fps`, `num_frames`, and
`do_sample_frames`. `fps` and `num_frames` are mutually exclusive. Decoded video
storage is capped at 1 GiB.

`--max-length` limits the total input to 1–16,384 tokens (default 16,384).
State text is truncated to fit; oversized schemas and media are rejected.
`--max-state-tokens` further limits state text. `--memory-gib` sets an allocation
budget (default 80), not a reservation. Run full-model jobs sequentially.

## Library

```rust
use clef_hrx::{ClefModel, EncodeOptions, LoadOptions, Request};

let request: Request = serde_json::from_str(include_str!("examples/invoice.json"))?;
let mut model = ClefModel::load(LoadOptions {
    encoding: EncodeOptions { max_length: 512, ..Default::default() },
    ..Default::default()
})?;
let response = model.systemone(&request)?;
```

`Encoder` works without a GPU or model weights. `media::prepare` accepts RGB
images and decoded video frames; `infer_with_media` uses that prepared input.
`load_in` accepts an existing HRX `ModelContext`. Inference is sequential and
caches one exact-input graph. Reload the model after a native execution failure.

## Development

```sh
cargo fmt --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --lib --bins --tests --examples --all-features
cargo test --locked --doc

# GPU operators, trained head, cached tokenizer, and video fixtures:
cargo test --locked --lib -- --include-ignored --skip full_checkpoint_corpus --test-threads=1
# Full checkpoint; run separately with at least 50 GiB available RAM:
cargo test --release --locked --lib full_checkpoint_corpus -- --ignored --nocapture --test-threads=1
```

GPU and checkpoint tests are explicitly ignored by default. Fixtures capture
Torch 2.14.0 / Transformers 5.10.2 outputs. `CLEF_MODEL_DIR` selects a local
checkpoint for the full-model test and benchmark.

```sh
cargo bench --locked --bench preprocessing
cargo bench --locked --features bench-internals --bench kernels
cargo bench --locked --features bench-internals --bench normalization
cargo bench --locked --features bench-internals --bench gemm
CLEF_BENCH_FULL_MODEL=1 cargo bench --locked --bench inference
```

The full-model benchmark reports load and first-request timings separately, then
measures exact-input graph replay and alternating equal-length requests. The
alternating case includes graph rebuilding and uploads with warm kernel
specializations; new token lengths can also incur compilation costs.

The normalization benchmark alternates the serial reference and parallel kernels
and reports paired timings. For full-model per-kernel diagnostics, run
`cargo run --release --locked --features bench-internals --example profile_inference`.
It accepts an optional request JSON path and uses the cached checkpoint with a
1,536-token limit. Diagnostic timings include serialization barriers; use the
inference benchmark for end-to-end latency.

The GEMM benchmark compares the original 32-element reduction tile with the
shape-selected kernel and checks identical outputs. Small matrices retain the
original tile. To compare normalization and GEMM changes within one loaded model:

```sh
cargo run --release --locked --features bench-internals --example compare_inference
# Check all corpus outputs without timing rounds:
cargo run --release --locked --features bench-internals --example compare_inference -- \
  --rounds 0 tests/fixtures/corpus/{0..7}.json
```

This rotates kernel variants, checks exact GEMM prediction parity, and reports
normalization probability differences. Request timings include graph rebuilding;
inference timings isolate graph execution. Run full-model jobs sequentially.

Kernel benchmarks include paired reference/optimized measurements. Check them
with `cargo run --locked --example check_benches -- target/criterion paired`;
thresholds are in [benches/criteria.json](benches/criteria.json). Use an idle GPU.
`CLEF_TRACE_DIR=/path` dumps FP32 intermediates for debugging and increases memory
use. Leave it unset for benchmarks.

## License

[Apache-2.0](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for
checkpoint, algorithm, and kernel attribution.
