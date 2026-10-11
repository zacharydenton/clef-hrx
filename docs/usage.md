# CLEF requests and Rust API

Commands run from the repository root.

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
For `score`, supply an ordered `criteria` array. See [examples/invoice.json](../examples/invoice.json).

Requests can include `"images": ["receipt.png"]` and `"videos": ["clip.mp4"]`.
Paths are relative to the working directory; images support PNG, JPEG, and WebP.
`media_kwargs` accepts `min_pixels`, `max_pixels`, `fps`, `num_frames`, and
`do_sample_frames`. `fps` and `num_frames` are mutually exclusive. Decoded video
storage is capped at 1 GiB.

`--max-length` limits the total input to 1–65,536 tokens (default 65,536).
State text is truncated to fit; oversized schemas and media are rejected.
`--max-state-tokens` further limits state text. `--memory-gib` sets an allocation
budget (default 80), not a reservation. Run full-model jobs sequentially.

## Precision

`--precision bf16` is the default. `--precision int4-mlp` packs the 192 MLP
projections; `--precision int4` additionally packs eligible large attention and
DeltaNet projections (368 matrices total). Both are available in ordinary builds.
The Rust API accepts `LoadOptions { precision: Precision::Int4, ..Default::default() }`.

Weights are rotated in 256-element groups and packed once when loading. Each
projection prepares activations on the GPU; integer accumulation is rescaled to
BF16 output. Vision, embeddings, small projections, normalization, recurrence and
the trained decision head retain their existing arithmetic. Packed weights are
prepared from the same pinned checkpoint, with no additional model download.

Full JevBench measurements were 752 tok/s and 199/231 for MLP-only INT4,
986–1,007 tok/s and 194/231 for broader INT4, and 516 tok/s and 202/231 in BF16.
The [latest qualification](evaluation.md#eight-row-causal-convolution) uses the
same binary for all modes and preserves every output of each prior baseline.
The broader INT4 range retains the initial run, unchanged-binary repeat and a
fresh control; the repeat remains below 1,000 tok/s. Peak engine allocation was
14.08 GiB for broader INT4 and 23.96 GiB for MLP-only INT4. Eight-row causal
convolution reuses overlapping inputs while preserving FMA order and both BF16
rounding boundaries; selected INT4 projection epilogues also fuse residuals.
A [compact-grid follow-up](evaluation.md#partial-mlp-down-raster-groups) retained
accuracy but failed to establish a model-level speedup, so it is not selected.
The modes are explicit accuracy/speed choices, not replacements for the default.
The ordinary CLI exposes the same tested packing and matrix kernels; retain the
exact binary, flags and results when comparing subsequent builds.

```sh
cargo build --release --locked --bin clef
python3 scripts/eval_jevbench.py --model-dir artifacts/model \
  --precision int4 --output tests/results/jevbench-int4
# Use --precision int4-mlp for the narrower choice.
```

Benchmark-only environment switches remain available with `bench-internals`;
explicit INT4 precision selects the corresponding fixed configuration.

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
