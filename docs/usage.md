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
