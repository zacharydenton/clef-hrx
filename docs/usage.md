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

Select Flash with `--model clef-flash`; default model selection is original CLEF.
The CLI selection controls loaded weights. The request's `model` field remains a
response label and does not switch checkpoints in a resident process.

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
State text is truncated to fit by default; `--strict-context` rejects truncation.
Oversized schemas and media are rejected in either mode.
`--max-state-tokens` further limits state text. `--memory-gib` sets an allocation
budget (default 80), not a reservation. Run full-model jobs sequentially.

## Omni prefix caching

Use `--prefix-cache` to reuse the image, audio, video and state prefix across
Omni requests in one resident process:

```sh
./target/release/clef --model clef-omni --precision int4 --prefix-cache \
  --strict-context decide requests.jsonl --jsonl --raw
```

The first request populates one cache entry during normal inference. Later
requests with the same encoded prefix reuse all 48 layers' attention keys and
values and the prefix hidden states. Only the question suffix runs through the
backbone; the decision head still runs over the complete context. CPU media
preprocessing still runs to validate and hash the input. Changing the encoded
state, media content or positions replaces the entry, even when a media filename
is unchanged. Request-dependent state truncation can also change the prefix;
use `--strict-context` when you require the full state.

For isolated questions, put **one question in each JSONL request** with the same
state and media. Multiple questions in one request retain the model's normal
cross-question attention. Caching does not change that contract or return joint
answer probabilities. Splitting computation into prefix and suffix can introduce
small floating-point differences compared with an ordinary full forward pass.

`timings.cached_prefix_tokens` is zero on a miss and reports the reused prefix
length on a hit. `input_tokens` always counts the complete logical request, so
cached tokens must not be described as freshly computed throughput. Report
request latency and cache-hit counts when comparing this mode. Miss timings
include prefix population; model loading is separate.

Caching is optional and currently supports Omni only, in all three precisions.
The entry uses approximately **196 KiB per prefix token** (about 1.53 GiB at
8,192 tokens), in addition to weights and execution workspace. It shares the
existing `--memory-gib` budget. There is no disk cache or cross-process reuse.

In Rust, call `model.set_prefix_cache(true)?` after loading Omni. Calling it again
clears the entry; `model.set_prefix_cache(false)?` clears it and disables reuse.
The cache lasts until it is replaced, cleared or the model is dropped.

Compare cached and uncached inference with a JSONL corpus, including changed
questions and changed prefixes:

```sh
cargo build --release --locked --example compare_prefix_cache
./target/release/examples/compare_prefix_cache \
  --model-dir artifacts/model-omni-tiled --precision int4 --memory-gib 28 \
  --output prefix-comparison.jsonl requests.jsonl
```

The runner records first-use and repeat timings for uncached, cached and reversed
request sequences. It checks repeated hits, unchanged winning answers, a default
maximum probability difference of 0.02, and restoration of uncached results after
disabling the cache. Use the existing memory guard for full-model runs on a shared
machine. The GPU mask oracle can be run separately with
`cargo test --release --lib omni_suffix_attention_oracle -- --ignored`.

The [recorded 28-request check](results/omni-prefix-cache-20261016.json) covers
isolated and grouped bin-image questions, two 1,669-token text prefixes, images,
speech, video, prefix replacement and reversed request order. Both INT4 modes
matched every uncached probability exactly in this check. Warm repeat totals:

| Workload | Precision | Uncached | Cached | Speedup |
| --- | --- | ---: | ---: | ---: |
| 17 isolated bin questions | INT4 | 5.07 s | 2.44 s | 2.08× |
| Four longer-text requests | INT4 | 1.81 s | 0.58 s | 3.12× |
| 17 isolated bin questions | INT4 MLP | 5.05 s | 2.94 s | 1.72× |
| Four longer-text requests | INT4 MLP | 1.58 s | 0.47 s | 3.35× |

These are single samples on a shared GPU, excluding model loading. First-use
timings and cache misses are retained in the result file. BF16 full-model
validation was blocked by the memory guard (102 GiB required); the BF16 attention
operator passed its separate 64K oracle. This is a cache equivalence check, not a
new JevBench or WorkflowEvals score. For reproduction, use the
[saved request corpus](results/omni-prefix-cache-20261016-requests.jsonl) with the
comparison runner above and the [pinned OpenDecision image](opendecision-examples.md).

## Precision

`--precision bf16` is the default for both models. For original CLEF,
`--precision int4-mlp` packs the 192 MLP
projections; `--precision int4` additionally packs eligible large attention and
DeltaNet projections (368 matrices total). Both modes also support Flash (96 MLP projections, 184 broader projections).
All modes are available in ordinary builds.
The Rust API accepts `LoadOptions { precision: Precision::Int4, ..Default::default() }`.

Weights are rotated in 256-element groups and packed once when loading. Each
projection prepares activations on the GPU; integer accumulation is rescaled to
BF16 output. Vision, embeddings, small projections, recurrence and the trained
decision head retain their existing precision. Selected INT4 normalization and
packing operations are fused, with a different FP32 RMS summation order and
the original BF16 rounding boundary. Packed weights are
prepared from the same pinned checkpoint, with no additional model download.

Original CLEF full JevBench measurements were 847 tok/s and 204/231 for MLP-only INT4,
1,032 tok/s and 194/231 for broader INT4, and 616 tok/s and 202/231 in BF16.
The [latest qualification](evaluation.md#tiled-weight-layout) uses the same
binary for all modes and reproduces every output of the prior runs exactly, so
same-mode accuracy, probability drift and calibration are unchanged. Peak
engine allocation is 13.88 GiB for broader INT4, 23.74 GiB for MLP-only INT4
and 47.77 GiB for BF16.
Fused RMS normalization and packing apply only to compatible INT4 projections
at 128–4,032 rows; longer requests retain their prior path and the default
context remains 65,536 tokens.
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
