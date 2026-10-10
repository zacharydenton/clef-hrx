# clef-hrx

[Cloudflare CLEF](https://huggingface.co/Cloudflare/clef) inference in Rust for
AMD Strix Halo. The text backbone, vision encoder and decision head run as
[Loom](https://github.com/ROCm/hrx-system) kernels through
[HRX](https://github.com/zacharydenton/hrx-rs).

CLEF answers structured questions about text, images and video:

| Question type | Output |
| --- | --- |
| `choice` | Selected option and probabilities |
| `score` | Expected score, legend and probabilities |
| `noul` | Probability that a proposition is true |

## Build and run

Requires Rust 1.91+, Linux, a `gfx1151` GPU and at least 50 GiB of available RAM
for short requests. Video input needs ffmpeg and ffprobe.

```sh
git clone https://github.com/zacharydenton/clef-hrx.git
cd clef-hrx
cargo build --release --locked
./target/release/clef --max-length 512 decide examples/invoice.json
```

For example, `invoice.json` can ask:

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

Add `--raw` for logits, unrounded probabilities and timings. For repeated
requests, keep the model loaded and pass newline-delimited JSON:

```sh
./target/release/clef --max-length 1536 decide requests.jsonl --jsonl
```

First use downloads roughly 51.2 GiB of BF16 weights. Resident weights occupy
about 46.5 GiB plus workspace. Use `--model-dir PATH` for a local snapshot or
`--offline` for cached files. `--memory-gib` sets the allocation budget (default
80 GiB); `--max-length` limits input tokens (default 16,384).

## Evaluation

On all 231 public JevBench cases, CLEF-HRX scored **202/231 (87.4%)**, compared
with Jev 1.13's published **200/231 (86.6%)** on the same cases. All requests
completed with valid probabilities and no input truncation.

On AMD Strix Halo, resident latency was **0.91 s median / 5.63 s p95**, with
**402 input tokens/s (33.3 decisions/minute)** throughput. Another GPU application
was present and the compiler cache was populated during this single sequential
run; model loading is excluded from these speed figures.
[Results, measurement details and reproduction instructions](docs/evaluation.md).

## Library

`ClefModel` exposes inference and `load_in` accepts an existing HRX `ModelContext`.
`Encoder` can prepare schemas and text without loading models or opening a GPU.
[Requests, media inputs and Rust example](docs/usage.md).

## Development

```sh
cargo test --locked --lib --bins --tests --examples --all-features
cargo test --locked --doc
cargo bench --locked --bench preprocessing
python3 scripts/bench_workload.py baseline --profile quick
CLEF_BENCH_FULL_MODEL=1 cargo bench --locked --bench inference
```

GPU and checkpoint tests are ignored by default. Full-model benchmarks report
loading and first-request time separately from resident requests.
[Validation, kernel benchmarks and profiling](docs/development.md).
[Workload benchmarks, recorded comparisons and evaluation gate](docs/optimization.md).

## License

[Apache-2.0](LICENSE). [Model and kernel attribution](THIRD_PARTY_NOTICES.md).
