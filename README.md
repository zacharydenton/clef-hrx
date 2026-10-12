# clef-hrx

[Cloudflare CLEF](https://huggingface.co/Cloudflare/clef) and
[CLEF Flash](https://huggingface.co/Cloudflare/clef-flash) inference in Rust for
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

Requires Rust 1.91+, Linux and a `gfx1151` GPU. Original CLEF BF16 needs at least 50 GiB of
available RAM for short requests. Video input needs ffmpeg and ffprobe.

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

Select Flash with `--model clef-flash`; original CLEF remains the default.
Use `--model-dir PATH` for a local snapshot or `--offline` for cached files.
First use downloads weights and creates a tiled projection cache alongside them.
Existing converted checkpoints remain readable.

`--max-length` sets the token limit (default 65,536); `--strict-context` rejects
requests that would truncate state. `--memory-gib` sets the allocation budget
(default 80 GiB). See [Flash qualification and reproduction](docs/clef-flash.md).

## Precision

BF16 is the default. Use `--precision` to select INT4:

```sh
./target/release/clef --precision int4-mlp decide examples/invoice.json
./target/release/clef --precision int4 decide examples/invoice.json
```

See [precision options](docs/usage.md#precision).

## Evaluation

Original CLEF on full JevBench, AMD Strix Halo, with the model already loaded:

| Runtime / mode | Full JevBench accuracy | Resident input tok/s |
| --- | ---: | ---: |
| clef-hrx BF16 | 202/231 (87.4%) | 616 |
| llama.cpp HIP BF16 | 203/231 (87.9%) | 343 |
| clef-hrx INT4 MLP only | 204/231 (88.3%) | 847 |
| llama.cpp HIP Q4_K_M | 204/231 (88.3%) | 335–351 |
| clef-hrx INT4 backbone | 194/231 (84.0%) | 1,032 |

BF16 is **1.79× faster** than llama.cpp HIP BF16. MLP-only INT4 is
**2.42–2.53× faster** than llama.cpp HIP Q4_K_M, with both scoring 204/231.
[llama.cpp comparison and reproduction](docs/llama-jevbench.md).

[Results and reproduction](docs/evaluation.md#tiled-weight-layout).

CLEF Flash with the model already loaded and a warm kernel cache for JevBench:

| Mode | Full JevBench accuracy | JevBench input tok/s | WorkflowEvals input tok/s |
| --- | ---: | ---: | ---: |
| BF16 | 191/231 (82.7%) | 1,794 | 531 |
| INT4 MLP only | 183/231 (79.2%) | 2,261 | 599 |
| INT4 backbone | 165/231 (71.4%) | 2,523 | 596 |

[Flash results and reproduction](docs/clef-flash.md).
WorkflowEvals rates cover [all 705 cases](docs/workflowevals.md#flash-full-suite-results)
with the model already loaded; original CLEF runs are in progress.

An experimental original CLEF BF16 kernel reaches **539 input tok/s** on a complete
30,342-token invoice with a shared GPU. Full WorkflowEvals qualification is running.
[Long-context results and reproduction](docs/long-context-bf16.md).

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

GPU and checkpoint tests are ignored by default.
See [development](docs/development.md) and [kernel benchmarks](docs/optimization.md).

## License

[Apache-2.0](LICENSE). [Model and kernel attribution](THIRD_PARTY_NOTICES.md).
