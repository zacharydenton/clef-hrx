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

Requires Rust 1.91+, Linux and a `gfx1151` GPU. BF16 needs at least 50 GiB of
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

First use downloads roughly 51.2 GiB of weights. Use `--model-dir PATH` for a
local snapshot or `--offline` for cached files. Inputs support up to 65,536 tokens,
subject to available memory. `--max-length` sets the token limit (default 65,536);
`--memory-gib` sets the allocation budget (default 80 GiB).

## Precision

BF16 is the default. Use `--precision` to select INT4:

```sh
./target/release/clef --precision int4-mlp decide examples/invoice.json
./target/release/clef --precision int4 decide examples/invoice.json
```

See [precision options](docs/usage.md#precision).

## Evaluation

Full JevBench results on AMD Strix Halo, with the model already loaded:

| Mode | Full JevBench accuracy | Resident input tok/s |
| --- | ---: | ---: |
| BF16 | 202/231 (87.4%) | 519 |
| INT4 MLP only | 204/231 (88.3%) | 747 |
| INT4 backbone | 194/231 (84.0%) | 1,003–1,035 |

[Results and reproduction](docs/evaluation.md#parallel-reduction-and-input-normalization).
[WorkflowEvals](docs/workflowevals.md) is still in progress.

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
