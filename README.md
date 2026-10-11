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
80 GiB); `--max-length` limits input tokens (default and maximum 65,536).
The checkpoint supports 262,144 positions; this build uses a 64k operating limit
without changing RoPE. Workspace is allocated for each actual request, with
8,192-row MLP chunks above 16k tokens. Available RAM and the allocation budget
still determine whether a request fits.

## Precision

BF16 is the default. Normal builds also support explicit INT4 modes:

```sh
./target/release/clef --precision int4-mlp decide examples/invoice.json
./target/release/clef --precision int4 decide examples/invoice.json
```

| Mode | Full JevBench accuracy | Resident input tok/s |
| --- | ---: | ---: |
| BF16 | 202/231 (87.4%) | 525 |
| INT4 MLP only | 199/231 (86.1%) | 625 |
| INT4 eligible backbone projections | 194/231 (84.0%) | 907 |

INT4 trades accuracy for speed; it is never selected automatically. These are
archived full-suite measurements of the respective kernel configurations. The
BF16 and broader INT4 results include fused SwiGLU, vectorized DeltaNet and
cached kernel-source fingerprints; broader INT4 also uses single-pass packing.
The MLP-only result is from an earlier build. Broader INT4
uses 14.51 GiB peak engine allocation and preserves all outputs of its previous
run. See [qualification and reproduction](docs/evaluation.md#kernel-source-fingerprint-cache),
[precision details](docs/usage.md#precision)
and [recorded experiments](docs/optimization.md). Vision, normalization and the
trained decision head retain their original precision.

## Evaluation

The [WorkflowEvals runner and reproduction notes](docs/workflowevals.md) cover
all four workflows in the pinned public collection. The first resumable batch
completed 4/705 cases, including a 36,916-token request. Full results are pending.

On all 231 public JevBench cases, CLEF-HRX scored **202/231 (87.4%)**, compared
with Jev 1.13's published **200/231 (86.6%)** on the same cases. All requests
completed with valid probabilities and no input truncation.

On AMD Strix Halo, BF16 resident latency was **0.66 s median / 4.42 s p95**, with
**525 input tokens/s (43.5 decisions/minute)** throughput. This single sequential
run used a populated compiler cache; the first request including model loading
is excluded from these speed figures.
[Results, measurement details and reproduction instructions](docs/evaluation.md#kernel-source-fingerprint-cache).

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
