# clef-hrx

Run [CLEF](https://huggingface.co/Cloudflare/clef),
[CLEF Flash](https://huggingface.co/Cloudflare/clef-flash), and
[CLEF Omni](https://huggingface.co/Cloudflare/clef-omni) locally in Rust on AMD
Strix Halo, using [HRX](https://github.com/zacharydenton/hrx-rs).

All three answer structured questions about text, images and video. Omni also
supports audio and video soundtracks. Outputs are choices, scores and
probabilities.

## Models and performance

Measured on a Radeon 8060S (`gfx1151`) with 128 GiB unified memory.
JevBench covers all 231 cases; WorkflowEvals covers all 705 cases.
Speed is input tokens per second with the model loaded, including encoding,
preparation, GPU execution and readback. Flash JevBench uses a warm kernel cache;
WorkflowEvals timings include shared-GPU runs.

| Model | Precision | JevBench accuracy | JevBench tok/s | WorkflowEvals tok/s |
| --- | --- | ---: | ---: | ---: |
| CLEF | BF16 | 87.4% | 616 | 548 |
| CLEF | INT4 MLP | 88.3% | 847 | — |
| CLEF | INT4 | 84.0% | 1,032 | — |
| Flash | BF16 | 82.7% | 1,794 | 1,557 |
| Flash | INT4 MLP | 79.2% | 2,261 | 599 |
| Flash | INT4 | 71.4% | 2,523 | 1,848 |
| Omni | BF16 | 80.1% | 1,145 | — |
| Omni | INT4 | 81.0% | 3,031 | 1,194 |

BF16 is the default. INT4 MLP quantizes only MLP projections; INT4 also quantizes
attention projections. A dash means no complete measurement for the current kernels.

Results and reproduction: [CLEF](docs/evaluation.md),
[Flash](docs/clef-flash.md), [Omni](docs/omni-memory-tuning.md),
[WorkflowEvals](docs/workflowevals.md), [llama.cpp comparison](docs/llama-jevbench.md).

## Compared with Cloudflare

WorkflowEvals agreement with consensus reference actions, in percent.
Each cell is **local / Cloudflare published**. Local CLEF and Flash use BF16;
local Omni uses INT4, compared with Cloudflare's unquantized model.
Published scores are from [Cloudflare's model cards](https://huggingface.co/Cloudflare/clef-omni#workflow-evals).

| Metric | CLEF BF16 | Flash BF16 | Omni INT4 |
| --- | ---: | ---: | ---: |
| Invoice exact actions | 64.7 / 64.7 | 57.1 / 57.1 | 57.3 / 60.2 |
| Invoice primary action | 86.4 / 86.2 | 73.6 / 73.3 | 79.3 / 82.0 |
| Customer service exact actions | 76.5 / 76.3 | 76.6 / 77.0 | 69.0 / 71.6 |
| Security exact actions | 62.9 / 62.9 | 61.3 / 61.7 | 56.7 / 61.7 |
| Agent trace primary action | 68.5 / 68.5 | 70.3 / 69.8 | 63.1 / 65.8 |

CLEF BF16 matches or slightly exceeds all five published scores. Flash BF16 is
within 0.5 percentage points; Omni INT4 is 2.6–5.0 points below published Omni.

Cloudflare's [hosted CLEF measurements](https://blog.cloudflare.com/clef-faster-cheaper-multimodal/#clef-model-is-now-faster)
report median latencies of 152 ms at ~800 tokens, 305 ms at ~3,400, and 1,635 ms
at ~16,000. Dividing input size by latency gives approximately **5,300–11,100
input tok/s**, versus **548–616** for local CLEF BF16 across our suites.
These use different workloads and hardware, so they are not a matched speed test.

## Build and run

Requires Linux, Rust 1.91+ and a `gfx1151` GPU. Original CLEF BF16 needs at least
50 GiB available RAM for short requests. Audio and video require ffmpeg and ffprobe.

```sh
git clone https://github.com/zacharydenton/clef-hrx.git
cd clef-hrx
cargo build --release --locked
./target/release/clef --max-length 512 decide examples/invoice.json
```

Select `--model clef`, `--model clef-flash` or `--model clef-omni`, and
`--precision bf16`, `--precision int4-mlp` or `--precision int4`.
Defaults are CLEF, BF16 and a 65,536-token context limit. Add `--jsonl` to process
multiple requests with the model loaded, or `--raw` for probabilities and timings.
For repeated Omni image/state inputs, add [`--prefix-cache`](docs/usage.md#omni-prefix-caching).

[CLI and Rust API](docs/usage.md) · [Omni media inputs](docs/clef-omni.md) ·
[Development](docs/development.md) · [Kernel benchmarks](docs/optimization.md)

## License

[Apache-2.0](LICENSE). [Model and kernel attribution](THIRD_PARTY_NOTICES.md).
