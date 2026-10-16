# CLEF Omni

Optimized attention and expert kernels are enabled by default in ordinary
builds. No benchmark flags or local kernel files are required. See
[performance results and configuration](omni-memory-tuning.md#default-configuration).

CLEF Omni uses the Qwen3-Omni thinker: 48 text layers, 128 experts with eight
selected per token, a vision encoder, an audio encoder, and Cloudflare's trained
decision head. The runtime is native Rust/HRX. Python is used only to generate
reference results from [Cloudflare's release](https://huggingface.co/Cloudflare/clef-omni).

The pinned revision is `0db1cd2607d76a7bdb2a382f659e7b313079f84b`.
Speech synthesis weights are excluded from resident storage. BF16 remains the
default; `--precision int4-mlp` packs expert projections and `--precision int4`
also packs eligible text attention projections. Vision, audio, routing and the
decision head retain BF16 weights.

BF16 resident weights require 58.12 GiB, plus host-backed embeddings and runtime
workspace. Allow additional RAM for the operating system and other applications.

## Run

```sh
cargo build --release --locked
./target/release/clef --model clef-omni --precision int4 \
  --max-length 1536 decide examples/omni-audio.json --raw
```

Replace the example's `recording.wav` with a local file. `images`, `audio` and
`videos` accept local paths. Audio is decoded to 16 kHz mono with ffmpeg. Videos
are sampled at two frames per second, matching the release. When every attached
video has an audio track, its audio tokens are interleaved with its visual tokens;
otherwise the release processes the videos without their soundtracks. Explicit
`audio` attachments are still processed.

`--model-dir` selects a local pinned snapshot; `--offline` uses cached files.
`--jsonl` keeps the model loaded across requests. The input token limit includes
all media and the question schema. `--strict-context` rejects state truncation.
Media that exceed the context budget are rejected.

Omni uses its own attention, expert routing, RoPE and vision feature injection.
The long-context kernel selections qualified for original CLEF and Flash do not
apply to Omni. Expert projections use grouped BF16/INT4 matrix kernels; see
[operator benchmarks and reproduction](omni-kernel-tuning.md).

## Qualification (October 15, 2026)

Seven short requests cover text, red and blue images, silent video, audio,
synchronized audio/video, and a spoken-word choice. INT4 agrees with the BF16
Cloudflare oracle on the highest-probability option for all nine questions;
repeating every request produces identical logits and probabilities. The wider-load
HRX 0.11 INT4 kernels complete full JevBench at **186/231 correct (80.52%)** and
**619.8 resident input tok/s**. BF16 scores **184/231 (79.65%)** at **414.2 tok/s**.
WorkflowEvals qualification remains pending. See
[the current experiments and reproduction](omni-memory-tuning.md).

The current HRX 0.11 build retains all nine INT4 choices. Its compiler changes
probabilities by up to 0.08994 versus 0.10.1. BF16 text and audio/video oracle
errors are 0.000622 and 0.005157, with unchanged top choices. See
[upgrade validation and timings](hrx-011.md#validation-and-performance).

The following table records the earlier HRX 0.10.1 grouped-kernel qualification:

| Input | Tokens | Initial INT4 warm input tok/s | Grouped INT4 warm input tok/s | Maximum probability difference from BF16 oracle |
| --- | ---: | ---: | ---: | ---: |
| Text, three questions | 258 | 79.6 | 720.1 | 0.0082 |
| Red image | 216 | 74.7 | 642.3 | 0.0027 |
| Silent video | 573 | 74.0 | 462.1 | 0.2118 |
| Audio | 160 | 76.2 | 482.6 | 0.0007 |
| Video with audio | 306 | 74.5 | 421.0 | 0.0494 |
| Blue image choice | 226 | 77.6 | 675.2 | 0.0022 |
| Spoken-word choice | 182 | 76.2 | 500.0 | 0.0022 |

Grouped expert kernels raise warm aggregate throughput from **75.7 to 520.3 input
tok/s (6.88×)**, with all 14 outputs exactly unchanged from the initial INT4 run.
These rates include encoding, preparation, GPU execution and readback on the
second request of each identical pair; model loading is excluded. Including both
runs of every pair gives 461.8 resident input tok/s, versus 66.4 initially.
Peak engine allocation remains 17.7 GiB. Measurements used a shared Strix Halo
GPU. The video probability shift from the BF16 oracle remains material even
though the selected answer is unchanged.
[Initial results](results/omni-20261015.json),
[grouped results](results/omni-grouped-experts-20261015.json), and
[kernel benchmarks and reproduction](omni-kernel-tuning.md).

That BF16 build also passed the text and synchronized audio/video oracle checks,
covering all three question types and both media encoders. Maximum probability
differences are 0.00242 and 0.00901, respectively, with unchanged highest-ranked
options. This validation loads one layer at a time and peaks at 2.26 GiB of
engine allocations. A resident BF16 attempt was stopped by the memory guard
before inference as another workload consumed RAM. The later full JevBench
measurement above supersedes that attempt.

## Verification and reproduction

The reference code is pinned by SHA-256:
`21d05ebb8cbea26a26af65eca3d16cf4b69bf80a320e2d5b203e86f0d632a670`.
The fixture manifest records the actual Torch and Transformers versions.

Checked-in fixtures cover text, images, video, audio, synchronized audio/video,
log-mel preprocessing, convolution, 128-wide grouped-query attention, RMSNorm and
RoPE. CPU tests use the fixtures without Python dependencies. GPU tests also
check selected expert reads, INT4 packing, weighted accumulation and replay.

```sh
cargo test --locked --lib
CLEF_OMNI_MODEL_DIR=artifacts/model-omni \
  cargo test --locked --lib omni_media::oracle_tests -- --ignored
python3 scripts/memory_guard.py \
  --output artifacts/omni-operator-check --memory-gib 8 -- \
  cargo test --locked --lib omni::tests -- \
    --ignored --nocapture --test-threads=1
```

Generate independent fixtures in a fresh directory with an oracle environment
containing Torch, Transformers 5.10.2, huggingface-hub 1.x, safetensors, NumPy,
Pillow and PyAV. Use a ROCm Torch build for GPU reference inference. ffmpeg and
ffprobe must be available for the media fixtures.

```sh
python scripts/generate_omni_reference.py \
  --model-dir artifacts/model-omni --output artifacts/omni-reference-fixtures
```

A full BF16 reference can be run without keeping all thinker weights resident.
The streaming oracle calls Cloudflare's unmodified forward implementation;
loading hooks materialize one backbone block at a time. Its elapsed time includes
weight transfers and is not an inference speed baseline.

```sh
python3 scripts/memory_guard.py \
  --output artifacts/omni-reference-guard --memory-gib 8 -- \
  python scripts/omni_streaming_oracle.py \
    --model-dir artifacts/model-omni --output artifacts/omni-reference-decisions \
    tests/fixtures/omni/encoding-0.json \
    tests/fixtures/omni/encoding-1.json \
    tests/fixtures/omni/encoding-2.json \
    tests/fixtures/omni/encoding-3.json \
    tests/fixtures/omni/encoding-4.json \
    tests/fixtures/omni/blue-request.json \
    tests/fixtures/omni/speech-request.json
```

Compare native outputs with those reference decisions and measure each request
twice in one resident process:

```sh
python3 scripts/compare_omni_reference.py \
  --reference artifacts/omni-reference-decisions/{0,1,2,3,4,5,6}.json \
  --write-requests artifacts/omni-requests.jsonl
python3 scripts/memory_guard.py \
  --output artifacts/omni-native-guard --memory-gib 24 -- \
  ./target/release/clef --model clef-omni --model-dir artifacts/model-omni \
    --precision int4 --memory-gib 24 --max-length 1536 \
    decide artifacts/omni-requests.jsonl --jsonl --raw
python3 scripts/compare_omni_reference.py \
  --reference artifacts/omni-reference-decisions/{0,1,2,3,4,5,6}.json \
  --native artifacts/omni-native-guard/run.log
```

The comparison reports answer differences and maximum probability error. Its
resident throughput includes encoding, graph preparation, GPU execution and
readback; model loading is excluded. Repeated requests reuse the graph.
The spoken-word fixture and its eSpeak NG generation command are checked in with
a SHA-256 manifest, so running these checks does not require a speech synthesizer.

The native BF16 validation test also supports bounded weight residency. It uses
the production operators, releasing each layer before loading the next and
preserving BF16 activations exactly between layers. By default it checks text
and synchronized audio/video against the saved oracle results. Its timing
includes loading and is not resident inference throughput. Consumed checkpoint
pages are discarded from the filesystem cache after each layer.

After the resident INT4 run has created the tiled cache, prepare a local view of
it for repeated validation. This avoids rehashing all pristine shards each time:

```sh
python3 - <<'PY'
import json
from pathlib import Path
source = Path('artifacts/model-omni').resolve()
revision = '0db1cd2607d76a7bdb2a382f659e7b313079f84b'
cache = source / '.clef-hrx' / 'clef-omni' / revision / 'tiled256x64'
target = Path('artifacts/model-omni-tiled')
target.mkdir(exist_ok=True)
index = json.loads((source / 'model.safetensors.index.json').read_text())
shards = {v for k, v in index['weight_map'].items() if k.startswith('thinker.')}
metadata = {'config.json', 'joint_head_config.json', 'processor_config.json',
            'tokenizer.json', 'model.safetensors.index.json', 'joint_head.safetensors'}
for name in sorted(shards | metadata):
    path = (cache if name in shards else source) / name
    assert path.is_file(), f'Complete the native model load first: {path}'
    link = target / name
    if not link.exists():
        link.symlink_to(path)
PY
```

```sh
CLEF_OMNI_MODEL_DIR=artifacts/model-omni-tiled \
CLEF_OMNI_BF16_OUTPUT=artifacts/omni-bf16-decisions \
python3 scripts/memory_guard.py \
  --output artifacts/omni-bf16-guard --memory-gib 8 -- \
  cargo test --release --locked --lib \
    omni::validation::streaming_bf16_matches_release -- --ignored --exact --nocapture
```

The guard requires its stated budget plus 32 GiB available before launch and
stops its own process group below a 24 GiB live reserve or after more than 1 GiB
additional swap. Choose the native model's allocation budget separately; an
8 GiB guard budget is for operator tests and the streaming oracle, not resident
Omni inference.
