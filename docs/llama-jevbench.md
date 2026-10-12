# CLEF through llama.cpp on JevBench

This comparison uses the published [ggml-org/Clef-GGUF](https://huggingface.co/ggml-org/Clef-GGUF)
models on the same Ryzen AI MAX+ 395 / Radeon 8060S machine as clef-hrx.

The longer-input comparison has a separate [WorkflowEvals runner and reproduction guide](llama-workflowevals.md).

## Results

Measured on 11 October 2026. Each resident measurement covers the same 166,394
input tokens across 230 requests; accuracy covers all 231 cases.

| Runtime / mode | Correct / 231 | Resident input tok/s | Resident wall seconds |
| --- | ---: | ---: | ---: |
| clef-hrx BF16 | 202 | 615.78 | 270.2 |
| llama.cpp HIP BF16 | 203 | 343.27 | 484.7 |
| clef-hrx MLP-only INT4 | 204 | 846.83 | 196.5 |
| llama.cpp HIP Q4_K_M | 204 | 334.84 | 496.9 |
| llama.cpp HIP Q4_K_M repeat | 204 | 350.62 | 474.6 |
| clef-hrx broader INT4 | 194 | 1,032.14 | 161.2 |

HRX BF16 is **1.79× faster** than llama.cpp BF16. MLP-only INT4 is
**2.42–2.53× faster** than llama.cpp Q4_K_M at the same 204/231 accuracy count.
Broader INT4 is **2.94–3.08× faster** than Q4_K_M, with ten fewer correct cases.
The lower speedup in each range uses the faster Q4_K_M repeat as its baseline.
These figures cover the pinned HIP configuration below.

All three completed llama.cpp runs produced 231/231 strictly valid responses,
with identical requests and input-token counts to HRX and no truncation.
Server logs confirm 65/65 layers offloaded, flash attention enabled, and zero
cached tokens for every request. No other KFD compute clients appeared in the
telemetry. Q4_K_M reproduced every response exactly on its second run; its
throughput increased by 4.7%. BF16 was measured in one complete run.

The BF16 implementations agree on 230/231 predictions. The only difference is
`hard-sol-c-judge_hard-13`, an invoice-rounding judgment: llama.cpp assigns the
correct answer 50.14%, while HRX assigns it 49.28%. Across all cases the maximum
absolute probability difference is 0.03419. Equal accuracy counts for Q4_K_M
and HRX MLP-only INT4 also do not mean identical predictions: they agree on
226/231, with two wins each and one disagreement where both are wrong.

HRX figures come from the [tiled-weight qualification](evaluation.md#tiled-weight-layout).
The archived HRX binary is `artifacts/throughput-1000/clef-tiled-weights`, SHA-256
`05207c8f415ccd7bd11320094b0346f84f1096428217480e623be88c9c0637d9`.
The llama-server executable is build 11552 (`0.6.0-dev`), SHA-256
`89445c2daf7746f0487cc7a4d78548bb251fe01e989dc31521befa9480778d5c`;
the binary and linked-library hashes match across all three runs.

Raw llama.cpp results are retained locally under
`tests/results/jevbench-20261011-llama-{bf16-full,q4,q4-repeat}/`.
HRX references are `tests/results/jevbench-20261011-bf16-tiled-repeat/`,
`tests/results/jevbench-20261011-int4-mlp-tiled/` and
`tests/results/jevbench-20261011-int4-tiled/`. The comparison audit, including
artifact hashes and exact-response checks, is
`artifacts/llama-jevbench/comparison.json`.

## Method

The runner sends all 231 public JevBench cases sequentially, in easy, original,
hard order, to llama.cpp's native `/v1/systemone` decision endpoint. The upstream
TypeSafe adapter and scorer handle choice, score and noul questions. Accuracy
includes every case; score questions use argmax, as in the HRX evaluation.

Throughput is the sum of input tokens divided by the sum of request wall times,
excluding only the first request to match the HRX resident comparison. It
includes request construction, JSON serialization, loopback HTTP, tokenization,
GPU execution and response parsing. Server startup is recorded separately.
These are input tokens, not generated tokens. There is no text decoding.

The server uses one slot, full GPU offload, flash attention, and 4,096-token
context, batch and microbatch limits. CLEF's joint decision graph processes the
whole request in one microbatch and has no persistent KV memory; no prompt
prefixes are reused. The full suite's longest HRX input is 4,032 tokens. The
benchmark's allocation does not change clef-hrx's 65,536-token default.

Model and executable hashing happen before server startup. The runner saves
raw responses, scores, requests, token counts, server logs, source/build
metadata, linked-library hashes and one-second GPU telemetry. Failed requests
remain in the accuracy denominator and suppress a throughput claim. An optional
HRX reference checks request equality and reports every input-token mismatch.

## Pinned inputs

- JevBench: `7625ecb1b075cfe4d09d34141d353325825eea1d`.
- llama.cpp: `23b0202a189c44a54625aadcb37a946dd1d6278d`, Release HIP build for
  `gfx1151`, ROCm 7.2.4; HIP graphs enabled, rocWMMA flash attention disabled.
- GGUF repository: `d2aa75cb425ae922b5c3c15ef3157d417b7f7641`.
- `Clef-BF16.gguf`: SHA-256
  `b23c810a85179244e2b2e722021bad950b4284e3595741b37820aa86e768071e`.
- `Clef-Q4_K_M.gguf`: SHA-256
  `9997675e4c9e9016273372b5fdedbdda7ad14285bc722949e4b9daca62bf3822`.

The GGUF repository's `.src_sha` records Cloudflare/clef revision
`0b331204bb13fbd2ca93a64df1956f5b55478ce5`. Comparing Hugging Face file hashes
against HRX's `2f3de3dd85f379784083b0814d997ab627200f0c` finds only a README
change: source weights and configuration are identical. Q4_K_M and HRX INT4
use different quantization schemes; compare their measured accuracy alongside
speed. This text-only evaluation does not load the GGUF vision projector.

## Reproduce

The runner requires Python 3.11+ and uses the standard library plus the pinned
JevBench checkout. Downloading the models additionally uses `huggingface_hub`.
Use a separate llama.cpp checkout if your existing checkout has local changes.

```sh
git clone https://github.com/ggml-org/llama.cpp artifacts/llama.cpp
git -C artifacts/llama.cpp checkout 23b0202a189c44a54625aadcb37a946dd1d6278d
cmake -S artifacts/llama.cpp -B artifacts/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_ROCWMMA_FATTN=OFF
cmake --build artifacts/llama.cpp/build --target llama-server -j 12

git clone https://github.com/fstandhartinger/jevbench artifacts/jevbench
git -C artifacts/jevbench checkout 7625ecb1b075cfe4d09d34141d353325825eea1d
```

Download the pinned published files with `huggingface_hub` installed in your
Python environment. They occupy about 73 GB together; the source model is not
needed by the llama.cpp runner.

```sh
python - <<'PY'
import json
from pathlib import Path
from urllib.request import urlopen
from huggingface_hub import hf_hub_download

repo = 'ggml-org/Clef-GGUF'
revision = 'd2aa75cb425ae922b5c3c15ef3157d417b7f7641'
root = Path('artifacts/llama-jevbench/gguf')
root.mkdir(parents=True, exist_ok=True)
with urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true') as response:
    manifest = json.load(response)
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
for name in ('.src_sha', 'Clef-BF16.gguf', 'Clef-Q4_K_M.gguf'):
    hf_hub_download(repo, name, revision=revision, local_dir=root)
PY
```

Wait for builds and downloads to finish, and run with the GPU otherwise idle.
Run each model separately, using a new output directory for every repeat:

```sh
python scripts/eval_jevbench_llama.py \
  --binary artifacts/llama.cpp/build/bin/llama-server \
  --llama-source artifacts/llama.cpp \
  --model artifacts/llama-jevbench/gguf/Clef-BF16.gguf \
  --manifest artifacts/llama-jevbench/gguf/manifest.json \
  --output tests/results/jevbench-llama-bf16

python scripts/eval_jevbench_llama.py \
  --binary artifacts/llama.cpp/build/bin/llama-server \
  --llama-source artifacts/llama.cpp \
  --model artifacts/llama-jevbench/gguf/Clef-Q4_K_M.gguf \
  --manifest artifacts/llama-jevbench/gguf/manifest.json \
  --output tests/results/jevbench-llama-q4
```

Add `--reference PATH` to compare against an HRX result directory produced by
[`eval_jevbench.py`](../scripts/eval_jevbench.py). Add `--smoke --reference PATH`
for a separate first/longest-case compatibility check. Neither downloads nor
runs require API credentials. Results are stored under the ignored
`tests/results/` directory.
