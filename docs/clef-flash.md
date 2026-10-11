# CLEF Flash qualification

Select the pinned Flash checkpoint with `--model clef-flash`. Original CLEF and
BF16 remain the defaults. Flash has 32 text layers, hidden width 4,096, 16 full
attention heads and 32 DeltaNet value heads. The tokenizer, processor and decision
head implementation match original CLEF; the vision output and head input
projections use Flash's hidden width.

Pinned release: [Cloudflare/clef-flash at 8b2e5fd](https://huggingface.co/Cloudflare/clef-flash/tree/8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3).
The native limit remains 65,536 total encoded tokens, including schema and media.

## Status

BF16 passes the eight-case independent text/image/video corpus with identical
winners, maximum probability error 0.002057 (limit 0.003), and exact changed-input replay. Reference
fixtures and their environment manifest are in `tests/fixtures/flash/`.
All three Flash JevBench runs and their warm-cache repeats are complete. The
six-configuration context/workflow matrix is running; full qualification remains
pending.

## JevBench results

The complete 231-case public suite on gfx1151, using the same binary and input
lengths in every mode:

| Precision | Correct | First run tok/s | Warm repeat tok/s | Peak engine GiB |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 191/231 | 1,165.68 | 1,793.70 | 14.91 |
| int4-mlp | 183/231 | 2,215.67 | 2,260.76 | 8.23 |
| int4 | 165/231 | 2,518.98 | 2,523.02 | 5.50 |

Relative to Flash BF16, MLP INT4 gains three correct cases and loses eleven;
broader INT4 gains thirteen and loses thirty-nine. Maximum probability changes
are 0.7532 and 0.9068, respectively. These are explicit quantization tradeoffs,
not numerical parity claims.

All responses were strictly valid. Every probability was identical on repeat,
and telemetry recorded at most one KFD client in each full run. Throughput counts
resident input tokens, including encoding, preparation, compilation, GPU execution
and readback. Only the first request containing model loading is excluded. The
first BF16 run includes first-use compilation for previously unseen Flash shapes;
the README uses the warm repeats. This is a cache effect with the same executable,
not a kernel optimization.

Implementation: `787b2bb`; evaluation tooling: `891353c`.
Frozen executable SHA256:
`3a430b14b418cb01c9f87aa9472848ee18a8f1ff773d2c8f745f9fa9521dd0d2`.
Raw results: `tests/results/flash-support-baseline/clef-flash-{bf16,int4-mlp,int4}`
and `tests/results/flash-support-repeat/clef-flash-{bf16,int4-mlp,int4}`.
Same-mode comparisons are in `artifacts/flash-qualification/flash-*-repeat-comparison.json`.


## Loading and storage

```sh
cargo build --release --locked
./target/release/clef --model clef-flash inspect
./target/release/clef --model clef-flash --strict-context decide examples/invoice.json --raw
```

Use `--model-dir artifacts/model-flash` for a local snapshot. The directory must
match the selected model's pinned configuration and tensor metadata. Flash weights
occupy approximately 19.1 GB. First load creates approximately 18.8 GB of derived
projection shards, plus temporary space for the largest shard. Source weights
remain intact. Derived files are keyed by model, revision, layout and source SHA256.
Previously converted original CLEF snapshots remain supported.

`--strict-context` rejects any state truncation, including that caused by
`--max-state-tokens`. Without it, existing truncation behavior remains. An exact
65,536-token request is accepted; larger requests are rejected in strict mode.
A context limit specifies capacity, not an allocation request.

## Independent reference

Use pristine Flash weights and the release's `joint_schema_model.py`. Run reference
and native GPU processes sequentially. The generator records package versions and
the release code hash; it does not use native predictions to construct references.

The following creates an isolated environment over the host's working ROCm
PyTorch/torchvision installation. Install the extra reference dependencies without
replacing that GPU stack:

```sh
uv venv --python /usr/bin/python3 --system-site-packages artifacts/flash-reference-py314
uv pip install --python artifacts/flash-reference-py314/bin/python \
  'transformers==5.10.2' 'tokenizers==0.22.2'
uv pip install --python artifacts/flash-reference-py314/bin/python --no-deps \
  'accelerate==1.15.0' 'flash-linear-attention==0.5.2' 'fla-core==0.5.2' \
  'av==19.0.1' 'sentencepiece==0.2.2'
artifacts/flash-reference-py314/bin/hf download Cloudflare/clef-flash \
  --revision 8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3 --local-dir artifacts/model-flash
artifacts/flash-reference-py314/bin/python scripts/generate_flash_reference.py \
  --model-dir artifacts/model-flash --output artifacts/flash-qualification/reference
```

The reference environment uses Transformers 5.10.2, tokenizers 0.22.2, Accelerate,
FLA 0.5.2, PyAV 19.0.1, and the host's ROCm PyTorch 2.14.0. The reference uses
[FLA's upstream GPU operators](https://github.com/fla-org/flash-linear-attention)
through Transformers. Video frames are decoded with PyAV and passed to the
unchanged processor with frame indices and FPS. Run the native comparison and CPU encoding boundary test:

```sh
CLEF_MODEL_DIR=artifacts/model-flash CLEF_TEST_MODEL=clef-flash \
  CLEF_REFERENCE_DIR=tests/fixtures/flash \
  cargo test --release --lib full_checkpoint_corpus -- --ignored --nocapture
```

```sh
CLEF_MODEL_DIR=artifacts/model cargo test --lib strict_context_preserves_exact_fit -- --ignored
```

Check all three precision modes against this corpus, including changed-input and
identical-input replay:

```sh
python3 scripts/qualify_corpus.py --model clef-flash --model-dir artifacts/model-flash \
  --reference tests/fixtures/flash --output tests/results/flash-multimodal-all-modes
```

The BF16 reference tolerance stays 0.003. Initial INT4 differences are reported;
all modes must produce valid probabilities and exact same-mode replay. The full
corpus passed in every mode: maximum errors were 0.002057 BF16, 0.066974 MLP INT4
and 0.261284 broader INT4, with unchanged winners and exact replay in all three.
These eight cases complement the larger JevBench accuracy results above.

## Evaluations

Run each command separately for `bf16`, `int4-mlp` and `int4`, with a distinct
output directory. Repeat with `--model clef --model-dir artifacts/model` for the
original model. WorkflowEvals uses all 705 cases unless explicitly limited.

```sh
python3 scripts/eval_jevbench.py --model clef-flash --model-dir artifacts/model-flash \
  --precision bf16 --memory-gib 80 --output tests/results/flash-jevbench-bf16
artifacts/WorkflowEvals/.venv/bin/python scripts/eval_workflowevals.py \
  --model clef-flash --model-dir artifacts/model-flash --precision bf16 \
  --memory-gib 80 --output tests/results/flash-workflows-bf16
```

Both runners use strict context handling and record the model revision. Workflow
results measure agreement with generated references, not human-labelled accuracy.
Resident native input tok/s and complete workflow wall time are separate metrics.
Resume only with the same binary, runner, settings and pinned cases.

Initial Flash INT4 results have no fixed accuracy cutoff: report their tradeoff
against Flash BF16. Subsequent kernel changes must preserve each mode's qualified
evaluation baseline. Original CLEF floors remain 202/231 BF16, 204/231 MLP INT4,
and 194/231 broader INT4 on JevBench.

## Benchmarks

```sh
python3 scripts/bench_workload.py flash-smoke --suite flash --test
CLEF_BENCH_LENGTHS=235,620,4032 python3 scripts/bench_workload.py flash-baseline --suite flash
CLEF_BENCH_MODEL=clef-flash CLEF_MODEL_DIR=artifacts/model-flash \
  CLEF_BENCH_FULL_MODEL=1 CLEF_BENCH_REQUESTS=tests/results/flash-jevbench-bf16/requests.jsonl \
  python3 scripts/bench_workload.py flash-inference --suite workload_inference
```

The Flash suite covers all projection shapes with repeated and rotating weights,
SwiGLU, attention and persistent DeltaNet. It uses 99% intervals and a 3% noise
threshold. Explicit lengths above 4,032 exercise attention and DeltaNet only, up
to 65,536. Operator timings exclude allocation, compilation and upload; full-model
evaluations include request preparation and compilation. GEMM operation rates are
not model token throughput. Backbone GEMMs use the production tiled layout;
small gates and decision-head projections retain row-major weights.

The full GPU smoke sweep passed 418 cases at all eleven default lengths from
1 through 4,032 tokens, including packing width 12,288 through the production
selector. Artifacts: `benches/results/flash-support-full-smoke`. Smoke timings are
not speed claims.

The measured 235/620-token baseline completed all 76 cases with 20 samples,
99% intervals and a 3% noise threshold. Telemetry recorded one KFD client at most.
Artifacts: `benches/results/flash-support-tiled-baseline`. Reproduce with:

```sh
CLEF_BENCH_LENGTHS=235,620 python3 scripts/bench_workload.py flash-tiled-baseline --suite flash
```

## Full context and workflow matrix

Flash BF16 has executed every length through 65,536 tokens, without truncation,
and answered all three beginning/middle/end retrieval questions correctly at
each length. The six-configuration matrix and final long replay are still running.

| Encoded tokens | Request wall seconds | Input tok/s | Peak engine GiB |
| --- | ---: | ---: | ---: |
| 4,096 | 12.80 | 320.06 | 14.81 |
| 8,192 | 5.40 | 1,517.57 | 15.87 |
| 16,384 | 16.65 | 983.86 | 18.00 |
| 32,768 | 62.04 | 528.21 | 20.00 |
| 49,152 | 127.72 | 384.84 | 23.00 |
| 65,536 | 309.38 | 211.83 | 28.00 |

The first request includes model loading. Later requests include encoding,
preparation, compilation, execution and readback. These are actual full-model
requests with three joint questions. Long-context throughput differs substantially
from short-request JevBench throughput. The subsequent 4k replay reproduced its
original output exactly in 2.02 seconds.

Raw records: `tests/results/flash-support-matrix/context-clef-flash-bf16`.
Matrix executable SHA256:
`18b7de8cc007ad00149cae61084de29d0a596c92e3fe370b25b7a300190ba708`.

The matrix runner freezes the executable and schedules the six combinations on
one GPU. It runs the actual context ladder before the full WorkflowEvals matrix.
Each context run includes facts at the beginning, middle and end, all three
question types, and long→short→long replay. Retrieval answers, token counts,
latency, memory and hardware telemetry are retained separately.

```sh
python3 scripts/qualify_matrix.py --binary target/release/clef \
  --output tests/results/flash-support-matrix
# Resume with the frozen executable if target/release/clef was rebuilt:
python3 scripts/qualify_matrix.py --binary tests/results/flash-support-matrix/clef \
  --output tests/results/flash-support-matrix --resume
```

`status.json` distinguishes running, failed and completed jobs. A lock rejects a
second simultaneous resume of the same matrix. Workflow runs
resume through case checkpoints. Failed context runs use fresh attempt
directories. A changed binary, runner or configuration requires a new matrix.
Completing this matrix can take many hours: the existing original CLEF invoice
example alone took about 314 seconds of GPU execution for 36,916 tokens and 55
joint questions. Partial matrices are not reported as full qualification.

For later kernel candidates, compare all consensus metrics per workflow; a gain
in one workflow must not conceal a regression in another:

```sh
python3 scripts/compare_workflowevals.py BASELINE_DIRECTORY CANDIDATE_DIRECTORY
```

Comparisons require the same model, precision, revisions, cases and scoring
denominators. They also report the other label sets and native timing boundaries.
Use frozen request corpora for performance comparisons when workflow branches
differ.
