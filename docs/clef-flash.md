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
winners, probability error below 0.003, and exact changed-input replay. Reference
fixtures and their environment manifest are in `tests/fixtures/flash/`.
Full JevBench, WorkflowEvals, throughput and 64k execution qualification are pending.

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

```sh
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
not model token throughput.
