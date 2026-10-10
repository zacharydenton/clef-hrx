# CLEF validation and benchmarks

Commands run from the repository root.

## Development

```sh
cargo fmt --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --lib --bins --tests --examples --all-features
cargo test --locked --doc

# GPU operators, trained head, cached tokenizer, and video fixtures:
cargo test --locked --lib -- --include-ignored --skip full_checkpoint_corpus --test-threads=1
# Full checkpoint; run separately with at least 50 GiB available RAM:
cargo test --release --locked --lib full_checkpoint_corpus -- --ignored --nocapture --test-threads=1
```

GPU and checkpoint tests are explicitly ignored by default. Fixtures capture
Torch 2.14.0 / Transformers 5.10.2 outputs. `CLEF_MODEL_DIR` selects a local
checkpoint for the full-model test and benchmark.

```sh
cargo bench --locked --bench preprocessing
cargo bench --locked --features bench-internals --bench kernels
cargo bench --locked --features bench-internals --bench normalization
cargo bench --locked --features bench-internals --bench gemm
CLEF_BENCH_FULL_MODEL=1 cargo bench --locked --bench inference
```

The full-model benchmark reports load and first-request timings separately, then
measures exact-input graph replay and alternating equal-length requests. The
alternating case includes graph rebuilding and uploads with warm kernel
specializations; new token lengths can also incur compilation costs.

The normalization benchmark alternates the serial reference and parallel kernels
and reports paired timings. For full-model per-kernel diagnostics, run
`cargo run --release --locked --features bench-internals --example profile_inference`.
It accepts an optional request JSON path and uses the cached checkpoint with a
1,536-token limit. Diagnostic timings include serialization barriers; use the
inference benchmark for end-to-end latency.

The GEMM benchmark compares the original kernel with automatic selection and
checks identical outputs. Large compatible matrices use the pipelined 128×256
kernel from qwen-image-hrx; other shapes use the K64 or original K32 kernel.
The `gemm_pipelined` group also compares against K64 directly. To compare
normalization and GEMM changes within one loaded model:

```sh
cargo run --release --locked --features bench-internals --example compare_inference
# Check all corpus outputs without timing rounds:
cargo run --release --locked --features bench-internals --example compare_inference -- \
  --rounds 0 tests/fixtures/corpus/{0..7}.json
```

This rotates kernel variants, checks exact GEMM prediction parity, and reports
normalization probability differences. Request timings include graph rebuilding;
inference timings isolate graph execution. Run full-model jobs sequentially.

Kernel benchmarks include paired reference/optimized measurements. Check them
with `cargo run --locked --example check_benches -- target/criterion paired`;
thresholds are in [benches/criteria.json](../benches/criteria.json). Use an idle GPU.
`CLEF_TRACE_DIR=/path` dumps FP32 intermediates for debugging and increases memory
use. Leave it unset for benchmarks.
