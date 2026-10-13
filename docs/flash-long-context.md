# Flash long-context attention

This optimization applies the [shared-memory attention kernel](long-context-bf16.md)
to CLEF Flash's 16 query heads and four KV heads. It uses the pinned Flash
checkpoint and BF16 projections. Flash retains its existing DeltaNet kernel.
It is now selected by default for Flash BF16 causal text attention from 8,192
through 65,536 tokens. Broader INT4 uses a separately qualified
[16-key kernel](attention-kernel-tuning.md#model-qualification). Original CLEF's
quantized modes and Flash MLP-only INT4 retain their previous paths.

The candidate measures **1,597.05 resident input tokens/s** on the complete
30,342-token invoice. Ten graph-rebuild Criterion samples give a 99% interval of
**1,490.91–1,689.51 tokens/s** and a mean latency of 19.00 seconds. Encoding,
preparation, GPU execution and readback are included; model loading is excluded.
The same-invoice production baseline measures 370.37 tokens/s, giving a 4.31×
ratio of means. These separate runs are not an alternating-order model comparison.

The complete 705-case WorkflowEvals qualification also passes, with no execution
failures and every reported accuracy metric unchanged. Resident throughput is
**1,556.96 input tokens/s**, versus **530.54** for production: **2.93× faster**.
Both runs contain 1,256 resident calls and 6,439,577 resident input tokens;
their summed native request times are 4,135.98 and 12,137.78 seconds. These
exclude model loading, CPU preflight and Python workflow execution. JevBench
retains 191/231 and the context gate passes all 15 retrieval answers with exact
replays. This qualification covers Flash BF16. Normal builds select the
qualified kernel without a feature flag or source override.
The ordinary release build also reproduces the qualified predictions exactly
across long, short, 8k-boundary and long-replay requests; its validation record
is `tests/results/attention-k16-qualification-20261012/default-bf16-validation.json`.

The qualification compares production attention against the single-component
BF16 probability variant at 8,192, 16,384 and 30,342 tokens, with a separate
65,536-token check. Each paired Criterion run alternates reference and candidate
execution order. Full-model benchmarks rebuild the graph on every request and
exclude initial model loading. Measurements share the GPU with another client.

The full-model request is WorkflowEvals invoice `ap_00051`, with all 30,342 encoded
tokens retained. A separate baseline run uses the same request, checkpoint and
benchmark executable without the attention override. These repeated-invoice
measurements are separate from aggregate WorkflowEvals throughput.

## Operator results

Measured on October 12, 2026 under shared GPU load:

| Tokens | Existing attention | Candidate | Speedup | 99% paired interval |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 168.17 ms | 27.29 ms | 6.16× | 6.11–6.22× |
| 16,384 | 1,219.28 ms | 154.46 ms | 7.89× | 6.95–8.64× |
| 30,342 | 6,282.63 ms | 542.24 ms | 11.59× | 10.61–12.69× |

These include value packing and completion fences. Every candidate replay is
exact; the maximum absolute output difference is 0.0078125 at each length.
The full-model result above measures the complete network; operator speedups do
not imply the same end-to-end speedup. Full-suite results are reported above.

The separate 65,536-token probe also passes the 0.008 error limit and exact
candidate replay. Its one timing pair measured 34.90 seconds for existing
attention and 3.84 seconds for the candidate; this is a correctness screen with
a timing observation, not a repeated Criterion speedup estimate.

## Reproduction

Generate the candidate and benchmark Flash's actual head geometry:

```sh
python3 scripts/attention_probability_candidate.py \
  --output artifacts/flash-attention-single.loom
cargo build --release --locked --features bench-internals \
  --bin clef --example probe_attention
target/release/examples/probe_attention --model clef-flash \
  --tokens 65536 --rounds 1 --reference-source kernels/attention_tiled.loom \
  artifacts/flash-attention-single.loom
CLEF_BENCH_MODEL=clef-flash CLEF_BENCH_LENGTHS=8192,16384,30342 \
CLEF_BENCH_ATTENTION_REFERENCE_SOURCE=kernels/attention_tiled.loom \
CLEF_BENCH_ATTENTION_SOURCE=artifacts/flash-attention-single.loom \
  python3 scripts/bench_workload.py flash-attention-repro --filter paired_attention
```

The explicit reference above selects the former production kernel; an unset
reference now measures the optimized default. Model-level experiment variables
do not alter the operator reference. Both
variants must produce finite outputs, pass the 0.008 maximum absolute difference
check and reproduce their own outputs exactly. Model evaluations gate acceptance.

Extract `request` for invoice `ap_00051` from the saved Flash baseline's
`native-calls.jsonl` as one line in `requests.jsonl`. See the
[WorkflowEvals adapter](workflowevals.md) to generate requests and reproduce the
baseline. For the full-model benchmark:

```sh
CLEF_BENCH_MODEL=clef-flash CLEF_MODEL_DIR=artifacts/model-flash \
CLEF_BENCH_FULL_MODEL=1 CLEF_BENCH_PRECISION=bf16 \
CLEF_BENCH_REQUESTS=requests.jsonl \
CLEF_BENCH_ATTENTION_SOURCE=artifacts/flash-attention-single.loom \
CLEF_EXPERIMENT_ATTENTION_SOURCE=artifacts/flash-attention-single.loom \
  python3 scripts/bench_workload.py flash-invoice-repro \
    --suite workload_inference --filter rebuild_wall
```

Repeat with a distinct benchmark name and
`CLEF_EXPERIMENT_ATTENTION_SOURCE="$PWD/kernels/attention_tiled.loom"`
to measure the former production path. Criterion collects ten samples and a
99% mean-latency confidence interval for each run. Operator speedup intervals use
20 paired batches and 10,000 paired bootstrap resamples.

Keep the candidate source and executable immutable during model evaluation:

```sh
export CLEF_EXPERIMENT_ATTENTION_SOURCE="$PWD/artifacts/flash-attention-single.loom"
python3 scripts/eval_jevbench.py --binary target/release/clef \
  --model clef-flash --model-dir artifacts/model-flash --precision bf16 \
  --output tests/results/flash-attention-jevbench
python3 scripts/qualify_context.py --binary target/release/clef \
  --model clef-flash --model-dir artifacts/model-flash --precision bf16 \
  --lengths 8192 32768 65536 --output tests/results/flash-attention-context
artifacts/WorkflowEvals/.venv/bin/python scripts/eval_workflowevals.py \
  --binary target/release/clef --model clef-flash \
  --model-dir artifacts/model-flash --precision bf16 \
  --output tests/results/flash-attention-workflows
python3 scripts/compare_jevbench.py \
  tests/results/flash-support-baseline/clef-flash-bf16 \
  tests/results/flash-attention-jevbench
python3 scripts/compare_workflowevals.py \
  tests/results/flash-support-matrix/workflowevals-clef-flash-bf16 \
  tests/results/flash-attention-workflows
unset CLEF_EXPERIMENT_ATTENTION_SOURCE
```

The JevBench floor is 191/231. WorkflowEvals must complete all 705 cases and
preserve every consensus metric per workflow. Its references score action
agreement, not human-labelled question accuracy. The context check requires
15/15 retrieval answers and exact 8k/64k replays.

## Run records

Frozen inputs, hashes, telemetry and completed qualification state are under
`tests/results/flash-long-attention-20261012/`. Its `workflowevals-comparison.json`
contains the per-workflow accuracy and throughput comparison. Original CLEF
BF16's separate [full-suite result](long-context-bf16.md) is now accepted against
Cloudflare's published scores.

An initial operator run inherited the model-level experiment override in its
reference. That run is invalidated and excluded. The corrected reference ignores
the override, and its measurements use the `flash-shared-attention-long-v2` name.
