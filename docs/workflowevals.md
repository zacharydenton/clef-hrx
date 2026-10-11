# WorkflowEvals reproduction

The runner in `scripts/eval_workflowevals.py` uses the upstream workflow code,
case checkpoints, consensus references and final-action scorer. It replaces only
model transport with a resident local CLEF CLI process. These are workflow action
agreement measurements against model-generated references, not human-labelled
question accuracy.

## Pinned suite

Upstream source: [typesafe-ai/WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals/tree/0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb),
commit `0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb`. The source must be unmodified.

| Workflow | Cases | Policies | Consensus metrics | Dataset revision |
| --- | ---: | ---: | --- | --- |
| Invoice processing | 150 | 3 | Exact action set; primary action | `6beeb2d2acd65c086c835022f5f4d7434114cafc` |
| Customer service | 204 | 3 | Exact action set | `b1342f5a704587dbc465867c38c2694348ff86e4` |
| Agent trace observability | 111 | 2 | Primary action | `8635540973910a92465fe2bc53e195375aa6e1a8` |
| Security incidents | 240 | 1 | Exact action set | `fbe1ea5c69cf494157fd23f2002a0d9d9a418443` |

The 705 cases contain 1,524 policy decisions. All policies are selected. Dynamic
branches follow the model's answers. The adapter never sends reference labels or
descriptive dataset metadata to the model.

## Input capacity and scoring

The configured context limit and runner default are 65,536 tokens, within the
pinned checkpoint’s native 262,144 positions. RoPE needs no extension or scaling. Requests are
jointly evaluated using the entire state and schema, including the first invoice
node's 55 questions (28,719 schema/prompt tokens before state).

Each request passes through the native CPU encoder before inference. A request
that exceeds the configured capacity triggers the upstream session's recursive question
splitting fallback so the evaluator never accepts silently truncated state.
Unsplittable requests fail and remain in scoring. Accepted native token counts
must match preflight. The manifest and request log record all batching decisions.
The earlier 16k smoke runs were interrupted or failed; they are not benchmark
results.

Noul probabilities, Choice distributions and Score expected values retain native
precision. No rounding or probability renormalization is applied. The adapter
checks question IDs, options, finite probability values and unit sums. The
upstream scorer compares full action sets (including arguments, ignoring order
and duplicates) or primary actions, as specified by each workflow. Failed cases
remain in the denominator under upstream scoring rules.

## Run

Use the pinned CLEF checkpoint described in [JevBench reproduction](evaluation.md#reproduce).
The upstream dependencies are installed in a separate environment; no API key is
needed for the local model or these public datasets.

```sh
git clone https://github.com/typesafe-ai/WorkflowEvals artifacts/WorkflowEvals
git -C artifacts/WorkflowEvals checkout 0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb
uv sync --locked --prerelease if-necessary --python 3.13 \
  --project artifacts/WorkflowEvals
cargo build --release --locked --bin clef

artifacts/WorkflowEvals/.venv/bin/python scripts/test_eval_workflowevals.py
artifacts/WorkflowEvals/.venv/bin/python scripts/eval_workflowevals.py \
  --model-dir artifacts/model --output tests/results/workflowevals-bf16
```

Use a fresh output directory. `--dry-run` validates the pinned inputs without GPU
work. `--limit 1` checks one case per workflow and is explicitly marked as a smoke
run. Omit it for the full suite. `--workflows` can select individual workflows.
`--resume` requires identical binary, runner, case selection and settings; the
upstream case checkpoints avoid repeating completed cases. A restart loads the
model again, and each process's first request is marked separately.

The default 80 GiB model allocation budget includes weights and workspace. Run
GPU jobs sequentially. The four workflows share one model process, one case runs
at a time, and all questions in a fitting node are evaluated jointly.

For resumable GPU scheduling, add `--new-cases-per-workflow 1` to execute one
additional case from each workflow's sorted full case list. Repeat with `--resume`
and any positive batch size, or omit the batch option to finish every remaining
case. The manifest keeps the full selected suite; completed case checkpoints are
reused, partial scores show completed/planned counts, and `complete` remains false
until all selected cases finish. This differs from `--limit`, which deliberately
selects a smaller smoke-only suite. Do not report partial scores as full results.

## Artifacts and timing

`native-manifest.json` records the binary and runner hashes, command, checkpoint
revision, dataset revisions and selected case IDs. Each workflow directory has
upstream `results.json`, `scores.json` and resumable case checkpoints. Raw native
requests, predictions, token counts and phase timings are in `native-calls.jsonl`;
capacity checks and rejected batch sizes are in `preflight.jsonl`. GPU telemetry
and stderr are retained alongside `summary.json`.

Native input throughput includes each accepted request's encoding, compilation,
preparation, GPU execution and readback. It excludes the first request of each
model process and the separate CPU preflight. Workflow case latency additionally
includes preflight, question splitting and Python workflow execution. Report
these timing boundaries separately; a kernel or native-call throughput figure
is not the complete workflow's decision throughput.

No full WorkflowEvals result has been qualified yet. The initial joint smoke run
failed on schema capacity and is retained in
`tests/results/workflowevals-bf16-smoke/`; it is not an accuracy result.

### First resumable 64k batch

`tests/results/workflowevals-swiglu-64k` has completed one case per workflow:
**4/705 cases**, covering nine policy decisions and eight native calls. All four
cases completed without execution failures; every accepted request retained its
full state and schema, and no capacity split was needed. These are partial
results, with `complete: false`; they do not estimate full-suite accuracy.

| Workflow | Completed / planned cases | Consensus agreement in this batch |
| --- | ---: | --- |
| Invoice processing | 1 / 150 | 0/3 exact action sets; 0/3 primary actions |
| Customer service | 1 / 204 | 3/3 exact action sets |
| Agent trace observability | 1 / 111 | 0/2 primary actions |
| Security incidents | 1 / 240 | 0/1 exact action sets |

The first invoice request contained **36,916 tokens and 55 questions**. It
completed in 352.900 seconds including model load; reported GPU inference was
313.691 seconds and engine allocation was 53.676 GiB. This establishes successful
execution of that long request, not a full 64k accuracy qualification.

The seven subsequent resident calls processed 13,235 tokens in 27.886 seconds:
**474.62 input tok/s**, including native encoding, preparation, compilation,
inference and readback. This excludes the first invoice call and separate CPU
preflight, so it must not be used as the speed of the long invoice workload or
compared directly with the mixed JevBench rate.

The batch uses the BF16 `clef-swiglu` binary and source archive recorded in
[JevBench qualification](evaluation.md#fused-swiglu). Resume the exact run with:

```sh
HF_HUB_CACHE="$PWD/artifacts/throughput-1000/hf-cache" \
  artifacts/WorkflowEvals/.venv/bin/python scripts/eval_workflowevals.py \
  --binary artifacts/throughput-1000/clef-swiglu \
  --output tests/results/workflowevals-swiglu-64k \
  --new-cases-per-workflow 1 --resume
```

Omit `--new-cases-per-workflow` to schedule all remaining cases. Keep the same
binary, runner, precision, context limit and dataset pins when resuming.

## Context validation

The current [six-configuration matrix](clef-flash.md#full-context-and-workflow-matrix)
has completed actual 65,536-token inference for CLEF and Flash in all three
precisions, with all retrieval answers correct and exact long/short replay.
Its full 705-case WorkflowEvals runs are in progress. Inspect current case counts
and request throughput with `python3 scripts/report_matrix.py tests/results/flash-support-matrix`.

The 64k build passes GPU attention indexing checks with 65,536 query rows and
65,536 key rows (tested separately against a uniform-softmax oracle), as well as
the existing causal, grouped-query, tail and graph-replay checks. Chunked MLP
output matches the unchunked operator bit-for-bit in the boundary/replay fixture.
CPU encoding accepts a 65,234-token synthetic state; the first invoice request
encodes to 36,916 tokens with all 55 questions. These checks do not constitute a
full 65,536-token model accuracy evaluation.

```sh
cargo test --locked --lib --all-features native_context_attention -- --ignored
cargo test --locked --lib --all-features chunked_mlp_parity -- --ignored
cargo test --locked --lib --all-features tiled_attention_tails_and_replay -- --ignored
```

The allocation cap is separate from context capacity: loading checks only the
minimum weight/workspace allowance, and actual buffer allocations are checked
against HRX's budget and available system RAM. Long requests process MLP rows in
8,192-token chunks; attention retains the full causal context.
