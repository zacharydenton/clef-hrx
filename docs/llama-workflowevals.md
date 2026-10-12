# CLEF through llama.cpp on WorkflowEvals

The BF16 comparison is queued. There are no llama.cpp WorkflowEvals accuracy or
throughput results yet. A [Q4_K_M invoice pilot](attention-precision.md#llamacpp-q4-comparison)
was stopped by the swap-pressure guard during loading, before any request completed.
[JevBench measurements](llama-jevbench.md) cover shorter
inputs and do not establish performance on this suite.

The adapter uses the same pinned WorkflowEvals source, four datasets, 705 cases,
policies and consensus scorer as the [HRX run](workflowevals.md). It calls the
native llama.cpp `/v1/systemone` endpoint sequentially with one GPU slot. Every
response must preserve the complete input-token count measured by the HRX CPU
encoder. Native probabilities and expected scores reach the scorer without
rounding or renormalization.

## Memory and scheduling

The runner shares `tests/results/flash-support-matrix/run.lock` with qualification
jobs. A busy slot or insufficient available RAM exits with code 75 before
starting a server. Other GPU clients may continue running; telemetry records
them and results must be labeled as shared-GPU measurements.

Start with a 32,768-token server and replay a full invoice three times. The
65,536-token run requires that pilot to pass with matching model, executable
and adapter hashes, exact replay, and at least 30,000 input tokens per request.
Context, batch and microbatch limits are equal because CLEF's joint decision
graph consumes the entire request in one microbatch.

Admission reserves the GGUF file size, estimated workspace of 12 GiB at 32k or
24 GiB at 64k, and another 16 GiB for headroom. For this BF16 GGUF, that requires
at least **78.4 GiB available for the pilot** and **90.4 GiB for 64k**. The 64k
estimate increases if the pilot's observed memory use, with scratch scaled by
context squared, exceeds that allowance. These are conservative admission
estimates, not proven allocator bounds.

A watchdog samples host memory approximately every 100 ms and kills only the
owned server process group if available memory falls below the reserve, swap
grows by more than 1 GiB, or monitoring fails. It cannot guarantee protection
against an allocation spike between samples. Guard failures stop the run; they
do not trigger retries, input truncation or an automatic precision change.

The local queue is `artifacts/llama-workflowevals/run_queue.py`, with progress in
`tests/results/llama-workflowevals-20261012/queue-status.json`. It waits for the
Flash and CLEF candidate qualifications, then for the shared qualification lock
and sufficient memory. It releases the lock while waiting for memory, allowing
other qualification work to continue.

## Reproduce

Use the pinned llama.cpp build, BF16 GGUF and download manifest from
[the JevBench setup](llama-jevbench.md#reproduce), plus the pinned WorkflowEvals
venv and HRX tokenizer model from [the WorkflowEvals setup](workflowevals.md).
The commands below use this machine's existing paths.

Save a complete invoice request from an HRX WorkflowEvals call log:

```sh
python3 - <<'PY'
import json
from pathlib import Path
source = Path('tests/results/bf16-long-20261012/workflowevals/native-calls.jsonl')
for line in source.open():
    call = json.loads(line)
    if call['workflow'] == 'invoice_processing' and 30000 <= call['input_tokens'] <= 32768:
        Path('artifacts/llama-workflowevals/invoice.json').write_text(json.dumps(call['request']) + '\n')
        break
else:
    raise SystemExit('No complete 30k–32k invoice request found')
PY
```

Run the pilot, then the full suite only after it succeeds:

```sh
artifacts/WorkflowEvals/.venv/bin/python scripts/eval_workflowevals_llama.py \
  --binary /home/zach/code/llama.cpp/build/bin/llama-server \
  --llama-source /home/zach/code/llama.cpp \
  --model artifacts/llama-jevbench/gguf/Clef-BF16.gguf \
  --manifest artifacts/llama-jevbench/gguf/manifest.json \
  --context 32768 \
  --smoke-request artifacts/llama-workflowevals/invoice.json \
  --output tests/results/llama-workflowevals-pilot

artifacts/WorkflowEvals/.venv/bin/python scripts/eval_workflowevals_llama.py \
  --binary /home/zach/code/llama.cpp/build/bin/llama-server \
  --llama-source /home/zach/code/llama.cpp \
  --model artifacts/llama-jevbench/gguf/Clef-BF16.gguf \
  --manifest artifacts/llama-jevbench/gguf/manifest.json \
  --pilot-output tests/results/llama-workflowevals-pilot \
  --output tests/results/llama-workflowevals-bf16
```

`--dry-run` validates inputs without starting a server. `--resume` requires the
same manifest and keeps completed case checkpoints. Failed checkpoints remain
visible and require investigation and a fresh output directory; the adapter
will not silently skip them. `--new-cases-per-workflow N` bounds a partial run
without changing the full plan or declaring it complete.

## Accounting

Resident throughput is total accepted input tokens divided by total HTTP
request wall time, excluding the first request of each server process. It
includes tokenization, GPU execution, response parsing and validation. Server
startup and CPU token-count preflight are recorded separately. End-to-end
workflow time includes preflight and workflow execution. Transport or memory
guard failures suppress the run's throughput claim.

Each output contains a manifest, raw requests and responses, token counts,
server logs, memory/GPU telemetry, upstream case checkpoints and scores.
`summary.json` reports completion only when all four workflows have their full
case counts and no failed cases. Compare accuracy per workflow against matching
HRX consensus metrics; a partial invoice run is not a full-suite comparison.
