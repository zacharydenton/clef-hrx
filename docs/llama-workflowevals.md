# CLEF through llama.cpp on WorkflowEvals

The Q4_K_M invoice pilot measures **202.81 resident input tokens/s** with exact
replay. This is one complete invoice, not the 705-case WorkflowEvals suite.
The BF16 comparison is held for input-serialization parity. The first
[Q4_K_M invoice pilot](attention-precision.md#llamacpp-q4-comparison)
was stopped by the swap-pressure guard during loading. A subsequent retry had
enough memory and exposed the serialization difference described below.
[JevBench measurements](llama-jevbench.md) cover shorter
inputs and do not establish performance on this suite.

The adapter uses the same pinned WorkflowEvals source, four datasets, 705 cases,
policies and consensus scorer as the [HRX run](workflowevals.md). It calls the
native llama.cpp `/v1/systemone` endpoint sequentially with one GPU slot. Every
response must preserve the complete input-token count measured by the HRX CPU
encoder. Native probabilities and expected scores reach the scorer without
rounding or renormalization.

## Q4 invoice result

On October 13, 2026, the pinned Q4_K_M GGUF completed three executions of
`ap_00051`, each containing all **30,342 input tokens**. The two resident calls
contain 60,684 tokens and take **299.22 seconds**, or **202.81 tokens/s**. The
first call is excluded. Server logs confirm zero cached prompt tokens on each
execution; answers reproduce exactly. There are no transport errors or truncated
inputs. Replaying the saved response through the pinned workflow matches
consensus on all three policy action sets and primary actions; this single case
does not establish suite accuracy.

Telemetry recorded only the owned GPU process. Peak available-memory drop was
**64.66 GiB** with a 20 GiB reserve, and the memory watchdog did not trigger.
The unchanged 64k admission formula requires **224.92 GiB available** from this
pilot's peak. The 64k run was therefore not launched. This is a conservative
admission estimate, not a measured 64k allocation or proof that a differently
configured implementation cannot fit.

Records are in
`tests/results/llama-q4-workflowevals-20261013/32k-canonical-pilot/`:
`summary.json` contains throughput, `process.json` contains replay and memory
checks, and `invoice-score.json` contains the three workflow decisions.
The parent directory's `64k-admission.json` records the refused escalation.

## Structured-state serialization

The pinned llama.cpp Jinja renderer writes floating-point JSON values through a
default C++ stream, using six significant digits and dropping integral `.0`
suffixes. On invoice `ap_00051`, this changes the input from **30,342 to 30,308
tokens**, including rounding `amount_vs_typical_ratio` from
`2.901720335244817` to `2.90172`. The unmodified structured-state attempt was
stopped before accepting a result.

CLEF accepts state as either structured JSON or text. For a text-only comparison,
serialize the state in advance with Python's compact, sorted, Unicode-preserving
JSON format. The model then receives the same text as HRX. CPU encoding confirmed
identical token IDs for the original request and its canonical-text version;
simulating the C++ float formatting reproduced the exact 34-token discrepancy.
Question schemas are preserved, and native model outputs are scored without
adjustment.

The evidence is in `artifacts/clef-comparisons-20261013/serialization-check.json`.
The corrected Q4 pilot uses a separate output directory,
`tests/results/llama-q4-workflowevals-20261013/32k-canonical-pilot/`.
This text-only workaround does not establish parity for media requests or
arbitrary structured question descriptions. The older BF16 queue is held until
its requests preserve the same model inputs.

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

The older local queue is `artifacts/llama-workflowevals/run_queue.py`. It was
stopped before starting a server after the serialization discrepancy was found;
the reason is recorded in
`tests/results/llama-workflowevals-20261012/serialization-hold.json`.

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
    if (call['workflow'] == 'invoice_processing'
            and call['request']['state']['case_id'] == 'ap_00051'):
        assert call['input_tokens'] == 30342
        request = call['request']
        if not isinstance(request['state'], str):
            request['state'] = json.dumps(request['state'], ensure_ascii=False,
                                          sort_keys=True, separators=(',', ':'),
                                          allow_nan=False)
        Path('artifacts/llama-workflowevals/invoice.json').write_text(json.dumps(request) + '\n')
        break
else:
    raise SystemExit('No complete ap_00051 invoice request found')
PY
```

Run the Q4 pilot (substitute `Clef-BF16.gguf` for BF16):

```sh
artifacts/WorkflowEvals/.venv/bin/python scripts/eval_workflowevals_llama.py \
  --binary /home/zach/code/llama.cpp/build/bin/llama-server \
  --llama-source /home/zach/code/llama.cpp \
  --model artifacts/llama-jevbench/gguf/Clef-Q4_K_M.gguf \
  --manifest artifacts/llama-jevbench/gguf/manifest.json \
  --context 32768 --reserve-gib 20 \
  --smoke-request artifacts/llama-workflowevals/invoice.json \
  --output tests/results/llama-workflowevals-q4-pilot
```

Full-suite execution remains held: dynamic workflow requests need the same
serialization correction, and 64k admission must pass using the pilot's measured
memory use. The frozen runner still sends structured state during full suites.

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
