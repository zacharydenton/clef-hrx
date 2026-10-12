# Attention precision experiments

These experiments compare BF16 and FP16 attention inside the same original CLEF
INT4 model. They use complete WorkflowEvals invoices, the pinned checkpoint,
and shared gfx1151 hardware. Production kernel selection is unchanged.

## Long-invoice profile

The diagnostic input is `ap_00000`, with 36,916 tokens. Optimized INT4 uses the
single-component shared BF16 attention experiment and long vector DeltaNet.
The ordinary executions surrounding profiling took 49.21 and 49.53 seconds,
including encoding, preparation and readback; model loading is excluded.
Their question probabilities were identical. Engine allocation was 20.11 GiB.

| Operation | Serialized GPU time | Share |
| --- | ---: | ---: |
| Full attention, 16 layers | 15.945 s | 33.95% |
| INT4 matrix multiplies | 20.595 s | 43.85% |
| DeltaNet recurrence | 3.752 s | 7.99% |
| Other kernels | 6.677 s | 14.22% |

The profile totals 46.969 seconds. Profiling introduces barriers, and background
GPU activity varies: use these shares to choose experiments, not as ordinary
request throughput. Halving attention time would save about 17% of this profile.
Projection work remains a substantial target.

## Precision experiment

`benches/kernels/attention_shared_fp16.loom` retains the shared attention tile,
converts BF16 Q/K/V loads to FP16 before shared-memory storage, and uses one FP16
probability component. Softmax and QK/PV accumulation remain FP32; outputs use
the existing BF16 store. Model weights and the decision head are unchanged.

Compiler reports confirm `amdgpu.v_wmma_f32_16x16x16_f16` instructions. Both this
candidate and single-component BF16 use 180 VGPRs and 64,768 LDS bytes, with
25% reported occupancy limited by LDS. Changing the datatype does not reduce
this resource footprint.

Paired Criterion measurements alternate BF16/FP16 order, exclude warmup and use
20 measured batches with 10,000 paired bootstrap resamples:

| Input tokens | Shared BF16 | FP16 | BF16 time / FP16 time | 99% interval |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 42.49 ms | 44.89 ms | 0.947× | 0.943–0.950× |
| 36,916 | 972.65 ms | 1,021.87 ms | 0.952× | 0.939–0.963× |

FP16 is slower at both lengths. It is retained as an accuracy experiment, not a
speed optimization. Scalar-reference checks at 256 tokens and production-reference
checks at 8,192 tokens pass the existing 0.008 absolute-error bound. The independent
uniform-softmax oracle also passes at 65, 8,191, 8,192, 8,193, 16,385 and 65,536
tokens, including sliced inputs and repeated execution, at its existing 0.002
tolerance. Full-model replay of `ap_00000` is exact for all three new variants.

## Accuracy investigation

The first seven saved invoices contain 21 policy decisions. Production INT4
matches consensus on 14 action sets and 18 primary actions; the single-component
optimized candidate matches 13 and 17. This partial result does not clear the
same-precision accuracy gate.

The changed policy is `high_volume_retailer` on `ap_00000`. Its `line_2_scope`
classification changes from `priced_unit` to `extra_unapproved`. Replaying the
pinned workflow on CPU and restoring only this question's original probabilities
restores the previous action set. Restoring the other changed classification,
`line_3_scope`, does not. These substitutions diagnose the change; they are not
model results or proposed output corrections.

Restoring the second BF16 probability component does not restore the original
policy answer on this invoice. Restoring the original DeltaNet kernel produces
bit-for-bit identical probabilities to the single-component candidate. This
isolates the observed change to attention, while leaving the complete-suite
accuracy question open.

FP16 restores the original action set on this invoice, with exact replay, but
does so through a different line classification: `line_1_scope` switches to
`priced_unit`, while `line_2_scope` still selects `extra_unapproved`. Matching
the final action therefore does not establish closer intermediate predictions.
The resident FP16 replay took 50.11 seconds (736.71 input tokens/s); a single
unpaired replay is insufficient to claim a speed difference.

The subsequent seven-invoice run matches production INT4 on every policy action
set and primary action: 14/21 action sets and 18/21 primary actions agree with
consensus. There are no failed cases. The six resident requests contain 164,064
tokens and take 210.23 seconds, or **780.40 input tokens/s**. The earlier shared
BF16 candidate measured 777.47 on the same requests; production attention measured
182.08. These historical runs were not interleaved, and the 0.4% difference
between FP16 and shared BF16 does not establish an improvement. Seven invoices
do not qualify the complete 705-case suite.

## llama.cpp Q4 comparison

The Q4_K_M pilot used the pinned llama.cpp HIP build and GGUF from the
[JevBench comparison](llama-jevbench.md), with a 32,768-token context and a
30,342-token complete invoice (`ap_00051`). The existing
[WorkflowEvals memory guard](llama-workflowevals.md) stopped the owned server
during weight loading because swap grew by more than 1 GiB. The last telemetry
sample still showed 54.5 GiB available; the swap-pressure condition triggered
independently of the available-memory reserve. System and cgroup OOM-kill
counters were zero when checked afterward.

No request completed, so there is no Q4 throughput or accuracy result from this
attempt. The 64k comparison was not launched. Preserve the failed pilot at
`tests/results/attention-precision-20261012/llama-q4-32k/`; a later retry needs a
fresh output directory and the same live memory guard. Admission based on GGUF
file size alone did not establish safe loading under the observed host load.

## Reproduce

Generate the single-component BF16 reference and build the development tools:

```sh
python3 scripts/attention_probability_candidate.py --output artifacts/attention-single.loom
cargo build --release --locked --features bench-internals \
  --example profile_inference --example probe_attention --bin clef
```

Save a complete invoice's `request` object from a WorkflowEvals native call log
as `invoice.json`. Profile with an explicit allocation cap:

```sh
CLEF_MODEL_DIR=artifacts/model \
CLEF_EXPERIMENT_ATTENTION_SOURCE=artifacts/attention-single.loom \
CLEF_EXPERIMENT_DELTA_VECTOR_LONG=1 \
  target/release/examples/profile_inference --precision int4 --memory-gib 40 \
    --max-length 65536 --rounds 1 --output profile.json invoice.json
```

Compile and check the FP16 operator, then measure alternating-order pairs:

```sh
target/release/examples/probe_attention --compile-only \
  benches/kernels/attention_shared_fp16.loom
target/release/examples/probe_attention --tokens 256 --rounds 3 \
  benches/kernels/attention_shared_fp16.loom
CLEF_BENCH_ATTENTION_REFERENCE_SOURCE=artifacts/attention-single.loom \
CLEF_BENCH_ATTENTION_SOURCE=benches/kernels/attention_shared_fp16.loom \
CLEF_BENCH_LENGTHS=8192,36916 \
  python3 scripts/bench_workload.py attention-precision-repro \
    --suite workload --filter paired_attention
CLEF_BENCH_ATTENTION_ORACLE_SOURCE=benches/kernels/attention_shared_fp16.loom \
  cargo test --release --locked --lib --features bench-internals \
    shared_attention_context_oracle -- --ignored --nocapture
```

For full-model replay, use `CLEF_EXPERIMENT_ATTENTION_SOURCE` with each candidate
and the same `--precision int4 --memory-gib 40 --strict-context` CLI options.
Keep DeltaNet fixed while changing attention. Test the original DeltaNet
separately by omitting `CLEF_EXPERIMENT_DELTA_VECTOR_LONG`.

Repeat the seven-invoice check with the pinned WorkflowEvals environment and a
fresh output directory:

```sh
CLEF_EXPERIMENT_ATTENTION_SOURCE=benches/kernels/attention_shared_fp16.loom \
CLEF_EXPERIMENT_DELTA_VECTOR_LONG=1 \
  artifacts/WorkflowEvals/.venv/bin/python scripts/eval_workflowevals.py \
    --binary target/release/clef --model-dir artifacts/model \
    --precision int4 --memory-gib 40 --workflows invoice_processing \
    --new-cases-per-workflow 7 --output tests/results/fp16-invoices-repro
```

Local records are under `tests/results/attention-precision-20261012/` and
`artifacts/attention-precision-20261012/`. The local supervisor runs one GPU job
at a time using the qualification lock. HRX model admission requires 60 GiB
available for its 40 GiB cap; the watchdog stops its own process group below
20 GiB available or after swap grows by more than 1 GiB. These direct commands
do not themselves provide that external watchdog.

Paired timing records, the source snapshot, compiler identity and hardware
telemetry are in `benches/results/attention-fp16-vs-single-bf16-20261012/`.
