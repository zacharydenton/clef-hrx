# Long-context BF16 experiments

The target is 500 resident input tokens/s for original CLEF on the long invoice
workload. These experiments use the pinned original checkpoint, BF16 projections,
and the complete 30,342-token `ap_00051` request from WorkflowEvals. No input is
truncated. Production dispatch is unchanged; experimental switches require
`--features bench-internals`.

The fastest candidate measured **539.25 resident input tokens/s** across ten
Criterion samples that rebuild the graph for every request. The 99% interval is
**535.51–541.60 tokens/s**, above the 500 tokens/s target throughout. Each sample
includes encoding, graph preparation, GPU execution and readback; initial model
loading is excluded. Mean request time is 56.27 seconds. These are repeated
measurements of one complete invoice, not throughput across the full suite.

Two preceding graph replays measured 557.62 tokens/s combined (560.76 and 554.51).
Every selected answer matched the original result; maximum probability change
was 0.008160 and replay was exact. Full JevBench is running, followed by 64k
retrieval/replay checks and all 705 WorkflowEvals cases in
`tests/results/bf16-long-20261012/`. Full evaluation gates production promotion.

Measurements on October 12, 2026 share the gfx1151 GPU with an unrelated compute
client. The first invoice experiment recorded up to three KFD clients; the paired
attention run recorded two. The user chose to continue on the shared GPU. Results
below describe those conditions, rather than exclusive-device peak throughput.

## Bottleneck and attention candidate

A serialized profile of the original BF16 request attributed 126.34 of 187.14
GPU seconds to the 16 full-attention layers. Other work took 60.80 seconds,
including 9.33 seconds of DeltaNet. Ordinary GPU executions surrounding the
profile took 203.59 and 175.19 seconds. Profiling barriers and changing background
load make this a diagnostic breakdown, not an end-to-end speed comparison.

`benches/kernels/attention_shared.loom` uses 64-query/32-key tiles and eight waves
per workgroup. Queries remain in LDS throughout the prefix, and keys and values
are shared across query groups. QK loads are scheduled in pairs before matrix
instructions. Online softmax and accumulation remain FP32; probabilities retain
both BF16 high and residual components. The different softmax reduction grouping
can change rounding.

The current padded kernel uses 232 VGPRs, 64,768 bytes of LDS and no spills. Its
initial scheduled version used 62,720 LDS bytes. Paired Criterion measurements include value packing and alternate
reference/candidate order. Each length has 20 measured batches; intervals use
10,000 paired bootstrap resamples at 99% confidence.

Initial scheduled tile, before probability-row padding:

| Input tokens | Existing attention | Candidate | Speedup | 99% interval |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 207.56 ms | 56.06 ms | 3.70× | 3.66–3.75× |
| 16,384 | 1,241.24 ms | 269.34 ms | 4.61× | 4.38–4.83× |
| 30,342 | 8,209.92 ms | 988.88 ms | 8.30× | 7.67–8.96× |

Maximum output change was 0.001953125 for these fixtures, below the 0.008 operator
gate. Every candidate replay was exact. An independent uniform-softmax oracle
also passes at 65, 8,191, 8,192, 8,193, 16,385 and 65,536 tokens, including sliced
inputs, grouped queries, causal boundaries and two replays. Its absolute tolerance
is 0.002. Both the padded two-component kernel and the single-component candidate
pass this oracle. Neither operator check substitutes for model evaluation.

The preceding shared-Q/K/V candidate, before paired QK scheduling, completed three
full invoice executions. Its two resident requests took 61.12 and 69.51 seconds:
496.41 and 436.52 tokens/s, or **464.54 tokens/s combined**. Every selected answer
matched the saved original result, maximum probability change was 0.005925, and
all repeated probabilities were exact. The 88.70-second first request includes
model loading and is excluded from resident throughput. This is one repeated
invoice, not the complete WorkflowEvals suite.

Adding paired QK scheduling and long-context vector DeltaNet measured 483.44
tokens/s over two resident replays, with exactly the same question arrays as the
first attention candidate. Padding probability rows from 32 to 40 BF16 elements
then improved the attention operator by 6.4%, 7.4% and 6.8% at 8k, 16k and 30k,
respectively, with bit-for-bit identical outputs. These gains have 99% paired
intervals entirely above 3%. This padding is retained in the checked-in kernel.

The fastest candidate additionally drops the residual BF16 probability component.
Weights, Q/K/V storage and projections remain BF16; accumulation and online
softmax remain FP32. Before probability padding this improved the shared kernel
by 17.1%, 18.9% and 5.6% at 16k, 30k and 64k. Padding then improved that variant
by 8.9% at 16k and 8.2% at 30k, reaching 599.44 ms per 30k attention layer. Maximum
operator difference versus production is 0.0078125. Its two resident full-invoice
requests took 54.11 and 54.72 seconds; the 85.28-second first request includes
model loading and is excluded. The model-level probability change of 0.008160
is a separate measurement from the operator error bound.

Score-row padding added only 0.9–1.9% and was not retained. GEMM row-group sweeps
at 8,192-row MLP and 30,342-row attention-output shapes found no useful change
from the existing group of eight. All nine GEMM probes retained exact outputs;
groups four and sixteen ranged from 0.976× to 1.016×, while one was substantially
slower. These were screening probes, not accepted Criterion improvements.

## Long DeltaNet scheduling

The existing eight-channel vector kernel also improves the scalar long-context
path. All five pairs below have bit-for-bit identical outputs and exact replays;
each uses 40 measured Criterion batches under shared GPU load.

| Input tokens | Speedup | 99% interval |
| ---: | ---: | ---: |
| 4,033 | 1.44× | 1.42–1.48× |
| 8,192 | 1.60× | 1.45–1.81× |
| 16,384 | 1.48× | 1.45–1.57× |
| 30,342 | 1.67× | 1.60–1.78× |
| 65,536 | 1.49× | 1.37–1.59× |

`CLEF_EXPERIMENT_DELTA_VECTOR_LONG=1` extends this kernel to original CLEF inputs
above 4,032 tokens in a benchmark build. Flash and shorter-input dispatch remain
unchanged. Raw records are in `benches/results/delta-vector8-long/`.

## Reproduce

Build the operator probes and run the independent oracle:

```sh
cargo build --release --locked --features bench-internals --example probe_attention
cargo test --release --locked --lib --features bench-internals \
  shared_attention_context_oracle -- --ignored --nocapture
target/release/examples/probe_attention --compile-only \
  benches/kernels/attention_shared.loom
target/release/examples/probe_attention --tokens 30342 --rounds 10 \
  --integrated-reference benches/kernels/attention_shared.loom
CLEF_BENCH_ATTENTION_SOURCE=benches/kernels/attention_shared.loom \
CLEF_BENCH_LENGTHS=8192,16384,30342 \
  python3 scripts/bench_workload.py attention-shared-repro --filter paired_attention
CLEF_BENCH_DELTA_SOURCE=kernels/delta_persistent_vector.loom \
CLEF_BENCH_DELTA_VALUES=8 CLEF_BENCH_LENGTHS=4033,8192,16384,30342,65536 \
  python3 scripts/bench_workload.py delta-vector-long-repro \
    --suite kernels --filter paired_delta_prefill
```

For full-model testing, set `CLEF_EXPERIMENT_ATTENTION_SOURCE` to the candidate
file. It replaces causal d256 attention only at 8,192 tokens and above. Short
requests retain their existing kernels. The source is read once when the engine
is created. Keep it immutable for the duration of an evaluation and record its
hash with the binary and result manifest.

```sh
cargo build --release --locked --features bench-internals --bin clef
CLEF_EXPERIMENT_ATTENTION_SOURCE=benches/kernels/attention_shared.loom \
  target/release/clef --model-dir artifacts/model --offline --strict-context \
  --precision bf16 decide requests.jsonl --jsonl --raw
```

Use the [WorkflowEvals adapter](workflowevals.md) to generate the request. The
existing local baseline is
`tests/results/flash-support-matrix/workflowevals-clef-bf16/native-calls.jsonl`;
select its record whose `request.state.case_id` is `ap_00051`. It has 30,342
encoded tokens. Write the `request` object as one JSONL line for CLI replay or
`CLEF_BENCH_REQUESTS` in the full-model Criterion suite. The latter's `rebuild_wall`
case includes rebuilding the graph for each resident request.

Generate the faster single-component experiment reproducibly from the checked-in
kernel; the generator records the source hash and refuses an unexpected structure:

```sh
python3 scripts/attention_probability_candidate.py \
  --output artifacts/attention-single.loom
CLEF_BENCH_ATTENTION_ORACLE_SOURCE=artifacts/attention-single.loom \
  cargo test --release --locked --lib --features bench-internals \
    shared_attention_context_oracle -- --ignored --nocapture
CLEF_BENCH_FULL_MODEL=1 CLEF_BENCH_REQUESTS=requests.jsonl \
CLEF_MODEL_DIR=artifacts/model CLEF_BENCH_PRECISION=bf16 \
CLEF_BENCH_ATTENTION_SOURCE=artifacts/attention-single.loom \
CLEF_EXPERIMENT_ATTENTION_SOURCE=artifacts/attention-single.loom \
CLEF_EXPERIMENT_DELTA_VECTOR_LONG=1 \
  python3 scripts/bench_workload.py bf16-long-rebuild-repro \
    --suite workload_inference --filter rebuild_wall
```

`CLEF_BENCH_ATTENTION_SOURCE` makes the benchmark runner archive the source;
`CLEF_EXPERIMENT_ATTENTION_SOURCE` selects it for the model. Set both to the same
file. Full-model Criterion rebuilds the graph for every timed request and uses
10 samples with a 99% interval. Keep model-level results separate from attention
operator throughput.

An optional guard is available for future exclusive-device measurements:

```sh
python3 scripts/exclusive_gpu.py --wait --log target/exclusive-attention.jsonl -- \
  env CLEF_BENCH_ATTENTION_SOURCE=benches/kernels/attention_shared.loom \
      CLEF_BENCH_LENGTHS=8192,16384,30342 \
  python3 scripts/bench_workload.py attention-exclusive-repro --filter paired_attention
```

It waits for idle KFD clients and stops only its own benchmark process group if
another client appears. It does not stop another application or detect every
possible graphics workload. Omit the guard for intentionally shared-GPU runs.

## Records

- `artifacts/bf16-invoice-500/profile.log`: ordinary and serialized profile timings.
- `artifacts/bf16-invoice-500/invoice-candidate/`: initial full-invoice predictions,
  timings, hashes and telemetry, explicitly marked as shared-GPU observations.
- `benches/results/attention-shared-long/`: scheduled candidate, source snapshot,
  compiler identity, telemetry, raw paired samples and confidence intervals.
- `artifacts/bf16-invoice-500/attention-oracle.log`: independent long-context test.
- `benches/results/attention-shared-probability-padding/`: exact probability-layout comparison.
- `benches/results/attention-shared-single-probability/` and
  `attention-shared-single-padding/`: numerical ablation and subsequent exact layout comparison.
- `artifacts/bf16-invoice-500/invoice-candidate-single/`: 557.62 tokens/s invoice
  measurements, complete predictions and shared-GPU telemetry.
- `benches/results/bf16-long-invoice-rebuild/`: ten-sample full-model Criterion
  benchmark, raw latency samples, source snapshots and telemetry.
- `tests/results/bf16-long-20261012/throughput.json`: 539.25 tokens/s graph-rebuild
  result and inverted 99% mean-latency confidence interval.
- `tests/results/bf16-long-20261012/manifest.json` and `status.json`: frozen candidate,
  experiment environment, evaluation queue and completion state.

The frozen six-mode qualification matrix retains its original executable and
completed checkpoints; experimental runs must not replace its binary or change
its pinned runner scripts.
The candidate queue resumes that matrix automatically after qualification finishes
or fails. Its status currently retains all completed baseline checkpoints.
