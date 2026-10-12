# Long-context INT4 matrix multiplies

The October 12, 2026 sweep tested ten tile and row-grouping choices on shared
gfx1151 hardware. All 40 paired cases preserved output bits and exact replay.
No candidate was selected for inference: the consistent improvements were too
small to establish a useful full-model gain.

## Workload and results

The [36,916-token invoice profile](attention-precision.md#long-invoice-profile)
spends 43.85% of serialized GPU time in INT4 matrix multiplies. Its chunked MLP
uses 8,192-row blocks and a 4,148-row tail. The previous paired integer benchmark
silently excluded lengths above 4,032; it now measures these actual shapes.

Up/gate multiplies `[M, 5120]` by `[17408, 5120]` transposed. Down multiplies
`[M, 17408]` by `[5120, 17408]` transposed. References use the native kernels for
these lengths: M256/N128/K128 for up/gate, M256/N128/K256 for down, row grouping
4 in both. Operands use padded INT4 storage, tiled weights and distinct FP32
row scales, including zero scales. Outputs are BF16.

Each table entry spans the replay and rotating-weight estimates. Ratios above
1 mean faster than the reference. Timing alternates execution order, excludes
warmup, and uses 20 paired batches with 10,000 bootstrap resamples and 99%
confidence intervals. Rotating cases retain at least 64 MiB of weights per
variant. These are prepacked operator timings, excluding activation packing,
compilation and model preparation.

| Candidate | 4,148 rows | 8,192 rows |
| --- | ---: | ---: |
| Up/gate, K256 | 1.008–1.010× | 1.011–1.013× |
| Up/gate, M128/K256 | 1.016× | 0.986–0.987× |
| Down, K128 | 0.983–0.985× | 0.975–0.978× |
| Down, M128/K256 | 0.997× | 0.964× |
| Up/gate, grouping 2 | 1.002× | 1.000–1.003× |
| Up/gate, grouping 8 | 0.985–0.986× | 0.995–0.996× |
| Up/gate, grouping 16 | 0.914–0.918× | 0.981–0.983× |
| Down, grouping 2 | 1.003–1.005× | 1.000–1.004× |
| Down, grouping 8 | 0.991–0.994× | 0.993–0.997× |
| Down, grouping 16 | 0.936–0.949× | 0.972–0.978× |

The K256 up/gate candidate's four 99% lower bounds exceed 1.0. Its gains are
0.8–1.3%, below the benchmark's 3% practical noise threshold. Applying these
operator estimates to the profile projects only about 0.2–0.3% less total GPU
time; that is an estimate, not an observed request speedup. M128 helps the tail
but regresses the dominant aligned block. Larger row groups regress both
projections. No full-model accuracy gate was run for these unselected candidates.

## Memory bounds

Packed integer fixtures accept lengths through 65,536, subject to a conservative
1 GiB allocation estimate per variant. The check runs before opening a GPU or
constructing host fixtures. It counts packed activations, scales, original and
tiled weight copies, rotating outputs, residual intermediates and allocator
headroom. The pair also checks actual combined engine allocation against 2 GiB.
Oversized shapes fail explicitly; they are not silently skipped. A CPU test
checks rejection without a GPU. BF16 pairs retain their existing 1 GiB cap.

The recorded sweep ran GPU jobs sequentially under the qualification lock, with
at least 26 GiB available at admission. Its external watchdog stopped owned
processes below 20 GiB available or if swap grew by more than 1 GiB. Every job
completed without triggering the guard. The benchmark's allocation check does
not itself provide that host-memory watchdog.

## Reproduce

Measure the wider up/gate reduction against its native reference:

```sh
CLEF_BENCH_INTEGER_BITS=4 CLEF_BENCH_INTEGER_PADDED=1 \
CLEF_BENCH_INTEGER_ROW_SCALES=1 CLEF_BENCH_LENGTHS=4148,8192 \
CLEF_BENCH_INTEGER_SOURCE=kernels/gemm_integer_wide_k.loom \
CLEF_BENCH_INTEGER_REFERENCE_SOURCE=kernels/gemm_integer.loom \
  python3 scripts/bench_workload.py int4-long-up-wide-repro \
    --suite workload --filter paired_integer_gemm/mlp_up_gate
```

Add `--test` for correctness only. For M128, use `gemm_integer_m128.loom` as
the candidate. For down projection, filter `paired_integer_gemm/mlp_down` and
use `gemm_integer_wide_k.loom` as the reference; test `gemm_integer.loom` and
`gemm_integer_m128.loom` as candidates. Use a fresh result name for each run.

For grouping sweeps, make copies with the configuration bound extended:

```sh
python3 - <<'PY'
from pathlib import Path
out = Path('artifacts/int4-long-groups')
out.mkdir(parents=True, exist_ok=True)
for name in ('gemm_integer', 'gemm_integer_wide_k'):
    source = Path('kernels', name + '.loom').read_text()
    source = source.replace('range(%value, 1, 4)', 'range(%value, 1, 16)')
    (out / (name + '.loom')).write_text(source)
PY
```

Use the generated `gemm_integer.loom` for up/gate or `gemm_integer_wide_k.loom`
for down as `CLEF_BENCH_INTEGER_SOURCE`. Keep the native reference and vary
`CLEF_BENCH_INTEGER_M_GROUP=2`, `8`, `16`. This changes only candidate grouping;
the reference remains at 4.

Local paired records, source snapshots, hashes and telemetry are under
`benches/results/int4-long-{up-wide,up-m128,down-narrow,down-m128,up-group2,up-group8,up-group16,down-group2,down-group8,down-group16}-20261012/`.
The guarded sweep status is
`tests/results/int4-long-gemm-20261012/sweep-status.json`; its supervisor and
generated grouping sources are under `artifacts/int4-long-gemm-20261012/`.
