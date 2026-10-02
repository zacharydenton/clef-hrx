use clef_hrx::benchmarking::{GemmBenchmark, GemmKernel};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::time::{Duration, Instant};

fn benches(c: &mut Criterion) {
    comparisons(c, false);
    comparisons(c, true);
}

fn comparisons(c: &mut Criterion, wide_reference: bool) {
    let mut group = c.benchmark_group(if wide_reference {
        "gemm_pipelined"
    } else {
        "gemm"
    });
    group
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    for (m, k, n) in [
        (7, 1024, 1024),
        (65, 5120, 5120),
        (128, 5120, 17408),
        (128, 5120, 5120),
        (260, 5120, 17408),
        (260, 17408, 5120),
        (260, 6144, 5120),
        (260, 5120, 1024),
        (256, 1280, 5120),
        (1024, 5120, 17408),
    ] {
        if wide_reference && (m < 128 || n < 4096) {
            continue;
        }
        let mut reference = GemmBenchmark::with_kernel(
            m,
            k,
            n,
            if wide_reference {
                GemmKernel::Wide
            } else {
                GemmKernel::Reference
            },
        )
        .unwrap();
        let mut optimized = GemmBenchmark::new(m, k, n, true).unwrap();
        assert_eq!(
            reference.output().unwrap(),
            optimized.output().unwrap(),
            "GEMM output must preserve rounding"
        );
        let mut old_total = Duration::ZERO;
        let mut new_total = Duration::ZERO;
        let mut runs = 0u64;
        group.bench_function(BenchmarkId::new("linear", format!("{m}x{k}x{n}")), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    for candidate in [runs.is_multiple_of(2), !runs.is_multiple_of(2)] {
                        let started = Instant::now();
                        if candidate {
                            optimized.run().unwrap();
                            let duration = started.elapsed();
                            elapsed += duration;
                            new_total += duration;
                        } else {
                            reference.run().unwrap();
                            old_total += started.elapsed();
                        }
                    }
                    runs += 1;
                }
                elapsed
            });
        });
        if runs > 0 {
            eprintln!(
                "{m}x{k}x{n}: reference={:.3} ms, optimized={:.3} ms, speedup={:.2}x ({runs} pairs)",
                old_total.as_secs_f64() * 1000. / runs as f64,
                new_total.as_secs_f64() * 1000. / runs as f64,
                old_total.as_secs_f64() / new_total.as_secs_f64()
            );
        }
    }
    group.finish();
}
criterion_group!(gemm, benches);
criterion_main!(gemm);
