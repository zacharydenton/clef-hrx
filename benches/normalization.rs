use clef_hrx::benchmarking::NormBenchmark;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::time::{Duration, Instant};

fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    for (rows, width, layer) in [
        (260, 5120, false),
        (260, 1024, true),
        (6240, 256, false),
        (7, 1024, true),
    ] {
        let mut reference = NormBenchmark::new(rows, width, layer, false).unwrap();
        let mut parallel = NormBenchmark::new(rows, width, layer, true).unwrap();
        let old = reference.output().unwrap();
        let new = parallel.output().unwrap();
        assert!(old.iter().zip(&new).all(|(a, b)| a.is_finite()
            && b.is_finite()
            && (a - b).abs() <= a.abs().max(b.abs()) / 128. + 1e-6));
        let mut old_total = Duration::ZERO;
        let mut new_total = Duration::ZERO;
        let mut runs = 0u64;
        group.bench_function(
            BenchmarkId::new(
                if layer { "layer" } else { "rms" },
                format!("{rows}x{width}"),
            ),
            |b| {
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for _ in 0..iterations {
                        for candidate in [runs.is_multiple_of(2), !runs.is_multiple_of(2)] {
                            let started = Instant::now();
                            if candidate {
                                parallel.run().unwrap();
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
            },
        );
        eprintln!(
            "{rows}x{width} layer={layer}: reference={:.3} ms, parallel={:.3} ms, speedup={:.2}x ({runs} pairs)",
            old_total.as_secs_f64() * 1000. / runs as f64,
            new_total.as_secs_f64() * 1000. / runs as f64,
            old_total.as_secs_f64() / new_total.as_secs_f64()
        );
    }
    group.finish();
}
criterion_group!(normalization, benches);
criterion_main!(normalization);
