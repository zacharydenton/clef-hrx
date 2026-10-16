use clef_hrx::benchmarking::OmniNormRopeBenchmark;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::time::{Duration, Instant};

fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("omni_norm_rope");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    let tokens = std::env::var("CLEF_BENCH_OMNI_TOKENS")
        .unwrap_or_else(|_| "1,17,128,512,4096,16384".into())
        .split(',')
        .map(|v| v.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    for tokens in tokens {
        for heads in [4, 32] {
            let id = format!("{tokens}/{heads}");
            let mut bench = OmniNormRopeBenchmark::new(tokens, heads).unwrap();
            let mut totals = [Duration::ZERO; 2];
            let mut runs = 0u64;
            group.bench_function(BenchmarkId::new("paired", &id), |b| {
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for _ in 0..iterations {
                        for candidate in [runs.is_multiple_of(2), !runs.is_multiple_of(2)] {
                            let start = Instant::now();
                            bench.run(candidate).unwrap();
                            let time = start.elapsed();
                            totals[usize::from(candidate)] += time;
                            if candidate {
                                elapsed += time;
                            }
                        }
                        runs += 1;
                    }
                    elapsed
                })
            });
            bench.check().unwrap();
            if runs > 0 {
                eprintln!(
                    "{id}: reference={:.3} us, fused={:.3} us, speedup={:.2}x ({runs} pairs)",
                    totals[0].as_secs_f64() * 1e6 / runs as f64,
                    totals[1].as_secs_f64() * 1e6 / runs as f64,
                    totals[0].as_secs_f64() / totals[1].as_secs_f64()
                );
            }
        }
    }
    group.finish();
}
criterion_group!(norm_rope, benches);
criterion_main!(norm_rope);
