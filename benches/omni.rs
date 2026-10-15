use clef_hrx::{
    Precision,
    benchmarking::{
        AttentionBenchmark, AttentionFixture, OmniCombineBenchmark, OmniExpertBenchmark,
        OmniRouteBenchmark, OmniRouterBenchmark, PackingBenchmark, SwiGluBenchmark,
    },
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::time::{Duration, Instant};
fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("omni_swiglu");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    for tokens in [1, 17, 256, 1024, 2048] {
        let id = format!("swiglu/{tokens}");
        if std::env::var("CLEF_BENCH_OMNI_FILTER").is_ok_and(|filter| !id.contains(&filter)) {
            continue;
        }
        let mut bench = SwiGluBenchmark::new(tokens * 8, 768).unwrap();
        let mut totals = [Duration::ZERO; 2];
        let mut runs = 0u64;
        group.bench_function(BenchmarkId::new("paired", &id), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    for new in [runs.is_multiple_of(2), !runs.is_multiple_of(2)] {
                        let start = Instant::now();
                        bench.run(new).unwrap();
                        let time = start.elapsed();
                        totals[usize::from(new)] += time;
                        if new {
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
                "{id}: reference={:.3} ms, fused={:.3} ms, speedup={:.2}x ({runs} pairs)",
                totals[0].as_secs_f64() * 1e3 / runs as f64,
                totals[1].as_secs_f64() * 1e3 / runs as f64,
                totals[0].as_secs_f64() / totals[1].as_secs_f64()
            );
        }
    }
    group.finish();
    let mut group = c.benchmark_group("omni_swiglu_packing");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    for tokens in [1, 17, 256, 1024, 2048] {
        let id = format!("packing/{tokens}");
        if std::env::var("CLEF_BENCH_OMNI_FILTER").is_ok_and(|filter| !id.contains(&filter)) {
            continue;
        }
        let mut bench = PackingBenchmark::swiglu(tokens * 8, 768, 4).unwrap();
        let mut totals = [Duration::ZERO; 2];
        let mut runs = 0u64;
        group.bench_function(BenchmarkId::new("paired", &id), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    for new in [runs.is_multiple_of(2), !runs.is_multiple_of(2)] {
                        let start = Instant::now();
                        bench.run(new).unwrap();
                        let time = start.elapsed();
                        totals[usize::from(new)] += time;
                        if new {
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
                "{id}: reference={:.3} ms, fused={:.3} ms, speedup={:.2}x ({runs} pairs)",
                totals[0].as_secs_f64() * 1e3 / runs as f64,
                totals[1].as_secs_f64() * 1e3 / runs as f64,
                totals[0].as_secs_f64() / totals[1].as_secs_f64()
            );
        }
    }
    group.finish();
    let mut group = c.benchmark_group("omni_experts");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    let token_counts = std::env::var("CLEF_BENCH_OMNI_TOKENS")
        .map(|value| {
            value
                .split(',')
                .map(|s| s.parse::<usize>().unwrap())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|_| vec![1, 17, 128, 256]);
    for precision in [Precision::Bf16, Precision::Int4] {
        for &tokens in &token_counts {
            for down in [false, true] {
                for skewed in [false, true] {
                    let id = format!(
                        "{precision:?}/{tokens}/{}/{skewed}",
                        if down { "down" } else { "up" }
                    );
                    // Select cases before allocating fixtures (Criterion filters only timing).
                    if std::env::var("CLEF_BENCH_OMNI_FILTER")
                        .is_ok_and(|filter| !id.contains(&filter))
                    {
                        continue;
                    }
                    let mut bench =
                        OmniExpertBenchmark::new(tokens, down, precision, skewed).unwrap();
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
                            "{id}: reference={:.3} ms, grouped={:.3} ms, speedup={:.2}x ({runs} pairs)",
                            totals[0].as_secs_f64() * 1000. / runs as f64,
                            totals[1].as_secs_f64() * 1000. / runs as f64,
                            totals[0].as_secs_f64() / totals[1].as_secs_f64()
                        );
                    }
                }
            }
        }
    }
    group.finish();
    let mut group = c.benchmark_group("omni_routes");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    for tokens in [1, 17, 64, 65, 128, 255, 256] {
        for skewed in [false, true] {
            let id = format!("routes/{tokens}/{skewed}");
            if std::env::var("CLEF_BENCH_OMNI_FILTER").is_ok_and(|filter| !id.contains(&filter)) {
                continue;
            }
            let mut bench = OmniRouteBenchmark::new(tokens, skewed).unwrap();
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
            if runs > 0 {
                eprintln!(
                    "{id}: serial={:.3} us, parallel={:.3} us, speedup={:.2}x ({runs} pairs)",
                    totals[0].as_secs_f64() * 1e6 / runs as f64,
                    totals[1].as_secs_f64() * 1e6 / runs as f64,
                    totals[0].as_secs_f64() / totals[1].as_secs_f64()
                );
            }
        }
    }
    group.finish();
    let mut group = c.benchmark_group("omni_topk");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    for (tokens, experts, topk) in [
        (1, 128, 8),
        (17, 128, 8),
        (256, 128, 8),
        (1024, 128, 8),
        (3, 1, 1),
        (7, 31, 8),
        (65, 33, 8),
        (129, 127, 8),
        (2, 128, 128),
    ] {
        for bf16 in [true, false] {
            let id = format!("topk/{tokens}/{experts}/{topk}/{bf16}");
            if std::env::var("CLEF_BENCH_OMNI_FILTER").is_ok_and(|filter| !id.contains(&filter)) {
                continue;
            }
            let mut bench = OmniRouterBenchmark::new(tokens, experts, topk, bf16).unwrap();
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
            if runs > 0 {
                eprintln!(
                    "{id}: serial={:.3} us, parallel={:.3} us, speedup={:.2}x ({runs} pairs)",
                    totals[0].as_secs_f64() * 1e6 / runs as f64,
                    totals[1].as_secs_f64() * 1e6 / runs as f64,
                    totals[0].as_secs_f64() / totals[1].as_secs_f64()
                );
            }
        }
    }
    group.finish();
    let mut group = c.benchmark_group("omni_attention");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    let attention_counts = std::env::var("CLEF_BENCH_OMNI_ATTENTION_TOKENS")
        .map(|value| {
            value
                .split(',')
                .map(|s| s.parse::<usize>().unwrap())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|_| vec![1, 17, 65, 127, 256, 1024, 2048, 4096]);
    for tokens in attention_counts {
        let id = format!("attention/{tokens}");
        if std::env::var("CLEF_BENCH_OMNI_FILTER").is_ok_and(|filter| !id.contains(&filter)) {
            continue;
        }
        let model = clef_hrx::ModelVariant::ClefOmni;
        let fixture = match std::env::var("CLEF_BENCH_OMNI_ATTENTION_FIXTURE").as_deref() {
            Ok("hashed") => AttentionFixture::Hashed,
            Ok("zero-queries") => AttentionFixture::ZeroQueries,
            Ok("patterned") | Err(_) => AttentionFixture::Patterned,
            Ok(value) => panic!("unknown attention fixture: {value}"),
        };
        let reference_source = std::env::var("CLEF_BENCH_OMNI_ATTENTION_REFERENCE")
            .ok()
            .map(|path| std::fs::read_to_string(path).unwrap());
        let reference_prepacked =
            std::env::var_os("CLEF_BENCH_OMNI_ATTENTION_REFERENCE_PREPACKED").is_some();
        let mut reference = if reference_prepacked {
            AttentionBenchmark::prepacked_case(
                tokens,
                reference_source
                    .as_deref()
                    .expect("prepacked reference requires an explicit source"),
                model,
                fixture,
            )
        } else {
            AttentionBenchmark::case(tokens, reference_source.as_deref(), false, model, fixture)
        }
        .unwrap();
        let source = std::env::var("CLEF_BENCH_OMNI_ATTENTION_SOURCE")
            .map(|path| std::fs::read_to_string(path).unwrap())
            .unwrap_or_else(|_| include_str!("../kernels/attention_tiled_128.loom").into());
        let prepacked = std::env::var_os("CLEF_BENCH_OMNI_PREPACK_QK").is_some();
        let mut candidate = if prepacked {
            assert!(
                std::env::var_os("CLEF_BENCH_OMNI_ATTENTION_SOURCE").is_some(),
                "prepacked attention requires an explicit compatible source"
            );
            AttentionBenchmark::prepacked_case(tokens, &source, model, fixture)
        } else {
            AttentionBenchmark::case(tokens, Some(&source), false, model, fixture)
        }
        .unwrap();
        let a = reference.output().unwrap();
        let b = candidate.output().unwrap();
        assert_eq!(a.len(), b.len());
        assert!(a.iter().all(|v| v.is_finite()));
        let max_error = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            b.iter().all(|v| v.is_finite()) && max_error <= 0.015625,
            "{id}: attention error {max_error}"
        );
        let mut totals = [Duration::ZERO; 2];
        let mut runs = 0u64;
        group.bench_function(BenchmarkId::new("paired", &id), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    for new in [runs.is_multiple_of(2), !runs.is_multiple_of(2)] {
                        let start = Instant::now();
                        if new {
                            candidate.run().unwrap();
                        } else {
                            reference.run().unwrap();
                        }
                        let time = start.elapsed();
                        totals[usize::from(new)] += time;
                        if new {
                            elapsed += time;
                        }
                    }
                    runs += 1;
                }
                elapsed
            })
        });
        assert_eq!(candidate.output().unwrap(), b);
        if runs > 0 {
            eprintln!(
                "{id}: reference={:.3} ms, candidate={:.3} ms, speedup={:.2}x, max_error={max_error}, prepacked={prepacked}, reference_prepacked={reference_prepacked}, fixture={fixture:?} ({runs} pairs)",
                totals[0].as_secs_f64() * 1e3 / runs as f64,
                totals[1].as_secs_f64() * 1e3 / runs as f64,
                totals[0].as_secs_f64() / totals[1].as_secs_f64()
            );
        }
    }
    group.finish();
    let mut group = c.benchmark_group("omni_combine");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    for tokens in [1, 17, 65, 256, 1024, 2048] {
        for skewed in [false, true] {
            let id = format!("combine/{tokens}/{skewed}");
            if std::env::var("CLEF_BENCH_OMNI_FILTER").is_ok_and(|filter| !id.contains(&filter)) {
                continue;
            }
            let mut bench = OmniCombineBenchmark::new(tokens, skewed).unwrap();
            let mut totals = [Duration::ZERO; 2];
            let mut runs = 0u64;
            group.bench_function(BenchmarkId::new("paired", &id), |b| {
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for _ in 0..iterations {
                        for new in [runs.is_multiple_of(2), !runs.is_multiple_of(2)] {
                            let start = Instant::now();
                            bench.run(new).unwrap();
                            let time = start.elapsed();
                            totals[usize::from(new)] += time;
                            if new {
                                elapsed += time;
                            }
                        }
                        runs += 1;
                    }
                    elapsed
                })
            });
            if runs > 0 {
                eprintln!(
                    "{id}: reference={:.3} ms, ordered={:.3} ms, speedup={:.2}x ({runs} pairs)",
                    totals[0].as_secs_f64() * 1e3 / runs as f64,
                    totals[1].as_secs_f64() * 1e3 / runs as f64,
                    totals[0].as_secs_f64() / totals[1].as_secs_f64()
                );
            }
        }
    }
    group.finish();
}
criterion_group!(omni, benches);
criterion_main!(omni);
