//! Flash shape coverage. Allocation, compilation and upload are outside timing.
use clef_hrx::{
    ModelVariant,
    benchmarking::{
        AttentionBenchmark, DeltaBenchmark, GemmBenchmark, GemmKernel, PackingBenchmark,
        SwiGluBenchmark,
    },
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::time::Duration;
fn lengths() -> Vec<usize> {
    std::env::var("CLEF_BENCH_LENGTHS")
        .unwrap_or_else(|_| "1,65,127,128,158,235,260,620,1024,2048,4032".into())
        .split(',')
        .map(|s| {
            let n = s.parse().unwrap();
            assert!((1..=65536).contains(&n));
            n
        })
        .collect()
}
fn flash(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .noise_threshold(0.03)
        .significance_level(0.01)
        .confidence_level(0.99);
    for rows in lengths() {
        if rows <= 4032 {
            for (role, k, n) in [
                ("mlp_up", 4096, 12288),
                ("mlp_down", 12288, 4096),
                ("attention_output", 4096, 4096),
                ("delta_qkv", 4096, 8192),
                ("delta_z", 4096, 4096),
                ("attention_q_gate", 4096, 8192),
                ("kv", 4096, 1024),
                ("delta_gate", 4096, 32),
                ("head_memory", 4096, 1024),
                ("joint_head", 1024, 1024),
            ] {
                for (cache, bytes) in [("replay", 0), ("rotating", 64 << 20)] {
                    if n >= 4096 {
                        group.throughput(Throughput::Elements(
                            2 * rows as u64 * k as u64 * n as u64,
                        ));
                        group.bench_function(
                            BenchmarkId::new(
                                format!("{role}/int4/{cache}"),
                                format!("{rows}x{k}x{n}"),
                            ),
                            |b| {
                                let mut graph = GemmBenchmark::rotated_integer(
                                    rows,
                                    k,
                                    n,
                                    include_str!("../kernels/gemm_integer.loom"),
                                    4,
                                    true,
                                    bytes,
                                )
                                .unwrap();
                                let expected = graph.output().unwrap();
                                assert!(expected.iter().all(|x| x.is_finite()));
                                b.iter(|| graph.run().unwrap());
                                assert_eq!(expected, graph.output().unwrap());
                            },
                        );
                    }
                    group.throughput(Throughput::Elements(2 * rows as u64 * k as u64 * n as u64));
                    group.bench_function(
                        BenchmarkId::new(format!("{role}/{cache}"), format!("{rows}x{k}x{n}")),
                        |b| {
                            let mut graph = GemmBenchmark::with_working_set(
                                rows,
                                k,
                                n,
                                if matches!(role, "delta_gate" | "head_memory" | "joint_head") {
                                    GemmKernel::Auto
                                } else {
                                    GemmKernel::Tiled
                                },
                                bytes,
                            )
                            .unwrap();
                            let expected = graph.output().unwrap();
                            assert!(expected.iter().all(|x| x.is_finite()));
                            b.iter(|| graph.run().unwrap());
                            assert_eq!(expected, graph.output().unwrap());
                        },
                    );
                }
            }
            for width in [4096, 8192, 12288] {
                group.throughput(Throughput::Elements(rows as u64 * width as u64));
                group.bench_function(
                    BenchmarkId::new("packing_int4", format!("{rows}x{width}")),
                    |b| {
                        let mut graph = PackingBenchmark::production(rows, width, 4).unwrap();
                        b.iter(|| graph.run(true).unwrap());
                        graph.check().unwrap();
                    },
                );
            }
            group.throughput(Throughput::Elements(rows as u64 * 12288));
            group.bench_function(BenchmarkId::new("swiglu", rows), |b| {
                let mut graph = SwiGluBenchmark::new(rows, 12288).unwrap();
                b.iter(|| graph.run(true).unwrap());
            });
        }
        group.throughput(Throughput::Elements(rows as u64));
        group.bench_function(BenchmarkId::new("attention_16q_4kv", rows), |b| {
            let mut graph =
                AttentionBenchmark::for_model(rows, ModelVariant::ClefFlash, true).unwrap();
            let expected = graph.output().unwrap();
            assert!(expected.iter().all(|x| x.is_finite()));
            b.iter(|| graph.run().unwrap());
            assert_eq!(expected, graph.output().unwrap());
        });
        group.bench_function(BenchmarkId::new("delta_32_heads", rows), |b| {
            let mut graph = DeltaBenchmark::for_model(rows, ModelVariant::ClefFlash).unwrap();
            let expected = graph.output().unwrap();
            assert!(expected.iter().all(|x| x.is_finite()));
            b.iter(|| graph.run().unwrap());
            assert_eq!(expected, graph.output().unwrap());
        });
    }
    group.finish();
}
criterion_group!(benches, flash);
criterion_main!(benches);
