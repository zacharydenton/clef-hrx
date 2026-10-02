use clef_hrx::benchmarking::DeltaBenchmark;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::time::{Duration, Instant};
mod support;

fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_prefill");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    for tokens in [65, 260, 1024] {
        group.throughput(Throughput::Elements(tokens as u64));
        for (name, chunked) in [("recurrent", false), ("chunked", true)] {
            // Each graph is released before creating the next; no model weights.
            let mut graph = DeltaBenchmark::new(tokens, chunked).unwrap();
            let allocated = graph.allocated_bytes();
            assert!(allocated <= allocation_limit());
            group.bench_function(BenchmarkId::new(name, tokens), |b| {
                b.iter(|| graph.run().unwrap())
            });
            assert_eq!(
                graph.allocated_bytes(),
                allocated,
                "replay must not grow allocation capacity"
            );
        }
    }
    group.finish();
}
fn allocation_limit() -> usize {
    let config: serde_json::Value = serde_json::from_str(include_str!("criteria.json")).unwrap();
    config["allocation_limit_bytes"].as_u64().unwrap() as usize
}

fn paired(c: &mut Criterion) {
    let run_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string();
    let fingerprint = support::fingerprint();
    let mut group = c.benchmark_group("paired_delta_prefill");
    group
        .sample_size(40)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(8));
    for tokens in [65, 260, 1024] {
        let mut reference = DeltaBenchmark::reference(tokens).unwrap();
        let mut optimized = DeltaBenchmark::new(tokens, true).unwrap();
        let allocated = reference.allocated_bytes() + optimized.allocated_bytes();
        assert!(
            allocated <= allocation_limit(),
            "combined old/new allocation exceeds budget"
        );
        {
            let before = reference.output().unwrap();
            let after = optimized.output().unwrap();
            assert_eq!(before.len(), after.len());
            assert!(
                before
                    .iter()
                    .zip(&after)
                    .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() < 0.002)
            );
        }
        let mut samples = Vec::new();
        let mut sequence = 0u64;
        group.throughput(Throughput::Elements(tokens as u64));
        group.bench_function(BenchmarkId::new("chunked",tokens),|b| b.iter_custom(|iterations| {
            let mut old = Duration::ZERO;
            let mut new = Duration::ZERO;
            for _ in 0..iterations {
                // Alternate order on each pair to reduce drift bias. Both
                // paths see the same background load within milliseconds.
                for candidate in [sequence.is_multiple_of(2),!sequence.is_multiple_of(2)] {
                    let started = Instant::now();
                    if candidate {optimized.run().unwrap();new+=started.elapsed();}
                    else {reference.run().unwrap();old+=started.elapsed();}
                }
                sequence+=1;
            }
            samples.push(serde_json::json!({"iterations":iterations,"reference_ns":old.as_nanos() as u64,"optimized_ns":new.as_nanos() as u64}));
            // Criterion reports candidate time; reference timing is saved
            // separately for paired acceptance checks.
            new
        }));
        assert_eq!(
            reference.allocated_bytes() + optimized.allocated_bytes(),
            allocated
        );
        let root = std::path::PathBuf::from("target/criterion/paired_delta_prefill/chunked")
            .join(tokens.to_string());
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("paired.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
            "tokens":tokens,"allocated_bytes":allocated,"sample_size":40,"samples":samples,
            "run_id":run_id,"fingerprint":fingerprint
            }))
            .unwrap(),
        )
        .unwrap();
    }
    group.finish();
}
criterion_group!(kernels, benches, paired);
criterion_main!(kernels);
