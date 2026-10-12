use clef_hrx::benchmarking::DeltaBenchmark;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::time::{Duration, Instant};
mod support;

fn benches(c: &mut Criterion) {
    if std::env::var("CLEF_BENCH_MODEL").as_deref() == Ok("clef-flash") {
        return;
    }
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
    let measurement_seconds = std::env::var("CLEF_BENCH_PAIRED_SECONDS")
        .map(|value| {
            value
                .parse::<u64>()
                .expect("CLEF_BENCH_PAIRED_SECONDS must be an integer")
        })
        .unwrap_or(8);
    assert!((1..=300).contains(&measurement_seconds));
    let flash = std::env::var("CLEF_BENCH_MODEL").as_deref() == Ok("clef-flash");
    let group_name = if flash {
        "paired_flash_delta"
    } else {
        "paired_delta_prefill"
    };
    let mut group = c.benchmark_group(group_name);
    group
        .sample_size(40)
        .confidence_level(0.99)
        .significance_level(0.01)
        .noise_threshold(0.03)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(measurement_seconds));
    let candidate_source = std::env::var_os("CLEF_BENCH_DELTA_SOURCE")
        .map(|path| std::fs::read_to_string(path).unwrap());
    let reference_source = std::env::var_os("CLEF_BENCH_DELTA_REFERENCE_SOURCE")
        .map(|path| std::fs::read_to_string(path).unwrap());
    let prepare_source = std::env::var_os("CLEF_BENCH_DELTA_PREPARE_SOURCE")
        .map(|path| std::fs::read_to_string(path).unwrap());
    assert!(prepare_source.is_none() || candidate_source.is_some());
    let values = std::env::var("CLEF_BENCH_DELTA_VALUES")
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(8);
    let mut cases = vec![
        ("chunked", 65),
        ("chunked", 260),
        ("chunked", 1024),
        ("persistent", 235),
        ("persistent", 620),
        ("persistent", 1024),
        ("prepared", 158),
        ("prepared", 235),
        ("prepared", 620),
        ("prepared", 1024),
        ("prepared", 2048),
        ("gated_norm", 260),
        ("gated_norm", 1024),
    ];
    if candidate_source.is_some() {
        cases.extend([128, 158, 235, 260, 620, 1024, 2048].map(|tokens| ("candidate", tokens)));
        // Two unprepared graphs fit the 256 MiB budget at the longest JevBench
        // input. Prepared graphs need extra token storage and do not fit.
        if prepare_source.is_none() {
            cases.push(("candidate", 4032));
        }
    }
    if flash {
        assert!(candidate_source.is_some() && prepare_source.is_none());
        cases = std::env::var("CLEF_BENCH_LENGTHS")
            .unwrap_or_else(|_| "1,65,127,128,158,235,260,620,1024,2048,4032".into())
            .split(',')
            .map(|n| ("candidate", n.parse().unwrap()))
            .collect();
    } else if candidate_source.is_some()
        && let Ok(lengths) = std::env::var("CLEF_BENCH_LENGTHS")
    {
        let mut lengths: Vec<usize> = lengths.split(',').map(|n| n.parse().unwrap()).collect();
        assert!(!lengths.is_empty() && lengths.len() <= 64);
        assert!(lengths.iter().all(|n| (1..=65536).contains(n)));
        assert!(prepare_source.is_none() || lengths.iter().all(|&n| n <= 4096));
        lengths.sort_unstable();
        lengths.dedup();
        cases = lengths.into_iter().map(|n| ("candidate", n)).collect();
    }
    for (name, tokens) in cases {
        let norm = name == "gated_norm";
        let long = !flash && tokens > 4096;
        let mut reference = if long {
            DeltaBenchmark::for_model_source(
                tokens,
                clef_hrx::ModelVariant::Clef,
                reference_source
                    .as_deref()
                    .unwrap_or(include_str!("../kernels/delta_persistent.loom")),
                8,
            )
        } else if flash {
            if let Some(source) = &reference_source {
                DeltaBenchmark::for_model_source(
                    tokens,
                    clef_hrx::ModelVariant::ClefFlash,
                    source,
                    8,
                )
            } else {
                DeltaBenchmark::for_model(tokens, clef_hrx::ModelVariant::ClefFlash)
            }
        } else if norm {
            DeltaBenchmark::gated_norm(tokens, false)
        } else if name == "candidate"
            && let Some(source) = &reference_source
        {
            DeltaBenchmark::persistent(tokens, source, 8)
        } else if matches!(name, "prepared" | "candidate") {
            DeltaBenchmark::resident(tokens)
        } else if name == "persistent" {
            DeltaBenchmark::new(tokens, true)
        } else {
            DeltaBenchmark::reference(tokens)
        }
        .unwrap();
        let mut optimized = if long {
            DeltaBenchmark::for_model_source(
                tokens,
                clef_hrx::ModelVariant::Clef,
                candidate_source.as_ref().unwrap(),
                values,
            )
        } else if flash {
            DeltaBenchmark::for_model_source(
                tokens,
                clef_hrx::ModelVariant::ClefFlash,
                candidate_source.as_ref().unwrap(),
                values,
            )
        } else if norm {
            DeltaBenchmark::gated_norm(tokens, true)
        } else if name == "candidate" {
            if let Some(prepare_source) = &prepare_source {
                DeltaBenchmark::persistent_prepared(
                    tokens,
                    prepare_source,
                    candidate_source.as_ref().unwrap(),
                    values,
                )
            } else {
                DeltaBenchmark::persistent(tokens, candidate_source.as_ref().unwrap(), values)
            }
        } else if name == "prepared" {
            DeltaBenchmark::persistent_prepared(
                tokens,
                include_str!("kernels/delta_prepare.loom"),
                include_str!("kernels/delta_segmented_prepared.loom"),
                4,
            )
        } else if name == "persistent" {
            DeltaBenchmark::resident(tokens)
        } else {
            DeltaBenchmark::new(tokens, true)
        }
        .unwrap();
        let allocated = reference.allocated_bytes() + optimized.allocated_bytes();
        assert!(
            allocated
                <= if long {
                    8usize << 30
                } else {
                    allocation_limit()
                },
            "combined old/new allocation exceeds budget"
        );
        let (max_abs_error, identical, reference_output, candidate_output) = {
            let before = reference.output().unwrap();
            let after = optimized.output().unwrap();
            assert_eq!(before.len(), after.len());
            assert!(before.iter().zip(&after).all(|(a, b)| a.is_finite()
                && b.is_finite()
                && (a - b).abs()
                    < if norm {
                        // Fusing removes intermediate BF16 rounding. FP64
                        // oracle tests separately bound each output's error.
                        a.abs().max(b.abs()) * 0.02 + 1e-6
                    } else {
                        0.002
                    }));
            (
                before
                    .iter()
                    .zip(&after)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max),
                before
                    .iter()
                    .zip(&after)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                before,
                after,
            )
        };
        let mut samples = Vec::new();
        let mut sequence = 0u64;
        group.throughput(Throughput::Elements(tokens as u64));
        group.bench_function(BenchmarkId::new(name, tokens), |b| {
            b.iter_custom(|iterations| {
                let mut old = Duration::ZERO;
                let mut new = Duration::ZERO;
                for _ in 0..iterations {
                    // Alternate order on each pair to reduce drift bias. Both
                    // paths see the same background load within milliseconds.
                    for candidate in [sequence.is_multiple_of(2), !sequence.is_multiple_of(2)] {
                        let started = Instant::now();
                        if candidate {
                            optimized.run().unwrap();
                            new += started.elapsed();
                        } else {
                            reference.run().unwrap();
                            old += started.elapsed();
                        }
                    }
                    sequence += 1;
                }
                samples.push(serde_json::json!({
                    "iterations": iterations,
                    "reference_ns": old.as_nanos() as u64,
                    "optimized_ns": new.as_nanos() as u64,
                }));
                // Criterion reports candidate time; reference timing is saved
                // separately for paired acceptance checks.
                new
            });
        });
        assert_eq!(
            reference.allocated_bytes() + optimized.allocated_bytes(),
            allocated
        );
        for (actual, expected) in [
            (reference.output().unwrap(), reference_output),
            (optimized.output().unwrap(), candidate_output),
        ] {
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "DeltaNet output changed on replay"
            );
        }
        let root = std::path::PathBuf::from("target/criterion")
            .join(group_name)
            .join(name)
            .join(tokens.to_string());
        std::fs::create_dir_all(&root).unwrap();
        // Criterion invokes iter_custom during warmup, then once per measured
        // sample. Keep those phases separate for paired confidence intervals.
        let measured = if samples.len() >= 40 {
            &samples[samples.len() - 40..]
        } else {
            &[]
        };
        std::fs::write(
            root.join("paired.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "tokens": tokens,
                "max_abs_error": max_abs_error,
                "bitwise_identical": identical,
                "allocated_bytes": allocated,
                "sample_size": 40,
                "samples": samples,
                "measurement_samples": measured,
                "run_id": run_id,
                "fingerprint": fingerprint,
                "measurement_seconds": measurement_seconds,
            }))
            .unwrap(),
        )
        .unwrap();
    }
    group.finish();
}
criterion_group!(kernels, benches, paired);
criterion_main!(kernels);
