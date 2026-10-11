//! Shape coverage for the public JevBench workload; no checkpoint required.
use clef_hrx::benchmarking::{
    AttentionBenchmark, DeltaBenchmark, GemmBenchmark, GemmKernel, PackingBenchmark,
    SwiGluBenchmark,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::time::{Duration, Instant};

fn lengths() -> Vec<usize> {
    match std::env::var("CLEF_BENCH_PROFILE")
        .as_deref()
        .unwrap_or("full")
    {
        "quick" => vec![235, 620, 4032],
        "full" => vec![1, 65, 127, 128, 158, 235, 260, 620, 1024, 2048, 4032],
        other => panic!("unknown CLEF_BENCH_PROFILE: {other}"),
    }
}
fn configure(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .noise_threshold(0.03)
        .significance_level(0.01)
        .confidence_level(0.99);
}
fn gemm(c: &mut Criterion) {
    let integer_source = std::env::var_os("CLEF_BENCH_INTEGER_SOURCE")
        .or_else(|| std::env::var_os("CLEF_BENCH_INT8_SOURCE"))
        .map(|p| std::fs::read_to_string(p).unwrap());
    let integer_padded = std::env::var("CLEF_BENCH_INTEGER_PADDED")
        .or_else(|_| std::env::var("CLEF_BENCH_INT8_PADDED"))
        .as_deref()
        == Ok("1");
    let integer_bits = std::env::var("CLEF_BENCH_INTEGER_BITS")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(8);
    let integer_prepare = std::env::var("CLEF_BENCH_INTEGER_PREPARE").as_deref() == Ok("1");
    assert!(
        !integer_prepare || integer_source.is_some(),
        "integer preparation requires an integer source"
    );
    assert!(
        [4, 8].contains(&integer_bits),
        "integer bits must be 4 or 8"
    );
    let padded = std::env::var("CLEF_BENCH_PAD_GEMM").as_deref() == Ok("1");
    let source =
        std::env::var_os("CLEF_BENCH_GEMM_SOURCE").map(|p| std::fs::read_to_string(p).unwrap());
    assert!(
        !padded || source.is_none(),
        "choose padding or a GEMM source override for each experiment"
    );
    assert!(
        integer_source.is_none() || (!padded && source.is_none()),
        "choose one GEMM experiment per run"
    );
    let mut group = c.benchmark_group("workload_gemm");
    configure(&mut group);
    for m in lengths() {
        // All text projection families, plus the trained decision head.
        for (role, k, n) in [
            ("mlp_up_gate", 5120, 17408),
            ("mlp_down", 17408, 5120),
            ("attention_output", 6144, 5120),
            ("delta_qkv", 5120, 10240),
            ("delta_z", 5120, 6144),
            ("attention_q_gate", 5120, 12288),
            ("kv_and_head", 5120, 1024),
            ("delta_gate", 5120, 48),
            ("joint_head", 1024, 1024),
        ] {
            for (cache, minimum_bytes) in [("replay", 0), ("rotating", 64 << 20)] {
                // FLOPs, not model input tokens. Each sample completes its GPU work.
                group.throughput(Throughput::Elements(2 * m as u64 * k as u64 * n as u64));
                let id = BenchmarkId::new(format!("{role}/{cache}"), format!("{m}x{k}x{n}"));
                let mut graph = None;
                let mut initial = Vec::new();
                group.bench_function(id, |b| {
                    let graph = graph.get_or_insert_with(|| {
                        let compatible = m >= 128 && n >= 4096 && n % 256 == 0;
                        let kernel = if compatible && padded {
                            GemmKernel::Padded
                        } else if compatible {
                            source
                                .as_ref()
                                .map(|s| GemmKernel::Source(s.clone()))
                                .unwrap_or(GemmKernel::Auto)
                        } else {
                            GemmKernel::Auto
                        };
                        let mut graph =
                            if let Some(source) = integer_source.as_ref().filter(|_| compatible) {
                                let build = if integer_prepare {
                                    GemmBenchmark::rotated_integer
                                } else {
                                    GemmBenchmark::packed_integer
                                };
                                build(m, k, n, source, integer_bits, integer_padded, minimum_bytes)
                            } else {
                                GemmBenchmark::with_working_set(m, k, n, kernel, minimum_bytes)
                            }
                            .unwrap();
                        let actual = graph.output().unwrap();
                        assert!(actual.iter().all(|x| x.is_finite()));
                        if (source.is_some() || padded || integer_source.is_some()) && compatible {
                            let expected = if integer_source.is_some() && !integer_prepare {
                                GemmBenchmark::quantized_reference(m, k, n, integer_bits)
                            } else {
                                GemmBenchmark::new(m, k, n, true)
                            }
                            .unwrap()
                            .output()
                            .unwrap();
                            let error = actual
                                .iter()
                                .zip(&expected)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0f32, f32::max);
                            eprintln!("GEMM {m}x{k}x{n}: maximum output change={error}");
                            if integer_prepare {
                                let squared_error: f64 = actual
                                    .iter()
                                    .zip(&expected)
                                    .map(|(a, b)| f64::from(a - b).powi(2))
                                    .sum();
                                let squared_norm: f64 =
                                    expected.iter().map(|x| f64::from(*x).powi(2)).sum();
                                eprintln!(
                                    "Rotated INT{integer_bits}: relative L2 error={}",
                                    (squared_error / squared_norm.max(f64::MIN_POSITIVE)).sqrt()
                                );
                            }
                            if padded {
                                assert_eq!(actual, expected, "stride padding changed arithmetic");
                            }
                            if integer_source.is_some() && !integer_prepare {
                                assert!(
                                    actual.iter().zip(&expected).all(|(a, b)| (a - b).abs()
                                        <= a.abs().max(b.abs()) / 128. + 1e-6),
                                    "exactly representable integer inputs exceed one BF16 step"
                                );
                            }
                        }
                        initial = actual;
                        graph
                    });
                    b.iter(|| graph.run().unwrap());
                });
                if let Some(mut graph) = graph {
                    assert_eq!(
                        graph.output().unwrap(),
                        initial,
                        "GEMM replay changed output"
                    );
                }
            }
        }
    }
    group.finish();
}
fn delta(c: &mut Criterion) {
    let source =
        std::env::var_os("CLEF_BENCH_DELTA_SOURCE").map(|p| std::fs::read_to_string(p).unwrap());
    let prepare_source = std::env::var_os("CLEF_BENCH_DELTA_PREPARE_SOURCE")
        .map(|p| std::fs::read_to_string(p).unwrap());
    assert!(
        prepare_source.is_none() || source.is_some(),
        "preparation requires CLEF_BENCH_DELTA_SOURCE"
    );
    let values = std::env::var("CLEF_BENCH_DELTA_VALUES")
        .map(|s| s.parse().unwrap())
        .unwrap_or(8);
    let mut group = c.benchmark_group("workload_delta");
    configure(&mut group);
    for tokens in lengths() {
        group.throughput(Throughput::Elements(tokens as u64));
        for name in ["chunked", "persistent"] {
            let mut graph = None;
            let mut expected = Vec::new();
            let mut allocated = 0;
            group.bench_function(BenchmarkId::new(name, tokens), |b| {
                let graph = graph.get_or_insert_with(|| {
                    let mut graph = if name == "chunked" {
                        DeltaBenchmark::new(tokens, true)
                    } else if let Some(source) = &source {
                        if let Some(prepare) = &prepare_source {
                            DeltaBenchmark::persistent_prepared(tokens, prepare, source, values)
                        } else {
                            DeltaBenchmark::persistent(tokens, source, values)
                        }
                    } else {
                        DeltaBenchmark::resident(tokens)
                    }
                    .unwrap();
                    expected = graph.output().unwrap();
                    assert!(expected.iter().all(|x| x.is_finite()));
                    if name == "persistent" {
                        let reference =
                            DeltaBenchmark::new(tokens, true).unwrap().output().unwrap();
                        let error = expected
                            .iter()
                            .zip(&reference)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f32, f32::max);
                        eprintln!(
                            "persistent DeltaNet tokens={tokens}: maximum output change={error}"
                        );
                        assert!(error < 0.002, "DeltaNet output error {error}");
                    }
                    allocated = graph.allocated_bytes();
                    graph
                });
                b.iter(|| graph.run().unwrap());
            });
            if let Some(mut graph) = graph {
                assert_eq!(
                    graph.output().unwrap(),
                    expected,
                    "state reset/replay changed output"
                );
                assert_eq!(
                    graph.allocated_bytes(),
                    allocated,
                    "replay allocated memory"
                );
            }
        }
    }
    group.finish();
}
fn attention(c: &mut Criterion) {
    let reference = std::env::var("CLEF_BENCH_ATTENTION_REFERENCE").as_deref() == Ok("1");
    let source = std::env::var_os("CLEF_BENCH_ATTENTION_SOURCE")
        .map(|p| std::fs::read_to_string(p).unwrap());
    assert!(
        !reference || source.is_none(),
        "choose reference attention or a source override"
    );
    let integrated = !reference && source.is_none();
    let mut group = c.benchmark_group("workload_attention");
    configure(&mut group);
    // Keep IDs stable for recorded baselines; operator tests cover partial tiles.
    for tokens in [16, 64, 128, 256, 624, 1024, 2048, 4032] {
        group.throughput(Throughput::Elements(tokens as u64));
        let mut graph = None;
        let mut actual = Vec::new();
        group.bench_function(BenchmarkId::new("causal_24q_4kv_d256", tokens), |b| {
            let graph = graph.get_or_insert_with(|| {
                let mut graph = if integrated {
                    AttentionBenchmark::integrated(tokens)
                } else {
                    AttentionBenchmark::new(tokens, source.as_deref())
                }
                .unwrap();
                actual = graph.output().unwrap();
                assert!(actual.iter().all(|x| x.is_finite()));
                if source.is_some() || integrated {
                    let expected = AttentionBenchmark::new(tokens, None)
                        .unwrap()
                        .output()
                        .unwrap();
                    let error = actual
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    eprintln!("attention tokens={tokens}: maximum output change={error}");
                }
                graph
            });
            b.iter(|| graph.run().unwrap());
        });
        if let Some(mut graph) = graph {
            assert_eq!(graph.output().unwrap(), actual);
        }
    }
    group.finish();
}
fn swiglu(c: &mut Criterion) {
    let mut group = c.benchmark_group("paired_swiglu");
    configure(&mut group);
    for rows in lengths() {
        let mut graph = None;
        let mut samples = Vec::new();
        let mut sequence = 0u64;
        group.throughput(Throughput::Elements(rows as u64 * 17408));
        group.bench_function(BenchmarkId::new("pair", format!("{rows}x17408")), |b| {
            let graph = graph.get_or_insert_with(|| SwiGluBenchmark::new(rows, 17408).unwrap());
            b.iter_custom(|iterations| {
                let mut times = [Duration::ZERO; 2];
                for _ in 0..iterations {
                    for candidate in [sequence.is_multiple_of(2), !sequence.is_multiple_of(2)] {
                        let started = Instant::now();
                        graph.run(candidate).unwrap();
                        times[usize::from(candidate)] += started.elapsed();
                    }
                    sequence += 1;
                }
                samples.push(serde_json::json!({
                    "iterations": iterations,
                    "reference_ns": times[0].as_nanos() as u64,
                    "optimized_ns": times[1].as_nanos() as u64,
                }));
                times[1]
            });
        });
        if let Some(mut graph) = graph {
            graph.check().unwrap();
            let root = std::path::PathBuf::from("target/criterion/paired_swiglu/pair")
                .join(format!("{rows}x17408"));
            std::fs::create_dir_all(&root).unwrap();
            let measured = if samples.len() >= 20 {
                &samples[samples.len() - 20..]
            } else {
                &[]
            };
            std::fs::write(
                root.join("paired.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "rows": rows, "width": 17408, "bitwise_identical": true,
                    "sample_size": 20, "samples": samples, "measurement_samples": measured,
                }))
                .unwrap(),
            )
            .unwrap();
        }
    }
    group.finish();
}
fn packing(c: &mut Criterion) {
    let mut group = c.benchmark_group("paired_packing");
    configure(&mut group);
    let widths = if std::env::var_os("CLEF_BENCH_PACKING_SOURCE").is_some() {
        vec![5120, 6144, 8192, 17408]
    } else {
        vec![5120, 6144, 8192]
    };
    for width in widths {
        for rows in lengths() {
            let mut graph = None;
            let mut samples = Vec::new();
            let mut sequence = 0u64;
            group.throughput(Throughput::Elements(rows as u64 * width as u64));
            group.bench_function(BenchmarkId::new("pair", format!("{rows}x{width}")), |b| {
                let graph =
                    graph.get_or_insert_with(|| PackingBenchmark::new(rows, width, 4).unwrap());
                b.iter_custom(|iterations| {
                    let mut times = [Duration::ZERO; 2];
                    for _ in 0..iterations {
                        for candidate in [sequence.is_multiple_of(2), !sequence.is_multiple_of(2)] {
                            let started = Instant::now();
                            graph.run(candidate).unwrap();
                            times[usize::from(candidate)] += started.elapsed();
                        }
                        sequence += 1;
                    }
                    samples.push(serde_json::json!({
                        "iterations": iterations,
                        "reference_ns": times[0].as_nanos() as u64,
                        "optimized_ns": times[1].as_nanos() as u64,
                    }));
                    times[1]
                });
            });
            if let Some(mut graph) = graph {
                graph.check().unwrap();
                let root = std::path::PathBuf::from("target/criterion/paired_packing/pair")
                    .join(format!("{rows}x{width}"));
                std::fs::create_dir_all(&root).unwrap();
                let measured = if samples.len() >= 20 {
                    &samples[samples.len() - 20..]
                } else {
                    &[]
                };
                std::fs::write(
                    root.join("paired.json"),
                    serde_json::to_vec_pretty(&serde_json::json!({
                        "rows": rows, "width": width, "bits": 4, "bitwise_identical": true,
                        "sample_size": 20, "samples": samples, "measurement_samples": measured,
                    }))
                    .unwrap(),
                )
                .unwrap();
            }
        }
    }
    group.finish();
}
/// Both packed-GEMM variants stay resident; timing order alternates each pair.
fn paired_integer_gemm(c: &mut Criterion) {
    let Some(path) = std::env::var_os("CLEF_BENCH_INTEGER_SOURCE") else {
        return;
    };
    let source = std::fs::read_to_string(path).unwrap();
    let bits = std::env::var("CLEF_BENCH_INTEGER_BITS")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(8);
    assert!([4, 8].contains(&bits));
    let padded = std::env::var("CLEF_BENCH_INTEGER_PADDED").as_deref() == Ok("1");
    let mut group = c.benchmark_group("paired_integer_gemm");
    configure(&mut group);
    for m in lengths() {
        for (role, k, n) in [
            ("mlp_up_gate", 5120, 17408),
            ("mlp_down", 17408, 5120),
            ("attention_output", 6144, 5120),
            ("delta_qkv", 5120, 10240),
            ("delta_z", 5120, 6144),
            ("attention_q_gate", 5120, 12288),
        ] {
            for (cache, minimum_bytes) in [("replay", 0), ("rotating", 64 << 20)] {
                let label = format!("{role}_{cache}");
                let shape = format!("{m}x{k}x{n}");
                let mut graphs = None;
                let mut expected = Vec::new();
                let mut samples = Vec::new();
                let mut sequence = 0u64;
                group.throughput(Throughput::Elements(2 * m as u64 * k as u64 * n as u64));
                group.bench_function(BenchmarkId::new(&label, &shape), |b| {
                    let graphs = graphs.get_or_insert_with(|| {
                        assert!(std::env::var("CLEF_BENCH_INTEGER_PREPARE").as_deref() != Ok("1"), "paired integer GEMM uses prepacked operands; measure preparation in workload_gemm");
                        let mut reference = GemmBenchmark::packed_integer(
                            m,
                            k,
                            n,
                            include_str!("../kernels/gemm_integer.loom"),
                            bits,
                            padded,
                            minimum_bytes,
                        )
                        .unwrap();
                        let mut candidate = GemmBenchmark::packed_integer(
                            m,
                            k,
                            n,
                            &source,
                            bits,
                            padded,
                            minimum_bytes,
                        )
                        .unwrap();
                        assert!(
                            reference.allocated_bytes() + candidate.allocated_bytes() <= 1 << 30,
                            "paired GEMM exceeds 1 GiB allocation budget"
                        );
                        let output = reference.output().unwrap();
                        assert!(output.iter().all(|x| x.is_finite()));
                        expected = output.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                        assert_eq!(
                            candidate
                                .output()
                                .unwrap()
                                .iter()
                                .map(|x| x.to_bits())
                                .collect::<Vec<_>>(),
                            expected,
                            "packed GEMM changed output"
                        );
                        [reference, candidate]
                    });
                    b.iter_custom(|iterations| {
                        let mut times = [Duration::ZERO; 2];
                        for _ in 0..iterations {
                            for candidate in
                                [sequence.is_multiple_of(2), !sequence.is_multiple_of(2)]
                            {
                                let started = Instant::now();
                                graphs[usize::from(candidate)].run().unwrap();
                                times[usize::from(candidate)] += started.elapsed();
                            }
                            sequence += 1;
                        }
                        samples.push(serde_json::json!({
                            "iterations": iterations,
                            "reference_ns": times[0].as_nanos() as u64,
                            "optimized_ns": times[1].as_nanos() as u64,
                        }));
                        times[1]
                    });
                });
                if let Some(mut graphs) = graphs {
                    for graph in &mut graphs {
                        assert_eq!(
                            graph
                                .output()
                                .unwrap()
                                .iter()
                                .map(|x| x.to_bits())
                                .collect::<Vec<_>>(),
                            expected,
                            "GEMM replay changed output"
                        );
                    }
                    let root = std::path::PathBuf::from("target/criterion/paired_integer_gemm")
                        .join(&label)
                        .join(&shape);
                    std::fs::create_dir_all(&root).unwrap();
                    let measured = if samples.len() >= 20 {
                        &samples[samples.len() - 20..]
                    } else {
                        &[]
                    };
                    std::fs::write(root.join("paired.json"), serde_json::to_vec_pretty(&serde_json::json!({
                        "rows": m, "k": k, "columns": n, "bits": bits,
                        "padded": padded, "minimum_weight_bytes_per_variant": minimum_bytes,
                        "varying_row_scales": std::env::var("CLEF_BENCH_INTEGER_ROW_SCALES").as_deref() == Ok("1"),
                        "bitwise_identical": true, "sample_size": 20,
                        "samples": samples, "measurement_samples": measured,
                        "note": "Prepacked operands; two resident variants with separate identical weight allocations; preparation excluded.",
                    })).unwrap()).unwrap();
                }
            }
        }
    }
    group.finish();
}
criterion_group!(
    workload,
    gemm,
    delta,
    attention,
    swiglu,
    packing,
    paired_integer_gemm
);
criterion_main!(workload);
