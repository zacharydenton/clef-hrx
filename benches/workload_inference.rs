use clef_hrx::{ClefModel, EncodeOptions, LoadOptions, Request, checkpoint::Source};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::{
    collections::BTreeSet,
    time::{Duration, Instant},
};

fn benches(c: &mut Criterion) {
    if std::env::var("CLEF_BENCH_FULL_MODEL").as_deref() != Ok("1") {
        eprintln!(
            "Set CLEF_BENCH_FULL_MODEL=1 and CLEF_BENCH_REQUESTS to a JevBench requests.jsonl."
        );
        return;
    }
    let requests: Vec<Request> = std::fs::read_to_string(
        std::env::var_os("CLEF_BENCH_REQUESTS").expect("CLEF_BENCH_REQUESTS is required"),
    )
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str(line).unwrap())
    .collect();
    assert!(!requests.is_empty());
    let start = Instant::now();
    let precision = <clef_hrx::Precision as clap::ValueEnum>::from_str(
        &std::env::var("CLEF_BENCH_PRECISION").unwrap_or_else(|_| "bf16".into()),
        false,
    )
    .unwrap();
    let mut model = ClefModel::load(LoadOptions {
        precision,
        source: Source {
            model: match std::env::var("CLEF_BENCH_MODEL")
                .as_deref()
                .unwrap_or("clef")
            {
                "clef" => clef_hrx::ModelVariant::Clef,
                "clef-flash" => clef_hrx::ModelVariant::ClefFlash,
                other => panic!("unknown benchmark model {other}"),
            },
            directory: std::env::var_os("CLEF_MODEL_DIR").map(Into::into),
            offline: true,
        },
        encoding: EncodeOptions {
            strict_context: true,
            max_length: 65536,
            ..Default::default()
        },
        memory_budget_bytes: 80usize << 30,
        ..Default::default()
    })
    .unwrap();
    eprintln!(
        "Checkpoint load: {:.3}s (excluded from resident samples)",
        start.elapsed().as_secs_f64()
    );
    let lengths: Vec<_> = requests
        .iter()
        .map(|r| model.encode_record(r).unwrap().input_ids.len())
        .collect();
    assert!(
        lengths.iter().all(|&n| n <= 65536),
        "check for input truncation"
    );
    let mut targets = vec![158usize, 235, 260, 620, 1024, 2048, 4032];
    if lengths.iter().any(|&n| n > 4032) {
        targets.extend([8192, 16384, 32768, 49152, 65536]);
    }
    let selected: BTreeSet<_> = targets
        .into_iter()
        .map(|target| {
            lengths
                .iter()
                .enumerate()
                .min_by_key(|(_, n)| n.abs_diff(target))
                .unwrap()
                .0
        })
        .collect();
    let mut group = c.benchmark_group("workload_inference");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(30))
        .noise_threshold(0.03)
        .significance_level(0.01)
        .confidence_level(0.99);
    for i in selected {
        let request = &requests[i];
        let tokens = lengths[i];
        group.throughput(Throughput::Elements(tokens as u64));
        for mode in ["replay_wall", "rebuild_wall"] {
            let mut warmed = false;
            let mut phases = Vec::new();
            group.bench_function(BenchmarkId::new(mode, format!("{tokens}_case{i}")), |b| {
                if !warmed {
                    let first = model.infer(request).unwrap();
                    eprintln!("case={i}, tokens={tokens}, first={:?}", first.timings);
                    warmed = true;
                }
                b.iter(|| {
                    if mode == "rebuild_wall" {
                        model.invalidate_benchmark_graph().unwrap();
                    }
                    let p = model.infer(request).unwrap();
                    assert_eq!(p.input_tokens, tokens);
                    phases.push(serde_json::json!({
                        "encode_ms": p.timings.encode_ms,
                        "prepare_ms": p.timings.prepare_ms,
                        "inference_ms": p.timings.inference_ms,
                        "readback_ms": p.timings.readback_ms,
                    }));
                });
            });
            if warmed {
                let dir = std::path::PathBuf::from("target/criterion/workload_phases");
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(dir.join(format!("{mode}_{tokens}_case{i}.json")),
                    serde_json::to_vec_pretty(&serde_json::json!({
                        "note": "Diagnostic phase observations include Criterion warmup; wall-time confidence intervals are in Criterion estimates.json.",
                        "tokens": tokens, "case": i, "mode": mode, "observations": phases,
                    })).unwrap()).unwrap();
            }
        }
    }
    group.finish();
}
criterion_group!(inference, benches);
criterion_main!(inference);
