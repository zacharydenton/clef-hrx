use clef_hrx::{ClefModel, EncodeOptions, LoadOptions, Request, checkpoint::Source};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::{hint::black_box, time::Duration};

fn benches(c: &mut Criterion) {
    if std::env::var("CLEF_BENCH_FULL_MODEL").as_deref() != Ok("1") {
        eprintln!(
            "Full-model benchmark disabled. Set CLEF_BENCH_FULL_MODEL=1 with at least 50 GiB available RAM and a cached checkpoint."
        );
        return;
    }
    let request: Request = serde_json::from_str(include_str!("../examples/invoice.json")).unwrap();
    let mut model = ClefModel::load(LoadOptions {
        source: Source {
            directory: std::env::var_os("CLEF_MODEL_DIR").map(Into::into),
            offline: true,
        },
        encoding: EncodeOptions {
            max_length: 512,
            max_state_tokens: None,
        },
        memory_budget_bytes: 60usize << 30,
        ..Default::default()
    })
    .expect("full-model memory preflight and checkpoint loading");
    let warmup = model.infer(&request).unwrap();
    let mut group = c.benchmark_group("full_model");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(30));
    group.throughput(Throughput::Elements(warmup.input_tokens as u64));
    group.bench_function("invoice_cached_graph", |b| {
        b.iter(|| model.infer(black_box(&request)).unwrap())
    });
    group.finish();
}
criterion_group!(inference, benches);
criterion_main!(inference);
