use clef_hrx::{
    encoding::canonical_json,
    media::{MediaOptions, prepare},
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn benches(c: &mut Criterion) {
    let request: serde_json::Value =
        serde_json::from_str(include_str!("../examples/invoice.json")).unwrap();
    c.bench_function("canonical_json/invoice", |b| {
        b.iter(|| canonical_json(black_box(&request)))
    });
    let mut group = c.benchmark_group("image_preprocessing");
    for (width, height) in [(32, 32), (117, 73), (777, 513)] {
        let images = [image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 127])
        })];
        let options = MediaOptions {
            min_pixels: Some(1024),
            max_pixels: Some(262144),
            ..Default::default()
        };
        group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
        group.bench_with_input(
            BenchmarkId::new("resize_patch", format!("{width}x{height}")),
            &images,
            |b, images| {
                b.iter(|| prepare(black_box(images), &[], black_box(&options), 16384).unwrap());
            },
        );
    }
    group.finish();
}
criterion_group!(preprocessing, benches);
criterion_main!(preprocessing);
