use clef_hrx::{
    encoding::canonical_json,
    media::{MediaOptions, prepare},
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn benches(c: &mut Criterion) {
    #[cfg(feature = "bench-internals")]
    {
        use clef_hrx::benchmarking::KernelCacheKeyBenchmark;
        let mut group = c.benchmark_group("kernel_cache_key");
        group
            .sample_size(50)
            .warm_up_time(std::time::Duration::from_millis(250))
            .measurement_time(std::time::Duration::from_secs(1))
            .confidence_level(0.99);
        for (name, source) in [
            (
                "pointwise",
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/addf.loom")
                ),
            ),
            ("integer", include_str!("../kernels/gemm_integer.loom")),
            (
                "delta",
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/delta_persistent_vector.loom")
                ),
            ),
        ] {
            let mut cache = KernelCacheKeyBenchmark::new(source);
            assert_eq!(cache.legacy(), (true, Some(7)));
            assert_eq!(cache.fingerprinted(), (true, Some(7)));
            group.bench_function(BenchmarkId::new("legacy", name), |b| {
                b.iter(|| black_box(cache.legacy()))
            });
            group.bench_function(BenchmarkId::new("fingerprint", name), |b| {
                b.iter(|| black_box(cache.fingerprinted()))
            });
        }
        group.finish();
    }
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
