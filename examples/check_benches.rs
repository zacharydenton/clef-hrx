//! Compare named Criterion baselines. Missing or invalid evidence fails closed.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::path::Path;
#[path = "../benches/support/mod.rs"]
mod support;

#[derive(Deserialize)]
struct Criteria {
    target: String,
    allocation_limit_bytes: usize,
    gates: Vec<Gate>,
}
#[derive(Deserialize)]
struct Gate {
    benchmark: String,
    max_ratio: f64,
}
#[derive(Deserialize)]
struct Estimates {
    mean: Estimate,
}
#[derive(Deserialize)]
struct Estimate {
    point_estimate: f64,
    confidence_interval: Interval,
}
#[derive(Deserialize)]
struct Interval {
    lower_bound: f64,
    upper_bound: f64,
}
#[derive(Deserialize)]
struct Paired {
    run_id: String,
    fingerprint: String,
    allocated_bytes: usize,
    sample_size: usize,
    samples: Vec<Sample>,
}
#[derive(Deserialize)]
struct Sample {
    iterations: u64,
    reference_ns: u64,
    optimized_ns: u64,
}

fn paired_ratio(data: &Paired) -> Result<(f64, f64)> {
    ensure!(
        data.sample_size >= 20 && data.samples.len() >= data.sample_size,
        "at least 20 paired measurement batches are required"
    );
    // Criterion's calibration/warmup calls precede its measurement batches.
    let samples = &data.samples[data.samples.len() - data.sample_size..];
    ensure!(
        samples
            .iter()
            .all(|s| s.iterations > 0 && s.reference_ns > 0 && s.optimized_ns > 0),
        "empty paired sample"
    );
    let sums = samples.iter().fold((0f64, 0f64), |(a, b), s| {
        (a + s.optimized_ns as f64, b + s.reference_ns as f64)
    });
    let mut rng = 0x6372_6974_6572_6961u64;
    let mut bootstrap = Vec::with_capacity(10000);
    for _ in 0..10000 {
        let mut old = 0.;
        let mut new = 0.;
        for _ in samples {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let sample = &samples[rng as usize % samples.len()];
            old += sample.reference_ns as f64;
            new += sample.optimized_ns as f64;
        }
        bootstrap.push(new / old);
    }
    bootstrap.sort_by(f64::total_cmp);
    // Upper endpoint of a deterministic 95% percentile bootstrap interval.
    Ok((sums.0 / sums.1, bootstrap[9749]))
}
fn read(root: &Path, id: &str, baseline: &str) -> Result<Estimate> {
    let path = root.join(id).join(baseline).join("estimates.json");
    let estimates: Estimates = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("missing {}", path.display()))?,
    )?;
    let mean = estimates.mean;
    let ci = &mean.confidence_interval;
    ensure!(
        [mean.point_estimate, ci.lower_bound, ci.upper_bound]
            .iter()
            .all(|v| v.is_finite() && *v > 0.)
            && ci.lower_bound <= mean.point_estimate
            && mean.point_estimate <= ci.upper_bound,
        "invalid timing estimate for {id}/{baseline}"
    );
    Ok(mean)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let paired = args.len() == 2 && args[1] == "paired";
    ensure!(
        paired || args.len() == 3,
        "usage: check_benches <criterion-directory> paired | <criterion-directory> <before-baseline> <after-baseline>"
    );
    if !paired {
        ensure!(args[1] != args[2], "use distinct baselines");
    }
    let criteria: Criteria = serde_json::from_str(include_str!("../benches/criteria.json"))?;
    println!(
        "Target {}; allocation cap {} MiB (enforced by benchmark harness)",
        criteria.target,
        criteria.allocation_limit_bytes / (1 << 20)
    );
    let mut passed = true;
    let mut paired_run = None;
    for gate in criteria.gates {
        if paired {
            let path = Path::new(&args[0])
                .join(format!("paired_{}", gate.benchmark))
                .join("paired.json");
            let data: Paired = serde_json::from_slice(
                &std::fs::read(&path).with_context(|| format!("missing {}", path.display()))?,
            )?;
            ensure!(
                data.fingerprint == support::fingerprint(),
                "stale benchmark evidence for {}: kernel sources changed",
                gate.benchmark
            );
            ensure!(!data.run_id.is_empty(), "missing paired run identity");
            if let Some(run) = &paired_run {
                ensure!(
                    run == &data.run_id,
                    "paired results must come from the same complete run"
                );
            } else {
                paired_run = Some(data.run_id.clone());
            }
            let (ratio, upper) = paired_ratio(&data)?;
            let memory_ok = data.allocated_bytes <= criteria.allocation_limit_bytes;
            let ok = upper <= gate.max_ratio && memory_ok;
            println!(
                "{}: paired ratio {ratio:.3}, upper 95% {upper:.3}, limit {:.3}, allocation {:.1} MiB: {}",
                gate.benchmark,
                gate.max_ratio,
                data.allocated_bytes as f64 / (1 << 20) as f64,
                if ok { "PASS" } else { "FAIL" }
            );
            passed &= ok;
            continue;
        }
        let before = read(Path::new(&args[0]), &gate.benchmark, &args[1])?;
        let after = read(Path::new(&args[0]), &gate.benchmark, &args[2])?;
        // Compare worst-case confidence bounds, not only a favorable mean.
        let ratio = after.confidence_interval.upper_bound / before.confidence_interval.lower_bound;
        let ok = ratio <= gate.max_ratio;
        println!(
            "{}: {:.3} -> {:.3} ms; conservative ratio {:.3}, limit {:.3}: {}",
            gate.benchmark,
            before.point_estimate / 1e6,
            after.point_estimate / 1e6,
            ratio,
            gate.max_ratio,
            if ok { "PASS" } else { "FAIL" }
        );
        passed &= ok;
    }
    ensure!(
        passed,
        "benchmark criteria failed; check correctness, machine load, and repeatability before accepting the change"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_estimate_preserves_speedup_under_changing_load() {
        let data = Paired {
            run_id: "test".into(),
            fingerprint: "test".into(),
            allocated_bytes: 100,
            sample_size: 20,
            samples: (1..=20)
                .map(|i| Sample {
                    iterations: i,
                    reference_ns: i * i * 1000,
                    optimized_ns: i * i * 600,
                })
                .collect(),
        };
        let (mean, upper) = paired_ratio(&data).unwrap();
        assert!((mean - 0.6).abs() < 1e-12 && (upper - 0.6).abs() < 1e-12);
    }
    #[test]
    fn paired_evidence_rejects_missing_and_zero_samples() {
        let mut data = Paired {
            run_id: "test".into(),
            fingerprint: "test".into(),
            allocated_bytes: 0,
            sample_size: 20,
            samples: vec![],
        };
        assert!(paired_ratio(&data).is_err());
        data.samples = (0..20)
            .map(|_| Sample {
                iterations: 1,
                reference_ns: 0,
                optimized_ns: 100,
            })
            .collect();
        assert!(paired_ratio(&data).is_err());
    }
}
