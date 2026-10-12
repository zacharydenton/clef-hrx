use clap::Parser;
use clef_hrx::{Result, benchmarking::DeltaBenchmark};
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 620)]
    tokens: usize,
    #[arg(long, default_value_t = 8)]
    values: usize,
    #[arg(long, default_value_t = 10)]
    rounds: usize,
    /// Compare against production selection, including the long-context scalar kernel.
    #[arg(long)]
    integrated_reference: bool,
    #[arg(long, default_value_t = 0.002)]
    max_error: f32,
    source: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    anyhow::ensure!(a.rounds > 0, "rounds must be positive");
    anyhow::ensure!(a.max_error.is_finite() && a.max_error >= 0.);
    let mut baseline = if a.integrated_reference {
        DeltaBenchmark::for_model(a.tokens, clef_hrx::ModelVariant::Clef)?
    } else {
        DeltaBenchmark::new(a.tokens, true)?
    };
    let expected = baseline.output()?;
    anyhow::ensure!(
        expected.iter().all(|v| v.is_finite()),
        "nonfinite reference"
    );
    let source = std::fs::read_to_string(&a.source)?;
    let mut candidate = if a.integrated_reference {
        DeltaBenchmark::for_model_source(a.tokens, clef_hrx::ModelVariant::Clef, &source, a.values)?
    } else {
        DeltaBenchmark::persistent(a.tokens, &source, a.values)?
    };
    let actual = candidate.output()?;
    anyhow::ensure!(actual.len() == expected.len(), "output shape mismatch");
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    anyhow::ensure!(actual.iter().all(|v| v.is_finite()), "nonfinite output");
    anyhow::ensure!(
        max_error <= a.max_error,
        "output error {max_error} exceeds {}",
        a.max_error
    );
    let mut samples = [Vec::new(), Vec::new()];
    for round in 0..a.rounds {
        for i in [round % 2, 1 - round % 2] {
            let start = Instant::now();
            if i == 0 {
                baseline.run()?;
            } else {
                candidate.run()?;
            }
            samples[i].push(start.elapsed().as_secs_f64());
        }
    }
    anyhow::ensure!(
        candidate.output()? == actual,
        "persistent state was not reset"
    );
    for s in &mut samples {
        s.sort_by(f64::total_cmp);
    }
    let old = samples[0][a.rounds / 2];
    let new = samples[1][a.rounds / 2];
    println!(
        "{} tokens={} max_error={} baseline {:.3} ms candidate {:.3} ms speedup {:.3}",
        a.source.display(),
        a.tokens,
        max_error,
        old * 1000.,
        new * 1000.,
        old / new
    );
    Ok(())
}
