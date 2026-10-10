use clap::Parser;
use clef_hrx::{Result, benchmarking::AttentionBenchmark};
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 256)]
    tokens: usize,
    #[arg(long, default_value_t = 10)]
    rounds: usize,
    source: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    anyhow::ensure!(a.rounds > 0, "rounds must be positive");
    let mut baseline = AttentionBenchmark::new(a.tokens, None)?;
    let expected = baseline.output()?;
    let mut candidate =
        AttentionBenchmark::new(a.tokens, Some(&std::fs::read_to_string(&a.source)?))?;
    let actual = candidate.output()?;
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    anyhow::ensure!(actual.iter().all(|v| v.is_finite()), "nonfinite output");
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
