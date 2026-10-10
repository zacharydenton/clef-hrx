use clap::Parser;
use clef_hrx::{
    Result,
    benchmarking::{GemmBenchmark, GemmKernel},
};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 620)]
    m: usize,
    #[arg(long, default_value_t = 5120)]
    k: usize,
    #[arg(long, default_value_t = 17408)]
    n: usize,
    #[arg(long, default_value_t = 20)]
    rounds: usize,
    sources: Vec<PathBuf>,
}
fn main() -> Result<()> {
    let a = Args::parse();
    anyhow::ensure!(a.rounds > 0, "rounds must be positive");
    let mut baseline = GemmBenchmark::with_kernel(a.m, a.k, a.n, GemmKernel::Pipelined)?;
    let expected = baseline.output()?;
    for source in &a.sources {
        let mut candidate = GemmBenchmark::with_kernel(
            a.m,
            a.k,
            a.n,
            GemmKernel::Source(std::fs::read_to_string(source)?),
        )?;
        let actual = candidate.output()?;
        anyhow::ensure!(actual.iter().all(|v| v.is_finite()), "nonfinite output");
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("{} maximum output change={error}", source.display());
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
        for s in &mut samples {
            s.sort_by(f64::total_cmp);
        }
        let old = samples[0][a.rounds / 2];
        let new = samples[1][a.rounds / 2];
        println!(
            "{} {}x{}x{} baseline {:.3} ms candidate {:.3} ms speedup {:.3} TFLOPS {:.2}",
            source.display(),
            a.m,
            a.k,
            a.n,
            old * 1000.,
            new * 1000.,
            old / new,
            2. * a.m as f64 * a.k as f64 * a.n as f64 / new / 1e12
        );
    }
    Ok(())
}
