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
    #[arg(long, conflicts_with = "sources")]
    tiled_group: Option<usize>,
    sources: Vec<PathBuf>,
}
fn main() -> Result<()> {
    let a = Args::parse();
    anyhow::ensure!(a.rounds > 0, "rounds must be positive");
    let mut baseline = GemmBenchmark::with_kernel(
        a.m,
        a.k,
        a.n,
        if a.tiled_group.is_some() {
            GemmKernel::TiledGroup(8)
        } else {
            GemmKernel::Pipelined
        },
    )?;
    let expected = baseline.output()?;
    let candidates = if let Some(group) = a.tiled_group {
        vec![(
            format!("tiled-group-{group}"),
            GemmKernel::TiledGroup(group),
        )]
    } else {
        a.sources
            .iter()
            .map(|source| {
                Ok((
                    source.display().to_string(),
                    GemmKernel::Source(std::fs::read_to_string(source)?),
                ))
            })
            .collect::<Result<Vec<_>>>()?
    };
    for (name, kernel) in candidates {
        let mut candidate = GemmBenchmark::with_kernel(a.m, a.k, a.n, kernel)?;
        let actual = candidate.output()?;
        anyhow::ensure!(actual.iter().all(|v| v.is_finite()), "nonfinite output");
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if a.tiled_group.is_some() {
            anyhow::ensure!(actual == expected, "row grouping changed arithmetic");
        }
        eprintln!("{name} maximum output change={error}");
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
            name,
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
