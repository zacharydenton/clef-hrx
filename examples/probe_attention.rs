use clap::Parser;
use clef_hrx::{Result, benchmarking::AttentionBenchmark};
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long, value_enum, default_value_t = clef_hrx::ModelVariant::Clef)]
    model: clef_hrx::ModelVariant,
    #[arg(long, default_value_t = 256)]
    tokens: usize,
    #[arg(long, default_value_t = 10)]
    rounds: usize,
    /// Compare with the production attention selector instead of the scalar oracle.
    #[arg(long)]
    integrated_reference: bool,
    /// Compare with an explicit source, for example a former production kernel.
    #[arg(long, conflicts_with = "integrated_reference")]
    reference_source: Option<PathBuf>,
    /// Specialize and report compiler resources without opening the GPU.
    #[arg(long)]
    compile_only: bool,
    #[arg(long, default_value_t = 0.008)]
    max_error: f32,
    source: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    anyhow::ensure!(a.rounds > 0, "rounds must be positive");
    anyhow::ensure!(a.max_error.is_finite() && a.max_error >= 0.);
    let source = std::fs::read_to_string(&a.source)?;
    if a.compile_only {
        let compiler = hrx::loom::Compiler::resolve(None)?;
        let mut spec = hrx::loom::Specialization::new("clef_op");
        let heads = match a.model {
            clef_hrx::ModelVariant::Clef => 24,
            clef_hrx::ModelVariant::ClefFlash => 16,
        };
        let heads_text = heads.to_string();
        let ratio = (heads / 4).to_string();
        let qw = (heads * 256).to_string();
        for (key, value) in [
            ("heads", heads_text.as_str()),
            ("ratio", ratio.as_str()),
            ("qw", qw.as_str()),
            ("kw", "1024"),
            ("scale", "6.25e-2"),
            ("causal", "1"),
        ] {
            spec.set_config(format!("clef.{key}"), value);
        }
        spec.set_report(hrx::loom::ReportMode::Details);
        let source = format!("{}\n{source}", include_str!("../kernels/common.loom"));
        let artifact = compiler.module(&source).compile(&spec)?;
        println!("{}", serde_json::to_string_pretty(&artifact.report())?);
        return Ok(());
    }
    let mut baseline = if let Some(path) = &a.reference_source {
        AttentionBenchmark::source_for_model(a.tokens, a.model, &std::fs::read_to_string(path)?)?
    } else {
        AttentionBenchmark::for_model(a.tokens, a.model, a.integrated_reference)?
    };
    let expected = baseline.output()?;
    anyhow::ensure!(
        expected.iter().all(|v| v.is_finite()),
        "nonfinite reference"
    );
    let mut candidate = AttentionBenchmark::source_for_model(a.tokens, a.model, &source)?;
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
