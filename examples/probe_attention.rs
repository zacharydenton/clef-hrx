use clap::Parser;
use clef_hrx::{
    Result,
    benchmarking::{AttentionBenchmark, AttentionFixture},
};
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long, value_enum, default_value_t = clef_hrx::ModelVariant::Clef)]
    model: clef_hrx::ModelVariant,
    #[arg(long, default_value_t = 256)]
    tokens: usize,
    #[arg(long, default_value_t = 10)]
    rounds: usize,
    #[arg(long, value_enum, default_value_t = AttentionFixture::Patterned)]
    fixture: AttentionFixture,
    /// Compare with the production attention selector instead of the scalar oracle.
    #[arg(long)]
    integrated_reference: bool,
    /// Compare with an explicit source, for example a former production kernel.
    #[arg(long, conflicts_with = "integrated_reference")]
    reference_source: Option<PathBuf>,
    /// Include Q/K packing in an explicitly selected reference graph.
    #[arg(long, requires = "reference_source")]
    reference_prepacked_qk: bool,
    /// Specialize and report compiler resources without opening the GPU.
    #[arg(long)]
    compile_only: bool,
    /// Compile the separate Q/K packing kernel instead of the attention kernel.
    #[arg(long, requires = "compile_only")]
    compile_packing: bool,
    /// Include separate Q/K packing before every candidate graph replay.
    #[arg(long)]
    prepacked_qk: bool,
    /// Save HRX 0.11's target assembly alongside the compile-only report.
    #[arg(long, requires = "compile_only")]
    assembly: Option<PathBuf>,
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
            clef_hrx::ModelVariant::ClefOmni => 32,
        };
        let heads_text = heads.to_string();
        let ratio = (heads / 4).to_string();
        let dim = if a.model == clef_hrx::ModelVariant::ClefOmni {
            128
        } else {
            256
        };
        let qw = (heads * dim).to_string();
        let kw = (4 * dim).to_string();
        let scale = format!("{:e}", 1.0 / (dim as f32).sqrt());
        for (key, value) in [
            ("heads", heads_text.as_str()),
            ("ratio", ratio.as_str()),
            ("qw", qw.as_str()),
            ("kw", kw.as_str()),
            ("scale", scale.as_str()),
            ("causal", "1"),
        ] {
            spec.set_config(format!("clef.{key}"), value);
        }
        if a.compile_packing {
            spec = hrx::loom::Specialization::new("clef_prepare_qk_int8");
            spec.set_config("clef.qk_i8.width", &qw);
        }
        spec.set_report(hrx::loom::ReportMode::Details);
        spec = spec.with_listing(a.assembly.is_some());
        let source = format!("{}\n{source}", include_str!("../kernels/common.loom"));
        let artifact = compiler.module(&source).compile(&spec)?;
        if let Some(path) = &a.assembly {
            std::fs::write(
                path,
                artifact
                    .listing()
                    .ok_or_else(|| anyhow::anyhow!("compiler returned no target assembly"))?,
            )?;
        }
        println!("{}", serde_json::to_string_pretty(&artifact.report())?);
        return Ok(());
    }
    let reference_source = a
        .reference_source
        .as_ref()
        .map(std::fs::read_to_string)
        .transpose()?;
    let mut baseline = if a.reference_prepacked_qk {
        AttentionBenchmark::prepacked_case(
            a.tokens,
            reference_source
                .as_deref()
                .expect("reference source required"),
            a.model,
            a.fixture,
        )?
    } else {
        AttentionBenchmark::case(
            a.tokens,
            reference_source.as_deref(),
            a.integrated_reference,
            a.model,
            a.fixture,
        )?
    };
    let expected = baseline.output()?;
    anyhow::ensure!(
        expected.iter().all(|v| v.is_finite()),
        "nonfinite reference"
    );
    let mut candidate = if a.prepacked_qk {
        AttentionBenchmark::prepacked_case(a.tokens, &source, a.model, a.fixture)?
    } else {
        AttentionBenchmark::case(a.tokens, Some(&source), false, a.model, a.fixture)?
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
        "{} tokens={} max_error={} baseline {:.3} ms candidate {:.3} ms speedup {:.3} fixture={:?}",
        a.source.display(),
        a.tokens,
        max_error,
        old * 1000.,
        new * 1000.,
        old / new,
        a.fixture
    );
    Ok(())
}
