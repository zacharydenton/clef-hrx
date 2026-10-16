use clap::Parser;
use clef_hrx::{
    Result,
    benchmarking::{AttentionBenchmark, AttentionFixture, AttentionProbeOptions},
};
use sha2::{Digest, Sha256};
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
    #[arg(long, requires = "compile_only", conflicts_with = "compile_key_mean")]
    compile_packing: bool,
    /// Compile one pass of the key mean reduction without opening the GPU.
    #[arg(long, requires = "compile_only", value_parser = ["partial", "finish"])]
    compile_key_mean: Option<String>,
    /// Include separate Q/K packing before every candidate graph replay.
    #[arg(long)]
    prepacked_qk: bool,
    /// Replace only candidate Q/K packing, including its cost on every replay.
    #[arg(long, requires = "prepacked_qk", conflicts_with = "compile_only")]
    packing_source: Option<PathBuf>,
    /// Center keys with two GPU mean passes and this four-binding packing source.
    /// Query and key packing must use the same rotation.
    #[arg(long, requires = "prepacked_qk", conflicts_with = "compile_only")]
    centered_key_source: Option<PathBuf>,
    /// Read exact-shape q.bf16/k.bf16/v.bf16 fixtures for both variants.
    #[arg(long, requires = "reference_source", conflicts_with_all = ["compile_only", "integrated_reference", "fixture"])]
    input_directory: Option<PathBuf>,
    /// Save HRX 0.11's target assembly alongside the compile-only report.
    #[arg(long, requires = "compile_only")]
    assembly: Option<PathBuf>,
    /// Save check outcomes and every alternating timing pair for noise analysis.
    #[arg(long, conflicts_with = "compile_only")]
    report_json: Option<PathBuf>,
    /// Time finite mismatching outputs for diagnostics, then still exit with failure.
    #[arg(long, requires = "report_json", conflicts_with = "compile_only")]
    time_on_mismatch: bool,
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
        if let Some(pass) = &a.compile_key_mean {
            spec = hrx::loom::Specialization::new("clef_key_mean");
            spec.set_config("clef.key_mean.width", &kw);
            spec.set_config(
                "clef.key_mean.finish",
                if pass == "finish" { "1" } else { "0" },
            );
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
    if let Some(directory) = &a.input_directory {
        for name in ["q", "k", "v"] {
            let path = directory.join(format!("{name}.bf16"));
            eprintln!(
                "fixture {} sha256={:x}",
                path.display(),
                Sha256::digest(std::fs::read(&path)?)
            );
        }
    }
    let mut baseline = if a.input_directory.is_some() {
        AttentionBenchmark::probe_case(
            a.tokens,
            reference_source.as_deref(),
            a.model,
            a.fixture,
            a.reference_prepacked_qk,
            AttentionProbeOptions {
                input_directory: a.input_directory.as_deref(),
                ..Default::default()
            },
        )?
    } else if a.reference_prepacked_qk {
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
    let packing_source = a
        .packing_source
        .as_ref()
        .map(std::fs::read_to_string)
        .transpose()?;
    let centered_key_source = a
        .centered_key_source
        .as_ref()
        .map(std::fs::read_to_string)
        .transpose()?;
    let mut candidate = AttentionBenchmark::probe_case(
        a.tokens,
        Some(&source),
        a.model,
        a.fixture,
        a.prepacked_qk,
        AttentionProbeOptions {
            packing_source: packing_source.as_deref(),
            centered_key_source: centered_key_source.as_deref(),
            input_directory: a.input_directory.as_deref(),
        },
    )?;
    let actual = candidate.output()?;
    anyhow::ensure!(actual.len() == expected.len(), "output shape mismatch");
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    anyhow::ensure!(actual.iter().all(|v| v.is_finite()), "nonfinite output");
    let numerical_check_passed = max_error <= a.max_error;
    anyhow::ensure!(
        numerical_check_passed || a.time_on_mismatch,
        "output error {max_error} exceeds {}",
        a.max_error
    );
    let mut samples = [Vec::new(), Vec::new()];
    let mut pairs = Vec::with_capacity(a.rounds);
    for round in 0..a.rounds {
        let mut seconds = [0.; 2];
        for i in [round % 2, 1 - round % 2] {
            let start = Instant::now();
            if i == 0 {
                baseline.run()?;
            } else {
                candidate.run()?;
            }
            seconds[i] = start.elapsed().as_secs_f64();
            samples[i].push(seconds[i]);
        }
        pairs.push(serde_json::json!({
            "round": round,
            "first": if round % 2 == 0 { "reference" } else { "candidate" },
            "reference_seconds": seconds[0],
            "candidate_seconds": seconds[1],
            "speedup": seconds[0] / seconds[1],
        }));
    }
    anyhow::ensure!(
        candidate.output()? == actual,
        "persistent state was not reset"
    );
    anyhow::ensure!(
        baseline.output()? == expected,
        "reference changed on replay"
    );
    for s in &mut samples {
        s.sort_by(f64::total_cmp);
    }
    let old = samples[0][a.rounds / 2];
    let new = samples[1][a.rounds / 2];
    let fixture_label = a.input_directory.as_ref().map_or_else(
        || format!("{:?}", a.fixture),
        |path| format!("files:{}", path.display()),
    );
    if let Some(path) = &a.report_json {
        let mut inputs = serde_json::Map::new();
        for input in std::iter::once(&a.source)
            .chain(a.reference_source.iter())
            .chain(a.packing_source.iter())
            .chain(a.centered_key_source.iter())
            .cloned()
            .chain(
                a.input_directory
                    .iter()
                    .flat_map(|dir| ["q", "k", "v"].map(|name| dir.join(format!("{name}.bf16")))),
            )
        {
            inputs.insert(
                input.display().to_string(),
                format!("{:x}", Sha256::digest(std::fs::read(&input)?)).into(),
            );
        }
        let rms_error = (actual
            .iter()
            .zip(&expected)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        let report = serde_json::json!({
            "schema_version": 1,
            "binary_sha256": format!("{:x}", Sha256::digest(std::fs::read(std::env::current_exe()?)?)),
            "model": format!("{:?}", a.model),
            "tokens": a.tokens,
            "fixture": fixture_label,
            "command": std::env::args().collect::<Vec<_>>(),
            "source_sha256": inputs,
            "reference_source": a.reference_source,
            "integrated_reference": a.integrated_reference,
            "reference_prepacked_qk": a.reference_prepacked_qk,
            "candidate_prepacked_qk": a.prepacked_qk,
            "max_error": max_error,
            "rms_error": rms_error,
            "max_error_limit": a.max_error,
            "numerical_check_passed": numerical_check_passed,
            "timed_after_numerical_failure": !numerical_check_passed,
            "reference_replay_exact": true,
            "candidate_replay_exact": true,
            "timing_scope": "resident operator graph replay, including configured Q/K preparation and V transpose; excludes compilation, upload and readback",
            "gpu_exclusivity": "not established by this probe",
            "reference_median_seconds": old,
            "candidate_median_seconds": new,
            "ratio_of_medians": old / new,
            "pairs": pairs,
        });
        std::fs::write(path, serde_json::to_string_pretty(&report)? + "\n")?;
    }
    println!(
        "{} tokens={} max_error={} baseline {:.3} ms candidate {:.3} ms speedup {:.3} fixture={}",
        a.source.display(),
        a.tokens,
        max_error,
        old * 1000.,
        new * 1000.,
        old / new,
        fixture_label
    );
    anyhow::ensure!(
        numerical_check_passed,
        "output error {max_error} exceeds {} (diagnostic timings recorded; numerical check failed)",
        a.max_error
    );
    Ok(())
}
