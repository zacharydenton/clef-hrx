//! Compare attention sources using one weight allocation and rebuilt request graphs.
use clap::Parser;
use clef_hrx::{
    ClefModel, EncodeOptions, LoadOptions, Precision, Request, Result, checkpoint::Source,
};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum, default_value_t = clef_hrx::ModelVariant::Clef)]
    model: clef_hrx::ModelVariant,
    #[arg(long, value_enum, default_value_t = Precision::Int4)]
    precision: Precision,
    #[arg(long, default_value_t = 28)]
    memory_gib: usize,
    #[arg(long, default_value_t = 65536)]
    max_length: usize,
    /// Alternating-order pairs after two untimed correctness/warmup requests.
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    /// Fresh result path, updated after every request.
    #[arg(long)]
    output: PathBuf,
    input: PathBuf,
}

fn save(path: &std::path::Path, report: &serde_json::Value) -> Result<()> {
    let temp = path.with_extension("tmp");
    std::fs::write(&temp, serde_json::to_vec_pretty(report)?)?;
    std::fs::rename(temp, path)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(!args.output.exists(), "use a fresh output path");
    anyhow::ensure!(args.memory_gib > 0, "memory budget must be positive");
    anyhow::ensure!((8192..=65536).contains(&args.max_length));
    let memory_budget_bytes = args
        .memory_gib
        .checked_mul(1 << 30)
        .ok_or_else(|| anyhow::anyhow!("memory budget overflow"))?;
    let sources = [
        std::fs::read_to_string(&args.reference)?,
        std::fs::read_to_string(&args.candidate)?,
    ];
    let bytes = std::fs::read(&args.input)?;
    let request: Request = serde_json::from_slice(&bytes)?;
    let hash = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    let mut report = serde_json::json!({
        "state": "loading", "model": format!("{:?}", args.model),
        "precision": format!("{:?}", args.precision), "memory_gib": args.memory_gib,
        "max_length": args.max_length, "rounds": args.rounds,
        "reference": {"path": args.reference, "sha256": hash(sources[0].as_bytes())},
        "candidate": {"path": args.candidate, "sha256": hash(sources[1].as_bytes())},
        "input": {"path": args.input, "sha256": hash(&bytes)},
        "binary_sha256": hash(&std::fs::read(std::env::current_exe()?)?),
        "environment": std::env::vars().filter(|(k, _)| k.starts_with("CLEF_")).collect::<std::collections::BTreeMap<_, _>>(),
        "warmup": [], "samples": [],
        "note": "One resident model. Source switching resets the graph before timing. Every timed request includes encoding, graph preparation, GPU execution and readback; loading and first compilation are excluded. Every prediction must match the reference exactly."
    });
    save(&args.output, &report)?;
    let start = Instant::now();
    let mut model = ClefModel::load(LoadOptions {
        source: Source {
            model: args.model,
            directory: std::env::var_os("CLEF_MODEL_DIR").map(Into::into),
            offline: true,
        },
        encoding: EncodeOptions {
            strict_context: true,
            max_length: args.max_length,
            ..Default::default()
        },
        memory_budget_bytes,
        precision: args.precision,
        ..Default::default()
    })?;
    report["load_seconds"] = start.elapsed().as_secs_f64().into();
    report["state"] = "warmup".into();
    save(&args.output, &report)?;
    let mut expected = None;
    for source in &sources {
        model.set_benchmark_attention_source(Some(source))?;
        let prediction = model.infer(&request)?;
        anyhow::ensure!(
            prediction.input_tokens >= 8192,
            "request does not select the attention override"
        );
        let questions = serde_json::to_value(&prediction.questions)?;
        report["warmup"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::to_value(&prediction)?);
        save(&args.output, &report)?;
        if let Some(reference) = &expected {
            anyhow::ensure!(
                reference == &questions,
                "attention candidate changed predictions; raw outputs saved"
            );
        } else {
            expected = Some(questions);
        }
    }
    report["state"] = "measuring".into();
    save(&args.output, &report)?;
    let mut totals = [0.; 2];
    for round in 0..args.rounds {
        for variant in [round % 2, 1 - round % 2] {
            model.set_benchmark_attention_source(Some(&sources[variant]))?;
            let started = Instant::now();
            let prediction = model.infer(&request)?;
            let seconds = started.elapsed().as_secs_f64();
            let exact = expected.as_ref() == Some(&serde_json::to_value(&prediction.questions)?);
            let sample = serde_json::json!({"round": round, "variant": variant,
                "wall_seconds": seconds, "input_tokens": prediction.input_tokens,
                "input_tok_s": prediction.input_tokens as f64 / seconds,
                "timings": prediction.timings, "exact": exact});
            report["samples"]
                .as_array_mut()
                .unwrap()
                .push(sample.clone());
            save(&args.output, &report)?;
            anyhow::ensure!(exact, "timed replay changed predictions");
            totals[variant] += seconds;
            println!("{sample}");
        }
    }
    if args.rounds > 0 {
        report["request_speedup"] = (totals[0] / totals[1]).into();
    }
    report["state"] = "complete".into();
    report["all_predictions_exact"] = true.into();
    save(&args.output, &report)?;
    println!(
        "Complete; paired request speedup: {}",
        report["request_speedup"]
    );
    Ok(())
}
