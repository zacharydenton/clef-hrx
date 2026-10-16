//! Compare Omni prefix reuse with ordinary full-request inference in one process.
use anyhow::{Context, ensure};
use clap::Parser;
use clef_hrx::{
    ClefModel, EncodeOptions, LoadOptions, ModelVariant, Precision, Prediction, Request,
    checkpoint::Source,
};
use std::{io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = Precision::Int4)]
    precision: Precision,
    #[arg(long, default_value_t = 28)]
    memory_gib: usize,
    #[arg(long, default_value_t = 0.02)]
    max_probability_delta: f32,
    #[arg(long)]
    output: PathBuf,
    input: PathBuf,
}

fn differences(a: &Prediction, b: &Prediction) -> anyhow::Result<(usize, f32)> {
    ensure!(
        a.input_tokens == b.input_tokens && a.questions.len() == b.questions.len(),
        "prediction shape changed"
    );
    let mut changed = 0;
    let mut delta = 0f32;
    for (a, b) in a.questions.iter().zip(&b.questions) {
        ensure!(
            a.question_id == b.question_id && a.option_ids == b.option_ids,
            "prediction identity changed"
        );
        let winner = |p: &[f32]| {
            p.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        changed += usize::from(winner(&a.probabilities) != winner(&b.probabilities));
        for (a, b) in a.probabilities.iter().zip(&b.probabilities) {
            ensure!(a.is_finite() && b.is_finite(), "nonfinite probability");
            delta = delta.max((a - b).abs());
        }
    }
    Ok((changed, delta))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let input = std::fs::read_to_string(&args.input)?;
    let requests: Vec<Request> = input
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    ensure!(!requests.is_empty(), "empty request corpus");
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let mut model = ClefModel::load(LoadOptions {
        source: Source {
            model: ModelVariant::ClefOmni,
            directory: Some(args.model_dir),
            offline: true,
        },
        precision: args.precision,
        memory_budget_bytes: args
            .memory_gib
            .checked_mul(1 << 30)
            .context("budget overflow")?,
        encoding: EncodeOptions {
            strict_context: true,
            ..Default::default()
        },
        ..Default::default()
    })?;
    let mut reference = Vec::new();
    let mut failed = false;
    for phase in ["uncached", "cached", "cached_reverse"] {
        // Explicitly resetting also exercises cache eviction and reseeding.
        model.set_prefix_cache(phase != "uncached")?;
        let indices: Vec<_> = if phase == "cached_reverse" {
            (0..requests.len()).rev().collect()
        } else {
            (0..requests.len()).collect()
        };
        for index in indices {
            let first = model.infer(&requests[index])?;
            let repeat = model.infer(&requests[index])?;
            if phase == "uncached" {
                ensure!(
                    differences(&first, &repeat)? == (0, 0.),
                    "uncached repeat differs"
                );
                reference.push(repeat.clone());
            } else {
                ensure!(
                    repeat.timings.cached_prefix_tokens > 0,
                    "identical prefix missed"
                );
                if first.timings.cached_prefix_tokens > 0 {
                    ensure!(
                        differences(&first, &repeat)? == (0, 0.),
                        "cached repeat differs"
                    );
                }
            }
            let first_diff = differences(&reference[index], &first)?;
            let repeat_diff = differences(&reference[index], &repeat)?;
            failed |= first_diff.0 != 0
                || repeat_diff.0 != 0
                || first_diff.1 > args.max_probability_delta
                || repeat_diff.1 > args.max_probability_delta;
            writeln!(
                out,
                "{}",
                serde_json::json!({"phase":phase,"index":index,"precision":args.precision,
                "first":first,"repeat":repeat,"first_changed":first_diff.0,"repeat_changed":repeat_diff.0,
                "first_max_probability_delta":first_diff.1,"repeat_max_probability_delta":repeat_diff.1})
            )?;
            out.flush()?;
            eprintln!(
                "{phase} {index}: changed={}, max delta={:.6}, reused={} tokens",
                repeat_diff.0, repeat_diff.1, repeat.timings.cached_prefix_tokens
            );
        }
    }
    model.set_prefix_cache(false)?;
    let restored = model.infer(&requests[0])?;
    ensure!(
        restored.timings.cached_prefix_tokens == 0
            && differences(&reference[0], &restored)? == (0, 0.),
        "disabling cache changed full inference"
    );
    ensure!(
        !failed,
        "prefix cache exceeded comparison gate; see {}",
        args.output.display()
    );
    Ok(())
}
