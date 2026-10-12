use clap::Parser;
use clef_hrx::{
    ClefModel, EncodeOptions, LoadOptions, Precision, Request, Result, checkpoint::Source,
};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum, default_value_t = clef_hrx::ModelVariant::Clef)]
    model: clef_hrx::ModelVariant,
    /// Requests to profile with one resident model.
    paths: Vec<PathBuf>,
    #[arg(long, default_value_t = 1536)]
    max_length: usize,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long, value_enum, default_value_t = Precision::Bf16)]
    precision: Precision,
    /// Maximum engine allocation in GiB.
    #[arg(long, default_value_t = 80)]
    memory_gib: usize,
    /// Write ordinary inference results for accuracy comparisons.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.rounds > 0, "profiling rounds must be positive");
    anyhow::ensure!(args.memory_gib > 0, "memory budget must be positive");
    let memory_budget_bytes = args
        .memory_gib
        .checked_mul(1usize << 30)
        .ok_or_else(|| anyhow::anyhow!("memory budget overflows usize"))?;
    let paths = if args.paths.is_empty() {
        vec![PathBuf::from("examples/invoice.json")]
    } else {
        args.paths
    };
    let started = Instant::now();
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
    eprintln!("Model load: {:.3} s", started.elapsed().as_secs_f64());
    let mut results = Vec::new();
    for path in paths {
        println!("Request: {}", path.display());
        let request: Request = serde_json::from_slice(&std::fs::read(path)?)?;
        results.push(profile(&mut model, &request, args.rounds)?);
    }
    if let Some(path) = args.output {
        std::fs::write(path, serde_json::to_vec_pretty(&results)?)?;
    }
    Ok(())
}

fn profile(model: &mut ClefModel, request: &Request, rounds: usize) -> Result<serde_json::Value> {
    let before = model.infer(request)?;
    eprintln!(
        "Ordinary graph, {} tokens: {:?}",
        before.input_tokens, before.timings
    );
    let mut totals = BTreeMap::<String, (usize, f64)>::new();
    for _ in 0..rounds {
        let profile = model.profile(request)?;
        for span in profile.intervals {
            let entry = totals.entry(span.label).or_default();
            entry.0 += 1;
            entry.1 +=
                (span.end_tick - span.start_tick) as f64 / profile.frequency_hz as f64 * 1000.;
        }
    }
    let after = model.infer(request)?;
    eprintln!("Post-profile ordinary graph: {:?}", after.timings);
    assert_eq!(
        serde_json::to_value(&before.questions)?,
        serde_json::to_value(&after.questions)?
    );
    let mut rows: Vec<_> = totals.into_iter().collect();
    rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    println!("Serialized diagnostic timings; includes profiling barriers.");
    let intervals: Vec<_> = rows.iter().map(|(label, (count, ms))| {
        serde_json::json!({"label": label, "dispatches": count / rounds, "milliseconds": ms / rounds as f64})
    }).collect();
    for (label, (count, ms)) in rows {
        println!(
            "{label:32} {:5} dispatches {:9.3} ms",
            count / rounds,
            ms / rounds as f64
        );
    }
    Ok(
        serde_json::json!({"before": before, "after": after, "intervals": intervals,
        "note": "Serialized diagnostic timings include profiling barriers; use ordinary timings for throughput."}),
    )
}
