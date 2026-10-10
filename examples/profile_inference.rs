use clap::Parser;
use clef_hrx::{ClefModel, EncodeOptions, LoadOptions, Request, Result, checkpoint::Source};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    /// Requests to profile with one resident model.
    paths: Vec<PathBuf>,
    #[arg(long, default_value_t = 1536)]
    max_length: usize,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.rounds > 0, "profiling rounds must be positive");
    let paths = if args.paths.is_empty() {
        vec![PathBuf::from("examples/invoice.json")]
    } else {
        args.paths
    };
    let started = Instant::now();
    let mut model = ClefModel::load(LoadOptions {
        source: Source {
            directory: std::env::var_os("CLEF_MODEL_DIR").map(Into::into),
            offline: true,
        },
        encoding: EncodeOptions {
            max_length: args.max_length,
            ..Default::default()
        },
        memory_budget_bytes: 60usize << 30,
        ..Default::default()
    })?;
    eprintln!("Model load: {:.3} s", started.elapsed().as_secs_f64());
    for path in paths {
        println!("Request: {}", path.display());
        let request: Request = serde_json::from_slice(&std::fs::read(path)?)?;
        profile(&mut model, &request, args.rounds)?;
    }
    Ok(())
}

fn profile(model: &mut ClefModel, request: &Request, rounds: usize) -> Result<()> {
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
    for (label, (count, ms)) in rows {
        println!(
            "{label:32} {:5} dispatches {:9.3} ms",
            count / rounds,
            ms / rounds as f64
        );
    }
    Ok(())
}
