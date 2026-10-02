use clef_hrx::{ClefModel, EncodeOptions, LoadOptions, Request, Result, checkpoint::Source};
use std::{collections::BTreeMap, time::Instant};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/invoice.json".into());
    let request: Request = serde_json::from_slice(&std::fs::read(path)?)?;
    let started = Instant::now();
    let mut model = ClefModel::load(LoadOptions {
        source: Source {
            directory: std::env::var_os("CLEF_MODEL_DIR").map(Into::into),
            offline: true,
        },
        encoding: EncodeOptions {
            max_length: 1536,
            ..Default::default()
        },
        memory_budget_bytes: 60usize << 30,
        ..Default::default()
    })?;
    eprintln!("Model load: {:.3} s", started.elapsed().as_secs_f64());
    let before = model.infer(&request)?;
    eprintln!(
        "Ordinary graph, {} tokens: {:?}",
        before.input_tokens, before.timings
    );
    let mut totals = BTreeMap::<String, (usize, f64)>::new();
    for _ in 0..3 {
        let profile = model.profile(&request)?;
        for span in profile.intervals {
            let entry = totals.entry(span.label).or_default();
            entry.0 += 1;
            entry.1 +=
                (span.end_tick - span.start_tick) as f64 / profile.frequency_hz as f64 * 1000.;
        }
    }
    let after = model.infer(&request)?;
    assert_eq!(
        serde_json::to_value(&before.questions)?,
        serde_json::to_value(&after.questions)?
    );
    let mut rows: Vec<_> = totals.into_iter().collect();
    rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    println!("Serialized diagnostic timings; includes profiling barriers.");
    for (label, (count, ms)) in rows {
        println!("{label:24} {:4} dispatches {:9.3} ms", count / 3, ms / 3.);
    }
    Ok(())
}
