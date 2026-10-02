use clef_hrx::{Result, benchmarking::DeltaBenchmark};
use std::collections::BTreeMap;

fn main() -> Result<()> {
    let tokens = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(260);
    let mut graph = DeltaBenchmark::profiled(tokens)?;
    let mut totals = BTreeMap::<String, (usize, f64)>::new();
    for _ in 0..5 {
        let profile = graph.profile()?;
        for span in profile.intervals {
            let entry = totals.entry(span.label).or_default();
            entry.0 += 1;
            entry.1 +=
                (span.end_tick - span.start_tick) as f64 / profile.frequency_hz as f64 * 1000.;
        }
    }
    let mut rows: Vec<_> = totals.into_iter().collect();
    rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    println!(
        "Serialized diagnostic timings; includes profiling barriers. tokens={tokens}, allocated_bytes={}",
        graph.allocated_bytes()
    );
    for (label, (count, ms)) in rows {
        println!("{label:24} {:3} dispatches {:8.3} ms", count / 5, ms / 5.);
    }
    Ok(())
}
