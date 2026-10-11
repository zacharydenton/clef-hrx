use clap::Parser;
use clef_hrx::{
    ClefModel, EncodeOptions, LoadOptions, Request, Result, benchmarking::ReferenceKernels,
    checkpoint::Source,
};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    paths: Vec<PathBuf>,
    #[arg(long, default_value_t = 4032)]
    max_length: usize,
    #[arg(long, default_value_t = 2)]
    rounds: usize,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let mut model = ClefModel::load(LoadOptions {
        source: Source {
            model: Default::default(),
            directory: std::env::var_os("CLEF_MODEL_DIR").map(Into::into),
            offline: true,
        },
        encoding: EncodeOptions {
            strict_context: false,
            max_length: a.max_length,
            ..Default::default()
        },
        memory_budget_bytes: 60usize << 30,
        ..Default::default()
    })?;
    for path in a.paths {
        let request: Request = serde_json::from_slice(&std::fs::read(&path)?)?;
        let mut reference = None;
        for round in 0..=a.rounds {
            for delta in [true, false] {
                model.set_reference_kernels(ReferenceKernels {
                    delta,
                    ..Default::default()
                })?;
                let p = model.infer(&request)?;
                println!(
                    "{} round={} chunked={} tokens={} timings={:?}",
                    path.display(),
                    round,
                    delta,
                    p.input_tokens,
                    p.timings
                );
                if delta {
                    reference = Some(p);
                } else {
                    let old = reference.as_ref().unwrap();
                    let error = old
                        .questions
                        .iter()
                        .zip(&p.questions)
                        .flat_map(|(a, b)| {
                            a.probabilities
                                .iter()
                                .zip(&b.probabilities)
                                .map(|(a, b)| (a - b).abs())
                        })
                        .fold(0f32, f32::max);
                    println!("maximum probability change: {error}");
                    anyhow::ensure!(
                        p.questions
                            .iter()
                            .flat_map(|q| &q.probabilities)
                            .all(|x| x.is_finite()),
                        "non-finite probabilities"
                    );
                }
            }
        }
    }
    Ok(())
}
