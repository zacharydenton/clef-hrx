use clap::Parser;
use clef_hrx::{
    ClefModel, EncodeOptions, LoadOptions, Request, Result, benchmarking::ReferenceKernels,
    checkpoint::Source,
};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    /// Requests to compare; defaults to examples/invoice.json.
    paths: Vec<PathBuf>,
    /// Paired timing rounds after correctness checks; 0 checks outputs only.
    #[arg(long, default_value_t = 6)]
    rounds: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
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
            max_length: 1536,
            ..Default::default()
        },
        memory_budget_bytes: 60usize << 30,
        ..Default::default()
    })?;
    eprintln!("Model load: {:.3} s", started.elapsed().as_secs_f64());
    for path in paths {
        eprintln!("Request: {}", path.display());
        let request: Request = serde_json::from_slice(&std::fs::read(&path)?)?;
        compare(&mut model, &request, args.rounds)?;
    }
    Ok(())
}

fn compare(model: &mut ClefModel, request: &Request, rounds: usize) -> Result<()> {
    let variants = [
        (
            "serial_norm_old_gemm",
            ReferenceKernels {
                norm: true,
                gemm: true,
            },
        ),
        (
            "parallel_norm_old_gemm",
            ReferenceKernels {
                norm: false,
                gemm: true,
            },
        ),
        ("parallel_norm_new_gemm", ReferenceKernels::default()),
    ];
    let mut predictions = Vec::new();
    for (_, config) in variants {
        model.set_reference_kernels(config)?;
        predictions.push(model.infer(request)?);
    }
    assert_eq!(
        serde_json::to_value(&predictions[1].questions)?,
        serde_json::to_value(&predictions[2].questions)?,
        "GEMM must preserve predictions exactly"
    );
    let max_norm_probability_change = predictions[0]
        .questions
        .iter()
        .zip(&predictions[1].questions)
        .flat_map(|(a, b)| {
            a.probabilities
                .iter()
                .zip(&b.probabilities)
                .map(|(a, b)| (a - b).abs())
        })
        .fold(0.0f32, f32::max);
    eprintln!(
        "tokens={}, max normalization probability change={max_norm_probability_change}",
        predictions[0].input_tokens
    );
    if rounds == 0 {
        return Ok(());
    }
    let mut gpu_ms = [0.; 3];
    let mut wall_ms = [0.; 3];
    // Rotate execution order to distribute clock and background-load drift.
    for round in 0..rounds {
        for j in 0..3 {
            let i = (round + j) % 3;
            model.set_reference_kernels(variants[i].1)?;
            let started = Instant::now();
            let prediction = model.infer(request)?;
            wall_ms[i] += started.elapsed().as_secs_f64() * 1000.;
            gpu_ms[i] += prediction.timings.inference_ms;
            assert_eq!(
                serde_json::to_value(&predictions[i].questions)?,
                serde_json::to_value(&prediction.questions)?,
                "graph reuse must be deterministic"
            );
        }
    }
    for (i, (name, _)) in variants.iter().enumerate() {
        println!(
            "{name}: inference={:.3} ms request={:.3} ms",
            gpu_ms[i] / rounds as f64,
            wall_ms[i] / rounds as f64
        );
    }
    println!(
        "Normalization speedup: {:.3}x; GEMM speedup: {:.3}x; combined: {:.3}x",
        gpu_ms[0] / gpu_ms[1],
        gpu_ms[1] / gpu_ms[2],
        gpu_ms[0] / gpu_ms[2]
    );
    Ok(())
}
