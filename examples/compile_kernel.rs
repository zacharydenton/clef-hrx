//! Inspect a specialized kernel's resources without opening the GPU.
use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    source: PathBuf,
    #[arg(long, default_value = "clef_op")]
    entry: String,
    /// Repeat for each specialization, for example -D clef.k=2048.
    #[arg(short = 'D', value_parser = parse_config)]
    config: Vec<(String, String)>,
    #[arg(long)]
    assembly: Option<PathBuf>,
}

fn parse_config(value: &str) -> Result<(String, String), String> {
    let (name, value) = value
        .split_once('=')
        .filter(|(name, value)| !name.is_empty() && !value.is_empty())
        .ok_or_else(|| "expected NAME=VALUE".to_owned())?;
    Ok((name.to_owned(), value.to_owned()))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let source = format!(
        "{}\n{}",
        include_str!("../kernels/common.loom"),
        std::fs::read_to_string(&args.source)?
    );
    let compiler = hrx::loom::Compiler::resolve(None)?;
    let mut spec = hrx::loom::Specialization::new(&args.entry);
    for (name, value) in args.config {
        spec.set_config(name, value);
    }
    spec.set_report(hrx::loom::ReportMode::Details);
    spec = spec.with_listing(args.assembly.is_some());
    let artifact = compiler.module(&source).compile(&spec)?;
    if let Some(path) = args.assembly {
        std::fs::write(path, artifact.listing().context("missing target assembly")?)?;
    }
    println!("{}", serde_json::to_string_pretty(&artifact.report())?);
    Ok(())
}
