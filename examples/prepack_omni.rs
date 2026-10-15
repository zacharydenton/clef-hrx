//! Prepare the existing INT4 weights one layer at a time under a 4 GiB budget.
use clap::Parser;
use clef_hrx::{ModelVariant, Result, benchmarking::prepack_omni_layers, checkpoint::Source};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    cache_dir: PathBuf,
    #[arg(long, default_value_t = 0)]
    first_layer: usize,
    #[arg(long, default_value_t = 48)]
    end_layer: usize,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let source = Source {
        model: ModelVariant::ClefOmni,
        directory: Some(args.model_dir),
        offline: true,
    };
    let reports = prepack_omni_layers(&source, args.cache_dir, args.first_layer, args.end_layer)?;
    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}
