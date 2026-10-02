use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use clef_hrx::{ClefModel, EncodeOptions, Encoder, LoadOptions, Request, checkpoint::Source};
use std::{
    io::{self, BufRead, Read},
    path::PathBuf,
};
#[derive(Parser)]
#[command(version, about = "Native CLEF typed decisions on AMD gfx1151")]
struct Args {
    #[arg(long, global = true)]
    model_dir: Option<PathBuf>,
    #[arg(long, global = true)]
    offline: bool,
    #[arg(long, global = true, default_value_t = 0)]
    device: i32,
    #[arg(long, global = true, default_value_t = 80)]
    memory_gib: usize,
    #[arg(long, global = true, default_value_t = 16384)]
    max_length: usize,
    #[arg(long, global = true)]
    max_state_tokens: Option<usize>,
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    Inspect,
    Encode {
        #[arg(default_value = "-")]
        input: PathBuf,
    },
    Decide {
        #[arg(default_value = "-")]
        input: PathBuf,
        #[arg(long)]
        jsonl: bool,
        #[arg(long)]
        raw: bool,
    },
}
fn reader(path: &PathBuf) -> Result<Box<dyn BufRead>> {
    Ok(if path.as_os_str() == "-" {
        Box::new(io::BufReader::new(io::stdin()))
    } else {
        Box::new(io::BufReader::new(std::fs::File::open(path)?))
    })
}
fn request(path: &PathBuf) -> Result<Request> {
    let mut s = String::new();
    reader(path)?.read_to_string(&mut s)?;
    Ok(serde_json::from_str(&s)?)
}
fn main() -> Result<()> {
    let args = Args::parse();
    let source = Source {
        directory: args.model_dir,
        offline: args.offline,
    };
    let encoding = EncodeOptions {
        max_length: args.max_length,
        max_state_tokens: args.max_state_tokens,
    };
    ensure!(
        (1..=16384).contains(&encoding.max_length),
        "max_length must be 1..=16384"
    );
    let options = LoadOptions {
        source: source.clone(),
        device: args.device,
        memory_budget_bytes: args
            .memory_gib
            .checked_mul(1 << 30)
            .context("memory budget overflow")?,
        encoding,
    };
    match args.command {
        Action::Inspect => println!("{}", serde_json::to_string_pretty(&source.inspect()?)?),
        Action::Encode { input } => {
            source.validate_config()?;
            let encoder = Encoder::load(source.resolve("tokenizer.json")?)?;
            println!(
                "{}",
                serde_json::to_string(&encoder.encode_record(&request(&input)?, encoding)?)?
            );
        }
        Action::Decide { input, jsonl, raw } => {
            let mut model = ClefModel::load(options)?;
            let requests: Box<dyn Iterator<Item = Result<Request>>> = if jsonl {
                Box::new(
                    reader(&input)?
                        .lines()
                        .filter(|line| line.as_ref().map_or(true, |s| !s.trim().is_empty()))
                        .map(|line| Ok(serde_json::from_str(&line?)?)),
                )
            } else {
                Box::new(std::iter::once(request(&input)))
            };
            for req in requests {
                let req = req?;
                let p = model.infer(&req)?;
                if raw {
                    println!("{}", serde_json::to_string(&p)?);
                } else {
                    println!(
                        "{}",
                        serde_json::to_string(&ClefModel::response(&req, &p)?)?
                    );
                }
            }
        }
    }
    Ok(())
}
