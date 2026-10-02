use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use clef_hrx::{ClefModel, EncodeOptions, Encoder, LoadOptions, Request, checkpoint::Source};
use std::{
    io::{self, BufRead, Read},
    path::{Path, PathBuf},
};
#[derive(Parser)]
#[command(version, about = "Native CLEF typed decisions on AMD gfx1151")]
struct Args {
    /// Read checkpoint files from a local snapshot.
    #[arg(long, global = true)]
    model_dir: Option<PathBuf>,
    /// Use cached checkpoint files without downloading.
    #[arg(long, global = true)]
    offline: bool,
    /// HRX GPU index.
    #[arg(long, global = true, default_value_t = 0)]
    device: i32,
    /// GPU allocation budget in GiB.
    #[arg(long, global = true, default_value_t = 80)]
    memory_gib: usize,
    /// Maximum input tokens, including schema and media (1–16384).
    #[arg(long, global = true, default_value_t = 16384)]
    max_length: usize,
    /// Additional limit on state text tokens.
    #[arg(long, global = true)]
    max_state_tokens: Option<usize>,
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Show checkpoint metadata without loading weights.
    Inspect,
    /// Encode a JSON request without loading GPU weights.
    Encode {
        /// JSON file, or - for stdin.
        #[arg(default_value = "-")]
        input: PathBuf,
    },
    /// Answer a request using the local model.
    Decide {
        /// JSON file, or - for stdin.
        #[arg(default_value = "-")]
        input: PathBuf,
        /// Read one request per line and keep the model loaded.
        #[arg(long)]
        jsonl: bool,
        /// Include logits, unrounded probabilities, and timings.
        #[arg(long)]
        raw: bool,
    },
}
fn reader(path: &Path) -> Result<Box<dyn BufRead>> {
    Ok(if path.as_os_str() == "-" {
        Box::new(io::BufReader::new(io::stdin()))
    } else {
        Box::new(io::BufReader::new(
            std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?,
        ))
    })
}
fn request(path: &Path) -> Result<Request> {
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
            let request = request(&input)?;
            request.validate()?;
            source.validate_config()?;
            let encoder = Encoder::load(source.resolve("tokenizer.json")?)?;
            println!(
                "{}",
                serde_json::to_string(&encoder.encode_record(&request, encoding)?)?
            );
        }
        Action::Decide { input, jsonl, raw } => {
            let requests: Box<dyn Iterator<Item = Result<Request>>> = if jsonl {
                Box::new(
                    reader(&input)?
                        .lines()
                        .enumerate()
                        .filter(|(_, line)| line.as_ref().map_or(true, |s| !s.trim().is_empty()))
                        .map(|(i, line)| {
                            serde_json::from_str(&line?)
                                .with_context(|| format!("request on line {}", i + 1))
                        }),
                )
            } else {
                Box::new(std::iter::once(request(&input)))
            };
            let mut model = None;
            for req in requests {
                let req = req?;
                req.validate()?;
                if model.is_none() {
                    model = Some(ClefModel::load(options.clone())?);
                }
                let p = model.as_mut().context("model not loaded")?.infer(&req)?;
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
