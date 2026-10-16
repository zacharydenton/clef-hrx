//! Compare expert/chunk/attention variants with one resident Omni checkpoint.
use clap::Parser;
use clef_hrx::{
    ClefModel, EncodeOptions, LoadOptions, ModelVariant, Precision, Request, Result,
    benchmarking::OmniKernelConfig, checkpoint::Source,
};
use sha2::{Digest, Sha256};
use std::{io::Write, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    reference_dir: PathBuf,
    #[arg(long)]
    candidate_dir: PathBuf,
    /// Use the default expert kernel below this routed chunk length.
    #[arg(long, default_value_t = 0)]
    candidate_min_tokens: usize,
    /// Optional shared-attention candidate, in addition to the tiled reference.
    #[arg(long)]
    attention_source: Option<PathBuf>,
    /// Add a shared-attention reference using 2,048-token chunks and padding.
    #[arg(long, requires = "attention_source")]
    reference_attention_source: Option<PathBuf>,
    /// Restrict diagnostics; the first retained variant supplies the reference.
    #[arg(long, value_delimiter = ',')]
    variants: Vec<String>,
    /// Use shared route ordering in every variant except the retained reference.
    #[arg(long)]
    sorted_combine: bool,
    /// Share gate/up packing and fuse SwiGLU in variants after the reference.
    #[arg(long)]
    fused_mlp: bool,
    /// Fuse Q/K RMS normalization with RoPE in variants after the reference.
    #[arg(long)]
    fused_norm_rope: bool,
    /// Fuse SwiGLU into INT4 expert packing in variants after the reference.
    #[arg(long)]
    pack_swiglu: bool,
    /// Pack Q/K once for a prepacked INT8 attention source (causal text only).
    #[arg(long)]
    prepack_qk: bool,
    /// Compact populated INT4 expert tiles in variants after the reference.
    #[arg(long)]
    compact_experts: bool,
    /// Reuse the padded text backbone graph across different requests.
    #[arg(long)]
    reuse_backbone: bool,
    #[arg(long, value_enum, default_value_t = Precision::Int4)]
    precision: Precision,
    #[arg(long, default_value_t = 24)]
    memory_gib: usize,
    #[arg(long, default_value_t = 4096)]
    max_length: usize,
    #[arg(long)]
    output: PathBuf,
    input: PathBuf,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let input = std::fs::read_to_string(&args.input)?;
    let requests: Vec<Request> = input
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let hash = |data: &[u8]| format!("{:x}", Sha256::digest(data));
    let sources: Vec<_> =
        [&args.reference_dir, &args.candidate_dir]
            .into_iter()
            .map(|dir| {
                ["omni_expert_bf16.loom", "omni_expert_i4.loom"].into_iter().map(|name| {
            let path = dir.join(name);
            Ok(serde_json::json!({"path": path, "sha256": hash(&std::fs::read(&path)?)}))
        }).collect::<Result<Vec<_>>>()
            })
            .collect::<Result<_>>()?;
    writeln!(
        output,
        "{}",
        serde_json::json!({"kind":"metadata", "precision":args.precision,
        "fused_norm_rope":args.fused_norm_rope,
        "reuse_backbone":args.reuse_backbone, "pack_swiglu":args.pack_swiglu, "prepack_qk":args.prepack_qk,
        "binary_sha256":hash(&std::fs::read(std::env::current_exe()?)?), "input_sha256":hash(input.as_bytes()),
        "sources":sources, "sorted_combine": args.sorted_combine, "fused_mlp":args.fused_mlp, "compact_experts":args.compact_experts, "candidate_min_tokens": args.candidate_min_tokens, "variants_requested": args.variants, "attention_source": args.attention_source.as_ref().map(|p| -> Result<_> { Ok(serde_json::json!({"path":p,"sha256":hash(&std::fs::read(p)?)})) }).transpose()?, "reference_attention_source":args.reference_attention_source.as_ref().map(|p| -> Result<_> { Ok(serde_json::json!({"path":p,"sha256":hash(&std::fs::read(p)?)})) }).transpose()?, "max_length":args.max_length, "memory_gib":args.memory_gib,
        "note":"Resident diagnostic, not full-suite qualification. Each sample includes encoding, preparation, GPU and readback. Two replays per configuration; first compilation is reported separately."})
    )?;
    output.flush()?;
    let start = Instant::now();
    let mut model = ClefModel::load(LoadOptions {
        source: Source {
            model: ModelVariant::ClefOmni,
            directory: Some(args.model_dir),
            offline: true,
        },
        encoding: EncodeOptions {
            strict_context: true,
            max_length: args.max_length,
            ..Default::default()
        },
        memory_budget_bytes: args
            .memory_gib
            .checked_mul(1 << 30)
            .ok_or_else(|| anyhow::anyhow!("memory budget overflow"))?,
        precision: args.precision,
        ..Default::default()
    })?;
    eprintln!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());
    let mut variants = vec![
        ("reference", Some(args.reference_dir), 256, false, 1),
        ("packets", None, 256, false, 1),
        ("packets_sorted", None, 256, false, 1),
        ("packets_fused", None, 256, false, 1),
        ("packets_compact", None, 256, false, 1),
        ("packets_1024", None, 1024, false, 1),
        ("packets_2048", None, 2048, false, 1),
        ("packets_compact_2048", None, 2048, false, 1),
        (
            "candidate_1024",
            Some(args.candidate_dir.clone()),
            1024,
            false,
            1,
        ),
        (
            "candidate_attention_1024",
            Some(args.candidate_dir.clone()),
            1024,
            true,
            1,
        ),
    ];
    let attention = args
        .attention_source
        .as_ref()
        .map(std::fs::read_to_string)
        .transpose()?;
    let reference_attention = args
        .reference_attention_source
        .as_ref()
        .map(std::fs::read_to_string)
        .transpose()?;
    if reference_attention.is_some() {
        variants.push((
            "shared_reference_2048",
            Some(args.candidate_dir.clone()),
            2048,
            true,
            128,
        ));
    }
    if attention.is_some() {
        variants.push((
            "shared_attention_1024",
            Some(args.candidate_dir.clone()),
            1024,
            true,
            1,
        ));
        variants.push((
            "shared_attention_2048",
            Some(args.candidate_dir.clone()),
            2048,
            true,
            1,
        ));
        variants.push((
            "shared_padded128_1024",
            Some(args.candidate_dir.clone()),
            1024,
            true,
            128,
        ));
        variants.push((
            "shared_padded128_2048",
            Some(args.candidate_dir.clone()),
            2048,
            true,
            128,
        ));
    }
    if !args.variants.is_empty() {
        anyhow::ensure!(
            args.variants
                .iter()
                .all(|selected| variants.iter().any(|v| v.0 == selected)),
            "unknown variant or missing attention source"
        );
        variants.retain(|v| args.variants.iter().any(|selected| selected == v.0));
    }
    let reference_name = variants[0].0;
    for (index, request) in requests.iter().enumerate() {
        let mut reference = None;
        for (name, directory, chunk, tiled, alignment) in &variants {
            let is_reference = *name == reference_name || *name == "shared_reference_2048";
            model.set_benchmark_omni_kernels(OmniKernelConfig {
                directory: directory.clone(),
                chunk: *chunk,
                tiled_attention: *tiled,
                prepack_qk: args.prepack_qk && !is_reference,
                alignment: *alignment,
                minimum_tokens: if *name == "reference" {
                    0
                } else {
                    args.candidate_min_tokens
                },
                sorted_combine: args.sorted_combine && !is_reference,
                fused_mlp: args.fused_mlp && !is_reference,
                fused_norm_rope: args.fused_norm_rope && !is_reference,
                pack_swiglu: args.pack_swiglu && !is_reference,
                compact_experts: args.compact_experts && !is_reference,
                reuse_backbone: args.reuse_backbone && !is_reference,
            })?;
            model.set_benchmark_attention_source(if *name == "shared_reference_2048" {
                reference_attention.as_deref()
            } else if name.starts_with("shared_") {
                attention.as_deref()
            } else {
                None
            })?;
            let mut previous = None;
            for replay in 0..2 {
                let started = Instant::now();
                let prediction = model.infer(request)?;
                let wall = started.elapsed().as_secs_f64();
                let questions = serde_json::to_value(&prediction.questions)?;
                if *name == reference_name {
                    reference = Some(questions.clone());
                }
                let exact = reference.as_ref() == Some(&questions);
                let repeated = previous.as_ref().is_none_or(|p| p == &questions);
                writeln!(
                    output,
                    "{}",
                    serde_json::json!({"kind":"sample", "request":index,
                    "variant":name, "replay":replay,"wall_seconds":wall,
                    "input_tok_s": prediction.input_tokens as f64/wall,
                    "exact_reference":exact,"exact_repeat":repeated,"prediction":prediction})
                )?;
                output.flush()?;
                anyhow::ensure!(repeated, "graph replay changed output");
                previous = Some(questions);
                eprintln!(
                    "request={index} variant={name} replay={replay} tok/s={:.1} exact={exact}",
                    prediction.input_tokens as f64 / wall
                );
            }
        }
    }
    Ok(())
}
