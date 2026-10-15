//! Memory-bounded full BF16 verification using the production operators.
//! Streaming changes weight residency and adds lossless BF16 round trips between
//! layers. It is deliberately a test, not a resident throughput benchmark.
use super::*;
use crate::{ModelVariant, checkpoint::Source, encoding::Encoder};
use half::bf16;
use std::{path::PathBuf, time::Instant};

#[cfg(target_os = "linux")]
fn discard_checkpoint_cache(files: &[std::fs::File]) -> Result<()> {
    use std::os::fd::AsRawFd;
    for file in files {
        // These are read-only checkpoint files. Completed uploads have their
        // own device storage, and done_with has released the mapped data pages.
        let code =
            unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        ensure!(
            code == 0,
            "checkpoint cache advice: {}",
            std::io::Error::from_raw_os_error(code)
        );
    }
    Ok(())
}
#[cfg(not(target_os = "linux"))]
fn discard_checkpoint_cache(_: &[std::fs::File]) -> Result<()> {
    Ok(())
}

fn load_matching(
    e: &mut Engine,
    checkpoint: &Checkpoint,
    predicate: impl Fn(&str) -> bool,
) -> Result<()> {
    for name in checkpoint.names().filter(|name| predicate(name)) {
        let tensor = checkpoint.tensor(name)?;
        let rows = tensor.shape.first().copied().unwrap_or(1);
        let cols = tensor.shape.iter().skip(1).product();
        e.weight(name.into(), rows, cols, tensor.bytes)?;
        checkpoint.done_with(name)?;
    }
    Ok(())
}
fn values(e: &mut Engine, tensor: &Tensor) -> Result<Vec<bf16>> {
    Ok(e.read_f32(tensor)?
        .into_iter()
        .map(bf16::from_f32)
        .collect())
}
fn input(e: &mut Engine, data: &[bf16]) -> Result<Tensor> {
    e.input(
        data.len() / 2048,
        2048,
        DType::Bf16,
        bytemuck::cast_slice(data),
    )
}

#[test]
#[ignore = "requires pinned Omni checkpoint and gfx1151; streams full BF16 layers"]
fn streaming_bf16_matches_release() -> Result<()> {
    let model = ModelVariant::ClefOmni;
    let source = Source {
        model,
        directory: std::env::var_os("CLEF_OMNI_MODEL_DIR").map(Into::into),
        offline: true,
    };
    let output = PathBuf::from(std::env::var("CLEF_OMNI_BF16_OUTPUT")?);
    std::fs::create_dir(&output)?;
    let encoder = Encoder::load_for_model(source.resolve("tokenizer.json")?, model)?;
    // The test holds the pinned, read-only snapshot throughout validation.
    let checkpoint = unsafe { Checkpoint::open(&source)? };
    let files = source
        .inspect()?
        .files
        .into_iter()
        .filter(|name| name.ends_with(".safetensors"))
        .map(|name| Ok(std::fs::File::open(source.resolve(&name)?)?))
        .collect::<Result<Vec<_>>>()?;
    discard_checkpoint_cache(&files)?;
    let context = hrx::inference::ModelContext::new(Default::default())?;
    let engine = || -> Result<Engine> {
        let mut e = Engine::new(&context)?;
        e.architecture = model.architecture();
        e.quantized = Default::default();
        Ok(e)
    };
    let mut passed = true;
    for case in std::env::var("CLEF_OMNI_BF16_CASES")
        .unwrap_or_else(|_| "0,4".into())
        .split(',')
    {
        let case: usize = case.parse()?;
        let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(format!(
            "tests/fixtures/omni/decisions-{case}.json"
        ))?)?;
        let request = serde_json::from_value(reference["request"].clone())?;
        let record = encoder.encode_record(
            &request,
            crate::EncodeOptions {
                max_length: 1536,
                strict_context: true,
                max_state_tokens: None,
            },
        )?;
        ensure!(
            serde_json::to_value(&record.input_ids)? == reference["input_ids"],
            "oracle tokens differ"
        );
        let start = Instant::now();
        let mut peak = 0;
        let (mut hidden, deepstack) = {
            let mut e = engine()?;
            e.embedding_rows.insert(
                "model.language_model.embed_tokens.weight".into(),
                checkpoint.embedding_rows("model.language_model.embed_tokens.weight")?,
            );
            load_matching(&mut e, &checkpoint, |name| {
                (!record.media.items.is_empty() && name.starts_with("model.visual."))
                    || (record
                        .media
                        .omni
                        .as_ref()
                        .is_some_and(|m| !m.audio.is_empty())
                        && name.starts_with("model.audio_tower."))
            })?;
            let (x, deepstack) = crate::model::embed_record(&mut e, &record)?;
            let mut graph = e.finish()?;
            e.run(&mut graph)?;
            peak = peak.max(e.allocated_bytes());
            let deepstack = deepstack
                .iter()
                .map(|items| {
                    items
                        .iter()
                        .map(|(at, tensor)| Ok((*at, values(&mut e, tensor)?)))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            (values(&mut e, &x)?, deepstack)
        };
        for layer in 0..48 {
            let mut e = engine()?;
            let prefix = format!("model.language_model.layers.{layer}.");
            load_matching(&mut e, &checkpoint, |name| {
                (name.starts_with(&prefix) && !name.contains(".mlp.experts."))
                    || (layer == 47 && name == "model.language_model.norm.weight")
            })?;
            load_expert_layer(&mut e, &checkpoint, layer)?;
            let x = input(&mut e, &hidden)?;
            let mut features = vec![Vec::new(); 3];
            if let Some(items) = deepstack.get(layer) {
                for (at, data) in items {
                    features[layer].push((*at, input(&mut e, data)?));
                }
            }
            let x = forward_layers(&mut e, x, &record.position_ids, &features, layer..layer + 1)?;
            let mut graph = e.finish()?;
            e.run(&mut graph)?;
            peak = peak.max(e.allocated_bytes());
            hidden = values(&mut e, &x)?;
            if (layer + 1) % 8 == 0 {
                eprintln!("BF16 case {case}: layer {}/48", layer + 1);
            }
            // Limit CPU page-cache pressure as well as device residency.
            discard_checkpoint_cache(&files)?;
        }
        let mut e = engine()?;
        e.embedding_rows.insert(
            "lm_head.weight".into(),
            checkpoint.embedding_rows("lm_head.weight")?,
        );
        load_matching(&mut e, &checkpoint, |name| name.starts_with("head."))?;
        let hidden = input(&mut e, &hidden)?;
        let logits = crate::head::forward(&mut e, &hidden, &record)?;
        let mut graph = e.finish()?;
        e.run(&mut graph)?;
        peak = peak.max(e.allocated_bytes());
        let logits = e.read_f32(&logits)?;
        let mut offset = 0;
        let mut error = 0.0f64;
        let mut differences = Vec::new();
        let mut questions = Vec::new();
        for (q, expected) in record
            .questions
            .iter()
            .zip(reference["questions"].as_array().unwrap())
        {
            let n = q.option_ids.len();
            let actual = &logits[offset..offset + n];
            offset += n;
            let probabilities = crate::schema::softmax(actual)?;
            let expected: Vec<_> = expected["probabilities"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.as_f64().unwrap())
                .collect();
            for (a, b) in probabilities.iter().zip(&expected) {
                error = error.max((*a as f64 - b).abs());
            }
            let best = |v: &[f64]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
            if best(&probabilities.iter().map(|&v| v as f64).collect::<Vec<_>>()) != best(&expected)
            {
                differences.push(q.question_id.clone());
            }
            questions.push(serde_json::json!({"question_id":q.question_id,"option_ids":q.option_ids,"logits":actual,"probabilities":probabilities}));
        }
        let result = serde_json::json!({"input_tokens":record.input_ids.len(),"questions":questions,
            "max_probability_error":error,"argmax_differences":differences,"peak_engine_bytes":peak,
            "elapsed_seconds_including_streaming":start.elapsed().as_secs_f64()});
        std::fs::write(
            output.join(format!("{case}.json")),
            serde_json::to_vec_pretty(&result)?,
        )?;
        eprintln!(
            "BF16 case {case}: max probability error {error:.6}, differences {differences:?}"
        );
        passed &= error <= 0.03 && differences.is_empty();
    }
    ensure!(passed, "BF16 oracle comparison failed; see saved results");
    Ok(())
}
