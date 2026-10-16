use super::*;
use crate::omni::{KvMode, LayerKv};
use sha2::{Digest, Sha256};

pub(super) struct PrefixCache {
    pub key: String,
    layers: Vec<LayerKv>,
    hidden: Tensor,
}

/// Hash the actual encoded prefix and media content, not filenames or questions.
/// Model, weights and precision are fixed for the lifetime of NativeState.
pub(super) fn key(record: &EncodedRecord) -> Result<String> {
    let n = record.prefix_len;
    ensure!(
        n > 0 && n < record.input_ids.len() && record.position_ids.len() == record.input_ids.len(),
        "invalid shared prefix extent"
    );
    let mut hash = Sha256::new();
    hash.update(serde_json::to_vec(&(
        &record.input_ids[..n],
        &record.position_ids[..n],
    ))?);
    for item in &record.media.items {
        ensure!(
            item.token_ranges.iter().all(|r| r[0] < r[1] && r[1] <= n),
            "visual tokens outside prefix"
        );
        hash.update(serde_json::to_vec(&(item.grid, &item.token_ranges))?);
        hash.update(bytemuck::cast_slice(&item.patches));
    }
    if let Some(omni) = &record.media.omni {
        for audio in &omni.audio {
            ensure!(
                audio.token_ranges.iter().all(|r| r[0] < r[1] && r[1] <= n),
                "audio tokens outside prefix"
            );
            hash.update(serde_json::to_vec(&(audio.frames, &audio.token_ranges))?);
            hash.update(bytemuck::cast_slice(&audio.mel));
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

impl NativeState {
    pub(super) fn build_with_prefix(
        &mut self,
        record: &EncodedRecord,
        key: &str,
        hit: bool,
    ) -> Result<Tensor> {
        self.plan = None;
        #[cfg(feature = "bench-internals")]
        {
            self.backbone = None;
        }
        if !hit {
            self.prefix = None;
        }
        self.engine.reset_plan()?;
        let n = record.prefix_len;
        let hidden = if hit {
            let saved = self.prefix.as_ref().context("missing prefix cache")?;
            let x = self.engine.embedding(
                "model.language_model.embed_tokens.weight",
                &record.input_ids[n..],
            )?;
            let suffix = crate::omni::forward_cached(
                &mut self.engine,
                x,
                &record.position_ids[n..],
                &[],
                &mut KvMode::Resume(&saved.layers),
            )?;
            self.engine.cat_rows(&[saved.hidden.clone(), suffix])?
        } else {
            // Populate from an ordinary full forward pass. In particular, the
            // first response retains the original model's graph geometry.
            let (x, deepstack) = embed_record(&mut self.engine, record)?;
            let mut layers = Vec::with_capacity(48);
            let hidden = crate::omni::forward_cached(
                &mut self.engine,
                x,
                &record.position_ids,
                &deepstack,
                &mut KvMode::Capture {
                    tokens: n,
                    layers: &mut layers,
                },
            )?;
            let saved = self.engine.alloc(n, hidden.cols, hidden.dtype)?;
            self.engine.copy_into(&hidden.slice_rows(0, n), &saved)?;
            self.prefix = Some(PrefixCache {
                key: key.into(),
                layers,
                hidden: saved,
            });
            hidden
        };
        crate::head::forward(&mut self.engine, &hidden, record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> EncodedRecord {
        EncodedRecord {
            input_ids: vec![1, 2, 3, 4, 5],
            prefix_len: 3,
            position_ids: (0..5).map(|i| [i; 3]).collect(),
            questions: Vec::new(),
            media: Default::default(),
        }
    }

    #[test]
    #[ignore = "requires pinned Omni tokenizer; checks the real image/state/schema boundary"]
    fn encoded_questions_share_only_the_context_prefix() -> Result<()> {
        let source = Source {
            model: crate::ModelVariant::ClefOmni,
            offline: true,
            ..Default::default()
        };
        let encoder = Encoder::load_for_model(source.resolve("tokenizer.json")?, source.model)?;
        let mut request: Request = serde_json::from_value(serde_json::json!({
            "model":"clef-omni", "state":"The invoice is approved. The total amount is 300 dollars.",
            "images":["tests/fixtures/corpus/blue.png"],
            "questions":{"approved":{"type":"noul","instructions":"Is the invoice approved?"}}
        }))?;
        let first = encoder.encode_record(&request, EncodeOptions::default())?;
        request.questions = serde_json::from_value(serde_json::json!({
            "color":{"type":"choice","instructions":"What color is visible?","criteria":{"blue":"Blue","red":"Red"}}
        }))?;
        let second = encoder.encode_record(&request, EncodeOptions::default())?;
        assert_eq!(key(&first)?, key(&second)?);
        assert_ne!(first.input_ids, second.input_ids);
        assert!(
            second
                .questions
                .iter()
                .all(|q| q.question_span[0] >= second.prefix_len)
        );
        let truncated = encoder.encode_record(
            &request,
            EncodeOptions {
                max_state_tokens: Some(3),
                ..Default::default()
            },
        )?;
        assert_ne!(key(&first)?, key(&truncated)?);
        Ok(())
    }

    #[test]
    fn prefix_identity_excludes_suffix_but_includes_tokens_positions_and_media() -> Result<()> {
        let mut r = record();
        let original = key(&r)?;
        r.input_ids[4] = 123;
        r.position_ids[4] = [99; 3];
        assert_eq!(key(&r)?, original);
        r.input_ids[0] = 123;
        assert_ne!(key(&r)?, original);
        r = record();
        r.position_ids[0][1] = 1;
        assert_ne!(key(&r)?, original);
        r = record();
        r.media.items.push(crate::media::VisualItem {
            grid: [1, 2, 2],
            patches: vec![0.; 16],
            video: false,
            timestamps: Vec::new(),
            token_ranges: vec![[1, 2]],
        });
        let image = key(&r)?;
        assert_ne!(image, original);
        r.media.items[0].patches[0] = 1.;
        assert_ne!(key(&r)?, image);
        r.media.items[0].token_ranges[0][1] = 4;
        assert!(key(&r).is_err());
        r = record();
        let mut audio = crate::omni_media::OmniMedia::default();
        audio.audio.push(crate::audio::AudioItem {
            frames: 1,
            mel: vec![0.; 128],
            token_ranges: vec![[1, 2]],
        });
        r.media.omni = Some(audio);
        let silent = key(&r)?;
        r.media.omni.as_mut().unwrap().audio[0].mel[7] = 1.;
        assert_ne!(key(&r)?, silent);
        r = record();
        r.prefix_len = 0;
        assert!(key(&r).is_err());
        r.prefix_len = 5;
        assert!(key(&r).is_err());
        Ok(())
    }
}
