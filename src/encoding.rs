use crate::{
    Result,
    media::PreparedMedia,
    schema::{QuestionType, Request},
};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

pub const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";
#[derive(Debug, Clone, Copy)]
pub struct EncodeOptions {
    pub max_length: usize,
    pub max_state_tokens: Option<usize>,
}
impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            max_length: 16384,
            max_state_tokens: None,
        }
    }
}
impl EncodeOptions {
    pub(crate) fn validate(self) -> Result<()> {
        ensure!(
            (1..=16384).contains(&self.max_length),
            "max_length must be 1..=16384"
        );
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedQuestion {
    pub question_id: String,
    pub question_type: QuestionType,
    pub question_span: [usize; 2],
    pub option_spans: Vec<[usize; 2]>,
    pub option_ids: Vec<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct EncodedRecord {
    pub input_ids: Vec<u32>,
    pub questions: Vec<EncodedQuestion>,
    pub position_ids: Vec<[u32; 3]>,
    #[serde(skip)]
    pub media: PreparedMedia,
}
pub struct Encoder {
    tokenizer: tokenizers::Tokenizer,
}
impl Encoder {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        tokenizer.with_padding(None);
        Ok(Self { tokenizer })
    }
    pub fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenization: {e}"))?
            .get_ids()
            .to_vec())
    }
    pub fn encode_record(
        &self,
        request: &Request,
        options: EncodeOptions,
    ) -> Result<EncodedRecord> {
        options.validate()?;
        request.validate()?;
        let media = crate::media::prepare_paths(request, options.max_length)?;
        self.encode_with_media(request, options, media)
    }
    pub fn encode_with_media(
        &self,
        request: &Request,
        options: EncodeOptions,
        mut media: PreparedMedia,
    ) -> Result<EncodedRecord> {
        options.validate()?;
        request.validate()?;
        let mut visual_tokens = 0usize;
        for item in &media.items {
            let [t, h, w] = item.grid;
            ensure!(
                t > 0 && h > 0 && w > 0 && h.is_multiple_of(2) && w.is_multiple_of(2),
                "invalid media grid"
            );
            let patches = t
                .checked_mul(h)
                .and_then(|n| n.checked_mul(w))
                .context("media grid overflow")?;
            ensure!(
                patches <= options.max_length.saturating_mul(4),
                "media exceeds token budget"
            );
            ensure!(
                item.patches.len() == patches * 1536 && item.patches.iter().all(|x| x.is_finite()),
                "invalid media patch tensor"
            );
            ensure!(
                (item.video
                    && item.timestamps.len() == t
                    && item.timestamps.iter().all(|x| x.is_finite() && *x >= 0.))
                    || (!item.video && t == 1),
                "invalid media timestamps"
            );
            visual_tokens = visual_tokens
                .checked_add(patches / 4)
                .context("media token count overflow")?;
        }
        ensure!(
            visual_tokens <= options.max_length,
            "media exceeds total token budget"
        );
        let mut schema = self.tokens("\n\nSCHEMA FIELDS:\n")?;
        let mut questions = Vec::new();
        for (i, (id, q)) in request.questions.iter().enumerate() {
            schema.extend(self.tokens(&format!(
                "\nFIELD {}\nID: {}\nTYPE: {}\nINSTRUCTION: ",
                i + 1,
                id,
                q.kind.name()
            ))?);
            let start = schema.len();
            let instruction = if q.instructions.is_null() || q.instructions == "" {
                id.clone()
            } else {
                render(&q.instructions)
            };
            schema.extend(self.tokens(&instruction)?);
            let end = schema.len();
            ensure!(end > start, "{id}: instruction span is empty");
            schema.extend(self.tokens("\nALLOWED OPTIONS:\n")?);
            let mut spans = Vec::new();
            let mut ids = Vec::new();
            for (j, (key, description)) in q.options()?.into_iter().enumerate() {
                schema.extend(self.tokens(&format!("OPTION {}: ", j + 1))?);
                let start = schema.len();
                let mut semantics = serde_json::Map::new();
                semantics.insert("option_id".into(), Value::String(key.clone()));
                if !description.is_null() {
                    semantics.insert("description".into(), description);
                }
                schema.extend(self.tokens(&render(&Value::Object(semantics)))?);
                spans.push([start, schema.len()]);
                ids.push(key);
                schema.extend(self.tokens("\n")?);
            }
            schema.extend(self.tokens("END FIELD\n")?);
            questions.push(EncodedQuestion {
                question_id: id.clone(),
                question_type: q.kind,
                question_span: [start, end],
                option_spans: spans,
                option_ids: ids,
            });
        }
        let mut prefix = self.tokens(&format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
        ))?;
        let media_offset = prefix.len();
        if !media.items.is_empty() {
            let media_tokens = self.tokens(&media.prompt)?;
            media.bind_tokens(&media_tokens, media_offset)?;
            prefix.extend(media_tokens);
        }
        let suffix = self.tokens(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
        )?;
        let fixed = prefix.len() + schema.len() + suffix.len();
        ensure!(
            fixed <= options.max_length,
            "schema and media require {fixed} tokens before state; maximum is {}",
            options.max_length
        );
        let mut state = self.tokens(&render(&request.state))?;
        state.truncate(
            options
                .max_state_tokens
                .unwrap_or(usize::MAX)
                .min(options.max_length - fixed),
        );
        let offset = prefix.len() + state.len();
        for q in &mut questions {
            q.question_span.iter_mut().for_each(|i| *i += offset);
            q.option_spans
                .iter_mut()
                .flatten()
                .for_each(|i| *i += offset);
        }
        prefix.extend(state);
        prefix.extend(schema);
        prefix.extend(suffix);
        let position_ids = media
            .position_ids(prefix.len())
            .context("multimodal positions")?;
        Ok(EncodedRecord {
            input_ids: prefix,
            questions,
            position_ids,
            media,
        })
    }
}

/// Python json.dumps(ensure_ascii=False, sort_keys=True, separators=(',', ':')).
/// Strings supplied as the whole state/instruction are rendered verbatim.
pub fn render(value: &Value) -> String {
    if let Value::String(s) = value {
        s.clone()
    } else {
        canonical_json(value)
    }
}
pub fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            format!(
                "{{{}}}",
                keys.into_iter()
                    .map(|k| format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap(),
                        canonical_json(&map[k])
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(canonical_json).collect::<Vec<_>>().join(",")
        ),
        Value::Number(n) if n.is_f64() => {
            let x = n.as_f64().unwrap();
            if x != 0.0 && (x.abs() < 1e-4 || x.abs() >= 1e16) {
                let s = format!("{x:e}");
                let (mantissa, exponent) = s.split_once('e').unwrap();
                let e: i32 = exponent.parse().unwrap();
                format!("{mantissa}e{}{:02}", if e < 0 { "-" } else { "+" }, e.abs())
            } else {
                let s = x.to_string();
                if s.contains('.') { s } else { format!("{s}.0") }
            }
        }
        _ => serde_json::to_string(value).unwrap(),
    }
}
