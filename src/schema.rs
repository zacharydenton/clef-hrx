use anyhow::{Result, bail, ensure};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Noul,
    Choice,
    Score,
}
impl QuestionType {
    pub fn id(self) -> u32 {
        match self {
            Self::Noul => 0,
            Self::Choice => 1,
            Self::Score => 2,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
        }
    }
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub kind: QuestionType,
    #[serde(default)]
    pub instructions: Value,
    #[serde(default)]
    pub criteria: Value,
}
impl Question {
    pub fn options(&self) -> Result<Vec<(String, Value)>> {
        Ok(match self.kind {
            QuestionType::Noul => {
                ensure!(
                    self.criteria.is_null() || self.criteria.is_object(),
                    "noul criteria must be an object"
                );
                [
                    ("true", "The proposition is true or the answer is yes."),
                    ("false", "The proposition is false or the answer is no."),
                ]
                .into_iter()
                .map(|(k, v)| (k.into(), self.criteria.get(k).cloned().unwrap_or(json!(v))))
                .collect()
            }
            QuestionType::Choice => {
                let obj = self
                    .criteria
                    .as_object()
                    .ok_or_else(|| anyhow::anyhow!("choice criteria must be an object"))?;
                ensure!(!obj.is_empty(), "choice criteria must not be empty");
                let mut pairs: Vec<_> = obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
                pairs
            }
            QuestionType::Score => {
                let a = self
                    .criteria
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("score criteria must be an array"))?;
                ensure!(!a.is_empty(), "score criteria must not be empty");
                a.iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), v.clone()))
                    .collect()
            }
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Request {
    pub model: String,
    pub state: Value,
    pub questions: IndexMap<String, Question>,
    #[serde(default)]
    pub images: Vec<String>,
    #[serde(default)]
    pub videos: Vec<String>,
    #[serde(default)]
    pub media_kwargs: crate::media::MediaOptions,
}
impl Request {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.questions.is_empty(),
            "at least one question is required"
        );
        for (id, q) in &self.questions {
            ensure!(
                !id.is_empty() || !(q.instructions.is_null() || q.instructions == ""),
                "question instruction span must not be empty"
            );
            q.options().map_err(|e| anyhow::anyhow!("{id}: {e}"))?;
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub model: String,
    pub answers: IndexMap<String, Value>,
    pub usage: Usage,
}

/// Matches Python round(value, 4), using decimal formatting with ties to even.
pub fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().expect("finite decimal")
}
pub fn softmax(logits: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
        "empty or non-finite logits"
    );
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = p.iter().sum();
    for x in &mut p {
        *x /= sum;
    }
    Ok(p)
}
pub fn answer(question: &Question, probabilities: &IndexMap<String, f32>) -> Result<Value> {
    let get = |k: &str| -> Result<f64> {
        probabilities
            .get(k)
            .map(|v| *v as f64)
            .ok_or_else(|| anyhow::anyhow!("missing probability {k}"))
    };
    ensure!(
        probabilities
            .values()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
        "invalid probabilities"
    );
    match question.kind {
        QuestionType::Noul => Ok(json!({"type":"noul", "noul":round4(get("true")?)})),
        QuestionType::Choice => {
            let options = question
                .criteria
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("invalid choice"))?;
            let mut winner = None;
            let mut best = -1.0;
            let mut probs = serde_json::Map::new();
            for key in options.keys() {
                let p = get(key)?;
                if p > best {
                    winner = Some(key);
                    best = p;
                }
                probs.insert(key.clone(), json!(round4(p)));
            }
            if winner.is_none() {
                bail!("empty choice");
            }
            Ok(
                json!({"type":"choice","choice":winner,"confidence":round4(best),"probabilities":probs}),
            )
        }
        QuestionType::Score => {
            let opts = question.options()?;
            let mut score = 0.;
            let mut best: f64 = 0.;
            let mut legend = serde_json::Map::new();
            let mut probs = serde_json::Map::new();
            for (i, (key, value)) in opts.into_iter().enumerate() {
                let p = get(&key)?;
                score += i as f64 * p;
                best = best.max(p);
                legend.insert(key.clone(), value);
                probs.insert(key, json!(round4(p)));
            }
            Ok(
                json!({"type":"score","score":round4(score),"confidence":round4(best),"legend":legend,"probabilities":probs}),
            )
        }
    }
}
