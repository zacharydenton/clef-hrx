use crate::{
    Result,
    checkpoint::{Checkpoint, Source},
    encoding::{EncodeOptions, EncodedRecord, Encoder},
    gpu::{Engine, Tensor},
    schema::{Request, Response, Usage, answer, softmax},
};
use anyhow::{Context, ensure};
use hrx::{
    GraphExec, execution::RuntimeOptions, inference::ModelContext, residency::ResidencyManager,
};
use indexmap::IndexMap;
use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub source: Source,
    pub device: i32,
    pub memory_budget_bytes: usize,
    pub encoding: EncodeOptions,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            source: Source::default(),
            device: 0,
            memory_budget_bytes: 80usize << 30,
            encoding: EncodeOptions::default(),
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct QuestionPrediction {
    pub question_id: String,
    pub option_ids: Vec<String>,
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
}
#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    pub questions: Vec<QuestionPrediction>,
    pub input_tokens: usize,
    pub timings: Timings,
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct Timings {
    pub encode_ms: f64,
    pub prepare_ms: f64,
    pub inference_ms: f64,
    pub readback_ms: f64,
    pub allocated_bytes: usize,
    pub uploaded_bytes: u64,
}
struct Plan {
    key: String,
    graph: GraphExec,
    output: Tensor,
}
pub struct ClefModel {
    encoder: Encoder,
    native: hrx::execution::NativeSession<NativeState>,
    context: ModelContext,
    options: LoadOptions,
    failed: bool,
}
struct NativeState {
    plan: Option<Plan>,
    engine: Engine,
}
impl NativeState {
    fn fence(&mut self) -> hrx::Result<()> {
        self.engine.fence()
    }
}
impl ClefModel {
    /// Loads the pinned checkpoint. Local files must remain unmodified while the model is alive.
    pub fn load(options: LoadOptions) -> Result<Self> {
        options.encoding.validate()?;
        let manager = ResidencyManager::new(options.memory_budget_bytes)?;
        let context = ModelContext::new(RuntimeOptions {
            gpu_index: options.device,
            memory_budget: Some(manager.budget()),
            ..Default::default()
        })?;
        Self::load_in(&context, options)
    }
    /// Uses an existing context. Checkpoint files must remain unmodified while the model is alive.
    pub fn load_in(context: &ModelContext, options: LoadOptions) -> Result<Self> {
        options.encoding.validate()?;
        let info = options.source.inspect()?;
        let required = info.resident_weight_bytes
            + crate::checkpoint::workspace_estimate(options.encoding.max_length);
        ensure!(
            required <= options.memory_budget_bytes as u64,
            "memory budget cannot fit weights and configured workspace"
        );
        crate::memory::require(required)?;
        let mut engine = Engine::new(context)?;
        let encoder = Encoder::load(options.source.resolve("tokenizer.json")?)?;
        // Files are only mapped while loading, before this method returns.
        let checkpoint = unsafe { Checkpoint::open(&options.source)? };
        for name in checkpoint.names() {
            if crate::embedding::TABLES.contains(&name) {
                engine
                    .embedding_rows
                    .insert(name.into(), checkpoint.embedding_rows(name)?);
                continue;
            }
            let tensor = checkpoint.tensor(name)?;
            let rows = tensor.shape.first().copied().unwrap_or(1);
            let cols = tensor.shape.iter().skip(1).product::<usize>();
            engine.weight(name.to_string(), rows, cols, tensor.bytes)?;
            checkpoint.done_with(name)?;
        }
        Ok(Self {
            encoder,
            native: unsafe {
                hrx::execution::NativeSession::new(
                    context.runtime(),
                    NativeState { plan: None, engine },
                    NativeState::fence,
                )
            },
            context: context.clone(),
            options,
            failed: false,
        })
    }
    pub fn context(&self) -> &ModelContext {
        &self.context
    }
    pub fn encode_record(&self, request: &Request) -> Result<EncodedRecord> {
        self.encoder.encode_record(request, self.options.encoding)
    }
    pub fn infer(&mut self, request: &Request) -> Result<Prediction> {
        ensure!(
            !self.failed,
            "model execution failed; reload model before reusing it"
        );
        let start = Instant::now();
        let record = self.encode_record(request)?;
        let encode_ms = start.elapsed().as_secs_f64() * 1000.;
        self.infer_encoded(record).map(|mut p| {
            p.timings.encode_ms = encode_ms;
            p
        })
    }
    /// Serialized diagnostic GPU timings, including profiling barriers.
    /// Rebuilds the graph and invalidates the ordinary exact-input cache.
    #[cfg(feature = "bench-internals")]
    pub fn profile(&mut self, request: &Request) -> Result<hrx::fabric::DeviceProfile> {
        ensure!(
            !self.failed,
            "model execution failed; reload model before reusing it"
        );
        let record = self.encode_record(request)?;
        ensure!(
            record.input_ids.iter().all(|id| *id < 248320),
            "token ID outside checkpoint vocabulary"
        );
        // Safety: all device resources stay in NativeState, as in infer_encoded.
        let result = unsafe {
            self.native.run(|state| -> Result<_> {
                let output = state.build(&record)?;
                let graph = state.engine.finish_profiled()?;
                // A blank key cannot match the SHA256 key used by infer_encoded.
                state.plan = Some(Plan {
                    key: String::new(),
                    graph,
                    output,
                });
                let plan = state.plan.as_mut().context("missing profiling plan")?;
                state.engine.run(&mut plan.graph)?;
                state.engine.profile(&mut plan.graph)
            })
        }
        .map_err(anyhow::Error::from)
        .and_then(|result| result);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    pub fn infer_with_media(
        &mut self,
        request: &Request,
        media: crate::media::PreparedMedia,
    ) -> Result<Prediction> {
        ensure!(
            !self.failed,
            "model execution failed; reload model before reusing it"
        );
        let start = Instant::now();
        let record = self
            .encoder
            .encode_with_media(request, self.options.encoding, media)?;
        let ms = start.elapsed().as_secs_f64() * 1000.;
        self.infer_encoded(record).map(|mut p| {
            p.timings.encode_ms = ms;
            p
        })
    }
    pub fn infer_batch(&mut self, requests: &[Request]) -> Result<Vec<Prediction>> {
        requests.iter().map(|r| self.infer(r)).collect()
    }
    fn infer_encoded(&mut self, record: EncodedRecord) -> Result<Prediction> {
        ensure!(
            record.input_ids.iter().all(|id| *id < 248320),
            "token ID outside checkpoint vocabulary"
        );
        // Safety: NativeState owns every stream, graph and allocation. Stages
        // return owned host results only; the fence drains even on errors.
        let result = unsafe { self.native.run(|state| state.execute(&record)) }
            .map_err(anyhow::Error::from)
            .and_then(|result| result);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    pub fn systemone(&mut self, request: &Request) -> Result<Response> {
        let prediction = self.infer(request)?;
        Self::response(request, &prediction)
    }
    pub fn response(request: &Request, prediction: &Prediction) -> Result<Response> {
        request.validate()?;
        ensure!(
            prediction.questions.len() == request.questions.len(),
            "prediction question count differs from request"
        );
        let mut answers = IndexMap::new();
        for q in &prediction.questions {
            ensure!(
                !answers.contains_key(&q.question_id),
                "duplicate prediction question {}",
                q.question_id
            );
            let question = request
                .questions
                .get(&q.question_id)
                .context("prediction question is absent from request")?;
            let options = question.options()?;
            ensure!(
                q.option_ids.len() == q.probabilities.len() && q.option_ids.len() == options.len(),
                "prediction option count differs from request"
            );
            let probabilities: IndexMap<_, _> = q
                .option_ids
                .iter()
                .cloned()
                .zip(q.probabilities.iter().copied())
                .collect();
            ensure!(
                probabilities.len() == options.len()
                    && options.iter().all(|(id, _)| probabilities.contains_key(id)),
                "prediction option IDs differ from request"
            );
            answers.insert(q.question_id.clone(), answer(question, &probabilities)?);
        }
        Ok(Response {
            model: request.model.clone(),
            answers,
            usage: Usage {
                input_tokens: prediction.input_tokens,
                output_tokens: 0,
            },
        })
    }
}

impl NativeState {
    fn build(&mut self, record: &EncodedRecord) -> Result<Tensor> {
        self.plan = None;
        self.engine.reset_plan()?;
        let x = self.engine.embedding(
            "model.language_model.embed_tokens.weight",
            &record.input_ids,
        )?;
        for item in &record.media.items {
            let features = crate::vision::forward(&mut self.engine, item)?;
            let mut offset = 0;
            for [start, end] in &item.token_ranges {
                let len = end - start;
                self.engine.copy_into(
                    &features.slice_rows(offset, len),
                    &x.slice_rows(*start, len),
                )?;
                offset += len;
            }
        }
        let hidden = crate::backbone::forward(&mut self.engine, x, &record.position_ids)?;
        let output = crate::head::forward(&mut self.engine, &hidden, record)?;
        Ok(output)
    }
    fn execute(&mut self, record: &EncodedRecord) -> Result<Prediction> {
        let start = Instant::now();
        let uploaded = self.engine.uploaded;
        // Include media bytes because EncodedRecord skips them during serialization.
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(serde_json::to_vec(record)?);
        for item in &record.media.items {
            hash.update(bytemuck::cast_slice(&item.patches));
            hash.update(serde_json::to_vec(&item.grid)?);
        }
        let key = format!("{:x}", hash.finalize());
        if self.plan.as_ref().is_none_or(|p| p.key != key) {
            let output = self.build(record)?;
            let graph = self.engine.finish()?;
            self.plan = Some(Plan { key, graph, output });
        }
        let prepare_ms = start.elapsed().as_secs_f64() * 1000.;
        let plan = self.plan.as_mut().context("missing execution plan")?;
        let start = Instant::now();
        self.engine.run(&mut plan.graph)?;
        let inference_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let logits = self.engine.read_f32(&plan.output)?;
        self.engine.dump_traces()?;
        let readback_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut offset = 0;
        let mut questions = Vec::new();
        for question in &record.questions {
            let n = question.option_ids.len();
            let l = logits[offset..offset + n].to_vec();
            offset += n;
            questions.push(QuestionPrediction {
                question_id: question.question_id.clone(),
                option_ids: question.option_ids.clone(),
                probabilities: softmax(&l)?,
                logits: l,
            });
        }
        Ok(Prediction {
            questions,
            input_tokens: record.input_ids.len(),
            timings: Timings {
                encode_ms: 0.,
                prepare_ms,
                inference_ms,
                readback_ms,
                allocated_bytes: self.engine.allocated_bytes(),
                uploaded_bytes: self.engine.uploaded - uploaded,
            },
        })
    }
}
