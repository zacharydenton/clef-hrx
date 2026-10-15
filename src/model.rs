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

/// Projection precision. INT4 trades some accuracy for lower latency and memory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Precision {
    #[default]
    Bf16,
    /// Quantize the language-model MLP projections only.
    Int4Mlp,
    /// Quantize MLP and other eligible large language-model projections.
    Int4,
}

#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub source: Source,
    pub device: i32,
    pub memory_budget_bytes: usize,
    pub encoding: EncodeOptions,
    pub precision: Precision,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            source: Source::default(),
            device: 0,
            memory_budget_bytes: 80usize << 30,
            encoding: EncodeOptions::default(),
            precision: Precision::default(),
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
#[cfg(feature = "bench-internals")]
struct BackbonePlan {
    positions: Vec<[u32; 3]>,
    graph: GraphExec,
    input: Tensor,
    hidden: Tensor,
    _pins: Vec<std::rc::Rc<()>>,
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
    #[cfg(feature = "bench-internals")]
    backbone: Option<BackbonePlan>,
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
        // Context capacity is not an allocation request. HRX enforces the budget
        // and Engine checks available RAM as actual request buffers are allocated.
        let required = crate::checkpoint::resident_weight_bytes(info.model, options.precision)?
            + options.source.model.architecture().workspace_estimate(0);
        ensure!(
            required <= options.memory_budget_bytes as u64,
            "memory budget cannot fit weights and minimum workspace"
        );
        crate::memory::require(required)?;
        let mut engine = Engine::new(context)?;
        engine.architecture = options.source.model.architecture();
        if options.precision != Precision::Bf16 {
            engine.quantized =
                crate::quantized::QuantizedProjections::for_precision(options.precision);
        }
        let encoder = Encoder::load_for_model(
            options.source.resolve("tokenizer.json")?,
            options.source.model,
        )?;
        // Files are only mapped while loading, before this method returns.
        let checkpoint = unsafe { Checkpoint::open(&options.source)? };
        if engine.architecture.is_omni() {
            crate::omni::load_experts(&mut engine, &checkpoint)?;
        }
        for name in checkpoint.names() {
            if engine.architecture.is_omni() && name.contains(".mlp.experts.") {
                continue;
            }
            if crate::embedding::TABLES.contains(&name) {
                engine
                    .embedding_rows
                    .insert(name.into(), checkpoint.embedding_rows(name)?);
                continue;
            }
            let tensor = checkpoint.tensor(name)?;
            let rows = tensor.shape.first().copied().unwrap_or(1);
            let cols = tensor.shape.iter().skip(1).product::<usize>();
            if engine.load_quantized_weight(name, rows, cols, tensor.bytes)? {
                checkpoint.done_with(name)?;
                continue;
            }
            engine.weight(name.to_string(), rows, cols, tensor.bytes)?;
            checkpoint.done_with(name)?;
        }
        engine.quantized.validate_loaded()?;
        #[cfg(feature = "bench-internals")]
        if let Some(cache) = &engine.packed_weight_cache {
            eprintln!(
                "Packed weight cache: {} hits, {} misses",
                cache.hits, cache.misses
            );
        }
        engine.bound_scratch_cache(options.memory_budget_bytes)?;
        Ok(Self {
            encoder,
            native: unsafe {
                hrx::execution::NativeSession::new(
                    context.runtime(),
                    NativeState {
                        plan: None,
                        #[cfg(feature = "bench-internals")]
                        backbone: None,
                        engine,
                    },
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
            record
                .input_ids
                .iter()
                .all(|id| (*id as usize) < self.options.source.model.architecture().vocabulary()),
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
    /// Drops the recorded graph so benchmarks can time preparation separately.
    #[cfg(feature = "bench-internals")]
    pub fn invalidate_benchmark_graph(&mut self) -> Result<()> {
        ensure!(
            !self.failed,
            "model execution failed; reload before reusing it"
        );
        // Safety: the closure only resets session-owned graph state.
        let result = unsafe {
            self.native.run(|state| -> Result<()> {
                state.plan = None;
                state.backbone = None;
                state.engine.reset_plan()?;
                Ok(())
            })
        }
        .map_err(anyhow::Error::from)
        .and_then(|result| result);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    /// Selects reference kernels for paired benchmarks, invalidating the graph.
    #[cfg(feature = "bench-internals")]
    pub fn set_reference_kernels(
        &mut self,
        kernels: crate::benchmarking::ReferenceKernels,
    ) -> Result<()> {
        ensure!(
            !self.failed,
            "model execution failed; reload model before reusing it"
        );
        // Safety: only owned host configuration crosses the native session.
        let result = unsafe {
            self.native.run(|state| -> Result<()> {
                if state.engine.reference_kernels != kernels {
                    state.plan = None;
                    state.backbone = None;
                    state.engine.reset_plan()?;
                    state.engine.reference_kernels = kernels;
                }
                Ok(())
            })
        }
        .map_err(anyhow::Error::from)
        .and_then(|result| result);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    /// Switches an attention source for paired diagnostics with one resident model.
    /// Rebuilds the graph; weights and unrelated kernel selections stay fixed.
    #[cfg(feature = "bench-internals")]
    pub fn set_benchmark_attention_source(&mut self, source: Option<&str>) -> Result<()> {
        ensure!(
            !self.failed,
            "model execution failed; reload model before reusing it"
        );
        let source: Option<std::sync::Arc<str>> = source
            .map(|source| format!("{}\n{source}", include_str!("../kernels/common.loom")).into());
        // Safety: only owned source text crosses the native session. Resetting
        // the plan follows the same lifetime rules as set_reference_kernels.
        let result = unsafe {
            self.native.run(|state| -> Result<()> {
                state.plan = None;
                state.backbone = None;
                state.engine.reset_plan()?;
                state.engine.experimental_attention = source;
                Ok(())
            })
        }
        .map_err(anyhow::Error::from)
        .and_then(|result| result);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    /// Switches Omni benchmark variants while retaining the checkpoint allocation.
    #[cfg(feature = "bench-internals")]
    pub fn set_benchmark_omni_kernels(
        &mut self,
        config: crate::benchmarking::OmniKernelConfig,
    ) -> Result<()> {
        ensure!(
            !self.failed,
            "model execution failed; reload model before reusing it"
        );
        ensure!(
            self.options.source.model == crate::ModelVariant::ClefOmni,
            "Omni model required"
        );
        ensure!(
            (1..=crate::omni::MAX_EXPERT_CHUNK).contains(&config.chunk),
            "Omni expert chunk limit"
        );
        ensure!(
            config.alignment.is_power_of_two() && config.alignment <= 256,
            "Omni padding alignment"
        );
        ensure!(
            config.minimum_tokens <= crate::omni::MAX_EXPERT_CHUNK,
            "Omni kernel minimum token count"
        );
        // Safety: only owned host configuration crosses the session, after the
        // previous graph is dropped and all temporary allocations are reset.
        let result = unsafe {
            self.native.run(|state| -> Result<()> {
                state.plan = None;
                state.backbone = None;
                state.engine.reset_plan()?;
                state.engine.omni_kernel_dir = config.directory;
                state.engine.omni_chunk = config.chunk;
                state.engine.omni_alignment = config.alignment;
                state.engine.omni_kernel_min_tokens = config.minimum_tokens;
                state.engine.omni_sorted_combine = config.sorted_combine;
                state.engine.omni_tiled_attention = config.tiled_attention;
                state.engine.omni_prepack_qk = config.prepack_qk;
                state.engine.omni_fused_mlp = config.fused_mlp;
                state.engine.omni_pack_swiglu = config.pack_swiglu;
                state.engine.omni_compact_experts = config.compact_experts;
                state.engine.omni_reuse_backbone = config.reuse_backbone;
                Ok(())
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
            record
                .input_ids
                .iter()
                .all(|id| (*id as usize) < self.options.source.model.architecture().vocabulary()),
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

pub(crate) type Deepstack = Vec<Vec<(usize, Tensor)>>;
pub(crate) fn embed_record(
    engine: &mut Engine,
    record: &EncodedRecord,
) -> Result<(Tensor, Deepstack)> {
    let x = engine.embedding(
        "model.language_model.embed_tokens.weight",
        &record.input_ids,
    )?;
    let mut deepstack: Vec<Vec<(usize, Tensor)>> = vec![Vec::new(); 3];
    for item in &record.media.items {
        let (features, intermediate) = crate::vision::forward_all(engine, item)?;
        let mut offset = 0;
        for [start, end] in &item.token_ranges {
            let len = end - start;
            engine.copy_into(
                &features.slice_rows(offset, len),
                &x.slice_rows(*start, len),
            )?;
            for (layer, features) in intermediate.iter().enumerate() {
                deepstack[layer].push((*start, features.slice_rows(offset, len)));
            }
            offset += len;
        }
    }
    if let Some(omni) = &record.media.omni {
        let chunk_width = omni
            .audio
            .iter()
            .map(|a| a.frames.min(100))
            .max()
            .unwrap_or(0);
        for audio in &omni.audio {
            let features = crate::audio::forward(engine, audio, chunk_width)?;
            let mut offset = 0;
            for &[start, end] in &audio.token_ranges {
                engine.copy_into(
                    &features.slice_rows(offset, end - start),
                    &x.slice_rows(start, end - start),
                )?;
                offset += end - start;
            }
            ensure!(offset == features.rows, "audio feature/token mismatch");
        }
    }
    Ok((x, deepstack))
}

impl NativeState {
    fn build(&mut self, record: &EncodedRecord) -> Result<Tensor> {
        self.plan = None;
        #[cfg(feature = "bench-internals")]
        {
            self.backbone = None;
        }
        self.engine.reset_plan()?;
        let (x, deepstack) = embed_record(&mut self.engine, record)?;
        let hidden = if self.engine.architecture.is_omni() {
            crate::omni::forward(&mut self.engine, x, &record.position_ids, &deepstack)?
        } else {
            crate::backbone::forward(&mut self.engine, x, &record.position_ids)?
        };
        let output = crate::head::forward(&mut self.engine, &hidden, record)?;
        Ok(output)
    }
    #[cfg(feature = "bench-internals")]
    fn build_cached_text(&mut self, record: &EncodedRecord) -> Result<Tensor> {
        let rows = record.input_ids.len();
        ensure!(
            (1..=crate::encoding::MAX_CONTEXT_TOKENS).contains(&rows)
                && self.engine.omni_alignment.is_power_of_two()
                && self.engine.omni_alignment <= 256,
            "Omni cached backbone extent/alignment"
        );
        ensure!(record.position_ids.len() == rows, "Omni position count");
        let padded = rows.div_ceil(self.engine.omni_alignment) * self.engine.omni_alignment;
        ensure!(
            padded <= crate::encoding::MAX_CONTEXT_TOKENS,
            "Omni padded context limit"
        );
        let positions = crate::omni::padded_positions(&record.position_ids, padded)?;
        self.plan = None;
        if self
            .backbone
            .as_ref()
            .is_some_and(|p| p.positions != positions)
        {
            self.backbone = None;
        }
        self.engine.reset_plan()?;
        let table = self
            .engine
            .embedding_rows
            .get("model.language_model.embed_tokens.weight")
            .context("missing Omni embedding table")?;
        let width = self.engine.architecture.hidden;
        let mut bytes = table.read(&record.input_ids)?;
        ensure!(bytes.len() == rows * width * 2, "Omni embedding width");
        bytes.resize(padded * width * 2, 0);
        if let Some(backbone) = &self.backbone {
            self.engine.upload(&backbone.input, &bytes)?;
        } else {
            let input = self
                .engine
                .input(padded, width, crate::gpu::DType::Bf16, &bytes)?;
            let hidden = crate::omni::forward(&mut self.engine, input.clone(), &positions, &[])?;
            let pins = self.engine.pin_plan_buffers();
            let graph = self.engine.finish()?;
            self.backbone = Some(BackbonePlan {
                positions,
                graph,
                input,
                hidden,
                _pins: pins,
            });
            // Pins keep backbone inputs/scratch alive while the head is built.
            self.engine.reset_plan()?;
        }
        let hidden = self
            .backbone
            .as_ref()
            .context("missing backbone plan")?
            .hidden
            .slice_rows(0, rows);
        crate::head::forward(&mut self.engine, &hidden, record)
    }
    fn execute(&mut self, record: &EncodedRecord) -> Result<Prediction> {
        let start = Instant::now();
        let uploaded = self.engine.uploaded;
        // Include media bytes because EncodedRecord skips them during serialization.
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        for dimension in [
            self.engine.architecture.layers,
            self.engine.architecture.hidden,
            self.engine.architecture.intermediate,
            self.engine.architecture.attention_heads,
            self.engine.architecture.delta_heads,
        ] {
            hash.update(dimension.to_le_bytes());
        }
        hash.update(serde_json::to_vec(record)?);
        if let Some(omni) = &record.media.omni {
            for audio in &omni.audio {
                hash.update(audio.frames.to_le_bytes());
                hash.update(bytemuck::cast_slice(&audio.mel));
                hash.update(serde_json::to_vec(&audio.token_ranges)?);
            }
        }

        for item in &record.media.items {
            hash.update(bytemuck::cast_slice(&item.patches));
            hash.update(serde_json::to_vec(&item.grid)?);
        }
        let key = format!("{:x}", hash.finalize());
        if self.plan.as_ref().is_none_or(|p| p.key != key) {
            #[cfg(feature = "bench-internals")]
            let cached = self.engine.omni_reuse_backbone
                && self.engine.architecture.is_omni()
                && record.media.items.is_empty()
                && record
                    .media
                    .omni
                    .as_ref()
                    .is_none_or(|m| m.audio.is_empty())
                && std::env::var_os("CLEF_TRACE_DIR").is_none();
            #[cfg(feature = "bench-internals")]
            let output = if cached {
                self.build_cached_text(record)?
            } else {
                self.build(record)?
            };
            #[cfg(not(feature = "bench-internals"))]
            let output = self.build(record)?;
            let graph = self.engine.finish()?;
            self.plan = Some(Plan { key, graph, output });
        }
        let prepare_ms = start.elapsed().as_secs_f64() * 1000.;
        let plan = self.plan.as_mut().context("missing execution plan")?;
        let start = Instant::now();
        #[cfg(feature = "bench-internals")]
        if let Some(backbone) = &mut self.backbone {
            self.engine.run(&mut backbone.graph)?;
        }
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
