//! Development-only graph harness for Criterion; no checkpoint allocation.
use crate::{
    Result,
    gpu::{DType, Engine, Tensor},
};
use hrx::{GraphExec, inference::ModelContext};
mod activation;
pub(crate) mod quantized;
pub use activation::SwiGluBenchmark;
pub use quantized::PackingBenchmark;

/// CPU-only cache-hit work with the same source/configuration in both maps.
pub struct KernelCacheKeyBenchmark {
    source: String,
    spec: hrx::loom::Specialization,
    fingerprints: crate::kernel_cache::SourceFingerprints,
    legacy: std::collections::HashMap<String, usize>,
    compact: std::collections::HashMap<crate::kernel_cache::KernelKey, usize>,
}
impl KernelCacheKeyBenchmark {
    pub fn new(source: &str) -> Self {
        let source = source.to_owned();
        let mut spec = hrx::loom::Specialization::new("clef_op");
        for (name, value) in [
            ("work", 620 * 5120),
            ("groups", 620 * 5120 / 64),
            ("out_len", 620 * 5120),
            ("out_dtype", 1),
            ("a_len", 620 * 5120),
            ("a_dtype", 1),
            ("b_len", 620 * 5120),
            ("b_dtype", 1),
        ] {
            spec.set_config(format!("clef.{name}"), value.to_string());
        }
        let mut fingerprints = crate::kernel_cache::SourceFingerprints::default();
        let compact_key = fingerprints.key(&source, &spec);
        let legacy_key = format!("{}{:?}", source, spec.configuration());
        Self {
            source,
            spec,
            fingerprints,
            legacy: [(legacy_key, 7)].into(),
            compact: [(compact_key, 7)].into(),
        }
    }
    pub fn legacy(&self) -> (bool, Option<usize>) {
        let key = format!("{}{:?}", self.source, self.spec.configuration());
        (
            self.legacy.contains_key(std::hint::black_box(&key)),
            self.legacy.get(std::hint::black_box(&key)).copied(),
        )
    }
    pub fn fingerprinted(&mut self) -> (bool, Option<usize>) {
        let key = self.fingerprints.key(&self.source, &self.spec);
        (
            self.compact.contains_key(std::hint::black_box(&key)),
            self.compact.get(std::hint::black_box(&key)).copied(),
        )
    }
}

/// Development-only switches for paired full-model comparisons.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReferenceKernels {
    pub gemm: bool,
    pub norm: bool,
    pub delta: bool,
    pub attention: bool,
}

fn gemm_value(i: usize, bits: usize) -> f32 {
    let value = ((i * 17 % 251) as f32 - 125.) / 128.;
    if bits == 4 {
        (value * 8.).round().clamp(-8., 7.) / 8.
    } else {
        value
    }
}

pub struct GemmBenchmark {
    graphs: Vec<GraphExec>,
    next_graph: usize,
    engine: Engine,
    output: Tensor,
    _context: ModelContext,
}
#[derive(Clone)]
pub enum GemmKernel {
    Reference,
    Wide,
    Pipelined,
    Auto,
    Source(String),
    Padded,
}
impl GemmBenchmark {
    pub fn allocated_bytes(&self) -> usize {
        self.engine.allocated_bytes()
    }

    /// Experimental packed integer dispatch; uniform scales are exactly representable.
    /// Paired comparisons can vary the two operands' row scales independently.
    /// Packing is outside timing; full-model qualification must include its cost.
    pub fn packed_integer(
        m: usize,
        k: usize,
        n: usize,
        source: &str,
        bits: usize,
        padded: bool,
        minimum_weight_bytes: usize,
    ) -> Result<Self> {
        Self::packed_integer_impl((m, k, n), source, bits, padded, minimum_weight_bytes, None)
    }
    /// Include a separate or fused BF16 residual addition in every replay.
    pub fn packed_integer_residual(
        m: usize,
        k: usize,
        n: usize,
        source: &str,
        padded: bool,
        minimum_weight_bytes: usize,
        fused: bool,
    ) -> Result<Self> {
        Self::packed_integer_impl(
            (m, k, n),
            source,
            4,
            padded,
            minimum_weight_bytes,
            Some(fused),
        )
    }
    fn packed_integer_impl(
        (m, k, n): (usize, usize, usize),
        source: &str,
        bits: usize,
        padded: bool,
        minimum_weight_bytes: usize,
        residual: Option<bool>,
    ) -> Result<Self> {
        anyhow::ensure!(
            [4, 8].contains(&bits)
                && (bits != 4 || k.is_multiple_of(128))
                && (1..=4096).contains(&m)
                && (64..=17408).contains(&k)
                && k.is_multiple_of(64)
                && (128..=17408).contains(&n)
                && n.is_multiple_of(128)
                && minimum_weight_bytes <= 512 << 20,
            "unsupported packed integer benchmark shape"
        );
        let context = ModelContext::new(Default::default())?;
        let mut engine = Engine::new(&context)?;
        let stride = k + if padded && k.is_multiple_of(1024) {
            512 / bits
        } else {
            0
        };
        let per_word = 32 / bits;
        let row_bytes = stride * bits / 8;
        let scale = 1.0 / (1usize << (bits - 1)) as f32;
        let varying_scales = std::env::var("CLEF_BENCH_INTEGER_ROW_SCALES").as_deref() == Ok("1");
        let scales = |rows, factor, zero_period| {
            (0..rows)
                .map(|row| {
                    if !varying_scales {
                        scale
                    } else if row % zero_period == 0 {
                        0.0
                    } else {
                        ((row * factor % 31 + 1) as f32 / 257.0)
                            * 2.0f32.powi(-((row * factor % 5) as i32 * 4))
                    }
                })
                .collect::<Vec<_>>()
        };
        let values = |rows| {
            let mut bytes = vec![0u8; rows * row_bytes];
            let mask = ((1u16 << bits) - 1) as u8;
            for row in 0..rows {
                for col in 0..k {
                    let value = (gemm_value(row * k + col, bits) / scale) as i8 as u8;
                    bytes[row * row_bytes + col * bits / 8] |=
                        (value & mask) << ((col % (8 / bits)) * bits);
                }
            }
            bytes
        };
        let x = engine.input(m, stride / per_word, DType::U32, &values(m))?;
        let xs = engine.input(1, m, DType::F32, bytemuck::cast_slice(&scales(m, 3, 17)))?;
        let ws = engine.input(1, n, DType::F32, bytemuck::cast_slice(&scales(n, 7, 23)))?;
        let residual_input = if residual.is_some() {
            let values: Vec<_> = (0..m * n)
                .map(|i| half::bf16::from_f32(((i * 43 % 997) as f32 - 498.) / 257.))
                .collect();
            Some(engine.input(m, n, DType::Bf16, bytemuck::cast_slice(&values))?)
        } else {
            None
        };
        let copies = minimum_weight_bytes.div_ceil(n * row_bytes).max(1);
        let mut graphs = Vec::with_capacity(copies);
        let mut first_output = None;
        for _ in 0..copies {
            let w = engine.input(n, stride / per_word, DType::U32, &values(n))?;
            let output = engine.alloc(m, n, DType::Bf16)?;
            let symbol = if residual == Some(true) {
                std::env::var("CLEF_BENCH_INTEGER_RESIDUAL_EXPORT")
                    .unwrap_or_else(|_| format!("h3_gemm_i{bits}_256"))
            } else {
                format!("h3_gemm_i{bits}_256")
            };
            let mut spec = hrx::loom::Specialization::new(symbol);
            for (key, value) in [
                ("k_size", k),
                ("n_size", n),
                ("k_stride", stride),
                ("m_group", m.div_ceil(256).min(4)),
            ] {
                spec.set_config(format!("h3.gemm_i{bits}_256.{key}"), value.to_string());
            }
            let mut constants = hrx::Constants::new();
            constants.push(m as u32)?;
            let mut inputs = vec![&x, &w, &ws, &xs];
            if residual == Some(true) {
                inputs.push(residual_input.as_ref().unwrap());
            }
            engine.emit_special(source, spec, constants, &[m as u64], &inputs, &output)?;
            let output = if residual == Some(false) {
                // Retain intermediates across all rotating graphs and replays.
                engine.retained.push(output.clone());
                engine.binary(residual_input.as_ref().unwrap(), &output, "addf")?
            } else {
                output
            };
            let mut graph = engine.finish()?;
            engine.run(&mut graph)?;
            graphs.push(graph);
            first_output.get_or_insert(output);
        }
        Ok(Self {
            graphs,
            next_graph: 0,
            engine,
            output: first_output.unwrap(),
            _context: context,
        })
    }
    pub fn new(m: usize, k: usize, n: usize, optimized: bool) -> Result<Self> {
        Self::with_kernel(
            m,
            k,
            n,
            if optimized {
                GemmKernel::Auto
            } else {
                GemmKernel::Reference
            },
        )
    }
    pub fn with_kernel(m: usize, k: usize, n: usize, kernel: GemmKernel) -> Result<Self> {
        Self::with_working_set(m, k, n, kernel, 0)
    }
    pub fn with_working_set(
        m: usize,
        k: usize,
        n: usize,
        kernel: GemmKernel,
        minimum_weight_bytes: usize,
    ) -> Result<Self> {
        Self::build(m, k, n, kernel, minimum_weight_bytes, 16)
    }
    pub fn quantized_reference(m: usize, k: usize, n: usize, bits: usize) -> Result<Self> {
        anyhow::ensure!([4, 8].contains(&bits), "unsupported integer precision");
        Self::build(m, k, n, GemmKernel::Auto, 0, bits)
    }
    fn build(
        m: usize,
        k: usize,
        n: usize,
        kernel: GemmKernel,
        minimum_weight_bytes: usize,
        bits: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            (1..=4096).contains(&m) && (1..=17408).contains(&k) && (1..=17408).contains(&n),
            "unsupported GEMM benchmark shape"
        );
        let context = ModelContext::new(Default::default())?;
        let mut engine = Engine::new(&context)?;
        let values = |len| {
            (0..len)
                .map(|i| half::bf16::from_f32(gemm_value(i, bits)))
                .collect::<Vec<_>>()
        };
        let stride = k + if matches!(kernel, GemmKernel::Padded) && k.is_multiple_of(1024) {
            64
        } else {
            0
        };
        let operands = |rows| {
            let dense = values(rows * k);
            if stride == k {
                return dense;
            }
            let mut padded = vec![half::bf16::ZERO; rows * stride];
            for r in 0..rows {
                padded[r * stride..r * stride + k].copy_from_slice(&dense[r * k..(r + 1) * k]);
            }
            padded
        };
        let x = engine.input(m, stride, DType::Bf16, bytemuck::cast_slice(&operands(m)))?;
        anyhow::ensure!(
            minimum_weight_bytes <= 512 << 20,
            "GEMM working set exceeds 512 MiB"
        );
        let copies = minimum_weight_bytes.div_ceil(n * stride * 2).max(1);
        let mut graphs = Vec::with_capacity(copies);
        let mut first_output = None;
        for _ in 0..copies {
            let w = engine.input(n, stride, DType::Bf16, bytemuck::cast_slice(&operands(n)))?;
            let output = match kernel.clone() {
                GemmKernel::Auto => engine.linear(&x, &w, None)?,
                GemmKernel::Reference => engine.linear_impl::<false>(&x, &w, None)?,
                GemmKernel::Wide => engine.linear_impl::<true>(&x, &w, None)?,
                GemmKernel::Pipelined => engine.linear_pipelined(&x, &w, None)?,
                GemmKernel::Source(source) => {
                    engine.linear_pipelined_source(&x, &w, None, &source)?
                }
                GemmKernel::Padded => engine.linear_pipelined_strided(
                    &x,
                    &w,
                    None,
                    k,
                    include_str!("../kernels/gemm_bf16_pipelined.loom"),
                )?,
            };
            let mut graph = engine.finish()?;
            engine.run(&mut graph)?;
            graphs.push(graph);
            first_output.get_or_insert(output);
        }
        Ok(Self {
            graphs,
            next_graph: 0,
            engine,
            output: first_output.unwrap(),
            _context: context,
        })
    }
    pub fn run(&mut self) -> Result<()> {
        self.engine.run(&mut self.graphs[self.next_graph])?;
        self.next_graph = (self.next_graph + 1) % self.graphs.len();
        Ok(())
    }
    pub fn output(&mut self) -> Result<Vec<f32>> {
        self.engine.read_f32(&self.output)
    }
}

pub struct NormBenchmark {
    graph: GraphExec,
    engine: Engine,
    output: Tensor,
    _context: ModelContext,
}

pub struct AttentionBenchmark {
    graph: GraphExec,
    engine: Engine,
    output: Tensor,
    _context: ModelContext,
}
impl AttentionBenchmark {
    pub fn new(tokens: usize, source: Option<&str>) -> Result<Self> {
        Self::build(tokens, source, false)
    }
    pub fn integrated(tokens: usize) -> Result<Self> {
        Self::build(tokens, None, true)
    }
    fn build(tokens: usize, source: Option<&str>, integrated: bool) -> Result<Self> {
        anyhow::ensure!(
            tokens > 0 && tokens <= 4096,
            "unsupported attention probe shape"
        );
        let context = ModelContext::new(Default::default())?;
        let mut engine = Engine::new(&context)?;
        let values = |len, factor| {
            (0..len)
                .map(|i| half::bf16::from_f32(((i * factor % 251) as f32 - 125.) / 64.))
                .collect::<Vec<_>>()
        };
        let q = engine.input(
            tokens,
            6144,
            DType::Bf16,
            bytemuck::cast_slice(&values(tokens * 6144, 17)),
        )?;
        let k = engine.input(
            tokens,
            1024,
            DType::Bf16,
            bytemuck::cast_slice(&values(tokens * 1024, 31)),
        )?;
        let v = engine.input(
            tokens,
            1024,
            DType::Bf16,
            bytemuck::cast_slice(&values(tokens * 1024, 43)),
        )?;
        let output = if let Some(source) = source {
            let source = format!("{}\n{}", include_str!("../kernels/common.loom"), source);
            engine.attention_tiled_source(&q, &k, &v, true, &source)?
        } else if integrated {
            engine.attention_impl::<true>(&q, &k, &v, 24, 4, true)?
        } else {
            engine.attention_impl::<false>(&q, &k, &v, 24, 4, true)?
        };
        let mut graph = engine.finish()?;
        engine.run(&mut graph)?;
        Ok(Self {
            graph,
            engine,
            output,
            _context: context,
        })
    }
    pub fn run(&mut self) -> Result<()> {
        self.engine.run(&mut self.graph)
    }
    pub fn output(&mut self) -> Result<Vec<f32>> {
        self.engine.read_f32(&self.output)
    }
}
impl NormBenchmark {
    pub fn new(rows: usize, width: usize, layer: bool, parallel: bool) -> Result<Self> {
        anyhow::ensure!(
            rows > 0 && rows <= 16384 && width > 0 && width <= 5120,
            "unsupported norm benchmark shape"
        );
        let context = ModelContext::new(Default::default())?;
        let mut engine = Engine::new(&context)?;
        let values: Vec<_> = (0..rows * width)
            .map(|i| half::bf16::from_f32(((i * 17 % 251) as f32 - 125.) / 16.))
            .collect();
        let weights: Vec<_> = (0..width)
            .map(|i| half::bf16::from_f32(((i * 7 % 67) as f32 - 33.) / 64.))
            .collect();
        let x = engine.input(rows, width, DType::Bf16, bytemuck::cast_slice(&values))?;
        let w = engine.input(1, width, DType::Bf16, bytemuck::cast_slice(&weights))?;
        let bias = layer.then_some(&w);
        let output = if parallel {
            engine.norm_impl::<true>(&x, &w, bias, 1e-6, !layer)?
        } else {
            engine.norm_impl::<false>(&x, &w, bias, 1e-6, !layer)?
        };
        let mut graph = engine.finish()?;
        engine.run(&mut graph)?;
        Ok(Self {
            graph,
            engine,
            output,
            _context: context,
        })
    }
    pub fn run(&mut self) -> Result<()> {
        self.engine.run(&mut self.graph)
    }
    pub fn output(&mut self) -> Result<Vec<f32>> {
        self.engine.read_f32(&self.output)
    }
}

pub struct DeltaBenchmark {
    graph: GraphExec,
    engine: Engine,
    _output: Tensor,
    _context: ModelContext,
}
impl DeltaBenchmark {
    /// Compare the previous three-dispatch gated norm with the FP32 fused kernel.
    pub fn gated_norm(tokens: usize, fused: bool) -> Result<Self> {
        anyhow::ensure!(
            (1..=1024).contains(&tokens),
            "benchmark supports 1..=1024 tokens"
        );
        let context = ModelContext::new(Default::default())?;
        let mut engine = Engine::new(&context)?;
        let rows = tokens * 48;
        let x: Vec<_> = (0..rows * 128)
            .map(|i| half::bf16::from_f32(((i * 17 % 251) as f32 - 125.) / 16.))
            .collect();
        let z: Vec<_> = (0..rows * 128)
            .map(|i| half::bf16::from_f32(((i * 31 % 131) as f32 - 65.) / 16.))
            .collect();
        let w: Vec<_> = (0..128)
            .map(|i| half::bf16::from_f32(0.5 + (i * 7 % 67) as f32 / 64.))
            .collect();
        let x = engine.input(rows, 128, DType::Bf16, bytemuck::cast_slice(&x))?;
        let z = engine.input(rows, 128, DType::Bf16, bytemuck::cast_slice(&z))?;
        let w = engine.input(1, 128, DType::Bf16, bytemuck::cast_slice(&w))?;
        let output = if fused {
            engine.delta_norm(&x, &w, &z)?
        } else {
            let norm = engine.norm_impl::<false>(&x, &w, None, 1e-6, false)?;
            let gate = engine.unary(&z, "silu")?;
            engine.binary(&norm, &gate, "mulf")?
        };
        let mut graph = engine.finish()?;
        engine.run(&mut graph)?;
        Ok(Self {
            graph,
            engine,
            _output: output,
            _context: context,
        })
    }
    pub fn new(tokens: usize, chunked: bool) -> Result<Self> {
        Self::build(tokens, chunked, false, true, None)
    }
    pub fn reference(tokens: usize) -> Result<Self> {
        Self::build(tokens, true, false, false, None)
    }
    pub fn profiled(tokens: usize) -> Result<Self> {
        Self::build(tokens, true, true, true, None)
    }
    pub fn persistent(tokens: usize, source: &str, values_per_wave: usize) -> Result<Self> {
        anyhow::ensure!(
            [4, 8, 16].contains(&values_per_wave),
            "unsupported values per wave"
        );
        Self::build(
            tokens,
            false,
            false,
            true,
            Some((source, values_per_wave, None)),
        )
    }
    /// Time Q/K and gate preparation together with the recurrent kernel.
    pub fn persistent_prepared(
        tokens: usize,
        prepare_source: &str,
        source: &str,
        values_per_wave: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            [4, 8, 16].contains(&values_per_wave),
            "unsupported values per wave"
        );
        Self::build(
            tokens,
            false,
            false,
            true,
            Some((source, values_per_wave, Some(prepare_source))),
        )
    }
    pub fn resident(tokens: usize) -> Result<Self> {
        Self::persistent(tokens, include_str!("../kernels/delta_persistent.loom"), 8)
    }
    fn build(
        tokens: usize,
        chunked: bool,
        profile: bool,
        optimized: bool,
        persistent: Option<(&str, usize, Option<&str>)>,
    ) -> Result<Self> {
        anyhow::ensure!(
            (1..=4096).contains(&tokens),
            "benchmark supports 1..=4096 tokens"
        );
        let context = ModelContext::new(Default::default())?;
        let mut engine = Engine::new(&context)?;
        let qkv: Vec<_> = (0..tokens * 10240)
            .map(|i| half::bf16::from_f32(((i * 17 % 251) as f32 - 125.) / 512.))
            .collect();
        let ab: Vec<_> = (0..tokens * 48)
            .map(|i| half::bf16::from_f32(((i * 13 % 131) as f32 - 65.) / 256.))
            .collect();
        let qkv = engine.input(tokens, 10240, DType::Bf16, bytemuck::cast_slice(&qkv))?;
        let a = engine.input(tokens, 48, DType::Bf16, bytemuck::cast_slice(&ab))?;
        let al = engine.input(
            1,
            48,
            DType::Bf16,
            bytemuck::cast_slice(&[half::bf16::ZERO; 48]),
        )?;
        let output = if let Some((source, values_per_wave, prepare_source)) = persistent {
            let prepared = if let Some(source) = prepare_source {
                let prepared = engine.alloc(tokens, 4192, DType::F32)?;
                let source = format!("{}\n{}", include_str!("../kernels/common.loom"), source);
                engine.dispatch(
                    &source,
                    tokens * 16 * 32,
                    &[
                        ("qkv", &qkv),
                        ("a", &a),
                        ("b", &a),
                        ("al", &al),
                        ("dt", &al),
                    ],
                    &prepared,
                    &[
                        ("n", tokens.to_string()),
                        ("output_len", (tokens * 4192).to_string()),
                    ],
                )?;
                Some(prepared)
            } else {
                None
            };
            let out = engine.alloc(tokens, 6144, DType::Bf16)?;
            let source = format!("{}\n{}", include_str!("../kernels/common.loom"), source);
            engine.dispatch(
                &source,
                48 * 128 / values_per_wave * 32,
                &[
                    ("qkv", &qkv),
                    ("a", prepared.as_ref().unwrap_or(&a)),
                    ("b", &a),
                    ("al", &al),
                    ("dt", &al),
                ],
                &out,
                &[
                    ("n", tokens.to_string()),
                    ("output_len", (tokens * 6144).to_string()),
                ],
            )?;
            out
        } else if chunked {
            if optimized {
                engine.delta_chunked(&qkv, &a, &a, &al, &al)?
            } else {
                engine.delta_chunked_impl::<false>(&qkv, &a, &a, &al, &al)?
            }
        } else {
            engine.delta(&qkv, &a, &a, &al, &al)?
        };
        let mut graph = if profile {
            engine.finish_profiled()?
        } else {
            engine.finish()?
        };
        engine.run(&mut graph)?;
        Ok(Self {
            graph,
            engine,
            _output: output,
            _context: context,
        })
    }
    /// Includes dispatch and completion fence; excludes compilation/upload/readback.
    pub fn run(&mut self) -> Result<()> {
        self.engine.run(&mut self.graph)
    }
    pub fn allocated_bytes(&self) -> usize {
        self.engine.allocated_bytes()
    }
    pub fn output(&mut self) -> Result<Vec<f32>> {
        self.engine.read_f32(&self._output)
    }
    pub fn profile(&mut self) -> Result<hrx::fabric::DeviceProfile> {
        self.engine.profile(&mut self.graph)
    }
}
