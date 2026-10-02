//! Development-only graph harness for Criterion; no checkpoint allocation.
use crate::{
    Result,
    gpu::{DType, Engine, Tensor},
};
use hrx::{GraphExec, inference::ModelContext};

pub struct NormBenchmark {
    graph: GraphExec,
    engine: Engine,
    output: Tensor,
    _context: ModelContext,
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
        Self::build(tokens, chunked, false, true)
    }
    pub fn reference(tokens: usize) -> Result<Self> {
        Self::build(tokens, true, false, false)
    }
    pub fn profiled(tokens: usize) -> Result<Self> {
        Self::build(tokens, true, true, true)
    }
    fn build(tokens: usize, chunked: bool, profile: bool, optimized: bool) -> Result<Self> {
        anyhow::ensure!(
            (1..=1024).contains(&tokens),
            "benchmark supports 1..=1024 tokens"
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
        let output = if chunked {
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
