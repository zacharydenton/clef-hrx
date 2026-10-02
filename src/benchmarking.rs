//! Development-only graph harness for Criterion; no checkpoint allocation.
use crate::{
    Result,
    gpu::{DType, Engine, Tensor},
};
use hrx::{GraphExec, inference::ModelContext};

pub struct DeltaBenchmark {
    graph: GraphExec,
    engine: Engine,
    _output: Tensor,
    _context: ModelContext,
}
impl DeltaBenchmark {
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
