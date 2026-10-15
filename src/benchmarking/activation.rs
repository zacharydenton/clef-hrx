use super::*;

/// Both activation paths share inputs and remain resident for paired timing.
pub struct SwiGluBenchmark {
    graphs: [GraphExec; 2],
    outputs: [Tensor; 2],
    engine: Engine,
    _context: ModelContext,
}

impl SwiGluBenchmark {
    pub fn new(rows: usize, width: usize) -> Result<Self> {
        anyhow::ensure!(
            (1..=16384).contains(&rows)
                && (1..=17408).contains(&width)
                && rows * width <= 8192 * 17408,
            "unsupported activation benchmark shape"
        );
        let context = ModelContext::new(Default::default())?;
        let mut engine = Engine::new(&context)?;
        let values = |factor| {
            (0..rows * width)
                .map(|i| half::bf16::from_f32(((i * factor % 1031) as f32 - 515.) / 16.))
                .collect::<Vec<_>>()
        };
        let gate = engine.input(rows, width, DType::Bf16, bytemuck::cast_slice(&values(17)))?;
        let up = engine.input(rows, width, DType::Bf16, bytemuck::cast_slice(&values(31)))?;
        let silu = engine.unary(&gate, "silu")?;
        let reference = engine.binary(&silu, &up, "mulf")?;
        let reference_graph = engine.finish()?;
        let candidate = engine.swiglu(&gate, &up)?;
        let candidate_graph = engine.finish()?;
        let mut benchmark = Self {
            graphs: [reference_graph, candidate_graph],
            outputs: [reference, candidate],
            engine,
            _context: context,
        };
        benchmark.run(false)?;
        benchmark.run(true)?;
        benchmark.check()?;
        Ok(benchmark)
    }

    pub fn run(&mut self, candidate: bool) -> Result<()> {
        self.engine.run(&mut self.graphs[usize::from(candidate)])
    }

    pub fn check(&mut self) -> Result<()> {
        let reference = self.engine.read_f32(&self.outputs[0])?;
        let candidate = self.engine.read_f32(&self.outputs[1])?;
        anyhow::ensure!(
            reference
                .iter()
                .zip(&candidate)
                .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits()),
            "fused SwiGLU differs from materialized BF16 reference"
        );
        Ok(())
    }
}
