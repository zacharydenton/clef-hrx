use super::*;

/// Matched causal four-tap convolution and SiLU, with independent rotating inputs.
pub struct ConvolutionBenchmark {
    graphs: [Vec<GraphExec>; 2],
    outputs: [Vec<Tensor>; 2],
    expected: Vec<u32>,
    next: [usize; 2],
    engine: Engine,
    _context: ModelContext,
}

impl ConvolutionBenchmark {
    pub fn new(rows: usize, source: &str, tile: usize, rotating: bool) -> Result<Self> {
        anyhow::ensure!((1..=4032).contains(&rows) && [1, 2, 4, 8].contains(&tile));
        let width = 10240;
        let copies = if rotating {
            (64usize << 20).div_ceil(rows * width * 2)
        } else {
            1
        };
        anyhow::ensure!(!rotating || rows >= 128);
        let context = ModelContext::new(Default::default())?;
        let mut engine = Engine::new(&context)?;
        let values: Vec<_> = (0..(rows + 1) * width)
            .map(|i| {
                half::bf16::from_f32(
                    ((i * 17 % 1031) as f32 - 515.) / if i % 7 == 0 { 65536. } else { 128. },
                )
            })
            .collect();
        let weights: Vec<_> = (0..(width + 1) * 4)
            .map(|i| half::bf16::from_f32(((i * 31 % 127) as f32 - 63.) / 128.))
            .collect();
        let source = format!("{}\n{source}", include_str!("../../kernels/common.loom"));
        let mut graphs = [Vec::new(), Vec::new()];
        let mut outputs = [Vec::new(), Vec::new()];
        for variant in 0..2 {
            for _ in 0..copies {
                let x = engine
                    .input(rows + 1, width, DType::Bf16, bytemuck::cast_slice(&values))?
                    .slice_rows(1, rows);
                let w = engine
                    .input(width + 1, 4, DType::Bf16, bytemuck::cast_slice(&weights))?
                    .slice_rows(1, width);
                let out = if variant == 0 {
                    engine.conv_delta_source(
                        &x,
                        &w,
                        concat!(
                            include_str!("../../kernels/common.loom"),
                            include_str!("../../kernels/conv_delta.loom")
                        ),
                        1,
                    )?
                } else {
                    engine.conv_delta_source(&x, &w, &source, tile)?
                };
                let mut graph = engine.finish()?;
                engine.run(&mut graph)?;
                graphs[variant].push(graph);
                outputs[variant].push(out);
            }
        }
        anyhow::ensure!(
            engine.allocated_bytes() <= 1 << 30,
            "paired convolution exceeds 1 GiB"
        );
        let expected = engine
            .read_f32(&outputs[0][0])?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let mut result = Self {
            graphs,
            outputs,
            expected,
            next: [0, 0],
            engine,
            _context: context,
        };
        result.check()?;
        Ok(result)
    }

    pub fn run(&mut self, candidate: bool) -> Result<()> {
        let v = usize::from(candidate);
        self.engine.run(&mut self.graphs[v][self.next[v]])?;
        self.next[v] = (self.next[v] + 1) % self.graphs[v].len();
        Ok(())
    }

    pub fn check(&mut self) -> Result<()> {
        for variant in 0..2 {
            for output in &self.outputs[variant] {
                let actual = self.engine.read_f32(output)?;
                anyhow::ensure!(
                    actual.iter().all(|v| v.is_finite())
                        && actual
                            .iter()
                            .map(|v| v.to_bits())
                            .eq(self.expected.iter().copied()),
                    "convolution output changed"
                );
            }
        }
        Ok(())
    }
}
