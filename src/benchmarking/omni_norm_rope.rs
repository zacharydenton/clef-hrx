use super::*;

/// Compare fused Q/K normalization and RoPE with materialized Omni operators.
pub struct OmniNormRopeBenchmark {
    engine: Engine,
    graphs: [GraphExec; 2],
    outputs: [Tensor; 2],
    _context: ModelContext,
}

impl OmniNormRopeBenchmark {
    pub fn new(tokens: usize, heads: usize) -> Result<Self> {
        anyhow::ensure!(
            (1..=65536).contains(&tokens) && [4, 32].contains(&heads),
            "unsupported Omni Q/K shape"
        );
        let budget = hrx::residency::ResidencyManager::new(3 << 30)?;
        let context = ModelContext::new(hrx::execution::RuntimeOptions {
            memory_budget: Some(budget.budget()),
            ..Default::default()
        })?;
        let mut engine = Engine::new(&context)?;
        engine.architecture = crate::ModelVariant::ClefOmni.architecture();
        let x = engine.alloc(tokens, heads * 128, DType::Bf16)?;
        let w = engine.alloc(1, 128, DType::Bf16)?;
        let cs = engine.alloc(tokens, 256, DType::Bf16)?;
        let normalized = engine.norm(&x.reshape(tokens * heads, 128), &w, None, 1e-6, false)?;
        let reference = engine.rope(&normalized.reshape(tokens, heads * 128), heads, 128, &cs)?;
        let reference_graph = engine.finish()?;
        let candidate = engine.omni_norm_rope(&x, &w, heads, &cs)?;
        let candidate_graph = engine.finish()?;
        let mut bench = Self {
            engine,
            graphs: [reference_graph, candidate_graph],
            outputs: [reference, candidate],
            _context: context,
        };
        // Changed inputs, learned weights and positions exercise graph replay.
        // Include zero rows, signed zeros, outliers and mixed head/token values.
        for phase in 0..3 {
            let values: Vec<_> = (0..x.len())
                .map(|i| {
                    let value = match phase {
                        0 => ((i * 37 + i / 128 * 11) % 1031) as f32 / 32. - 16.,
                        1 if (i / 128) % 7 == 0 => {
                            if i % 2 == 0 {
                                0.
                            } else {
                                -0.
                            }
                        }
                        1 => ((i * 53 + i / 128 * 7) % 257) as f32 / 1024. - 0.125,
                        _ if i % 131 == 0 => 1024.,
                        _ => ((i * 79 + i / 128 * 17) % 263) as f32 / 8. - 16.,
                    };
                    half::bf16::from_f32(value)
                })
                .collect();
            bench.engine.upload(&x, bytemuck::cast_slice(&values))?;
            drop(values);
            let weights: Vec<_> = (0..128)
                .map(|i| half::bf16::from_f32(((i * 19 + phase * 13) % 67) as f32 / 32. - 1.))
                .collect();
            bench.engine.upload(&w, bytemuck::cast_slice(&weights))?;
            let positions: Vec<_> = (0..cs.len())
                .map(|i| {
                    let column = i % 256;
                    let angle = ((i / 256 * 7 + column % 64 * 11 + phase * 31) % 4093) as f32 / 64.;
                    half::bf16::from_f32(if column < 128 {
                        angle.cos()
                    } else {
                        angle.sin()
                    })
                })
                .collect();
            bench.engine.upload(&cs, bytemuck::cast_slice(&positions))?;
            for _ in 0..2 {
                bench.run(false)?;
                bench.run(true)?;
                bench.check()?;
            }
        }
        Ok(bench)
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
            "fused Omni normalization/RoPE differs from materialized BF16 reference; max error {}",
            reference
                .iter()
                .zip(&candidate)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        );
        Ok(())
    }
}
