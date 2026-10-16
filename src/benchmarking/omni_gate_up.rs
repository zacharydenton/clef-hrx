use super::*;
use crate::Precision;

/// Exact grouped INT4 gate/up plus SwiGLU comparison, including packing/grouping.
pub struct OmniGateUpBenchmark {
    engine: Engine,
    graphs: [GraphExec; 2],
    outputs: [Tensor; 2],
    _context: ModelContext,
}

impl OmniGateUpBenchmark {
    pub fn new(tokens: usize, skewed: bool) -> Result<Self> {
        anyhow::ensure!((1..=2048).contains(&tokens), "unsupported expert chunk");
        crate::memory::require(2 << 30)?;
        let budget = hrx::residency::ResidencyManager::new(4 << 30)?;
        let context = ModelContext::new(hrx::execution::RuntimeOptions {
            memory_budget: Some(budget.budget()),
            ..Default::default()
        })?;
        let mut engine = Engine::new(&context)?;
        engine.omni_fast_defaults = false;
        engine.quantized = crate::quantized::QuantizedProjections::for_precision(Precision::Int4);
        engine.omni_fused_mlp = true;
        // Pin the reference independently of other experimental environment selectors.
        engine.omni_kernel_dir = std::env::var_os("CLEF_BENCH_OMNI_REFERENCE_DIR").map(Into::into);
        engine.omni_kernel_min_tokens = 0;
        let (k, n) = (2048, 768);
        let gate_name = "model.language_model.layers.0.mlp.experts.gate_proj";
        let up_name = "model.language_model.layers.0.mlp.experts.up_proj";
        for (name, seed) in [(gate_name, 12345u32), (up_name, 67891)] {
            let mut state = seed;
            let weights: Vec<_> = (0..128 * n * k)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    half::bf16::from_f32((state % 127) as f32 / 1024. - 63. / 1024.)
                })
                .collect();
            let tiled = crate::checkpoint::tile_rows(bytemuck::cast_slice(&weights), 128 * n, k);
            drop(weights);
            anyhow::ensure!(
                engine.load_quantized_weight(&format!("{name}.weight"), 128 * n, k, &tiled)?,
                "expected INT4 expert weights"
            );
        }
        let x = engine.alloc(tokens, k, DType::Bf16)?;
        let routes = engine.alloc(tokens, 16, DType::F32)?;
        let groups = engine.omni_group_routes(&routes)?;
        let gate = engine.omni_expert_impl(&x, &routes, Some(&groups), gate_name, false)?;
        let up = engine.omni_expert_impl(&x, &routes, Some(&groups), up_name, false)?;
        let reference = engine.swiglu(&gate, &up)?;
        let reference_graph = engine.finish()?;
        let groups = engine.omni_group_routes(&routes)?;
        let candidate_source = std::env::var_os("CLEF_BENCH_OMNI_GATE_UP_SOURCE")
            .map(std::fs::read_to_string)
            .transpose()?
            .unwrap_or_else(|| include_str!("../../benches/kernels/omni_gate_up_i4.loom").into());
        let source = format!(
            "{}\n{candidate_source}",
            include_str!("../../kernels/common.loom")
        );
        engine.omni_gate_up_source = Some(source.into());
        let candidate = engine.omni_gate_up(&x, &routes, &groups, gate_name, up_name)?;
        let candidate_graph = engine.finish()?;
        let mut bench = Self {
            engine,
            graphs: [reference_graph, candidate_graph],
            outputs: [reference, candidate],
            _context: context,
        };
        for phase in 0..3 {
            bench.engine.upload(
                &routes,
                bytemuck::cast_slice(&OmniExpertBenchmark::routes(tokens, skewed, phase)),
            )?;
            let input: Vec<_> = (0..tokens * k)
                .map(|i| {
                    let value = match phase {
                        0 => ((i * 17) % 251) as f32 / 128. - 1.,
                        1 if (i / k) % 7 == 0 => {
                            if i % 2 == 0 {
                                0.
                            } else {
                                -0.
                            }
                        }
                        1 => ((i * 37 + 13) % 257) as f32 / 1024. - 0.125,
                        _ if i % 131 == 0 => 16.,
                        _ => ((i * 79 + 17) % 263) as f32 / 32. - 4.,
                    };
                    half::bf16::from_f32(value)
                })
                .collect();
            bench.engine.upload(&x, bytemuck::cast_slice(&input))?;
            for replay in 0..2 {
                // Poison both output buffers: missing route/tail writes must fail.
                let poison = vec![half::bf16::NAN; tokens * 8 * n];
                for output in &bench.outputs {
                    bench.engine.upload(output, bytemuck::cast_slice(&poison))?;
                }
                bench.run(replay == 0)?;
                bench.run(replay != 0)?;
                bench.check()?;
            }
        }
        Ok(bench)
    }

    pub fn run(&mut self, candidate: bool) -> Result<()> {
        self.engine.run(&mut self.graphs[usize::from(candidate)])
    }

    pub fn check(&mut self) -> Result<()> {
        let a = self.engine.read_f32(&self.outputs[0])?;
        let b = self.engine.read_f32(&self.outputs[1])?;
        for (i, (a, b)) in a.into_iter().zip(b).enumerate() {
            anyhow::ensure!(
                a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
                "fused gate/up mismatch at {i}: reference={a}, candidate={b}"
            );
        }
        Ok(())
    }
}
