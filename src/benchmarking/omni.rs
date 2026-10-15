//! Routed expert benchmarks, including grouping/packing and graph replay.
use super::*;
use crate::Precision;

/// Parallel top-k selection against the original ascending serial scan.
pub struct OmniRouterBenchmark {
    engine: Engine,
    graphs: [GraphExec; 2],
    _outputs: [Tensor; 2],
    _context: ModelContext,
}

impl OmniRouterBenchmark {
    pub fn new(tokens: usize, experts: usize, topk: usize, dtype_bf16: bool) -> Result<Self> {
        anyhow::ensure!(
            tokens > 0 && experts <= 128 && topk > 0 && topk <= experts,
            "invalid router benchmark shape"
        );
        let budget = hrx::residency::ResidencyManager::new(64 << 20)?;
        let context = ModelContext::new(hrx::execution::RuntimeOptions {
            memory_budget: Some(budget.budget()),
            ..Default::default()
        })?;
        let mut engine = Engine::new(&context)?;
        let dtype = if dtype_bf16 { DType::Bf16 } else { DType::F32 };
        let logits = engine.alloc(tokens, experts, dtype)?;
        let old = engine.alloc(tokens, topk * 2, DType::F32)?;
        engine.dispatch(
            concat!(
                include_str!("../../kernels/common.loom"),
                include_str!("../../benches/kernels/omni_route_serial.loom")
            ),
            tokens,
            &[("x", &logits)],
            &old,
            &[("experts", experts.to_string()), ("topk", topk.to_string())],
        )?;
        let mut old_graph = engine.finish()?;
        let new = engine.omni_route(&logits, topk)?;
        let mut new_graph = engine.finish()?;
        for phase in 0..4 {
            let mut values: Vec<f32> = (0..tokens * experts)
                .map(|i| match phase {
                    0 => ((i * 37 + i / experts * 11) % 257) as f32 / 16. - 8.,
                    1 => ((i * 53 + i / experts * 7) % 5) as f32 - 2.,
                    2 => {
                        if i % 2 == 0 {
                            0.0
                        } else {
                            -0.0
                        }
                    }
                    _ => ((i * 79 + i / experts * 17) % 263) as f32 * 16. - 2048.,
                })
                .collect();
            if dtype_bf16 {
                let rounded: Vec<_> = values.iter().map(|&v| half::bf16::from_f32(v)).collect();
                values = rounded.iter().map(|v| v.to_f32()).collect();
                engine.upload(&logits, bytemuck::cast_slice(&rounded))?;
            } else {
                engine.upload(&logits, bytemuck::cast_slice(&values))?;
            }
            engine.run(&mut old_graph)?;
            engine.run(&mut new_graph)?;
            let a = engine.read_f32(&old)?;
            let b = engine.read_f32(&new)?;
            anyhow::ensure!(
                a == b,
                "router changed output: tokens={tokens}, experts={experts}, topk={topk}, bf16={dtype_bf16}, phase={phase}; max error={}",
                a.iter()
                    .zip(&b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max)
            );
            for token in 0..tokens {
                let mut expected: Vec<_> = (0..experts).collect();
                expected.sort_by(|&a, &b| {
                    values[token * experts + b]
                        .partial_cmp(&values[token * experts + a])
                        .unwrap()
                });
                for rank in 0..topk {
                    anyhow::ensure!(
                        b[token * topk * 2 + rank] == expected[rank] as f32,
                        "router CPU selection mismatch"
                    );
                }
            }
        }
        Ok(Self {
            engine,
            graphs: [old_graph, new_graph],
            _outputs: [old, new],
            _context: context,
        })
    }
    pub fn run(&mut self, candidate: bool) -> Result<()> {
        self.engine.run(&mut self.graphs[usize::from(candidate)])?;
        Ok(())
    }
}

pub struct OmniRouteBenchmark {
    engine: Engine,
    graphs: [GraphExec; 2],
    _outputs: [Tensor; 2],
    _context: ModelContext,
}

impl OmniRouteBenchmark {
    pub fn new(tokens: usize, skewed: bool) -> Result<Self> {
        anyhow::ensure!(
            (1..=crate::omni::MAX_EXPERT_CHUNK).contains(&tokens),
            "Omni benchmark token count"
        );
        let budget = hrx::residency::ResidencyManager::new(64 << 20)?;
        let context = ModelContext::new(hrx::execution::RuntimeOptions {
            memory_budget: Some(budget.budget()),
            ..Default::default()
        })?;
        let mut engine = Engine::new(&context)?;
        let routes = engine.input(
            tokens,
            16,
            DType::F32,
            bytemuck::cast_slice(&OmniExpertBenchmark::routes(tokens, skewed, 0)),
        )?;
        let old = engine.alloc(128, tokens + 1, DType::F32)?;
        engine.dispatch(
            concat!(
                include_str!("../../kernels/common.loom"),
                include_str!("../../benches/kernels/omni_group_routes_serial.loom")
            ),
            128,
            &[("routes", &routes)],
            &old,
            &[("tokens", tokens.to_string()), ("topk", "8".into())],
        )?;
        let mut old_graph = engine.finish()?;
        let new = engine.omni_group_routes(&routes)?;
        let mut new_graph = engine.finish()?;
        for phase in 0..3 {
            engine.upload(
                &routes,
                bytemuck::cast_slice(&OmniExpertBenchmark::routes(tokens, skewed, phase)),
            )?;
            engine.run(&mut old_graph)?;
            engine.run(&mut new_graph)?;
            let a = engine.read_f32(&old)?;
            let b = engine.read_f32(&new)?;
            for expert in 0..128 {
                let base = expert * (tokens + 1);
                let count = a[base] as usize;
                anyhow::ensure!(count <= tokens, "invalid reference route count");
                anyhow::ensure!(
                    a[base..base + count + 1] == b[base..base + count + 1],
                    "route table differs at expert {expert}, replay {phase}"
                );
            }
        }
        Ok(Self {
            engine,
            graphs: [old_graph, new_graph],
            _outputs: [old, new],
            _context: context,
        })
    }

    pub fn run(&mut self, candidate: bool) -> Result<()> {
        self.engine.run(&mut self.graphs[usize::from(candidate)])
    }
}

pub struct OmniExpertBenchmark {
    graphs: [GraphExec; 2],
    outputs: [Tensor; 2],
    routes: Tensor,
    input: Tensor,
    engine: Engine,
    tokens: usize,
    width: usize,
    routed: bool,
    precision: Precision,
    _context: ModelContext,
}
impl OmniExpertBenchmark {
    pub fn new(tokens: usize, down: bool, precision: Precision, skewed: bool) -> Result<Self> {
        anyhow::ensure!(
            (1..=crate::omni::MAX_EXPERT_CHUNK).contains(&tokens),
            "Omni benchmark token count"
        );
        let (k, n) = if down { (768, 2048) } else { (2048, 768) };
        crate::memory::require(2 << 30)?;
        let budget = hrx::residency::ResidencyManager::new(4 << 30)?;
        let context = ModelContext::new(hrx::execution::RuntimeOptions {
            memory_budget: Some(budget.budget()),
            ..Default::default()
        })?;
        let mut engine = Engine::new(&context)?;
        engine.quantized = crate::quantized::QuantizedProjections::for_precision(precision);
        // Nonperiodic, exactly representable fixtures exercise cancellation and all experts.
        let mut state = 12345u32;
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
        let name = "model.language_model.layers.0.mlp.experts.up_proj";
        let weight = format!("{name}.weight");
        if !engine.load_quantized_weight(&weight, 128 * n, k, &tiled)? {
            engine.weight(weight, 128 * n, k, &tiled)?;
        }
        drop(tiled);
        let rows = tokens * if down { 8 } else { 1 };
        let input = engine.input(
            rows,
            k,
            DType::Bf16,
            bytemuck::cast_slice(&Self::inputs(rows * k, 0)),
        )?;
        let routes = engine.input(
            tokens,
            16,
            DType::F32,
            bytemuck::cast_slice(&Self::routes(tokens, skewed, 0)),
        )?;
        let old = if let Some(directory) = std::env::var_os("CLEF_BENCH_OMNI_REFERENCE_DIR") {
            let candidate = engine.omni_kernel_dir.replace(directory.into());
            let groups = engine.omni_group_routes(&routes)?;
            let old = engine.omni_expert_impl(&input, &routes, Some(&groups), name, down)?;
            engine.omni_kernel_dir = candidate;
            old
        } else {
            engine.omni_expert(&input, &routes, name, down)?
        };
        let old_graph = engine.finish()?;
        let groups = if engine.omni_compact_experts && precision == Precision::Int4 {
            engine.omni_compact_routes(&routes)?.0
        } else {
            engine.omni_group_routes(&routes)?
        };
        let new = engine.omni_expert_impl(&input, &routes, Some(&groups), name, down)?;
        let new_graph = engine.finish()?;
        let mut bench = Self {
            graphs: [old_graph, new_graph],
            outputs: [old, new],
            routes,
            input,
            engine,
            tokens,
            width: k,
            routed: down,
            precision,
            _context: context,
        };
        // Change route membership and activation values without rebuilding either graph.
        for phase in 0..3 {
            bench.engine.upload(
                &bench.routes,
                bytemuck::cast_slice(&Self::routes(tokens, skewed, phase)),
            )?;
            bench.engine.upload(
                &bench.input,
                bytemuck::cast_slice(&Self::inputs(rows * k, phase)),
            )?;
            bench.run(false)?;
            bench.run(true)?;
            bench.check()?;
            if groups.rows == 1 {
                let actual = bench.engine.read_f32(&groups)?;
                let routes = Self::routes(tokens, skewed, phase);
                let mut expected = Vec::new();
                for expert in 0..128 {
                    let members: Vec<_> = (0..tokens)
                        .flat_map(|token| (0..8).map(move |rank| (token, rank)))
                        .filter(|&(token, rank)| routes[token * 16 + rank] == expert as f32)
                        .map(|(token, rank)| (token * 8 + rank) as f32)
                        .collect();
                    let base = expert * (tokens + 1);
                    anyhow::ensure!(
                        actual[base] == members.len() as f32
                            && actual[base + 1..base + 1 + members.len()] == members,
                        "compact route CPU mismatch: expert={expert}, phase={phase}"
                    );
                    for row in (0..members.len()).step_by(16) {
                        expected.extend([expert as f32, row as f32]);
                    }
                }
                let base = 128 * (tokens + 1) + tokens * 8;
                anyhow::ensure!(
                    actual[base] == (expected.len() / 2) as f32
                        && actual[base + 1..base + 1 + expected.len()] == expected,
                    "compact task CPU mismatch: tokens={tokens}, skewed={skewed}, phase={phase}"
                );
            }
        }
        Ok(bench)
    }
    fn inputs(count: usize, phase: usize) -> Vec<half::bf16> {
        (0..count)
            .map(|i| half::bf16::from_f32(((i * 17 + phase * 7) % 251) as f32 / 128. - 1.))
            .collect()
    }
    fn routes(tokens: usize, skewed: bool, phase: usize) -> Vec<f32> {
        (0..tokens)
            .flat_map(|t| {
                (0..16).map(move |r| {
                    if r < 8 {
                        ((if skewed { 0 } else { t * 13 } + r * 7 + phase * 19) % 128) as f32
                    } else {
                        0.125
                    }
                })
            })
            .collect()
    }
    pub fn run(&mut self, candidate: bool) -> Result<()> {
        self.engine.run(&mut self.graphs[usize::from(candidate)])
    }
    pub fn check(&mut self) -> Result<()> {
        let old = self.engine.read_f32(&self.outputs[0])?;
        let new = self.engine.read_f32(&self.outputs[1])?;
        let mut worst = 0.0f32;
        let exact = std::env::var_os("CLEF_BENCH_OMNI_EXACT").is_some();
        for (i, (&a, &b)) in old.iter().zip(&new).enumerate() {
            worst = worst.max((a - b).abs());
            let tolerance = if self.precision == Precision::Bf16 {
                0.0078125 * a.abs().max(0.01)
            } else {
                0.0
            };
            let matches = if self.precision == Precision::Bf16 && !exact {
                (a - b).abs() <= tolerance
            } else {
                a.to_bits() == b.to_bits()
            };
            anyhow::ensure!(
                a.is_finite() && b.is_finite() && matches,
                "Omni expert mismatch {:?} tokens={} k={} down={}: index={i}, old={a}, new={b}, error={}, max={worst}",
                self.precision,
                self.tokens,
                self.width,
                self.routed,
                (a - b).abs()
            );
        }
        Ok(())
    }
}

/// Grouping plus weighted accumulation; keeps the original BF16 rounding order.
pub struct OmniCombineBenchmark {
    engine: Engine,
    graphs: [GraphExec; 2],
    _outputs: Vec<Tensor>,
    _context: ModelContext,
}
impl OmniCombineBenchmark {
    pub fn new(tokens: usize, skewed: bool) -> Result<Self> {
        anyhow::ensure!(
            (1..=crate::omni::MAX_EXPERT_CHUNK).contains(&tokens),
            "combine benchmark tokens"
        );
        let budget = hrx::residency::ResidencyManager::new(256 << 20)?;
        let context = ModelContext::new(hrx::execution::RuntimeOptions {
            memory_budget: Some(budget.budget()),
            ..Default::default()
        })?;
        let mut engine = Engine::new(&context)?;
        let routes = engine.alloc(tokens, 16, DType::F32)?;
        let values = engine.alloc(tokens * 8, 2048, DType::Bf16)?;
        let old_groups = engine.omni_group_routes(&routes)?;
        let old = engine.omni_combine(&values, &routes)?;
        let mut old_graph = engine.finish()?;
        let (new_groups, order) = engine.omni_group_routes_ordered(&routes)?;
        let new = engine.omni_combine_ordered(&values, &routes, &order)?;
        let mut new_graph = engine.finish()?;
        for phase in 0..3 {
            let mut data = OmniExpertBenchmark::routes(tokens, skewed, phase);
            for token in 0..tokens {
                for rank in 0..8 {
                    data[token * 16 + rank + 8] =
                        half::bf16::from_f32(((rank + phase) % 8 + 1) as f32 / 36.).to_f32();
                }
            }
            engine.upload(&routes, bytemuck::cast_slice(&data))?;
            engine.upload(
                &values,
                bytemuck::cast_slice(&OmniExpertBenchmark::inputs(tokens * 8 * 2048, phase)),
            )?;
            engine.run(&mut old_graph)?;
            engine.run(&mut new_graph)?;
            let a = engine.read_f32(&old)?;
            let b = engine.read_f32(&new)?;
            anyhow::ensure!(
                a.iter()
                    .zip(&b)
                    .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits()),
                "combine changed BF16 output: tokens={tokens}, skewed={skewed}, phase={phase}"
            );
            let a = engine.read_f32(&old_groups)?;
            let b = engine.read_f32(&new_groups)?;
            for expert in 0..128 {
                let base = expert * (tokens + 1);
                let len = a[base] as usize + 1;
                anyhow::ensure!(
                    a[base..base + len] == b[base..base + len],
                    "ordered grouping changed stable routes"
                );
            }
            let actual = engine.read_f32(&order)?;
            for token in 0..tokens {
                let mut expected: Vec<_> = (0..8).collect();
                expected.sort_by(|&a, &b| data[token * 16 + a].total_cmp(&data[token * 16 + b]));
                for rank in 0..8 {
                    anyhow::ensure!(
                        actual[token * 8 + rank] == expected[rank] as f32,
                        "incorrect ascending expert order"
                    );
                }
            }
        }
        Ok(Self {
            engine,
            graphs: [old_graph, new_graph],
            _outputs: vec![routes, values, old_groups, new_groups, order, old, new],
            _context: context,
        })
    }
    pub fn run(&mut self, candidate: bool) -> Result<()> {
        self.engine.run(&mut self.graphs[usize::from(candidate)])
    }
}
