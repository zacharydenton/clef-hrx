//! Benchmark harness for the native integer path.
use super::*;
use crate::quantized::prepare;
use anyhow::ensure;
use hrx::{Constants, loom::Specialization};

/// Paired activation packing; logical codes and scales must match exactly.
pub struct PackingBenchmark {
    graphs: [GraphExec; 2],
    outputs: [(Tensor, Tensor); 2],
    // A replay of either graph must not overwrite the other graph's outputs.
    _reference_intermediates: Vec<Tensor>,
    engine: Engine,
    logical_words: usize,
    _context: ModelContext,
}

impl PackingBenchmark {
    pub fn new(rows: usize, width: usize, bits: usize) -> Result<Self> {
        ensure!(
            (1..=8192).contains(&rows)
                && (256..=32768).contains(&width)
                && width.is_multiple_of(256)
                && [4, 8].contains(&bits),
            "unsupported packing benchmark shape"
        );
        let context = ModelContext::new(Default::default())?;
        let mut engine = Engine::new(&context)?;
        let values: Vec<_> = (0..rows * width)
            .map(|i| half::bf16::from_f32(((i * 17 % 1031) as f32 - 515.) / 16.))
            .collect();
        let input = engine.input(rows, width, DType::Bf16, bytemuck::cast_slice(&values))?;
        let stride = width
            + if width.is_multiple_of(1024) {
                512 / bits
            } else {
                0
            };
        let candidate_source = std::env::var("CLEF_BENCH_PACKING_SOURCE")
            .ok()
            .map(std::fs::read_to_string)
            .transpose()?;
        let fused = std::env::var("CLEF_BENCH_PACKING_SWIGLU").as_deref() == Ok("1");
        let attention_gate =
            std::env::var("CLEF_BENCH_PACKING_ATTENTION_GATE").as_deref() == Ok("1");
        ensure!(!(fused && attention_gate), "choose one packing activation");
        ensure!(!attention_gate || width == 6144, "attention packing width");
        ensure!(
            !(fused || attention_gate) || candidate_source.is_some(),
            "fused packing requires a candidate source"
        );
        let up = if fused || attention_gate {
            let up_width = width * if attention_gate { 2 } else { 1 };
            let values: Vec<_> = (0..rows * up_width)
                .map(|i| half::bf16::from_f32(((i * 29 % 997) as f32 - 498.) / 32.))
                .collect();
            Some(engine.input(rows, up_width, DType::Bf16, bytemuck::cast_slice(&values))?)
        } else {
            None
        };
        let mut reference_intermediates = Vec::new();
        let activated = if attention_gate {
            let gate = engine.columns(&up.as_ref().unwrap().reshape(rows * 24, 512), 256, 256)?;
            reference_intermediates.push(gate.clone());
            let sigmoid = engine.unary(&gate.reshape(rows, width), "sigmoid")?;
            reference_intermediates.push(sigmoid.clone());
            engine.binary(&input, &sigmoid, "mulf")?
        } else if let Some(up) = &up {
            engine.swiglu(&input, up)?
        } else {
            input.clone()
        };
        reference_intermediates.push(activated.clone());
        // Keep the pre-register production baseline reproducible after selection.
        let reference = crate::quantized::prepare_with_source(
            &mut engine,
            &activated,
            bits,
            stride,
            if fused || attention_gate {
                include_str!("../../kernels/prepare_integer_register.loom")
            } else if candidate_source.is_some() && width <= 6144 {
                include_str!("../../kernels/prepare_integer_single.loom")
            } else {
                include_str!("../../kernels/prepare_integer.loom")
            },
        )?;
        let reference_graph = engine.finish()?;
        let inputs = if let Some(up) = &up {
            vec![&input, up]
        } else {
            vec![&input]
        };
        let candidate = if attention_gate {
            crate::quantized::prepare_attention_gate_with_source(
                &mut engine,
                &input,
                up.as_ref().unwrap(),
                bits,
                stride,
                candidate_source.as_deref().unwrap(),
            )?
        } else {
            crate::quantized::prepare_with_inputs(
                &mut engine,
                &inputs,
                bits,
                stride,
                candidate_source
                    .as_deref()
                    .unwrap_or(include_str!("../../kernels/prepare_integer_single.loom")),
            )?
        };
        let candidate_graph = engine.finish()?;
        let mut benchmark = Self {
            graphs: [reference_graph, candidate_graph],
            outputs: [reference, candidate],
            _reference_intermediates: reference_intermediates,
            engine,
            logical_words: width / (32 / bits),
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
        let codes: Vec<_> = self
            .outputs
            .iter()
            .map(|(codes, _)| {
                self.engine
                    .read_f32(codes)
                    .map(|v| v.into_iter().map(f32::to_bits).collect::<Vec<_>>())
            })
            .collect::<Result<_>>()?;
        for (a, b) in codes[0]
            .chunks_exact(self.outputs[0].0.cols)
            .zip(codes[1].chunks_exact(self.outputs[1].0.cols))
        {
            ensure!(
                a[..self.logical_words] == b[..self.logical_words],
                "packed codes differ"
            );
        }
        let a = self.engine.read_f32(&self.outputs[0].1)?;
        let b = self.engine.read_f32(&self.outputs[1].1)?;
        ensure!(
            a.iter()
                .zip(b)
                .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits()),
            "packing scales differ"
        );
        Ok(())
    }
}

impl GemmBenchmark {
    /// Rotate and pack activations on every timed replay; prepare weights once.
    pub fn rotated_integer(
        m: usize,
        k: usize,
        n: usize,
        source: &str,
        bits: usize,
        padded: bool,
        minimum_weight_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            [4, 8].contains(&bits)
                && (1..=4096).contains(&m)
                && (256..=17408).contains(&k)
                && k.is_multiple_of(256)
                && (128..=17408).contains(&n)
                && n.is_multiple_of(128)
                && minimum_weight_bytes <= 512 << 20,
            "unsupported rotated integer GEMM shape"
        );
        let context = ModelContext::new(Default::default())?;
        let mut engine = Engine::new(&context)?;
        let stride = k + if padded && k.is_multiple_of(1024) {
            512 / bits
        } else {
            0
        };
        let values = |len| {
            (0..len)
                .map(|i| half::bf16::from_f32(super::gemm_value(i, 16)))
                .collect::<Vec<_>>()
        };
        let x = engine.input(m, k, DType::Bf16, bytemuck::cast_slice(&values(m * k)))?;
        let dense_w = engine.input(n, k, DType::Bf16, bytemuck::cast_slice(&values(n * k)))?;
        let copies = minimum_weight_bytes
            .div_ceil(n * (stride * bits / 8 + 4))
            .max(1);
        let mut graphs = Vec::with_capacity(copies);
        let mut first_output = None;
        for _ in 0..copies {
            let (w, ws) = prepare(&mut engine, &dense_w, bits, stride)?;
            // Each rotating graph needs its own immutable packed weight buffer.
            // Ordinary model weights are leased by QuantizedProjections instead.
            engine.retained.push(w.clone());
            let mut prepack = engine.finish()?;
            engine.run(&mut prepack)?;
            let (a, a_scale) = prepare(&mut engine, &x, bits, stride)?;
            let output = engine.alloc(m, n, DType::Bf16)?;
            let mut spec = Specialization::new(format!("h3_gemm_i{bits}_256"));
            for (key, value) in [
                ("k_size", k),
                ("n_size", n),
                ("k_stride", stride),
                ("m_group", m.div_ceil(256).min(4)),
            ] {
                spec.set_config(format!("h3.gemm_i{bits}_256.{key}"), value.to_string());
            }
            let mut constants = Constants::new();
            constants.push(m as u32)?;
            engine.emit_special(
                source,
                spec,
                constants,
                &[m as u64],
                &[&a, &w, &ws, &a_scale],
                &output,
            )?;
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
}
