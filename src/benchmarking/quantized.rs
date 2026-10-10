//! Benchmark harness for the native integer path.
use super::*;
use crate::quantized::prepare;
use anyhow::ensure;
use hrx::{Constants, loom::Specialization};

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
