use super::*;

/// Matched named RMSNorm and integer packing, with independent rotating inputs.
pub struct NormPackingBenchmark {
    graphs: [Vec<GraphExec>; 2],
    outputs: [Vec<(Tensor, Tensor)>; 2],
    norm_outputs: [Vec<Tensor>; 2],
    expected_norm: Vec<u32>,
    candidate_norm: Vec<u32>,
    expected: (Vec<u32>, Vec<u32>),
    candidate_expected: (Vec<u32>, Vec<u32>),
    _intermediates: Vec<Tensor>,
    next: [usize; 2],
    engine: Engine,
    _context: ModelContext,
}

impl NormPackingBenchmark {
    pub fn new(rows: usize, source: &str, rotating: bool) -> Result<Self> {
        anyhow::ensure!((1..=4032).contains(&rows));
        let store_norm = std::env::var("CLEF_BENCH_NORM_PACK_STORE").as_deref() == Ok("1");
        let width = 5120;
        let stride = width + 128;
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
        let weights: Vec<_> = (0..2 * width)
            .map(|i| half::bf16::from_f32(((i * 31 % 127) as f32 - 63.) / 128.))
            .collect();
        let mut intermediates = Vec::new();
        let mut graphs = [Vec::new(), Vec::new()];
        let mut outputs = [Vec::new(), Vec::new()];
        let mut norm_outputs = [Vec::new(), Vec::new()];
        for variant in 0..2 {
            for _ in 0..copies {
                let x = engine
                    .input(rows + 1, width, DType::Bf16, bytemuck::cast_slice(&values))?
                    .slice_rows(1, rows);
                let w = engine
                    .input(2, width, DType::Bf16, bytemuck::cast_slice(&weights))?
                    .slice_rows(1, 1);
                let out = if variant == 0 {
                    let normalized = engine.norm(&x, &w, None, 1e-6, true)?;
                    let result = crate::quantized::prepare(&mut engine, &normalized, 4, stride)?;
                    if store_norm {
                        norm_outputs[variant].push(normalized.clone());
                    }
                    intermediates.push(normalized);
                    result
                } else if store_norm {
                    let (codes, scales, normalized) =
                        crate::quantized::prepare_norm_store_with_source(
                            &mut engine,
                            &x,
                            &w,
                            4,
                            stride,
                            source,
                        )?;
                    norm_outputs[variant].push(normalized);
                    (codes, scales)
                } else {
                    crate::quantized::prepare_norm_with_source(
                        &mut engine,
                        &x,
                        &w,
                        4,
                        stride,
                        source,
                    )?
                };
                let mut graph = engine.finish()?;
                engine.run(&mut graph)?;
                graphs[variant].push(graph);
                outputs[variant].push(out);
            }
        }
        anyhow::ensure!(
            engine.allocated_bytes() <= 1 << 30,
            "paired normalization packing exceeds 1 GiB"
        );
        let expected = Self::read(&mut engine, &outputs[0][0])?;
        let candidate_expected = Self::read(&mut engine, &outputs[1][0])?;
        let expected_norm = if store_norm {
            Self::read_norm(&mut engine, &norm_outputs[0][0])?
        } else {
            Vec::new()
        };
        let candidate_norm = if store_norm {
            Self::read_norm(&mut engine, &norm_outputs[1][0])?
        } else {
            Vec::new()
        };
        let mut result = Self {
            graphs,
            outputs,
            norm_outputs,
            expected_norm,
            candidate_norm,
            expected,
            candidate_expected,
            _intermediates: intermediates,
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

    fn read(engine: &mut Engine, output: &(Tensor, Tensor)) -> Result<(Vec<u32>, Vec<u32>)> {
        let codes = engine.read_f32(&output.0)?;
        let logical = codes
            .chunks_exact(output.0.cols)
            .flat_map(|row| row[..5120 / 8].iter().map(|x| x.to_bits()))
            .collect();
        let scales = engine.read_f32(&output.1)?;
        anyhow::ensure!(
            scales.iter().all(|v| v.is_finite()),
            "nonfinite benchmark scales"
        );
        Ok((logical, scales.into_iter().map(f32::to_bits).collect()))
    }

    fn read_norm(engine: &mut Engine, output: &Tensor) -> Result<Vec<u32>> {
        let values = engine.read_f32(output)?;
        anyhow::ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite normalized benchmark output"
        );
        Ok(values.into_iter().map(f32::to_bits).collect())
    }

    pub fn differences(&self) -> Result<serde_json::Value> {
        let mut max_code_change = 0i32;
        let mut changed_codes = 0usize;
        let mut total_codes = 0usize;
        let mut changed_scales = 0usize;
        let mut max_scale_relative_error = 0f32;
        {
            let actual = &self.candidate_expected;
            for (a, b) in actual.0.iter().zip(&self.expected.0) {
                for lane in 0..8 {
                    let code = |word: u32| ((word << (28 - lane * 4)) as i32) >> 28;
                    let delta = (code(*a) - code(*b)).abs();
                    max_code_change = max_code_change.max(delta);
                    changed_codes += usize::from(delta != 0);
                    total_codes += 1;
                }
            }
            for (a, b) in actual.1.iter().zip(&self.expected.1) {
                changed_scales += usize::from(a != b);
                let a = f32::from_bits(*a);
                let b = f32::from_bits(*b);
                max_scale_relative_error =
                    max_scale_relative_error.max((a - b).abs() / b.abs().max(1e-30));
            }
        }
        let mut changed_normalized = 0usize;
        let mut max_normalized_bf16_ulp = 0u32;
        for (a, b) in self.candidate_norm.iter().zip(&self.expected_norm) {
            if f32::from_bits(*a) == f32::from_bits(*b) {
                continue;
            }
            changed_normalized += 1;
            let ordered = |bits: u32| {
                let v = bits >> 16;
                if v & 0x8000 != 0 {
                    (!v) & 0xffff
                } else {
                    v | 0x8000
                }
            };
            max_normalized_bf16_ulp =
                max_normalized_bf16_ulp.max(ordered(*a).abs_diff(ordered(*b)));
        }
        Ok(serde_json::json!({
            "bitwise_identical": changed_codes == 0 && changed_scales == 0 && self.candidate_norm == self.expected_norm,
            "changed_normalized": changed_normalized, "max_normalized_bf16_ulp": max_normalized_bf16_ulp,
            "changed_normalized_fraction": changed_normalized as f64 / self.expected_norm.len().max(1) as f64,
            "max_code_change": max_code_change, "changed_codes": changed_codes,
            "total_codes": total_codes, "changed_code_fraction": changed_codes as f64 / total_codes as f64,
            "changed_scales": changed_scales, "max_scale_relative_error": max_scale_relative_error,
        }))
    }

    pub fn check(&mut self) -> Result<()> {
        for output in &self.outputs[0] {
            anyhow::ensure!(
                Self::read(&mut self.engine, output)? == self.expected,
                "reference normalized codes or scales changed"
            );
        }
        for output in &self.outputs[1] {
            anyhow::ensure!(
                Self::read(&mut self.engine, output)? == self.candidate_expected,
                "candidate normalized codes or scales changed across replays"
            );
        }
        for (variant, expected) in [&self.expected_norm, &self.candidate_norm]
            .into_iter()
            .enumerate()
        {
            for output in &self.norm_outputs[variant] {
                anyhow::ensure!(
                    Self::read_norm(&mut self.engine, output)? == *expected,
                    "normalized BF16 output changed across replays"
                );
            }
        }
        let d = self.differences()?;
        if std::env::var("CLEF_BENCH_NORM_PACK_NUMERIC").as_deref() == Ok("1") {
            anyhow::ensure!(
                d["max_code_change"].as_i64().unwrap() <= 1
                    && d["max_normalized_bf16_ulp"].as_u64().unwrap() <= 1
                    && d["changed_normalized_fraction"].as_f64().unwrap() <= 0.01
                    && d["changed_code_fraction"].as_f64().unwrap() <= 0.01
                    && d["max_scale_relative_error"].as_f64().unwrap() <= 0.004,
                "normalized packing exceeds numeric bounds: {d}"
            );
        } else {
            anyhow::ensure!(
                d["bitwise_identical"] == true,
                "normalized codes or scales changed: {d}"
            );
        }
        Ok(())
    }
}
