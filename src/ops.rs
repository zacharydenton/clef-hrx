use crate::{
    Result,
    gpu::{DType, Engine, Tensor},
};
use anyhow::ensure;
use hrx::{Constants, loom::Specialization};

impl Engine {
    pub fn unary(&mut self, x: &Tensor, op: &str) -> Result<Tensor> {
        let out = self.alloc(x.rows, x.cols, x.dtype)?;
        let source = match op {
            "silu" => concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/silu.loom")
            ),
            "sigmoid" => concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/sigmoid.loom")
            ),
            "gelu" => concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/gelu.loom")
            ),
            "gelu_tanh" => concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/gelu_tanh.loom")
            ),
            _ => anyhow::bail!("unknown activation {op}"),
        };
        self.dispatch(source, out.len(), &[("x", x)], &out, &[])?;
        Ok(out)
    }
    pub fn binary(&mut self, a: &Tensor, b: &Tensor, op: &str) -> Result<Tensor> {
        ensure!(
            a.cols == b.cols && (b.rows == 1 || b.rows == a.rows),
            "binary shape mismatch"
        );
        let out = self.alloc(a.rows, a.cols, a.dtype)?;
        let source = match op {
            "addf" => concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/addf.loom")
            ),
            "mulf" => concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/mulf.loom")
            ),
            "subf" => concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/subf.loom")
            ),
            _ => anyhow::bail!("unknown binary operation {op}"),
        };
        self.dispatch(source, out.len(), &[("a", a), ("b", b)], &out, &[])?;
        Ok(out)
    }
    pub fn cast(&mut self, x: &Tensor, dtype: DType) -> Result<Tensor> {
        if dtype == x.dtype {
            return Ok(x.clone());
        }
        let out = self.alloc(x.rows, x.cols, dtype)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/cast.loom")
            ),
            out.len(),
            &[("x", x)],
            &out,
            &[],
        )?;
        Ok(out)
    }
    pub fn linear(&mut self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        ensure!(
            x.cols == w.cols && x.dtype == DType::Bf16 && w.dtype == DType::Bf16,
            "linear shape/dtype mismatch"
        );
        let (m, k, n) = (x.rows, x.cols, w.rows);
        let out = self.alloc(m, n, DType::Bf16)?;
        let symbol = "krea2_gemm_bf16_bf16_nt";
        let mut spec = Specialization::new(symbol);
        for (key, value) in [
            ("m", m),
            ("n", n),
            ("k", k),
            ("asize", m * k),
            ("bsize", n * k),
            ("csize", m * n),
            ("astride", m * k),
            ("bstride", n * k),
            ("grid_x", n.div_ceil(64)),
            ("grid_y", m.div_ceil(64)),
        ] {
            spec.set_config(format!("krea2.gemm_bf16_bf16_nt.{key}"), value.to_string());
        }
        let mut constants = Constants::new();
        constants.push((m * n) as u32)?;
        constants.push(1.0f32)?;
        self.emit_special(
            include_str!("../kernels/gemm_bf16.loom"),
            spec,
            constants,
            [n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1],
            &[x, w],
            &out,
        )?;
        if let Some(b) = bias {
            self.binary(&out, &b.reshape(1, n), "addf")
        } else {
            Ok(out)
        }
    }
    pub fn project(&mut self, x: &Tensor, name: &str, bias: bool) -> Result<Tensor> {
        let w = self.w(&format!("{name}.weight"))?;
        let b = if bias {
            Some(self.w(&format!("{name}.bias"))?)
        } else {
            None
        };
        self.linear(x, &w, b.as_ref())
    }
    pub fn norm(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        b: Option<&Tensor>,
        eps: f32,
        offset: bool,
    ) -> Result<Tensor> {
        self.norm_impl::<true>(x, w, b, eps, offset)
    }
    pub(crate) fn norm_impl<const PARALLEL: bool>(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        b: Option<&Tensor>,
        eps: f32,
        offset: bool,
    ) -> Result<Tensor> {
        let width = x.cols;
        ensure!(
            w.len() == width && b.is_none_or(|b| b.len() == width),
            "norm dimensions"
        );
        let out = self.alloc(x.rows, width, x.dtype)?;
        self.dispatch(
            if PARALLEL {
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/norm.loom")
                )
            } else {
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/norm_reference.loom")
                )
            },
            x.rows * if PARALLEL { 32 } else { 1 },
            &[("x", x), ("w", w), ("b", b.unwrap_or(w))],
            &out,
            &[
                ("width", width.to_string()),
                ("width_float", format!("{width}.0")),
                ("eps", format!("{eps:e}")),
                ("layer", usize::from(b.is_some()).to_string()),
                ("weight_add", if offset { "1.0" } else { "0.0" }.into()),
            ],
        )?;
        Ok(out)
    }
    pub fn named_norm(&mut self, x: &Tensor, name: &str, layer: bool, eps: f32) -> Result<Tensor> {
        let w = self.w(&format!("{name}.weight"))?;
        let b = if layer {
            Some(self.w(&format!("{name}.bias"))?)
        } else {
            None
        };
        self.norm(x, &w, b.as_ref(), eps, !layer)
    }
    /// Contiguous columns from every row; supports interleaved attention projections.
    pub fn columns(&mut self, x: &Tensor, start: usize, cols: usize) -> Result<Tensor> {
        ensure!(start + cols <= x.cols, "column slice");
        let out = self.alloc(x.rows, cols, x.dtype)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/columns.loom")
            ),
            out.len(),
            &[("x", x)],
            &out,
            &[
                ("cols", (cols).to_string()),
                ("p0", (x.cols).to_string()),
                ("start", (start).to_string()),
            ],
        )?;
        Ok(out)
    }
    pub fn gather(&mut self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let cols = x.cols;
        let out = self.alloc(ids.len(), cols, x.dtype)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/gather.loom")
            ),
            out.len(),
            &[("x", x), ("ids", ids)],
            &out,
            &[("cols", (cols).to_string())],
        )?;
        Ok(out)
    }
    pub fn pool_spans(&mut self, x: &Tensor, spans: &[[usize; 2]]) -> Result<Tensor> {
        let ids: Vec<u32> = spans.iter().flatten().map(|v| *v as u32).collect();
        let index = self.input(spans.len(), 2, DType::U32, bytemuck::cast_slice(&ids))?;
        let out = self.alloc(spans.len(), x.cols, x.dtype)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/pool_spans.loom")
            ),
            out.len(),
            &[("x", x), ("spans", &index)],
            &out,
            &[("cols", (x.cols).to_string())],
        )?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires gfx1151; validates parallel normalization against an FP64 oracle"]
    fn parallel_norm_accuracy_and_tails() -> Result<()> {
        let context = hrx::inference::ModelContext::new(Default::default())?;
        let mut engine = Engine::new(&context)?;
        for dtype in [DType::Bf16, DType::F32] {
            let round = |x: f32| match dtype {
                DType::Bf16 => half::bf16::from_f32(x).to_f32(),
                _ => x,
            };
            for width in [1, 7, 32, 33, 128, 256, 1024, 1280, 5120] {
                let rows = 5;
                let values: Vec<_> = (0..rows * width)
                    .map(|i| {
                        let value = ((i * 17 % 251) as f32 - 125.) / 16.;
                        round(match i / width {
                            0 => 0.,
                            1 => 2.,
                            2 => value,
                            3 => 128. + (i % 3) as f32,
                            _ => value * 1e-4,
                        })
                    })
                    .collect();
                let weights: Vec<_> = (0..width)
                    .map(|i| round(((i * 7 % 67) as f32 - 33.) / 64.))
                    .collect();
                let biases: Vec<_> = (0..width).map(|i| round((i % 11) as f32 / 32.)).collect();
                let mut input = |rows, cols, values: &[f32]| {
                    if dtype == DType::Bf16 {
                        let bytes: Vec<_> =
                            values.iter().map(|&x| half::bf16::from_f32(x)).collect();
                        engine.input(rows, cols, dtype, bytemuck::cast_slice(&bytes))
                    } else {
                        engine.input(rows, cols, dtype, bytemuck::cast_slice(values))
                    }
                };
                let x = input(rows, width, &values)?;
                let w = input(1, width, &weights)?;
                let b = input(1, width, &biases)?;
                for (layer, offset) in [(false, false), (false, true), (true, false)] {
                    let eps = if layer { 1e-5 } else { 1e-6 };
                    let out = engine.norm(&x, &w, layer.then_some(&b), eps, offset)?;
                    let mut graph = engine.finish()?;
                    engine.run(&mut graph)?;
                    let actual = engine.read_f32(&out)?;
                    for row in 0..rows {
                        let v = &values[row * width..(row + 1) * width];
                        let mean = if layer {
                            v.iter().map(|&x| x as f64).sum::<f64>() / width as f64
                        } else {
                            0.
                        };
                        let variance = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>()
                            / width as f64;
                        for col in 0..width {
                            let weight = weights[col] as f64 + if offset { 1. } else { 0. };
                            let bias = if layer { biases[col] as f64 } else { 0. };
                            let expected = (v[col] as f64 - mean) / (variance + eps as f64).sqrt()
                                * weight
                                + bias;
                            let actual = actual[row * width + col] as f64;
                            let relative = if dtype == DType::Bf16 {
                                1. / 256.
                            } else {
                                5e-5
                            };
                            assert!(
                                actual.is_finite()
                                    && (actual - expected).abs()
                                        <= expected.abs() * relative + 2e-5,
                                "{dtype:?} width={width} row={row} col={col} layer={layer} offset={offset}: {actual} vs {expected}"
                            );
                        }
                    }
                }
                engine.reset_plan()?;
            }
        }
        Ok(())
    }
}
