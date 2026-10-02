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
        let width = x.cols;
        ensure!(
            w.len() == width && b.is_none_or(|b| b.len() == width),
            "norm dimensions"
        );
        let out = self.alloc(x.rows, width, x.dtype)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/norm.loom")
            ),
            x.rows,
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
