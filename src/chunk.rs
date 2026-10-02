//! Chunkwise Gated DeltaNet prefill, following the pinned Transformers FP32
//! triangular-solve formulation. State crosses chunks; token work inside each
//! chunk is parallel. The recurrent kernel remains an independent oracle.
use crate::{
    Result,
    gpu::{DType, Engine, Tensor},
};

impl Engine {
    fn zero_f32(&mut self, rows: usize, cols: usize) -> Result<Tensor> {
        let out = self.alloc(rows, cols, DType::F32)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/zero.loom")
            ),
            out.len(),
            &[],
            &out,
            &[],
        )?;
        Ok(out)
    }
    fn bmm_f32<const OPTIMIZED: bool>(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            a.len() == batch * m * k && b.len() == batch * k * n && (1..=64).contains(&m),
            "chunk contraction shape"
        );
        let out = self.alloc(batch * m, n, DType::F32)?;
        let source = concat!(
            include_str!("../kernels/common.loom"),
            include_str!("../kernels/bmm_f32.loom")
        );
        #[cfg(any(test, feature = "bench-internals"))]
        let source = if OPTIMIZED {
            source
        } else {
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/bmm_f32_reference.loom")
            )
        };
        let mut parameters = vec![
            ("k", k.to_string()),
            ("kn", (k * n).to_string()),
            ("mk", (m * k).to_string()),
            ("mn", (m * n).to_string()),
            ("n", n.to_string()),
        ];
        if OPTIMIZED {
            parameters.extend([
                ("m", m.to_string()),
                ("tilesn", (m.div_ceil(4) * n).to_string()),
            ]);
        }
        self.dispatch(
            source,
            batch * if OPTIMIZED { m.div_ceil(4) } else { m } * n,
            &[("a", a), ("b", b)],
            &out,
            &parameters,
        )?;
        Ok(out)
    }
    fn chunk_prepare<const OPTIMIZED: bool>(
        &mut self,
        qkv: &Tensor,
        a: &Tensor,
        b: &Tensor,
        al: &Tensor,
        dt: &Tensor,
    ) -> Result<Tensor> {
        let n = qkv.rows;
        let width = 641;
        let out = self.alloc(48 * n, width, DType::F32)?;
        let source = concat!(
            include_str!("../kernels/common.loom"),
            include_str!("../kernels/chunk_prepare.loom")
        );
        #[cfg(any(test, feature = "bench-internals"))]
        let source = if OPTIMIZED {
            source
        } else {
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/chunk_prepare_reference.loom")
            )
        };
        self.dispatch(
            source,
            48 * n * if OPTIMIZED { 32 } else { 1 },
            &[("qkv", qkv), ("a", a), ("b", b), ("al", al), ("dt", dt)],
            &out,
            &[("n", (n).to_string())],
        )?;
        Ok(out)
    }
    fn chunk_pair(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        kbeta: &Tensor,
        g: &Tensor,
        n: usize,
    ) -> Result<Tensor> {
        let out = self.alloc(48 * n, n * 2, DType::F32)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/chunk_pair.loom")
            ),
            48 * n * n,
            &[("q", q), ("k", k), ("b", kbeta), ("g", g)],
            &out,
            &[
                ("n", (n).to_string()),
                ("nn", (n * n).to_string()),
                ("twon", (n * 2).to_string()),
            ],
        )?;
        Ok(out)
    }
    fn triangular_inverse<const OPTIMIZED: bool>(
        &mut self,
        l: &Tensor,
        n: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            (1..=64).contains(&n),
            "triangular solve chunk must be 1..=64"
        );
        let out = self.alloc(48 * n, n, DType::F32)?;
        let source = concat!(
            include_str!("../kernels/common.loom"),
            include_str!("../kernels/triangular_inverse.loom")
        );
        #[cfg(any(test, feature = "bench-internals"))]
        let source = if OPTIMIZED {
            source
        } else {
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/triangular_inverse_reference.loom")
            )
        };
        self.dispatch(
            source,
            48 * if OPTIMIZED { 64 } else { n },
            &[("l", l)],
            &out,
            &[("n", (n).to_string()), ("nn", (n * n).to_string())],
        )?;
        Ok(out)
    }
    fn scale_rows_exp(&mut self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        let out = self.alloc(x.rows, x.cols, DType::F32)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/scale_rows_exp.loom")
            ),
            out.len(),
            &[("x", x), ("g", g)],
            &out,
            &[("p0", (x.cols).to_string())],
        )?;
        Ok(out)
    }
    fn chunk_state(
        &mut self,
        state: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        n: usize,
    ) -> Result<Tensor> {
        let out = self.alloc(48 * 128, 128, DType::F32)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/chunk_state.loom")
            ),
            out.len(),
            &[("state", state), ("k", k), ("v", v), ("g", g)],
            &out,
            &[("last", (n - 1).to_string()), ("n", (n).to_string())],
        )?;
        Ok(out)
    }
    pub fn delta_chunked(
        &mut self,
        qkv: &Tensor,
        a: &Tensor,
        b: &Tensor,
        al: &Tensor,
        dt: &Tensor,
    ) -> Result<Tensor> {
        self.delta_chunked_impl::<true>(qkv, a, b, al, dt)
    }
    pub(crate) fn delta_chunked_impl<const OPTIMIZED: bool>(
        &mut self,
        qkv: &Tensor,
        a: &Tensor,
        b: &Tensor,
        al: &Tensor,
        dt: &Tensor,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            OPTIMIZED || cfg!(any(test, feature = "bench-internals")),
            "reference kernels require tests or bench-internals"
        );
        let output = self.alloc(qkv.rows, 6144, DType::Bf16)?;
        let mut state = self.zero_f32(48 * 128, 128)?;
        for start in (0..qkv.rows).step_by(64) {
            let n = (qkv.rows - start).min(64);
            let prep = self.chunk_prepare::<OPTIMIZED>(
                &qkv.slice_rows(start, n),
                &a.slice_rows(start, n),
                &b.slice_rows(start, n),
                al,
                dt,
            )?;
            let q = self.columns(&prep, 0, 128)?;
            let k = self.columns(&prep, 128, 128)?;
            let vb = self.columns(&prep, 256, 128)?;
            let kb = self.columns(&prep, 384, 128)?;
            let ke = self.columns(&prep, 512, 128)?;
            let g = self.columns(&prep, 640, 1)?;
            let pair = self.chunk_pair(&q, &k, &kb, &g, n)?;
            let l = self.columns(&pair, 0, n)?;
            let gamma = self.columns(&pair, n, n)?;
            let inv = self.triangular_inverse::<OPTIMIZED>(&l, n)?;
            let vt = self.bmm_f32::<OPTIMIZED>(&inv, &vb, 48, n, n, 128)?;
            let kt = self.bmm_f32::<OPTIMIZED>(&inv, &ke, 48, n, n, 128)?;
            let vp = self.bmm_f32::<OPTIMIZED>(&kt, &state, 48, n, 128, 128)?;
            let vn = self.binary(&vt, &vp, "subf")?;
            let qe = self.scale_rows_exp(&q, &g)?;
            let inter = self.bmm_f32::<OPTIMIZED>(&qe, &state, 48, n, 128, 128)?;
            let intra = self.bmm_f32::<OPTIMIZED>(&gamma, &vn, 48, n, n, 128)?;
            let y = self.binary(&inter, &intra, "addf")?;
            state = self.chunk_state(&state, &k, &vn, &g, n)?;
            let target = output.slice_rows(start, n);
            self.dispatch(
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/chunk_reorder.loom")
                ),
                target.len(),
                &[("x", &y)],
                &target,
                &[("n", (n).to_string())],
            )?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires gfx1151; checks tiled matmul row/column tails"]
    fn tiled_matmul_tails() -> Result<()> {
        let mut e = Engine::new(&hrx::inference::ModelContext::new(Default::default())?)?;
        for (m, k, n) in [
            (1, 7, 3),
            (2, 13, 37),
            (3, 64, 128),
            (4, 128, 17),
            (5, 3, 65),
            (63, 128, 128),
            (64, 64, 128),
        ] {
            let av: Vec<_> = (0..2 * m * k)
                .map(|i| ((i * 13 % 101) as f32 - 50.) / 128.)
                .collect();
            let bv: Vec<_> = (0..2 * k * n)
                .map(|i| ((i * 17 % 97) as f32 - 48.) / 128.)
                .collect();
            let a = e.input(2 * m, k, DType::F32, bytemuck::cast_slice(&av))?;
            let b = e.input(2 * k, n, DType::F32, bytemuck::cast_slice(&bv))?;
            let out = e.bmm_f32::<true>(&a, &b, 2, m, k, n)?;
            let mut graph = e.finish()?;
            e.run(&mut graph)?;
            let values = e.read_f32(&out)?;
            for batch in 0..2 {
                for row in 0..m {
                    for col in 0..n {
                        let expected = (0..k).fold(0f32, |acc, j| {
                            av[(batch * m + row) * k + j]
                                .mul_add(bv[(batch * k + j) * n + col], acc)
                        });
                        let actual = values[(batch * m + row) * n + col];
                        assert_eq!(actual, expected, "{m}x{k}x{n}, ({batch},{row},{col})");
                    }
                }
            }
            e.reset_plan()?;
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires gfx1151; checks shared-memory solve tails and inverse residual"]
    fn triangular_solve_tails() -> Result<()> {
        let mut e = Engine::new(&hrx::inference::ModelContext::new(Default::default())?)?;
        for n in [1, 2, 3, 5, 31, 63, 64] {
            let values: Vec<_> = (0..48 * n * n)
                .map(|i| {
                    let row = i / n % n;
                    let col = i % n;
                    if col < row {
                        ((i * 7 % 31) as f32 - 15.) / 256.
                    } else {
                        0.
                    }
                })
                .collect();
            let l = e.input(48 * n, n, DType::F32, bytemuck::cast_slice(&values))?;
            let reference = e.triangular_inverse::<false>(&l, n)?;
            let actual = e.triangular_inverse::<true>(&l, n)?;
            let mut graph = e.finish()?;
            e.run(&mut graph)?;
            let actual = e.read_f32(&actual)?;
            assert_eq!(actual, e.read_f32(&reference)?);
            for h in 0..48 {
                for row in 0..n {
                    for col in 0..n {
                        let residual = (0..n).fold(0f64, |acc, j| {
                            acc + ((if row == j { 1. } else { 0. })
                                - values[(h * n + row) * n + j] as f64)
                                * actual[(h * n + j) * n + col] as f64
                        });
                        let expected = if row == col { 1. } else { 0. };
                        assert!((residual - expected).abs() < 1e-6);
                    }
                }
            }
            e.reset_plan()?;
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires gfx1151; checks FP32 normalization against an independent FP64 oracle"]
    fn normalization_fp64_accuracy() -> Result<()> {
        let mut e = Engine::new(&hrx::inference::ModelContext::new(Default::default())?)?;
        let n = 3;
        let values: Vec<_> = (0..n * 10240)
            .map(|i| half::bf16::from_f32(((i * 13 % 251) as f32 - 125.) / 32.))
            .collect();
        let qkv = e.input(n, 10240, DType::Bf16, bytemuck::cast_slice(&values))?;
        let ab = e.input(
            n,
            48,
            DType::Bf16,
            bytemuck::cast_slice(&vec![half::bf16::ZERO; n * 48]),
        )?;
        let al = e.input(
            1,
            48,
            DType::Bf16,
            bytemuck::cast_slice(&[half::bf16::ZERO; 48]),
        )?;
        let prepared = e.chunk_prepare::<true>(&qkv, &ab, &ab, &al, &al)?;
        let mut graph = e.finish()?;
        e.run(&mut graph)?;
        let actual = e.read_f32(&prepared)?;
        let bf = |x: f64| half::bf16::from_f64(x).to_f64();
        let mut native_error = 0.;
        let mut fallback_error = 0.;
        for head in 0..48 {
            for token in 0..n {
                for (offset, output_offset, scale) in [(0, 0, 1. / 128f64.sqrt()), (2048, 128, 1.)]
                {
                    let start = token * 10240 + (head / 3) * 128 + offset;
                    let row = &values[start..start + 128];
                    let sum: f64 = row.iter().map(|v| v.to_f64().powi(2)).sum();
                    let inv = (sum + 1e-6).sqrt().recip();
                    // BF16 CPU fallback rounds squares, sum, epsilon, rsqrt, and product.
                    let low_sum = bf(row.iter().map(|v| bf(v.to_f64().powi(2))).sum());
                    let low_inv = bf(bf(low_sum + 1e-6).sqrt().recip());
                    for (j, value) in row.iter().enumerate() {
                        let expected = value.to_f64() * inv * scale;
                        let native = actual[(head * n + token) * 641 + output_offset + j] as f64;
                        let fallback = bf(value.to_f64() * low_inv) * scale;
                        assert!((native - expected).abs() < 1e-6);
                        native_error += (native - expected).powi(2);
                        fallback_error += (fallback - expected).powi(2);
                    }
                }
            }
        }
        eprintln!(
            "normalization FP64 squared error: native={native_error:e}, BF16 fallback={fallback_error:e}"
        );
        assert!(native_error < fallback_error / 10000.);
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151; compares chunked versus recurrent prefill"]
    fn prefill_comparison() -> Result<()> {
        let mut e = Engine::new(&hrx::inference::ModelContext::new(Default::default())?)?;
        for n in [65, 260, 1024] {
            let qkv: Vec<_> = (0..n * 10240)
                .map(|i| half::bf16::from_f32(((i * 17 % 251) as f32 - 125.) / 512.))
                .collect();
            let ab: Vec<_> = (0..n * 48)
                .map(|i| half::bf16::from_f32(((i * 13 % 131) as f32 - 65.) / 256.))
                .collect();
            let qkv = e.input(n, 10240, DType::Bf16, bytemuck::cast_slice(&qkv))?;
            let a = e.input(n, 48, DType::Bf16, bytemuck::cast_slice(&ab))?;
            let b = a.clone();
            let al = e.input(
                1,
                48,
                DType::Bf16,
                bytemuck::cast_slice(&[half::bf16::ZERO; 48]),
            )?;
            let dt = al.clone();
            let r = e.delta(&qkv, &a, &b, &al, &dt)?;
            let mut rg = e.finish()?;
            let c = e.delta_chunked(&qkv, &a, &b, &al, &dt)?;
            let mut cg = e.finish()?;
            e.run(&mut rg)?;
            let expected = e.read_f32(&r)?;
            e.run(&mut cg)?;
            let actual = e.read_f32(&c)?;
            assert_eq!(actual.len(), expected.len());
            assert!(actual.iter().chain(&expected).all(|v| v.is_finite()));
            e.run(&mut cg)?;
            assert_eq!(actual, e.read_f32(&c)?, "chunk state must reset on replay");
            let error = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            assert!(error < 0.002);
        }
        Ok(())
    }
}
