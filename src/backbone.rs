use crate::{
    Result,
    gpu::{DType, Engine, Tensor},
};
use anyhow::ensure;
use hrx::{Constants, loom::Specialization};

impl Engine {
    /// Online softmax attention with bounded memory and tiled long-sequence execution.
    pub fn attention(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        heads: usize,
        kv_heads: usize,
        causal: bool,
    ) -> Result<Tensor> {
        #[cfg(feature = "bench-internals")]
        if self.reference_kernels.attention {
            return self.attention_impl::<false>(q, k, v, heads, kv_heads, causal);
        }
        self.attention_impl::<true>(q, k, v, heads, kv_heads, causal)
    }
    pub(crate) fn attention_impl<const TILED: bool>(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        heads: usize,
        kv_heads: usize,
        causal: bool,
    ) -> Result<Tensor> {
        ensure!(
            heads > 0
                && kv_heads > 0
                && q.rows > 0
                && k.rows > 0
                && q.cols.is_multiple_of(heads)
                && k.cols.is_multiple_of(kv_heads),
            "attention layout"
        );
        let d = q.cols / heads;
        ensure!(
            d == k.cols / kv_heads
                && k.rows == v.rows
                && k.cols == v.cols
                && heads.is_multiple_of(kv_heads),
            "attention shape"
        );
        if TILED
            && d == 256
            && (1024..=16384).contains(&q.rows)
            && k.rows <= 16384
            && (!causal || q.rows == k.rows)
            && q.dtype == DType::Bf16
            && k.dtype == DType::Bf16
            && v.dtype == DType::Bf16
        {
            return self.attention_tiled(q, k, v, heads, kv_heads, causal);
        }
        let out = self.alloc(q.rows, q.cols, DType::Bf16)?;
        let source = match d {
            64 => concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/attention_64.loom")
            ),
            72 => concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/attention_72.loom")
            ),
            256 => concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/attention_256.loom")
            ),
            1024 => concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/attention_1024.loom")
            ),
            _ => anyhow::bail!("unsupported attention head dimension {d}"),
        };
        self.dispatch(
            source,
            q.rows * heads * 32,
            &[("q", q), ("k", k), ("v", v)],
            &out,
            &[
                ("heads", heads.to_string()),
                ("ratio", (heads / kv_heads).to_string()),
                ("qw", q.cols.to_string()),
                ("kw", k.cols.to_string()),
                ("scale", format!("{:e}", 1.0 / (d as f32).sqrt())),
                ("causal", usize::from(causal).to_string()),
                ("keys", k.rows.to_string()),
            ],
        )?;
        Ok(out)
    }
    pub(crate) fn attention_tiled(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        heads: usize,
        kv_heads: usize,
        causal: bool,
    ) -> Result<Tensor> {
        ensure!(
            q.cols == heads * 256 && k.cols == kv_heads * 256,
            "tiled head dimensions"
        );
        self.attention_tiled_source(
            q,
            k,
            v,
            causal,
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/attention_tiled.loom")
            ),
        )
    }
    pub(crate) fn attention_tiled_source(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
        source: &str,
    ) -> Result<Tensor> {
        ensure!(
            (1..=16384).contains(&q.rows)
                && (1..=16384).contains(&k.rows)
                && (256..=65536).contains(&q.cols)
                && (256..=65536).contains(&k.cols)
                && q.cols.is_multiple_of(256)
                && k.cols.is_multiple_of(256)
                && q.cols.is_multiple_of(k.cols)
                && k.rows == v.rows
                && k.cols == v.cols
                && (!causal || q.rows == k.rows)
                && q.dtype == DType::Bf16
                && k.dtype == DType::Bf16
                && v.dtype == DType::Bf16,
            "tiled attention shape or storage"
        );
        let heads = q.cols / 256;
        let kv_heads = k.cols / 256;
        let packed = self.transpose_attention_v(v)?;
        let out = self.alloc(q.rows, q.cols, DType::Bf16)?;
        let mut spec = Specialization::new("clef_op");
        for (key, value) in [
            ("heads", heads.to_string()),
            ("ratio", (heads / kv_heads).to_string()),
            ("qw", q.cols.to_string()),
            ("kw", k.cols.to_string()),
            ("scale", "6.25e-2".into()),
            ("causal", usize::from(causal).to_string()),
        ] {
            spec.set_config(format!("clef.{key}"), value);
        }
        let mut constants = Constants::new();
        constants.push(q.rows as u32)?;
        constants.push(k.rows as u32)?;
        constants.push(packed.cols as u32)?;
        self.emit_special(
            source,
            spec,
            constants,
            &[q.rows as u64, k.rows as u64, packed.cols as u64],
            &[q, k, &packed],
            &out,
        )?;
        Ok(out)
    }
    pub(crate) fn transpose_attention_v(&mut self, v: &Tensor) -> Result<Tensor> {
        ensure!(
            (1..=16384).contains(&v.rows)
                && (64..=65536).contains(&v.cols)
                && v.cols.is_multiple_of(64)
                && v.dtype == DType::Bf16,
            "attention value packing shape"
        );
        let capacity = v.rows.div_ceil(64) * 64;
        let packed = self.alloc(v.cols, capacity, DType::Bf16)?;
        let mut spec = Specialization::new("clef_transpose_v");
        spec.set_config("clef.transpose_v.width", v.cols.to_string());
        spec.set_config("clef.transpose_v.row_capacity", capacity.to_string());
        let mut constants = Constants::new();
        constants.push(v.rows as u32)?;
        self.emit_special(
            include_str!("../kernels/transpose_v.loom"),
            spec,
            constants,
            &[v.rows as u64],
            &[v],
            &packed,
        )?;
        Ok(packed)
    }
    pub fn rope(&mut self, x: &Tensor, heads: usize, rotary: usize, cs: &Tensor) -> Result<Tensor> {
        let d = x.cols / heads;
        ensure!(
            rotary <= d && rotary.is_multiple_of(2) && cs.rows == x.rows && cs.cols == rotary * 2,
            "rope shape"
        );
        let out = self.alloc(x.rows, x.cols, x.dtype)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/rope.loom")
            ),
            out.len(),
            &[("x", x), ("cs", cs)],
            &out,
            &[
                ("cswidth", (cs.cols).to_string()),
                ("d", (d).to_string()),
                ("half", (rotary / 2).to_string()),
                ("rotary", (rotary).to_string()),
                ("width", (x.cols).to_string()),
            ],
        )?;
        Ok(out)
    }
    pub fn conv_delta(&mut self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let out = self.alloc(x.rows, x.cols, DType::Bf16)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/conv_delta.loom")
            ),
            out.len(),
            &[("x", x), ("w", w)],
            &out,
            &[("width", (x.cols).to_string())],
        )?;
        Ok(out)
    }
    pub fn delta_persistent(
        &mut self,
        qkv: &Tensor,
        a: &Tensor,
        b: &Tensor,
        alog: &Tensor,
        dt: &Tensor,
    ) -> Result<Tensor> {
        let n = qkv.rows;
        ensure!(
            n > 0
                && qkv.cols == 10240
                && a.rows == n
                && a.cols == 48
                && b.rows == n
                && b.cols == 48
                && alog.len() == 48
                && dt.len() == 48,
            "persistent DeltaNet shape"
        );
        #[cfg(feature = "bench-internals")]
        if n >= 128 && n <= self.prepared_delta_max {
            let prepared = self.alloc(n, 4192, DType::F32)?;
            self.dispatch(
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../benches/kernels/delta_prepare.loom")
                ),
                n * 16 * 32,
                &[("qkv", qkv), ("a", a), ("b", b), ("al", alog), ("dt", dt)],
                &prepared,
                &[
                    ("n", n.to_string()),
                    ("output_len", prepared.len().to_string()),
                ],
            )?;
            let out = self.alloc(n, 6144, DType::Bf16)?;
            self.dispatch(
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../benches/kernels/delta_segmented_prepared.loom")
                ),
                48 * 32 * 32,
                &[
                    ("qkv", qkv),
                    ("a", &prepared),
                    ("b", b),
                    ("al", alog),
                    ("dt", dt),
                ],
                &out,
                &[("n", n.to_string()), ("output_len", out.len().to_string())],
            )?;
            return Ok(out);
        }
        let out = self.alloc(n, 6144, DType::Bf16)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/delta_persistent.loom")
            ),
            48 * 16 * 32,
            &[("qkv", qkv), ("a", a), ("b", b), ("al", alog), ("dt", dt)],
            &out,
            &[("n", n.to_string()), ("output_len", out.len().to_string())],
        )?;
        Ok(out)
    }
    pub fn delta(
        &mut self,
        qkv: &Tensor,
        a: &Tensor,
        b: &Tensor,
        alog: &Tensor,
        dt: &Tensor,
    ) -> Result<Tensor> {
        // One lane per value channel. FP32 state persists across the entire
        // sequence inside the dispatch and is reinitialized on every replay.
        let n = qkv.rows;
        let output_len = n * 6144;
        let state_len = 48 * 128 * 128;
        let workspace = self.alloc(1, output_len + state_len, DType::F32)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/delta_recurrent.loom")
            ),
            6144,
            &[("qkv", qkv), ("a", a), ("b", b), ("al", alog), ("dt", dt)],
            &workspace,
            &[
                ("n", (n).to_string()),
                ("output_len", (output_len).to_string()),
            ],
        )?;
        let prefix = workspace
            .reshape(workspace.len(), 1)
            .slice_rows(0, output_len)
            .reshape(n, 6144);
        self.cast(&prefix, DType::Bf16)
    }
    pub fn delta_norm(&mut self, x: &Tensor, w: &Tensor, z: &Tensor) -> Result<Tensor> {
        anyhow::ensure!(
            x.cols == 128 && w.len() == 128 && z.len() == x.len(),
            "DeltaNet gated norm dimensions"
        );
        let out = self.alloc(x.rows, x.cols, DType::Bf16)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/delta_norm.loom")
            ),
            x.rows * 32,
            &[("x", x), ("w", w), ("z", z)],
            &out,
            &[],
        )?;
        Ok(out)
    }
}

pub(crate) fn text_rope(positions: &[[u32; 3]]) -> Vec<half::bf16> {
    let mut out = Vec::with_capacity(positions.len() * 128);
    for pos in positions {
        let angles: Vec<_> = (0..32)
            .map(|i| {
                let axis = if i % 3 == 1 && i < 33 {
                    1
                } else if i % 3 == 2 && i < 30 {
                    2
                } else {
                    0
                };
                pos[axis] as f32 / 10_000_000f32.powf(i as f32 / 32.)
            })
            .collect();
        for sin in [false, true] {
            for i in 0..64 {
                out.push(half::bf16::from_f32(if sin {
                    angles[i % 32].sin()
                } else {
                    angles[i % 32].cos()
                }));
            }
        }
    }
    out
}
pub(crate) fn forward(e: &mut Engine, mut x: Tensor, positions: &[[u32; 3]]) -> Result<Tensor> {
    let n = x.rows;
    let cs = e.input(
        n,
        128,
        DType::Bf16,
        bytemuck::cast_slice(&text_rope(positions)),
    )?;
    for i in 0..64 {
        let p = format!("model.language_model.layers.{i}");
        let norm = e.named_norm(&x, &format!("{p}.input_layernorm"), false, 1e-6)?;
        let attn = if i % 4 == 3 {
            let p = format!("{p}.self_attn");
            let qg = e
                .project(&norm, &format!("{p}.q_proj"), false)?
                .reshape(n * 24, 512);
            let q = e.columns(&qg, 0, 256)?;
            let gate = e.columns(&qg, 256, 256)?.reshape(n, 6144);
            let q = e
                .named_norm(&q, &format!("{p}.q_norm"), false, 1e-6)?
                .reshape(n, 6144);
            let q = e.rope(&q, 24, 64, &cs)?;
            let k = e
                .project(&norm, &format!("{p}.k_proj"), false)?
                .reshape(n * 4, 256);
            let k = e
                .named_norm(&k, &format!("{p}.k_norm"), false, 1e-6)?
                .reshape(n, 1024);
            let k = e.rope(&k, 4, 64, &cs)?;
            let v = e.project(&norm, &format!("{p}.v_proj"), false)?;
            let a = e.attention(&q, &k, &v, 24, 4, true)?;
            let gate = e.unary(&gate, "sigmoid")?;
            let a = e.binary(&a, &gate, "mulf")?;
            e.project(&a, &format!("{p}.o_proj"), false)?
        } else {
            let p = format!("{p}.linear_attn");
            let qkv = e.project(&norm, &format!("{p}.in_proj_qkv"), false)?;
            let w = e.w(&format!("{p}.conv1d.weight"))?;
            let qkv = e.conv_delta(&qkv, &w)?;
            let a = e.project(&norm, &format!("{p}.in_proj_a"), false)?;
            let b = e.project(&norm, &format!("{p}.in_proj_b"), false)?;
            let al = e.w(&format!("{p}.A_log"))?;
            let dt = e.w(&format!("{p}.dt_bias"))?;
            #[cfg(not(feature = "bench-internals"))]
            let use_chunked = false;
            #[cfg(feature = "bench-internals")]
            let use_chunked = e.reference_kernels.delta;
            let d = if use_chunked && n > 64 {
                e.delta_chunked(&qkv, &a, &b, &al, &dt)?
            } else if use_chunked {
                e.delta(&qkv, &a, &b, &al, &dt)?
            } else {
                e.delta_persistent(&qkv, &a, &b, &al, &dt)?
            };
            let d = d.reshape(n * 48, 128);
            let w = e.w(&format!("{p}.norm.weight"))?;
            let z = e.project(&norm, &format!("{p}.in_proj_z"), false)?;
            let d = e.delta_norm(&d, &w, &z)?.reshape(n, 6144);
            e.project(&d, &format!("{p}.out_proj"), false)?
        };
        x = e.binary(&x, &attn, "addf")?;
        let norm = e.named_norm(&x, &format!("{p}.post_attention_layernorm"), false, 1e-6)?;
        let gate = e.project(&norm, &format!("{p}.mlp.gate_proj"), false)?;
        let up = e.project(&norm, &format!("{p}.mlp.up_proj"), false)?;
        let gate = e.unary(&gate, "silu")?;
        let gate = e.binary(&gate, &up, "mulf")?;
        let down = e.project(&gate, &format!("{p}.mlp.down_proj"), false)?;
        x = e.binary(&x, &down, "addf")?;
        e.trace(format!("layer_{i}"), &x);
    }
    let x = e.named_norm(&x, "model.language_model.norm", false, 1e-6)?;
    e.trace("hidden", &x);
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires gfx1151; compares fused gated RMSNorm with an FP64 oracle"]
    fn gated_norm_fp64_accuracy() -> Result<()> {
        let mut e = Engine::new(&hrx::inference::ModelContext::new(Default::default())?)?;
        let rows = 13;
        let x: Vec<_> = (0..rows * 128)
            .map(|i| half::bf16::from_f32(((i * 17 % 251) as f32 - 125.) / 16.))
            .collect();
        let z: Vec<_> = (0..rows * 128)
            .map(|i| half::bf16::from_f32(((i * 31 % 131) as f32 - 65.) / 16.))
            .collect();
        let w: Vec<_> = (0..128)
            .map(|i| half::bf16::from_f32(0.5 + (i * 7 % 67) as f32 / 64.))
            .collect();
        let tx = e.input(rows, 128, DType::Bf16, bytemuck::cast_slice(&x))?;
        let tz = e.input(rows, 128, DType::Bf16, bytemuck::cast_slice(&z))?;
        let tw = e.input(1, 128, DType::Bf16, bytemuck::cast_slice(&w))?;
        let fused = e.delta_norm(&tx, &tw, &tz)?;
        let old = e.norm(&tx, &tw, None, 1e-6, false)?;
        let gate = e.unary(&tz, "silu")?;
        let old = e.binary(&old, &gate, "mulf")?;
        let mut graph = e.finish()?;
        e.run(&mut graph)?;
        let actual = e.read_f32(&fused)?;
        let old = e.read_f32(&old)?;
        let mut fused_error = 0.;
        let mut old_error = 0.;
        for row in 0..rows {
            let variance = x[row * 128..(row + 1) * 128]
                .iter()
                .map(|v| v.to_f64().powi(2))
                .sum::<f64>()
                / 128.;
            for (col, weight) in w.iter().enumerate() {
                let i = row * 128 + col;
                let gate = z[i].to_f64();
                let expected = x[i].to_f64() / (variance + 1e-6).sqrt() * weight.to_f64() * gate
                    / (1. + (-gate).exp());
                let error = (actual[i] as f64 - expected).abs();
                assert!(error <= expected.abs() / 256. + 1e-6);
                fused_error += error.powi(2);
                old_error += (old[i] as f64 - expected).powi(2);
            }
        }
        eprintln!("gated RMSNorm FP64 squared error: fused={fused_error:e}, old={old_error:e}");
        assert!(fused_error < old_error);
        Ok(())
    }
}
