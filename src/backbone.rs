use crate::{
    Result,
    gpu::{DType, Engine, Tensor},
};
use anyhow::ensure;
use hrx::{Constants, loom::Specialization};

fn qualified_shared_attention(
    architecture: crate::architecture::Architecture,
    quantized: &crate::quantized::QuantizedProjections,
    tokens: usize,
    heads: usize,
    kv_heads: usize,
    causal: bool,
) -> Option<&'static str> {
    let flash = architecture == crate::ModelVariant::ClefFlash.architecture() && heads == 16;
    let clef = architecture == crate::ModelVariant::Clef.architecture() && heads == 24;
    if !(flash || clef)
        || !causal
        || !(8192..=crate::encoding::MAX_CONTEXT_TOKENS).contains(&tokens)
        || kv_heads != 4
    {
        return None;
    }
    if quantized.is_bf16() {
        Some(concat!(
            include_str!("../kernels/common.loom"),
            include_str!("../kernels/attention_shared_bf16.loom")
        ))
    } else if flash && quantized.is_full_int4() {
        Some(concat!(
            include_str!("../kernels/common.loom"),
            include_str!("../kernels/attention_shared_k16.loom")
        ))
    } else {
        None
    }
}

fn qualified_long_vector_delta(
    architecture: crate::architecture::Architecture,
    quantized: &crate::quantized::QuantizedProjections,
    tokens: usize,
    heads: usize,
) -> bool {
    architecture == crate::ModelVariant::Clef.architecture()
        && quantized.is_bf16()
        && heads == 48
        && (4033..=crate::encoding::MAX_CONTEXT_TOKENS).contains(&tokens)
}

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
            && (1024..=crate::encoding::MAX_CONTEXT_TOKENS).contains(&q.rows)
            && k.rows <= crate::encoding::MAX_CONTEXT_TOKENS
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
        #[cfg(feature = "bench-internals")]
        if causal
            && q.rows >= 8192
            && let Some(source) = self.experimental_attention.clone()
        {
            return self.attention_tiled_source(q, k, v, causal, &source);
        }
        if let Some(source) = qualified_shared_attention(
            self.architecture,
            &self.quantized,
            q.rows,
            heads,
            kv_heads,
            causal,
        ) {
            return self.attention_tiled_source(q, k, v, causal, source);
        }
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
            (1..=crate::encoding::MAX_CONTEXT_TOKENS).contains(&q.rows)
                && (1..=crate::encoding::MAX_CONTEXT_TOKENS).contains(&k.rows)
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
            (1..=crate::encoding::MAX_CONTEXT_TOKENS).contains(&v.rows)
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
        let tiled = (128..=4032).contains(&x.rows)
            && x.cols == 10240
            && x.dtype == DType::Bf16
            && w.dtype == DType::Bf16;
        self.conv_delta_source(
            x,
            w,
            if tiled {
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/conv_delta_rows8.loom")
                )
            } else {
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/conv_delta.loom")
                )
            },
            if tiled { 8 } else { 1 },
        )
    }
    pub(crate) fn conv_delta_source(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        source: &str,
        tile: usize,
    ) -> Result<Tensor> {
        ensure!(
            x.rows > 0 && x.cols > 0 && w.len() == x.cols * 4 && [1, 2, 4, 8].contains(&tile),
            "causal convolution dimensions"
        );
        let out = self.alloc(x.rows, x.cols, DType::Bf16)?;
        self.dispatch(
            source,
            x.rows.div_ceil(tile) * x.cols,
            &[("x", x), ("w", w)],
            &out,
            &[("width", x.cols.to_string())],
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
                && [32, 48].contains(&a.cols)
                && qkv.cols == 4096 + a.cols * 128
                && a.rows == n
                && a.cols == b.cols
                && b.rows == n
                && alog.len() == a.cols
                && dt.len() == a.cols,
            "persistent DeltaNet shape"
        );
        #[cfg(feature = "bench-internals")]
        if a.cols == 48 && n >= 128 && n <= self.prepared_delta_max {
            return self.delta_prepared::<4>(
                qkv,
                a,
                b,
                alog,
                dt,
                (
                    concat!(
                        include_str!("../kernels/common.loom"),
                        include_str!("../benches/kernels/delta_prepare.loom")
                    ),
                    concat!(
                        include_str!("../kernels/common.loom"),
                        include_str!("../benches/kernels/delta_segmented_prepared.loom")
                    ),
                ),
            );
        }
        // Operator savings were offset by preparation in complete evaluations.
        // Retain this ablation without changing production selection.
        #[cfg(feature = "bench-internals")]
        if a.cols == 48 && n >= 128 && n <= self.prepared_delta_v8_max {
            return self.delta_prepared::<8>(
                qkv,
                a,
                b,
                alog,
                dt,
                (
                    concat!(
                        include_str!("../kernels/common.loom"),
                        include_str!("../kernels/delta_prepare.loom")
                    ),
                    concat!(
                        include_str!("../kernels/common.loom"),
                        include_str!("../kernels/delta_persistent_prepared.loom")
                    ),
                ),
            );
        }
        let out = self.alloc(n, a.cols * 128, DType::Bf16)?;
        let long_vector =
            qualified_long_vector_delta(self.architecture, &self.quantized, n, a.cols);
        #[cfg(feature = "bench-internals")]
        let long_vector = self
            .long_vector_delta
            .map(|enabled| n > 4032 && enabled)
            .unwrap_or(long_vector);
        // Qualified vector scheduling range; preserve the scalar implementation
        // for shorter inputs and long-context workloads outside this range.
        let (source, values_per_wave) = if a.cols == 32 {
            (
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/delta_persistent_flash.loom")
                ),
                8,
            )
        } else if (128..=1024).contains(&n) {
            (
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/delta_persistent_vector16.loom")
                ),
                16,
            )
        } else if (1025..=4032).contains(&n) || long_vector {
            (
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/delta_persistent_vector.loom")
                ),
                8,
            )
        } else {
            (
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/delta_persistent.loom")
                ),
                8,
            )
        };
        self.dispatch(
            source,
            a.cols * 128 / values_per_wave * 32,
            &[("qkv", qkv), ("a", a), ("b", b), ("al", alog), ("dt", dt)],
            &out,
            &[("n", n.to_string()), ("output_len", out.len().to_string())],
        )?;
        Ok(out)
    }
    #[cfg(any(feature = "bench-internals", test))]
    fn delta_prepared<const VALUES: usize>(
        &mut self,
        qkv: &Tensor,
        a: &Tensor,
        b: &Tensor,
        alog: &Tensor,
        dt: &Tensor,
        sources: (&str, &str),
    ) -> Result<Tensor> {
        let n = qkv.rows;
        let prepared = self.alloc(n, 4192, DType::F32)?;
        self.dispatch(
            sources.0,
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
            sources.1,
            48 * (128 / VALUES) * 32,
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
    let arch = e.architecture;
    let attention_width = arch.attention_heads * 256;
    let delta_width = arch.delta_heads * 128;
    let cs = e.input(
        n,
        128,
        DType::Bf16,
        bytemuck::cast_slice(&text_rope(positions)),
    )?;
    for i in 0..arch.layers {
        let p = format!("model.language_model.layers.{i}");
        let first_projection = if i % 4 == 3 {
            "self_attn.q_proj"
        } else {
            "linear_attn.in_proj_qkv"
        };
        let norm = e.normalized_input(
            &x,
            &format!("{p}.input_layernorm"),
            &format!("{p}.{first_projection}"),
        )?;
        x = if i % 4 == 3 {
            let p = format!("{p}.self_attn");
            let qg = e
                .project(&norm, &format!("{p}.q_proj"), false)?
                .reshape(n * arch.attention_heads, 512);
            let q = e.columns(&qg, 0, 256)?;
            let output_projection = format!("{p}.o_proj");
            let fuse_gate = arch.attention_heads == 24
                && n >= 128
                && e.quantized
                    .weights
                    .contains_key(&format!("{output_projection}.weight"));
            let gate = if fuse_gate {
                None
            } else {
                Some(e.columns(&qg, 256, 256)?.reshape(n, attention_width))
            };
            let q = e
                .named_norm(&q, &format!("{p}.q_norm"), false, 1e-6)?
                .reshape(n, attention_width);
            let q = e.rope(&q, arch.attention_heads, 64, &cs)?;
            let k = e
                .project(&norm, &format!("{p}.k_proj"), false)?
                .reshape(n * 4, 256);
            let k = e
                .named_norm(&k, &format!("{p}.k_norm"), false, 1e-6)?
                .reshape(n, 1024);
            let k = e.rope(&k, 4, 64, &cs)?;
            let v = e.project(&norm, &format!("{p}.v_proj"), false)?;
            let a = e.attention(&q, &k, &v, arch.attention_heads, 4, true)?;
            if let Some(gate) = gate {
                let gate = e.unary(&gate, "sigmoid")?;
                let a = e.binary(&a, &gate, "mulf")?;
                e.project_residual(&a, &output_projection, &x)?
            } else {
                e.project_attention_gate(
                    &a,
                    &qg.reshape(n, attention_width * 2),
                    &output_projection,
                    &x,
                )?
            }
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
            let use_chunked = arch.delta_heads == 48 && e.reference_kernels.delta;
            let d = if use_chunked && n > 64 {
                e.delta_chunked(&qkv, &a, &b, &al, &dt)?
            } else if use_chunked {
                e.delta(&qkv, &a, &b, &al, &dt)?
            } else {
                e.delta_persistent(&qkv, &a, &b, &al, &dt)?
            };
            let d = d.reshape(n * arch.delta_heads, 128);
            let w = e.w(&format!("{p}.norm.weight"))?;
            let z = e.project(&norm, &format!("{p}.in_proj_z"), false)?;
            let d = e.delta_norm(&d, &w, &z)?.reshape(n, delta_width);
            e.project_residual(&d, &format!("{p}.out_proj"), &x)?
        };
        if n > 16_384 {
            drop(norm);
        }
        let norm_name = format!("{p}.post_attention_layernorm");
        let mlp_name = format!("{p}.mlp");
        if let Some(output) = e.normalized_mlp(&x, &norm_name, &mlp_name)? {
            x = output;
        } else {
            let norm = e.named_norm(&x, &norm_name, false, 1e-6)?;
            x = if n > 16_384 {
                // MLP rows are independent. Bound the wide activations and keep all
                // kernel extents below u32::MAX throughout the supported context.
                let down = mlp_chunked(e, &norm, &mlp_name, 8192)?;
                e.binary(&x, &down, "addf")?
            } else {
                mlp(e, &norm, &mlp_name, Some(&x))?
            };
        }
        e.trace(format!("layer_{i}"), &x);
    }
    let x = e.named_norm(&x, "model.language_model.norm", false, 1e-6)?;
    e.trace("hidden", &x);
    Ok(x)
}

fn mlp(e: &mut Engine, x: &Tensor, prefix: &str, residual: Option<&Tensor>) -> Result<Tensor> {
    let gate = e.project(x, &format!("{prefix}.gate_proj"), false)?;
    let up = e.project(x, &format!("{prefix}.up_proj"), false)?;
    e.project_swiglu(&gate, &up, &format!("{prefix}.down_proj"), residual)
}

fn mlp_chunked(e: &mut Engine, x: &Tensor, prefix: &str, chunk: usize) -> Result<Tensor> {
    ensure!(chunk > 0, "MLP chunk must be positive");
    let out = e.alloc(x.rows, x.cols, x.dtype)?;
    for start in (0..x.rows).step_by(chunk) {
        let rows = chunk.min(x.rows - start);
        let part = mlp(e, &x.slice_rows(start, rows), prefix, None)?;
        e.copy_into(&part, &out.slice_rows(start, rows))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires gfx1151; independent FP64 Flash recurrence and replay oracle"]
    fn flash_delta_fp64_oracle() -> crate::Result<()> {
        use crate::gpu::{DType, Engine};
        use hrx::inference::ModelContext;
        let context = ModelContext::new(Default::default())?;
        let mut e = Engine::new(&context)?;
        let (n, heads, width) = (7, 32, 8192);
        let values: Vec<_> = (0..n * width)
            .map(|i| half::bf16::from_f32(((i * 17 % 251) as f32 - 125.) / 512.))
            .collect();
        let qkv = e.input(n, width, DType::Bf16, bytemuck::cast_slice(&values))?;
        let ab = e.input(
            n,
            heads,
            DType::Bf16,
            bytemuck::cast_slice(&vec![half::bf16::ZERO; n * heads]),
        )?;
        let al = e.input(
            1,
            heads,
            DType::Bf16,
            bytemuck::cast_slice(&vec![half::bf16::ZERO; heads]),
        )?;
        let out = e.delta_persistent(&qkv, &ab, &ab, &al, &al)?;
        let mut graph = e.finish()?;
        e.run(&mut graph)?;
        let actual = e.read_f32(&out)?;
        let mut maximum = 0f64;
        for h in 0..heads {
            let mut state = vec![0f64; 128 * 128];
            for t in 0..n {
                let base = t * width + (h / 2) * 128;
                let q: Vec<_> = values[base..base + 128]
                    .iter()
                    .map(|v| v.to_f64())
                    .collect();
                let k: Vec<_> = values[base + 2048..base + 2176]
                    .iter()
                    .map(|v| v.to_f64())
                    .collect();
                let qi =
                    (q.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt().recip() / 128f64.sqrt();
                let ki = (k.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt().recip();
                for channel in 0..128 {
                    let row = &mut state[channel * 128..(channel + 1) * 128];
                    for s in row.iter_mut() {
                        *s *= 0.5;
                    }
                    let prediction: f64 = row.iter().zip(&k).map(|(s, k)| s * k * ki).sum();
                    let value = values[t * width + 4096 + h * 128 + channel].to_f64();
                    let delta = (value - prediction) * 0.5;
                    for (s, k) in row.iter_mut().zip(&k) {
                        *s += delta * k * ki;
                    }
                    let expected: f64 = row.iter().zip(&q).map(|(s, q)| s * q * qi).sum();
                    maximum = maximum
                        .max((f64::from(actual[t * 4096 + h * 128 + channel]) - expected).abs());
                }
            }
        }
        assert!(maximum < 0.002, "Flash recurrence error {maximum}");
        e.run(&mut graph)?;
        assert_eq!(actual, e.read_f32(&out)?);
        Ok(())
    }

    use super::*;

    #[test]
    fn shared_attention_stays_within_qualified_model_precision_and_context() {
        use crate::{ModelVariant, Precision, quantized::QuantizedProjections};
        for model in [ModelVariant::Clef, ModelVariant::ClefFlash] {
            for precision in [Precision::Bf16, Precision::Int4Mlp, Precision::Int4] {
                let quantized = QuantizedProjections::for_precision(precision);
                let heads = model.architecture().attention_heads;
                for tokens in [8191, 8192, 65536, 65537] {
                    let selected = qualified_shared_attention(
                        model.architecture(),
                        &quantized,
                        tokens,
                        heads,
                        4,
                        true,
                    );
                    assert_eq!(
                        selected.is_some(),
                        (precision == Precision::Bf16
                            || (model == ModelVariant::ClefFlash && precision == Precision::Int4))
                            && (8192..=65536).contains(&tokens),
                    );
                }
                for (heads, kv_heads, causal) in
                    [(heads, 4, false), (heads + 8, 4, true), (heads, 8, true)]
                {
                    assert!(
                        qualified_shared_attention(
                            model.architecture(),
                            &quantized,
                            8192,
                            heads,
                            kv_heads,
                            causal,
                        )
                        .is_none()
                    );
                }
            }
        }
    }

    #[test]
    fn long_vector_delta_stays_within_qualified_model_precision_and_context() {
        use crate::{ModelVariant, Precision, quantized::QuantizedProjections};
        for model in [ModelVariant::Clef, ModelVariant::ClefFlash] {
            for precision in [Precision::Bf16, Precision::Int4Mlp, Precision::Int4] {
                let quantized = QuantizedProjections::for_precision(precision);
                for tokens in [4032, 4033, 8192, 65536, 65537] {
                    for heads in [32, 48] {
                        assert_eq!(
                            qualified_long_vector_delta(
                                model.architecture(),
                                &quantized,
                                tokens,
                                heads
                            ),
                            model == ModelVariant::Clef
                                && precision == Precision::Bf16
                                && heads == 48
                                && (4033..=65536).contains(&tokens)
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires gfx1151; checks causal history, row tails, slices, rounding and replay"]
    fn convolution_row_reuse_preserves_outputs() -> Result<()> {
        let context = hrx::inference::ModelContext::new(Default::default())?;
        let width = 10240;
        for rows in [
            1, 2, 3, 4, 7, 8, 9, 127, 128, 129, 158, 235, 620, 1023, 1024, 2048, 4031, 4032, 4033,
        ] {
            let mut e = Engine::new(&context)?;
            let xv: Vec<_> = (0..(rows + 1) * width)
                .map(|i| {
                    half::bf16::from_f32(match i % width {
                        0 => f32::NAN,
                        1 => f32::INFINITY,
                        2 => f32::NEG_INFINITY,
                        3 => -0.,
                        4 => 0.,
                        5 => half::bf16::from_bits(1).to_f32(),
                        6 => half::bf16::MAX.to_f32(),
                        _ => {
                            ((i * 17 % 1031) as f32 - 515.) / if i % 7 == 0 { 65536. } else { 128. }
                        }
                    })
                })
                .collect();
            let wv: Vec<_> = (0..(width + 2) * 4)
                .map(|i| {
                    half::bf16::from_f32(match (i / 4, i % 4) {
                        (9, 0) => f32::NAN,
                        (10, 1) => f32::INFINITY,
                        (11, 2) => f32::NEG_INFINITY,
                        _ => ((i * 31 % 127) as f32 - 63.) / 128.,
                    })
                })
                .collect();
            let x = e
                .input(rows + 1, width, DType::Bf16, bytemuck::cast_slice(&xv))?
                .slice_rows(1, rows);
            let w = e
                .input(width + 2, 4, DType::Bf16, bytemuck::cast_slice(&wv))?
                .slice_rows(2, width);
            let reference = e.conv_delta_source(
                &x,
                &w,
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/conv_delta.loom")
                ),
                1,
            )?;
            let candidate = e.conv_delta_source(
                &x,
                &w,
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/conv_delta_rows8.loom")
                ),
                8,
            )?;
            let selected = e.conv_delta(&x, &w)?;
            let mut graph = e.finish()?;
            let mut first = None;
            for _ in 0..2 {
                e.run(&mut graph)?;
                let expected = e.read_f32(&reference)?;
                for output in [&candidate, &selected] {
                    let actual = e.read_f32(output)?;
                    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                        if b.is_nan() {
                            assert!(a.is_nan(), "NaN at {rows}, {i}");
                        } else {
                            assert_eq!(a.to_bits(), b.to_bits(), "convolution {rows}, {i}");
                        }
                    }
                }
                let bits: Vec<_> = expected.iter().map(|x| x.to_bits()).collect();
                if let Some(first) = &first {
                    assert_eq!(first, &bits, "reference replay {rows}");
                } else {
                    first = Some(bits);
                }
                // Independent scalar oracle at startup, interior and final rows.
                for row in [0, 1.min(rows - 1), 3.min(rows - 1), rows / 2, rows - 1] {
                    for col in [31, 127, 5120, width - 1] {
                        let mut sum = 0f64;
                        for tap in 0..4 {
                            if row + tap >= 3 {
                                sum += xv[(row + tap - 3 + 1) * width + col].to_f64()
                                    * wv[(col + 2) * 4 + tap].to_f64();
                            }
                        }
                        let rounded = half::bf16::from_f64(sum).to_f64();
                        let oracle = rounded / (1. + (-rounded).exp());
                        let actual = f64::from(expected[row * width + col]);
                        assert!(
                            (actual - oracle).abs() <= 0.004 * oracle.abs() + 1e-6,
                            "convolution oracle row={row} col={col}: {actual} vs {oracle}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151; checks prepared recurrence boundaries, slices and replay"]
    fn delta_variants_preserve_state_arithmetic() -> Result<()> {
        let context = hrx::inference::ModelContext::new(Default::default())?;
        for n in [127, 128, 129, 620, 1023, 1024, 1025, 2048, 4032, 4033] {
            let mut e = Engine::new(&context)?;
            let values = |len, factor, divisor| {
                (0..len)
                    .map(|i| half::bf16::from_f32(((i * factor % 251) as f32 - 125.) / divisor))
                    .collect::<Vec<_>>()
            };
            let qkv = e
                .input(
                    n + 1,
                    10240,
                    DType::Bf16,
                    bytemuck::cast_slice(&values((n + 1) * 10240, 17, 512.)),
                )?
                .slice_rows(1, n);
            let a = e
                .input(
                    n + 1,
                    48,
                    DType::Bf16,
                    bytemuck::cast_slice(&values((n + 1) * 48, 13, 64.)),
                )?
                .slice_rows(1, n);
            let b = e
                .input(
                    n + 1,
                    48,
                    DType::Bf16,
                    bytemuck::cast_slice(&values((n + 1) * 48, 19, 128.)),
                )?
                .slice_rows(1, n);
            let alog = e.input(
                1,
                48,
                DType::Bf16,
                bytemuck::cast_slice(&values(48, 7, 256.)),
            )?;
            let dt = e.input(
                1,
                48,
                DType::Bf16,
                bytemuck::cast_slice(&values(48, 11, 256.)),
            )?;
            let vector = e.delta_persistent(&qkv, &a, &b, &alog, &dt)?;
            let actual = e.delta_prepared::<8>(
                &qkv,
                &a,
                &b,
                &alog,
                &dt,
                (
                    concat!(
                        include_str!("../kernels/common.loom"),
                        include_str!("../kernels/delta_prepare.loom")
                    ),
                    concat!(
                        include_str!("../kernels/common.loom"),
                        include_str!("../kernels/delta_persistent_prepared.loom")
                    ),
                ),
            )?;
            let expected = e.alloc(n, 6144, DType::Bf16)?;
            e.dispatch(
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/delta_persistent.loom")
                ),
                48 * 16 * 32,
                &[
                    ("qkv", &qkv),
                    ("a", &a),
                    ("b", &b),
                    ("al", &alog),
                    ("dt", &dt),
                ],
                &expected,
                &[
                    ("n", n.to_string()),
                    ("output_len", expected.len().to_string()),
                ],
            )?;
            let mut graph = e.finish()?;
            for _ in 0..2 {
                e.run(&mut graph)?;
                let expected_bits = e
                    .read_f32(&expected)?
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>();
                for (variant, tensor) in [("prepared", &actual), ("vector", &vector)] {
                    let output = e.read_f32(tensor)?;
                    assert!(output.iter().all(|x| x.is_finite()));
                    assert_eq!(
                        output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        expected_bits,
                        "{variant} n={n}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151; independent long causal shared-attention oracle and replay"]
    fn shared_attention_context_oracle() -> Result<()> {
        let source = if let Some(path) = std::env::var_os("CLEF_BENCH_ATTENTION_ORACLE_SOURCE") {
            format!(
                "{}\n{}",
                include_str!("../kernels/common.loom"),
                std::fs::read_to_string(path)?
            )
        } else {
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../benches/kernels/attention_shared.loom")
            )
            .to_owned()
        };
        for n in [65, 8191, 8192, 8193, 16385, 65536] {
            let mut e = Engine::new(&hrx::inference::ModelContext::new(Default::default())?)?;
            let q = e
                .input(n + 1, 512, DType::Bf16, &vec![0; (n + 1) * 512 * 2])?
                .slice_rows(1, n);
            let k = e.input(n, 256, DType::Bf16, &vec![0; n * 256 * 2])?;
            let row_value = |row: usize| match row {
                0..32 => -0.5f64,
                32..8192 => 0.25,
                8192..16384 => -0.25,
                _ => 0.5,
            };
            let channel_value = |col: usize| ((col % 256 % 9) as f64 - 4.) / 16.;
            let values: Vec<_> = (0..(n + 1) * 256)
                .map(|i| {
                    half::bf16::from_f64(row_value((i / 256).saturating_sub(1)) + channel_value(i))
                })
                .collect();
            let v = e
                .input(n + 1, 256, DType::Bf16, bytemuck::cast_slice(&values))?
                .slice_rows(1, n);
            let output = e.attention_tiled_source(&q, &k, &v, true, &source)?;
            let mut graph = e.finish()?;
            let mut previous = None;
            for _ in 0..2 {
                e.run(&mut graph)?;
                let actual = e.read_f32(&output)?;
                let mut sum = 0.;
                for (row, channels) in actual.as_chunks::<512>().0.iter().enumerate() {
                    sum += row_value(row);
                    for (col, &value) in channels.iter().enumerate() {
                        let expected = sum / (row + 1) as f64 + channel_value(col);
                        assert!(
                            value.is_finite() && (f64::from(value) - expected).abs() <= 0.002,
                            "n={n} row={row} col={col}: {value} != {expected}"
                        );
                    }
                }
                if let Some(previous) = &previous {
                    assert_eq!(&actual, previous, "shared attention replay n={n}");
                }
                previous = Some(actual);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151; checks long attention indexing against uniform-softmax oracle"]
    fn native_context_attention() -> Result<()> {
        for (queries, keys) in [
            (17, 16_385),
            (17, crate::encoding::MAX_CONTEXT_TOKENS),
            (crate::encoding::MAX_CONTEXT_TOKENS, 17),
        ] {
            let mut e = Engine::new(&hrx::inference::ModelContext::new(Default::default())?)?;
            // Include a nonzero slice offset, a partial query tile and key tail.
            let q = e
                .input(
                    queries + 1,
                    256,
                    DType::Bf16,
                    &vec![0; (queries + 1) * 256 * 2],
                )?
                .slice_rows(1, queries);
            let k = e.input(keys, 256, DType::Bf16, &vec![0; keys * 256 * 2])?;
            let values: Vec<_> = (0..keys * 256)
                .map(|i| half::bf16::from_f32(if i / 256 >= 16_384 { 2. } else { 1. }))
                .collect();
            let v = e.input(keys, 256, DType::Bf16, bytemuck::cast_slice(&values))?;
            let output = e.attention_tiled(&q, &k, &v, 1, 1, false)?;
            let mut graph = e.finish()?;
            e.run(&mut graph)?;
            let prefix = keys.min(16_384);
            let expected = (prefix as f32 + (keys - prefix) as f32 * 2.) / keys as f32;
            for value in e.read_f32(&output)? {
                assert!(
                    (value - expected).abs() < 0.008,
                    "keys={keys}: {value} != {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151; checks chunk boundaries, tails and graph buffer reuse"]
    fn chunked_mlp_parity() -> Result<()> {
        let mut e = Engine::new(&hrx::inference::ModelContext::new(Default::default())?)?;
        for (name, rows, cols) in [
            ("gate_proj", 256, 128),
            ("up_proj", 256, 128),
            ("down_proj", 128, 256),
        ] {
            let values: Vec<_> = (0..rows * cols)
                .map(|i| half::bf16::from_f32((i as i32 % 17 - 8) as f32 / 128.))
                .collect();
            e.weight(
                format!("test.{name}.weight"),
                rows,
                cols,
                bytemuck::cast_slice(&values),
            )?;
        }
        let values: Vec<_> = (0usize..131 * 128)
            .map(|i| half::bf16::from_f32((i as i32 % 23 - 11) as f32 / 32.))
            .collect();
        let x = e
            .input(131, 128, DType::Bf16, bytemuck::cast_slice(&values))?
            .slice_rows(1, 130);
        let expected = mlp(&mut e, &x, "test", None)?;
        let actual = mlp_chunked(&mut e, &x, "test", 64)?;
        let mut graph = e.finish()?;
        for _ in 0..2 {
            e.run(&mut graph)?;
            assert_eq!(e.read_f32(&actual)?, e.read_f32(&expected)?);
        }
        Ok(())
    }

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
