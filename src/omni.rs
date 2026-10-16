//! Native Qwen3-Omni thinker operators. Speech generation is not part of CLEF.
use crate::{
    Result,
    checkpoint::Checkpoint,
    gpu::{DType, Engine, Layout, Tensor},
};
use anyhow::ensure;

const EXPERTS: usize = 128;
const TOP_K: usize = 8;
pub(crate) const MAX_EXPERT_CHUNK: usize = 2048;

/// Embedded sources are byte-identical to the accepted full-evaluation variants.
fn fast_expert_source(bits: usize, tokens: usize) -> Option<&'static str> {
    match (bits, tokens) {
        (4, 512..) => Some(concat!(
            include_str!("../kernels/common.loom"),
            include_str!("../kernels/omni_expert_i4_k128_down.loom")
        )),
        (0, 128..) => Some(concat!(
            include_str!("../kernels/common.loom"),
            include_str!("../kernels/omni_expert_bf16_m64.loom")
        )),
        _ => None,
    }
}

#[cfg(test)]
#[path = "omni_validation.rs"]
mod validation;

/// Concatenate expert matrices in numeric expert order. One projection at a time
/// bounds host staging; untouched speech-output tensors never become resident.
pub(crate) fn load_experts(e: &mut Engine, checkpoint: &Checkpoint) -> Result<()> {
    for layer in 0..48 {
        load_expert_layer(e, checkpoint, layer)?;
    }
    Ok(())
}
pub(crate) fn load_expert_layer(
    e: &mut Engine,
    checkpoint: &Checkpoint,
    layer: usize,
) -> Result<()> {
    for projection in ["gate", "up", "down"] {
        let prefix = format!("model.language_model.layers.{layer}.mlp.experts");
        let name = format!("{prefix}.{projection}_proj.weight");
        let (rows, cols) = if projection == "down" {
            (2048, 768)
        } else {
            (768, 2048)
        };
        let bytes = EXPERTS * rows * cols * 2;
        crate::memory::before_allocation(bytes)?;
        let mut combined = Vec::with_capacity(bytes);
        for expert in 0..EXPERTS {
            let key = format!("{prefix}.{expert}.{projection}_proj.weight");
            let tensor = checkpoint.tensor(&key)?;
            ensure!(
                tensor.shape == [rows, cols],
                "invalid Omni expert shape: {key}"
            );
            combined.extend_from_slice(tensor.bytes);
            checkpoint.done_with(&key)?;
        }
        // Concatenating complete 256-row tiles preserves the tiled layout.
        if !e.load_quantized_weight(&name, EXPERTS * rows, cols, &combined)? {
            e.weight(name, EXPERTS * rows, cols, &combined)?;
        }
    }
    Ok(())
}

impl Engine {
    #[cfg(feature = "bench-internals")]
    pub(crate) fn omni_norm_rope(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        heads: usize,
        cs: &Tensor,
    ) -> Result<Tensor> {
        ensure!(
            self.architecture.is_omni()
                && [4, 32].contains(&heads)
                && x.cols == heads * 128
                && w.len() == 128
                && cs.rows == x.rows
                && cs.cols == 256
                && [x, w, cs].iter().all(|t| t.dtype == DType::Bf16),
            "unsupported fused Omni normalization/RoPE shape or dtype"
        );
        let out = self.alloc(x.rows, x.cols, DType::Bf16)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../benches/kernels/omni_norm_rope.loom")
            ),
            x.rows * heads * 32,
            &[("x", x), ("w", w), ("cs", cs)],
            &out,
            &[("heads", heads.to_string())],
        )?;
        Ok(out)
    }

    fn omni_normalized_rope(
        &mut self,
        x: &Tensor,
        name: &str,
        heads: usize,
        cs: &Tensor,
    ) -> Result<Tensor> {
        #[cfg(feature = "bench-internals")]
        if self.omni_fused_norm_rope {
            let w = self.w(&format!("{name}.weight"))?;
            return self.omni_norm_rope(x, &w, heads, cs);
        }
        let norm = self
            .named_norm(&x.reshape(x.rows * heads, 128), name, false, 1e-6)?
            .reshape(x.rows, heads * 128);
        self.rope(&norm, heads, 128, cs)
    }

    /// Prefix-compatible grouped routes, rank order, then a compact tile schedule.
    pub(crate) fn omni_compact_routes(&mut self, routes: &Tensor) -> Result<(Tensor, Tensor)> {
        ensure!(
            (1..=MAX_EXPERT_CHUNK).contains(&routes.rows)
                && routes.cols == TOP_K * 2
                && routes.dtype == DType::F32,
            "unsupported compact Omni routes"
        );
        let group_len = EXPERTS * (routes.rows + 1);
        let order_len = routes.rows * TOP_K;
        let task_len = 1 + 2 * ((routes.rows * TOP_K).div_ceil(16) + EXPERTS - 1);
        let combined = self.alloc(group_len + order_len + task_len, 1, DType::F32)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/omni_group_routes_ordered.loom")
            ),
            EXPERTS * 64,
            &[("routes", routes)],
            &combined,
            &[
                ("tokens", routes.rows.to_string()),
                ("topk", TOP_K.to_string()),
            ],
        )?;
        let groups = combined
            .slice_rows(0, group_len)
            .reshape(EXPERTS, routes.rows + 1);
        let tasks = combined.slice_rows(group_len + order_len, task_len);
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/omni_expert_tasks.loom")
            ),
            EXPERTS,
            &[("routes", &groups)],
            &tasks,
            &[("tokens", routes.rows.to_string())],
        )?;
        let order = combined
            .slice_rows(group_len, order_len)
            .reshape(routes.rows, TOP_K);
        let len = combined.len();
        Ok((combined.reshape(1, len), order))
    }
    pub(crate) fn omni_route(&mut self, logits: &Tensor, topk: usize) -> Result<Tensor> {
        ensure!(
            topk > 0 && topk <= logits.cols && logits.cols <= 128,
            "invalid expert router shape"
        );
        let out = self.alloc(logits.rows, topk * 2, DType::F32)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/omni_route.loom")
            ),
            logits.rows * 32,
            &[("x", logits)],
            &out,
            &[
                ("experts", logits.cols.to_string()),
                ("topk", topk.to_string()),
            ],
        )?;
        Ok(out)
    }
    /// Stable per-expert route lists; each token selects an expert at most once.
    pub(crate) fn omni_group_routes(&mut self, routes: &Tensor) -> Result<Tensor> {
        ensure!(
            (1..=MAX_EXPERT_CHUNK).contains(&routes.rows) && routes.cols == TOP_K * 2,
            "unsupported Omni route grouping shape"
        );
        let out = self.alloc(EXPERTS, routes.rows + 1, DType::F32)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/omni_group_routes.loom")
            ),
            EXPERTS * 64,
            &[("routes", routes)],
            &out,
            &[
                ("tokens", routes.rows.to_string()),
                ("topk", TOP_K.to_string()),
            ],
        )?;
        Ok(out)
    }
    pub(crate) fn omni_group_routes_ordered(
        &mut self,
        routes: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        ensure!(
            (1..=MAX_EXPERT_CHUNK).contains(&routes.rows)
                && routes.cols == TOP_K * 2
                && routes.dtype == DType::F32,
            "unsupported ordered Omni routes"
        );
        let group_len = EXPERTS * (routes.rows + 1);
        let combined = self.alloc(group_len + routes.rows * TOP_K, 1, DType::F32)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/omni_group_routes_ordered.loom")
            ),
            EXPERTS * 64,
            &[("routes", routes)],
            &combined,
            &[
                ("tokens", routes.rows.to_string()),
                ("topk", TOP_K.to_string()),
            ],
        )?;
        let groups = combined
            .slice_rows(0, group_len)
            .reshape(EXPERTS, routes.rows + 1);
        let order = combined
            .slice_rows(group_len, routes.rows * TOP_K)
            .reshape(routes.rows, TOP_K);
        Ok((groups, order))
    }
    pub(crate) fn omni_combine_ordered(
        &mut self,
        values: &Tensor,
        routes: &Tensor,
        order: &Tensor,
    ) -> Result<Tensor> {
        ensure!(
            routes.cols == TOP_K * 2
                && order.rows == routes.rows
                && order.cols == TOP_K
                && values.rows == routes.rows * TOP_K
                && values.cols.is_multiple_of(8)
                && values.dtype == DType::Bf16
                && routes.dtype == DType::F32
                && order.dtype == DType::F32,
            "unsupported ordered Omni accumulation"
        );
        let out = self.alloc(routes.rows, values.cols, DType::Bf16)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/omni_combine_ordered.loom")
            ),
            out.len() / 8,
            &[("x", values), ("routes", routes), ("order", order)],
            &out,
            &[
                ("width", values.cols.to_string()),
                ("topk", TOP_K.to_string()),
            ],
        )?;
        Ok(out)
    }
    #[cfg(feature = "bench-internals")]
    pub(crate) fn omni_expert(
        &mut self,
        x: &Tensor,
        routes: &Tensor,
        name: &str,
        input_routed: bool,
    ) -> Result<Tensor> {
        self.omni_expert_impl(x, routes, None, name, input_routed)
    }
    pub(crate) fn omni_expert_impl(
        &mut self,
        x: &Tensor,
        routes: &Tensor,
        groups: Option<&Tensor>,
        name: &str,
        input_routed: bool,
    ) -> Result<Tensor> {
        self.omni_expert_inputs(x, None, routes, groups, name, input_routed)
    }

    /// Fused INT4 gate/up and SwiGLU using the benchmark N32 launch ABI.
    #[cfg(feature = "bench-internals")]
    pub(crate) fn omni_gate_up(
        &mut self,
        x: &Tensor,
        routes: &Tensor,
        groups: &Tensor,
        gate_name: &str,
        up_name: &str,
    ) -> Result<Tensor> {
        ensure!(
            (1..=MAX_EXPERT_CHUNK).contains(&routes.rows)
                && routes.cols == TOP_K * 2
                && routes.dtype == DType::F32
                && x.rows == routes.rows
                && x.dtype == DType::Bf16
                && groups.rows == EXPERTS
                && groups.cols == routes.rows + 1
                && groups.dtype == DType::F32,
            "fused gate/up requires regular grouped routes"
        );
        let gw = self
            .quantized
            .weights
            .get(&format!("{gate_name}.weight"))
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("fused gate requires INT4 weights"))?;
        let uw = self
            .quantized
            .weights
            .get(&format!("{up_name}.weight"))
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("fused up requires INT4 weights"))?;
        ensure!(
            gw.bits == 4
                && uw.bits == 4
                && gw.cols == x.cols
                && uw.cols == x.cols
                && gw.rows == uw.rows
                && gw.stride == uw.stride
                && gw.rows.is_multiple_of(EXPERTS * 64),
            "incompatible fused expert weights"
        );
        let source = self
            .omni_gate_up_source
            .clone()
            .ok_or_else(|| anyhow::anyhow!("missing fused gate/up source"))?;
        let (packed, scales) = self.prepare_quantized_input(x, &gw)?;
        let n = gw.rows / EXPERTS;
        let out = self.alloc(routes.rows * TOP_K, n, DType::Bf16)?;
        self.dispatch(
            &source,
            EXPERTS * (n / 32) * 32,
            &[
                ("x", &packed),
                ("w", &gw.codes),
                ("routes", groups),
                ("xs", &scales),
                ("ws", &gw.scales),
                ("u", &uw.codes),
                ("us", &uw.scales),
            ],
            &out,
            &[
                ("tokens", routes.rows.to_string()),
                ("k", x.cols.to_string()),
                ("n", n.to_string()),
                ("topk", TOP_K.to_string()),
                ("input_routed", "0".into()),
                ("stride", gw.stride.to_string()),
            ],
        )?;
        Ok(out)
    }

    fn omni_expert_inputs(
        &mut self,
        x: &Tensor,
        up: Option<&Tensor>,
        routes: &Tensor,
        groups: Option<&Tensor>,
        name: &str,
        input_routed: bool,
    ) -> Result<Tensor> {
        let topk = routes.cols / 2;
        let compact = groups.is_some_and(|g| g.rows == 1);
        if let Some(groups) = groups {
            let compact_len = EXPERTS * (routes.rows + 1)
                + routes.rows * TOP_K
                + 1
                + 2 * ((routes.rows * TOP_K).div_ceil(16) + EXPERTS - 1);
            ensure!(
                (1..=MAX_EXPERT_CHUNK).contains(&routes.rows)
                    && topk == TOP_K
                    && ((groups.rows == EXPERTS && groups.cols == routes.rows + 1)
                        || (compact && groups.cols == compact_len))
                    && groups.dtype == DType::F32,
                "invalid grouped expert routes"
            );
        }
        let weight = format!("{name}.weight");
        let (a, w, xs, ws, rows, stride, bits) =
            if let Some(w) = self.quantized.weights.get(&weight).cloned() {
                ensure!(
                    w.bits == 4 && w.cols == x.cols,
                    "unsupported Omni expert packing"
                );
                #[cfg(feature = "bench-internals")]
                let shared = self.omni_fast_defaults || self.omni_fused_mlp;
                #[cfg(not(feature = "bench-internals"))]
                let shared = self.omni_fast_defaults;
                let (a, scales) = if let Some(up) = up {
                    let packed = crate::quantized::prepare_with_inputs(
                        self,
                        &[x, up],
                        w.bits,
                        w.stride,
                        include_str!("../kernels/prepare_integer_register.loom"),
                    )?;
                    self.quantized.clear_activation();
                    packed
                } else if shared {
                    self.prepare_quantized_input(x, &w)?
                } else {
                    crate::quantized::prepare(self, x, w.bits, w.stride)?
                };
                (
                    a,
                    w.codes,
                    scales,
                    w.scales,
                    w.rows / EXPERTS,
                    w.stride,
                    w.bits,
                )
            } else {
                ensure!(up.is_none(), "fused expert packing requires INT4 weights");
                let w = self.w(&weight)?;
                ensure!(
                    w.cols == x.cols && w.layout == Layout::Tiled && w.rows.is_multiple_of(EXPERTS),
                    "invalid expert matrix"
                );
                let rows = w.rows / EXPERTS;
                // Scale bindings are unused by the specialized BF16 kernel.
                (x.clone(), w, x.clone(), x.clone(), rows, x.cols, 0)
            };
        ensure!(
            x.rows == routes.rows * if input_routed { topk } else { 1 },
            "expert input routes mismatch"
        );
        ensure!(
            rows.is_multiple_of(256) && x.cols.is_multiple_of(256),
            "unsupported Omni expert tile shape"
        );
        let out = self.alloc(routes.rows * topk, rows, DType::Bf16)?;
        ensure!(
            !compact || (bits == 4 && rows <= 2048),
            "compact expert schedule requires INT4 and at most 2048 output channels"
        );
        let source = if groups.is_some() {
            if compact {
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/omni_expert_i4_compact.loom")
                )
            } else if bits == 4 {
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/omni_expert_i4.loom")
                )
            } else {
                concat!(
                    include_str!("../kernels/common.loom"),
                    include_str!("../kernels/omni_expert_bf16.loom")
                )
            }
        } else {
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/omni_expert.loom")
            )
        };
        let source = if self.omni_fast_defaults && groups.is_some() && !compact {
            fast_expert_source(bits, routes.rows).unwrap_or(source)
        } else {
            source
        };
        #[cfg(feature = "bench-internals")]
        let experimental =
            if !compact && groups.is_some() && routes.rows >= self.omni_kernel_min_tokens {
                self.omni_kernel_dir
                    .as_ref()
                    .map(|dir| {
                        let file = if bits == 4 {
                            "omni_expert_i4.loom"
                        } else {
                            "omni_expert_bf16.loom"
                        };
                        std::fs::read_to_string(dir.join(file)).map(|source| {
                            format!("{}\n{source}", include_str!("../kernels/common.loom"))
                        })
                    })
                    .transpose()?
            } else {
                None
            };
        #[cfg(feature = "bench-internals")]
        let source = experimental.as_deref().unwrap_or(source);
        self.dispatch(
            source,
            if groups.is_some() {
                EXPERTS * (rows / 64) * 32
            } else {
                out.len() * 32
            },
            &[
                ("x", &a),
                ("w", &w),
                ("routes", groups.unwrap_or(routes)),
                ("xs", &xs),
                ("ws", &ws),
            ],
            &out,
            &[
                ("tokens", routes.rows.to_string()),
                ("k", x.cols.to_string()),
                ("n", rows.to_string()),
                ("topk", topk.to_string()),
                ("input_routed", usize::from(input_routed).to_string()),
                ("bits", bits.to_string()),
                ("stride", stride.to_string()),
            ],
        )?;
        Ok(out)
    }
    pub(crate) fn omni_combine(&mut self, values: &Tensor, routes: &Tensor) -> Result<Tensor> {
        let topk = routes.cols / 2;
        ensure!(
            values.rows == routes.rows * topk,
            "expert output routes mismatch"
        );
        let out = self.alloc(routes.rows, values.cols, DType::Bf16)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/omni_combine.loom")
            ),
            out.len(),
            &[("x", values), ("routes", routes)],
            &out,
            &[
                ("width", values.cols.to_string()),
                ("topk", topk.to_string()),
            ],
        )?;
        Ok(out)
    }
}

pub(crate) fn text_rope(positions: &[[u32; 3]]) -> Vec<half::bf16> {
    positions
        .iter()
        .flat_map(|pos| {
            let angles: Vec<_> = (0..64)
                .map(|i| {
                    let axis = if i < 60 && i % 3 != 0 { i % 3 } else { 0 };
                    pos[axis] as f32 / 1_000_000f32.powf(i as f32 / 64.)
                })
                .collect();
            [false, true].into_iter().flat_map(move |sin| {
                let angles = angles.clone();
                (0..128).map(move |i| {
                    half::bf16::from_f32(if sin {
                        angles[i % 64].sin()
                    } else {
                        angles[i % 64].cos()
                    })
                })
            })
        })
        .collect()
}

pub(crate) fn padded_positions(positions: &[[u32; 3]], padded: usize) -> Result<Vec<[u32; 3]>> {
    ensure!(padded >= positions.len(), "Omni padded position extent");
    let last = *positions
        .last()
        .ok_or_else(|| anyhow::anyhow!("empty Omni positions"))?;
    let mut extended = positions.to_vec();
    for step in 1..=padded - positions.len() {
        let mut position = last;
        for value in &mut position {
            *value = value
                .checked_add(step as u32)
                .ok_or_else(|| anyhow::anyhow!("Omni position overflow"))?;
        }
        extended.push(position);
    }
    Ok(extended)
}

/// Prefix tensors own their allocator leases across graph resets.
pub(crate) struct LayerKv {
    pub key: Tensor,
    pub value: Tensor,
}
pub(crate) enum KvMode<'a> {
    None,
    Capture {
        tokens: usize,
        layers: &'a mut Vec<LayerKv>,
    },
    Resume(&'a [LayerKv]),
}

pub(crate) fn forward(
    e: &mut Engine,
    x: Tensor,
    positions: &[[u32; 3]],
    deepstack: &[Vec<(usize, Tensor)>],
) -> Result<Tensor> {
    forward_cached(e, x, positions, deepstack, &mut KvMode::None)
}

pub(crate) fn forward_cached(
    e: &mut Engine,
    x: Tensor,
    positions: &[[u32; 3]],
    deepstack: &[Vec<(usize, Tensor)>],
    cache: &mut KvMode<'_>,
) -> Result<Tensor> {
    let prefix = match cache {
        KvMode::Resume(layers) => {
            ensure!(layers.len() == 48, "Omni prefix layer count");
            ensure!(deepstack.is_empty(), "suffix cannot contain media features");
            layers[0].key.rows
        }
        KvMode::Capture { tokens, layers } => {
            ensure!(
                *tokens > 0 && *tokens < x.rows && layers.is_empty(),
                "Omni prefix capture extent"
            );
            0
        }
        KvMode::None => 0,
    };
    #[cfg(feature = "bench-internals")]
    let alignment = if e.omni_fast_defaults {
        128
    } else {
        e.omni_alignment
    };
    #[cfg(not(feature = "bench-internals"))]
    let alignment: usize = 128;
    if alignment != 1 {
        ensure!(
            alignment.is_power_of_two() && alignment <= 256,
            "Omni padding alignment"
        );
        let actual = x.rows;
        let rounded = actual.div_ceil(alignment) * alignment;
        let padded = if prefix + rounded <= crate::encoding::MAX_CONTEXT_TOKENS {
            rounded
        } else {
            actual
        };
        ensure!(
            prefix + padded <= crate::encoding::MAX_CONTEXT_TOKENS,
            "Omni padded context limit"
        );
        if actual != padded {
            // Padding follows every real token. Causal attention prevents it
            // from influencing real queries; remove it before the decision head.
            let zeros = e.input(
                padded - actual,
                x.cols,
                DType::Bf16,
                &vec![0u8; (padded - actual) * x.cols * 2],
            )?;
            let x = e.cat_rows(&[x, zeros])?;
            let extended = padded_positions(positions, padded)?;
            return Ok(
                forward_layers_cached(e, x, &extended, deepstack, 0..48, cache)?
                    .slice_rows(0, actual),
            );
        }
    }
    forward_layers_cached(e, x, positions, deepstack, 0..48, cache)
}
#[cfg(test)]
fn forward_layers(
    e: &mut Engine,
    x: Tensor,
    positions: &[[u32; 3]],
    deepstack: &[Vec<(usize, Tensor)>],
    layers: std::ops::Range<usize>,
) -> Result<Tensor> {
    forward_layers_cached(e, x, positions, deepstack, layers, &mut KvMode::None)
}
fn forward_layers_cached(
    e: &mut Engine,
    mut x: Tensor,
    positions: &[[u32; 3]],
    deepstack: &[Vec<(usize, Tensor)>],
    layers: std::ops::Range<usize>,
    cache: &mut KvMode<'_>,
) -> Result<Tensor> {
    ensure!(
        x.cols == 2048 && x.rows == positions.len(),
        "Omni text input shape"
    );
    let n = x.rows;
    #[cfg(feature = "bench-internals")]
    let chunk = if e.omni_fast_defaults {
        2048
    } else {
        e.omni_chunk
    };
    #[cfg(not(feature = "bench-internals"))]
    let chunk = 2048;
    ensure!(
        (1..=MAX_EXPERT_CHUNK).contains(&chunk),
        "Omni expert chunk limit"
    );
    let cs = e.input(
        n,
        256,
        DType::Bf16,
        bytemuck::cast_slice(&text_rope(positions)),
    )?;
    let normalize = layers.end == 48;
    for layer in layers {
        let p = format!("model.language_model.layers.{layer}");
        let norm = e.named_norm(&x, &format!("{p}.input_layernorm"), false, 1e-6)?;
        let a = format!("{p}.self_attn");
        let q = e.project(&norm, &format!("{a}.q_proj"), false)?;
        let q = e.omni_normalized_rope(&q, &format!("{a}.q_norm"), 32, &cs)?;
        let k = e.project(&norm, &format!("{a}.k_proj"), false)?;
        let k = e.omni_normalized_rope(&k, &format!("{a}.k_norm"), 4, &cs)?;
        let v = e.project(&norm, &format!("{a}.v_proj"), false)?;
        let attn = match &mut *cache {
            KvMode::Resume(layers) => {
                let saved = &layers[layer];
                let keys = e.cat_rows(&[saved.key.clone(), k])?;
                let values = e.cat_rows(&[saved.value.clone(), v])?;
                e.attention_suffix(&q, &keys, &values, saved.key.rows)?
            }
            mode => {
                if let KvMode::Capture { tokens, layers } = mode {
                    let key = e.alloc(*tokens, k.cols, k.dtype)?;
                    let value = e.alloc(*tokens, v.cols, v.dtype)?;
                    e.copy_into(&k.slice_rows(0, *tokens), &key)?;
                    e.copy_into(&v.slice_rows(0, *tokens), &value)?;
                    layers.push(LayerKv { key, value });
                }
                e.attention(&q, &k, &v, 32, 4, true)?
            }
        };
        let attn = e.project(&attn, &format!("{a}.o_proj"), false)?;
        x = e.binary(&x, &attn, "addf")?;
        let norm = e.named_norm(&x, &format!("{p}.post_attention_layernorm"), false, 1e-6)?;
        let mut chunks = Vec::new();
        for start in (0..n).step_by(chunk) {
            let norm = norm.slice_rows(start, (n - start).min(chunk));
            let logits = e.project(&norm, &format!("{p}.mlp.gate"), false)?;
            let routes = e.omni_route(&logits, TOP_K)?;
            #[cfg(feature = "bench-internals")]
            let reference = std::env::var_os("CLEF_BENCH_OMNI_REFERENCE").is_some();
            #[cfg(not(feature = "bench-internals"))]
            let reference = false;
            #[cfg(feature = "bench-internals")]
            let ordered = e.omni_fast_defaults || e.omni_sorted_combine;
            #[cfg(not(feature = "bench-internals"))]
            let ordered = e.omni_fast_defaults;
            #[cfg(feature = "bench-internals")]
            let compact = e.omni_compact_experts && !e.quantized.is_bf16();
            #[cfg(not(feature = "bench-internals"))]
            let compact = false;
            let (groups, order) = if reference {
                (None, None)
            } else if compact {
                let (groups, order) = e.omni_compact_routes(&routes)?;
                (Some(groups), ordered.then_some(order))
            } else if ordered {
                let (groups, order) = e.omni_group_routes_ordered(&routes)?;
                (Some(groups), Some(order))
            } else {
                (Some(e.omni_group_routes(&routes)?), None)
            };
            let down = format!("{p}.mlp.experts.down_proj");
            #[cfg(feature = "bench-internals")]
            let fused_gate_up = if e.omni_gate_up_source.is_some() {
                ensure!(
                    !e.omni_pack_swiglu,
                    "gate/up fusion and direct SwiGLU packing are mutually exclusive"
                );
                let groups = groups
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("gate/up fusion requires grouped routes"))?;
                Some(e.omni_gate_up(
                    &norm,
                    &routes,
                    groups,
                    &format!("{p}.mlp.experts.gate_proj"),
                    &format!("{p}.mlp.experts.up_proj"),
                )?)
            } else {
                None
            };
            #[cfg(not(feature = "bench-internals"))]
            let fused_gate_up: Option<Tensor> = None;
            let values = if let Some(hidden) = fused_gate_up {
                e.omni_expert_impl(&hidden, &routes, groups.as_ref(), &down, true)?
            } else {
                let gate = e.omni_expert_impl(
                    &norm,
                    &routes,
                    groups.as_ref(),
                    &format!("{p}.mlp.experts.gate_proj"),
                    false,
                )?;
                let up = e.omni_expert_impl(
                    &norm,
                    &routes,
                    groups.as_ref(),
                    &format!("{p}.mlp.experts.up_proj"),
                    false,
                )?;
                #[cfg(feature = "bench-internals")]
                let fused = e.omni_fast_defaults || e.omni_fused_mlp;
                #[cfg(not(feature = "bench-internals"))]
                let fused = e.omni_fast_defaults;
                #[cfg(feature = "bench-internals")]
                let pack_swiglu = e.omni_pack_swiglu
                    && e.quantized.weights.contains_key(&format!("{down}.weight"));
                #[cfg(not(feature = "bench-internals"))]
                let pack_swiglu = false;
                if pack_swiglu {
                    e.omni_expert_inputs(&gate, Some(&up), &routes, groups.as_ref(), &down, true)?
                } else {
                    let hidden = if fused {
                        e.swiglu(&gate, &up)?
                    } else {
                        let gate = e.unary(&gate, "silu")?;
                        e.binary(&gate, &up, "mulf")?
                    };
                    e.omni_expert_impl(&hidden, &routes, groups.as_ref(), &down, true)?
                }
            };
            let values = if let Some(order) = &order {
                e.omni_combine_ordered(&values, &routes, order)?
            } else {
                e.omni_combine(&values, &routes)?
            };
            chunks.push(values);
        }
        let values = e.cat_rows(&chunks)?;
        x = e.binary(&x, &values, "addf")?;
        if let Some(features) = deepstack.get(layer) {
            for (start, feature) in features {
                let target = x.slice_rows(*start, feature.rows);
                let added = e.binary(&target, feature, "addf")?;
                e.copy_into(&added, &target)?;
            }
        }
        e.trace(format!("layer_{layer}"), &x);
    }
    if normalize {
        x = e.named_norm(&x, "model.language_model.norm", false, 1e-6)?;
        e.trace("hidden", &x);
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_kernel_sources_and_thresholds() {
        use sha2::{Digest, Sha256};
        // Pin the sources actually used by the completed evaluation runs.
        for (source, expected) in [
            (
                include_bytes!("../kernels/omni_attention_transposed.loom").as_slice(),
                "3b819d2f638449880eb58446d825e6130e21855d6b55b33bae5c1666c67eaf41",
            ),
            (
                include_bytes!("../kernels/omni_expert_bf16_m64.loom").as_slice(),
                "d542ea05ea4c963e008b291bc3779b5767261bf748f3ea10f4edd5832ccf7d30",
            ),
            (
                include_bytes!("../kernels/omni_expert_i4_k128_down.loom").as_slice(),
                "cbc7ea40c31cc05555d3f1b82ff1c1fa65422433d33adbaa592d8e7889f5cb08",
            ),
        ] {
            assert_eq!(format!("{:x}", Sha256::digest(source)), expected);
        }
        for (bits, minimum) in [(0, 128), (4, 512)] {
            assert!(fast_expert_source(bits, minimum - 1).is_none());
            assert!(fast_expert_source(bits, minimum).is_some());
            assert!(fast_expert_source(bits, MAX_EXPERT_CHUNK).is_some());
        }
        assert!(fast_expert_source(8, MAX_EXPERT_CHUNK).is_none());
    }

    #[test]
    #[ignore = "requires gfx1151; checks stable parallel route compaction and graph replay"]
    fn grouped_routes_match_cpu() -> Result<()> {
        let context = hrx::inference::ModelContext::new(Default::default())?;
        for tokens in [
            1, 17, 64, 65, 128, 255, 256, 511, 512, 1023, 1024, 2047, 2048,
        ] {
            let mut e = Engine::new(&context)?;
            let mut data = vec![0f32; tokens * TOP_K * 2];
            let routes = e.input(tokens, TOP_K * 2, DType::F32, bytemuck::cast_slice(&data))?;
            let groups = e.omni_group_routes(&routes)?;
            let mut graph = e.finish()?;
            // Replay with different expert membership, including empty experts and
            // a completely full expert row. IDs remain distinct within each token.
            for phase in 0..3 {
                let mut expected = vec![Vec::new(); EXPERTS];
                for token in 0..tokens {
                    for rank in 0..TOP_K {
                        let expert =
                            (if phase == 1 { 0 } else { token * 13 } + rank * 7 + phase * 19)
                                % EXPERTS;
                        data[token * TOP_K * 2 + rank] = expert as f32;
                        expected[expert].push((token * TOP_K + rank) as f32);
                    }
                }
                e.upload(&routes, bytemuck::cast_slice(&data))?;
                e.run(&mut graph)?;
                let actual = e.read_f32(&groups)?;
                for (expert, ordinals) in expected.iter().enumerate() {
                    let base = expert * (tokens + 1);
                    assert_eq!(actual[base], ordinals.len() as f32);
                    assert_eq!(&actual[base + 1..base + 1 + ordinals.len()], ordinals);
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151; checks routing, selected experts and weighted accumulation"]
    fn routed_experts_match_sparse_cpu_oracle() -> Result<()> {
        let context = hrx::inference::ModelContext::new(Default::default())?;
        for precision in [crate::Precision::Bf16, crate::Precision::Int4] {
            let mut e = Engine::new(&context)?;
            e.quantized = crate::quantized::QuantizedProjections::for_precision(precision);
            let logits: Vec<_> = (0..3 * EXPERTS)
                .map(|i| half::bf16::from_f32(((i * 37 + i / EXPERTS * 11) % EXPERTS) as f32 / 16.))
                .collect();
            let input: Vec<_> = (0..3 * 256)
                .map(|i| half::bf16::from_f32((i % 17) as f32 / 16. - 0.5))
                .collect();
            let weights: Vec<_> = (0..EXPERTS * 256 * 256)
                .map(|i| {
                    let row = i / 256;
                    half::bf16::from_f32(if i % 256 == row % 256 {
                        (row / 256 + 1) as f32 / 128.
                    } else {
                        0.
                    })
                })
                .collect();
            let bytes =
                crate::checkpoint::tile_rows(bytemuck::cast_slice(&weights), EXPERTS * 256, 256);
            let name = "model.language_model.layers.0.mlp.experts.up_proj.weight";
            if !e.load_quantized_weight(name, EXPERTS * 256, 256, &bytes)? {
                e.weight(name.into(), EXPERTS * 256, 256, &bytes)?;
            }
            let x = e.input(3, 256, DType::Bf16, bytemuck::cast_slice(&input))?;
            let l = e.input(3, EXPERTS, DType::Bf16, bytemuck::cast_slice(&logits))?;
            let routes = e.omni_route(&l, TOP_K)?;
            let groups = e.omni_group_routes(&routes)?;
            let values = e.omni_expert_impl(
                &x,
                &routes,
                Some(&groups),
                "model.language_model.layers.0.mlp.experts.up_proj",
                false,
            )?;
            let values = e.omni_expert_impl(
                &values,
                &routes,
                Some(&groups),
                "model.language_model.layers.0.mlp.experts.up_proj",
                true,
            )?;
            let out = e.omni_combine(&values, &routes)?;
            let mut graph = e.finish()?;
            for _ in 0..2 {
                e.run(&mut graph)?;
                let actual_routes = e.read_f32(&routes)?;
                let actual = e.read_f32(&out)?;
                for token in 0..3 {
                    let mut ranked: Vec<_> = (0..EXPERTS).collect();
                    ranked.sort_by(|&a, &b| {
                        logits[token * EXPERTS + b]
                            .partial_cmp(&logits[token * EXPERTS + a])
                            .unwrap()
                    });
                    for rank in 0..TOP_K {
                        assert_eq!(
                            actual_routes[token * TOP_K * 2 + rank] as usize,
                            ranked[rank]
                        );
                    }
                    let max = logits[token * EXPERTS + ranked[0]].to_f32();
                    let mut selected: Vec<_> = ranked[..TOP_K]
                        .iter()
                        .map(|&expert| {
                            (
                                expert,
                                (logits[token * EXPERTS + expert].to_f32() - max).exp(),
                            )
                        })
                        .collect();
                    let sum: f32 = selected.iter().map(|p| p.1).sum();
                    for (_, p) in &mut selected {
                        *p = half::bf16::from_f32(*p / sum).to_f32();
                    }
                    selected.sort_by_key(|p| p.0);
                    for col in 0..256 {
                        let mut expected = 0.;
                        for &(expert, p) in &selected {
                            let y = half::bf16::from_f32(
                                input[token * 256 + col].to_f32() * (expert + 1) as f32 / 128.,
                            )
                            .to_f32();
                            let y = half::bf16::from_f32(y * (expert + 1) as f32 / 128.).to_f32();
                            let y = half::bf16::from_f32(y * p).to_f32();
                            expected = half::bf16::from_f32(expected + y).to_f32();
                        }
                        let error = (actual[token * 256 + col] - expected).abs();
                        assert!(
                            error
                                <= if precision == crate::Precision::Bf16 {
                                    0.
                                } else {
                                    0.08
                                },
                            "{precision:?}: {error}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires gfx1151; compares Omni RMSNorm and RoPE with upstream classes"]
    fn norm_and_rope_match_upstream() -> Result<()> {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/omni/operators.json"))?;
        let values = |name: &str| -> Vec<half::bf16> {
            fixture[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| half::bf16::from_f64(v.as_f64().unwrap()))
                .collect()
        };
        let context = hrx::inference::ModelContext::new(Default::default())?;
        let mut e = Engine::new(&context)?;
        e.architecture = crate::ModelVariant::ClefOmni.architecture();
        e.weight(
            "test.weight".into(),
            1,
            128,
            bytemuck::cast_slice(&values("norm_weight")),
        )?;
        let x = e.input(3, 128, DType::Bf16, bytemuck::cast_slice(&values("x")))?;
        let norm = e.named_norm(&x, "test", false, 1e-6)?;
        let positions: Vec<[u32; 3]> = fixture["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| std::array::from_fn(|i| v[i].as_u64().unwrap() as u32))
            .collect();
        let cs = e.input(
            3,
            256,
            DType::Bf16,
            bytemuck::cast_slice(&text_rope(&positions)),
        )?;
        let rotated = e.rope(&norm, 1, 128, &cs)?;
        let mut graph = e.finish()?;
        e.run(&mut graph)?;
        for (tensor, key) in [(&norm, "normalized"), (&rotated, "rotated")] {
            let actual = e.read_f32(tensor)?;
            let expected = fixture[key].as_array().unwrap();
            let error = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (*a as f64 - b.as_f64().unwrap()).abs())
                .fold(0., f64::max);
            assert!(error <= 0.015625, "{key} maximum error {error}");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires gfx1151; 128-wide GQA, causal masking and audio convolution oracle"]
    fn attention_and_audio_conv_match_torch() -> Result<()> {
        let context = hrx::inference::ModelContext::new(Default::default())?;
        let mut e = Engine::new(&context)?;
        let conv: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/omni/conv.json"))?;
        let bf16 = |fixture: &serde_json::Value, name: &str| -> Vec<half::bf16> {
            fixture[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| half::bf16::from_f64(v.as_f64().unwrap()))
                .collect()
        };
        e.weight(
            "conv.weight".into(),
            4,
            27,
            bytemuck::cast_slice(&bf16(&conv, "w")),
        )?;
        e.weight(
            "conv.bias".into(),
            1,
            4,
            bytemuck::cast_slice(&bf16(&conv, "b")),
        )?;
        let x = e.input(63, 3, DType::Bf16, bytemuck::cast_slice(&bf16(&conv, "x")))?;
        let (y, h, w) = e.audio_conv(&x, 7, 9, "conv")?;
        assert_eq!((h, w), (4, 5));
        let attention: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/omni/attention.json"))?;
        let q = e.input(
            7,
            512,
            DType::Bf16,
            bytemuck::cast_slice(&bf16(&attention, "q")),
        )?;
        let k = e.input(
            7,
            256,
            DType::Bf16,
            bytemuck::cast_slice(&bf16(&attention, "k")),
        )?;
        let v = e.input(
            7,
            256,
            DType::Bf16,
            bytemuck::cast_slice(&bf16(&attention, "v")),
        )?;
        let causal = e.attention(&q, &k, &v, 4, 2, true)?;
        let noncausal = e.attention(&q, &k, &v, 4, 2, false)?;
        let mut graph = e.finish()?;
        for _ in 0..2 {
            e.run(&mut graph)?;
            for (tensor, expected, tolerance) in [
                (&y, &conv["y"], 0.008),
                (&causal, &attention["True"], 0.004),
                (&noncausal, &attention["False"], 0.004),
            ] {
                let actual = e.read_f32(tensor)?;
                let expected = expected.as_array().unwrap();
                assert_eq!(actual.len(), expected.len());
                let error = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (*a as f64 - b.as_f64().unwrap()).abs())
                    .fold(0., f64::max);
                assert!(error <= tolerance, "maximum error {error} > {tolerance}");
            }
        }
        Ok(())
    }
}
