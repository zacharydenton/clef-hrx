//! Native Qwen3-Omni thinker operators. Speech generation is not part of CLEF.
use crate::{
    Result,
    checkpoint::Checkpoint,
    gpu::{DType, Engine, Layout, Tensor},
};
use anyhow::ensure;

const EXPERTS: usize = 128;
const TOP_K: usize = 8;

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
fn load_expert_layer(e: &mut Engine, checkpoint: &Checkpoint, layer: usize) -> Result<()> {
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
            logits.rows,
            &[("x", logits)],
            &out,
            &[
                ("experts", logits.cols.to_string()),
                ("topk", topk.to_string()),
            ],
        )?;
        Ok(out)
    }
    pub(crate) fn omni_expert(
        &mut self,
        x: &Tensor,
        routes: &Tensor,
        name: &str,
        input_routed: bool,
    ) -> Result<Tensor> {
        let topk = routes.cols / 2;
        let weight = format!("{name}.weight");
        let (a, w, xs, ws, rows, stride, bits) =
            if let Some(w) = self.quantized.weights.get(&weight).cloned() {
                ensure!(
                    w.bits == 4 && w.cols == x.cols,
                    "unsupported Omni expert packing"
                );
                let (a, scales) = crate::quantized::prepare(self, x, w.bits, w.stride)?;
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
        let out = self.alloc(routes.rows * topk, rows, DType::Bf16)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/omni_expert.loom")
            ),
            out.len() * 32,
            &[
                ("x", &a),
                ("w", &w),
                ("routes", routes),
                ("xs", &xs),
                ("ws", &ws),
            ],
            &out,
            &[
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

pub(crate) fn forward(
    e: &mut Engine,
    x: Tensor,
    positions: &[[u32; 3]],
    deepstack: &[Vec<(usize, Tensor)>],
) -> Result<Tensor> {
    forward_layers(e, x, positions, deepstack, 0..48)
}
fn forward_layers(
    e: &mut Engine,
    mut x: Tensor,
    positions: &[[u32; 3]],
    deepstack: &[Vec<(usize, Tensor)>],
    layers: std::ops::Range<usize>,
) -> Result<Tensor> {
    ensure!(
        x.cols == 2048 && x.rows == positions.len(),
        "Omni text input shape"
    );
    let n = x.rows;
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
        let q = e
            .project(&norm, &format!("{a}.q_proj"), false)?
            .reshape(n * 32, 128);
        let q = e
            .named_norm(&q, &format!("{a}.q_norm"), false, 1e-6)?
            .reshape(n, 4096);
        let q = e.rope(&q, 32, 128, &cs)?;
        let k = e
            .project(&norm, &format!("{a}.k_proj"), false)?
            .reshape(n * 4, 128);
        let k = e
            .named_norm(&k, &format!("{a}.k_norm"), false, 1e-6)?
            .reshape(n, 512);
        let k = e.rope(&k, 4, 128, &cs)?;
        let v = e.project(&norm, &format!("{a}.v_proj"), false)?;
        let attn = e.attention(&q, &k, &v, 32, 4, true)?;
        let attn = e.project(&attn, &format!("{a}.o_proj"), false)?;
        x = e.binary(&x, &attn, "addf")?;
        let norm = e.named_norm(&x, &format!("{p}.post_attention_layernorm"), false, 1e-6)?;
        let mut chunks = Vec::new();
        for start in (0..n).step_by(256) {
            let norm = norm.slice_rows(start, (n - start).min(256));
            let logits = e.project(&norm, &format!("{p}.mlp.gate"), false)?;
            let routes = e.omni_route(&logits, TOP_K)?;
            let gate =
                e.omni_expert(&norm, &routes, &format!("{p}.mlp.experts.gate_proj"), false)?;
            let up = e.omni_expert(&norm, &routes, &format!("{p}.mlp.experts.up_proj"), false)?;
            let gate = e.unary(&gate, "silu")?;
            let hidden = e.binary(&gate, &up, "mulf")?;
            let values = e.omni_expert(
                &hidden,
                &routes,
                &format!("{p}.mlp.experts.down_proj"),
                true,
            )?;
            let values = e.omni_combine(&values, &routes)?;
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
            let values = e.omni_expert(
                &x,
                &routes,
                "model.language_model.layers.0.mlp.experts.up_proj",
                false,
            )?;
            let values = e.omni_expert(
                &values,
                &routes,
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
