//! Optional rotated INT4 projections; BF16 remains the default.
use crate::{
    Result,
    gpu::{DType, Engine, Tensor},
    model::Precision,
};
use anyhow::ensure;
#[cfg(test)]
use hrx::inference::ModelContext;
use hrx::{Constants, loom::Specialization};

#[derive(Clone)]
pub(crate) struct QuantizedWeight {
    codes: Tensor,
    scales: Tensor,
    rows: usize,
    cols: usize,
    stride: usize,
    bits: usize,
}

#[derive(Default)]
pub(crate) struct QuantizedProjections {
    bits: Option<usize>,
    backbone_bits: Option<usize>,
    projections: Vec<String>,
    share_packing: bool,
    activation: Option<PreparedActivation>,
    pub weights: std::collections::HashMap<String, QuantizedWeight>,
}
struct PreparedActivation {
    // Hold the source lease: allocation slots may otherwise be reused within a graph.
    input: Tensor,
    codes: Tensor,
    scales: Tensor,
    bits: usize,
    stride: usize,
}
impl QuantizedProjections {
    pub(crate) fn for_precision(precision: Precision) -> Self {
        Self {
            bits: (precision == Precision::Int4Mlp).then_some(4),
            backbone_bits: (precision == Precision::Int4).then_some(4),
            projections: vec!["up".into(), "gate".into(), "down".into()],
            share_packing: precision != Precision::Bf16,
            ..Default::default()
        }
    }

    #[cfg(feature = "bench-internals")]
    pub(crate) fn from_env() -> Result<Self> {
        let bits = std::env::var("CLEF_EXPERIMENT_MLP_BITS")
            .ok()
            .map(|s| s.parse::<usize>())
            .transpose()?;
        ensure!(
            bits.is_none_or(|b| [4, 8].contains(&b)),
            "MLP bits must be 4 or 8"
        );
        let backbone_bits = std::env::var("CLEF_EXPERIMENT_BACKBONE_BITS")
            .ok()
            .map(|s| s.parse::<usize>())
            .transpose()?;
        ensure!(
            backbone_bits.is_none_or(|b| [4, 8].contains(&b)),
            "backbone bits must be 4 or 8"
        );
        let projections: Vec<_> = std::env::var("CLEF_EXPERIMENT_MLP_PROJECTIONS")
            .unwrap_or_else(|_| "up,gate,down".into())
            .split(',')
            .map(str::to_owned)
            .collect();
        ensure!(
            projections
                .iter()
                .all(|p| ["up", "gate", "down"].contains(&p.as_str())),
            "unknown MLP projection"
        );
        Ok(Self {
            bits,
            backbone_bits,
            projections,
            share_packing: std::env::var("CLEF_EXPERIMENT_SHARE_PACKING").as_deref() == Ok("1"),
            ..Self::default()
        })
    }
    pub(crate) fn clear_activation(&mut self) {
        self.activation = None;
    }
    fn precision(&self, name: &str, rows: usize, cols: usize) -> Option<usize> {
        if !name.starts_with("model.language_model.layers.") || !name.ends_with(".weight") {
            return None;
        }
        if name.contains(".mlp.") {
            return self
                .projections
                .iter()
                .any(|p| name.ends_with(&format!(".mlp.{p}_proj.weight")))
                .then_some(self.bits.or(self.backbone_bits))
                .flatten();
        }
        (rows >= 4096 && cols >= 1024 && rows.is_multiple_of(256) && cols.is_multiple_of(256))
            .then_some(self.backbone_bits)
            .flatten()
    }
    pub(crate) fn validate_loaded(&self) -> Result<()> {
        if self.bits.is_some() || self.backbone_bits.is_some() {
            ensure!(
                !self.weights.is_empty(),
                "integer experiment selected no checkpoint weights"
            );
            for bits in [4, 8] {
                let count = self.weights.values().filter(|w| w.bits == bits).count();
                if count > 0 {
                    eprintln!("INT{bits}: {count} packed backbone projection weights");
                }
            }
        }
        Ok(())
    }
}

impl Engine {
    /// Replace selected immutable projection weights at load time. No dense copy survives.
    pub(crate) fn load_quantized_weight(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
        bytes: &[u8],
    ) -> Result<bool> {
        let Some(bits) = self.quantized.precision(name, rows, cols) else {
            return Ok(false);
        };
        ensure!(
            rows.is_multiple_of(256) && cols.is_multiple_of(256),
            "unsupported quantized weight shape"
        );
        let stride = cols
            + if cols.is_multiple_of(1024) {
                512 / bits
            } else {
                0
            };
        let dense = self.input(rows, cols, DType::Bf16, bytes)?;
        let (codes, scales) = prepare(self, &dense, bits, stride)?;
        let mut graph = self.finish()?;
        self.run(&mut graph)?;
        self.quantized.weights.insert(
            name.into(),
            QuantizedWeight {
                codes,
                scales,
                rows,
                cols,
                stride,
                bits,
            },
        );
        // The map keeps packed weights leased across reset and all later plans.
        // Drop staging leases so the next checkpoint tensor can reuse its upload buffer.
        self.reset_plan()?;
        Ok(true)
    }
    pub(crate) fn project_quantized(&mut self, x: &Tensor, w: &QuantizedWeight) -> Result<Tensor> {
        ensure!(
            x.cols == w.cols && x.dtype == DType::Bf16,
            "quantized projection shape/dtype mismatch"
        );
        let cached = self
            .quantized
            .activation
            .as_ref()
            .filter(|a| a.input.same_view(x) && a.bits == w.bits && a.stride == w.stride);
        let (a, scales) = if let Some(cached) = cached {
            (cached.codes.clone(), cached.scales.clone())
        } else {
            let (a, scales) = prepare(self, x, w.bits, w.stride)?;
            if self.quantized.share_packing {
                self.quantized.activation = Some(PreparedActivation {
                    input: x.clone(),
                    codes: a.clone(),
                    scales: scales.clone(),
                    bits: w.bits,
                    stride: w.stride,
                });
            }
            (a, scales)
        };
        let out = self.alloc(x.rows, w.rows, DType::Bf16)?;
        let mut spec = Specialization::new(format!("h3_gemm_i{}_256", w.bits));
        for (key, value) in [
            ("k_size", w.cols),
            ("n_size", w.rows),
            ("k_stride", w.stride),
            ("m_group", x.rows.div_ceil(256).min(4)),
        ] {
            spec.set_config(format!("h3.gemm_i{}_256.{key}", w.bits), value.to_string());
        }
        let mut constants = Constants::new();
        constants.push(x.rows as u32)?;
        self.emit_special(
            include_str!("../kernels/gemm_integer.loom"),
            spec,
            constants,
            &[x.rows as u64],
            &[&a, &w.codes, &w.scales, &scales],
            &out,
        )?;
        Ok(out)
    }
}

pub(crate) fn prepare(
    e: &mut Engine,
    x: &Tensor,
    bits: usize,
    stride: usize,
) -> Result<(Tensor, Tensor)> {
    prepare_with_source(
        e,
        x,
        bits,
        stride,
        if x.cols <= 6144 {
            include_str!("../kernels/prepare_integer_single.loom")
        } else {
            include_str!("../kernels/prepare_integer.loom")
        },
    )
}

pub(crate) fn prepare_with_source(
    e: &mut Engine,
    x: &Tensor,
    bits: usize,
    stride: usize,
    source: &str,
) -> Result<(Tensor, Tensor)> {
    ensure!(
        [4, 8].contains(&bits)
            && (256..=32768).contains(&x.cols)
            && x.cols.is_multiple_of(256)
            && (1..=1048576).contains(&x.rows)
            && x.dtype == DType::Bf16
            && (x.cols..=65536).contains(&stride)
            && stride.is_multiple_of(512 / bits),
        "unsupported rotated integer operand"
    );
    let words = stride / (32 / bits);
    let storage = e.alloc(1, x.rows * (words + 1), DType::U32)?;
    let mut codes = storage.clone();
    codes.rows = x.rows;
    codes.cols = words;
    let mut scales = storage.clone();
    scales.offset += x.rows * words * 4;
    scales.cols = x.rows;
    scales.dtype = DType::F32;
    let mut spec = Specialization::new(format!("h3_prepare_plain_tiled_i{bits}"));
    for (key, value) in [("width", x.cols), ("lanes", 256), ("out_stride", stride)] {
        spec.set_config(
            format!("h3.prepare_plain_tiled_i{bits}.{key}"),
            value.to_string(),
        );
    }
    let mut constants = Constants::new();
    constants.push(x.rows as u32)?;
    e.emit_special(source, spec, constants, &[x.rows as u64], &[x], &storage)?;
    // The returned views lease this storage through their last recorded read.
    // Graph access declarations order later scratch reuse; immutable weights
    // are kept alive by QuantizedProjections rather than pinning every activation.
    Ok((codes, scales))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotated(row: &[half::bf16]) -> Vec<f32> {
        let mut out: Vec<_> = row.iter().map(|x| x.to_f32()).collect();
        for group in out.chunks_exact_mut(256) {
            for stride in [1, 4, 16, 64] {
                for base in (0..256).step_by(4 * stride) {
                    for i in 0..stride {
                        let j = base + i;
                        let [a, b, c, d] = [
                            group[j],
                            group[j + stride],
                            group[j + 2 * stride],
                            group[j + 3 * stride],
                        ];
                        let (s01, s23, d01, d23) = (a + b, c + d, a - b, c - d);
                        group[j] = s01 + d23;
                        group[j + stride] = s01 - d23;
                        group[j + 2 * stride] = d01 + s23;
                        group[j + 3 * stride] = s23 - d01;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn explicit_precision_selects_only_requested_projections() {
        assert_eq!(crate::LoadOptions::default().precision, Precision::Bf16);
        let mlp = "model.language_model.layers.0.mlp.up_proj.weight";
        let qkv = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        for (mode, mlp_bits, qkv_bits) in [
            (Precision::Bf16, None, None),
            (Precision::Int4Mlp, Some(4), None),
            (Precision::Int4, Some(4), Some(4)),
        ] {
            let config = QuantizedProjections::for_precision(mode);
            assert_eq!(config.precision(mlp, 17408, 5120), mlp_bits);
            assert_eq!(config.precision(qkv, 10240, 5120), qkv_bits);
            assert_eq!(
                config.precision("model.visual.blocks.0.mlp.up_proj.weight", 17408, 5120),
                None
            );
            assert_eq!(config.precision("joint_head.weight", 5120, 5120), None);
            assert_eq!(
                config.precision(
                    "model.language_model.layers.3.self_attn.k_proj.weight",
                    1024,
                    5120
                ),
                None
            );
        }
    }

    #[test]
    fn rotation_preserves_inner_product() {
        let a: Vec<_> = (0..512)
            .map(|i| half::bf16::from_f32(((i * 17 % 71) as f32 - 35.) / 64.))
            .collect();
        let b: Vec<_> = (0..512)
            .map(|i| half::bf16::from_f32(((i * 31 % 59) as f32 - 29.) / 64.))
            .collect();
        let expected: f64 = a
            .iter()
            .zip(&b)
            .map(|(a, b)| f64::from(a.to_f32()) * f64::from(b.to_f32()))
            .sum();
        let actual: f64 = rotated(&a)
            .iter()
            .zip(rotated(&b))
            .map(|(a, b)| f64::from(*a) * f64::from(b) / 256.)
            .sum();
        assert!((actual - expected).abs() < 1e-8);
    }

    #[test]
    #[ignore = "requires gfx1151; checks bounded scratch across a deep quantized graph"]
    fn activation_scratch_is_bounded() -> Result<()> {
        let mut e = Engine::new(&ModelContext::new(Default::default())?)?;
        e.quantized = QuantizedProjections::for_precision(Precision::Int4Mlp);
        let name = "model.language_model.layers.0.mlp.up_proj.weight";
        let weights: Vec<_> = (0..256 * 256)
            .map(|i| half::bf16::from_f32(if i / 256 == i % 256 { 1. } else { 0. }))
            .collect();
        e.load_quantized_weight(name, 256, 256, bytemuck::cast_slice(&weights))?;
        let w = e.quantized.weights[name].clone();
        let values: Vec<_> = (0..17 * 256)
            .map(|i| half::bf16::from_f32(1. + (i / 256) as f32 / 16.))
            .collect();
        let mut x = e.input(17, 256, DType::Bf16, bytemuck::cast_slice(&values))?;
        let mut warmed_bytes = 0;
        for layer in 0..32 {
            x = e.project_quantized(&x, &w)?;
            if layer == 7 {
                warmed_bytes = e.allocated_bytes();
            }
            if layer > 7 {
                assert_eq!(
                    e.allocated_bytes(),
                    warmed_bytes,
                    "scratch grew at layer {layer}"
                );
            }
        }
        let mut graph = e.finish()?;
        let expected: Vec<_> = values.iter().map(|v| v.to_f32()).collect();
        for _ in 0..2 {
            e.run(&mut graph)?;
            assert_eq!(e.read_f32(&x)?, expected);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151; checks packed weight lifetime across loads and rebuilt plans"]
    fn packed_weights_survive_plan_reset() -> Result<()> {
        let context = ModelContext::new(Default::default())?;
        for (bits, share_packing) in [(4, false), (4, true), (8, false), (8, true)] {
            let mut e = Engine::new(&context)?;
            e.quantized.bits = Some(bits);
            e.quantized.share_packing = share_packing;
            e.quantized.projections = vec!["up".into()];
            let names = [
                "model.language_model.layers.0.mlp.up_proj.weight",
                "model.language_model.layers.1.mlp.up_proj.weight",
            ];
            let quantize = |row: &[half::bf16]| {
                let values = rotated(row);
                let max_code = ((1 << (bits - 1)) - 1) as f32;
                let scale = (values.iter().map(|v| v.abs()).fold(0f32, f32::max) / 16.).max(1e-30)
                    / max_code;
                let codes: Vec<_> = values
                    .iter()
                    .map(|v| {
                        (v * (0.0625 / scale))
                            .round_ties_even()
                            .clamp(-max_code, max_code) as i32
                    })
                    .collect();
                (codes, scale)
            };
            let mut weight_oracles = Vec::new();
            for (index, name) in names.iter().enumerate() {
                let values: Vec<_> = (0..256 * 256)
                    .map(|i| {
                        half::bf16::from_f32((((i * (17 + index * 4)) % 251) as f32 - 125.) / 128.)
                    })
                    .collect();
                assert!(e.load_quantized_weight(name, 256, 256, bytemuck::cast_slice(&values))?);
                weight_oracles.push(values.chunks_exact(256).map(quantize).collect::<Vec<_>>());
            }
            assert!(e.weights.is_empty());
            for rows in [1, 17, 1] {
                e.reset_plan()?;
                assert!(e.quantized.activation.is_none());
                let values: Vec<_> = (0..(rows + 1) * 256)
                    .map(|i| half::bf16::from_f32(((i * 31 % 257) as f32 - 128.) / 64.))
                    .collect();
                let input = e.input(rows + 1, 256, DType::Bf16, bytemuck::cast_slice(&values))?;
                let oracles: Vec<_> = values.chunks_exact(256).map(quantize).collect();
                // Shared input reuse, then a different view of the same allocation.
                // Matching only the slot or lease would return stale packed rows.
                let mut outputs = Vec::new();
                for start in [0, 1] {
                    let x = input.slice_rows(start, rows);
                    for (index, name) in names.iter().enumerate() {
                        let w = e.quantized.weights[*name].clone();
                        let output = e.project_quantized(&x, &w)?;
                        outputs.push((start, index, output));
                    }
                }
                let mut graph = e.finish()?;
                assert!(e.quantized.activation.is_none());
                e.run(&mut graph)?;
                for (start, index, output) in outputs {
                    let actual = e.read_f32(&output)?;
                    for row in 0..rows {
                        for col in 0..256 {
                            let (a, a_scale) = &oracles[row + start];
                            let (w, w_scale) = &weight_oracles[index][col];
                            let dot: i32 = a.iter().zip(w).map(|(a, w)| a * w).sum();
                            let expected =
                                half::bf16::from_f32(dot as f32 * a_scale * w_scale).to_f32();
                            let actual = actual[row * 256 + col];
                            assert!(
                                actual.is_finite()
                                    && (actual - expected).abs()
                                        <= actual.abs().max(expected.abs()) / 128. + 1e-6,
                                "INT{bits} rows={rows} row={row} col={col}: {actual} vs {expected}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151; validates GPU rotation, scale and packed codes"]
    fn rotated_packing_cpu_oracle() -> Result<()> {
        let context = ModelContext::new(Default::default())?;
        let mut e = Engine::new(&context)?;
        for bits in [4, 8] {
            for width in [256, 768, 1280, 5120, 6144, 8192, 17408] {
                let mut values = vec![half::bf16::ZERO; width * 3];
                values[width] = half::bf16::ONE;
                for i in 0..width {
                    values[width * 2 + i] =
                        half::bf16::from_f32(((i * 37 % 257) as f32 - 128.) / 91.);
                }
                let input = e.input(3, width, DType::Bf16, bytemuck::cast_slice(&values))?;
                let stride = width + 512 / bits;
                let (codes, scales) = prepare(&mut e, &input, bits, stride)?;
                let mut graph = e.finish()?;
                e.run(&mut graph)?;
                // read_f32 bitcasts four-byte storage; retain raw U32 code bits.
                let actual: Vec<_> = e.read_f32(&codes)?.into_iter().map(f32::to_bits).collect();
                let scales = e.read_f32(&scales)?;
                let max_code = ((1 << (bits - 1)) - 1) as f32;
                for row in 0..3 {
                    let rotation = rotated(&values[row * width..(row + 1) * width]);
                    let scale = (rotation.iter().map(|x| x.abs()).fold(0f32, f32::max) / 16.)
                        .max(1e-30)
                        / max_code;
                    assert!((scales[row] - scale).abs() <= scale * 1e-6);
                    let inv = 0.0625 / scale;
                    for col in 0..width {
                        let word = actual[row * codes.cols + col / (32 / bits)];
                        let code = (word >> ((col % (32 / bits)) * bits)) & ((1 << bits) - 1);
                        let expected = (rotation[col] * inv)
                            .round_ties_even()
                            .clamp(-max_code, max_code)
                            as i32 as u32
                            & ((1 << bits) - 1);
                        assert_eq!(
                            code, expected,
                            "bits={bits}, width={width}, row={row}, col={col}"
                        );
                    }
                }
                e.run(&mut graph)?;
                let replay: Vec<_> = e.read_f32(&codes)?.into_iter().map(f32::to_bits).collect();
                // Padding is unused; compare only logical columns.
                for row in 0..3 {
                    let start = row * codes.cols;
                    assert_eq!(
                        &actual[start..start + width / (32 / bits)],
                        &replay[start..start + width / (32 / bits)]
                    );
                }
                e.reset_plan()?;
            }
        }
        Ok(())
    }
}
