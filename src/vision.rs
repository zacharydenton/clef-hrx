use crate::{
    Result,
    gpu::{DType, Engine, Tensor},
    media::VisualItem,
};

fn positions(h: usize, w: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for by in 0..h / 2 {
        for bx in 0..w / 2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    out.push((by * 2 + dy, bx * 2 + dx));
                }
            }
        }
    }
    out
}
fn rope(h: usize, w: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for (y, x) in positions(h, w) {
        let angles: Vec<_> = [y, x]
            .into_iter()
            .flat_map(|p| (0..18).map(move |i| p as f32 / 10000f32.powf(i as f32 / 18.)))
            .collect();
        // Vision RoPE's cos/sin are FP32 in the reference. Stored as F32 below.
        for sin in [false, true] {
            for i in 0..72 {
                out.push(if sin {
                    angles[i % 36].sin()
                } else {
                    angles[i % 36].cos()
                });
            }
        }
    }
    out
}
fn position_embedding(e: &mut Engine, h: usize, w: usize) -> Result<Tensor> {
    let mut ids = Vec::new();
    let mut weights = Vec::new();
    for (y, x) in positions(h, w) {
        let yy = y as f32 * 47. / (h - 1) as f32;
        let xx = x as f32 * 47. / (w - 1) as f32;
        let y0 = yy as usize;
        let x0 = xx as usize;
        let y1 = (y0 + 1).min(47);
        let x1 = (x0 + 1).min(47);
        let dy = yy - y0 as f32;
        let dx = xx - x0 as f32;
        ids.extend([
            (y0 * 48 + x0) as u32,
            (y0 * 48 + x1) as u32,
            (y1 * 48 + x0) as u32,
            (y1 * 48 + x1) as u32,
        ]);
        weights.extend([
            (1. - dy) * (1. - dx),
            (1. - dy) * dx,
            dy * (1. - dx),
            dy * dx,
        ]);
    }
    let ids = e.input(h * w, 4, DType::U32, bytemuck::cast_slice(&ids))?;
    let weights = e.input(h * w, 4, DType::F32, bytemuck::cast_slice(&weights))?;
    let table = e.w("model.visual.pos_embed.weight")?;
    let out = e.alloc(h * w, 1152, DType::Bf16)?;
    e.dispatch(
        concat!(
            include_str!("../kernels/common.loom"),
            include_str!("../kernels/vision_position.loom")
        ),
        out.len(),
        &[("ids", &ids), ("weights", &weights), ("table", &table)],
        &out,
        &[],
    )?;
    Ok(out)
}
pub(crate) fn forward(e: &mut Engine, item: &VisualItem) -> Result<Tensor> {
    let [t, h, w] = item.grid;
    let n = h * w;
    let pos = position_embedding(e, h, w)?;
    let cs = e.input(n, 144, DType::F32, bytemuck::cast_slice(&rope(h, w)))?;
    let mut outputs = Vec::new();
    for frame in 0..t {
        let pixels: Vec<_> = item.patches[frame * n * 1536..(frame + 1) * n * 1536]
            .iter()
            .map(|p| half::bf16::from_f32(*p))
            .collect();
        let pixels = e.input(n, 1536, DType::Bf16, bytemuck::cast_slice(&pixels))?;
        let mut x = e.project(&pixels, "model.visual.patch_embed.proj", true)?;
        x = e.binary(&x, &pos, "addf")?;
        for i in 0..27 {
            let p = format!("model.visual.blocks.{i}");
            let norm = e.named_norm(&x, &format!("{p}.norm1"), true, 1e-6)?;
            let qkv = e.project(&norm, &format!("{p}.attn.qkv"), true)?;
            let q = e.columns(&qkv, 0, 1152)?;
            let k = e.columns(&qkv, 1152, 1152)?;
            let v = e.columns(&qkv, 2304, 1152)?;
            let q = e.rope(&q, 16, 72, &cs)?;
            let k = e.rope(&k, 16, 72, &cs)?;
            let a = e.attention(&q, &k, &v, 16, 16, false)?;
            let a = e.project(&a, &format!("{p}.attn.proj"), true)?;
            x = e.binary(&x, &a, "addf")?;
            let f = e.named_norm(&x, &format!("{p}.norm2"), true, 1e-6)?;
            let f = e.project(&f, &format!("{p}.mlp.linear_fc1"), true)?;
            let f = e.unary(&f, "gelu_tanh")?;
            let f = e.project(&f, &format!("{p}.mlp.linear_fc2"), true)?;
            x = e.binary(&x, &f, "addf")?;
        }
        let x = e
            .named_norm(&x, "model.visual.merger.norm", true, 1e-6)?
            .reshape(n / 4, 4608);
        let x = e.project(&x, "model.visual.merger.linear_fc1", true)?;
        let x = e.unary(&x, "gelu")?;
        outputs.push(e.project(&x, "model.visual.merger.linear_fc2", true)?);
    }
    e.cat_rows(&outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires gfx1151; uses a small synthetic position table"]
    fn interpolated_position_embedding() -> Result<()> {
        let context = hrx::inference::ModelContext::new(Default::default())?;
        let mut e = Engine::new(&context)?;
        let table: Vec<_> = (0..48 * 48 * 1152)
            .map(|i| {
                let row = i / 1152;
                half::bf16::from_f32((row / 48) as f32 / 64. + (row % 48) as f32 / 1024.)
            })
            .collect();
        e.weight(
            "model.visual.pos_embed.weight".into(),
            48 * 48,
            1152,
            bytemuck::cast_slice(&table),
        )?;
        for (h, w) in [(2, 4), (4, 2), (6, 8)] {
            let out = position_embedding(&mut e, h, w)?;
            let mut graph = e.finish()?;
            e.run(&mut graph)?;
            let actual = e.read_f32(&out)?;
            for y in 0..h {
                for x in 0..w {
                    // Independent spatial-merge ordering and interpolation of
                    // an affine table, including both rectangular orientations.
                    let row = ((y / 2) * (w / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2;
                    let expected = y as f32 * 47. / (h - 1) as f32 / 64.
                        + x as f32 * 47. / (w - 1) as f32 / 1024.;
                    for value in &actual[row * 1152..(row + 1) * 1152] {
                        assert!(
                            (value - expected).abs() < 0.004,
                            "({h},{w}) at ({y},{x}): {value} vs {expected}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
