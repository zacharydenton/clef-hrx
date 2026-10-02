//! Qwen3-VL media preprocessing and multimodal position bookkeeping.
use crate::{Result, schema::Request};
use anyhow::{Context, ensure};
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::{
    io::Read,
    path::Path,
    process::{Command, Stdio},
};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MediaOptions {
    pub min_pixels: Option<usize>,
    pub max_pixels: Option<usize>,
    pub fps: Option<f64>,
    pub num_frames: Option<usize>,
    pub do_sample_frames: Option<bool>,
}
#[derive(Debug, Clone)]
pub struct VideoFrames {
    pub frames: Vec<RgbImage>,
    pub fps: f64,
}
#[derive(Debug, Clone)]
pub struct VisualItem {
    pub grid: [usize; 3],
    pub patches: Vec<f32>,
    pub video: bool,
    pub timestamps: Vec<f64>,
    pub token_ranges: Vec<[usize; 2]>,
}
#[derive(Debug, Clone, Default)]
pub struct PreparedMedia {
    pub items: Vec<VisualItem>,
    pub prompt: String,
}
impl PreparedMedia {
    pub fn bind_tokens(&mut self, ids: &[u32], offset: usize) -> Result<()> {
        let mut cursor = 0;
        for item in &mut self.items {
            let expected = if item.video { 248057 } else { 248056 };
            let per_frame = item.grid[1] * item.grid[2] / 4;
            item.token_ranges.clear();
            for _ in 0..item.grid[0] {
                while cursor < ids.len() && ids[cursor] != expected {
                    cursor += 1;
                }
                ensure!(
                    cursor + per_frame <= ids.len()
                        && ids[cursor..cursor + per_frame]
                            .iter()
                            .all(|v| *v == expected),
                    "visual token count mismatch"
                );
                item.token_ranges
                    .push([offset + cursor, offset + cursor + per_frame]);
                cursor += per_frame;
            }
        }
        Ok(())
    }
    pub fn position_ids(&self, len: usize) -> Result<Vec<[u32; 3]>> {
        let mut result = Vec::with_capacity(len);
        let mut pos = 0u32;
        for item in &self.items {
            let h = item.grid[1] / 2;
            let w = item.grid[2] / 2;
            for range in &item.token_ranges {
                ensure!(
                    range[0] >= result.len() && range[1] <= len,
                    "invalid visual token range"
                );
                while result.len() < range[0] {
                    result.push([pos; 3]);
                    pos += 1;
                }
                for y in 0..h {
                    for x in 0..w {
                        result.push([pos, pos + y as u32, pos + x as u32]);
                    }
                }
                pos += h.max(w) as u32;
                ensure!(result.len() == range[1], "visual grid/token mismatch");
            }
        }
        while result.len() < len {
            result.push([pos; 3]);
            pos += 1;
        }
        Ok(result)
    }
}
fn limits(options: &MediaOptions, video: bool) -> Result<(usize, usize)> {
    let min = options
        .min_pixels
        .unwrap_or(if video { 4096 } else { 65536 });
    let max = options
        .max_pixels
        .unwrap_or(if video { 25165824 } else { 16777216 });
    ensure!(
        min >= 1024 && min <= max && max <= 25165824,
        "invalid media pixel limits"
    );
    Ok((min, max))
}
pub fn smart_resize(
    h: usize,
    w: usize,
    frames: usize,
    min: usize,
    max: usize,
    video: bool,
) -> Result<(usize, usize)> {
    ensure!(h > 0 && w > 0 && frames > 0, "empty media");
    ensure!(
        min >= 1024 && min <= max && max <= 25165824,
        "invalid media pixel limits"
    );
    let pixels = frames
        .checked_mul(h)
        .and_then(|n| n.checked_mul(w))
        .context("media dimensions overflow")?;
    ensure!(
        h.max(w) as f64 / h.min(w) as f64 <= 200.,
        "media aspect ratio exceeds 200"
    );
    if video {
        ensure!(h >= 32 && w >= 32, "video dimensions must be at least 32");
    }
    let mut hh = (h as f64 / 32.).round_ties_even() as usize * 32;
    let mut ww = (w as f64 / 32.).round_ties_even() as usize * 32;
    let t = if video {
        frames
            .checked_add(frames % 2)
            .context("frame count overflow")?
    } else {
        1
    };
    let rounded_pixels = t.saturating_mul(hh).saturating_mul(ww);
    if rounded_pixels > max {
        let beta = (frames as f64 * h as f64 * w as f64 / max as f64).sqrt();
        hh = ((h as f64 / beta / 32.).floor() as usize * 32).max(32);
        ww = ((w as f64 / beta / 32.).floor() as usize * 32).max(32);
    } else if rounded_pixels < min {
        let beta = (min as f64 / pixels as f64).sqrt();
        hh = (h as f64 * beta / 32.).ceil() as usize * 32;
        ww = (w as f64 * beta / 32.).ceil() as usize * 32;
    }
    Ok((hh.max(32), ww.max(32)))
}
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1. {
        ((1.5 * x - 2.5) * x) * x + 1.
    } else if x < 2. {
        ((-0.5 * x + 2.5) * x - 4.) * x + 2.
    } else {
        0.
    }
}
fn taps(src: usize, dst: usize) -> Vec<Vec<(usize, f64)>> {
    let scale = src as f64 / dst as f64;
    let support = scale.max(1.);
    (0..dst)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let start = ((center - support * 2. + 0.5) as isize).max(0) as usize;
            let end = ((center + support * 2. + 0.5) as usize).min(src);
            let mut t: Vec<_> = (start..end)
                .map(|j| (j, cubic((j as f64 - center + 0.5) / support)))
                .collect();
            let sum: f64 = t.iter().map(|(_, w)| w).sum();
            for (_, w) in &mut t {
                *w /= sum;
            }
            t
        })
        .collect()
}
// Torch's uint8 antialias path quantizes coefficients to signed 16 bits.
// A floating-point convolution differs by up to two byte values after both passes.
fn quantized_taps(src: usize, dst: usize) -> (Vec<Vec<(usize, i32)>>, u32) {
    let taps = taps(src, dst);
    let max = taps.iter().flatten().map(|(_, w)| *w).fold(0., f64::max);
    let precision = (0..22)
        .find(|p| (0.5 + max * ((1u32 << (p + 1)) as f64)) as i32 >= (1 << 15))
        .unwrap_or(22);
    let scale = (1u32 << precision) as f64;
    (
        taps.into_iter()
            .map(|t| {
                t.into_iter()
                    .map(|(i, w)| (i, (w * scale).round() as i32))
                    .collect()
            })
            .collect(),
        precision,
    )
}
/// Antialiased bicubic RGB8 resize, with byte rounding between separable passes.
fn resize(image: &RgbImage, h: usize, w: usize) -> RgbImage {
    let (sw, sh) = (image.width() as usize, image.height() as usize);
    if (sh, sw) == (h, w) {
        return image.clone();
    }
    let (xt, xprecision) = quantized_taps(sw, w);
    let (yt, yprecision) = quantized_taps(sh, h);
    let mut mid = vec![0u8; sh * w * 3];
    for y in 0..sh {
        let source = &image.as_raw()[y * sw * 3..(y + 1) * sw * 3];
        for (x, t) in xt.iter().enumerate() {
            let mut sums = [0i32; 3];
            for &(xx, a) in t {
                for c in 0..3 {
                    sums[c] += source[xx * 3 + c] as i32 * a;
                }
            }
            for c in 0..3 {
                mid[(y * w + x) * 3 + c] =
                    ((sums[c] + (1 << (xprecision - 1))) >> xprecision).clamp(0, 255) as u8;
            }
        }
    }
    let mut out = vec![0u8; h * w * 3];
    // Accumulate contiguous rows so the inner loop can vectorize. Keep the
    // coefficient order and integer rounding identical to the Torch path.
    let mut sums = vec![0i32; w * 3];
    for (y, t) in yt.iter().enumerate() {
        sums.fill(0);
        for &(yy, a) in t {
            let source = &mid[yy * w * 3..(yy + 1) * w * 3];
            for (sum, &pixel) in sums.iter_mut().zip(source) {
                *sum += pixel as i32 * a;
            }
        }
        for (pixel, &sum) in out[y * w * 3..(y + 1) * w * 3].iter_mut().zip(&sums) {
            *pixel = ((sum + (1 << (yprecision - 1))) >> yprecision).clamp(0, 255) as u8;
        }
    }
    RgbImage::from_raw(w as u32, h as u32, out).unwrap()
}
pub fn sample_indices(total: usize, fps: f64, options: &MediaOptions) -> Result<Vec<usize>> {
    ensure!(
        total > 0 && fps.is_finite() && fps > 0.,
        "invalid video metadata"
    );
    ensure!(
        options.num_frames.is_none() || options.fps.is_none(),
        "num_frames and fps are mutually exclusive"
    );
    if options.do_sample_frames == Some(false) {
        return Ok((0..total).collect());
    }
    let rate = options.fps.unwrap_or(2.);
    ensure!(rate.is_finite() && rate > 0., "invalid sampling fps");
    let n = options.num_frames.unwrap_or(
        ((total as f64 / fps * rate) as usize)
            .clamp(4, 768)
            .min(total),
    );
    ensure!((1..=768).contains(&n), "sample frame count must be 1..=768");
    Ok((0..n)
        .map(|i| {
            if n == 1 {
                0
            } else {
                (i as f64 * (total - 1) as f64 / (n - 1) as f64).round_ties_even() as usize
            }
        })
        .collect())
}
fn item(
    frames: &[RgbImage],
    indices: &[usize],
    fps: f64,
    video: bool,
    options: &MediaOptions,
    budget: usize,
) -> Result<VisualItem> {
    ensure!(!frames.is_empty(), "empty frame sequence");
    let w = frames[0].width() as usize;
    let h = frames[0].height() as usize;
    ensure!(
        frames
            .iter()
            .all(|f| f.width() as usize == w && f.height() as usize == h),
        "video frame dimensions differ"
    );
    let (min, max) = limits(options, video)?;
    let (rh, rw) = smart_resize(h, w, frames.len(), min, max, video)?;
    let t = if video { frames.len().div_ceil(2) } else { 1 };
    let gh = rh / 16;
    let gw = rw / 16;
    ensure!(
        t * gh * gw / 4 <= budget,
        "media alone exceeds token budget; lower max_pixels/num_frames"
    );
    let resized: Vec<_> = frames.iter().map(|im| resize(im, rh, rw)).collect();
    let mut patches = Vec::with_capacity(t * gh * gw * 1536);
    for ti in 0..t {
        for by in 0..gh / 2 {
            for bx in 0..gw / 2 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        for c in 0..3 {
                            for dt in 0..2 {
                                for py in 0..16 {
                                    for px in 0..16 {
                                        let f = if video {
                                            (ti * 2 + dt).min(resized.len() - 1)
                                        } else {
                                            0
                                        };
                                        let y = (by * 2 + dy) * 16 + py;
                                        let x = (bx * 2 + dx) * 16 + px;
                                        let pixel = resized[f].as_raw()[(y * rw + x) * 3 + c];
                                        patches.push((pixel as f32 * (1.0 / 255.0) - 0.5) / 0.5);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let timestamps = if video {
        (0..t)
            .map(|i| {
                (indices[i * 2] as f64 + indices[(i * 2 + 1).min(indices.len() - 1)] as f64)
                    / (2. * fps)
            })
            .collect()
    } else {
        vec![]
    };
    Ok(VisualItem {
        grid: [t, gh, gw],
        patches,
        video,
        timestamps,
        token_ranges: vec![],
    })
}
fn append(media: &mut PreparedMedia, item: VisualItem) {
    let n = item.grid[1] * item.grid[2] / 4;
    if item.video {
        media.prompt.push_str("<|vision_start|>");
        for time in &item.timestamps {
            media.prompt.push_str(&format!(
                "<{time:.1} seconds><|vision_start|>{}<|vision_end|>",
                "<|video_pad|>".repeat(n)
            ));
        }
        media.prompt.push_str("<|vision_end|>");
    } else {
        media.prompt.push_str(&format!(
            "<|vision_start|>{}<|vision_end|>",
            "<|image_pad|>".repeat(n)
        ));
    }
    media.items.push(item);
}
pub fn prepare(
    images: &[RgbImage],
    videos: &[VideoFrames],
    options: &MediaOptions,
    budget: usize,
) -> Result<PreparedMedia> {
    let mut out = PreparedMedia::default();
    let mut remaining = budget;
    for im in images {
        let v = item(
            std::slice::from_ref(im),
            &[0],
            1.,
            false,
            options,
            remaining,
        )?;
        remaining -= v.grid.iter().product::<usize>() / 4;
        append(&mut out, v);
    }
    for video in videos {
        let indices = sample_indices(video.frames.len(), video.fps, options)?;
        let frames: Vec<_> = indices.iter().map(|i| video.frames[*i].clone()).collect();
        let v = item(&frames, &indices, video.fps, true, options, remaining)?;
        remaining -= v.grid.iter().product::<usize>() / 4;
        append(&mut out, v);
    }
    if !out.items.is_empty() {
        out.prompt.push('\n');
    }
    Ok(out)
}
pub fn prepare_paths(request: &Request, budget: usize) -> Result<PreparedMedia> {
    let mut out = PreparedMedia::default();
    let mut remaining = budget;
    for path in &request.images {
        let image = image::open(path)
            .with_context(|| format!("image {path}"))?
            .to_rgb8();
        let v = item(&[image], &[0], 1., false, &request.media_kwargs, remaining)?;
        remaining -= v.grid.iter().product::<usize>() / 4;
        append(&mut out, v);
    }
    for path in &request.videos {
        let (frames, indices, fps) = decode_video(Path::new(path), &request.media_kwargs)?;
        let v = item(
            &frames,
            &indices,
            fps,
            true,
            &request.media_kwargs,
            remaining,
        )?;
        remaining -= v.grid.iter().product::<usize>() / 4;
        append(&mut out, v);
    }
    if !out.items.is_empty() {
        out.prompt.push('\n');
    }
    Ok(out)
}
fn decode_video(path: &Path, options: &MediaOptions) -> Result<(Vec<RgbImage>, Vec<usize>, f64)> {
    ensure!(
        path.is_file(),
        "video must be a local file: {}",
        path.display()
    );
    // Absolute paths also prevent leading '-' filenames from becoming options.
    let path = path.canonicalize()?;
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames",
            "-of",
            "json",
        ])
        .arg(&path)
        .stdin(Stdio::null())
        .output()
        .context("ffprobe is required for video files")?;
    ensure!(
        probe.status.success(),
        "ffprobe failed: {}",
        String::from_utf8_lossy(&probe.stderr)
    );
    let meta: serde_json::Value = serde_json::from_slice(&probe.stdout)?;
    let s = &meta["streams"][0];
    let w = s["width"].as_u64().context("video width")? as usize;
    let h = s["height"].as_u64().context("video height")? as usize;
    ensure!(
        w > 0 && h > 0 && w <= 8192 && h <= 8192,
        "unsupported video dimensions"
    );
    // r_frame_rate can reflect the container timebase for short clips.
    let fps = ["avg_frame_rate", "r_frame_rate"]
        .into_iter()
        .find_map(|key| {
            let (a, b) = s[key].as_str()?.split_once('/')?;
            let rate = a.parse::<f64>().ok()? / b.parse::<f64>().ok()?;
            (rate.is_finite() && rate > 0.).then_some(rate)
        })
        .context("video fps")?;
    let total = s["nb_read_frames"]
        .as_str()
        .context("video frame count")?
        .parse()?;
    let indices = sample_indices(total, fps, options)?;
    let unique: std::collections::BTreeSet<_> = indices.iter().copied().collect();
    let decoded_bytes = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_mul(unique.len() + indices.len()))
        .context("decoded video size overflow")?;
    ensure!(
        decoded_bytes <= 1 << 30,
        "decoded video exceeds 1 GiB; reduce num_frames or source resolution"
    );
    crate::memory::before_allocation(decoded_bytes)?;
    let filter = format!(
        "select={}",
        unique
            .iter()
            .map(|i| format!("eq(n\\,{i})"))
            .collect::<Vec<_>>()
            .join("+")
    );
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-noautorotate", "-i"])
        .arg(&path)
        .args([
            "-map",
            "0:v:0",
            "-vf",
            &filter,
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("ffmpeg is required for video files")?;
    let read = (|| -> Result<_> {
        let mut stdout = child.stdout.take().unwrap();
        let mut frames = std::collections::BTreeMap::new();
        for i in unique {
            let mut bytes = vec![0; w * h * 3];
            stdout
                .read_exact(&mut bytes)
                .context("reading video frame")?;
            frames.insert(i, RgbImage::from_raw(w as u32, h as u32, bytes).unwrap());
        }
        Ok(indices
            .iter()
            .map(|i| frames[i].clone())
            .collect::<Vec<_>>())
    })();
    if read.is_err() {
        let _ = child.kill();
    }
    let status = child.wait()?;
    let frames = read?;
    ensure!(status.success(), "ffmpeg decode failed");
    Ok((frames, indices, fps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires ffmpeg and ffprobe"]
    fn decode_uses_the_probed_video_stream() -> Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("streams.mkv");
        let output = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "color=red:s=32x32:r=2:d=1",
                "-f",
                "lavfi",
                "-i",
                "color=blue:s=64x64:r=2:d=1",
                "-map",
                "0:v",
                "-map",
                "1:v",
                "-c:v",
                "ffv1",
            ])
            .arg(&path)
            .output()?;
        ensure!(
            output.status.success(),
            "fixture: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        // FFmpeg's automatic selection prefers the larger second stream.
        let (frames, indices, fps) = decode_video(&path, &MediaOptions::default())?;
        assert_eq!(indices, [0, 1]);
        assert_eq!(fps, 2.);
        assert_eq!(frames.len(), 2);
        for frame in frames {
            assert_eq!(frame.dimensions(), (32, 32));
            assert!(frame.pixels().all(|p| p[0] > 240 && p[1] < 10 && p[2] < 10));
        }
        Ok(())
    }
}
