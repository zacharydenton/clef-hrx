//! Qwen3-Omni audio preprocessing and encoder. Samples stay at 16 kHz mono.
use crate::{
    Result,
    gpu::{DType, Engine, Tensor},
};
use anyhow::{Context, ensure};
use rustfft::{FftPlanner, num_complex::Complex};
use std::{
    io::Read,
    path::Path,
    process::{Command, Stdio},
};

#[derive(Debug, Clone)]
pub struct AudioItem {
    /// Valid mel frames; storage is frequency-major [128, frames].
    pub frames: usize,
    pub mel: Vec<f32>,
    pub token_ranges: Vec<[usize; 2]>,
}
impl AudioItem {
    pub fn tokens(&self) -> usize {
        output_length(self.frames)
    }
}
pub fn output_length(frames: usize) -> usize {
    frames / 100 * 13 + (frames % 100).div_ceil(8)
}

/// Decode the first audio stream. A video without audio returns None.
pub(crate) fn decode(path: &Path, budget: usize) -> Result<Option<Vec<f32>>> {
    let path = path
        .canonicalize()
        .with_context(|| format!("audio {}", path.display()))?;
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
        ])
        .arg(&path)
        .stdin(Stdio::null())
        .output()
        .context("ffprobe is required for audio")?;
    ensure!(
        probe.status.success(),
        "audio probe failed: {}",
        String::from_utf8_lossy(&probe.stderr)
    );
    let meta: serde_json::Value = serde_json::from_slice(&probe.stdout)?;
    if meta["streams"].as_array().is_none_or(|s| s.is_empty()) {
        return Ok(None);
    }
    let limit = budget
        .checked_mul(8 * 160 * 4)
        .context("audio budget overflow")?;
    crate::memory::before_allocation(limit)?;
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(&path)
        .args([
            "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("ffmpeg is required for audio")?;
    let mut bytes = Vec::new();
    let read = child
        .stdout
        .take()
        .context("audio stdout")?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes);
    if read.is_err() || bytes.len() > limit {
        let _ = child.kill();
    }
    let status = child.wait()?;
    read?;
    ensure!(bytes.len() <= limit, "audio exceeds context capacity");
    ensure!(
        status.success() && !bytes.is_empty() && bytes.len().is_multiple_of(4),
        "audio decode failed"
    );
    let samples: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        samples.iter().all(|v| v.is_finite()),
        "non-finite audio samples"
    );
    Ok(Some(samples))
}

/// Whisper's Slaney-normalized, 128-band log-mel features. Padding is shared
/// across clips in a request, matching the upstream processor's longest batch.
pub fn mel_features(clips: &[Vec<f32>]) -> Result<Vec<AudioItem>> {
    if clips.is_empty() {
        return Ok(Vec::new());
    }
    let length = clips.iter().map(Vec::len).max().unwrap();
    ensure!(
        length > 200
            && clips
                .iter()
                .all(|s| !s.is_empty() && s.iter().all(|v| v.is_finite())),
        "audio is too short or non-finite"
    );
    let frames = length / 160;
    crate::memory::before_allocation(
        frames
            .checked_mul(128 * 4 * clips.len())
            .context("mel extent overflow")?,
    )?;
    let hz_to_mel = |hz: f64| {
        if hz < 1000. {
            hz * 3. / 200.
        } else {
            15. + (hz / 1000.).ln() * 27. / 6.4f64.ln()
        }
    };
    let mel_to_hz = |mel: f64| {
        if mel < 15. {
            mel * 200. / 3.
        } else {
            1000. * ((mel - 15.) * 6.4f64.ln() / 27.).exp()
        }
    };
    let edges: Vec<_> = (0..130)
        .map(|i| mel_to_hz(hz_to_mel(8000.) * i as f64 / 129.))
        .collect();
    let filters: Vec<Vec<f32>> = (0..128)
        .map(|band| {
            (0..201)
                .map(|bin| {
                    let hz = bin as f64 * 40.;
                    let triangle = ((hz - edges[band]) / (edges[band + 1] - edges[band]))
                        .min((edges[band + 2] - hz) / (edges[band + 2] - edges[band + 1]))
                        .max(0.);
                    (triangle * 2. / (edges[band + 2] - edges[band])) as f32
                })
                .collect()
        })
        .collect();
    let window: Vec<_> = (0..400)
        .map(|i| (1. - (2. * std::f64::consts::PI * i as f64 / 400.).cos()) as f32 * 0.5)
        .collect();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(400);
    clips
        .iter()
        .map(|samples| {
            let valid = samples.len().div_ceil(160).min(frames);
            let mut mel = vec![0.; 128 * frames];
            let mut work = vec![Complex::default(); 400];
            let mut peak = f32::NEG_INFINITY;
            for time in 0..frames {
                for j in 0..400 {
                    let at = time as isize * 160 + j as isize - 200;
                    let at = if at < 0 {
                        -at
                    } else if at >= length as isize {
                        2 * length as isize - 2 - at
                    } else {
                        at
                    } as usize;
                    work[j] = Complex::new(samples.get(at).copied().unwrap_or(0.) * window[j], 0.);
                }
                fft.process(&mut work);
                let power: Vec<_> = work[..201].iter().map(|v| v.norm_sqr()).collect();
                for band in 0..128 {
                    let energy: f32 = filters[band].iter().zip(&power).map(|(f, p)| f * p).sum();
                    let v = energy.max(1e-10).log10();
                    peak = peak.max(v);
                    mel[band * frames + time] = v;
                }
            }
            let values = (0..128)
                .flat_map(|band| {
                    mel[band * frames..band * frames + valid]
                        .iter()
                        .map(|&v| (v.max(peak - 8.) + 4.) / 4.)
                })
                .collect();
            Ok(AudioItem {
                frames: valid,
                mel: values,
                token_ranges: Vec::new(),
            })
        })
        .collect()
}

impl Engine {
    pub(crate) fn audio_conv(
        &mut self,
        x: &Tensor,
        h: usize,
        w: usize,
        name: &str,
    ) -> Result<(Tensor, usize, usize)> {
        ensure!(x.rows == h * w, "audio convolution geometry");
        let oh = h.div_ceil(2);
        let ow = w.div_ceil(2);
        let patches = self.alloc(oh * ow, x.cols * 9, DType::Bf16)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/omni_im2col.loom")
            ),
            patches.len(),
            &[("x", x)],
            &patches,
            &[
                ("height", h.to_string()),
                ("width", w.to_string()),
                ("channels", x.cols.to_string()),
                ("out_width", ow.to_string()),
            ],
        )?;
        let x = self.project(&patches, name, true)?;
        Ok((self.unary(&x, "gelu")?, oh, ow))
    }
}

pub(crate) fn forward(e: &mut Engine, audio: &AudioItem, chunk_width: usize) -> Result<Tensor> {
    ensure!(
        audio.frames > 0
            && audio.mel.len() == audio.frames * 128
            && (1..=100).contains(&chunk_width),
        "invalid audio features"
    );
    let mut chunks = Vec::new();
    for start in (0..audio.frames).step_by(100) {
        let length = (audio.frames - start).min(100);
        ensure!(length <= chunk_width, "audio chunk padding");
        let mel: Vec<_> = (0..128)
            .flat_map(|f| {
                (0..chunk_width).map(move |t| {
                    half::bf16::from_f32(if t < length {
                        audio.mel[f * audio.frames + start + t]
                    } else {
                        0.
                    })
                })
            })
            .collect();
        let mut x = e.input(
            128 * chunk_width,
            1,
            DType::Bf16,
            bytemuck::cast_slice(&mel),
        )?;
        let (mut h, mut w) = (128, chunk_width);
        for i in 1..=3 {
            (x, h, w) = e.audio_conv(&x, h, w, &format!("model.audio_tower.conv2d{i}"))?;
        }
        let reordered = e.alloc(w, h * x.cols, DType::Bf16)?;
        e.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/omni_audio_reorder.loom")
            ),
            reordered.len(),
            &[("x", &x)],
            &reordered,
            &[
                ("height", h.to_string()),
                ("width", w.to_string()),
                ("channels", x.cols.to_string()),
            ],
        )?;
        let x = e.project(&reordered, "model.audio_tower.conv_out", false)?;
        let pos: Vec<_> = (0..w)
            .flat_map(|time| {
                (0..1280).map(move |c| {
                    let phase = time as f64 * (-10000f64.ln() * (c % 640) as f64 / 639.).exp();
                    half::bf16::from_f64(if c < 640 { phase.sin() } else { phase.cos() })
                })
            })
            .collect();
        let pos = e.input(w, 1280, DType::Bf16, bytemuck::cast_slice(&pos))?;
        let x = e.binary(&x, &pos, "addf")?;
        chunks.push(x.slice_rows(0, output_length(length)));
    }
    let mut x = e.cat_rows(&chunks)?;
    let window = output_length(chunk_width) * 8;
    for i in 0..32 {
        let p = format!("model.audio_tower.layers.{i}");
        let norm = e.named_norm(&x, &format!("{p}.self_attn_layer_norm"), true, 1e-5)?;
        let q = e.project(&norm, &format!("{p}.self_attn.q_proj"), true)?;
        let k = e.project(&norm, &format!("{p}.self_attn.k_proj"), true)?;
        let v = e.project(&norm, &format!("{p}.self_attn.v_proj"), true)?;
        let mut attention = Vec::new();
        for start in (0..x.rows).step_by(window) {
            let n = (x.rows - start).min(window);
            attention.push(e.attention(
                &q.slice_rows(start, n),
                &k.slice_rows(start, n),
                &v.slice_rows(start, n),
                20,
                20,
                false,
            )?);
        }
        let a = e.cat_rows(&attention)?;
        let a = e.project(&a, &format!("{p}.self_attn.out_proj"), true)?;
        x = e.binary(&x, &a, "addf")?;
        let f = e.named_norm(&x, &format!("{p}.final_layer_norm"), true, 1e-5)?;
        let f = e.project(&f, &format!("{p}.fc1"), true)?;
        let f = e.unary(&f, "gelu")?;
        let f = e.project(&f, &format!("{p}.fc2"), true)?;
        x = e.binary(&x, &f, "addf")?;
    }
    let x = e.named_norm(&x, "model.audio_tower.ln_post", true, 1e-5)?;
    let x = e.project(&x, "model.audio_tower.proj1", true)?;
    let x = e.unary(&x, "gelu")?;
    e.project(&x, "model.audio_tower.proj2", true)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mel_matches_cloudflare_processor() -> Result<()> {
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/omni/mel.json"))?;
        let clips: Vec<Vec<f32>> = reference
            .as_array()
            .unwrap()
            .iter()
            .map(|case| {
                (0..case["samples"].as_u64().unwrap() as usize)
                    .map(|i| ((i * 17 % 101) as f32 - 50.) / 256.)
                    .collect()
            })
            .collect();
        for (actual, expected) in mel_features(&clips)?
            .iter()
            .zip(reference.as_array().unwrap())
        {
            assert_eq!(actual.frames, expected["frames"].as_u64().unwrap() as usize);
            let expected = expected["mel"].as_array().unwrap();
            assert_eq!(actual.mel.len(), expected.len());
            let error = actual
                .mel
                .iter()
                .zip(expected)
                .map(|(a, b)| (*a as f64 - b.as_f64().unwrap()).abs())
                .fold(0., f64::max);
            assert!(error < 0.0001, "log-mel maximum error: {error}");
        }
        Ok(())
    }
    #[test]
    fn cnn_lengths_cover_chunk_boundaries() {
        for (input, expected) in [
            (1, 1),
            (8, 1),
            (9, 2),
            (99, 13),
            (100, 13),
            (101, 14),
            (199, 26),
            (200, 26),
            (201, 27),
        ] {
            assert_eq!(output_length(input), expected);
        }
    }
}
