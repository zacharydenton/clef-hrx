//! Omni media ordering, audio/video synchronization and native preprocessing.
use crate::{
    Result,
    audio::{AudioItem, mel_features},
    media::{MediaOptions, PreparedMedia, VisualItem},
    schema::Request,
};
use anyhow::{Context, ensure};
use image::RgbImage;
use std::{
    io::Read,
    path::Path,
    process::{Command, Stdio},
};

#[derive(Debug, Clone)]
enum Block {
    Image(usize),
    Audio(usize),
    Video(usize, Option<usize>),
}
#[derive(Debug, Clone, Default)]
pub struct OmniMedia {
    pub audio: Vec<AudioItem>,
    blocks: Vec<Block>,
    positions: Vec<[u32; 3]>,
    offset: usize,
    next_position: u32,
}
impl OmniMedia {
    pub fn position_ids(&self, len: usize) -> Result<Vec<[u32; 3]>> {
        ensure!(
            self.offset + self.positions.len() <= len,
            "Omni media position extent"
        );
        let mut out: Vec<_> = (0..self.offset as u32).map(|i| [i; 3]).collect();
        out.extend_from_slice(&self.positions);
        let mut pos = self.next_position;
        while out.len() < len {
            out.push([pos; 3]);
            pos += 1;
        }
        Ok(out)
    }
}
fn vision_positions(item: &VisualItem) -> Vec<[u32; 3]> {
    let [t, h, w] = item.grid;
    (0..t)
        .flat_map(|time| {
            (0..h / 2).flat_map(move |y| {
                (0..w / 2).map(move |x| [(time * 13) as u32, y as u32, x as u32])
            })
        })
        .collect()
}
fn token_sequence(item: &VisualItem, audio: Option<&AudioItem>) -> Vec<(u32, [u32; 3])> {
    let visual = vision_positions(item);
    let token = if item.video { 151656 } else { 151655 };
    let mut out = Vec::with_capacity(visual.len() + audio.map_or(0, AudioItem::tokens));
    let (mut v, mut a) = (0, 0);
    let audio_len = audio.map_or(0, AudioItem::tokens);
    while v < visual.len() || a < audio_len {
        if v < visual.len() && (a == audio_len || visual[v][0] <= a as u32) {
            out.push((token, visual[v]));
            v += 1;
        } else {
            out.push((151675, [a as u32; 3]));
            a += 1;
        }
    }
    out
}
fn push_range(ranges: &mut Vec<[usize; 2]>, at: usize) {
    if let Some(last) = ranges.last_mut()
        && last[1] == at
    {
        last[1] += 1;
        return;
    }
    ranges.push([at, at + 1]);
}
/// Bind only the processor-generated prefix, never tokens in user state/schema.
pub(crate) fn bind(media: &mut PreparedMedia, ids: &[u32], offset: usize) -> Result<()> {
    let omni = media.omni.as_mut().context("missing Omni media")?;
    omni.positions.clear();
    omni.offset = offset;
    for item in &mut media.items {
        item.token_ranges.clear();
    }
    for item in &mut omni.audio {
        item.token_ranges.clear();
    }
    let mut cursor = 0;
    let mut pos = offset as u32;
    for block in &omni.blocks {
        let (bos, eos, sequence, visual, audio) = match *block {
            Block::Image(i) => (
                vec![151652],
                vec![151653],
                token_sequence(&media.items[i], None),
                Some(i),
                None,
            ),
            Block::Audio(a) => (
                vec![151669],
                vec![151670],
                (0..omni.audio[a].tokens())
                    .map(|t| (151675, [t as u32; 3]))
                    .collect(),
                None,
                Some(a),
            ),
            Block::Video(i, a) => (
                if a.is_some() {
                    vec![151652, 151669]
                } else {
                    vec![151652]
                },
                if a.is_some() {
                    vec![151670, 151653]
                } else {
                    vec![151653]
                },
                token_sequence(&media.items[i], a.map(|a| &omni.audio[a])),
                Some(i),
                a,
            ),
        };
        while cursor < ids.len() && ids[cursor] != bos[0] {
            omni.positions.push([pos; 3]);
            pos += 1;
            cursor += 1;
        }
        for token in bos {
            ensure!(
                ids.get(cursor) == Some(&token),
                "Omni media beginning mismatch"
            );
            omni.positions.push([pos; 3]);
            cursor += 1;
            pos += 1;
        }
        let base = pos;
        for &(token, coords) in &sequence {
            ensure!(
                ids.get(cursor) == Some(&token),
                "Omni media token mismatch at {cursor}"
            );
            omni.positions.push(coords.map(|c| c + base));
            if token == 151675 {
                push_range(
                    &mut omni.audio[audio.context("audio token without clip")?].token_ranges,
                    cursor + offset,
                );
            } else {
                push_range(
                    &mut media.items[visual.context("visual token without item")?].token_ranges,
                    cursor + offset,
                );
            }
            cursor += 1;
        }
        // Upstream advances after its last appended position segment. In a
        // synchronized video that is the remaining tail after one stream ends.
        let tail = if audio.is_some() && visual.is_some() {
            let last_token = sequence.last().context("empty media")?.0;
            sequence
                .iter()
                .rposition(|p| p.0 != last_token)
                .map_or(0, |i| i + 1)
        } else {
            0
        };
        pos = base
            + sequence[tail..]
                .iter()
                .flat_map(|p| p.1)
                .max()
                .context("empty media")?
            + 1;
        for token in eos {
            ensure!(
                ids.get(cursor) == Some(&token),
                "Omni media ending mismatch"
            );
            omni.positions.push([pos; 3]);
            pos += 1;
            cursor += 1;
        }
    }
    while cursor < ids.len() {
        omni.positions.push([pos; 3]);
        pos += 1;
        cursor += 1;
    }
    omni.next_position = pos;
    Ok(())
}
fn append_prompt(media: &mut PreparedMedia) {
    let omni = media.omni.as_ref().unwrap();
    for block in &omni.blocks {
        match *block {
            Block::Image(i) => {
                media.prompt.push_str("<|vision_start|>");
                media.prompt.push_str(
                    &"<|image_pad|>".repeat(media.items[i].grid.iter().product::<usize>() / 4),
                );
                media.prompt.push_str("<|vision_end|>\n");
            }
            Block::Audio(a) => {
                media.prompt.push_str("<|audio_start|>");
                media
                    .prompt
                    .push_str(&"<|audio_pad|>".repeat(omni.audio[a].tokens()));
                media.prompt.push_str("<|audio_end|>\n");
            }
            Block::Video(i, a) => {
                media.prompt.push_str("<|vision_start|>");
                if a.is_some() {
                    media.prompt.push_str("<|audio_start|>");
                }
                for (token, _) in token_sequence(&media.items[i], a.map(|a| &omni.audio[a])) {
                    media.prompt.push_str(if token == 151675 {
                        "<|audio_pad|>"
                    } else {
                        "<|video_pad|>"
                    });
                }
                if a.is_some() {
                    media.prompt.push_str("<|audio_end|>");
                }
                media.prompt.push_str("<|vision_end|>\n");
            }
        }
    }
}

pub(crate) fn prepare(request: &Request, budget: usize) -> Result<PreparedMedia> {
    let options = &request.media_kwargs;
    ensure!(
        options.fps.is_none_or(|f| f == 2.)
            && options.num_frames.is_none()
            && options.do_sample_frames != Some(true),
        "Omni uses the release's 2 fps sampling; num_frames and resampling overrides are unsupported"
    );
    let mut media = PreparedMedia {
        omni: Some(OmniMedia::default()),
        ..Default::default()
    };
    let mut remaining = budget;
    let image_options = MediaOptions {
        min_pixels: Some(options.min_pixels.unwrap_or(65536)),
        max_pixels: Some(options.max_pixels.unwrap_or(1048576)),
        ..Default::default()
    };
    for path in &request.images {
        let image = image::open(path)
            .with_context(|| format!("image {path}"))?
            .to_rgb8();
        let item = crate::media::omni_item(&[image], false, &image_options, remaining)?;
        remaining -= item.grid.iter().product::<usize>() / 4;
        media
            .omni
            .as_mut()
            .unwrap()
            .blocks
            .push(Block::Image(media.items.len()));
        media.items.push(item);
    }
    let mut clips = Vec::new();
    for path in &request.audio {
        let samples = crate::audio::decode(Path::new(path), remaining)?
            .context("audio has no decodable stream")?;
        media
            .omni
            .as_mut()
            .unwrap()
            .blocks
            .push(Block::Audio(clips.len()));
        clips.push(samples);
    }
    let mut videos = Vec::new();
    let mut tracks = Vec::new();
    let video_options = MediaOptions {
        min_pixels: Some(options.min_pixels.unwrap_or(128 * 32 * 32)),
        max_pixels: Some(options.max_pixels.unwrap_or(768 * 32 * 32)),
        ..Default::default()
    };
    for path in &request.videos {
        let frames = decode_video(Path::new(path), remaining)?;
        let item = crate::media::omni_item(&frames, true, &video_options, remaining)?;
        remaining -= item.grid.iter().product::<usize>() / 4;
        videos.push(item);
        tracks.push(crate::audio::decode(Path::new(path), budget)?);
    }
    let synchronized = !videos.is_empty() && tracks.iter().all(Option::is_some);
    for (item, track) in videos.into_iter().zip(tracks) {
        let audio = if synchronized {
            let a = clips.len();
            clips.push(track.unwrap());
            Some(a)
        } else {
            None
        };
        media
            .omni
            .as_mut()
            .unwrap()
            .blocks
            .push(Block::Video(media.items.len(), audio));
        media.items.push(item);
    }
    let audio = mel_features(&clips)?;
    ensure!(
        audio.iter().map(AudioItem::tokens).sum::<usize>() <= remaining,
        "context capacity exceeded: audio and video exceed token budget"
    );
    media.omni.as_mut().unwrap().audio = audio;
    append_prompt(&mut media);
    Ok(media)
}

fn decode_video(path: &Path, budget: usize) -> Result<Vec<RgbImage>> {
    let path = path
        .canonicalize()
        .with_context(|| format!("video {}", path.display()))?;
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
        ])
        .arg(&path)
        .stdin(Stdio::null())
        .output()
        .context("ffprobe is required for video")?;
    ensure!(probe.status.success(), "video probe failed");
    let meta: serde_json::Value = serde_json::from_slice(&probe.stdout)?;
    let w = meta["streams"][0]["width"]
        .as_u64()
        .context("video width")? as usize;
    let h = meta["streams"][0]["height"]
        .as_u64()
        .context("video height")? as usize;
    ensure!(
        w > 0 && h > 0 && w <= 8192 && h <= 8192,
        "unsupported video dimensions"
    );
    let scale = (262144.0 / (w * h) as f64).sqrt().min(1.);
    let w = ((w as f64 * scale / 2.).round_ties_even() as usize * 2).max(2);
    let h = ((h as f64 * scale / 2.).round_ties_even() as usize * 2).max(2);
    let frame_bytes = w * h * 3;
    let limit = (budget / 128 + 1)
        .saturating_mul(2)
        .saturating_mul(frame_bytes)
        .min(1 << 30);
    crate::memory::before_allocation(limit)?;
    let filter = format!("select=gte(t+0.000001\\,selected_n/2),scale={w}:{h}:flags=bilinear");
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-noautorotate", "-copyts", "-i"])
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
        .context("ffmpeg is required for video")?;
    let mut bytes = Vec::new();
    let read = child
        .stdout
        .take()
        .context("video stdout")?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes);
    if read.is_err() || bytes.len() > limit {
        let _ = child.kill();
    }
    let status = child.wait()?;
    read?;
    ensure!(
        bytes.len() <= limit,
        "decoded video exceeds memory/context budget"
    );
    ensure!(
        status.success() && !bytes.is_empty() && bytes.len().is_multiple_of(frame_bytes),
        "video decode failed"
    );
    let mut frames: Vec<_> = bytes
        .chunks_exact(frame_bytes)
        .map(|b| RgbImage::from_raw(w as u32, h as u32, b.to_vec()).unwrap())
        .collect();
    if !frames.len().is_multiple_of(2) {
        frames.push(frames.last().unwrap().clone());
    }
    Ok(frames)
}

#[cfg(test)]
mod oracle_tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned Omni tokenizer and ffmpeg; CPU only"]
    fn encoding_matches_cloudflare_release() -> Result<()> {
        let source = crate::checkpoint::Source {
            model: crate::ModelVariant::ClefOmni,
            offline: true,
            directory: std::env::var_os("CLEF_OMNI_MODEL_DIR").map(Into::into),
        };
        let encoder =
            crate::Encoder::load_for_model(source.resolve("tokenizer.json")?, source.model)?;
        for text in [
            include_str!("../tests/fixtures/omni/encoding-0.json"),
            include_str!("../tests/fixtures/omni/encoding-1.json"),
            include_str!("../tests/fixtures/omni/encoding-2.json"),
            include_str!("../tests/fixtures/omni/encoding-3.json"),
            include_str!("../tests/fixtures/omni/encoding-4.json"),
        ] {
            let fixture: serde_json::Value = serde_json::from_str(text)?;
            let request: Request = serde_json::from_value(fixture["request"].clone())?;
            let record = encoder.encode_record(&request, Default::default())?;
            assert_eq!(
                serde_json::to_value(&record.input_ids)?,
                fixture["input_ids"]
            );
            // Upstream returns integral floating-point positions for the fixed 2 fps release contract.
            let positions: Vec<[u32; 3]> = fixture["position_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| std::array::from_fn(|i| v[i].as_f64().unwrap() as u32))
                .collect();
            assert_eq!(record.position_ids, positions);
            for (actual, expected) in record
                .questions
                .iter()
                .zip(fixture["questions"].as_array().unwrap())
            {
                assert_eq!(
                    serde_json::to_value(actual.question_span)?,
                    expected["question_span"]
                );
                assert_eq!(
                    serde_json::to_value(&actual.option_spans)?,
                    expected["option_spans"]
                );
            }
        }
        Ok(())
    }
}
