//! Pinned Hub resolution and strict, memory-mapped checkpoint validation.
use crate::{ModelVariant, Result};
use anyhow::{Context, ensure};
use hrx::artifacts::{
    hf::{HubFile, Repository, Resolver},
    safetensors::{DType, FileView, Tensor},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

pub const REVISION: &str = "2f3de3dd85f379784083b0814d997ab627200f0c";

/// Safetensors metadata marker for shards whose projection weights are tiled.
pub const TILED_LAYOUT_KEY: &str = "clef_weight_layout";
pub const TILED_LAYOUT: &str = "tiled256x64";
const TILE_ROWS: usize = 256;
const TILE_COLS: usize = 64;

/// Text-backbone projections stored in the tiled GEMM layout on disk.
///
/// A `[rows][cols]` BF16 matrix is stored as `[rows / 256][cols / 64][256][64]`,
/// so every 256-output by 64-reduction block that a GEMM K-step consumes is one
/// contiguous 32 KiB stream instead of 256 separate 128-byte row segments. The
/// logical shape in the safetensors header is unchanged. Other tensors, including
/// the vision encoder, decision head, norms, convolutions and the 48-wide
/// DeltaNet gates, keep their row-major layout.
pub fn tiled_projection(name: &str, rows: usize, cols: usize) -> bool {
    const PROJECTIONS: [&str; 10] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ];
    rows > 0
        && cols > 0
        && rows.is_multiple_of(TILE_ROWS)
        && cols.is_multiple_of(TILE_COLS)
        && name
            .strip_prefix("model.language_model.layers.")
            .and_then(|rest| rest.strip_suffix(".weight"))
            .and_then(|rest| rest.rsplit('.').next())
            .is_some_and(|projection| PROJECTIONS.contains(&projection))
}
fn rows_cols(shape: &[usize]) -> (usize, usize) {
    (
        shape.first().copied().unwrap_or(1),
        shape.iter().skip(1).product::<usize>(),
    )
}
/// Permutes row-major BF16 bytes into the tiled layout.
pub fn tile_rows(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    permute_tiles(src, rows, cols, true)
}
/// Restores row-major BF16 bytes from the tiled layout.
pub fn untile_rows(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    permute_tiles(src, rows, cols, false)
}
fn permute_tiles(src: &[u8], rows: usize, cols: usize, to_tiled: bool) -> Vec<u8> {
    assert!(
        rows.is_multiple_of(TILE_ROWS) && cols.is_multiple_of(TILE_COLS),
        "tiled layout requires 256-row and 64-column multiples"
    );
    assert_eq!(src.len(), rows * cols * 2, "BF16 byte length");
    let mut dst = vec![0u8; src.len()];
    let k_steps = cols / TILE_COLS;
    let row_bytes = cols * 2;
    let segment = TILE_COLS * 2;
    let tile_bytes = TILE_ROWS * row_bytes;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per_thread = (rows / TILE_ROWS).div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (chunk, dst) in dst.chunks_mut(per_thread * tile_bytes).enumerate() {
            let first_tile = chunk * per_thread;
            scope.spawn(move || {
                for (local, dst) in dst.chunks_mut(tile_bytes).enumerate() {
                    let tile = first_tile + local;
                    let src = &src[tile * tile_bytes..(tile + 1) * tile_bytes];
                    for k_step in 0..k_steps {
                        for r in 0..TILE_ROWS {
                            let dense = r * row_bytes + k_step * segment;
                            let tiled = (k_step * TILE_ROWS + r) * segment;
                            let (from, to) = if to_tiled {
                                (dense, tiled)
                            } else {
                                (tiled, dense)
                            };
                            dst[to..to + segment].copy_from_slice(&src[from..from + segment]);
                        }
                    }
                }
            });
        }
    });
    dst
}
#[derive(Debug, Deserialize, Serialize, Clone)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}
fn read_header(file: &mut std::fs::File) -> Result<(u64, serde_json::Map<String, Value>)> {
    let mut len = [0u8; 8];
    file.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    ensure!(len <= 100 << 20, "unreasonable safetensors header");
    let mut header = vec![0u8; len as usize];
    file.read_exact(&mut header)?;
    let header: Value = serde_json::from_slice(&header)?;
    let Value::Object(header) = header else {
        anyhow::bail!("safetensors header is not an object");
    };
    Ok((len, header))
}
/// Rewrites a shard in place so its backbone projections use the tiled layout.
///
/// Returns whether the file changed. Shards already carrying the layout marker,
/// and shards without any tiled projection, are left untouched. The rewrite goes
/// through a temporary file in the shard's directory and an atomic rename, so an
/// interrupted conversion leaves the original intact. Checkpoints converted this
/// way can only be loaded by binaries that understand the marker.
pub fn ensure_tiled_shard(path: &Path) -> Result<bool> {
    let target = std::fs::canonicalize(path)?;
    let mut file = std::fs::File::open(&target)?;
    let (header_len, mut header) = read_header(&mut file)?;
    let data_start = 8 + header_len;
    let metadata = header
        .get(TILED_LAYOUT_KEY_METADATA)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if metadata.get(TILED_LAYOUT_KEY).and_then(Value::as_str) == Some(TILED_LAYOUT) {
        return Ok(false);
    }
    let mut entries: Vec<(String, HeaderEntry)> = header
        .iter()
        .filter(|(name, _)| name.as_str() != TILED_LAYOUT_KEY_METADATA)
        .map(|(name, value)| Ok((name.clone(), serde_json::from_value(value.clone())?)))
        .collect::<Result<_>>()?;
    entries.sort_by_key(|(_, entry)| entry.data_offsets[0]);
    let tiled: Vec<bool> = entries
        .iter()
        .map(|(name, entry)| {
            let (rows, cols) = rows_cols(&entry.shape);
            entry.dtype == "BF16" && tiled_projection(name, rows, cols)
        })
        .collect();
    if !tiled.contains(&true) {
        return Ok(false);
    }
    let total: u64 = entries
        .iter()
        .map(|(_, entry)| entry.data_offsets[1] - entry.data_offsets[0])
        .sum();
    eprintln!(
        "Converting {} ({:.1} GiB) to the tiled projection layout; this happens once",
        target.display(),
        total as f64 / (1u64 << 30) as f64
    );
    let mut metadata = metadata;
    metadata.insert(TILED_LAYOUT_KEY.into(), Value::String(TILED_LAYOUT.into()));
    header.insert(TILED_LAYOUT_KEY_METADATA.into(), Value::Object(metadata));
    let mut encoded = serde_json::to_vec(&Value::Object(header))?;
    while !encoded.len().is_multiple_of(8) {
        encoded.push(b' ');
    }
    let temporary = target.with_extension(format!("safetensors.retiling.{}", std::process::id()));
    let result = (|| -> Result<()> {
        let mut out = BufWriter::with_capacity(16 << 20, std::fs::File::create(&temporary)?);
        out.write_all(&(encoded.len() as u64).to_le_bytes())?;
        out.write_all(&encoded)?;
        let mut expected = 0u64;
        for ((name, entry), tiled) in entries.iter().zip(tiled) {
            let [start, end] = entry.data_offsets;
            ensure!(
                start == expected && end >= start,
                "non-contiguous tensor data in {name}"
            );
            expected = end;
            file.seek(SeekFrom::Start(data_start + start))?;
            let bytes = (end - start) as usize;
            if tiled {
                let (rows, cols) = rows_cols(&entry.shape);
                let mut dense = vec![0u8; bytes];
                file.read_exact(&mut dense)?;
                out.write_all(&tile_rows(&dense, rows, cols))?;
            } else {
                let mut remaining = bytes;
                let mut chunk = vec![0u8; (64 << 20).min(bytes.max(1))];
                while remaining > 0 {
                    let n = chunk.len().min(remaining);
                    file.read_exact(&mut chunk[..n])?;
                    out.write_all(&chunk[..n])?;
                    remaining -= n;
                }
            }
        }
        out.flush()?;
        out.get_ref().sync_all()?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(error).with_context(|| format!("converting {}", target.display()));
    }
    std::fs::rename(&temporary, &target)
        .with_context(|| format!("replacing {}", target.display()))?;
    Ok(true)
}
const TILED_LAYOUT_KEY_METADATA: &str = "__metadata__";
#[derive(Debug, Clone, Default)]
pub struct Source {
    pub model: ModelVariant,
    pub directory: Option<PathBuf>,
    pub offline: bool,
}
impl Source {
    pub fn resolve(&self, name: &str) -> Result<PathBuf> {
        ensure!(
            Path::new(name)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
            "invalid checkpoint filename {name}"
        );
        if let Some(root) = &self.directory {
            let p = root.join(name);
            ensure!(p.is_file(), "missing checkpoint file {}", p.display());
            return Ok(p);
        }
        Ok(Resolver::new(
            Repository::new("Cloudflare", self.model.name()).at(self.model.revision()),
        )
        .offline(self.offline)
        .resolve(&HubFile::new(name))?)
    }
    pub fn json(&self, name: &str) -> Result<Value> {
        Ok(serde_json::from_slice(&std::fs::read(
            self.resolve(name)?,
        )?)?)
    }
    pub fn validate_config(&self) -> Result<()> {
        for name in [
            "config.json",
            "joint_head_config.json",
            "processor_config.json",
        ] {
            let reference = self.model.reference(name);
            let actual = self.json(name)?;
            let expected: Value = serde_json::from_str(reference)?;
            // Only informational producer metadata may differ.
            let mut actual = actual;
            let mut expected = expected;
            for v in [&mut actual, &mut expected] {
                if let Some(m) = v.as_object_mut() {
                    m.remove("transformers_version");
                    m.remove("_name_or_path");
                }
            }
            ensure!(
                actual == expected,
                "unsupported {name}; expected pinned {} architecture/processor",
                self.model.name()
            );
        }
        Ok(())
    }
    pub fn inspect(&self) -> Result<Inspection> {
        self.validate_config()?;
        let index: Index = serde_json::from_value(self.json("model.safetensors.index.json")?)?;
        validate_index(&index, self.model)?;
        let mut files: Vec<_> = index
            .weight_map
            .values()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        files.extend(
            [
                "joint_head.safetensors",
                "config.json",
                "joint_head_config.json",
                "processor_config.json",
                "tokenizer.json",
                "model.safetensors.index.json",
            ]
            .map(String::from),
        );
        let head: BTreeMap<String, Meta> =
            serde_json::from_str(self.model.reference("joint_head_metadata.json"))?;
        let head_bytes: usize = head
            .values()
            .map(|m| m.shape.iter().product::<usize>() * 2)
            .sum();
        Ok(Inspection {
            model: self.model,
            revision: self.model.revision().into(),
            architecture: "Qwen3_5 + JointSchemaHead".into(),
            backbone_layers: self.model.architecture().layers,
            vision_layers: 27,
            weight_bytes: index.metadata.total_size + head_bytes as u64,
            resident_weight_bytes: index.metadata.total_size + head_bytes as u64
                - self.model.architecture().embedding_bytes(),
            files,
            workspace_estimate_bytes: self
                .model
                .architecture()
                .workspace_estimate(crate::encoding::MAX_CONTEXT_TOKENS),
            qualification: "see model-specific qualification in docs".into(),
        })
    }
}
#[derive(Debug, Deserialize)]
struct Index {
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}
#[derive(Debug, Deserialize)]
struct IndexMetadata {
    total_size: u64,
}
#[derive(Debug, Deserialize)]
struct Meta {
    shape: Vec<usize>,
    dtype: String,
}
/// Resident storage after optional packing; excludes host-backed embedding tables.
pub(crate) fn resident_weight_bytes(
    model: ModelVariant,
    precision: crate::Precision,
) -> Result<u64> {
    let packing = crate::quantized::QuantizedProjections::for_precision(precision);
    let mut bytes = 0;
    for metadata in ["backbone_metadata.json", "joint_head_metadata.json"] {
        let tensors: BTreeMap<String, Meta> = serde_json::from_str(model.reference(metadata))?;
        for (name, tensor) in tensors {
            if crate::embedding::TABLES.contains(&name.as_str()) {
                continue;
            }
            let (rows, cols) = rows_cols(&tensor.shape);
            bytes += packing.resident_bytes(&name, rows, cols);
        }
    }
    Ok(bytes)
}
#[derive(Debug, Serialize)]
pub struct Inspection {
    pub model: ModelVariant,
    pub revision: String,
    pub architecture: String,
    pub backbone_layers: usize,
    pub vision_layers: usize,
    pub weight_bytes: u64,
    pub resident_weight_bytes: u64,
    pub workspace_estimate_bytes: u64,
    pub files: Vec<String>,
    pub qualification: String,
}
pub fn workspace_estimate(tokens: usize) -> u64 {
    // Conservative live activation allowance, plus recurrent state, scratch and staging.
    ModelVariant::Clef.architecture().workspace_estimate(tokens)
}
fn validate_index(index: &Index, model: ModelVariant) -> Result<()> {
    let expected: Index = serde_json::from_str(model.reference("model.safetensors.index.json"))?;
    ensure!(
        index.metadata.total_size == expected.metadata.total_size,
        "unexpected checkpoint size"
    );
    ensure!(
        index.weight_map == expected.weight_map,
        "checkpoint tensor-to-shard index differs from pinned release"
    );
    Ok(())
}
/// Keep pristine downloaded weights intact; derived layouts live beside the snapshot.
fn tiled_path(source: &Source, name: &str, path: &Path) -> Result<PathBuf> {
    let (_, header) = read_header(&mut std::fs::File::open(path)?)?;
    if header
        .get(TILED_LAYOUT_KEY_METADATA)
        .and_then(|v| v.get(TILED_LAYOUT_KEY))
        .and_then(Value::as_str)
        == Some(TILED_LAYOUT)
        || !header.iter().any(|(key, value)| {
            serde_json::from_value::<HeaderEntry>(value.clone()).is_ok_and(|entry| {
                let (rows, cols) = rows_cols(&entry.shape);
                tiled_projection(key, rows, cols)
            })
        })
    {
        return Ok(path.to_path_buf());
    }
    let root = path
        .parent()
        .context("checkpoint has no parent")?
        .join(".clef-hrx")
        .join(source.model.name())
        .join(source.model.revision())
        .join(TILED_LAYOUT);
    std::fs::create_dir_all(&root)?;
    let target = root.join(name);
    // Hash source bytes before conversion. A different local checkpoint must not
    // accidentally reuse a derived layout with matching tensor shapes.
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    let mut input = std::fs::File::open(path)?;
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let digest = format!("{:x}", hash.finalize());
    let receipt = target.with_extension("sha256");
    if target.is_file() && std::fs::read_to_string(&receipt).ok().as_deref() == Some(&digest) {
        return Ok(target);
    }
    let temporary = target.with_extension(format!("copy.{}", std::process::id()));
    std::fs::copy(path, &temporary)?;
    ensure_tiled_shard(&temporary)?;
    std::fs::rename(&temporary, &target)?;
    std::fs::write(receipt, digest)?;
    Ok(target)
}
pub struct Checkpoint {
    files: Vec<FileView>,
    tensors: BTreeMap<String, usize>,
}
impl Checkpoint {
    pub(crate) fn embedding_rows(&self, name: &str) -> Result<crate::embedding::EmbeddingRows> {
        let i = *self.tensors.get(name).context("missing embedding tensor")?;
        crate::embedding::EmbeddingRows::open(&self.files[i], name)
    }
    /// # Safety
    /// Checkpoint files must remain unchanged while this mapping is alive.
    pub unsafe fn open(source: &Source) -> Result<Self> {
        source.validate_config()?;
        let index: Index = serde_json::from_value(source.json("model.safetensors.index.json")?)?;
        validate_index(&index, source.model)?;
        let backbone: BTreeMap<String, Meta> =
            serde_json::from_str(source.model.reference("backbone_metadata.json"))?;
        let head: BTreeMap<String, Meta> =
            serde_json::from_str(source.model.reference("joint_head_metadata.json"))?;
        let mut result = Self {
            files: Vec::new(),
            tensors: BTreeMap::new(),
        };
        for name in index
            .weight_map
            .values()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .chain(std::iter::once("joint_head.safetensors".into()))
        {
            let path = source.resolve(&name)?;
            // Projection weights are stored tiled; convert a pristine shard first.
            let path = tiled_path(source, &name, &path)?;
            // Safety: caller guarantees immutable checkpoint files.
            let file = unsafe { FileView::map(&path)? };
            let is_head = name == "joint_head.safetensors";
            let expected = if is_head { &head } else { &backbone };
            for (key, entry) in file.entries() {
                let meta = expected
                    .get(key)
                    .with_context(|| format!("unexpected tensor {key}"))?;
                ensure!(
                    meta.dtype == "BF16" && entry.dtype == DType::BF16 && entry.shape == meta.shape,
                    "invalid dtype or shape for {key}"
                );
                if !is_head {
                    ensure!(
                        index.weight_map.get(key) == Some(&name),
                        "{key} in wrong shard"
                    );
                }
                let key = if is_head {
                    format!("head.{key}")
                } else {
                    key.clone()
                };
                ensure!(
                    result.tensors.insert(key, result.files.len()).is_none(),
                    "duplicate tensor"
                );
            }
            result.files.push(file);
        }
        ensure!(
            result.tensors.len() == backbone.len() + head.len(),
            "missing checkpoint tensors"
        );
        Ok(result)
    }
    pub fn done_with(&self, name: &str) -> Result<()> {
        let i = *self
            .tensors
            .get(name)
            .with_context(|| format!("missing {name}"))?;
        self.files[i].done_with(self.tensor(name)?);
        Ok(())
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
    pub fn tensor(&self, name: &str) -> Result<Tensor<'_>> {
        let i = *self
            .tensors
            .get(name)
            .with_context(|| format!("missing {name}"))?;
        Ok(self.files[i].get(name.strip_prefix("head.").unwrap_or(name))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_estimate_respects_precision_and_pinned_mlp_shapes() -> Result<()> {
        for model in [ModelVariant::Clef, ModelVariant::ClefFlash] {
            let source = Source {
                model,
                ..Default::default()
            };
            // Inspection reads pinned metadata; this check needs no model or GPU.
            let index: Index =
                serde_json::from_str(model.reference("model.safetensors.index.json"))?;
            let head: BTreeMap<String, Meta> =
                serde_json::from_str(model.reference("joint_head_metadata.json"))?;
            let dense = index.metadata.total_size
                + head
                    .values()
                    .map(|m| m.shape.iter().product::<usize>() as u64 * 2)
                    .sum::<u64>()
                - source.model.architecture().embedding_bytes();
            assert_eq!(resident_weight_bytes(model, crate::Precision::Bf16)?, dense);
            let arch = model.architecture();
            let (hidden, intermediate, layers) = (
                arch.hidden as u64,
                arch.intermediate as u64,
                arch.layers as u64,
            );
            let mlp_elements = 3 * layers * hidden * intermediate;
            let mlp_scales = layers * (2 * intermediate + hidden) * 4;
            let mlp = resident_weight_bytes(model, crate::Precision::Int4Mlp)?;
            assert_eq!(mlp, dense - mlp_elements * 3 / 2 + mlp_scales);
            let broader = resident_weight_bytes(model, crate::Precision::Int4)?;
            assert!(broader < mlp && mlp < dense);
            // Regression: CLEF INT4 must pass the 40 GiB admission that BF16 rejects.
            if model == ModelVariant::Clef {
                let workspace = arch.workspace_estimate(0);
                assert!(dense + workspace > 40 << 30);
                assert!(broader + workspace < 40 << 30);
                assert!(mlp + workspace < 48 << 30);
            }
        }
        Ok(())
    }

    fn position(cols: usize, n: usize, k: usize) -> usize {
        (((n / TILE_ROWS) * (cols / TILE_COLS) + k / TILE_COLS) * TILE_ROWS + n % TILE_ROWS)
            * TILE_COLS
            + k % TILE_COLS
    }

    #[test]
    fn local_model_variant_mismatch_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "config.json",
            "joint_head_config.json",
            "processor_config.json",
        ] {
            std::fs::write(
                dir.path().join(name),
                ModelVariant::ClefFlash.reference(name),
            )
            .unwrap();
        }
        let mut source = Source {
            model: ModelVariant::ClefFlash,
            directory: Some(dir.path().into()),
            offline: true,
        };
        source.validate_config().unwrap();
        source.model = ModelVariant::Clef;
        assert!(source.validate_config().is_err());
    }

    #[test]
    fn derived_layout_preserves_source_and_invalidates_changed_weights() {
        let dir = tempfile::tempdir().unwrap();
        let name = "test.safetensors";
        let path = dir.path().join(name);
        let key = "model.language_model.layers.0.mlp.up_proj.weight";
        let dense: Vec<_> = (0..256 * 64 * 2).map(|i| (i % 251) as u8).collect();
        write_safetensors(&path, &[(key, vec![256, 64], dense.clone())]);
        let before = std::fs::read(&path).unwrap();
        let source = Source {
            directory: Some(dir.path().into()),
            ..Default::default()
        };
        let derived = tiled_path(&source, name, &path).unwrap();
        assert_ne!(path, derived);
        assert_eq!(before, std::fs::read(&path).unwrap());
        assert_eq!(derived, tiled_path(&source, name, &path).unwrap());
        let old = std::fs::read(&derived).unwrap();
        let other = Source {
            model: ModelVariant::ClefFlash,
            ..source.clone()
        };
        assert_ne!(derived, tiled_path(&other, name, &path).unwrap());
        write_safetensors(&path, &[(key, vec![256, 64], vec![7; dense.len()])]);
        tiled_path(&source, name, &path).unwrap();
        assert_ne!(old, std::fs::read(derived).unwrap());
    }

    #[test]
    fn tiled_projection_policy() {
        let layer = "model.language_model.layers.3";
        assert!(tiled_projection(
            &format!("{layer}.mlp.up_proj.weight"),
            17408,
            5120
        ));
        assert!(tiled_projection(
            &format!("{layer}.self_attn.k_proj.weight"),
            1024,
            5120
        ));
        assert!(tiled_projection(
            &format!("{layer}.linear_attn.in_proj_qkv.weight"),
            10240,
            5120
        ));
        assert!(tiled_projection(
            &format!("{layer}.linear_attn.out_proj.weight"),
            5120,
            6144
        ));
        assert!(!tiled_projection(
            &format!("{layer}.linear_attn.in_proj_a.weight"),
            48,
            5120
        ));
        assert!(!tiled_projection(
            &format!("{layer}.linear_attn.conv1d.weight"),
            10240,
            4
        ));
        assert!(!tiled_projection(
            &format!("{layer}.mlp.up_proj.bias"),
            17408,
            1
        ));
        assert!(!tiled_projection(
            &format!("{layer}.input_layernorm.weight"),
            5120,
            1
        ));
        assert!(!tiled_projection(
            "model.visual.merger.linear_fc2.weight",
            5120,
            4608
        ));
        assert!(!tiled_projection(
            "head.memory_projection.weight",
            1024,
            5120
        ));
        assert!(!tiled_projection("lm_head.weight", 248320, 5120));
        assert!(!tiled_projection(
            &format!("{layer}.mlp.up_proj.weight"),
            17400,
            5120
        ));
        assert!(!tiled_projection(
            &format!("{layer}.mlp.up_proj.weight"),
            17408,
            5100
        ));
    }

    #[test]
    fn tile_and_untile_round_trip() {
        let (rows, cols) = (512, 192);
        let dense: Vec<u8> = (0..rows * cols * 2).map(|i| (i * 7 % 251) as u8).collect();
        let tiled = tile_rows(&dense, rows, cols);
        assert_eq!(tiled.len(), dense.len());
        for (n, k) in [
            (0, 0),
            (1, 0),
            (0, 1),
            (255, 63),
            (256, 64),
            (300, 150),
            (511, 191),
        ] {
            let dense_offset = (n * cols + k) * 2;
            let tiled_offset = position(cols, n, k) * 2;
            assert_eq!(
                tiled[tiled_offset..tiled_offset + 2],
                dense[dense_offset..dense_offset + 2],
                "element ({n}, {k})"
            );
        }
        // Each 256x64 block is contiguous and equal to the dense block.
        // Tile row 1, K-step 2 of three.
        let block = &tiled[5 * TILE_ROWS * TILE_COLS * 2..][..TILE_ROWS * TILE_COLS * 2];
        for r in 0..TILE_ROWS {
            let from = ((256 + r) * cols + 128) * 2;
            assert_eq!(&block[r * 128..(r + 1) * 128], &dense[from..from + 128]);
        }
        assert_eq!(untile_rows(&tiled, rows, cols), dense);
    }

    fn write_safetensors(path: &Path, tensors: &[(&str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        let mut data = Vec::new();
        for (name, shape, bytes) in tensors {
            let end = offset + bytes.len() as u64;
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [offset, end]}),
            );
            data.extend_from_slice(bytes);
            offset = end;
        }
        let mut encoded = serde_json::to_vec(&Value::Object(header)).unwrap();
        while !encoded.len().is_multiple_of(8) {
            encoded.push(b' ');
        }
        let mut out = Vec::new();
        out.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
        out.extend_from_slice(&encoded);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    /// Converts every shard of `CLEF_MODEL_DIR` without loading the model.
    /// Ordinary loading performs the same conversion on first use.
    #[test]
    #[ignore = "rewrites the local checkpoint named by CLEF_MODEL_DIR"]
    fn retile_local_checkpoint() -> Result<()> {
        let source = Source {
            model: Default::default(),
            directory: std::env::var_os("CLEF_MODEL_DIR").map(Into::into),
            offline: true,
        };
        for name in source.inspect()?.files {
            if name.ends_with(".safetensors") {
                let converted = ensure_tiled_shard(&source.resolve(&name)?)?;
                eprintln!(
                    "{name}: {}",
                    if converted { "converted" } else { "unchanged" }
                );
            }
        }
        Ok(())
    }

    #[test]
    fn shard_conversion_is_idempotent_and_exact() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model-00001-of-00001.safetensors");
        let (rows, cols) = (256, 128);
        let projection: Vec<u8> = (0..rows * cols * 2).map(|i| (i % 253) as u8).collect();
        let norm: Vec<u8> = (0..cols * 2).map(|i| (i % 7) as u8).collect();
        let gate: Vec<u8> = (0..48 * cols * 2).map(|i| (i % 11) as u8).collect();
        write_safetensors(
            &path,
            &[
                (
                    "model.language_model.layers.0.input_layernorm.weight",
                    vec![cols],
                    norm.clone(),
                ),
                (
                    "model.language_model.layers.0.mlp.down_proj.weight",
                    vec![rows, cols],
                    projection.clone(),
                ),
                (
                    "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                    vec![48, cols],
                    gate.clone(),
                ),
            ],
        );
        let original = std::fs::read(&path)?;
        assert!(ensure_tiled_shard(&path)?);
        assert!(!ensure_tiled_shard(&path)?);
        let converted = std::fs::read(&path)?;
        assert_eq!(converted.len() % 8, original.len() % 8);
        let view = unsafe { FileView::map(&path)? };
        let tensor = view.get("model.language_model.layers.0.mlp.down_proj.weight")?;
        assert_eq!(tensor.shape, vec![rows, cols]);
        assert_eq!(tensor.bytes, tile_rows(&projection, rows, cols).as_slice());
        assert_eq!(untile_rows(tensor.bytes, rows, cols), projection);
        assert_eq!(
            view.get("model.language_model.layers.0.input_layernorm.weight")?
                .bytes,
            norm.as_slice()
        );
        assert_eq!(
            view.get("model.language_model.layers.0.linear_attn.in_proj_a.weight")?
                .bytes,
            gate.as_slice()
        );
        let mut file = std::fs::File::open(&path)?;
        let (_, header) = read_header(&mut file)?;
        assert_eq!(
            header[TILED_LAYOUT_KEY_METADATA][TILED_LAYOUT_KEY],
            Value::String(TILED_LAYOUT.into())
        );
        // Shards without tiled projections keep their bytes and gain no marker.
        let plain = dir.path().join("joint_head.safetensors");
        write_safetensors(
            &plain,
            &[("memory_projection.weight", vec![rows, cols], projection)],
        );
        let before = std::fs::read(&plain)?;
        assert!(!ensure_tiled_shard(&plain)?);
        assert_eq!(std::fs::read(&plain)?, before);
        Ok(())
    }
}
