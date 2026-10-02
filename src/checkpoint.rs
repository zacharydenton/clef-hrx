//! Pinned Hub resolution and strict, memory-mapped checkpoint validation.
use crate::Result;
use anyhow::{Context, ensure};
use hrx::artifacts::{
    hf::{HubFile, Repository, Resolver},
    safetensors::{DType, FileView, Tensor},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

pub const REVISION: &str = "2f3de3dd85f379784083b0814d997ab627200f0c";
#[derive(Debug, Clone, Default)]
pub struct Source {
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
        Ok(
            Resolver::new(Repository::new("Cloudflare", "clef").at(REVISION))
                .offline(self.offline)
                .resolve(&HubFile::new(name))?,
        )
    }
    pub fn json(&self, name: &str) -> Result<Value> {
        Ok(serde_json::from_slice(&std::fs::read(
            self.resolve(name)?,
        )?)?)
    }
    pub fn validate_config(&self) -> Result<()> {
        for (name, reference) in [
            ("config.json", include_str!("../reference/config.json")),
            (
                "joint_head_config.json",
                include_str!("../reference/joint_head_config.json"),
            ),
            (
                "processor_config.json",
                include_str!("../reference/processor_config.json"),
            ),
        ] {
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
                "unsupported {name}; expected pinned CLEF architecture/processor"
            );
        }
        Ok(())
    }
    pub fn inspect(&self) -> Result<Inspection> {
        self.validate_config()?;
        let index: Index = serde_json::from_value(self.json("model.safetensors.index.json")?)?;
        validate_index(&index)?;
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
            serde_json::from_str(include_str!("../reference/joint_head_metadata.json"))?;
        let head_bytes: usize = head
            .values()
            .map(|m| m.shape.iter().product::<usize>() * 2)
            .sum();
        Ok(Inspection {
            revision: REVISION.into(),
            architecture: "Qwen3_5 + JointSchemaHead".into(),
            backbone_layers: 64,
            vision_layers: 27,
            weight_bytes: index.metadata.total_size + head_bytes as u64,
            files,
            workspace_estimate_bytes: workspace_estimate(16384),
            qualification: "experimental; full-checkpoint parity required".into(),
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
#[derive(Debug, Serialize)]
pub struct Inspection {
    pub revision: String,
    pub architecture: String,
    pub backbone_layers: usize,
    pub vision_layers: usize,
    pub weight_bytes: u64,
    pub workspace_estimate_bytes: u64,
    pub files: Vec<String>,
    pub qualification: String,
}
pub fn workspace_estimate(tokens: usize) -> u64 {
    // Conservative live activation allowance, plus recurrent state, scratch and staging.
    (tokens as u64 * 17408 * 2 * 12) + (2 << 30)
}
fn validate_index(index: &Index) -> Result<()> {
    let expected: Index =
        serde_json::from_str(include_str!("../reference/model.safetensors.index.json"))?;
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
pub struct Checkpoint {
    files: Vec<FileView>,
    tensors: BTreeMap<String, usize>,
}
impl Checkpoint {
    /// # Safety
    /// Checkpoint files must remain unchanged while this mapping is alive.
    pub unsafe fn open(source: &Source) -> Result<Self> {
        source.validate_config()?;
        let index: Index = serde_json::from_value(source.json("model.safetensors.index.json")?)?;
        validate_index(&index)?;
        let backbone: BTreeMap<String, Meta> =
            serde_json::from_str(include_str!("../reference/backbone_metadata.json"))?;
        let head: BTreeMap<String, Meta> =
            serde_json::from_str(include_str!("../reference/joint_head_metadata.json"))?;
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
