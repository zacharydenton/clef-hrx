//! Read exact BF16 lookup rows without keeping vocabulary tables resident.
use crate::{
    Result,
    gpu::{DType, Engine, Tensor},
};
use anyhow::{Context, ensure};
use hrx::artifacts::safetensors::{DType as FileDType, FileView};
use std::{fs::File, os::unix::fs::FileExt};

pub(crate) const TABLES: [&str; 2] = ["model.language_model.embed_tokens.weight", "lm_head.weight"];
pub(crate) const TABLE_BYTES: u64 = 2 * 248320 * 5120 * 2;

pub(crate) struct EmbeddingRows {
    file: File,
    offset: u64,
    rows: usize,
    cols: usize,
}

impl EmbeddingRows {
    /// The view has already validated the SafeTensors header and data bounds.
    pub fn open(view: &FileView, name: &str) -> Result<Self> {
        let entry = view
            .entries()
            .get(name)
            .context("missing embedding table")?;
        ensure!(
            entry.dtype == FileDType::BF16
                && entry.shape.len() == 2
                && entry.shape.iter().all(|&n| n > 0),
            "embedding table must be a nonempty BF16 matrix"
        );
        let file = File::open(view.path())?;
        let mut prefix = [0; 8];
        file.read_exact_at(&mut prefix, 0)?;
        let offset = u64::from_le_bytes(prefix)
            .checked_add(8)
            .and_then(|n| n.checked_add(entry.offset as u64))
            .context("embedding offset overflow")?;
        let file_bytes = file.metadata()?.len();
        ensure!(
            offset
                .checked_add(entry.bytes as u64)
                .is_some_and(|end| end <= file_bytes),
            "embedding table exceeds file bounds"
        );
        Ok(Self {
            file,
            offset,
            rows: entry.shape[0],
            cols: entry.shape[1],
        })
    }

    pub fn read(&self, ids: &[u32]) -> Result<Vec<u8>> {
        ensure!(
            ids.iter().all(|&id| (id as usize) < self.rows),
            "embedding token ID out of bounds"
        );
        let row_bytes = self
            .cols
            .checked_mul(2)
            .context("embedding row size overflow")?;
        let bytes = ids
            .len()
            .checked_mul(row_bytes)
            .context("embedding input size overflow")?;
        crate::memory::before_allocation(bytes)?;
        let mut output = vec![0; bytes];
        for (&id, row) in ids.iter().zip(output.chunks_exact_mut(row_bytes)) {
            self.file
                .read_exact_at(row, self.offset + id as u64 * row_bytes as u64)?;
        }
        Ok(output)
    }
}

impl Engine {
    pub fn embedding(&mut self, name: &str, ids: &[u32]) -> Result<Tensor> {
        let table = self
            .embedding_rows
            .get(name)
            .with_context(|| format!("missing embedding table {name}"))?;
        let cols = table.cols;
        let bytes = table.read(ids)?;
        self.input(ids.len(), cols, DType::Bf16, &bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn lookup_preserves_order_duplicates_and_bf16_bits() -> Result<()> {
        let mut file = tempfile::NamedTempFile::new()?;
        let header = br#"{"prefix":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"x":{"dtype":"BF16","shape":[3,2],"data_offsets":[2,14]}}"#;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(header)?;
        file.write_all(&[42, 42])?;
        let bytes = [0, 0, 128, 63, 128, 127, 192, 127, 128, 255, 0, 128];
        file.write_all(&bytes)?;
        let view = FileView::read(file.path())?;
        let table = EmbeddingRows::open(&view, "x")?;
        assert!(EmbeddingRows::open(&view, "prefix").is_err());
        assert!(EmbeddingRows::open(&view, "missing").is_err());
        assert_eq!(
            table.read(&[2, 0, 2, 1])?,
            [
                bytes[8..12].to_vec(),
                bytes[0..4].to_vec(),
                bytes[8..12].to_vec(),
                bytes[4..8].to_vec()
            ]
            .concat()
        );
        assert!(table.read(&[3]).is_err());
        assert!(table.read(&[u32::MAX]).is_err());
        assert!(table.read(&[])?.is_empty());
        file.as_file().set_len(8 + header.len() as u64 + 2 + 4)?;
        assert!(table.read(&[1]).is_err());
        assert!(EmbeddingRows::open(&view, "x").is_err());
        Ok(())
    }
}
