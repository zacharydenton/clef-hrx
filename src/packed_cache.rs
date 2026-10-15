//! Opt-in development cache of the exact GPU-produced integer weight bytes.
use crate::Result;
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const MAGIC: &[u8; 8] = b"CLFPACK1";
const PACKING: &str = concat!(
    "tiled256x64-to-rotated-codes128x32-v1\n",
    include_str!("../kernels/common.loom"),
    include_str!("../kernels/prepare_integer.loom"),
    include_str!("../kernels/prepare_integer_single.loom"),
    include_str!("../kernels/prepare_integer_register.loom"),
    include_str!("../kernels/retile_codes.loom"),
);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Key {
    name: String,
    rows: usize,
    cols: usize,
    bits: usize,
    compiler: String,
    target: String,
    source_sha256: String,
    packing_sha256: String,
}
impl Key {
    fn sizes(&self) -> Result<(usize, usize)> {
        ensure!(
            [4, 8].contains(&self.bits)
                && self.rows > 0
                && self.cols > 0
                && self.rows.is_multiple_of(256)
                && self.cols.is_multiple_of(256),
            "invalid packed-cache extent"
        );
        let codes = self
            .rows
            .checked_mul(self.cols)
            .and_then(|n| n.checked_mul(self.bits))
            .context("packed-cache size overflow")?
            / 8;
        let scales = self
            .rows
            .checked_mul(4)
            .context("packed-cache scale overflow")?;
        Ok((codes, scales))
    }
}

#[derive(Serialize, Deserialize)]
struct Header {
    key: Key,
    codes_sha256: String,
    scales_sha256: String,
}
#[derive(Debug)]
pub(crate) struct PackedBytes {
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
}
pub(crate) struct PackedWeightCache {
    directory: PathBuf,
    compiler: String,
    target: String,
    pub hits: usize,
    pub misses: usize,
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
impl PackedWeightCache {
    pub fn new(directory: PathBuf, compiler: &str, target: &str) -> Self {
        Self {
            directory,
            compiler: compiler.into(),
            target: target.into(),
            hits: 0,
            misses: 0,
        }
    }
    pub fn key(&self, name: &str, rows: usize, cols: usize, bits: usize, source: &[u8]) -> Key {
        Key {
            name: name.into(),
            rows,
            cols,
            bits,
            compiler: self.compiler.clone(),
            target: self.target.clone(),
            source_sha256: digest(source),
            packing_sha256: digest(PACKING.as_bytes()),
        }
    }
    fn path(&self, key: &Key) -> Result<PathBuf> {
        Ok(self
            .directory
            .join(format!("{}.pack", digest(&serde_json::to_vec(key)?))))
    }
    pub fn read(&mut self, key: &Key) -> Result<Option<PackedBytes>> {
        let path = self.path(key)?;
        let mut file = match std::fs::File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                self.misses += 1;
                return Ok(None);
            }
            Err(error) => return Err(error.into()),
        };
        let (code_bytes, scale_bytes) = key.sizes()?;
        let mut prefix = [0u8; 12];
        file.read_exact(&mut prefix)
            .context("truncated packed-cache prefix")?;
        ensure!(&prefix[..8] == MAGIC, "invalid packed-cache magic");
        let header_bytes = u32::from_le_bytes(prefix[8..].try_into()?) as usize;
        ensure!(header_bytes <= 65536, "oversized packed-cache header");
        let expected_len = 12u64 + header_bytes as u64 + code_bytes as u64 + scale_bytes as u64;
        ensure!(
            file.metadata()?.len() == expected_len,
            "packed-cache length mismatch"
        );
        let mut header = vec![0; header_bytes];
        file.read_exact(&mut header)?;
        let header: Header = serde_json::from_slice(&header)?;
        ensure!(header.key == *key, "packed-cache key mismatch");
        let mut codes = vec![0; code_bytes];
        let mut scales = vec![0; scale_bytes];
        file.read_exact(&mut codes)?;
        file.read_exact(&mut scales)?;
        ensure!(
            digest(&codes) == header.codes_sha256 && digest(&scales) == header.scales_sha256,
            "packed-cache checksum mismatch: {}",
            path.display()
        );
        self.hits += 1;
        Ok(Some(PackedBytes { codes, scales }))
    }
    pub fn write(&self, key: &Key, codes: &[u8], scales: &[u8]) -> Result<()> {
        ensure!(
            key.sizes()? == (codes.len(), scales.len()),
            "packed-cache data size mismatch"
        );
        let header = serde_json::to_vec(&Header {
            key: key.clone(),
            codes_sha256: digest(codes),
            scales_sha256: digest(scales),
        })?;
        ensure!(header.len() <= 65536, "oversized packed-cache header");
        std::fs::create_dir_all(&self.directory)?;
        let path = self.path(key)?;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let temporary = path.with_extension(format!(
            "tmp.{}.{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let result = write_atomic(&path, &temporary, &header, codes, scales);
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        result
    }
}
fn write_atomic(
    path: &Path,
    temporary: &Path,
    header: &[u8],
    codes: &[u8],
    scales: &[u8],
) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temporary)?;
    file.write_all(MAGIC)?;
    file.write_all(&(header.len() as u32).to_le_bytes())?;
    file.write_all(header)?;
    file.write_all(codes)?;
    file.write_all(scales)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_and_compiler_changes_miss_and_corruption_is_rejected() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let mut cache = PackedWeightCache::new(directory.path().into(), "compiler-a", "gfx1151");
        let key = cache.key("weight", 256, 256, 4, b"source");
        assert!(cache.read(&key)?.is_none());
        let codes = vec![0x5a; 32768];
        let scales = vec![0x17; 1024];
        cache.write(&key, &codes, &scales)?;
        let actual = cache.read(&key)?.unwrap();
        assert_eq!(actual.codes, codes);
        assert_eq!(actual.scales, scales);
        let changed = cache.key("weight", 256, 256, 4, b"changed source");
        assert!(cache.read(&changed)?.is_none());
        let mut other = PackedWeightCache::new(directory.path().into(), "compiler-b", "gfx1151");
        assert!(
            other
                .read(&other.key("weight", 256, 256, 4, b"source"))?
                .is_none()
        );
        let path = cache.path(&key)?;
        let mut bytes = std::fs::read(&path)?;
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(&path, &bytes)?;
        assert!(
            cache
                .read(&key)
                .unwrap_err()
                .to_string()
                .contains("checksum")
        );
        std::fs::write(&path, &bytes[..bytes.len() - 1])?;
        assert!(cache.read(&key).unwrap_err().to_string().contains("length"));
        Ok(())
    }
}
