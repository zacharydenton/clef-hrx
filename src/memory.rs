//! System memory checks complement HRX's per-context residency budget.
use crate::Result;
use anyhow::ensure;

pub(crate) fn available_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines().find_map(|line| {
        line.strip_prefix("MemAvailable:")?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()?
            .checked_mul(1024)
    })
}

pub(crate) fn require(bytes: u64) -> Result<()> {
    if let Some(available) = available_bytes() {
        ensure!(
            available > bytes,
            "insufficient available RAM: need approximately {:.1} GiB, available {:.1} GiB",
            bytes as f64 / (1u64 << 30) as f64,
            available as f64 / (1u64 << 30) as f64
        );
    }
    Ok(())
}

pub(crate) fn before_allocation(bytes: usize) -> Result<()> {
    if bytes >= 16 << 20 {
        // Recheck during loading: other applications may consume RAM after
        // the initial preflight. Preserve headroom for the OS and staging.
        require(bytes as u64 + (1 << 30))?;
    }
    Ok(())
}
