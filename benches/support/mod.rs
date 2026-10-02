use sha2::{Digest, Sha256};

pub fn fingerprint() -> String {
    let mut hash = Sha256::new();
    for source in [
        include_str!("../../Cargo.lock"),
        include_str!("../../src/benchmarking.rs"),
        include_str!("../kernels.rs"),
        include_str!("../criteria.json"),
        include_str!("../../src/chunk.rs"),
        include_str!("../../src/gpu.rs"),
        include_str!("../../src/ops.rs"),
        include_str!("../../src/backbone.rs"),
        include_str!("../../kernels/delta_norm.loom"),
        include_str!("../../kernels/common.loom"),
        include_str!("../../kernels/bmm_f32.loom"),
        include_str!("../../kernels/bmm_f32_reference.loom"),
        include_str!("../../kernels/triangular_inverse.loom"),
        include_str!("../../kernels/triangular_inverse_reference.loom"),
        include_str!("../../kernels/chunk_prepare.loom"),
        include_str!("../../kernels/chunk_prepare_reference.loom"),
        include_str!("../../kernels/chunk_pair.loom"),
        include_str!("../../kernels/chunk_state.loom"),
        include_str!("../../kernels/chunk_reorder.loom"),
        include_str!("../../kernels/delta_recurrent.loom"),
        include_str!("../../kernels/columns.loom"),
        include_str!("../../kernels/cast.loom"),
        include_str!("../../kernels/addf.loom"),
        include_str!("../../kernels/subf.loom"),
        include_str!("../../kernels/mulf.loom"),
        include_str!("../../kernels/norm.loom"),
        include_str!("../../kernels/silu.loom"),
        include_str!("../../kernels/scale_rows_exp.loom"),
        include_str!("../../kernels/zero.loom"),
    ] {
        hash.update(source.len().to_le_bytes());
        hash.update(source.as_bytes());
    }
    format!("{:x}", hash.finalize())
}
