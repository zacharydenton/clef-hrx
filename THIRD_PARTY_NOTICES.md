# Third-party sources

- Cloudflare CLEF release (`2f3de3dd85f379784083b0814d997ab627200f0c`), Apache-2.0.
  Configurations, tensor metadata, and shard index under `reference/` come from
  https://huggingface.co/Cloudflare/clef/tree/2f3de3dd85f379784083b0814d997ab627200f0c.
- The architecture and processor behavior follow Hugging Face Transformers
  v5.10.2 (Apache-2.0), https://github.com/huggingface/transformers/tree/v5.10.2/src/transformers.
  Captured numerical fixtures record the reference versions used; no Python
  implementation is distributed or required.
- `kernels/gemm_bf16.loom` and `kernels/gemm_bf16_reference.loom` are adapted from
  Zachary Denton's `qwen-image-hrx` `gemm_bf16_bf16_nt.loom` (MIT); see
  `licenses/qwen-image-hrx-MIT.txt`.
- `kernels/gemm_bf16_pipelined.loom` comes from qwen-image-hrx's
  `gemm_bf16_fast.loom`, itself adapted from h3-hrx's `gemm_f16_fast_256b.loom`
  (MIT). Its source header is retained; see `licenses/qwen-image-hrx-MIT.txt`.
- `kernels/transpose_v.loom` adapts h3-hrx's `transpose_qkv_v_f16.loom`
  at `95ed12f43bd3e596cbfad175c22de9560447e9e5` (Apache-2.0), changing its
  input layout to contiguous BF16 values and retaining its padded LDS transpose.
- `benches/kernels/gemm_integer.loom` adapts h3-hrx's `gemm_packed_256.loom`
  at the same revision (Apache-2.0), retaining its INT4/INT8 exports and changing
  FP16 outputs and saturation bounds to BF16 for benchmark experiments.
- Resize coefficient quantization follows PyTorch's
  `aten/src/ATen/native/cpu/UpSampleKernel.cpp` algorithm. PyTorch is BSD-3-Clause
  (see `licenses/pytorch-BSD.txt`); the bicubic filter and resampling algorithm
  also acknowledge Pillow's work.

The project license is Apache-2.0. Runtime/compiler dependencies retain their
own licenses, as documented in hrx-rs/THIRD-PARTY.md.
