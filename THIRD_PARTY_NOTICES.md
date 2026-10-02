# Third-party sources

- Cloudflare CLEF release (`2f3de3dd85f379784083b0814d997ab627200f0c`), Apache-2.0.
  Configurations, tensor metadata, and shard index under `reference/` come from
  https://huggingface.co/Cloudflare/clef/tree/2f3de3dd85f379784083b0814d997ab627200f0c.
- The architecture and processor behavior follow Hugging Face Transformers
  v5.10.2 (Apache-2.0), https://github.com/huggingface/transformers/tree/v5.10.2/src/transformers.
  Captured numerical fixtures record the reference versions used; no Python
  implementation is distributed or required.
- `kernels/gemm_bf16.loom` is adapted from Zachary Denton's `qwen-image-hrx`
  `gemm_bf16_bf16_nt.loom` (MIT); see `licenses/qwen-image-hrx-MIT.txt`.
- Resize coefficient quantization follows PyTorch's
  `aten/src/ATen/native/cpu/UpSampleKernel.cpp` algorithm. PyTorch is BSD-3-Clause
  (see `licenses/pytorch-BSD.txt`); the bicubic filter and resampling algorithm
  also acknowledge Pillow's work.

The project license is Apache-2.0. Runtime/compiler dependencies retain their
own licenses, as documented in hrx-rs/THIRD-PARTY.md.
