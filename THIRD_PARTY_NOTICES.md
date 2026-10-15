# Third-party sources

- Cloudflare CLEF release (`2f3de3dd85f379784083b0814d997ab627200f0c`), Apache-2.0.
  Configurations, tensor metadata, and shard index under `reference/` come from
  https://huggingface.co/Cloudflare/clef/tree/2f3de3dd85f379784083b0814d997ab627200f0c.
- Cloudflare CLEF Flash release (`8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3`), Apache-2.0.
  Configurations, tensor metadata and shard index under `reference/clef-flash/`
  come from https://huggingface.co/Cloudflare/clef-flash/tree/8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3.
- Cloudflare CLEF Omni release (`0db1cd2607d76a7bdb2a382f659e7b313079f84b`), Apache-2.0.
  Configurations, tensor metadata and shard index under `reference/clef-omni/`
  come from https://huggingface.co/Cloudflare/clef-omni/tree/0db1cd2607d76a7bdb2a382f659e7b313079f84b.
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
- `kernels/omni_expert_bf16.loom` adapts the WMMA fragments and output publishing
  from the tiled BF16 kernel. `kernels/omni_expert_i4.loom` also uses the packed
  INT4 operand encoding from the h3-hrx integer GEMM (Apache-2.0), with routed
  gathers, smaller tiles, and reference-compatible scaling.
  `scripts/omni_expert_k128_candidate.py` derives its smaller register packets
  and M32/N64 experimental tile from that kernel, retaining those attributions.
  `scripts/omni_expert_row_loop_candidate.py` changes its row scheduling while
  retaining the same arithmetic and attribution.
  `scripts/omni_expert_workgroup_candidate.py` regroups its independent waves,
  retaining the same arithmetic and source attribution.
- `kernels/transpose_v.loom` adapts h3-hrx's `transpose_qkv_v_f16.loom`
  at `95ed12f43bd3e596cbfad175c22de9560447e9e5` (Apache-2.0), changing its
  input layout to contiguous BF16 values and retaining its padded LDS transpose.
- `scripts/omni_attention_transposed_candidate.py` adapts the transposed WMMA,
  scalar softmax statistics and packed lane exchange from h3-hrx's
  `attention_mha64t32_lds_f16_wmma.loom` at the same revision (Apache-2.0).
  Its optional lane controls and paired accumulators also adapt
  `attention_i8qkhm_mha8_k64_lds_f16_wmma.loom` at that revision, keeping BF16
  operands and the original floating-point operation order.
  Generated experimental kernels use 128-wide BF16 heads, grouped-query
  attention, causal masks and CLEF's transposed value layout.
- `scripts/omni_attention_int8_candidate.py` and
  `benches/kernels/omni_prepare_attention_i8.loom` adapt H3's signed INT8 WMMA
  operand layout and per-row quantization/packing from
  `attention_i8qkhm_mha8_k64_lds_f16_wmma.loom` and `prepare_qk_i8hm.loom`
  at the same Apache-2.0 revision. The generator supports inline or separate
  Q/K packing, omits H3's rotation, and keeps BF16 probability/value operands
  and FP32 softmax/accumulation.
- `scripts/omni_attention_mask_candidate.py` adds an interior-tile mask branch
  to the transposed and prefetch sources, retaining their provenance.
  `scripts/omni_attention_qk_candidate.py` reorders their independent QK
  accumulators while preserving each summation order and the same provenance.
- `scripts/omni_attention_probability_candidate.py` reorders operations produced
  by the transposed and prefetch generators below, retaining their provenance.
- `scripts/omni_attention_prefetch_candidate.py` adapts the next-tile loading
  approach from krea2-hrx's `experiments/sage/attention_sage_i8_fast_prefetch.loom`
  at `0fdd677970baf5fb11cd176c625cac93d5a6eff8` (MIT; see
  `licenses/krea2-hrx-MIT.txt`). It keeps one LDS
  tile and both barriers, uses register-carried loads, and clamps final prefetches
  instead of requiring extra input padding. Its arithmetic comes from the
  transposed and INT8 generators described above.
- `kernels/gemm_integer.loom` adapts h3-hrx's `gemm_packed_256.loom`
  at the same revision (Apache-2.0), retaining its INT4/INT8 exports and changing
  FP16 outputs and saturation bounds to BF16 for the optional integer projection path.
- `kernels/prepare_integer.loom` adapts h3-hrx's
  `prepare_plain_tiled.loom` at that revision (Apache-2.0), accepting BF16 input
  and placing packed codes and FP32 row scales in one output allocation.
- Resize coefficient quantization follows PyTorch's
  `aten/src/ATen/native/cpu/UpSampleKernel.cpp` algorithm. PyTorch is BSD-3-Clause
  (see `licenses/pytorch-BSD.txt`); the bicubic filter and resampling algorithm
  also acknowledge Pillow's work.

The project license is Apache-2.0. Runtime/compiler dependencies retain their
own licenses, as documented in hrx-rs/THIRD-PARTY.md.
