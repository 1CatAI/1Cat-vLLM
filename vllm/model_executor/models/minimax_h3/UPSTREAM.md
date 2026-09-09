# Native MiniMax H3 provenance

The numerical model and media helpers are adapted from Apache-2.0 licensed
vLLM-Omni commit `7be014bce6374f06c95b703763bdbac4c6198f31`.
Comfy checkpoint handling is informed by vLLM-Omni PR #6894 at
`be89929001563a882f1f783fb9ee37b4d5a970f4` (not merged at port time).
Original copyright and SPDX notices remain on adapted files.

This package is part of 1Cat-vLLM. It does not import or install vllm-omni.
The initial scope excluded LoRA. Native LightX2V Turbo support now follows the
artifact contracts in Omni `b58ff5cb8b17250b76f9cdf9b9b46385cdda4376`
(`diffusion/models/minimax_h3/lora.py`, unchanged from the initial port's
reference). Native execution uses registered, staged A/B buffers and H3 SM70
GEMMs instead of Omni's generic LoRA manager. FlashGen's native grouped-QKV
layout and metadata schedule follow `diffusion/models/minimax_h3/npu/lora.py`
at the initial pinned Omni revision. The export is device-independent; native
execution uses the same SM70 staged delta path. The pruned INT8 integration
restores original dense AdaLN/time tensors, preserving backbone INT8 data.
FastH3 Dense mapping and fusion follow `diffusion/models/minimax_h3/fasth3.py`
at the initial pinned Omni revision. Original weights are fused before native
TP loading and staging; native INT8 fusion and VSA remain unimplemented.
Approximate diffusion caches, step batching, Ulysses/Ring parallelism and other
model families remain outside this implementation. Model weights and checkpoint
remote code retain their respective upstream terms; they are not vendored here.

Frozen model revisions:

- MiniMaxAI/MiniMax-H3: `42ed227ee7df40d41602854ae760620d6eb651fe`.
- Comfy-Org/MiniMax-H3: `a98869194787969724c7425d95d0ed73ce9202af`.
- lightx2v/Minimax-h3-Turbo: `2f015e66b37c585cea9dc4ae6f1850ea8788e742`.
- ModelScope FlashGen/Minimax-H3-4step-lora-flashgen, file revision:
  `11ccc3e67cbe0a4b83ae0b6d95a55a5eb27008ee`.
  File: `minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors`;
  size 1,258,539,599 bytes;
  SHA256 `0e17fff71d76db497707328a49d61dd7bcd0a375beabef5ae7e93234d15dff00`.
- FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA:
  `f509e629374cac104e7f62daecce6d1488a3041d`.
  File: `dense-datafree/adapter_model.safetensors`, 1,485,626,152 bytes;
  verified SHA256 `4ce198c83132251b7fd0de2503823aa49c53983f068318f66cb19eaefb7fcc12`.
  Its declared base revision `9bfb6693f2cf6de171db46d1aa586f67d773a1da`
  has the same blob/LFS hashes for all 81 FL2VA files as the native pinned base.
