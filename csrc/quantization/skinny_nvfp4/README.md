# SM70 skinny NVFP4 kernels

This directory contains the small-M production subset of the MIT-licensed
[`v100-skinny`](https://github.com/Leonccaa/v100-skinny) CUDA kernels, pinned
to commit `f8194f7c3c9269fa74ee70b5029d53c20098f4c8`.

The extension consumes the checkpoint-native W4A16 NVFP4 layout: packed E2M1
codes `[N, K / 2]`, FP8-E4M3 group-16 scales `[N, K / 16]`, and one
multiplicative global scale. It provides two SM70-only operators:

- SIMT for FP16 `M <= 3`.
- QPN `mma.sync.m8n8k4` for FP16 `4 <= M <= 16`, using an offline
  fragment-order prepack.

The Python adapter owns runtime dispatch. BF16 is converted explicitly to
FP16 at the adapter boundary and restored on output. Larger M, unsupported
shapes, or a failed validation use the existing SM70 TurboMind kernel. The
skinny backend is opt-in through `VLLM_SM70_QUANT_BACKEND=skinny`; it is not a
global `--linear-backend` because mixed-format models still need their normal
per-format backends.
