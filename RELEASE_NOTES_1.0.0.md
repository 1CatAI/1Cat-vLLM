# 1Cat-vLLM 1.0.0 Release Notes

This release focuses on the V100/SM70 serving path for Qwen3.5/Qwen3.6-class models, long-context stability, and reproducible local wheel deployment.

## Highlights

- Added FP8 KV cache support for the V100 FlashAttention path, including regression coverage for operator correctness and model-level generation behavior.
- Fixed model output quality regressions observed in long-context and MTP serving, including repeated output, abnormal punctuation-only output, and unstable NaN-like decode behavior.
- Fixed OpenAI-compatible tool calling and chat serving behavior, improving compatibility with Cherry Studio, OpenClaw, OpenCode, and OpenAI-style clients.
- Improved runtime stability for Qwen3.5/Qwen3.6 model families, including Qwen3.6-27B-AWQ serving with 256K context.
- Reduced unnecessary startup memory pressure and tightened V100 memory defaults for more predictable deployment.
- Added MTP speculative decoding support and serving flags, with regression tooling for acceptance length, quality, and throughput.
- Introduced DFlash as an experimental speculative decoding path. It is included for validation and continued tuning, not as the default production path.
- Improved FlashAttention V100 dense prefill and paged decode paths, with stricter operator-level quality and speed regression scripts.
- Added benchmark and audit utilities for FA2/Triton comparison, prefix-cache plus MTP serving, Qwen3.6 output quality checks, and OpenAI API compatibility testing.

## Recommended Runtime Baseline

- GPU: NVIDIA V100 / SM70
- CUDA: 12.8
- Python: 3.12
- PyTorch: 2.9.1 + cu128
- Default context target: 256K
- Recommended deployment path: install the local `flash_attn_v100` wheel together with the local `vllm` wheel.

## Packaging

The local release wheel directory for this build is expected to be:

```bash
../dist-cu128-sm70-1.0.0
```

Install both wheels together:

```bash
python -m pip install \
  ../dist-cu128-sm70-1.0.0/flash_attn_v100-*.whl \
  ../dist-cu128-sm70-1.0.0/vllm-*.whl
```

## Notes

- DFlash remains experimental and should be benchmarked against the normal MTP path before production use.
- The V100 FP8 work targets FP8 KV/cache and V100-specific fast paths. It is not Hopper-style native FP8 Tensor Core W8A8 inference.
- Generated benchmark outputs, server logs, model weights, and local cache artifacts are not part of the source release.
