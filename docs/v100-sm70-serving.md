# Serving Qwen3.5/3.6 on V100 (SM70) with 1Cat-vLLM 1.2.1

Field notes for running W4A16 (+ MTP speculative decoding) Qwen3.6-27B-class checkpoints on
2x Tesla V100-PCIE-32GB (compute capability 7.0) with 1Cat-vLLM 1.2.1. As of 1.2.1 the Volta
serving stack is in-release; only the fp8-KV prefill gather (this PR) is an add-on.

## What 1.2.1 already handles (no patch needed)
- **W4A16 on Volta**: `CompressedTensorsWNA16` admits CC 7.0 and binds the SM70 TurboMind kernel.
- **fp8_e5m2 KV on W4A16**: `CompressedTensorsKVCacheMethod.validate_kv_cache_scheme(None)` returns
  cleanly, so `--kv-cache-dtype fp8_e5m2` works on a no-KV-scale W4A16 checkpoint.
- **SM70 rotary**: pure-PyTorch `forward_native` fallback when `_vllm_fa2_C` is unavailable.
- **Reasoning**: `--reasoning-parser qwen3` populates the reasoning field and keeps partial content
  when truncated by `max_tokens`.

## This PR: fast fp8-KV prefill gather
`_extract_contiguous_kv_from_paged_cache` skips the native `paged_kv_to_contiguous` kernel for
`uint8` (the kernel is fp16-typed), so fp8 KV falls to a per-block Python loop -- the dominant
chunked-prefill cost on long prompts for fp8-KV models. The patch views the uint8 bytes as fp16
(a gather is a bitwise copy) to run the fast kernel. Correctness-equivalent to the fallback;
only prefill speed changes. Decode (incl. MTP) is unaffected.

## MTP (speculative decoding) packed-head recipe
To enable `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` on a W4A16 checkpoint:

- The MTP head must be **quantized (packed W4A16)** to match the body. A bf16 head on a W4A16 body
  crashes SM70 spec-decode (mixed-precision GDN spec forward).
- **But `mtp.fc` must stay fp16** and be listed in `quantization_config.ignore` (and excluded from any
  quant `config_group` target). If `mtp.fc` is marked for quantization while stored plain fp16, the
  loader logs `Parameter fc.weight not found in params_dict, skip loading`, the drafter's fusion
  projection is left uninitialized, and **MTP accept rate is 0%** -- silently, since the target model
  still emits coherent text. With `mtp.fc` in `ignore`, accept is ~76-80%.
- This matches the base Qwen3.6 packed-head layout: 36 mtp tensors, each `mtp.layers.0.*_proj` with
  `weight_packed/weight_scale/weight_shape/weight_zero_point`; `mtp.fc` + norms fp16.

## Recommended serving config (measured, 2x V100-PCIE-32GB)
```
VLLM_SM70_QUANT_BACKEND=turbomind \
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
  --model <qwen3.6-27b W4A16 + packed MTP head> \
  --trust-remote-code --dtype half --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.58 \
  --max-model-len 262144 --max-num-seqs 6 --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8_e5m2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"attention_backend":"FLASH_ATTN_V100"}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1,2,4,8]}' \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

- **fp8_e5m2 KV** roughly doubles the KV pool vs fp16, enabling the full 262144 context; this PR keeps
  its prefill fast. (fp16 KV also works and is simpler if you only need ~130k context.)
- **MTP K=2** is the sweet spot for stock Qwen MTP on V100 (~76-80% accept, ~1.5x single-stream decode);
  K=4 lowers accept without a purpose-built head.
- `--attention-backend FLASH_ATTN_V100` and `VLLM_SM70_QUANT_BACKEND=turbomind` are required on Volta.
