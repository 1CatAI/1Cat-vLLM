# Serving Qwen3.5/3.6 on V100 (SM70) with 1Cat-vLLM 1.2.1

Field notes for running W4A16 (+ MTP speculative decoding, + `fp8_e5m2` KV) Qwen3.6-27B-class
checkpoints on 2× Tesla V100-PCIE-32GB (compute capability 7.0) with 1Cat-vLLM 1.2.1. As of 1.2.1
most of the Volta stack is in-release; this PR adds the **two** remaining pieces needed to run the
full W4A16 + `fp8_e5m2` + MTP config.

## What 1.2.1 already handles (no patch)
- **W4A16 on Volta**: `CompressedTensorsWNA16` admits CC 7.0 and binds the SM70 TurboMind kernel.
- **SM70 rotary**: pure-PyTorch `forward_native` fallback when `_vllm_fa2_C` is unavailable (vision/MM).
- **Reasoning**: `--reasoning-parser qwen3` populates the reasoning field and keeps partial content
  when truncated by `max_tokens`.
- **An SM70 `fp8_e5m2` unit-scale override path exists** (`attention.py`, `_force_unit_fp8_e5m2_kv_scales`)
  — but it is gated on the layer using `BaseKVCacheMethod.process_weights_after_loading`. See P7 below.

## This PR — two patches

### 1. P7: `fp8_e5m2` KV on compressed-tensors W4A16 (SM70)
A W4A16 checkpoint ships **no KV scales** (`kv_cache_scheme=None`), yet `get_quant_method` still returns
`CompressedTensorsKVCacheMethod` for the attention layer. That class **overrides**
`process_weights_after_loading`, so 1.2.1's SM70 unit-scale override (which requires the *base* method's
`process_weights_after_loading`) does **not** apply, and `attention.py` raises:

```
ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints
outside the SM70 Flash-V100 unit-scale override path.
```

Fix: when no KV scheme ships, skip the CT KV method entirely so the layer takes the plain
`fp8_e5m2` unit-scale path:

```python
if isinstance(layer, Attention):
    if self.kv_cache_scheme is None:
        return None
    return CompressedTensorsKVCacheMethod(self)
```

Without this, `--kv-cache-dtype fp8_e5m2` cannot be used with a W4A16 checkpoint on V100 — which means
no `fp8_e5m2` KV, which means you cannot fit the full 262144 context (fp16 KV ~halves the pool).

### 2. Fast `fp8`-KV prefill gather
`_extract_contiguous_kv_from_paged_cache` skips the native `paged_kv_to_contiguous` kernel for `uint8`
(the kernel is fp16-typed), so `fp8` KV falls to a per-block Python loop — the dominant chunked-prefill
cost on long prompts for fp8-KV models. A gather is a bitwise copy, so view two `uint8` bytes as one
`float16`, run the fast kernel, view back. Correctness-equivalent to the fallback; decode (incl. MTP)
unaffected. (This pairs with P7: once `fp8_e5m2` KV is usable on W4A16, this keeps its prefill fast.)

## MTP (speculative decoding) packed-head recipe
To enable `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` on a W4A16 checkpoint:

- The MTP head must be **quantized (packed W4A16)** to match the body. A bf16 head on a W4A16 body
  crashes SM70 spec-decode (mixed-precision GDN spec forward).
- **But `mtp.fc` must stay fp16** and be listed in `quantization_config.ignore` (and excluded from any
  quant `config_group` target). If `mtp.fc` is marked for quantization while stored plain fp16, the
  loader logs `Parameter fc.weight not found in params_dict, skip loading`, the drafter's fusion
  projection is left uninitialized, and **MTP accept rate is 0%** — silently, since the target model
  still emits coherent text. With `mtp.fc` in `ignore`, accept is ~76–84%.
- This matches the base Qwen3.6 packed-head layout: 36 mtp tensors, each `mtp.layers.0.*_proj` with
  `weight_packed/weight_scale/weight_shape/weight_zero_point`; `mtp.fc` + norms fp16.

## Recommended serving config (measured, 2× V100-PCIE-32GB, 1.2.1 + this PR)
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

Measured (Qwen3.6-27B-Uncensored W4A16-AWQ g128 + packed MTP head):
- **MTP K=2 accept ~84%**, **~57 tok/s** single-stream decode (≈1.5× vs no-MTP).
- **fp8_e5m2 KV** → **427k-token KV pool** at `util 0.58` (1.63× concurrency at 262144) while coexisting
  with an ASR+TTS stack on the same 2 GPUs. fp16 KV also works (~213k pool, ~192k usable context) and
  needs neither patch if you don't need the full 262k context.
- `num_speculative_tokens=2` is the sweet spot for stock Qwen MTP on V100; K=4 lowers accept.
- `--attention-backend FLASH_ATTN_V100` and `VLLM_SM70_QUANT_BACKEND=turbomind` are required on Volta.
