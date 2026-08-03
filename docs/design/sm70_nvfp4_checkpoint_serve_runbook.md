# SM70 NVFP4 checkpoint scale + serve runbook

Date: 2026-08-03  
Related: PR scale normalize (`normalize_nvfp4_global_scale_for_sm70`),
`docs/design/sm70_nvfp4_turbomind_operator_optimization.md`

## Problem summary

### 1) Scale convention (Python fix in this PR)

Some GPTQ / Magpie NVFP4 exports (Medium, ThinkingCap, cyber) store **tiny**
FP8 block scales with a large disk global. CT turns global into `1/G`;
TurboMind multiplies `block * global` and under-scales → token salad.

**Fix:** detect tiny block scales and set `weight_global_scale = 1.0`
(leave block scales untouched).

### 2) CUDA-graph garbage with identical wheel (serve flags)

Third-party V100 hosts with the **same** public 1.2.2 `_C.abi3.so` have seen
coherent output under `--enforce-eager` but salad under default graphs.

**Fix (preferred over permanent eager):** pin graph capture mode and greedy
MTP draft sampling:

```bash
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1,2,4,8]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"attention_backend":"FLASH_ATTN_V100","draft_sample_method":"greedy"}'
```

Why these matter on 1.2.2:

- **`full_and_piecewise` + explicit capture sizes** — avoids the
  “MTP accepting but not amortized / wrong graph partition” failure mode
  (piecewise-only or missing size-1 graphs). See deckard tok/s runbook notes.
- **`draft_sample_method: greedy`** — 1.2.2 auto-promotes MTP draft to
  probabilistic, which requires `draft_probs` only the dynamic-vocab path
  emits. With dynamic draft vocab off (or multi-seq), non-greedy draft can
  corrupt or crash; greedy is the safe production pin (same as
  voice-coexist-mtp122).

Also set `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0` when
`max_num_seqs != 1`.

## Wheel identity

Public **1Cat-vLLM 1.2.2** release:

| Artifact | sha256 |
| --- | --- |
| wheel | `8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6` |
| `_C.abi3.so` | `ffc6271aaf25d96fe690d0f899544801cc1e39486905c49ad0090cb2cbe7a147` |

https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.2.2  
Tag commit: `644d8a7cd05ed4ecd1cd188e3c05b4bbd074f504`

Scale normalize is **Python-only**. Matching `_C` + still garbage → serve
flags / overlay, not a second wheel.

```bash
python3 scripts/sm70_nvfp4_verify_identity.py
```

## Recommended serve (wrapper)

```bash
export VLLM_SM70_NVFP4_TURBOMIND=1
bash scripts/sm70_nvfp4_campaign_serve.sh /path/to/nvfp4-ckpt my-nvfp4
```

Full argv (graph-safe):

```bash
export VLLM_SM70_FLASH_ATTN_V100=1
export VLLM_SM70_QUANT_BACKEND=turbomind
export VLLM_SM70_NVFP4_TURBOMIND=1
export VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/nvfp4-ckpt \
  --served-model-name my-nvfp4 \
  --trust-remote-code --dtype half \
  --attention-backend FLASH_ATTN_V100 \
  --host 0.0.0.0 --port 8003 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8_e5m2 \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1,2,4,8]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"attention_backend":"FLASH_ATTN_V100","draft_sample_method":"greedy"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
```

Smoke:

```bash
python3 scripts/sm70_nvfp4_smoke_paris.py --model my-nvfp4
# expects Paris + 17*19 → 323
```

### Measured (pve3, 2×V100, 1.2.2 + this PR)

| Model | smoke | c1 tok/s |
| --- | --- | ---: |
| tc-medium-nvfp4 | Paris + 323 | ~66 |
| tc-aggressive-nvfp4 | Paris + 323 | ~70 |

## Graph vs eager bisect

```bash
bash scripts/sm70_nvfp4_ab_graph_eager.sh /path/to/ckpt served-name
```

| Result | Action |
| --- | --- |
| graph PASS (with flags above) | Ship that serve line |
| graph FAIL, eager PASS | Missing compilation/spec pins — add flags above before permanent eager |
| both FAIL | Overlay / TINY-scale normalize / env |

## Checklist for third-party V100 hosts (JJ)

1. Public `1cat_vllm==1.2.2` — `_C` sha matches above.
2. This PR’s scale normalize (or release that includes it).
3. Serve with **compilation-config + greedy MTP draft** (not default graphs alone).
4. `VLLM_SM70_NVFP4_TURBOMIND=1`.
5. Smoke Paris + 323 before long benches.
