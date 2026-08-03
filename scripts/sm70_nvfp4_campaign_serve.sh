#!/usr/bin/env bash
# SM70 NVFP4 campaign serve (graph-safe + coherent on TC/Medium TINY-scale quants).
#
# Critical flags that fix "garbage under CUDA graphs / OK in eager" on 1.2.2:
#   --compilation-config full_and_piecewise + capture sizes [1,2,4,8]
#   --speculative-config ... draft_sample_method=greedy  (with MTP k=4)
#
# Also requires: VLLM_SM70_NVFP4_TURBOMIND=1 and scale-normalize (this PR).
#
# Usage: sm70_nvfp4_campaign_serve.sh MODEL_DIR SERVED_NAME [--eager]
#   --eager  adds --enforce-eager (bisect only; not the preferred fix)
set -euo pipefail
DIR="${1:?model dir}"
NAME="${2:?served name}"
EAGER=0
if [ "${3:-}" = "--eager" ] || [ "${ENFORCE_EAGER:-0}" = "1" ]; then
  EAGER=1
fi

if [ -x /opt/1cat-release-122/.venv/bin/python3 ]; then
  PY=/opt/1cat-release-122/.venv/bin/python3
  export PATH=/opt/1cat-release-122/.venv/bin:$PATH
else
  PY="${VLLM_PYTHON:-python3}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_SM70_FLASH_ATTN_V100=1
export VLLM_SM70_QUANT_BACKEND=turbomind
export VLLM_SM70_NVFP4_TURBOMIND=1
# When max_num_seqs>1, dynamic draft vocab must stay off (1.2.2).
export VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT="${VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT:-0}"

MTP_K="${MTP_K:-4}"
SPEC="{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_K}, \"attention_backend\": \"FLASH_ATTN_V100\", \"draft_sample_method\": \"greedy\"}"
COMP='{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1,2,4,8]}'

ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "$DIR"
  --served-model-name "$NAME" qwen-27b deckard-40b
  --trust-remote-code
  --dtype half
  --attention-backend FLASH_ATTN_V100
  --host 0.0.0.0 --port "${PORT:-8003}"
  --tensor-parallel-size "${TP:-2}"
  --gpu-memory-utilization "${UTIL:-0.85}"
  --max-model-len "${MAX_LEN:-32768}"
  --max-num-seqs "${MAX_SEQS:-1}"
  --max-num-batched-tokens 8192
  --kv-cache-dtype fp8_e5m2
  --speculative-config "$SPEC"
  --compilation-config "$COMP"
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --default-chat-template-kwargs '{"enable_thinking": true, "preserve_thinking": true}'
)
if [ "$EAGER" = "1" ]; then
  ARGS+=(--enforce-eager)
  echo "[serve] ENFORCE_EAGER=1 (no CUDA graphs) — bisect only"
else
  echo "[serve] CUDA graphs: full_and_piecewise sizes=[1,2,4,8]; MTP k=${MTP_K} draft=greedy"
fi

exec "$PY" "${ARGS[@]}"
