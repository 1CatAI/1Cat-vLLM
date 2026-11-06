#!/bin/bash
# Serves a release model on 4x V100-32GB with the same arguments as its Studio preset.
#   MODEL=27b        Qwen3.8-27B NVFP4 (Unsloth) + DFlash2
#   MODEL=flashnext  Qwen3.8-Flash-Next NVFP4 + MTP4, FP8 KV history in host memory
# Model directories live under MODEL_DIR. EXTRA_ARGS is appended to the command line.
set -euo pipefail

MODEL=${MODEL:-27b}
MODEL_DIR=${MODEL_DIR:-/models}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
CACHE_DIR=${CACHE_DIR:-/cache}

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_DISABLE_PERF_BOOST=1
export VLLM_CACHE_ROOT=$CACHE_DIR/vllm TORCHINDUCTOR_CACHE_DIR=$CACHE_DIR/inductor TRITON_CACHE_DIR=$CACHE_DIR/triton TORCH_EXTENSIONS_DIR=$CACHE_DIR/extensions
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

thinking_on='{"enable_thinking":true}'
thinking_off='{"enable_thinking":false}'

args=(--host "$HOST" --port "$PORT" --tensor-parallel-size 4 --dtype half
  --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens 4096 --max-num-seqs "$MAX_NUM_SEQS"
  --attention-backend FLASH_ATTN_V100 --enable-prefix-caching --partial-prefix-cache
  --block-size 2048 --mamba-block-size 2048 --mixed-prefill-step-latency-ms 0
  --enable-prompt-tokens-details --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3 --trust-remote-code --seed 0)
[ -n "${VLLM_API_KEY:-}" ] && args+=(--api-key "$VLLM_API_KEY")

case "$MODEL" in
  27b)
    target=$MODEL_DIR/Qwen3.8-27B-NVFP4
    draft=$MODEL_DIR/Qwen3.8-27B-DFlash2
    args+=(--served-model-name "${SERVED_MODEL_NAME:-qwen3.8-27b-dflash2}"
      --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.83}"
      --mamba-cache-mode align --prefix-cache-retention-interval 8192
      --limit-mm-per-prompt '{"video": 0, "audio": 0}'
      --default-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS:-$thinking_on}"
      --speculative-config "{\"method\": \"dflash\", \"model\": \"$draft\", \"revision\": \"dedf8df68adfb1afeaf7b7480c0a0243108177b4\", \"num_speculative_tokens\": 7, \"kv_cache_dtype\": \"auto\", \"attention_backend\": \"FLASH_ATTN_V100\", \"draft_sample_method\": \"probabilistic\", \"enforce_eager\": false}")
    ;;
  flashnext)
    target=$MODEL_DIR/Qwen3.8-Flash-Next-NVFP4
    draft=$MODEL_DIR/Qwen3.8-Flash-Next-mtp
    args+=(--served-model-name "${SERVED_MODEL_NAME:-qwen3.8-flash-next-nvfp4}"
      --kv-cache-dtype float16 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.89}"
      --language-model-only --mamba-ssm-cache-dtype float32
      --prefix-cache-retention-interval 8192
      --default-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS:-$thinking_off}"
      --kernel-config '{"sm70_hcx": true, "sm70_qsa_prep": true, "sm70_hcx_local_schedule": true, "qsa_host_kv": true, "qsa_host_kv_dtype": "fp8_e4m3", "qsa_host_kv_hot_dtype": "fp8_e4m3", "qsa_host_kv_hot_tokens": 32768, "qsa_host_kv_prefill_stage_tokens": 262144}'
      --speculative-config "{\"method\": \"mtp\", \"model\": \"$draft\", \"num_speculative_tokens\": 4, \"draft_sample_method\": \"greedy\"}")
    ;;
  *)
    echo "Unknown MODEL=$MODEL (use 27b or flashnext)" >&2
    exit 2
    ;;
esac

for path in "$target" "$draft"; do
  [ -d "$path" ] || { echo "Model directory not found: $path" >&2; exit 3; }
done

# shellcheck disable=SC2086
exec python -m vllm.entrypoints.cli.main serve "$target" "${args[@]}" ${EXTRA_ARGS:-}
