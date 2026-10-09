#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Flash-Next GGUF (Qwen4Exp IQ3_S + MTP4) OpenAI-compatible server on 4x V100.
#
# Usage:
#   MODEL=/path/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
#   DRAFT=/path/mtp ./serve.sh
#
# Optional environment:
#   PORT (8000)  MAX_MODEL_LEN (32768)  MAX_NUM_SEQS (4)  PREFILL_CHUNK (2048)
#   KV_PLACEMENT (host|device, default host)  CACHE_DIR (~/.cache/onecat-flashnext)
#   GPUS (0,1,2,3)  LOCK (1: follow the shared /tmp GPU lock protocol)
#   HOT_TOKENS (MAX_MODEL_LEN)  HCX_LOCAL_SCHEDULE (0; 1 needs a build with KernelConfig.sm70_hcx_local_schedule)
set -euo pipefail

: "${MODEL:?set MODEL to the first GGUF shard}"
: "${DRAFT:?set DRAFT to the MTP draft (safetensors) directory}"
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
KV_PLACEMENT=${KV_PLACEMENT:-host}
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/onecat-flashnext}
GPUS=${GPUS:-0,1,2,3}
LOCK=${LOCK:-1}
HCX_LOCAL_SCHEDULE=${HCX_LOCAL_SCHEDULE:-0}
# Device hot cache per QSA layer; keep >= the longest prompt (1 KiB/token/layer).
HOT_TOKENS=${HOT_TOKENS:-$MAX_MODEL_LEN}
LOCAL_SCHEDULE_FIELD=""
if [ "$HCX_LOCAL_SCHEDULE" = 1 ]; then
  LOCAL_SCHEDULE_FIELD='"sm70_hcx_local_schedule": true,'
fi

case "$KV_PLACEMENT" in
  host) DEVICE_REFERENCE=false ;;
  device) DEVICE_REFERENCE=true ;;
  *) echo "KV_PLACEMENT must be host or device" >&2; exit 2 ;;
esac

mkdir -p "$CACHE_DIR"
export CUDA_VISIBLE_DEVICES=$GPUS CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Persist compile, Triton and TurboMind GEMM plans across restarts.
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$CACHE_DIR/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$CACHE_DIR/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$CACHE_DIR/inductor}
export VLLM_SM70_GEMM_LUT_PATH=${VLLM_SM70_GEMM_LUT_PATH:-$CACHE_DIR/gemm-lut-{device}.bin}

# Target and draft QSA history: FP16 (bit-identical to device FP16 KV).
# KV_PLACEMENT=host keeps it in pinned host memory with a HOT_TOKENS device
# hot cache per layer; KV_PLACEMENT=device keeps the history on the GPU
# (fastest, uses more device memory). PLE n-gram tables stay on disk
# (file-backed, the SM70 Qwen3.8 default).
KERNEL_CONFIG=$(cat <<EOF
{"sm70_gguf": {"small_m_dp4a": true, "q8_expert_intermediate": true,
               "small_m_hmma": true, "lut4_expert_dp4a": true,
               "device_transcode": true},
 "sm70_hcx": true, "sm70_hcx_output_projection": false, $LOCAL_SCHEDULE_FIELD
 "sm70_qsa_shared_key": true, "sm70_qsa_device_history": true,
 "qsa_host_kv": true, "qsa_host_kv_dtype": "float16",
 "qsa_host_kv_draft_dtype": "float16",
 "qsa_host_kv_device_reference": $DEVICE_REFERENCE,
 "qsa_host_kv_hot_tokens": $HOT_TOKENS}
EOF
)
SPEC_CONFIG=$(cat <<EOF
{"method": "mtp", "model": "$DRAFT", "num_speculative_tokens": 4,
 "draft_load_config": {"load_format": "safetensors"},
 "draft_sample_method": "greedy"}
EOF
)

cmd=(vllm serve "$MODEL"
  --served-model-name flash-next
  --port "$PORT"
  --quantization gguf
  --tensor-parallel-size 4
  --dtype half
  --kv-cache-dtype float16
  --mamba-ssm-cache-dtype float32
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$PREFILL_CHUNK"
  --gpu-memory-utilization 0.95
  --no-enable-prefix-caching
  --language-model-only
  --compilation-config '{"mode": 3, "cudagraph_mode": "FULL"}'
  --kernel-config "$KERNEL_CONFIG"
  --speculative-config "$SPEC_CONFIG")

if [ "$LOCK" = 1 ]; then
  # Shared-host etiquette: hold the four-GPU and per-GPU locks while serving.
  exec 9>/tmp/gpu0-3.lock
  flock -E 75 -w 3600 9
  exec 8>/tmp/1cat-vllm-v100-gpus0123.lock
  flock -E 75 -w 600 8
  for g in ${GPUS//,/ }; do
    for l in /tmp/gpu$g.lock /tmp/1cat-vllm-v100-gpu$g.lock; do
      exec {fd}>"$l"
      flock -E 75 -w 600 "$fd"
    done
  done
fi
exec "${cmd[@]}"
