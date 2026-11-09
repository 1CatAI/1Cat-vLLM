#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Flash-Next NVFP4 / MTP4 release recipe, 4x V100-SXM2-32GB.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: serve_flash_next_nvfp4_v100.sh MODEL [--draft LOCAL_PATH] [vllm serve options...]

MODEL is a local checkpoint directory or a Hugging Face model ID.
Defaults: TP4, FP16 compute/KV, MTP4, 256K context with FP8 KV history in
pinned host memory, 4096-token prefill, up to 8 sequences, prefix caching and
GPU memory utilization 0.89. Without --draft the MTP layers inside MODEL are
used; pass --draft for a separate MTP checkpoint directory.
Requires four peer-connected V100-SXM2 32GB GPUs, 64 GB or more host RAM,
and a standard CUDA 12.8 Toolkit on PATH for remaining kernel JIT.
Additional CLI options override these defaults.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
if [[ $# -eq 0 || $1 == -* ]]; then
  usage >&2
  exit 2
fi
model=$1
shift
draft=""
serve_options=()
while [[ $# -gt 0 ]]; do
  if [[ $1 == --draft ]]; then
    if [[ $# -lt 2 || $2 == -* ]]; then
      usage >&2
      exit 2
    fi
    draft=$2
    shift 2
  else
    serve_options+=("$1")
    shift
  fi
done
speculative='{"method": "mtp", "num_speculative_tokens": 4, "draft_sample_method": "greedy"}'
if [[ -n $draft ]]; then
  draft_json=$(printf '%s' "$draft" | sed 's/\\/\\\\/g; s/"/\\"/g')
  speculative="{\"method\": \"mtp\", \"model\": \"$draft_json\", \"num_speculative_tokens\": 4, \"draft_sample_method\": \"greedy\"}"
fi
kernel_config='{"sm70_hcx": true, "sm70_qsa_prep": true, "sm70_hcx_local_schedule": true, "qsa_host_kv": true, "qsa_host_kv_dtype": "fp8_e4m3", "qsa_host_kv_hot_dtype": "fp8_e4m3", "qsa_host_kv_hot_tokens": 32768, "qsa_host_kv_prefill_stage_tokens": 262144}'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
vllm_cli=vllm
if [[ -x "$script_dir/vllm" ]]; then
  vllm_cli="$script_dir/vllm"
fi
vllm_cli=$(command -v "$vllm_cli")
exec "$vllm_cli" serve "$model" \
  --tensor-parallel-size 4 --dtype half --kv-cache-dtype float16 \
  --language-model-only --max-model-len 262144 \
  --max-num-batched-tokens 4096 --max-num-seqs 8 \
  --gpu-memory-utilization 0.89 --attention-backend FLASH_ATTN_V100 \
  --enable-prefix-caching --partial-prefix-cache \
  --block-size 2048 --mamba-block-size 2048 --mixed-prefill-step-latency-ms 0 \
  --prefix-cache-retention-interval 8192 \
  --mamba-ssm-cache-dtype float32 --enable-prompt-tokens-details \
  --kernel-config "$kernel_config" \
  --speculative-config "$speculative" "${serve_options[@]}"
