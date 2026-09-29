#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Release profile: Qwen3.8-27B NVFP4 + DFlash2, 4x V100-SXM2-32GB.
# Installed beside vllm by the wheel, so parameters follow the package version.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: serve_qwen38_27b_nvfp4_v100.sh MODEL [vllm serve options...]

MODEL is a local checkpoint directory or a Hugging Face model ID.
Uses the installed vllm, bundled kernels and automatic SM70 operator defaults.
The pinned DFlash2 checkpoint is downloaded normally; no offline mode is forced.

Profile: FP16, TP4, E5M2 KV, 256K context, 8192-token prefill, up to 4 sequences.
Requires four peer-connected V100-SXM2 32GB GPUs. Other hardware/capacities
need their own memory and performance validation. Additional CLI options
override the profile, for example:

  serve_qwen38_27b_nvfp4_v100.sh /models/Qwen3.8-27B-NVFP4 --port 8001
  serve_qwen38_27b_nvfp4_v100.sh /models/Qwen3.8-27B-NVFP4 --max-model-len 32768

To use a local draft, pass a replacement --speculative-config JSON after the
profile arguments. The release default uses the pinned DFlash2 revision.
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

# Prefer the CLI installed alongside this script, including when the caller
# invokes it by absolute path without activating that Python environment.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
vllm_cli=vllm
if [[ -x "$script_dir/vllm" ]]; then
  vllm_cli="$script_dir/vllm"
fi

# Keep the prefill budget and recurrent-state grid at 8192 together: the
# long-prefill specialization admits Q=8000..8192, and the grid must be a
# multiple of the KV block size. Smaller prompts use the normal fallback.
exec "$vllm_cli" serve "$model" \
  --host 127.0.0.1 --port 8000 \
  --served-model-name qwen3.8-27b-dflash2 \
  --trust-remote-code \
  --dtype half --tensor-parallel-size 4 --attention-backend FLASH_ATTN_V100 \
  --kv-cache-dtype fp8_e5m2 --max-model-len 262144 \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 8192 --max-num-seqs 4 \
  --enable-prefix-caching --mamba-cache-mode align \
  --block-size 2048 --mamba-block-size 8192 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --seed 0 \
  --speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","revision":"dedf8df68adfb1afeaf7b7480c0a0243108177b4","num_speculative_tokens":7,"kv_cache_dtype":"auto","attention_backend":"FLASH_ATTN_V100","draft_sample_method":"probabilistic","enforce_eager":false}' \
  "$@"
