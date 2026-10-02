#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Flash-Next NVFP4 / MTP4 release recipe, 4x V100-SXM2-32GB.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: serve_flash_next_nvfp4_v100.sh MODEL [vllm serve options...]

MODEL is a local checkpoint directory or a Hugging Face model ID.
Defaults: TP4, FP16 compute/KV, MTP4, prefix caching, 128K context,
8192-token prefill, one sequence, GPU memory utilization 0.90.
FP16 acceleration and hybrid PLE host placement are selected automatically.
Requires four peer-connected V100-SXM2 32GB GPUs, sufficient host RAM,
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
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
vllm_cli=vllm
if [[ -x "$script_dir/vllm" ]]; then
  vllm_cli="$script_dir/vllm"
fi
vllm_cli=$(command -v "$vllm_cli")
exec "$vllm_cli" serve "$model" \
  --tensor-parallel-size 4 --dtype half --kv-cache-dtype auto \
  --language-model-only --max-model-len 131072 \
  --max-num-batched-tokens 8192 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --no-async-scheduling \
  --mamba-cache-mode align --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' "$@"
