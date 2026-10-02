#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Release profile: Qwen3.8-27B NVFP4 + DFlash2, 4x V100-SXM2-32GB.
# Installed beside vllm by the wheel, so parameters follow the package version.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: serve_qwen38_27b_nvfp4_v100.sh MODEL [--draft LOCAL_PATH] [vllm serve options...]

MODEL is a local checkpoint directory or a Hugging Face model ID.
Uses the installed vllm, bundled kernels and automatic SM70 operator defaults.
The pinned DFlash2 checkpoint is downloaded normally; no offline mode is forced.

Profile: FP16, TP4, E4M3 KV, 256K context, 8192-token prefill, up to 4 sequences.
Requires four peer-connected V100-SXM2 32GB GPUs. Other hardware/capacities
need their own memory and performance validation. Additional CLI options
override the profile, for example:

  serve_qwen38_27b_nvfp4_v100.sh /models/Qwen3.8-27B-NVFP4 --port 8001
  serve_qwen38_27b_nvfp4_v100.sh /models/Qwen3.8-27B-NVFP4 --max-model-len 32768

To use a downloaded draft, pass --draft /models/Qwen3.8-27B-DFlash2.
Without --draft, the release default downloads the pinned DFlash2 revision.
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
PROFILE_NAME=qwen38_27b_nvfp4_dflash2
profile_options=()
serve_options=()
while [[ $# -gt 0 ]]; do
  if [[ $1 == --draft ]]; then
    if [[ $# -lt 2 || $2 == -* ]]; then
      usage >&2
      exit 2
    fi
    profile_options+=(--draft "$2")
    shift 2
  else
    serve_options+=("$1")
    shift
  fi
done

# Prefer the CLI installed alongside this script, including when the caller
# invokes it by absolute path without activating that Python environment.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
vllm_cli=vllm
if [[ -x "$script_dir/vllm" ]]; then
  vllm_cli="$script_dir/vllm"
fi

vllm_cli=$(command -v "$vllm_cli")
python_bin="$(dirname "$vllm_cli")/python"
profile_output=$("$python_bin" -m vllm.sm70_profiles argv "$PROFILE_NAME" \
  --argv-lines "${profile_options[@]}")
mapfile -t profile_args <<< "$profile_output"
exec "$vllm_cli" serve "$model" "${profile_args[@]}" "${serve_options[@]}"
