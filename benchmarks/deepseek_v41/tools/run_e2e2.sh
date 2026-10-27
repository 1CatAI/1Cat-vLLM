#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Usage: run_e2e2.sh <name> [v41_e2e.py args]; inside ds41-compute. Cleans up workers.
set -uo pipefail
dev_root=/home/ymzx/ds41-dev
state=$dev_root/state
name=${1:?name}; shift
cd "$dev_root/src"
export PYTHONPATH="$dev_root/src:$dev_root/tools"
export XDG_CACHE_HOME="$dev_root/cache" XDG_CONFIG_HOME="$dev_root/cache/config"
export FLASHINFER_WORKSPACE_BASE="$dev_root/cache" TRITON_CACHE_DIR="$dev_root/cache/triton"
export VLLM_NO_USAGE_STATS=1
cleanup() { status=$?; pkill -9 -f "VLLM::Worker" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null; printf "%s\n" "$status" > "$state/$name.exit"; }
trap cleanup EXIT
pkill -9 -f "VLLM::Worker" 2>/dev/null; sleep 2
[[ -z $(nvidia-smi --id=0,1,2,3 --query-compute-apps=pid --format=csv,noheader) ]] || { echo "GPU busy" > "$state/$name.log"; exit 3; }
.venv/bin/python "$dev_root/tools/v41_e2e.py" --label "$name" --output "$state/$name.json" "$@" > "$state/$name.log" 2>&1
