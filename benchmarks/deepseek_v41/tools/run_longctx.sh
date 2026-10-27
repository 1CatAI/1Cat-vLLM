#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -uo pipefail
dev_root=/home/ymzx/ds41-dev; state=$dev_root/state; name=$1; shift
cd "$dev_root/src"
export PYTHONPATH="$dev_root/src:$dev_root/tools" XDG_CACHE_HOME="$dev_root/cache" XDG_CONFIG_HOME="$dev_root/cache/config"
export FLASHINFER_WORKSPACE_BASE="$dev_root/cache" TRITON_CACHE_DIR="$dev_root/cache/triton" VLLM_NO_USAGE_STATS=1 VLLM_V41_ASYNC=${VLLM_V41_ASYNC:-1}
cleanup() { status=$?; pkill -9 -f "VLLM::Worker" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null; printf "%s\n" "$status" > "$state/$name.exit"; }
trap cleanup EXIT
.venv/bin/python "$dev_root/tools/longctx_test.py" "$state/$name.json" "$@" > "$state/$name.log" 2>&1
