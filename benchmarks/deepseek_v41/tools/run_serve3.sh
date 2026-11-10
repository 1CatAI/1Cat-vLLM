#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# run_serve.sh <name>; inside ds41-compute. OpenAI API on 0.0.0.0:18000 (container bridge only).
set -uo pipefail
dev_root=${DS41_ROOT:-/home/ymzx/ds41-dev}; state=$dev_root/state; name=${1:-serve}
cd "$dev_root/src"
export PYTHONPATH="$dev_root/src:$dev_root/tools" XDG_CACHE_HOME="$dev_root/cache" XDG_CONFIG_HOME="$dev_root/cache/config"
export FLASHINFER_WORKSPACE_BASE="$dev_root/cache" TRITON_CACHE_DIR="$dev_root/cache/triton" VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_V41_ASYNC=1 OMP_NUM_THREADS=1
export VLLM_V41_DOWN_ROWS=1 VLLM_V41_SESSION_PREFIX=1 VLLM_V41_PACKED_KV=1 VLLM_V41_INDEX_TILE_MB=32
pkill -9 -f "VLLM::Worker" 2>/dev/null; sleep 2
exec .venv/bin/python -m vllm.entrypoints.openai.api_server --config benchmarks/deepseek_v41/serve-ds41-phase3.yaml > "$state/$name.log" 2>&1
