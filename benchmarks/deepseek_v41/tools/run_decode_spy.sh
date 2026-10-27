#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# run_decode_spy.sh NAME: sync decode loop + py-spy of EngineCore and Worker_TP0 (inside ds41-compute, as root)
set -uo pipefail
dev_root=/home/ymzx/ds41-dev; state=$dev_root/state; name=$1
cd "$dev_root/src"
export PYTHONPATH="$dev_root/src:$dev_root/tools" XDG_CACHE_HOME="$dev_root/cache" XDG_CONFIG_HOME="$dev_root/cache/config"
export FLASHINFER_WORKSPACE_BASE="$dev_root/cache" TRITON_CACHE_DIR="$dev_root/cache/triton" VLLM_NO_USAGE_STATS=1 VLLM_V41_ASYNC=0
cleanup() { status=$?; pkill -9 -f "VLLM::Worker" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null; printf "%s\n" "$status" > "$state/$name.exit"; }
trap cleanup EXIT
.venv/bin/python "$dev_root/tools/decode_loop.py" > "$state/$name.log" 2>&1 &
main=$!
until grep -q "DECODE START" "$state/$name.log" 2>/dev/null || ! kill -0 $main 2>/dev/null; do sleep 2; done
sleep 3
ec=$(pgrep -f "VLLM::EngineCore" | head -1); w0=$(pgrep -f "VLLM::Worker_TP0" | head -1)
.venv/bin/py-spy record -p "$ec" -d 12 -r 200 -f raw -o "$state/$name.engine.txt" --nonblocking > "$state/$name.spy1.log" 2>&1 &
.venv/bin/py-spy record -p "$w0" -d 12 -r 200 -f raw -o "$state/$name.worker.txt" --nonblocking > "$state/$name.spy2.log" 2>&1 &
.venv/bin/py-spy record -p "$main" -d 12 -r 200 -f raw -o "$state/$name.main.txt" --nonblocking > "$state/$name.spy3.log" 2>&1 &
wait $main
