#!/bin/bash
# Build all four nodes and propagate any node's compiler or SSH failure.
. "$(dirname "$0")/common.sh"

pids=()
for host in $NODES; do
  ssh "$host" '
    cd "$HOME/v41/src" || exit 1
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=/usr/local/cuda-12.8/bin:$HOME/v41env/bin:$PATH
    "$HOME/v41env/bin/python" benchmarks/deepseek_v41/v41k/build.py > "$HOME/v41/logs/build.log" 2>&1
    status=$?
    printf "%s: build exit=%s; %s\n" "$(hostname)" "$status" "$(tail -n 1 "$HOME/v41/logs/build.log")"
    exit "$status"
  ' &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
