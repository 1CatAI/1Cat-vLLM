#!/bin/bash
# run_pp4_nsys.sh RANK: run_pp4.sh under nsys; capture starts at /start_profile.
RANK=$1; shift
mkdir -p $HOME/v41/prof
exec /usr/local/bin/nsys profile -o $HOME/v41/prof/stage$RANK -f true \
  --trace=cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=stop --trace-fork-before-exec=true --sample=none --cpuctxsw=none \
  $HOME/v41/run_pp4.sh $RANK --profiler-config '{"profiler": "cuda", "max_iterations": 120}' "$@"
