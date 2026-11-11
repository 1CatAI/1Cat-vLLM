#!/bin/bash
# bench.sh ["ENV=VAL ..."] LABEL [prefill]
# Restart, warm up, then measure fixed-DSpark5 greedy decode.
set -eo pipefail
. "$(dirname "$0")/common.sh"

EXTRA=${1:-}
LABEL=${2:-benchmark}
CLUSTER_DIR=$(dirname "$0")
"$CLUSTER_DIR/stop.sh" > /dev/null
"$CLUSTER_DIR/start.sh" run_pp4.sh "$EXTRA" > /dev/null
"$CLUSTER_DIR/wait_ready.sh"
echo "== $LABEL ($EXTRA)"
cd "$HOME/v41"
"$HOME/v41env/bin/python" bench_client.py --tokens 64 --temperature 0 > /dev/null 2>&1
"$HOME/v41env/bin/python" bench_client.py --tokens 256 --temperature 0 --repeat 2 2>&1 | grep MEAN
if [ "${3:-}" = prefill ]; then
  "$HOME/v41env/bin/python" -u prefill_bench.py 2>&1 | grep --line-buffered -E "prompt|tok/s"
fi
