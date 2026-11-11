#!/bin/bash
# start.sh [launcher] ["ENV=VAL ..."]: start all four ranks, rank 0 (Dev-1, API) last.
# The old log is removed first: processes still exiting from a previous run keep
# writing to their open file, which would otherwise look like a new failure.
# launcher: run_pp4.sh (default) | run_pp4_nsys.sh. Extra env overrides launcher defaults.
. $(dirname $0)/common.sh
L=${1:-run_pp4.sh}; EXTRA="$2"
for r in 3 2 1 0; do
  ssh $(rank_ip $r) "mkdir -p ~/v41/logs; rm -f ~/v41/logs/serve.log; (setsid nohup env $EXTRA ~/v41/$L $r > ~/v41/logs/serve.log 2>&1 < /dev/null &); echo started-rank$r"
done
