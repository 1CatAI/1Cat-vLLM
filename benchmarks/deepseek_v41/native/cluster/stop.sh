#!/bin/bash
# stop.sh: stop the PP4 server on all four nodes and wait until every GPU is empty
# (a new server fails its free-memory check if an old worker is still exiting).
. $(dirname $0)/common.sh
for h in $NODES; do
  ssh $h 'p1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); p2=$(ps -eo pid,args | awk "/^ *[0-9]+ ([V]LLM::|.*[v]41env\/bin\/python -m vllm|.*[n]sys profile)/ {print \$1}"); kill $p1 $p2 2>/dev/null
    for i in $(seq 60); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
      [ "$used" -lt 1000 ] && break
      [ $i -gt 5 ] && kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) 2>/dev/null
      sleep 1
    done
    echo "$(hostname) gpu_mem_MiB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" ",")"' &
done
wait
