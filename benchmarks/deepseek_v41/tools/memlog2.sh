#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# memlog2.sh <out> <progress-file>: host memcg + per-GPU free VRAM every 20 s
out=$1; prog=$2
while true; do
  a=$(docker exec ds41-compute-20261024 sh -c "grep -E \"^(anon|shmem) \" /sys/fs/cgroup/memory.stat | tr \"\n\" \" \"")
  avail=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
  gfree=$(nvidia-smi --id=0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits | tr "\n" " ")
  q=$(wc -l < "$prog" 2>/dev/null)
  echo "$(date +%H:%M:%S) n=$q avail_mb=$((avail/1024)) $a gpu_free_mib: $gfree" >> "$out"
  sleep 20
done
