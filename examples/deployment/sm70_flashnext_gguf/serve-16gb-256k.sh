#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Source deployment with one full 256K request on four 16 GiB V100s.
set -euo pipefail
: "${MODEL:?set MODEL to the first IQ3_S GGUF shard}"
: "${DRAFT:?set DRAFT to the compact MTP directory}"
: "${VLLM_PYTHON:?set VLLM_PYTHON to the source environment Python}"
SRC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/onecat-flashnext-16gb}
PORT=${PORT:-8000}
HOST=${HOST:-127.0.0.1}
GPU_UTIL=${GPU_UTIL:-0.95}
mkdir -p "$CACHE_DIR"
export PATH="$(dirname "$VLLM_PYTHON"):$PATH"
export PYTHONPATH="$SRC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export VLLM_CACHE_ROOT="$CACHE_DIR/vllm" TRITON_CACHE_DIR="$CACHE_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/inductor"
export TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch_extensions"
export VLLM_SM70_GEMM_LUT_PATH="$CACHE_DIR/gemm-lut-{device}.bin"

# Host history pools are about 15.3 GiB for this geometry. Reserve another 6 GiB
# for workers, staging and the bounded CPU row cache before any GPU is touched.
"$VLLM_PYTHON" - <<'PY'
import shutil
from pathlib import Path
if not shutil.which('ninja'):
    raise SystemExit('Ninja is required in the source environment for FlashQLA')
if not shutil.which('nvcc'):
    raise SystemExit('The CUDA source toolchain is required in PATH')
info = {line.split(':')[0]: int(line.split()[1]) * 1024
        for line in Path('/proc/meminfo').read_text().splitlines()}
if info['MemAvailable'] < 21 * 1024**3:
    raise SystemExit('Not enough available host memory for 256K FP16 history and workers')
PY

KERNEL_CONFIG='{"sm70_gguf":{"expert_storage":"original","embedding_storage":"original","dense_storage":"canonical","small_m_dp4a":true,"q8_expert_intermediate":true,"small_m_hmma":true,"lut4_expert_dp4a":true,"device_transcode":true,"dequant_workspace_bytes":33554432},"sm70_router_weight_storage":"row_major","sm70_mtp_lossless_storage":true,"hc_weight_storage":"sharded","hc_ll_shard":true,"hc_ll_optimized_loads":false,"sm70_hcx":true,"sm70_hcx_output_projection":false,"sm70_hcx_local_schedule":true,"sm70_qsa_shared_key":true,"sm70_qsa_device_history":true,"qsa_host_kv":true,"qsa_host_indexer_history":true,"qsa_host_kv_dtype":"float16","qsa_host_kv_draft_dtype":"float16","qsa_host_kv_device_reference":false,"qsa_host_kv_hot_tokens":8192,"qsa_host_kv_state_blocks":29,"qsa_auto_e4m3":false,"ple_disk_only":true,"ple_row_cache_mib":512,"ple_input_prepare":true,"ple_pinned_decode":false,"sm70_greedy_verify":true,"sm70_draft_single_graph":true,"sm70_fused_side_projections":true,"sm70_top1x":true}'
SPEC_CONFIG=$("$VLLM_PYTHON" - "$DRAFT" <<'PY'
import json, sys
print(json.dumps({'method': 'mtp', 'model': sys.argv[1], 'num_speculative_tokens': 4,
                  'draft_load_config': {'load_format': 'safetensors'},
                  'draft_sample_method': 'greedy'}))
PY
)

# Hold every lock for the entire server lifetime.
exec 9>/tmp/gpu0-3.lock
flock -E 75 -w 3600 9
exec 8>/tmp/1cat-vllm-v100-gpus0123.lock
flock -E 75 -w 600 8
for g in 0 1 2 3; do
  for name in /tmp/gpu$g.lock /tmp/1cat-vllm-v100-gpu$g.lock; do
    exec {fd}>"$name"
    flock -E 75 -w 600 "$fd"
  done
done

exec "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name flash-next --host "$HOST" --port "$PORT" \
  --quantization gguf --tensor-parallel-size 4 --dtype half \
  --kv-cache-dtype float16 --mamba-ssm-cache-dtype float32 \
  --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 512 \
  --gpu-memory-utilization "$GPU_UTIL" --enable-prefix-caching --mamba-cache-mode align \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --compilation-config '{"mode":3,"cudagraph_mode":"FULL"}' \
  --kernel-config "$KERNEL_CONFIG" --speculative-config "$SPEC_CONFIG"
