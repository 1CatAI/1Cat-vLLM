#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Qualified Flash-Next GGUF TP4 settings: 256K total tokens, host KV and disk PLE.
set -euo pipefail
SOURCE_DIR=${SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-$SOURCE_DIR/.venv/bin/python}
: "${MODEL_FILE:?Set MODEL_FILE to the first IQ3_S GGUF shard}"
: "${MTP_DIR:?Set MTP_DIR to the matching stock MTP safetensors directory}"
: "${CACHE_DIR:?Set CACHE_DIR to a dedicated runtime cache}"
: "${VLLM_API_KEY:?Set VLLM_API_KEY for the public API}"
PORT=${PORT:-8000}
GPU_IDS=${GPU_IDS:-4,5,6,7}
CPU_SET=${CPU_SET:-24-47,72-95}
EXPERT_PROFILE=${EXPERT_PROFILE:-}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$CUDA_HOME/bin:$(dirname "$PYTHON_BIN"):$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$SOURCE_DIR"
export TORCH_CUDA_ARCH_LIST=7.0
export PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_EXPERT_CACHE_STACKS=1 VLLM_BALANCED_PREFILL_CHUNKS=1
export VLLM_SM70_MTP_PROFILE=0 VLLM_NVTX_SCOPES_FOR_PROFILING=0
export MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$CACHE_DIR/vllm" TRITON_CACHE_DIR="$CACHE_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/inductor"
export TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch_extensions"
export VLLM_SM70_GEMM_LUT_PATH="$CACHE_DIR/gemm-lut-{device}.bin"
unset LD_PRELOAD LD_AUDIT ONECAT_VLLM_EXTENSION_DIR
mkdir -p "$CACHE_DIR"
KC=$("$PYTHON_BIN" - "$EXPERT_PROFILE" <<'PY'
import json
import sys
config = {
    "sm70_gguf": {
        "expert_storage": "original", "expert_arena": True,
        "embedding_storage": "original", "dense_storage": "canonical",
        "small_m_dp4a": True, "q8_expert_intermediate": True,
        "small_m_hmma": True, "lut4_expert_dp4a": True,
        "device_transcode": True, "dequant_workspace_bytes": 33554432,
        "expert_cache_fraction": 0.8, "expert_cache_host_slack": 0.05,
        "expert_cache_prefill": "turbomind", "expert_cache_prefill_tokens": 4096,
    },
    "sm70_turbomind_workspace_bytes": 8388608,
    "sm70_router_weight_storage": "row_major", "sm70_mtp_q8_0_experts": True,
    "hc_weight_storage": "sharded", "hc_ll_shard": True,
    "hc_ll_optimized_loads": False, "sm70_hcx": True,
    "sm70_hcx_output_projection": False, "sm70_hcx_local_schedule": True,
    "sm70_qsa_shared_key": True, "sm70_qsa_device_history": True,
    "sm70_qsa_host_native": True, "qsa_host_kv": True,
    "qsa_host_indexer_history": False, "qsa_host_kv_dtype": "fp8_e4m3",
    "qsa_host_kv_draft_dtype": "float16", "qsa_host_kv_device_reference": False,
    "qsa_host_kv_hot_tokens": 32768, "qsa_host_kv_hot_dtype": "fp8_e4m3",
    "qsa_host_kv_draft_hot_tokens": 16384, "qsa_host_kv_staging_rows": 8,
    "qsa_host_kv_prefill_stage_tokens": 131072,
    "sm70_sparse": {"qsa_score_tile_mb": 32}, "qsa_host_kv_state_blocks": 29,
    "qsa_auto_e4m3": False, "ple_row_cache_mib": 512,
    "ple_input_prepare": True, "ple_pinned_decode": False,
    "ple_fast_lane": True, "ple_draft_prefetch": True,
    "sm70_greedy_verify": True, "sm70_sparse_target_sampling": True,
    "sm70_sparse_target_reference_check": False, "sm70_draft_single_graph": True,
    "sm70_fused_side_projections": True, "sm70_top1x": True,
    "sm70_draft_hot_vocab": 0,
}
if sys.argv[1]:
    config["sm70_gguf"]["expert_cache_profile"] = sys.argv[1]
print(json.dumps(config))
PY
)
SPEC=$("$PYTHON_BIN" - "$MTP_DIR" <<'PY'
import json
import sys
print(json.dumps({
    "method": "mtp", "model": sys.argv[1], "num_speculative_tokens": 4,
    "draft_load_config": {"load_format": "safetensors"},
    "draft_sample_method": "greedy",
}))
PY
)
cd "$SOURCE_DIR"
exec taskset -c "$CPU_SET" "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL_FILE" \
  --served-model-name flash-next-abliterated flash-next qwen3.8-flash-next \
  --host 127.0.0.1 --port "$PORT" --quantization gguf --tensor-parallel-size 4 \
  --dtype half --kv-cache-dtype float16 --mamba-ssm-cache-dtype float32 \
  --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 --enable-prefix-caching \
  --mamba-cache-mode align --no-mamba-tail-checkpoint --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}' \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL","cudagraph_capture_sizes":[5],"inductor_compile_config":{"benchmark_combo_kernel":false}}' \
  --kernel-config "$KC" --speculative-config "$SPEC"

