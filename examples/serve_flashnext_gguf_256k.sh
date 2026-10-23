#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Qualified Flash-Next GGUF TP4 settings: 256K total tokens, host KV and disk PLE.
set -euo pipefail
SOURCE_DIR=${SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# An in-place build (setup.py build_ext --inplace) runs from SOURCE_DIR;
# otherwise use the installed 1cat-vllm wheel.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN=$SOURCE_DIR/.venv/bin/python
  [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=$(command -v python3)
fi
INPLACE_BUILD=false
compgen -G "$SOURCE_DIR/vllm/_C*.so" >/dev/null && INPLACE_BUILD=true
: "${MODEL_FILE:?Set MODEL_FILE to the first IQ3_S GGUF shard}"
: "${MTP_DIR:?Set MTP_DIR to the matching stock MTP safetensors directory}"
: "${CACHE_DIR:?Set CACHE_DIR to a dedicated runtime cache}"
: "${VLLM_API_KEY:?Set VLLM_API_KEY for the public API}"
PORT=${PORT:-8000}
# Bind address; 0.0.0.0 inside a container published with -p.
HOST=${HOST:-127.0.0.1}
GPU_IDS=${GPU_IDS:-0,1,2,3}
# Optional CPU pinning, e.g. the cores of the socket that owns the GPUs.
CPU_SET=${CPU_SET:-}
EXPERT_PROFILE=${EXPERT_PROFILE:-}
QSA_HOST_INDEXER_HISTORY=${QSA_HOST_INDEXER_HISTORY:-false}
# Concurrent requests. MTP4 verifies 5 rows per request: capture one CUDA
# graph per 5 * {1, 2, 4, ...} up to the limit (uncaptured batches run eager).
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
CAPTURE_SIZES=$(r=1; s=""; while [ "$r" -lt "$MAX_NUM_SEQS" ]; do
  s="$s$((5 * r)),"; r=$((r * 2)); done; echo "${s}$((5 * MAX_NUM_SEQS))")
# Context window. Beyond the native 262144 positions RoPE uses YaRN scaling
# (factor = ceil(MAX_MODEL_LEN / 262144)); up to 262144 keeps the stock RoPE.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
NATIVE_LEN=262144
# Recurrent-state pool: 8 blocks per request (4 GDN groups x running state
# and aligned checkpoint; speculative states are committed by replay), plus
# 4 shared blocks for each of up to two requests entering a new block.
STATE_BLOCKS=$((1 + 8 * MAX_NUM_SEQS + 4 * (MAX_NUM_SEQS < 2 ? MAX_NUM_SEQS : 2)))
# Windows past the native length hold a 4x larger indexer history on the GPU:
# halve the prefill chunk (activation peak) and trim the hot cache to fit 16 GB.
PREFILL_CHUNK=8192 HOT_TOKENS=32768 STAGE_TOKENS=525504 EXPERT_FRACTION=0.8
# Concurrent requests hold more recurrent states and larger verification
# graphs, and prefill shares steps with resident decodes: halve the prefill
# chunk (activation peak) and stage prefill history in smaller windows to fit.
if (( MAX_NUM_SEQS > 1 )); then PREFILL_CHUNK=4096 STAGE_TOKENS=131072; fi
# Sparse attention touches most of each request's history within ~100 decode
# steps, so the shared hot cache should cover every concurrent context:
# misses cross PCIe. A smaller state pool leaves room for it on 16 GB cards:
# two to four requests get 64K tokens per layer (96K leaves under 100 MiB for
# mixed prefill activations), more keep 32K.
if (( MAX_NUM_SEQS >= 2 && MAX_NUM_SEQS <= 4 )); then HOT_TOKENS=65536; fi
# Prefill sharing a step with resident decodes keeps at least half a chunk:
# the adaptive budget charges the decodes' fixed cost to the prefill tokens
# and would otherwise shrink to a few tokens per step.
MIXED_ARGS=()
# The draft hot cache is one layer; concurrent requests share it.
DRAFT_HOT_TOKENS=16384
(( MAX_NUM_SEQS > 1 )) && DRAFT_HOT_TOKENS=32768
if (( MAX_NUM_SEQS > 1 )); then
  MIXED_ARGS=(--mixed-prefill-min-tokens "$((PREFILL_CHUNK / 2))")
fi
if (( MAX_MODEL_LEN > NATIVE_LEN )); then
  PREFILL_CHUNK=4096 HOT_TOKENS=16384 STAGE_TOKENS=525504 EXPERT_FRACTION=0.77
fi
# Save the recurrent state at the end of each prompt so the next chat turn
# reuses the whole previous prompt from the prefix cache (one extra prefill
# step per prompt). Set to 0 to skip it.
MAMBA_TAIL_CHECKPOINT=${MAMBA_TAIL_CHECKPOINT:-1}
TAIL_ARG=--mamba-tail-checkpoint
[[ "$MAMBA_TAIL_CHECKPOINT" == 0 ]] && TAIL_ARG=--no-mamba-tail-checkpoint
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$CUDA_HOME/bin:$(dirname "$PYTHON_BIN"):$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_IDS"
if [[ "$INPLACE_BUILD" == true ]]; then export PYTHONPATH="$SOURCE_DIR"; fi
export TORCH_CUDA_ARCH_LIST=7.0
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:16:8}
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-4194304} NCCL_MAX_CTAS=${NCCL_MAX_CTAS:-6}
export PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2
# Serve large transient host buffers (images, prefill staging) from mmap so
# glibc returns them to the OS instead of growing the heap.
export MALLOC_MMAP_THRESHOLD_=${MALLOC_MMAP_THRESHOLD_:-4194304}
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_EXPERT_CACHE_STACKS=1 VLLM_BALANCED_PREFILL_CHUNKS=1
export VLLM_SM70_MTP_PROFILE=0 VLLM_NVTX_SCOPES_FOR_PROFILING=0
export MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Compiled graphs embed window-sized tables (scaled RoPE): keep one compile
# cache per non-default window so switching windows never reuses stale code.
VLLM_CACHE_SUBDIR=vllm
(( MAX_MODEL_LEN != NATIVE_LEN )) && VLLM_CACHE_SUBDIR=vllm-$MAX_MODEL_LEN
export VLLM_CACHE_ROOT="$CACHE_DIR/$VLLM_CACHE_SUBDIR" TRITON_CACHE_DIR="$CACHE_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/inductor"
export TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch_extensions"
export VLLM_SM70_GEMM_LUT_PATH="$CACHE_DIR/gemm-lut-{device}.bin"
unset LD_PRELOAD LD_AUDIT ONECAT_VLLM_EXTENSION_DIR
mkdir -p "$CACHE_DIR"
KC=$("$PYTHON_BIN" - "$EXPERT_PROFILE" "$QSA_HOST_INDEXER_HISTORY" "$STATE_BLOCKS" "$HOT_TOKENS" "$STAGE_TOKENS" "$EXPERT_FRACTION" "$DRAFT_HOT_TOKENS" <<'PY'
import json
import sys
if sys.argv[2].lower() not in ("true", "false"):
    raise ValueError("QSA_HOST_INDEXER_HISTORY must be true or false")
config = {
    "sm70_gguf": {
        "expert_storage": "original", "expert_arena": True,
        "embedding_storage": "original", "dense_storage": "canonical",
        "small_m_dp4a": True, "q8_expert_intermediate": True,
        "small_m_hmma": True, "lut4_expert_dp4a": True,
        "device_transcode": True, "dequant_workspace_bytes": 33554432,
        "expert_cache_fraction": float(sys.argv[6]), "expert_cache_host_slack": 0.0,
        "expert_cache_prefill": "turbomind", "expert_cache_prefill_tokens": 2048,
        "expert_load_prefetch_banks": 1,
    },
    "sm70_turbomind_workspace_bytes": 8388608,
    "sm70_router_weight_storage": "row_major", "sm70_mtp_q8_0_experts": True,
    "hc_weight_storage": "sharded", "hc_ll_shard": True,
    "hc_ll_optimized_loads": False, "sm70_hcx": True,
    "sm70_hcx_output_projection": False, "sm70_hcx_local_schedule": True,
    "sm70_qsa_shared_key": True, "sm70_qsa_device_history": True,
    "sm70_qsa_host_native": True, "qsa_host_kv": True,
    "qsa_host_indexer_history": sys.argv[2].lower() == "true",
    "qsa_host_kv_dtype": "fp8_e4m3",
    "qsa_host_kv_draft_dtype": "float16", "qsa_host_kv_device_reference": False,
    "qsa_host_kv_hot_tokens": int(sys.argv[4]), "qsa_host_kv_hot_dtype": "fp8_e4m3",
    "qsa_host_kv_draft_hot_tokens": int(sys.argv[7]), "qsa_host_kv_staging_rows": 8,
    "qsa_host_kv_prefill_stage_tokens": int(sys.argv[5]),
    "qsa_host_kv_prefill_stage_slots": 1,
    "qsa_host_kv_trim_warmup_cache": True,
    "qsa_host_kv_allocator_headroom_mib": 192,
    "sm70_sparse": {
        "qsa_score_tile_mb": 32, "qsa_prefill_tp_rows": True,
        "qsa_decode_partition_topk": True,
    },
    "layer_execution": {
        "ple_prefill_conv": True, "hc_prefill_sequence_parallel": True,
        "hc_mtp_prefill_sequence_parallel": True, "hc_prefill_peer_norm": True,
    },
    "qsa_host_kv_state_blocks": int(sys.argv[3]),
    "qsa_auto_e4m3": False, "sm70_gdn_verify": True, "sm70_gdn_spec_replay": True, "ple_row_cache_mib": 512,
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
if [[ "$INPLACE_BUILD" == true ]]; then cd "$SOURCE_DIR"; else cd "$CACHE_DIR"; fi
MM_ARGS=(--language-model-only)
if [[ -n "${VISION_DIR:-}" ]]; then
  test -f "$VISION_DIR/visual.safetensors"
  MM_LOAD=$("$PYTHON_BIN" - "$VISION_DIR/visual.safetensors" <<'PY'
import json
import sys
print(json.dumps({"multimodal_safetensors": sys.argv[1]}))
PY
)
  MM_ARGS=(--hf-config-path "$VISION_DIR" --tokenizer "$VISION_DIR"
    --model-loader-extra-config "$MM_LOAD"
    --limit-mm-per-prompt '{"image":2147483647,"video":0}'
    --skip-mm-profiling --mm-encoder-tp-mode weights --mm-encoder-batch-size 1
    --mm-processor-cache-gb 0.25 --mm-processor-kwargs '{"max_pixels":4194304}')
fi
LONG_ARGS=()
if (( MAX_MODEL_LEN > NATIVE_LEN )); then
  ROPE=$("$PYTHON_BIN" - "$MAX_MODEL_LEN" "$NATIVE_LEN" "${VISION_DIR:+nested}" <<'PY'
import json
import sys
length, native, nested = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "nested"
rope = {
    "mrope_interleaved": True, "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25, "rope_theta": 10000000,
    "rope_type": "yarn", "factor": float(-(-length // native)),
    "original_max_position_embeddings": native,
}
text = {"rope_parameters": rope, "max_position_embeddings": length}
print(json.dumps({"text_config": text} if nested else text))
PY
)
  LONG_ARGS=(--hf-overrides "$ROPE")
fi
PIN=()
if [[ -n "$CPU_SET" ]]; then PIN=(taskset -c "$CPU_SET"); fi
exec "${PIN[@]}" "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL_FILE" \
  --served-model-name flash-next-abliterated flash-next qwen3.8-flash-next \
  --host "$HOST" --port "$PORT" --quantization gguf --tensor-parallel-size 4 \
  --dtype half --kv-cache-dtype float16 --mamba-ssm-cache-dtype float32 \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$PREFILL_CHUNK" "${MIXED_ARGS[@]}" \
  --gpu-memory-utilization 0.95 --enable-prefix-caching \
  --mamba-cache-mode align "$TAIL_ARG" "${MM_ARGS[@]}" "${LONG_ARGS[@]}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}' \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL","cudagraph_capture_sizes":['"$CAPTURE_SIZES"'],"inductor_compile_config":{"benchmark_combo_kernel":false}}' \
  --kernel-config "$KC" --speculative-config "$SPEC"

