#!/usr/bin/env bash
# Qwen3.8-27B NVFP4 (QUASAR QAT) on 4x V100-SXM2-32GB, TP4, 256K, DFlash2,
# full-acceleration path, thinking off, prefix caching on.
#
# Two flags carry the long-context default and are easy to get wrong:
#
#   --block-size 2048        KV block / prefix-cache hash granularity
#   --mamba-block-size 8192  GDN recurrent-state checkpoint grid
#
# The prefill chunk is min(max_num_batched_tokens, state grid), and the 75T
# Q8000 dense prefill route only fires for a chunk in [8000, 8192]. Hybrid
# page-size unification otherwise scales the state grid together with the block
# size, so a 4096 block reached an 8192 grid by accident while a 2048 block fell
# to 4096 and silently lost the route. Pinning the grid keeps the chunk at 8192
# with the finer block: ~29% more KV capacity for free, at identical prefill
# speed and identical 75T dispatch.
#
# The grid MUST be a multiple of the block size. Otherwise no prefix length is
# simultaneously block-aligned (KV blocks) and grid-aligned (recurrent state),
# and prefix caching collapses to zero hits -- measured, not theoretical.
#
# Measured on this box, unique-salt 100K prompt, fp8_e4m3 KV, DFlash2:
#   block 4096 (grid 8192 auto)                  3426 tok/s    818,142 KV
#   block 2048 (grid 4096 auto)                  2954 tok/s  1,058,133 KV   no 75T
#   block 2048 + --mamba-block-size 8192         3336 tok/s  1,058,133 KV   75T
# Raw probe output: docs/design/... see 75T-PREFIX-CACHING-FINDINGS.md.
#
# Note that a smaller block does NOT give finer prefix-cache reuse: reuse is
# quantised by the state grid, so it stays at 8192 here. The block size buys KV
# capacity only.
set -euo pipefail

SRC=${SRC:-/home/ymzx/桌面/1cat-vllm/worktrees/v100-75t-prefix-bridge-20260916}
BASE=${BASE:-/data/minimax-h3/task-cache/v100-quasar-dflash2-15ms-20260908}
NATIVE_LIB_DIR=$BASE/lib-v4
MODEL=${MODEL:-/data/models/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4-d8e6fbfa}
DRAFT=${DRAFT:-/data/models/v100-dflash2-20260820/draft}
API_GPU_GROUP=${API_GPU_GROUP:-0,1,2,3}
CHAT_KWARGS=${CHAT_KWARGS:-'{"enable_thinking":false}'}

cd "$SRC"
export CUDA_VISIBLE_DEVICES=$API_GPU_GROUP
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=${CUDA_HOME:-/data/minimax-h3/task-cache/v100-quasar-w4a4-audit-20260902/cuda128-exact}
export PATH="$CUDA_HOME/bin:/home/ymzx/1cat-build/conc-defaults/.venv/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SRC:$NATIVE_LIB_DIR:$SRC/flash-attention-v100"
export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=2
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/home/ymzx/.cache/qwen38-mixed-20260913/torchinductor
export TRITON_CACHE_DIR=/home/ymzx/.cache/qwen38-mixed-20260913/triton
export TORCH_EXTENSIONS_DIR=/home/ymzx/.cache/qwen38-mixed-20260913/torch_extensions
export VLLM_CACHE_ROOT=/home/ymzx/.cache/qwen38-mixed-20260913/vllm-cache
export VLLM_USE_V2_MODEL_RUNNER=1

# Full-acceleration path: these bring the 1K verification round to ~16 ms.
export VLLM_SM70_DFLASH2_FP32_LOGITS=1
export VLLM_SM70_DFLASH2_QUANT_LM_HEAD=1
export VLLM_SM70_DFLASH2_CONTEXT_PIPELINE=1
export VLLM_SM70_DFLASH2_CONTEXT_KV_GRAPH=1
export VLLM_SM70_DFLASH2_FIXED_GEMMA_RMS=1
export VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY=1
export VLLM_SM70_DFLASH2_FUSED_GDN_COMBINED_SPLIT=1
export VLLM_SM70_DFLASH2_DIRECT_ATTENTION_OUTPUT=0
export VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL=1
export FLASH_QLA_SM70_USE_ORIGINAL_TILELANG=1
export FLASH_QLA_SM70_PREBUILT_EXTENSION_PATH="$BASE/cache/flashqla-build/flash_qla_sm70_gdn_strided.so"
export VLLM_FLASH_V100_ROUTE_SUMMARY=1
export VLLM_SM70_MTP_PROFILE=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_SERVER_DEV_MODE=0

exec /home/ymzx/1cat-build/conc-defaults/.venv/bin/python -u -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host 127.0.0.1 --port "${API_PORT:-18301}" \
  --served-model-name "${SERVED_NAME:-quasar-27b-nvfp4-256k}" \
  --dtype half --tensor-parallel-size 4 --attention-backend FLASH_ATTN_V100 \
  --kv-cache-dtype fp8_e4m3 --max-model-len 262144 \
  --gpu-memory-utilization "${GPU_UTIL:-0.8}" \
  --max-num-batched-tokens 8192 --max-num-seqs 4 \
  --enable-prefix-caching --mamba-cache-mode align \
  --block-size 2048 --mamba-block-size 8192 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs "$CHAT_KWARGS" \
  --seed 0 \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"revision\":\"dedf8df68adfb1afeaf7b7480c0a0243108177b4\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN_V100\",\"draft_sample_method\":\"probabilistic\",\"enforce_eager\":false}"
