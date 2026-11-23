#!/bin/bash
set -eu
# DeepSeek-V4.1 target on DGX-2H (16 V100s), with DSpark5 verification.
here=${V41_DIR:-/data/ds41}
export V41_DIR="$here"
# Run this mode once after model loading to populate both host Engram tables.
if [[ "${1:-}" == "--warm-engram" ]]; then
    for file in /data/ds41/engram/{engram-01,engram-14}.rows; do
        [[ -r "$file" ]] || exit 1
    done
    dd if=/data/ds41/engram/engram-01.rows of=/dev/null bs=16M & first=$!
    dd if=/data/ds41/engram/engram-14.rows of=/dev/null bs=16M & second=$!
    wait "$first"
    wait "$second"
    exec python3 /data/ds41/tools/incore.py /data/ds41/engram/*.rows
fi
export V41_CONFIG="$here"/src/benchmarks/deepseek_v41/dgx/serve-dgx-1m.yaml
export VLLM_V41_SESSION_PREFIX=1 VLLM_DP_PREFIX_AFFINITY=1 VLLM_V41_SWA_FP8=1
export VLLM_V41_BATCH=1 VLLM_V41_WINDOW_RING=0 VLLM_V41_ASYNC=1
export VLLM_V41_PACKED_KV_PAGES=1 VLLM_V41_INDEX_SHARD=1 VLLM_V41_LIVE_TABLE=1 VLLM_V41_TM_CODES=1
export VLLM_V41_COMBINE_TP=1 VLLM_V41_PROPOSE_GRAPH=1 VLLM_V41_BATCH_GREEDY_VERIFY=1
export VLLM_V41_Q8_ROWS_MAX=96 VLLM_V41_DENSE_TM_ROWS=0 VLLM_V41_DENSE_SINGLE=1 VLLM_V41_PREFILL_DEC_ATTN=1
export VLLM_V41_HC_PARTS_CUDA=1 VLLM_V41_EXPERT_REPLICAS=/data/ds41/tools/replicas_all.json
export VLLM_V41_EP4_TP4=1 VLLM_V41_DRAFT_PROJ_TP=1
export VLLM_V41_DRAFT_HEAD_FP8=${VLLM_V41_DRAFT_HEAD_FP8:-1}
export VLLM_V41_SOLO_DP_GRAPH=${VLLM_V41_SOLO_DP_GRAPH:-1}
export VLLM_V41_BATCH_SHORT_GRAPH=${VLLM_V41_BATCH_SHORT_GRAPH:-1}
# Long prefill on the 16-V100 EP4 x TP4 target.
export VLLM_V41_INDEX_PREFILL_CUDA=${VLLM_V41_INDEX_PREFILL_CUDA:-1}
export VLLM_V41_MOE_PF_GEMM=${VLLM_V41_MOE_PF_GEMM:-1}
export VLLM_V41_ATTN_H64=${VLLM_V41_ATTN_H64:-1}
export VLLM_V41_MOE_PF_DIRECT=${VLLM_V41_MOE_PF_DIRECT:-1}
export VLLM_V41_HC_PF_TP4=${VLLM_V41_HC_PF_TP4:-1}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$here/src"
# Leave workspace for mixed prefill and the DSpark graphs on 32 GiB ranks.
# The 2 GiB KV allocation still covers a 1M-token request per DP engine.
exec bash benchmarks/deepseek_v41/dgx/run_dgx.sh --kv-cache-memory-bytes 2147483648 "$@"
