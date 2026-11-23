#!/bin/bash
# run_dgx.sh [extra vllm args]: DeepSeek-V4.1-Flash on one DGX-2H (16 x V100-32GB,
# NVSwitch), TP4 x DP4 with the routed experts in EP16 (all-gather / reduce-scatter
# dispatch across the DP groups). Config: $V41_CONFIG (default serve-dgx.yaml here).
D=${V41_DIR:-/data/ds41}
export PYTHONPATH=$D/src
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$HOME/1Cat-vLLM/.venv/bin:$PATH
export VLLM_V41_NATIVE=1
export VLLM_GGUF_EXPERT_CACHE_JIT=1 VLLM_V41K_LIB=$D/src/benchmarks/deepseek_v41/v41k/build/v41k.so
export VLLM_V41_FP8_GEMV=1 VLLM_V41_EM=1 VLLM_V41_FUSE_ATTN_PREP=1 VLLM_V41_FUSE_ATTN_POST=1 VLLM_V41_MARKOV_SHARD=1 VLLM_SM70_TP4_PUSH_ALLREDUCE=0
export VLLM_V41_ROUTE_CUDA=${VLLM_V41_ROUTE_CUDA:-1} VLLM_V41_SHARED_STREAM=${VLLM_V41_SHARED_STREAM:-1} VLLM_V41_Q8_GROUPED=${VLLM_V41_Q8_GROUPED:-1} VLLM_V41_MOE_DEC=${VLLM_V41_MOE_DEC:-1} VLLM_V41_ATTN_DEC=${VLLM_V41_ATTN_DEC:-1} VLLM_V41_H16=${VLLM_V41_H16:-1} VLLM_V41_ATTN_RING=${VLLM_V41_ATTN_RING:-1} VLLM_V41_DEQUANT_MM_ROWS=${VLLM_V41_DEQUANT_MM_ROWS:-64} VLLM_SM70_FP8_TUNE_SMALL_SHAPES=${VLLM_SM70_FP8_TUNE_SMALL_SHAPES:-1} VLLM_SM70_FP8_DENSE_TUNE_MAX_M=${VLLM_SM70_FP8_DENSE_TUNE_MAX_M:-192} VLLM_V41_LL_AR=${VLLM_V41_LL_AR:-1} VLLM_V41_HC_SIDE=${VLLM_V41_HC_SIDE:-1} VLLM_V41_HC_SIDE_MAX=${VLLM_V41_HC_SIDE_MAX:-192} VLLM_FAST_ARGMAX=${VLLM_FAST_ARGMAX:-1} VLLM_V41_FUSED_CLIP=${VLLM_V41_FUSED_CLIP:-1} VLLM_V41_ATTN_RING_PREFILL=${VLLM_V41_ATTN_RING_PREFILL:-1}
# Top-p without the reference-row host sync (T > 0); never empty the allocator
# on one DP rank alone (a device sync while its peers wait in a collective).
# Batched experts: one weight read serves up to 4 x 8 rows (split-K 16 fits).
export V41SK_RB_MAX=${V41SK_RB_MAX:-4}
export VLLM_V41_TOPP_NO_FENCE=${VLLM_V41_TOPP_NO_FENCE:-1} VLLM_V41_EMPTY_CACHE_TOKENS=${VLLM_V41_EMPTY_CACHE_TOKENS:-0}
export VLLM_V41_ATTN_RING_POST=${VLLM_V41_ATTN_RING_POST:-1} VLLM_V41_PREP_REGS=${VLLM_V41_PREP_REGS:-1} VLLM_V41_GREEDY_VERIFY=${VLLM_V41_GREEDY_VERIFY:-1} VLLM_V41_ROUTE_MATCH=${VLLM_V41_ROUTE_MATCH:-1} VLLM_V41_W13_SWIGLU=${VLLM_V41_W13_SWIGLU:-1}
# fork: workers skip a second ~10-20 s import of vLLM (vLLM falls back to spawn
# by itself if the engine process has already initialized CUDA).
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-fork}
# 16 ranks each scan the checkpoint: loading takes longer than the 600 s default.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
# If host RAM runs out, the OOM killer takes the server, not the ssh/tunnel processes.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
exec python -m vllm.entrypoints.cli.main serve --config ${V41_CONFIG:-$D/src/benchmarks/deepseek_v41/dgx/serve-dgx.yaml} ${V41_ARGS-} "$@"
