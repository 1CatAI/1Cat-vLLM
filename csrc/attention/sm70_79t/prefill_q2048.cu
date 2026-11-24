// SPDX-License-Identifier: BSD-3-Clause

// Build the V3 FP32-PV architecture with a native Q2048 causal tail for
// medium-length prefill tail chunks (1024..2048 tokens) while sharing the
// half-height Q4096 score workspace.
#define FLASH_NAMESPACE onecat_79t_q2048
#define PREFIX_TORCH_QUERY_TOKENS 2048
#define PREFIX_TORCH_ARCHITECTURE_FUNCTION sm70_d256_gqa_architecture_q2048_fwd
#undef PREFIX_BATCHED_TAIL_TILE_TOKENS
#define PREFIX_BATCHED_TAIL_TILE_TOKENS 128

#include "prefill.cu"
