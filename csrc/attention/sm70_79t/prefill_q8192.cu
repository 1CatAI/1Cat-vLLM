// SPDX-License-Identifier: BSD-3-Clause

// Build the same FP32-accumulated architecture with a native Q8192 causal
// tail.  A private namespace keeps its device globals and workspace cache
// independent from the qualified Q8000 specialization.
#define FLASH_NAMESPACE onecat_79t_q8192
#define PREFIX_TORCH_QUERY_TOKENS 8192
#define PREFIX_TORCH_ARCHITECTURE_FUNCTION sm70_d256_gqa_architecture_q8192_fwd
#undef PREFIX_BATCHED_TAIL_TILE_TOKENS
#define PREFIX_BATCHED_TAIL_TILE_TOKENS 256

#include "prefill.cu"
