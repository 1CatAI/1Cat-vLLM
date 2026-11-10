// SPDX-License-Identifier: BSD-3-Clause

// Build the same FP32-accumulated architecture with a native Q4096 causal
// tail for 4096-token prefill chunks. A private namespace keeps its device
// globals and workspace cache independent from the Q8000/Q8192 builds.
#define FLASH_NAMESPACE onecat_79t_q4096
#define PREFIX_TORCH_QUERY_TOKENS 4096
#define PREFIX_TORCH_ARCHITECTURE_FUNCTION sm70_d256_gqa_architecture_q4096_fwd
#undef PREFIX_BATCHED_TAIL_TILE_TOKENS
#define PREFIX_BATCHED_TAIL_TILE_TOKENS 256

#include "prefill.cu"
