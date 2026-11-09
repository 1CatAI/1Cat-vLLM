// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/ATen.h>
#include <torch/library.h>

namespace onecat_v37 {
at::Tensor sm70_d256_gqa_v37_fwd(const at::Tensor& q, const at::Tensor& k,
                                 const at::Tensor& v, at::Tensor& out,
                                 double scale, bool causal);
}

// Separate symbol: a stale legacy FA2 library must never masquerade as v37.
TORCH_LIBRARY_FRAGMENT(_vllm_fa2_C, ops) {
  ops.def(
      "sm70_d256_gqa_v37_fwd(Tensor q, Tensor k, Tensor v, Tensor(a!) out, "
      "float softmax_scale, bool causal) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(_vllm_fa2_C, CUDA, ops) {
  ops.impl("sm70_d256_gqa_v37_fwd", &onecat_v37::sm70_d256_gqa_v37_fwd);
}
