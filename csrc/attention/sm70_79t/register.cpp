// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/cuda/Exceptions.h>
#include <map>
#include "shared_workspace.h"

namespace onecat_sm70_prefill {
ScoreWorkspace::ScoreWorkspace(const at::Tensor& query, int64_t block_n)
    : scores(at::empty({block_n * 8192 * 6}, query.options())) {
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming));
}
ScoreWorkspace::~ScoreWorkspace() {
  if (completion != nullptr) cudaEventDestroy(completion);
}
std::shared_ptr<ScoreWorkspace> get_score_workspace(const at::Tensor& query,
                                                    int64_t block_n) {
  static std::mutex mutex;
  static std::map<std::pair<int, int64_t>, std::shared_ptr<ScoreWorkspace>>
      cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto& entry = cache[{query.get_device(), block_n}];
  if (!entry) entry = std::make_shared<ScoreWorkspace>(query, block_n);
  return entry;
}
}  // namespace onecat_sm70_prefill

extern "C" int64_t onecat_sm70_q8000_accumulation_bits();
extern "C" int64_t onecat_sm70_q8192_accumulation_bits();

namespace onecat_79t_q8192 {
at::Tensor sm70_d256_gqa_architecture_q8192_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    at::Tensor& out, double softmax_scale, bool causal);
}

TORCH_LIBRARY_FRAGMENT(_vllm_fa2_C, ops) {
  ops.def("sm70_d256_gqa_accumulation_bits() -> int", []() -> int64_t {
    return onecat_sm70_q8000_accumulation_bits() == 32 &&
                   onecat_sm70_q8192_accumulation_bits() == 32
               ? 32
               : 16;
  });
  ops.def(
      "sm70_d256_gqa_architecture_q8192_fwd(Tensor q, Tensor k, Tensor v, "
      "Tensor(a!) out, float softmax_scale, bool causal) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(_vllm_fa2_C, CUDA, ops) {
  ops.impl("sm70_d256_gqa_architecture_q8192_fwd",
           &onecat_79t_q8192::sm70_d256_gqa_architecture_q8192_fwd);
}
