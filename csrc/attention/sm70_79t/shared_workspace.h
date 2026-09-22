// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>

namespace onecat_sm70_prefill {
struct ScoreWorkspace {
  at::Tensor scores;
  std::mutex mutex;
  cudaEvent_t completion = nullptr;
  bool completion_recorded = false;
  explicit ScoreWorkspace(const at::Tensor& query, int64_t block_n);
  ~ScoreWorkspace();
};
std::shared_ptr<ScoreWorkspace> get_score_workspace(const at::Tensor& query,
                                                    int64_t block_n);
}  // namespace onecat_sm70_prefill
