// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>

namespace onecat_sm70_prefill {
// The kernels bind device-global pointers. All engines on a physical device
// must share this gate, even though their score tensors are independent.
struct ExecutionGate {
  const int device;
  std::mutex mutex;
  cudaEvent_t completion = nullptr;
  bool completion_recorded = false;
  explicit ExecutionGate(int device);
  ~ExecutionGate();
};
std::shared_ptr<ExecutionGate> execution_gate(int device);

struct ScoreWorkspace {
  const int64_t block_n;
  at::Tensor scores;
  std::shared_ptr<ExecutionGate> gate;
  std::mutex& mutex;
  cudaEvent_t& completion;
  bool& completion_recorded;
  explicit ScoreWorkspace(const at::Tensor& query, int64_t block_n);
  ~ScoreWorkspace();
};
std::shared_ptr<ScoreWorkspace> get_score_workspace(const at::Tensor& query,
                                                    int64_t block_n);
}  // namespace onecat_sm70_prefill
