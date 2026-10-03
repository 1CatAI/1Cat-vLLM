// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>
#include "common.cuh"

using torch::stable::Tensor;
using torch::stable::accelerator::DeviceGuard;

// Every bridge allocation is owned by a Torch tensor for the duration of
// one upstream call. GGML borrows the current Torch CUDA stream.
class TorchScratchPool final : public ggml_cuda_pool {
 public:
  explicit TorchScratchPool(const Tensor& prototype);
  void* alloc(size_t size, size_t* actual_size) override;
  void* alloc_workspace(size_t bytes, size_t* actual_size);
  void free(void* ptr, size_t size) override;

 private:
  void* alloc_impl(size_t bytes, size_t* actual_size);
  const Tensor& prototype_;
  std::vector<Tensor> owners_;
};

struct ProjectionBuffers {
  Tensor output;
  void* q8;
  const float* input;
  float* result;
};

struct UpstreamCall {
  explicit UpstreamCall(const Tensor& prototype);
  DeviceGuard device_guard;
  cudaStream_t stream;
  ggml_backend_cuda_context context;
  TorchScratchPool* scratch_pool = nullptr;
};

cudaStream_t current_stream(int32_t device_index);
void check_launch(const char* op_name);
bool has_weight_padding(const Tensor& W, int64_t k, int64_t type,
                        const char* op_name);
void cast_contiguous_async(void* dst, torch::headeronly::ScalarType dst_type,
                           const void* src,
                           torch::headeronly::ScalarType src_type,
                           int64_t count, cudaStream_t stream);
ProjectionBuffers projection_buffers(const Tensor& X, int64_t output_rows,
                                     int64_t row, size_t q8_bytes,
                                     TorchScratchPool& scratch_pool,
                                     cudaStream_t stream);
Tensor finish_output(const ProjectionBuffers& buffers, cudaStream_t stream);
