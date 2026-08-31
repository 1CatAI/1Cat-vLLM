// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kGlm53K = 4096;
constexpr int kGlm53N = 6416;
constexpr int kLanesPerRow = 16;
constexpr int kChunkK = 512;
constexpr int kChunkCount = kGlm53K / kChunkK;
constexpr int kThreads = kLanesPerRow * kChunkCount;

template <int kBatch>
__global__ void glm53_fp16_gemv_sm70_kernel(half* __restrict__ output,
                                            const half* __restrict__ input,
                                            const half* __restrict__ weight) {
  const int thread = threadIdx.x;
  const int chunk = thread / kLanesPerRow;
  const int lane = thread & (kLanesPerRow - 1);
  const int row = blockIdx.x;

  // Each (chunk, lane) pair preserves cuBLAS's 32-element ascending FMA
  // chain. Shared memory then joins chunk 0..7 followed by lane 0..15.
  float chunk_sum[kBatch] = {};
#pragma unroll 4
  for (int tile = 0; tile < kChunkK / kLanesPerRow; ++tile) {
    const int k = chunk * kChunkK + tile * kLanesPerRow + lane;
    const half w = __ldg(weight + static_cast<size_t>(row) * kGlm53K + k);
#pragma unroll
    for (int batch = 0; batch < kBatch; ++batch) {
      const half x = __ldg(input + static_cast<size_t>(batch) * kGlm53K + k);
      chunk_sum[batch] =
          __fmaf_rn(__half2float(x), __half2float(w), chunk_sum[batch]);
    }
  }

  __shared__ float chunk_partials[kBatch][kThreads];
#pragma unroll
  for (int batch = 0; batch < kBatch; ++batch) {
    chunk_partials[batch][chunk * kLanesPerRow + lane] = chunk_sum[batch];
  }
  __syncthreads();

  if (chunk == 0) {
#pragma unroll
    for (int batch = 0; batch < kBatch; ++batch) {
      float lane_sum = chunk_partials[batch][lane];
#pragma unroll
      for (int source_chunk = 1; source_chunk < kChunkCount; ++source_chunk) {
        lane_sum = __fadd_rn(
            lane_sum,
            chunk_partials[batch][source_chunk * kLanesPerRow + lane]);
      }

      float row_sum = 0.0f;
#pragma unroll
      for (int source_lane = 0; source_lane < kLanesPerRow; ++source_lane) {
        const float value =
            __shfl_sync(0x0000ffffu, lane_sum, source_lane, kLanesPerRow);
        if (lane == 0) {
          row_sum = source_lane == 0 ? value : __fadd_rn(row_sum, value);
        }
      }
      if (lane == 0) {
        output[static_cast<size_t>(batch) * kGlm53N + row] =
            __float2half_rn(row_sum);
      }
    }
  }
}

void validate_glm53_fp16_gemv_tensors(const torch::Tensor& output,
                                      const torch::Tensor& input,
                                      const torch::Tensor& weight) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda(),
              "sm70_glm53_fp16_gemv_out: tensors must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half &&
                  weight.scalar_type() == at::ScalarType::Half &&
                  output.scalar_type() == at::ScalarType::Half,
              "sm70_glm53_fp16_gemv_out: tensors must be float16");
  TORCH_CHECK(
      input.is_contiguous() && weight.is_contiguous() && output.is_contiguous(),
      "sm70_glm53_fp16_gemv_out: tensors must be contiguous");
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= 8 &&
                  input.size(1) == kGlm53K,
              "sm70_glm53_fp16_gemv_out: input must be [M, 4096] with "
              "1 <= M <= 8");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == kGlm53N &&
                  weight.size(1) == kGlm53K,
              "sm70_glm53_fp16_gemv_out: weight must be [6416, 4096]");
  TORCH_CHECK(output.dim() == 2 && output.size(0) == input.size(0) &&
                  output.size(1) == kGlm53N,
              "sm70_glm53_fp16_gemv_out: output must be [M, 6416]");
  TORCH_CHECK(
      input.device() == weight.device() && input.device() == output.device(),
      "sm70_glm53_fp16_gemv_out: tensors must share a device");
}

}  // namespace

void sm70_glm53_fp16_gemv_out(torch::Tensor output, torch::Tensor input,
                              torch::Tensor weight) {
  validate_glm53_fp16_gemv_tensors(output, input, weight);
  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "sm70_glm53_fp16_gemv_out: requires SM70");
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(input.device().index());
#define VLLM_LAUNCH_GLM53_GEMV(batch)                                      \
  case batch:                                                              \
    glm53_fp16_gemv_sm70_kernel<batch><<<kGlm53N, kThreads, 0, stream>>>(  \
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),              \
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),         \
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()));       \
    break
  switch (input.size(0)) {
    VLLM_LAUNCH_GLM53_GEMV(1);
    VLLM_LAUNCH_GLM53_GEMV(2);
    VLLM_LAUNCH_GLM53_GEMV(3);
    VLLM_LAUNCH_GLM53_GEMV(4);
    VLLM_LAUNCH_GLM53_GEMV(5);
    VLLM_LAUNCH_GLM53_GEMV(6);
    VLLM_LAUNCH_GLM53_GEMV(7);
    VLLM_LAUNCH_GLM53_GEMV(8);
  }
#undef VLLM_LAUNCH_GLM53_GEMV
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef VLLM_GLM53_GEMV_STANDALONE
TORCH_LIBRARY_FRAGMENT(_C_glm53_gemv_bench, ops) {
  ops.def(
      "sm70_glm53_fp16_gemv_out(Tensor(a!) output, Tensor input, Tensor "
      "weight) -> ()");
  ops.impl("sm70_glm53_fp16_gemv_out", torch::kCUDA, &sm70_glm53_fp16_gemv_out);
}
#elif defined(VLLM_GLM53_GEMV_SIDECAR)
TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "sm70_glm53_fp16_gemv_out(Tensor(a!) output, Tensor input, Tensor "
      "weight) -> ()");
  ops.impl("sm70_glm53_fp16_gemv_out", torch::kCUDA, &sm70_glm53_fp16_gemv_out);
}
#endif
