// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026, 1CatAI.

// Research-only ABI adapter for the historical 79-TFLOP frontier.  The old
// endpoint used an FP16 numerator, while the current qualified kernel uses an
// FP32 numerator.  Keep the conversion explicit so this path is source-
// complete and cannot silently reinterpret arguments or buffers.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

extern "C" cudaError_t onecat_v37_dense_state_float_raw(
    const void* q, const void* k, const void* v, float* state_max,
    float* state_sum, void* out, int query_len, int kv_len, int heads_q,
    int heads_kv, float softmax_scale, int unnormalized, cudaStream_t stream);

extern "C" cudaError_t onecat_sm70_d256_dense_state_legacy_raw(
    const void* q, const void* k, const void* v, float* state_max,
    float* state_sum, void* out, int query_len, int kv_len, int heads_q,
    int heads_kv, float softmax_scale, cudaStream_t stream);

namespace {

constexpr int kTail = 8000;
constexpr int kHeadsQ = 6;
constexpr int kHeadsKV = 1;
constexpr int kHeadDim = 256;

__global__ void fp32_to_fp16_kernel(const float* src, half* dst,
                                    std::size_t elements) {
  std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    dst[index] = __float2half_rn(src[index]);
  }
}

cudaError_t alloc_async(void** ptr, std::size_t bytes, cudaStream_t stream) {
  return cudaMallocAsync(ptr, bytes, stream);
}

}  // namespace

extern "C" cudaError_t onecat_sm70_d256_dense_state_raw(
    const void* q, const void* k, const void* v, float* state_max,
    float* state_sum, void* out, int query_len, int kv_len, int heads_q,
    int heads_kv, float softmax_scale, cudaStream_t stream) {
  // Preserve the legacy implementation for aligned or unsupported shapes;
  // the historical architecture endpoint uses exactly this fixed tail.
  if (query_len != kTail || kv_len != kTail || heads_q != kHeadsQ ||
      heads_kv != kHeadsKV || !std::isfinite(softmax_scale)) {
    return onecat_sm70_d256_dense_state_legacy_raw(
        q, k, v, state_max, state_sum, out, query_len, kv_len, heads_q,
        heads_kv, softmax_scale, stream);
  }
  if (q == nullptr || k == nullptr || v == nullptr || state_max == nullptr ||
      state_sum == nullptr || out == nullptr) {
    return cudaErrorInvalidValue;
  }

  float* max_pad = nullptr;
  float* sum_pad = nullptr;
  float* out_pad = nullptr;
  cudaError_t status = cudaSuccess;
  auto cleanup = [&]() {
    if (max_pad != nullptr) cudaFreeAsync(max_pad, stream);
    if (sum_pad != nullptr) cudaFreeAsync(sum_pad, stream);
    if (out_pad != nullptr) cudaFreeAsync(out_pad, stream);
  };

  status = alloc_async(
      reinterpret_cast<void**>(&max_pad),
      static_cast<std::size_t>(kTail) * kHeadsQ * sizeof(float), stream);
  if (status != cudaSuccess) {
    cleanup();
    return status;
  }
  status = alloc_async(
      reinterpret_cast<void**>(&sum_pad),
      static_cast<std::size_t>(kTail) * kHeadsQ * sizeof(float), stream);
  if (status != cudaSuccess) {
    cleanup();
    return status;
  }
  status = alloc_async(
      reinterpret_cast<void**>(&out_pad),
      static_cast<std::size_t>(kTail) * kHeadsQ * kHeadDim * sizeof(float),
      stream);
  if (status != cudaSuccess) {
    cleanup();
    return status;
  }

  status = onecat_v37_dense_state_float_raw(
      q, k, v, max_pad, sum_pad, out_pad, kTail, kTail, kHeadsQ, kHeadsKV,
      softmax_scale, /*unnormalized=*/0, stream);
  if (status == cudaSuccess) {
    constexpr std::size_t kOutputElements =
        static_cast<std::size_t>(kTail) * kHeadsQ * kHeadDim;
    constexpr int kThreads = 256;
    const int blocks =
        static_cast<int>((kOutputElements + kThreads - 1) / kThreads);
    fp32_to_fp16_kernel<<<blocks, kThreads, 0, stream>>>(
        out_pad, static_cast<half*>(out), kOutputElements);
    status = cudaPeekAtLastError();
  }
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(
        state_max, max_pad,
        static_cast<std::size_t>(kTail) * kHeadsQ * sizeof(float),
        cudaMemcpyDeviceToDevice, stream);
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(
        state_sum, sum_pad,
        static_cast<std::size_t>(kTail) * kHeadsQ * sizeof(float),
        cudaMemcpyDeviceToDevice, stream);
  cleanup();
  return status;
}
