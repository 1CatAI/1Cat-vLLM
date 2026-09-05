// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team
// Adapted by 1Cat-vLLM contributors from FlashInfer norm.cuh at
// 6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f. HC injection and materialized
// residual rounding are model semantics absent from upstream fused-add norm.
#pragma once
#include <flashinfer/norm.cuh>
namespace flashinfer::sm70::hc {
template <uint32_t VEC_SIZE, typename T, typename B, typename W>
__global__ void HCCombineNormKernel(
    const B* __restrict__ input, const T* __restrict__ residual,
    const W* __restrict__ weight, const W* __restrict__ injection,
    T* __restrict__ combined, T* __restrict__ output, uint32_t groups,
    uint32_t d, uint32_t stride_input, uint32_t stride_residual,
    uint32_t stride_injection, bool shared_weight, float eps) {
  const uint32_t bx = blockIdx.x, group = blockIdx.y;
  const float gate =
      2.f /
      (1.f + __expf(-float(injection[bx * stride_injection + group]) / groups));
  const uint32_t weight_offset = shared_weight ? 0 : group * d;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + ceil_div(num_warps, 4) * 4;

  float sum_sq = 0.f;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && \
     (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<B, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    vec_t<T, VEC_SIZE> residual_vec;
    residual_vec.fill(0.f);
    vec_t<float, VEC_SIZE> x_vec;
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE +
                     thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * stride_residual + group * d +
                        i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      // Match HC combine -> materialized residual -> Gemma norm rounding.
      float x =
          float(T(fmaf(float(input_vec[j]), gate, float(residual_vec[j]))));
      sum_sq += x * x;
      residual_vec[j] = (T)x;
      x_vec[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(combined + bx * stride_residual + group * d +
                         i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      x_vec.store(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += math::shfl_xor_sync(sum_sq, offset);
  }

  if (tx == 0) smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += math::shfl_xor_sync(sum_sq, offset);
    }
    if (tx == 0) smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    vec_t<W, VEC_SIZE> weight_vec;
    vec_t<float, VEC_SIZE> x_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + weight_offset + i * num_threads * VEC_SIZE +
                      thread_id * VEC_SIZE);
      x_vec.load(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      const float y = x_vec[j] * rms_rcp;
      input_vec[j] = fmaf(y, float(weight_vec[j]), y);
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.store(output + bx * stride_residual + group * d +
                      i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && \
     (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace flashinfer::sm70::hc
