// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Uses a dedicated CustomAllreduce instance: never share its packet/epoch
// storage with auxiliary-stream collectives, or cross the owning DSO ABI.
#include "custom_all_reduce.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

namespace sm70_hc_batch {
using namespace vllm;
using Pack = packed_t<half>::P;

// Match the existing Triton HC PTX, including div.full rather than replacing
// its division with a different approximate reciprocal or changing rounding.
__device__ __forceinline__ float full_div(float a, float b) {
  float result;
  asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

__device__ __forceinline__ float sigmoid(float x) {
  float exponent, denominator;
  asm("mul.f32 %0, %1, 0fBFB8AA3B;" : "=f"(exponent) : "f"(x));
  asm("ex2.approx.f32 %0, %1;" : "=f"(exponent) : "f"(exponent));
  asm("add.f32 %0, %1, 0f3F800000;" : "=f"(denominator) : "f"(exponent));
  return full_div(1.0f, denominator);
}

template <bool Mix>
__global__ void gather_kernel(RankData buffers, const half* input,
                              const half* x, half* output, half* injection,
                              int rank, int rows) {
  constexpr int local_cols = Mix ? 640 : 88;
  constexpr int packs_per_row = local_cols / Pack::size;
  constexpr int stride = kSm70Tp4PushAllreduceBytes / sizeof(Pack);
  auto* local =
      const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[rank]));
  auto* epochs = reinterpret_cast<uint32_t*>(local);
  const unsigned epoch = epochs[blockIdx.x];
  const int base = epoch * 4 * stride;
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < rows * packs_per_row) {
    const int row = offset / packs_per_row;
    const int col = offset % packs_per_row * Pack::size;
    Pack value;
    if constexpr (Mix) {
      float accum[Pack::size] = {};
#pragma unroll
      for (int branch = 0; branch < 4; ++branch) {
        const Pack g = *reinterpret_cast<const Pack*>(input + row * 2560 +
                                                      branch * 640 + col);
        const Pack v = *reinterpret_cast<const Pack*>(
            x + row * 10240 + branch * 2560 + rank * 640 + col);
#pragma unroll
        for (int i = 0; i < Pack::size; ++i)
          accum[i] = fmaf(sigmoid(__half2float(g.data[i])),
                          __half2float(v.data[i]), accum[i]);
      }
#pragma unroll
      for (int i = 0; i < Pack::size; ++i) {
        value.data[i] = __float2half_rn(full_div(accum[i], 4.0f));
        // The unfused disjoint all-reduce adds positive zero to each rank's
        // materialized FP16 result, canonicalizing signed zero.
        if (__half2float(value.data[i]) == 0)
          value.data[i] = __float2half_rn(0);
      }
    } else {
      value = *reinterpret_cast<const Pack*>(input + row * 88 + col);
#pragma unroll
      for (int i = 0; i < Pack::size; ++i) {
        float v = __half2float(value.data[i]);
        if (v == 0) v = 0.0f;
        if (col < 80) {
          v = full_div(v, 4.0f);
          v = __fmul_rn(v, sigmoid(v));
        }
        value.data[i] = __float2half_rn(v);
      }
    }
#pragma unroll
    for (int i = 0; i < Pack::size; ++i)
      sm70_push_escape_sentinel(value.data[i]);
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      auto* destination =
          const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[peer]));
      destination += kSm70Tp4PushAllreduceSignalBytes +
                     (base + rank * stride) * sizeof(Pack);
      sm70_push_store_volatile_16b(value, destination, offset);
    }
    Pack values[4];
    while (true) {
      bool pending = false;
#pragma unroll
      for (int peer = 0; peer < 4; ++peer) {
        const void* source = local + kSm70Tp4PushAllreduceSignalBytes +
                             (base + peer * stride) * sizeof(Pack);
        sm70_push_load_volatile_16b(values[peer], source, offset);
#pragma unroll
        for (int i = 0; i < Pack::size; ++i)
          pending |= sm70_push_is_sentinel(values[peer].data[i]);
      }
      if (!pending) break;
    }
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      if constexpr (Mix) {
        *reinterpret_cast<Pack*>(output + row * 2560 + peer * 640 + col) =
            values[peer];
      } else {
        if (col < 80)
          *reinterpret_cast<Pack*>(output + row * 320 + peer * 80 + col) =
              values[peer];
        else if (peer == 3)
          *reinterpret_cast<uint2*>(injection + row * 4) =
              *reinterpret_cast<uint2*>(&values[peer]);
      }
    }
    Pack empty;
#pragma unroll
    for (int i = 0; i < Pack::size; ++i)
      *reinterpret_cast<uint16_t*>(&empty.data[i]) =
          kSm70Tp4PushAllreduceSentinel;
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      void* source = local + kSm70Tp4PushAllreduceSignalBytes +
                     (base + peer * stride) * sizeof(Pack);
      sm70_push_store_volatile_16b(empty, source, offset);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) epochs[blockIdx.x] = (epoch + 1) % 2;
}

template <bool Mix>
void run(int64_t ptr, torch::Tensor input, torch::Tensor aux,
         torch::Tensor output) {
  const c10::cuda::CUDAGuard guard(input.device());
  auto* ca = reinterpret_cast<CustomAllreduce*>(ptr);
  TORCH_CHECK(ca && ca->world_size_ == 4 && ca->fully_connected_ &&
              ca->sm70_tp4_push_buffers_registered_ &&
              custom_allreduce_current_device_is_sm70());
  for (const auto& t : {input, aux, output})
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() &&
                t.device() == input.device() && t.scalar_type() == at::kHalf &&
                t.dim() == 2);
  const int rows = input.size(0);
  TORCH_CHECK(rows > 0 && aux.size(0) == rows && output.size(0) == rows);
  TORCH_CHECK(input.size(1) == (Mix ? 2560 : 88) &&
              aux.size(1) == (Mix ? 10240 : 4) &&
              output.size(1) == (Mix ? 2560 : 320));
  const int packs = rows * (Mix ? 80 : 11);
  TORCH_CHECK(packs <= kSm70Tp4PushAllreduceBytes / sizeof(Pack));
  constexpr int threads = kSm70Tp4PushAllreduceThreads;
  const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  gather_kernel<Mix><<<(packs + threads - 1) / threads, threads, 0, stream>>>(
      ca->sm70_tp4_push_buffers_,
      reinterpret_cast<const half*>(input.data_ptr()),
      Mix ? reinterpret_cast<const half*>(aux.data_ptr()) : nullptr,
      reinterpret_cast<half*>(output.data_ptr()),
      Mix ? nullptr : reinterpret_cast<half*>(aux.data_ptr()), ca->rank_, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace sm70_hc_batch
