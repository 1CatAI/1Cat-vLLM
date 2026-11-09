// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Register-list merging is inspired by NVIDIA FlashInfer's Apache-2.0
// moeTopKFuncs.cuh at c1af49ecbad5433d4332aaec986d1a3bd1e1e610.
#pragma once
#include <cuda_fp16.h>
#include "gguf_q8_1.cuh"

namespace vllm::sm70_router {
__device__ __forceinline__ unsigned key(half h, int id) {
  unsigned b = __half_as_ushort(h);
  if ((b & 0x7fff) == 0) b = 0;
  unsigned ordered = (b & 0x8000) ? (~b & 0xffff) : (b ^ 0x8000);
  return (ordered << 9) | (511 - id);
}
__device__ __forceinline__ unsigned warp_max(unsigned value) {
#pragma unroll
  for (int d = 16; d; d >>= 1)
    value = max(value, __shfl_xor_sync(0xffffffff, value, d));
  return value;
}
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int d = 16; d; d >>= 1) value += __shfl_xor_sync(0xffffffff, value, d);
  return value;
}
__device__ __forceinline__ float unpack(unsigned packed) {
  const unsigned ordered = packed >> 9;
  return __half2float(__ushort_as_half(
      (ordered & 0x8000) ? (ordered ^ 0x8000) : (~ordered & 0xffff)));
}
__device__ __forceinline__ void select_row(const half* logits, float* weights,
                                           int* ids, int* source, int m,
                                           int row) {
  const int lane = threadIdx.x & 31;
  unsigned keys[16];
  bool invalid = false, finite_max = false;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const half h = logits[i * 32 + lane];
    unsigned b = __half_as_ushort(h);
    invalid |= (b & 0x7fff) > 0x7c00 || b == 0x7c00;
    finite_max |= b != 0xfc00;
    keys[i] = key(h, i * 32 + lane);
  }
  invalid =
      __any_sync(0xffffffff, invalid) || !__any_sync(0xffffffff, finite_max);
  if (invalid) {
    if (lane < 10) {
      weights[lane] = 0;
      ids[lane] = lane;
      source[lane] = lane * m + row;
    }
    return;
  }
  // Sort only each lane's sixteen keys, then merge the 32 list heads.
#pragma unroll
  for (int width = 2; width <= 16; width *= 2) {
#pragma unroll
    for (int distance = width / 2; distance; distance /= 2) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int other = j ^ distance;
        if (other > j) {
          const unsigned a = keys[j], b = keys[other];
          const bool descending = (j & width) == 0;
          keys[j] = descending ? max(a, b) : min(a, b);
          keys[other] = descending ? min(a, b) : max(a, b);
        }
      }
    }
  }
  unsigned own_winner = 0;
  float largest = 0;
#pragma unroll
  for (int rank = 0; rank < 10; ++rank) {
    unsigned head = keys[0];
    const unsigned winner = warp_max(head);
    if (rank == 0) largest = unpack(winner);
    if (lane == rank) own_winner = winner;
    const bool pop = head == winner;
#pragma unroll
    for (int j = 0; j < 15; ++j) keys[j] = pop ? keys[j + 1] : keys[j];
    keys[15] = pop ? 0 : keys[15];
  }
  float exponential =
      lane < 10 ? exp2f((unpack(own_winner) - largest) * 1.4426950408889634f)
                : 0;
  const float denominator = warp_sum(exponential);
  if (lane < 10) {
    weights[lane] = exponential / denominator;
    ids[lane] = 511 - (own_winner & 511);
    source[lane] = lane * m + row;
  }
}
__global__ void select_kernel(const half* logits, float* weights, int* ids,
                              int* source, int m) {
  const int row = blockIdx.x;
  select_row(logits + row * 512, weights + row * 10, ids + row * 10,
             source + row * 10, m, row);
}

// Preserve the quantizer's independent 256-column blocks. In the first
// block of each row, one warp also produces routing results. Both outputs
// become ready at the same kernel boundary; no cross-CTA handoff is needed.
__global__ void select_quantize_kernel(const half* logits, const half* x,
                                       float* weights, int* ids, int* source,
                                       vllm::sm70_gguf::Q8_1* q8, int m) {
  const int row = blockIdx.y, lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (blockIdx.x == 0 && warp == 0)
    select_row(logits + row * 512, weights + row * 10, ids + row * 10,
               source + row * 10, m, row);
  const int packet = row * 80 + blockIdx.x * 8 + warp;
  const int k = blockIdx.x * 256 + warp * 32 + lane;
  vllm::sm70_gguf::quantize_q8_1_warp(q8 + packet,
                                      __half2float(x[row * 2560 + k]));
}

}  // namespace vllm::sm70_router
