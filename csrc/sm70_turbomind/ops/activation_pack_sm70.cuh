// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace vllm::sm70 {

// [M, K] -> [K/16, M, 16]. This packs only activations; quantized weights
// and their numerics remain unchanged. Callers admit contiguous K%16==0.
static __global__ void pack_k16_input(const half* __restrict__ input,
                                      half* __restrict__ packed, int m, int k) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m * k / 2) return;
  const int pair = index & 7;
  const int row = (index >> 3) % m;
  const int group = (index >> 3) / m;
  reinterpret_cast<half2*>(packed)[index] = reinterpret_cast<const half2*>(
      input)[static_cast<size_t>(row) * k / 2 + group * 8 + pair];
}

}  // namespace vllm::sm70
