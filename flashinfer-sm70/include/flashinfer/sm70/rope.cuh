// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_fp16.h>

#include <cstdint>

namespace flashinfer::sm70 {
template <typename Position>
__global__ void rope_pos_ids(half* out, const half* x, const Position* positions, int64_t count,
                             int64_t heads, int64_t dim, int64_t sx0, int64_t sx1, int64_t so0,
                             int64_t so1, int rotary_dim, bool interleave, float scale,
                             float theta) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int d = index % dim;
  const int64_t head = index / dim % heads, row = index / (dim * heads);
  const half* input = x + row * sx0 + head * sx1;
  half* output = out + row * so0 + head * so1;
  if (d >= rotary_dim) {
    output[d] = input[d];
    return;
  }
  // A single thread reads and writes both members so the in-place API has no
  // read/write race, including independent warps and partial rotary dimensions.
  const int pairs = rotary_dim / 2;
  if (d >= pairs) return;
  const int first = interleave ? 2 * d : d;
  const int second = interleave ? 2 * d + 1 : d + pairs;
  const float angle =
      static_cast<float>(positions[row]) / scale * powf(theta, -static_cast<float>(d) / pairs);
  float s, c;
  sincosf(angle, &s, &c);
  const float a = __half2float(input[first]), b = __half2float(input[second]);
  output[first] = __float2half_rn(a * c - b * s);
  output[second] = __float2half_rn(b * c + a * s);
}
}  // namespace flashinfer::sm70
