// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>
#include "src/turbomind/kernels/gemm/transform.h"

namespace vllm::sm70_gguf {
// Aligned N32/K32 packets retain all original bits. The three metadata planes
// store original d16, scales_h16 and scales_l32, without coefficient expansion.
struct Iq4XsNativeDecoder {
  struct Parameters {
    float d;
    uint16_t scales_hi;
    uint32_t scales_lo;
  };

  __device__ static float scale(Parameters p, int group32) {
    const int small = int(((p.scales_lo >> (4 * group32)) & 15) |
                          (((p.scales_hi >> (2 * group32)) & 3) << 4)) -
                      32;
    return __fmul_rn(p.d, float(small));
  }

  __device__ static void initialize_float_book(uint8_t* book) {
    if (threadIdx.x < 16) {
      // Reuse the canonical codebook definition, without duplicating values.
      const uint32_t biased =
          turbomind::gemm::Transform_HMMA_SM70_Lut4<0>::iq_values(threadIdx.x);
      reinterpret_cast<float*>(book)[threadIdx.x] =
          float(int(biased & 255) - 128);
    }
  }

  template <class Output = half>
  __device__ static turbomind::Array<Output, 8> fragment_from_float_book(
      Parameters p, int group32, uint32_t packet, const uint8_t* book) {
    static_assert(std::is_same_v<Output, half> ||
                  std::is_same_v<Output, float>);
    const float coefficient = scale(p, group32);
    turbomind::Array<Output, 8> result;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float integer =
          reinterpret_cast<const float*>(book)[(packet >> (4 * i)) & 15];
      // Every codebook integer is exactly representable in FP16 and FP32.
      // This removes the old unity-scale Half lookup and promotion only;
      // original FP32 coefficient arithmetic and final rounding are identical.
      const float value = __fmul_rn(coefficient, integer);
      if constexpr (std::is_same_v<Output, half>)
        result[i] = __float2half_rn(value);
      else
        result[i] = value;
    }
    return result;
  }

  __device__ static Parameters parameters(const uint8_t* tile, int blocks_k,
                                          int block, int col) {
    Parameters result;
    result.d = __half2float(*reinterpret_cast<const half*>(
        tile + blocks_k * 4096 + block * 64 + col * 2));
    result.scales_hi = *reinterpret_cast<const uint16_t*>(
        tile + blocks_k * 4160 + block * 64 + col * 2);
    result.scales_lo = *reinterpret_cast<const uint32_t*>(
        tile + blocks_k * 4224 + block * 128 + col * 4);
    return result;
  }

  template <class Output = half>
  __device__ static turbomind::Array<Output, 8> fragment(Parameters p,
                                                         int group32,
                                                         uint32_t packet) {
    static_assert(std::is_same_v<Output, half> ||
                  std::is_same_v<Output, float>);
    // Existing LUT4 transform returns exact signed integers with a unity scale.
    // Reusing it keeps the codebook and nibble conversion in one
    // implementation.
    turbomind::Array<turbomind::uint4_t, 8> data[1][1];
    (uint32_t&)data[0][0] = packet;
    turbomind::Array<uint16_t, 1> stat[1][1];
    stat[0][0][0] = 0x3c00U;  // FP16 one, not an expanded source coefficient.
    turbomind::Array<half, 8> integers[1][1];
    turbomind::gemm::Transform_HMMA_SM70_Lut4<0>::apply(integers, 0, data, stat,
                                                        1);
    const float coefficient = scale(p, group32);
    turbomind::Array<Output, 8> result;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      // The canonical LUT transform emits interleaved half2 pairs
      // [q0,q4,q1,q5,q2,q6,q3,q7]. Native packets use logical [q0..q7].
      const int canonical_lane = (i & 3) * 2 + (i >> 2);
      const float value =
          __fmul_rn(coefficient, __half2float(integers[0][0][canonical_lane]));
      if constexpr (std::is_same_v<Output, half>)
        result[i] = __float2half_rn(value);
      else
        result[i] = value;
    }
    return result;
  }
};
}  // namespace vllm::sm70_gguf
