// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace flash_v100 {

// Addressing is borrowed from the cache view, independent of scale placement.
struct KVWritableScalePlane {
  float* data = nullptr;
  int64_t block_stride = 0;
  int64_t token_stride = 0;
  int64_t head_stride = 0;

  __device__ __forceinline__ void store(int64_t block, int token, int head,
                                        float scale) const {
    data[block * block_stride + token * token_stride + head * head_stride] =
        scale;
  }
};

template <bool ROUND_EVEN>
struct KVInt8RowEncoder {
  template <typename T>
  __device__ __forceinline__ static float reduce_scale(const T* first,
                                                       int first_width,
                                                       const T* second,
                                                       int second_width) {
    // Whole-row writers use a CTA; per-head writers retain their warp owner.
    __shared__ float warp_max[32];
    __shared__ int warp_bad[32];
    __shared__ float result;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float amax = 0.f;
    bool invalid = false;
    for (int dim = threadIdx.x; dim < first_width + second_width;
         dim += blockDim.x) {
      const float value = dim < first_width
                              ? static_cast<float>(first[dim])
                              : static_cast<float>(second[dim - first_width]);
      amax = fmaxf(amax, fabsf(value));
      invalid |= !isfinite(value);
    }
    for (int delta = 16; delta; delta >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, delta));
    invalid = __any_sync(0xffffffffu, invalid);
    if (lane == 0) {
      warp_max[warp] = amax;
      warp_bad[warp] = invalid;
    }
    __syncthreads();
    if (warp == 0) {
      amax = lane < blockDim.x / 32 ? warp_max[lane] : 0.f;
      invalid = lane < blockDim.x / 32 ? warp_bad[lane] : false;
      for (int delta = 16; delta; delta >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, delta));
      invalid = __any_sync(0xffffffffu, invalid);
      if (lane == 0) result = scale(amax, invalid);
    }
    __syncthreads();
    return result;
  }

  __device__ __forceinline__ static float legacy_divide(float x, float y) {
    // Preserve Triton's legacy div.full operation before the truncating cast.
    // RN division/reciprocal can change the byte at an integer boundary.
    float result;
    asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(x), "f"(y));
    return result;
  }

  __device__ __forceinline__ static float scale(float absolute_max,
                                                bool invalid) {
    if (invalid) return __int_as_float(0x7fffffff);
    const float scale = ROUND_EVEN ? __fdiv_rn(absolute_max, 127.f)
                                   : legacy_divide(absolute_max, 127.f);
    return fmaxf(scale, 1e-6f);
  }

  __device__ __forceinline__ static int8_t encode(float value, float scale) {
    float normalized;
    if constexpr (ROUND_EVEN) {
      normalized = __fdiv_rn(value, scale);
    } else {
      normalized = __fmul_rn(value, legacy_divide(1.f, scale));
    }
    if (!isfinite(normalized)) return 0;
    normalized = fminf(fmaxf(normalized, -128.f), 127.f);
    return static_cast<int8_t>(ROUND_EVEN ? __float2int_rn(normalized)
                                          : __float2int_rz(normalized));
  }
};

}  // namespace flash_v100
