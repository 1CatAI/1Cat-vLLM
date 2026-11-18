#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace flash_v100 {

constexpr int KV_CACHE_DTYPE_FP16 = 0;
constexpr int KV_CACHE_DTYPE_FP8_E4M3 = 1;
constexpr int KV_CACHE_DTYPE_FP8_E5M2 = 2;
constexpr int KV_CACHE_DTYPE_INT8_PTH = 3;

// Exact zero-LUT signed int8 -> half2 conversion on SM70+.
// In IEEE-754 binary16, exponent 25 (0x6400 = 1024.0) has ULP = 1.0, so
// (0x6400 | (uint8_t(b) ^ 0x80)) equals 1024.0 + (b + 128) = b + 1152.0
// without rounding for all b in [-128, 127].
__device__ __forceinline__ __half2 int8_pair_to_half2_unscaled(uint32_t word,
                                                               uint32_t sel) {
  const uint32_t biased = __byte_perm(word, 0x64646464u, sel) ^ 0x00800080u;
  const uint32_t offset = 0x64806480u;  // half2(1152.0, 1152.0)
  return __hsub2(*reinterpret_cast<const __half2*>(&biased),
                 *reinterpret_cast<const __half2*>(&offset));
}

__device__ __forceinline__ uint4 int8_vector_to_half8_unscaled(uint64_t raw) {
  const uint32_t lo = static_cast<uint32_t>(raw);
  const uint32_t hi = static_cast<uint32_t>(raw >> 32);
  union {
    uint4 u;
    __half2 h[4];
  } out;
  out.h[0] = int8_pair_to_half2_unscaled(lo, 0x5140u);
  out.h[1] = int8_pair_to_half2_unscaled(lo, 0x5342u);
  out.h[2] = int8_pair_to_half2_unscaled(hi, 0x5140u);
  out.h[3] = int8_pair_to_half2_unscaled(hi, 0x5342u);
  return out.u;
}

__device__ __forceinline__ uint4 int8_vector_to_half8_scaled(uint64_t raw,
                                                             __half2 scale2) {
  const uint32_t lo = static_cast<uint32_t>(raw);
  const uint32_t hi = static_cast<uint32_t>(raw >> 32);
  union {
    uint4 u;
    __half2 h[4];
  } out;
  out.h[0] = __hmul2(int8_pair_to_half2_unscaled(lo, 0x5140u), scale2);
  out.h[1] = __hmul2(int8_pair_to_half2_unscaled(lo, 0x5342u), scale2);
  out.h[2] = __hmul2(int8_pair_to_half2_unscaled(hi, 0x5140u), scale2);
  out.h[3] = __hmul2(int8_pair_to_half2_unscaled(hi, 0x5342u), scale2);
  return out.u;
}

__device__ __forceinline__ float quiet_nan_f() {
  return __int_as_float(0x7fffffff);
}

__device__ __forceinline__ float inf_f() { return __int_as_float(0x7f800000); }

__device__ __forceinline__ float exp2_int(const int exponent) {
  return __uint_as_float(static_cast<uint32_t>(exponent + 127) << 23);
}

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t raw) {
  const int sign = raw >> 7;
  const int exp = (raw >> 3) & 0x0f;
  const int mant = raw & 0x07;

  if ((raw & 0x7f) == 0) {
    return sign ? -0.0f : 0.0f;
  }

  float value;
  if (exp == 0) {
    value = static_cast<float>(mant) * 0.001953125f;  // 2^-9
  } else {
    if (exp == 0x0f && mant == 0x07) {
      return quiet_nan_f();
    }
    value = (1.0f + static_cast<float>(mant) * 0.125f) * exp2_int(exp - 7);
  }
  return sign ? -value : value;
}

__device__ __forceinline__ __half fp8_e5m2_to_half(uint8_t raw) {
  // E5M2 and IEEE fp16 use the same sign bit, exponent width, and exponent
  // bias. Expanding the two mantissa bits into the high fp16 mantissa bits is
  // exact for normals, subnormals, zero, inf, and NaN.
  return __ushort_as_half(static_cast<unsigned short>(raw) << 8);
}

__device__ __forceinline__ __half2 fp8_e5m2_pair_to_half2(uint16_t raw_pair) {
  const uint32_t half2_bits = (static_cast<uint32_t>(raw_pair & 0x00ffu) << 8) |
                              (static_cast<uint32_t>(raw_pair & 0xff00u) << 16);
  union {
    uint32_t u;
    __half2 h2;
  } converter;
  converter.u = half2_bits;
  return converter.h2;
}

__device__ __forceinline__ __half2 load_fp8_e5m2_half2_unscaled(
    const void* __restrict__ cache, const int64_t byte_index) {
  const uint16_t* cache_u16 = reinterpret_cast<const uint16_t*>(cache);
  return fp8_e5m2_pair_to_half2(cache_u16[byte_index >> 1]);
}

__device__ __forceinline__ float fp8_e5m2_to_float(uint8_t raw) {
  return __half2float(fp8_e5m2_to_half(raw));
}

template <int KV_DTYPE>
__device__ __forceinline__ float load_kv_cache_float_unscaled(
    const void* __restrict__ cache, const int64_t index) {
  if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP16) {
    const __half* cache_h = reinterpret_cast<const __half*>(cache);
    return __half2float(cache_h[index]);
  } else {
    const uint8_t* cache_u8 = reinterpret_cast<const uint8_t*>(cache);
    const uint8_t raw = cache_u8[index];
    const float value = KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3
                            ? fp8_e4m3fn_to_float(raw)
                            : fp8_e5m2_to_float(raw);
    return value;
  }
}

template <int KV_DTYPE>
__device__ __forceinline__ float load_kv_cache_float(
    const void* __restrict__ cache, const int64_t index, const float scale) {
  return load_kv_cache_float_unscaled<KV_DTYPE>(cache, index) * scale;
}

template <int KV_DTYPE>
__device__ __forceinline__ __half load_kv_cache_half(
    const void* __restrict__ cache, const int64_t index, const float scale) {
  if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP8_E5M2) {
    const uint8_t* cache_u8 = reinterpret_cast<const uint8_t*>(cache);
    return __float2half_rn(__half2float(fp8_e5m2_to_half(cache_u8[index])) *
                           scale);
  }
  return __float2half_rn(load_kv_cache_float<KV_DTYPE>(cache, index, scale));
}

}  // namespace flash_v100
