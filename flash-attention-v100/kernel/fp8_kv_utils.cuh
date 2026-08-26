#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace flash_v100 {

constexpr int KV_CACHE_DTYPE_FP16 = 0;
constexpr int KV_CACHE_DTYPE_FP8_E4M3 = 1;
constexpr int KV_CACHE_DTYPE_FP8_E5M2 = 2;

__device__ __forceinline__ float quiet_nan_f() {
  return __int_as_float(0x7fffffff);
}

__device__ __forceinline__ float inf_f() {
  return __int_as_float(0x7f800000);
}

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
    value = (1.0f + static_cast<float>(mant) * 0.125f) *
            exp2_int(exp - 7);
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
  const uint32_t half2_bits =
      (static_cast<uint32_t>(raw_pair & 0x00ffu) << 8) |
      (static_cast<uint32_t>(raw_pair & 0xff00u) << 16);
  union {
    uint32_t u;
    __half2 h2;
  } converter;
  converter.u = half2_bits;
  return converter.h2;
}

__device__ __forceinline__ __half2 load_fp8_e5m2_half2_unscaled(
    const void* __restrict__ cache,
    const int64_t byte_index) {
  const uint16_t* cache_u16 = reinterpret_cast<const uint16_t*>(cache);
  return fp8_e5m2_pair_to_half2(cache_u16[byte_index >> 1]);
}

__device__ __forceinline__ float fp8_e5m2_to_float(uint8_t raw) {
  return __half2float(fp8_e5m2_to_half(raw));
}

template<int KV_DTYPE>
__device__ __forceinline__ float load_kv_cache_float_unscaled(
    const void* __restrict__ cache,
    const int64_t index) {
  if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP16) {
    const __half* cache_h = reinterpret_cast<const __half*>(cache);
    return __half2float(cache_h[index]);
  } else {
    const uint8_t* cache_u8 = reinterpret_cast<const uint8_t*>(cache);
    const uint8_t raw = cache_u8[index];
    const float value =
        KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3 ? fp8_e4m3fn_to_float(raw)
                                            : fp8_e5m2_to_float(raw);
    return value;
  }
}

template<int KV_DTYPE>
__device__ __forceinline__ float load_kv_cache_float(
    const void* __restrict__ cache,
    const int64_t index,
    const float scale) {
  return load_kv_cache_float_unscaled<KV_DTYPE>(cache, index) * scale;
}

template<int KV_DTYPE>
__device__ __forceinline__ __half load_kv_cache_half(
    const void* __restrict__ cache,
    const int64_t index,
    const float scale) {
  if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP8_E5M2) {
    const uint8_t* cache_u8 = reinterpret_cast<const uint8_t*>(cache);
    return __float2half_rn(__half2float(fp8_e5m2_to_half(cache_u8[index])) *
                           scale);
  }
  return __float2half_rn(load_kv_cache_float<KV_DTYPE>(cache, index, scale));
}

}  // namespace flash_v100
