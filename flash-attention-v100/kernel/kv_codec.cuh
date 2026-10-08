// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, D.Skryabin
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Shared SM70 KV readers. Storage conversion belongs here; attention scheduling
// stays in the caller. Writers are migrated separately after their parity gate.
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

// Finite E4M3 values map exactly to FP16 bits followed by FP32 scaling.
// Preserve the original NaN payload and signed zeros, including subnormals.
__device__ __forceinline__ float fp8_e4m3fn_to_float_bits(uint8_t raw) {
  const uint16_t half_bits = ((static_cast<uint16_t>(raw) << 7) & 0x3f80u) |
                             ((static_cast<uint16_t>(raw) << 8) & 0x8000u);
  const float value = __half2float(__ushort_as_half(half_bits)) * 256.0f;
  return (raw & 0x7fu) == 0x7fu ? quiet_nan_f() : value;
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

__device__ __forceinline__ uint32_t
fp8_e5m2_pair_to_half2_bits(const uint16_t raw_pair) {
  return (static_cast<uint32_t>(raw_pair & 0x00ffu) << 8) |
         (static_cast<uint32_t>(raw_pair & 0xff00u) << 16);
}

__device__ __forceinline__ uint4 fp8_e5m2_vector_to_half8(const uint64_t raw) {
  return make_uint4(
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 16)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 32)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 48)));
}

__device__ __forceinline__ uint16_t fp8_e4m3fn_to_half_bits(const uint8_t raw) {
  const uint16_t sign = static_cast<uint16_t>(raw & 0x80u) << 8;
  const uint8_t magnitude = raw & 0x7fu;
  const uint8_t exponent = magnitude >> 3;
  const uint8_t mantissa = magnitude & 0x07u;
  if (magnitude == 0) {
    return sign;
  }
  if (exponent == 0) {
    // E4M3 subnormals are exact fp16 normals: mantissa * 2^-9.
    const uint16_t magnitude_bits =
        mantissa < 2
            ? 0x1800u
            : (mantissa < 4
                   ? static_cast<uint16_t>(0x1c00u | ((mantissa - 2) << 9))
                   : static_cast<uint16_t>(0x2000u | ((mantissa - 4) << 8)));
    return sign | magnitude_bits;
  }
  if (magnitude == 0x7fu) {
    return sign | 0x7e00u;
  }
  return sign | static_cast<uint16_t>((exponent + 8) << 10) |
         static_cast<uint16_t>(mantissa << 7);
}

__device__ __forceinline__ uint32_t
fp8_e4m3fn_pair_to_half2_bits(const uint16_t raw_pair) {
  return static_cast<uint32_t>(
             fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(raw_pair))) |
         (static_cast<uint32_t>(
              fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(raw_pair >> 8)))
          << 16);
}

__device__ __forceinline__ uint32_t
fp8_e4m3fn_pair_to_half2_bits_fast(const uint16_t raw_pair) {
  const uint8_t raw0 = static_cast<uint8_t>(raw_pair);
  const uint8_t raw1 = static_cast<uint8_t>(raw_pair >> 8);
  if ((raw0 & 0x7fu) == 0x7fu || (raw1 & 0x7fu) == 0x7fu) {
    return fp8_e4m3fn_pair_to_half2_bits(raw_pair);
  }

  // Moving a finite E4M3 encoding into the corresponding fp16 sign,
  // exponent, and mantissa fields represents exactly value / 256. A packed
  // half2 multiply restores both values without per-byte exponent branches.
  const uint32_t expanded = (static_cast<uint32_t>(raw_pair & 0x0080u) << 8) |
                            (static_cast<uint32_t>(raw_pair & 0x007fu) << 7) |
                            (static_cast<uint32_t>(raw_pair & 0x8000u) << 16) |
                            (static_cast<uint32_t>(raw_pair & 0x7f00u) << 15);
  union {
    uint32_t u;
    __half2 h2;
  } converter;
  converter.u = expanded;
  converter.h2 = __hmul2(converter.h2, __float2half2_rn(256.0f));
  return converter.u;
}

__device__ __forceinline__ uint4
fp8_e4m3fn_vector_to_half8(const uint64_t raw) {
  return make_uint4(
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw)),
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw >> 16)),
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw >> 32)),
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw >> 48)));
}

__device__ __forceinline__ uint4
fp8_e4m3fn_vector_to_half8_fast(const uint64_t raw) {
  return make_uint4(
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw)),
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw >> 16)),
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw >> 32)),
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw >> 48)));
}

__device__ __forceinline__ uint4 fp8_e4m3fn_vector_to_half8_lut(
    const uint64_t raw, const uint16_t* __restrict__ lut) {
  return make_uint4(
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 8)]) << 16),
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 16)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 24)]) << 16),
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 32)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 40)]) << 16),
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 48)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 56)]) << 16));
}

template <int KV_DTYPE, bool E4M3_BITS = false>
struct KVReader {
  static_assert(KV_DTYPE == KV_CACHE_DTYPE_FP16 ||
                    KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3 ||
                    KV_DTYPE == KV_CACHE_DTYPE_FP8_E5M2,
                "KV format has no Flash-V100 reader");
  static constexpr int element_bytes = KV_DTYPE == KV_CACHE_DTYPE_FP16 ? 2 : 1;

  // Eight unscaled half values. Scale placement/rounding remains the existing
  // caller's contract; per-token codecs need their own staged tile policy.
  template <bool PACKED_FAST = false, bool SHARED_LUT = false>
  __device__ __forceinline__ static uint4 half8_from_packed(
      const uint64_t raw, const uint16_t* __restrict__ lut = nullptr) {
    static_assert(KV_DTYPE != KV_CACHE_DTYPE_FP16,
                  "Eight FP16 values require sixteen payload bytes");
    if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3) {
      if constexpr (SHARED_LUT) {
        return fp8_e4m3fn_vector_to_half8_lut(raw, lut);
      } else if constexpr (PACKED_FAST) {
        return fp8_e4m3fn_vector_to_half8_fast(raw);
      } else {
        return fp8_e4m3fn_vector_to_half8(raw);
      }
    } else {
      // Packed panels historically allow an unused LUT hint for E5M2.
      return fp8_e5m2_vector_to_half8(raw);
    }
  }

  // Keep base and vector column separate: old formats divide the base by
  // eight, while a future inline-scale reader must retain the byte offset.
  template <bool SHARED_LUT = false>
  __device__ __forceinline__ static uint4 load_half8(
      const void* __restrict__ cache, const int64_t physical_offset,
      const int vec_col, const uint16_t* __restrict__ lut = nullptr) {
    if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP16) {
      const uint4* cache_vec = reinterpret_cast<const uint4*>(cache);
      return __ldg(&cache_vec[physical_offset / 8 + vec_col]);
    } else {
      static_assert(KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3 || !SHARED_LUT,
                    "The E4M3 conversion LUT requires E4M3 KV");
      const uint64_t* cache_vec = reinterpret_cast<const uint64_t*>(cache);
      const uint64_t raw = __ldg(&cache_vec[physical_offset / 8 + vec_col]);
      return half8_from_packed<false, SHARED_LUT>(raw, lut);
    }
  }

  __device__ __forceinline__ static float unscaled(
      const void* __restrict__ cache, const int64_t index) {
    if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP16) {
      const __half* cache_h = reinterpret_cast<const __half*>(cache);
      return __half2float(cache_h[index]);
    } else {
      const uint8_t* cache_u8 = reinterpret_cast<const uint8_t*>(cache);
      const uint8_t raw = cache_u8[index];
      if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3 && E4M3_BITS) {
        return fp8_e4m3fn_to_float_bits(raw);
      }
      const float value = KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3
                              ? fp8_e4m3fn_to_float(raw)
                              : fp8_e5m2_to_float(raw);
      return value;
    }
  }

  __device__ __forceinline__ static float scaled(const void* __restrict__ cache,
                                                 const int64_t index,
                                                 const float scale) {
    return unscaled(cache, index) * scale;
  }

  __device__ __forceinline__ static __half half(const void* __restrict__ cache,
                                                const int64_t index,
                                                const float scale) {
    if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP8_E5M2) {
      const uint8_t* cache_u8 = reinterpret_cast<const uint8_t*>(cache);
      return __float2half_rn(__half2float(fp8_e5m2_to_half(cache_u8[index])) *
                             scale);
    }
    return __float2half_rn(scaled(cache, index, scale));
  }
};

// Source-compatible entry points keep existing attention launch ABIs unchanged.
template <int KV_DTYPE, bool E4M3_BITS = false>
__device__ __forceinline__ float load_kv_cache_float_unscaled(
    const void* __restrict__ cache, const int64_t index) {
  return KVReader<KV_DTYPE, E4M3_BITS>::unscaled(cache, index);
}

template <int KV_DTYPE>
__device__ __forceinline__ float load_kv_cache_float(
    const void* __restrict__ cache, const int64_t index, const float scale) {
  return KVReader<KV_DTYPE>::scaled(cache, index, scale);
}

template <int KV_DTYPE>
__device__ __forceinline__ __half load_kv_cache_half(
    const void* __restrict__ cache, const int64_t index, const float scale) {
  return KVReader<KV_DTYPE>::half(cache, index, scale);
}

}  // namespace flash_v100
