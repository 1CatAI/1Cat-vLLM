// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#ifdef USE_ROCM
  #include "quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

// Existing calibrated scalar writers; page addressing belongs to the caller.
// Dynamic token/group scales and the Flash-V100 reader boundary are separate
// migration steps. Preserve the original conversion and saturation contracts.
namespace vllm {

// Used to copy/convert one element
template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct KVWriter {
  float scale;

  __device__ __forceinline__ void operator()(OutT& dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);
    } else {
      dst = fp8::scaled_convert<OutT, InT, kv_dt>(src, scale);
    }
  }
};

__device__ __forceinline__ uint8_t
fp16_bits_to_e5m2_satfinite_rn(const uint16_t bits) {
  const uint16_t sign = (bits >> 8) & 0x80u;
  const uint16_t magnitude = bits & 0x7fffu;
  const uint16_t exponent = magnitude >> 10;
  const uint16_t mantissa = magnitude & 0x03ffu;

  if (exponent == 0x1fu) {
    // Match CUDA's __NV_SATFINITE behavior: infinities clamp to max finite,
    // while every NaN payload and sign canonicalizes to positive E5M2 NaN.
    return mantissa == 0 ? static_cast<uint8_t>(sign | 0x7bu) : 0x7fu;
  }

  uint16_t rounded = magnitude >> 8;
  const uint16_t remainder = magnitude & 0x00ffu;
  rounded += remainder > 0x80u || (remainder == 0x80u && (rounded & 1u) != 0u);
  rounded = rounded > 0x7bu ? 0x7bu : rounded;
  return static_cast<uint8_t>(sign | rounded);
}

struct CopyFp16ToE5M2UnitScaleOp {
  __device__ __forceinline__ void operator()(uint8_t& dst,
                                             const uint16_t src) const {
    dst = fp16_bits_to_e5m2_satfinite_rn(src);
  }
};

}  // namespace vllm
