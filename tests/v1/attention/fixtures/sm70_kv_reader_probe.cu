// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Compile both versions with identical entry names and runtime inputs. No GPU
// execution: PTX parity checks compiler transformations, not measured outputs.
#ifdef LEGACY_READER
  #include "sm70_kv_reader_legacy.cuh"
#else
  #include "kv_codec_traits.cuh"
  #include <type_traits>
static_assert(
    std::is_same<flash_v100::KVCodecTraits<0>::storage_type, __half>::value);
static_assert(
    std::is_same<flash_v100::KVCodecTraits<1>::storage_type, uint8_t>::value);
static_assert(
    std::is_same<flash_v100::KVCodecTraits<2>::storage_type, uint8_t>::value);
static_assert(flash_v100::KVCodecTraits<0>::vector_bytes == 16);
static_assert(flash_v100::KVCodecTraits<1>::vector_bytes == 8);
static_assert(flash_v100::KVCodecTraits<2>::vector_bytes == 8);
static_assert(!flash_v100::KVCodecTraits<0>::quantized);
static_assert(flash_v100::KVCodecTraits<1>::quantized);
static_assert(flash_v100::KVCodecTraits<2>::quantized);
#endif

template <int CODEC, bool BITS>
__device__ __forceinline__ void scalar_reader(const void* cache, int64_t index,
                                              float scale, float* out,
                                              __half* half_out) {
#ifdef LEGACY_READER
  out[0] = flash_v100::load_kv_cache_float_unscaled<CODEC, BITS>(cache, index);
  out[1] = flash_v100::load_kv_cache_float<CODEC>(cache, index, scale);
  half_out[0] = flash_v100::load_kv_cache_half<CODEC>(cache, index, scale);
#else
  out[0] = flash_v100::KVCodecTraits<CODEC, BITS>::unscaled(cache, index);
  out[1] = flash_v100::KVCodecTraits<CODEC>::scaled(cache, index, scale);
  half_out[0] = flash_v100::KVCodecTraits<CODEC>::half(cache, index, scale);
#endif
}

extern "C" __global__ void scalar(const void* cache, int64_t index, float scale,
                                  float* out, __half* half_out) {
  scalar_reader<0, false>(cache, index, scale, out, half_out);
  scalar_reader<1, false>(cache, index, scale, out + 2, half_out + 1);
  scalar_reader<1, true>(cache, index, scale, out + 4, half_out + 2);
  scalar_reader<2, false>(cache, index, scale, out + 6, half_out + 3);
}

extern "C" __global__ void packed(uint64_t raw, const uint16_t* lut,
                                  uint4* out) {
#ifdef LEGACY_READER
  out[0] = flash_v100::fp8_e4m3fn_vector_to_half8(raw);
  out[1] = flash_v100::fp8_e4m3fn_vector_to_half8_fast(raw);
  out[2] = flash_v100::fp8_e4m3fn_vector_to_half8_lut(raw, lut);
  out[3] = flash_v100::fp8_e5m2_vector_to_half8(raw);
  out[4] = flash_v100::fp8_e5m2_vector_to_half8(raw);
  out[5] = flash_v100::fp8_e4m3fn_vector_to_half8_lut(raw, lut);
#else
  out[0] = flash_v100::KVCodecTraits<1>::half8_from_packed<>(raw);
  out[1] = flash_v100::KVCodecTraits<1>::half8_from_packed<true>(raw);
  out[2] =
      flash_v100::KVCodecTraits<1>::half8_from_packed<false, true>(raw, lut);
  out[3] = flash_v100::KVCodecTraits<2>::half8_from_packed<>(raw);
  out[4] =
      flash_v100::KVCodecTraits<2>::half8_from_packed<true, true>(raw, lut);
  out[5] =
      flash_v100::KVCodecTraits<1>::half8_from_packed<true, true>(raw, lut);
#endif
}

extern "C" __global__ void vector_load(const void* cache, int64_t offset,
                                       int col, const uint16_t* lut,
                                       uint4* out) {
#ifdef LEGACY_READER
  out[0] = flash_v100::legacy_load_half8<0>(cache, offset, col);
  out[1] = flash_v100::legacy_load_half8<1>(cache, offset, col);
  out[2] = flash_v100::legacy_load_half8<1, true>(cache, offset, col, lut);
  out[3] = flash_v100::legacy_load_half8<2>(cache, offset, col);
#else
  out[0] = flash_v100::KVCodecTraits<0>::load_half8<>(cache, offset, col);
  out[1] = flash_v100::KVCodecTraits<1>::load_half8<>(cache, offset, col);
  out[2] =
      flash_v100::KVCodecTraits<1>::load_half8<true>(cache, offset, col, lut);
  out[3] = flash_v100::KVCodecTraits<2>::load_half8<>(cache, offset, col);
#endif
}
