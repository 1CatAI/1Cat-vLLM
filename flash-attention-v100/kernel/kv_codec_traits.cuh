// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, D.Skryabin
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "kv_codec.cuh"

namespace flash_v100 {

// Describes payload storage only. Scale placement and accumulator precision
// remain attention-operator contracts; these formats use separate K/V scalars.
template <int KV_DTYPE>
struct KVStorageTraits;

template <>
struct KVStorageTraits<KV_CACHE_DTYPE_FP16> {
  using storage_type = __half;
  static constexpr bool quantized = false;
};

template <>
struct KVStorageTraits<KV_CACHE_DTYPE_FP8_E4M3> {
  using storage_type = uint8_t;
  static constexpr bool quantized = true;
};

template <>
struct KVStorageTraits<KV_CACHE_DTYPE_FP8_E5M2> {
  using storage_type = uint8_t;
  static constexpr bool quantized = true;
};

// One format contract supplies scalar and packed-vector readers to every
// attention family. Keep KVReader available for existing native callers.
template <int KV_DTYPE, bool E4M3_BITS = false>
struct KVCodecTraits : KVStorageTraits<KV_DTYPE>,
                       KVReader<KV_DTYPE, E4M3_BITS> {
  static constexpr int vector_elements = 8;
  static constexpr int vector_bytes =
      vector_elements * KVReader<KV_DTYPE, E4M3_BITS>::element_bytes;
  static_assert(sizeof(typename KVStorageTraits<KV_DTYPE>::storage_type) ==
                    KVReader<KV_DTYPE, E4M3_BITS>::element_bytes,
                "KV storage type must match reader addressing");
};

}  // namespace flash_v100
