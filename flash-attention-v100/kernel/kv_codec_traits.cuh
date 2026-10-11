// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, D.Skryabin
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "kv_codec.cuh"

namespace flash_v100 {

// Describes payload storage only. Scale placement and accumulator precision
// remain explicit codec and attention-operator contracts.
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

template <>
struct KVStorageTraits<KV_CACHE_DTYPE_INT8> {
  using storage_type = int8_t;
  static constexpr bool quantized = true;
};

// Borrowed scale addresses use element strides from the public cache view.
// No algorithm infers an inline/planar layout from a payload dtype.
struct KVScalePlane {
  const float* data = nullptr;
  int64_t block_stride = 0;
  int64_t token_stride = 0;
  int64_t head_stride = 0;

  __device__ __forceinline__ float load(int block, int token, int head) const {
    return data[static_cast<int64_t>(block) * block_stride +
                static_cast<int64_t>(token) * token_stride +
                static_cast<int64_t>(head) * head_stride];
  }
};

struct KVScaleView {
  KVScalePlane key;
  KVScalePlane value;
};

// One format contract supplies scalar and packed-vector readers to every
// attention family. Keep KVReader available for existing native callers.
template <int KV_DTYPE, bool E4M3_BITS = false>
struct KVCodecTraits : KVStorageTraits<KV_DTYPE>,
                       KVReader<KV_DTYPE, E4M3_BITS> {
  static constexpr int vector_elements = 8;
  static constexpr bool per_token_head = KV_DTYPE == KV_CACHE_DTYPE_INT8;

  __device__ __forceinline__ static float score_scale(float softmax_scale,
                                                      float tensor_scale,
                                                      const KVScalePlane& rows,
                                                      int block, int token,
                                                      int head) {
    if constexpr (per_token_head) {
      return softmax_scale * rows.load(block, token, head);
    } else if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP16) {
      return softmax_scale;
    } else {
      return softmax_scale * tensor_scale;
    }
  }

  __device__ __forceinline__ static float value_weight(float weight,
                                                       const KVScalePlane& rows,
                                                       int block, int token,
                                                       int head) {
    if constexpr (per_token_head) {
      return weight * rows.load(block, token, head);
    } else {
      return weight;
    }
  }

  __device__ __forceinline__ static float output_scale(float inverse_sum,
                                                       float tensor_scale) {
    if constexpr (per_token_head || KV_DTYPE == KV_CACHE_DTYPE_FP16) {
      return inverse_sum;
    } else {
      return inverse_sum * tensor_scale;
    }
  }
  static constexpr int vector_bytes =
      vector_elements * KVReader<KV_DTYPE, E4M3_BITS>::element_bytes;
  static_assert(sizeof(typename KVStorageTraits<KV_DTYPE>::storage_type) ==
                    KVReader<KV_DTYPE, E4M3_BITS>::element_bytes,
                "KV storage type must match reader addressing");
};

}  // namespace flash_v100
