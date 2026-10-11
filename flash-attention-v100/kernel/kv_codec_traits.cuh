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
  static constexpr int vector_alignment = 16;
};

template <>
struct KVStorageTraits<KV_CACHE_DTYPE_FP8_E4M3> {
  using storage_type = uint8_t;
  static constexpr bool quantized = true;
  static constexpr int vector_alignment = 8;
};

template <>
struct KVStorageTraits<KV_CACHE_DTYPE_FP8_E5M2> {
  using storage_type = uint8_t;
  static constexpr bool quantized = true;
  static constexpr int vector_alignment = 8;
};

template <>
struct KVStorageTraits<KV_CACHE_DTYPE_INT8> {
  using storage_type = int8_t;
  static constexpr bool quantized = true;
  static constexpr int vector_alignment = 4;
};

// Borrowed scale addresses use element strides from the public cache view.
// No algorithm infers an inline/planar layout from a payload dtype.
template <typename Scalar>
struct KVScalePlaneT {
  Scalar* data = nullptr;
  int64_t block_stride = 0;
  int64_t token_stride = 0;
  int64_t head_stride = 0;

  __device__ __forceinline__ int64_t offset(int block, int token,
                                            int head) const {
    return static_cast<int64_t>(block) * block_stride +
           static_cast<int64_t>(token) * token_stride +
           static_cast<int64_t>(head) * head_stride;
  }

  __device__ __forceinline__ float load(int block, int token, int head) const {
    return data[offset(block, token, head)];
  }
};

using KVScalePlane = KVScalePlaneT<const float>;
using KVScaleOutputPlane = KVScalePlaneT<float>;

struct KVScaleView {
  KVScalePlane key;
  KVScalePlane value;
};

struct KVScaleOutputView {
  KVScaleOutputPlane key;
  KVScaleOutputPlane value;
};

// One format contract supplies scalar and packed-vector readers to every
// attention family. Keep KVReader available for existing native callers.
template <int KV_DTYPE, bool E4M3_BITS = false>
struct KVCodecTraits : KVStorageTraits<KV_DTYPE>,
                       KVReader<KV_DTYPE, E4M3_BITS> {
  static constexpr int vector_elements = 8;
  static constexpr bool per_token_head = KV_DTYPE == KV_CACHE_DTYPE_INT8;
  static constexpr bool fp32_partials =
      per_token_head || KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3;
  static constexpr bool paired_byte_load =
      KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3 ||
      KV_DTYPE == KV_CACHE_DTYPE_FP8_E5M2 || per_token_head;

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

// Apply row scales to an existing score or probability tile. Softmax,
// masking and reduction remain owned by the attention algorithm.
template <int KV_DTYPE>
__device__ __forceinline__ void scale_attention_tile(
    float* tile, int rows, int columns, int stride, const KVScalePlane& scale,
    const int* page_ids, int first_token, int block_size, int head) {
  if constexpr (KVCodecTraits<KV_DTYPE>::per_token_head) {
    for (int i = threadIdx.x; i < rows * columns; i += blockDim.x) {
      const int row = i / columns;
      const int col = i - row * columns;
      const int token = first_token + col;
      tile[row * stride + col] = KVCodecTraits<KV_DTYPE>::value_weight(
          tile[row * stride + col], scale, page_ids[token / block_size],
          token % block_size, head);
    }
  }
}

}  // namespace flash_v100
