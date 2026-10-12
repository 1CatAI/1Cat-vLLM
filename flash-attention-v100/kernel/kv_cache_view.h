// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <ATen/ATen.h>
#include <optional>
#include <type_traits>

#include "kv_codec_traits.cuh"

namespace flash_v100 {

// Compatible algorithms instantiate from one catalog. A codec implements its
// reader/scale traits once; it does not add another algorithm dispatch tree.
template <typename Operation>
decltype(auto) visit_kv_codec(int codec, Operation&& operation) {
  switch (codec) {
#define BIND_CODEC(ID) \
  case ID:             \
    return operation(std::integral_constant<int, ID>{});
    BIND_CODEC(KV_CACHE_DTYPE_FP16)
    BIND_CODEC(KV_CACHE_DTYPE_FP8_E4M3)
    BIND_CODEC(KV_CACHE_DTYPE_FP8_E5M2)
    BIND_CODEC(KV_CACHE_DTYPE_INT8)
#undef BIND_CODEC
    default:
      TORCH_CHECK(false, "Unregistered native KV codec: ", codec);
  }
}

template <int KV_DTYPE>
constexpr at::ScalarType kv_storage_type() {
  using Storage = typename KVStorageTraits<KV_DTYPE>::storage_type;
  static_assert(std::is_same_v<Storage, __half> ||
                    std::is_same_v<Storage, uint8_t> ||
                    std::is_same_v<Storage, int8_t>,
                "Native KV storage needs an ATen dtype binding");
  return std::is_same_v<Storage, __half>   ? at::kHalf
         : std::is_same_v<Storage, int8_t> ? at::kChar
                                           : at::kByte;
}

inline KVScalePlane bind_scale_plane(const at::Tensor& scale,
                                     const at::Tensor& payload,
                                     int page_group_size = 1,
                                     int subpage_tokens = 0) {
  TORCH_CHECK(scale.device() == payload.device(),
              "KV scales and payload must use the same device");
  TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.dim() == 3,
              "KV scales must be FP32 [page, token, head]");
  TORCH_CHECK(page_group_size >= 1, "Invalid KV scale page mapping");
  if (page_group_size == 1) {
    TORCH_CHECK(scale.sizes() == payload.sizes().slice(0, 3),
                "KV scale shape must match payload rows");
  } else {
    TORCH_CHECK(
        subpage_tokens > 0 && scale.sizes() == payload.sizes().slice(0, 3) &&
            payload.size(1) ==
                static_cast<int64_t>(page_group_size) * subpage_tokens,
        "KV scales must cover the owning pages and their mapped subpages");
  }
  TORCH_CHECK(scale.stride(0) > 0 && scale.stride(1) > 0 && scale.stride(2) > 0,
              "KV scales require positive strides");
  return {scale.data_ptr<float>(), scale.stride(0), scale.stride(1),
          scale.stride(2),         page_group_size, subpage_tokens};
}

inline KVScaleView bind_scale_view(int codec, const at::Tensor& key,
                                   const at::Tensor& value,
                                   const std::optional<at::Tensor>& key_scale,
                                   const std::optional<at::Tensor>& value_scale,
                                   int page_group_size = 1,
                                   int subpage_tokens = 0) {
  TORCH_CHECK(page_group_size >= 1 && subpage_tokens >= 0,
              "Invalid KV subpage mapping");
  if (subpage_tokens > 0) {
    TORCH_CHECK(key.dim() == 4 && value.dim() == 4 &&
                    key.size(1) == static_cast<int64_t>(page_group_size) *
                                       subpage_tokens &&
                    value.size(1) == key.size(1),
                "KV subpage mapping must match the payload owner");
  }
  return visit_kv_codec(codec, [&](auto tag) -> KVScaleView {
    if constexpr (KVCodecTraits<decltype(tag)::value>::per_token_head) {
      TORCH_CHECK(key_scale.has_value() && value_scale.has_value(),
                  "Token/head KV codecs require both scale planes");
      return {
          bind_scale_plane(*key_scale, key, page_group_size, subpage_tokens),
          bind_scale_plane(*value_scale, value, page_group_size,
                           subpage_tokens),
          {page_group_size, subpage_tokens}};
    } else {
      TORCH_CHECK(!key_scale.has_value() && !value_scale.has_value(),
                  "This KV codec uses tensor scales, not scale planes");
      return {{}, {}, {page_group_size, subpage_tokens}};
    }
  });
}

inline KVScaleOutputPlane bind_scale_output_plane(const at::Tensor& scale,
                                                  const at::Tensor& payload) {
  const auto plane = bind_scale_plane(scale, payload);
  return {scale.data_ptr<float>(), plane.block_stride, plane.token_stride,
          plane.head_stride};
}

// Dense inputs may be exact, unscaled codec expansions. The same address
// contract applies to [batch, token, head] scales without a page table.
inline KVScaleView bind_dense_scale_view(
    const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& key_scale,
    const std::optional<at::Tensor>& value_scale) {
  TORCH_CHECK(key_scale.has_value() == value_scale.has_value(),
              "Dense KV requires both scale planes or neither");
  return key_scale ? KVScaleView{bind_scale_plane(*key_scale, key),
                                 bind_scale_plane(*value_scale, value),
                                 {}}
                   : KVScaleView{};
}

}  // namespace flash_v100
