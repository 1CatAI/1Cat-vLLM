// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <ATen/ATen.h>
#include <optional>

#include "kv_codec_traits.cuh"

namespace flash_v100 {

inline KVScalePlane bind_scale_plane(const at::Tensor& scale,
                                     const at::Tensor& payload) {
  TORCH_CHECK(scale.device() == payload.device(),
              "KV scales and payload must use the same device");
  TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.dim() == 3,
              "KV scales must be FP32 [page, token, head]");
  TORCH_CHECK(scale.sizes() == payload.sizes().slice(0, 3),
              "KV scale shape must match payload rows");
  TORCH_CHECK(scale.stride(0) > 0 && scale.stride(1) > 0 && scale.stride(2) > 0,
              "KV scales require positive strides");
  return {scale.data_ptr<float>(), scale.stride(0), scale.stride(1),
          scale.stride(2)};
}

inline KVScaleView bind_scale_view(
    int codec, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& key_scale,
    const std::optional<at::Tensor>& value_scale) {
  if (codec == KV_CACHE_DTYPE_INT8) {
    TORCH_CHECK(key_scale.has_value() && value_scale.has_value(),
                "Token/head KV codecs require both scale planes");
    return {bind_scale_plane(*key_scale, key),
            bind_scale_plane(*value_scale, value)};
  }
  TORCH_CHECK(!key_scale.has_value() && !value_scale.has_value(),
              "This KV codec uses tensor scales, not scale planes");
  return {};
}

}  // namespace flash_v100
