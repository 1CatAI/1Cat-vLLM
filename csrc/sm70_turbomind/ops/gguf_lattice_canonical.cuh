// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <type_traits>
#include "src/turbomind/kernels/gemm/lattice_transform.h"

namespace vllm::sm70_gguf {
template <int Type>
struct LatticeCanonicalDecoder {
  static constexpr int kGroup =
      Type == 17 || Type == 22 || Type == 29 ? 16 : 32;
  using Transform = turbomind::gemm::Transform_HMMA_SM70_Lattice<Type, kGroup>;
  using Stats =
      std::conditional_t<Type == 18 || Type == 21 || Type == 19 || Type == 29,
                         uint16_t, uint32_t>;
  using Carrier = std::conditional_t<Type == 18 || Type == 21,
                                     turbomind::uint4_t, turbomind::uint2_t>;

  __device__ static turbomind::Array<half, 8> fragment(const void* weight,
                                                       const void* stats, int k,
                                                       int n, int col, int base,
                                                       const uint8_t* grid) {
    const int64_t packet =
        (static_cast<int64_t>(col / 32) * (k / 8) + base / 8) * 32 + col % 32;
    uint64_t metadata = static_cast<const Stats*>(
        stats)[static_cast<int64_t>(base / kGroup) * n + col];
    const int within = base % kGroup;
    if constexpr (Type != 18 && Type != 21 && Type != 19 && Type != 29) {
      metadata = (metadata & 65535U) |
                 (((metadata >> (16 + 2 * (within / 8))) & 3U) << 16);
    }
    turbomind::Array<Carrier, 8> data[1][1];
    data[0][0] =
        reinterpret_cast<const turbomind::Array<Carrier, 8>*>(weight)[packet];
    turbomind::Array<Stats, 1> coefficients[1][1];
    coefficients[0][0][0] = static_cast<Stats>(metadata);
    turbomind::Array<half, 8> decoded[1][1];
    Transform::apply(decoded, 0, data, coefficients, 1, grid);
    return decoded[0][0];
  }
};
}  // namespace vllm::sm70_gguf
