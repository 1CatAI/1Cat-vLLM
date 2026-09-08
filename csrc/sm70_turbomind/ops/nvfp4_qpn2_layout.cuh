// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cuda_runtime.h>

// Both layouts use E2M1 nibbles ordered [0,2,4,6,1,3,5,7] per K=8.
// TurboMind SM70 HMMA884 B/Pack1 stores [N/32, K/8, column, word].
// QPN2 stores [N/32, K/16, lane, uint2]. Only the word addresses differ.
template <bool TurboMindLayout>
struct Nvfp4Qpn2CodeReader {
  const uint8_t* base;

  __device__ __forceinline__ Nvfp4Qpn2CodeReader(const uint8_t* codes, int tile,
                                                 int groups_k16, int lane) {
    if constexpr (TurboMindLayout) {
      const int col =
          ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
      base = codes + (static_cast<size_t>(tile) * groups_k16 * 64 + col) * 4;
    } else {
      base = codes + (static_cast<size_t>(tile) * groups_k16 * 32 + lane) * 8;
    }
  }

  __device__ __forceinline__ uint2 load(int group) const {
    if constexpr (TurboMindLayout) {
      const auto* ptr = reinterpret_cast<const uint32_t*>(base) +
                        static_cast<size_t>(group) * 64;
      return make_uint2(__ldcs(ptr), __ldcs(ptr + 32));
    } else {
      return __ldcs(reinterpret_cast<const uint2*>(base) +
                    static_cast<size_t>(group) * 32);
    }
  }
};
