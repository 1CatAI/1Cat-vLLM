// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Layout/formulas follow llama.cpp bed0a856606ee4a24a164066f73d2379447033f5
// ggml-common.h / ggml-quants.c (MIT). Shared codebooks retain their MIT
// license.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/gemm/lattice_codebooks.h"

namespace vllm::sm70_gguf {
template <int Type>
struct LatticeRawDecoder {
  static_assert(Type == 21 || Type == 22);
  using Codebook = turbomind::gemm::LatticeCodebook<Type>;
  static constexpr int kBlockBytes = Type == 21 ? 110 : 82;
  static constexpr int kCodebookBytes = Codebook::kBytes;

  __device__ static void initialize(uint8_t* grid) {
    auto* words = reinterpret_cast<uint32_t*>(grid);
    for (int i = threadIdx.x; i < kCodebookBytes / 4; i += blockDim.x)
      words[i] = Codebook::word(i);
    __syncthreads();
  }

  // Eight consecutive K values. Both scales are multiplied in FP32 before
  // multiplying grid values. No FP16 expanded coefficient exists in storage.
  __device__ static turbomind::Array<float, 8> fragment(const uint8_t* block,
                                                        int base,
                                                        const uint8_t* grid) {
    const float d = __half2float(*reinterpret_cast<const half*>(block));
    const int octet = base / 8;
    const uint8_t high = block[66 + base / 32];
    const uint8_t signs = block[(Type == 21 ? 74 : 34) + octet];
    turbomind::Array<float, 8> values;
    if constexpr (Type == 21) {
      const int nibble =
          (block[106 + base / 64] >> (4 * ((base / 32) % 2))) & 15;
      const float scale = d * (1 + 2 * nibble);
      const int sub = octet % 4;
      const int first = block[2 + 2 * octet] | (((high >> (2 * sub)) & 1) << 8);
      const int second =
          block[3 + 2 * octet] | (((high >> (2 * sub + 1)) & 1) << 8);
      const uint32_t a = *reinterpret_cast<const uint32_t*>(grid + first * 4);
      const uint32_t b = *reinterpret_cast<const uint32_t*>(grid + second * 4);
      const uint64_t packed = a | (static_cast<uint64_t>(b) << 32);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float value = static_cast<int>((packed >> (8 * i)) & 255) - 128;
        values[i] = scale * value * ((signs >> i) & 1 ? -1.f : 1.f);
      }
    } else {
      const int nibble =
          (block[74 + base / 32] >> (4 * ((base / 16) % 2))) & 15;
      const float scale = (d * (0.5f + nibble)) * 0.25f;
      const int index =
          block[2 + octet] | (((high >> (2 * (octet % 4))) & 3) << 8);
      const uint64_t packed =
          *reinterpret_cast<const uint64_t*>(grid + index * 8);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float value = static_cast<int>((packed >> (8 * i)) & 255) - 128;
        values[i] = scale * value * ((signs >> i) & 1 ? -1.f : 1.f);
      }
    }
    return values;
  }
};
}  // namespace vllm::sm70_gguf
