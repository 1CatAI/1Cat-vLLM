// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>
#include "src/turbomind/kernels/core/array.h"
#include "gguf_iq3_signed_codebook.cuh"

namespace vllm::sm70_gguf {
// Signed IQ3_S four-tuples stored as four nibbles. Original d and small
// scales remain separate; final FP16 operands match official dequantization.
struct Iq3NibbleBookDecoder {
  static constexpr int kSignedCodebookBytes = 512 * 16 * 2;
  struct Parameters {
    float d;
    uint32_t scales;
    half2 base;
  };
  template <bool SignedBook = true>
  __device__ static void initialize(uint8_t* grid) {
    static_assert(SignedBook);
    auto* vectors = reinterpret_cast<uint4*>(grid);
    const auto* source = reinterpret_cast<const uint4*>(signed_iq3_grid_words);
    for (int i = threadIdx.x; i < kSignedCodebookBytes / 16; i += blockDim.x)
      vectors[i] = __ldg(source + i);
    __syncthreads();
  }

  template <int Bit, int Words>
  __device__ static __forceinline__ uint32_t
  signed_index(const uint32_t (&words)[Words]) {
    static_assert(Bit + 13 <= Words * 32);
    constexpr int word = Bit / 32, offset = Bit % 32;
    if constexpr (offset + 13 <= 32)
      return (words[word] >> offset) & 8191;
    else {
      constexpr int byte = offset / 8;
      constexpr int selector =
          (byte + 3) * 4096 + (byte + 2) * 256 + (byte + 1) * 16 + byte;
      uint32_t window;
      asm("prmt.b32 %0,%1,%2,%3;"
          : "=r"(window)
          : "r"(words[word]), "r"(words[word + 1]), "n"(selector));
      return (window >> (offset % 8)) & 8191;
    }
  }

  template <class Output = float, bool CachedBase = false>
  __device__ static turbomind::Array<Output, 8> fragment_signed(
      Parameters parameters, int nibble, uint32_t first, uint32_t second,
      const uint8_t* grid) {
    static_assert(std::is_same_v<Output, half> && CachedBase);
    const uint32_t first_word =
        *reinterpret_cast<const uint16_t*>(grid + first * 2);
    const uint32_t second_word =
        *reinterpret_cast<const uint16_t*>(grid + second * 2);
    const uint32_t packed = __byte_perm(first_word, second_word, 0x5140);
    const uint32_t upper = __byte_perm(packed, 0, 0x4321);
    turbomind::Array<half, 8> result;
    const half2 factor = __float2half2_rn(float(2 + 4 * nibble));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t original = i < 2 ? packed : upper;
      const uint32_t values = (i & 1) ? original : (original << 4);
      constexpr uint32_t mask = 0x00f000f0;
      constexpr uint32_t magic = 0x54005400;
      uint32_t halves;
      asm("lop3.b32 %0,%1,%2,%3,0xea;"
          : "=r"(halves)
          : "r"(values), "r"(mask), "r"(magic));
      const half2 bias = __float2half2_rn(71.5f);
      const half2 integer_half = __hsub2((const half2&)halves, bias);
      const half2 scaled = __hmul2(integer_half, factor);
      (half2&)result[i * 2] = __hmul2(scaled, parameters.base);
    }
    return result;
  }
};
}  // namespace vllm::sm70_gguf
