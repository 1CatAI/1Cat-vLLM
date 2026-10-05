// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Integer-dot formulas and Q8_1 layout follow ggml-org/llama.cpp
// ggml-cuda/vecdotq.cuh and quantize.cu (MIT; see packaged llama.cpp LICENSE).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include "src/turbomind/kernels/gemm/lattice_codebooks.h"

namespace vllm::sm70_gguf {
struct Q8_1 {
  half2 ds;
  int8_t qs[32];
};
static_assert(sizeof(Q8_1) == 36);

__device__ __forceinline__ uint32_t load_u32_2(const uint8_t* p) {
  return uint32_t(*reinterpret_cast<const uint16_t*>(p)) |
         (uint32_t(*reinterpret_cast<const uint16_t*>(p + 2)) << 16);
}

// Shared by dense and routed kernels. Only integer codebook values reach
// dp4a; original weight and activation scales are applied after the dot.
struct IQ3SDot {
  using Book = turbomind::gemm::LatticeCodebook<21>;
  static constexpr int kBookWords = Book::kBytes / 4;
  static constexpr int kBlockBytes = 110;

  __device__ static void initialize(uint32_t* book, uint32_t* masks) {
    for (int i = threadIdx.x; i < kBookWords; i += blockDim.x)
      book[i] = Book::word(i) ^ 0x80808080U;
    if (threadIdx.x < 16) {
      const int s = threadIdx.x;
      masks[s] = ((s & 1) ? 0x000000ffU : 0) | ((s & 2) ? 0x0000ff00U : 0) |
                 ((s & 4) ? 0x00ff0000U : 0) | ((s & 8) ? 0xff000000U : 0);
    }
    __syncthreads();
  }

  __device__ static float dot(const uint8_t* row, int group, const Q8_1& x,
                              const uint32_t* book, const uint32_t* masks) {
    const uint8_t* b = row + (group / 8) * kBlockBytes;
    const int sub = group % 8;
    const uint8_t* qs = b + 2 + sub * 8;
    const uint32_t low = load_u32_2(qs), high = load_u32_2(qs + 4);
    const int qh = b[66 + sub];
    const uint32_t signs = load_u32_2(b + 74 + sub * 4);
    const int* activation = reinterpret_cast<const int*>(x.qs);
    int sum = 0;
#pragma unroll
    for (int octet = 0; octet < 4; ++octet) {
      const uint32_t codes = octet < 2 ? low : high;
      const int shift = (octet % 2) * 16;
      const int first =
          ((codes >> shift) & 255) | (((qh >> (2 * octet)) & 1) << 8);
      const int second =
          ((codes >> (shift + 8)) & 255) | (((qh >> (2 * octet + 1)) & 1) << 8);
      const int sign = (signs >> (8 * octet)) & 255;
      const uint32_t s0 = masks[sign & 15], s1 = masks[sign >> 4];
      const int w0 = __vsub4(book[first] ^ s0, s0);
      const int w1 = __vsub4(book[second] ^ s1, s1);
      sum = __dp4a(w0, activation[2 * octet], sum);
      sum = __dp4a(w1, activation[2 * octet + 1], sum);
    }
    const int scale = (b[106 + sub / 2] >> (4 * (sub % 2))) & 15;
    const float d =
        __half2float(*reinterpret_cast<const half*>(b)) * __low2float(x.ds);
    return d * float(sum * (1 + 2 * scale));
  }
};
}  // namespace vllm::sm70_gguf
