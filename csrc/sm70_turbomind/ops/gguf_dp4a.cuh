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
template <int Type>
struct LatticeDot {
  static_assert(Type == 18 || Type == 21 || Type == 22);
  using Book = turbomind::gemm::LatticeCodebook<Type>;
  static constexpr int kBookWords = Book::kBytes / 4;
  static constexpr int kBlockBytes = Type == 18 ? 98 : Type == 21 ? 110 : 82;

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
    const uint32_t low = load_u32_2(b + 2 + sub * (Type == 22 ? 4 : 8));
    uint32_t high = 0, signs = 0;
    if constexpr (Type != 22) high = load_u32_2(b + 6 + sub * 8);
    if constexpr (Type == 18)
      signs = load_u32_2(b + 66 + sub * 4);
    else
      signs = load_u32_2(b + (Type == 21 ? 74 : 34) + sub * 4);
    const int qh = Type == 18 ? 0 : b[66 + sub];
    const int* activation = reinterpret_cast<const int*>(x.qs);
    int sum0 = 0, sum1 = 0;
#pragma unroll
    for (int octet = 0; octet < 4; ++octet) {
      int first, second, sign;
      if constexpr (Type == 22) {
        first =
            (((low >> (octet * 8)) & 255) | (((qh >> (2 * octet)) & 3) << 8)) *
            2;
        second = first + 1;
      } else {
        const uint32_t codes = octet < 2 ? low : high;
        const int shift = (octet % 2) * 16;
        first = ((codes >> shift) & 255);
        second = ((codes >> (shift + 8)) & 255);
        if constexpr (Type == 21) {
          first |= ((qh >> (2 * octet)) & 1) << 8;
          second |= ((qh >> (2 * octet + 1)) & 1) << 8;
        }
      }
      if constexpr (Type == 18) {
        sign = (signs >> (7 * octet)) & 127;
        sign |= (__popc(sign) & 1) << 7;
      } else {
        sign = (signs >> (8 * octet)) & 255;
      }
      const uint32_t s0 = masks[sign & 15], s1 = masks[sign >> 4];
      const int w0 = __vsub4(book[first] ^ s0, s0);
      const int w1 = __vsub4(book[second] ^ s1, s1);
      if constexpr (Type == 22) {
        if (octet < 2) {
          sum0 = __dp4a(w0, activation[2 * octet], sum0);
          sum0 = __dp4a(w1, activation[2 * octet + 1], sum0);
        } else {
          sum1 = __dp4a(w0, activation[2 * octet], sum1);
          sum1 = __dp4a(w1, activation[2 * octet + 1], sum1);
        }
      } else {
        sum0 = __dp4a(w0, activation[2 * octet], sum0);
        sum0 = __dp4a(w1, activation[2 * octet + 1], sum0);
      }
    }
    const float d =
        __half2float(*reinterpret_cast<const half*>(b)) * __low2float(x.ds);
    if constexpr (Type == 21) {
      const int scale = (b[106 + sub / 2] >> (4 * (sub % 2))) & 15;
      return d * float(sum0 * (1 + 2 * scale));
    } else if constexpr (Type == 18) {
      return d * (float(sum0) * float(1 + 2 * (signs >> 28)) * .25f);
    } else {
      const int scale = b[74 + sub];
      return d *
             float(sum0 * (1 + 2 * (scale & 15)) +
                   sum1 * (1 + 2 * (scale >> 4))) *
             .125f;
    }
  }
};
using IQ3SDot = LatticeDot<21>;
}  // namespace vllm::sm70_gguf
