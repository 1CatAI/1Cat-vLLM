// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Integer-dot formulas and Q8_1 layout follow ggml-org/llama.cpp
// ggml-cuda/vecdotq.cuh and quantize.cu (MIT; see packaged llama.cpp LICENSE).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include "gguf_q8_1.cuh"
#include "src/turbomind/kernels/gemm/lattice_codebooks.h"
#include "src/turbomind/kernels/gemm/transform.h"

namespace vllm::sm70_gguf {
__device__ __forceinline__ uint32_t load_u32_2(const uint8_t* p) {
  return uint32_t(*reinterpret_cast<const uint16_t*>(p)) |
         (uint32_t(*reinterpret_cast<const uint16_t*>(p + 2)) << 16);
}

// Shared by dense and routed kernels. Only integer codebook values reach
// dp4a; original weight and activation scales are applied after the dot.
template <int Type, bool BankAware = false>
struct LatticeDot {
  static_assert(Type == 18 || Type == 21 || Type == 22);
  using Book = turbomind::gemm::LatticeCodebook<Type>;
  static constexpr int kBookWords = Book::kBytes / 4;
  static constexpr int kBlockBytes = Type == 18 ? 98 : Type == 21 ? 110 : 82;

  __device__ static void initialize(uint32_t* book, uint32_t* masks) {
    for (int i = threadIdx.x; i < kBookWords; i += blockDim.x) {
      // IQ2 entries have two words. Separate their planes so each lookup
      // can address all 32 banks, rather than only even or odd banks.
      const int slot =
          BankAware && Type == 22 ? (i / 2 + (i % 2) * (kBookWords / 2)) : i;
      book[slot] = Book::word(i) ^ 0x80808080U;
    }
    if (!BankAware && threadIdx.x < 16) {
      const int s = threadIdx.x;
      masks[s] = ((s & 1) ? 0x000000ffU : 0) | ((s & 2) ? 0x0000ff00U : 0) |
                 ((s & 4) ? 0x00ff0000U : 0) | ((s & 8) ? 0xff000000U : 0);
    }
    __syncthreads();
  }

  __device__ static uint32_t sign_mask(int nibble) {
    // Spread the four bits into byte sign positions, then sign-extend each
    // byte with PRMT. __byte_perm clears selector sign bits; use PTX here.
    const uint32_t bits = uint32_t(nibble) * 0x10204080U;
    uint32_t mask;
    asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(mask) : "r"(bits));
    return mask;
  }

  __device__ static uint32_t word(const uint32_t* book, int index) {
    if constexpr (BankAware && Type == 22)
      return book[index / 2 + (index % 2) * (kBookWords / 2)];
    return book[index];
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
      const uint32_t s0 = BankAware ? sign_mask(sign & 15) : masks[sign & 15];
      const uint32_t s1 = BankAware ? sign_mask(sign >> 4) : masks[sign >> 4];
      const int w0 = __vsub4(word(book, first) ^ s0, s0);
      const int w1 = __vsub4(word(book, second) ^ s1, s1);
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

// Lossless scalar LUT expansion avoids the correlated codebook in shared
// memory. Records preserve the source base and integer subscales, so dot
// arithmetic and the final FP16 boundary remain the same as LatticeDot.
template <int Type>
struct SignedLutDot {
  static_assert(Type == 18 || Type == 21 || Type == 22);
  static constexpr int kBookWords = 1;
  __device__ static void initialize(uint32_t*, uint32_t*) { __syncthreads(); }
  __host__ __device__ static constexpr int value(int i) {
    if constexpr (Type == 21) return 2 * i - 15;
    if constexpr (Type == 18) {
      const int levels[16] = {-62, -52, -44, -36, -28, -20, -12, -4,
                              4,   12,  20,  28,  36,  44,  52,  62};
      return levels[i];
    }
    const int levels[16] = {-43, -25, -8, 8, 25, 43};
    return levels[i];
  }
  __host__ __device__ static constexpr uint32_t table(int start) {
    uint32_t word = 0;
    for (int i = 0; i < 4; ++i)
      word |= uint32_t(value(start + i) + 128) << (8 * i);
    return word;
  }
  __device__ static uint32_t decode(uint32_t nibbles) {
    const uint32_t selector = nibbles & 0x7777U;
    const uint32_t low = __byte_perm(table(0), table(4), selector);
    const uint32_t high = __byte_perm(table(8), table(12), selector);
    return __byte_perm(low, high, ((nibbles & 0x8888U) >> 1) | 0x3210U) ^
           0x80808080U;
  }
  __device__ static float dot(const uint8_t* row, int group, const Q8_1& x,
                              const uint32_t*, const uint32_t*) {
    const uint8_t* b = row + group * 20;
    const auto* activation = reinterpret_cast<const int*>(x.qs);
    int sum0 = 0, sum1 = 0;
#pragma unroll
    for (int fragment = 0; fragment < 4; ++fragment) {
      const uint32_t packed = reinterpret_cast<const uint32_t*>(b)[fragment];
      int& sum = Type == 22 && fragment >= 2 ? sum1 : sum0;
      const int low = static_cast<int>(decode(packed));
      const int high = static_cast<int>(decode(packed >> 16));
      sum = __dp4a(low, activation[2 * fragment], sum);
      sum = __dp4a(high, activation[2 * fragment + 1], sum);
    }
    const float d = __half2float(*reinterpret_cast<const half*>(b + 16)) *
                    __low2float(x.ds);
    if constexpr (Type == 18)
      return d * (float(sum0) * float(b[18]) * .25f);
    else if constexpr (Type == 21)
      return d * float(sum0 * b[18]);
    else
      return d * float(sum0 * b[18] + sum1 * b[19]) * .125f;
  }
};

// Existing N32/K8 storage handles TP boundaries inside Q2_0's source K64
// blocks without expanded FP16 weights or a second layout. Its scale and
// centered integer values are exact; IQ4_NL uses the shared TurboMind LUT.
template <int Type>
struct CanonicalIntegerDot {
  static_assert(Type == 20 || Type == 42);
  static constexpr int kBookWords = 1;
  __device__ static float dot(const void* weight, const void* stats, int n,
                              int k, int col, int group, const Q8_1& x) {
    int sum = 0;
    const int* activation = reinterpret_cast<const int*>(x.qs);
#pragma unroll
    for (int fragment = 0; fragment < 4; ++fragment) {
      const int64_t packet =
          (int64_t{col / 32} * (k / 8) + group * 4 + fragment) * 32 + col % 32;
      uint32_t even, odd;
      if constexpr (Type == 20) {
        const uint32_t packed = static_cast<const uint32_t*>(weight)[packet];
        using Lut = turbomind::gemm::Transform_HMMA_SM70_Lut4<0>;
        even = Lut::iq_values(packed) ^ 0x80808080U;
        odd = Lut::iq_values(packed >> 16) ^ 0x80808080U;
      } else {
        const uint32_t packed = static_cast<const uint16_t*>(weight)[packet];
        const auto expand = [](uint32_t p) {
          return (p & 3) | (((p >> 2) & 3) << 8) | (((p >> 4) & 3) << 16) |
                 (((p >> 6) & 3) << 24);
        };
        even = __vsub4(expand(packed), 0x01010101U);
        odd = __vsub4(expand(packed >> 8), 0x01010101U);
      }
      const int w0 = __byte_perm(even, odd, 0x5140);
      const int w1 = __byte_perm(even, odd, 0x7362);
      sum = __dp4a(w0, activation[fragment * 2], sum);
      sum = __dp4a(w1, activation[fragment * 2 + 1], sum);
    }
    const int64_t coefficient = int64_t{group} * n + col;
    const half scale =
        Type == 20 ? static_cast<const half*>(stats)[coefficient]
                   : __low2half(static_cast<const half2*>(stats)[coefficient]);
    return float(sum) * (__half2float(scale) * __low2float(x.ds));
  }
};
}  // namespace vllm::sm70_gguf
