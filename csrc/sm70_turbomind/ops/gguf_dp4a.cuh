// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Integer-dot formulas and Q8_1 layout follow ggml-org/llama.cpp
// ggml-cuda/vecdotq.cuh and quantize.cu (MIT; see packaged llama.cpp LICENSE).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include "gguf_lattice_group.cuh"
#include "src/turbomind/kernels/gemm/lattice_codebooks.h"
#include "src/turbomind/kernels/gemm/transform.h"

namespace vllm::sm70_gguf {
struct Q8_1 {
  half2 ds;
  int8_t qs[32];
};
static_assert(sizeof(Q8_1) == 36);

__device__ __forceinline__ void quantize_q8_1_warp(Q8_1* out, float value) {
  const int lane = threadIdx.x % 32;
  float maximum = fabsf(value), sum = value;
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  const float d = maximum / 127.f;
  out->qs[lane] = maximum == 0.f ? 0 : int8_t(roundf(value / d));
  if (!lane) out->ds = __floats2half2_rn(d, sum);
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

  __device__ static LatticeGroup32 load_group(const uint8_t* row, int group,
                                              const uint32_t* book,
                                              const uint32_t* masks) {
    return load_lattice_group<Type, BankAware>(row + (group / 8) * kBlockBytes,
                                               group % 8, book, masks);
  }

  __device__ static float dot(const LatticeGroup32& w, const Q8_1& x) {
    const int* activation = reinterpret_cast<const int*>(x.qs);
    int sum0 = 0, sum1 = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if constexpr (Type == 22) {
        if (i < 4)
          sum0 = __dp4a(w.words[i], activation[i], sum0);
        else
          sum1 = __dp4a(w.words[i], activation[i], sum1);
      } else {
        sum0 = __dp4a(w.words[i], activation[i], sum0);
      }
    }
    const float d =
        __half2float(__ushort_as_half(w.scale_bits)) * __low2float(x.ds);
    if constexpr (Type == 21)
      return d * float(sum0 * w.scale0);
    else if constexpr (Type == 18)
      return d * (float(sum0) * float(w.scale0) * .25f);
    else
      return d * float(sum0 * w.scale0 + sum1 * w.scale1) * .125f;
  }

  __device__ static float dot(const uint8_t* row, int group, const Q8_1& x,
                              const uint32_t* book, const uint32_t* masks) {
    return dot(load_group(row, group, book, masks), x);
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
  struct Group {
    int words[8];
    half scale;
  };
  __device__ static Group load_group(const void* weight, const void* stats,
                                     int n, int k, int col, int group) {
    Group result;
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
      result.words[fragment * 2] = w0;
      result.words[fragment * 2 + 1] = w1;
    }
    const int64_t coefficient = int64_t{group} * n + col;
    result.scale =
        Type == 20 ? static_cast<const half*>(stats)[coefficient]
                   : __low2half(static_cast<const half2*>(stats)[coefficient]);
    return result;
  }
  __device__ static float dot(const Group& group, const Q8_1& x) {
    int sum = 0;
    const int* activation = reinterpret_cast<const int*>(x.qs);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      sum = __dp4a(group.words[i], activation[i], sum);
    return float(sum) * (__half2float(group.scale) * __low2float(x.ds));
  }
  __device__ static float dot(const void* weight, const void* stats, int n,
                              int k, int col, int group, const Q8_1& x) {
    return dot(load_group(weight, stats, n, k, col, group), x);
  }
};
}  // namespace vllm::sm70_gguf
