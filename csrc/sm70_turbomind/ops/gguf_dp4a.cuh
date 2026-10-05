// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Integer-dot formulas and Q8_1 layout follow ggml-org/llama.cpp
// ggml-cuda/vecdotq.cuh and quantize.cu (MIT; see packaged llama.cpp LICENSE).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
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

__device__ __forceinline__ uint32_t unpack_u4_word(uint32_t nibbles) {
  return __byte_perm(nibbles & 0x0f0fU, (nibbles >> 4) & 0x0f0fU, 0x5140);
}

template <int Kind>
struct RowIntegerGroup {
  int words[8];
  float scale0, scale1, minimum;
  __device__ static RowIntegerGroup load(const uint8_t* row,
                                         const float* scales, const float* mins,
                                         int group) {
    RowIntegerGroup result;
    if constexpr (Kind == 0) {
      const auto* packets = reinterpret_cast<const int4*>(row + group * 32);
      const int4 a = __ldg(packets), b = __ldg(packets + 1);
      result.words[0] = a.x;
      result.words[1] = a.y;
      result.words[2] = a.z;
      result.words[3] = a.w;
      result.words[4] = b.x;
      result.words[5] = b.y;
      result.words[6] = b.z;
      result.words[7] = b.w;
      result.scale0 = __ldg(scales + group * 2);
      result.scale1 = __ldg(scales + group * 2 + 1);
      result.minimum = 0.f;
    } else {
      const int4 packet =
          __ldg(reinterpret_cast<const int4*>(row + group * 16));
      const int packed[4] = {packet.x, packet.y, packet.z, packet.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t value = packed[i];
        if constexpr (Kind == 1) {
          result.words[2 * i] = unpack_u4_word(value);
          result.words[2 * i + 1] = unpack_u4_word(value >> 16);
        } else {
          using Lut = turbomind::gemm::Transform_HMMA_SM70_Lut4<0>;
          result.words[2 * i] = Lut::iq_values(value) ^ 0x80808080U;
          result.words[2 * i + 1] = Lut::iq_values(value >> 16) ^ 0x80808080U;
        }
      }
      result.scale0 = result.scale1 = __ldg(scales + group);
      result.minimum = Kind == 1 ? __ldg(mins + group) : 0.f;
    }
    return result;
  }
  __device__ float dot(const Q8_1& x) const {
    const int* activation = reinterpret_cast<const int*>(x.qs);
    int a = 0, b = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) a = __dp4a(words[i], activation[i], a);
#pragma unroll
    for (int i = 4; i < 8; ++i) b = __dp4a(words[i], activation[i], b);
    float result;
    if constexpr (Kind == 0)
      result = (float(a) * scale0 + float(b) * scale1) * __low2float(x.ds);
    else
      result = float(a + b) * scale0 * __low2float(x.ds);
    if constexpr (Kind == 1) result -= minimum * __high2float(x.ds);
    return result;
  }
};

// Raw Q6_K integer groups share Q8_1 and dp4a arithmetic with lattice
// projections. Signed subgroup scales stay in the integer dot domain.
struct RawQ6KGroup {
  int words[8];
  int scale0, scale1;
  float d;
  __device__ static RawQ6KGroup load(const uint8_t* row, int group) {
    RawQ6KGroup result;
    const uint8_t* block = row + (group / 8) * 210;
    const int sub = group % 8, half_block = sub / 4;
    const uint8_t* low = block + half_block * 64 + (sub % 2) * 32;
    const uint8_t* high = block + 128 + half_block * 32;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t ql =
          (load_u32_2(low + i * 4) >> ((sub % 4 >= 2) * 4)) & 0x0f0f0f0fU;
      const uint32_t qh =
          (load_u32_2(high + i * 4) >> ((sub % 4) * 2)) & 0x03030303U;
      result.words[i] = __vsub4(ql | (qh << 4), 0x20202020U);
    }
    const auto* scales = reinterpret_cast<const int8_t*>(block + 192);
    result.scale0 = scales[2 * sub];
    result.scale1 = scales[2 * sub + 1];
    result.d = __half2float(*reinterpret_cast<const half*>(block + 208));
    return result;
  }
  __device__ float dot(const Q8_1& x) const {
    const int* activation = reinterpret_cast<const int*>(x.qs);
    int a = 0, b = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) a = __dp4a(words[i], activation[i], a);
#pragma unroll
    for (int i = 4; i < 8; ++i) b = __dp4a(words[i], activation[i], b);
    return float(a * scale0 + b * scale1) * d * __low2float(x.ds);
  }
};

// Existing N32/K8 storage handles TP boundaries inside Q2_0's source K64
// blocks without expanded FP16 weights or a second layout. Its scale and
// centered integer values are exact; IQ4_NL uses the shared TurboMind LUT.
template <int Type>
struct CanonicalIntegerDot {
  static_assert(Type == 20 || Type == 42);
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
