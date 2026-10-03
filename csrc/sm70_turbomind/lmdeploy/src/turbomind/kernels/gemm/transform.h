// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

struct Transform_Default {
  template <class T, int Nf, int Mf, int K, int Nd, int Md, class S>
  __device__ static void apply(Array<T, Nf> (&frag)[K][Mf], int k,
                               Array<T, Nd> (&data)[K][Md], S&, int div) {
    static_assert(Nf * Mf == Nd * Md);
    static_assert(Nd % Nf == 0 && Mf % Md == 0);
    static_assert(sizeof(frag) == sizeof(data));

    // Alignment must be manually enforced for `reinterpret_cast`
    auto& frag_k = reinterpret_cast<Array<T, Nd>(&)[Md]>(frag[k]);
    auto& data_k = data[k];

    PRAGMA_UNROLL
    for (int i = 0; i < std::size(frag_k); ++i) {
      frag_k[i] = data_k[i];
    }
  }
};

template <int StatStepS, int StatStepC>
struct Transform_HMMA_16816 {
  template <class F, int Nf, int Mf, int K, class D, int Nd, int Md, class S,
            int Ns, int Ms, int Ks>
  __device__ static void apply(Array<F, Nf> (&frag)[K][Mf], int k,
                               Array<D, Nd> (&data)[K][Md],
                               Array<S, Ns> (&stat)[Ks][Ms], int div) {
    static_assert(Nf * Mf == Nd * Md);
    static_assert(Nd % Nf == 0 && Mf % Md == 0);
    static_assert(Nf * Mf == Ns * Ms * 4);

    auto& frag_k = reinterpret_cast<Array<F, Nd>(&)[Md]>(frag[k]);
    auto& stat_k = reinterpret_cast<Array<S, 1>(&)[Ns * Ms]>(stat[k / div]);
    auto& data_k = data[k];

    PRAGMA_UNROLL
    for (int m = 0; m < Md; ++m) {
      auto tmp = ConvertKvCache<D, F>::convert(data_k[m]);
      static_assert(Nd % 8 == 0);
      PRAGMA_UNROLL
      for (int i = 0; i < Nd; i += 8) {
        PRAGMA_UNROLL
        for (int s = 0; s < 2; ++s) {
          PRAGMA_UNROLL
          for (int c = 0; c < 2; ++c) {
            const int idx =
                (m * Nd + i) / 8 * 2 + s * StatStepS + c * StatStepC;
            dequant((Array<F, 2>&)tmp[i + s * 4 + c * 2], stat_k[idx]);
          }
        }
      }

      frag_k[m] = tmp;
    }
  }

  template <class F>
  __device__ static void dequant(Array<F, 2>& x, Array<uint32_t, 1> s) {
    Array<F, 2>& _s = (Array<F, 2>&)s;
    x[0] = __hfma(x[0], _s[0], _s[1]);
    x[1] = __hfma(x[1], _s[0], _s[1]);
  }

  __device__ static void dequant(Array<bfloat16_t, 2>& x, Array<uint8_t, 1> s) {
    bfloat16_t s1 = __ushort_as_bfloat16((uint16_t)s[0] << 7);
    x[0] = __hmul(x[0], s1);
    x[1] = __hmul(x[1], s1);
  }

  __device__ static void dequant(Array<half_t, 2>& x, Array<uint8_t, 1> s) {
    // half_t s1 = __ushort_as_half(((uint16_t)s[0] + 15 - 127) << 10);
    // Adjusted in `AdjustUe8m0ScaleForHalf`
    half_t s1 = __ushort_as_half((uint16_t)s[0] << 10);
    x[0] = __hmul(x[0], s1);
    x[1] = __hmul(x[1], s1);
  }

  __device__ static void dequant(Array<bfloat16_t, 2>& x,
                                 Array<uint16_t, 1> s) {
    auto s1 = __ushort_as_bfloat16(s[0]);
    x[0] = __hmul(x[0], s1);
    x[1] = __hmul(x[1], s1);
  }

  __device__ static void dequant(Array<half, 2>& x, Array<uint16_t, 1> s) {
    auto s1 = __ushort_as_half(s[0]);
    x[0] = __hmul(x[0], s1);
    x[1] = __hmul(x[1], s1);
  }
};

// Used by SM70 MMA
struct Transform_HMMA_SIMT_B {
  template <class F, class D, int N>
  __device__ static auto decode(const Array<D, N>& data) {
    if constexpr (std::is_same_v<D, uint8_t> && std::is_same_v<F, half>) {
      // Converter<uint16_t,uint8_t> interleaves the middle two bytes for
      // packed MMA operands. KV-cache uint8 storage has a different order.
      static_assert(N % 4 == 0);
      Array<F, N> decoded;
      PRAGMA_UNROLL
      for (int i = 0; i < N; i += 4) {
        (Array<F, 4>&)decoded[i] =
            cvt_f16x2x2_u8_trans<true>((const Array<uint8_t, 4>&)data[i]);
      }
      return decoded;
    } else if constexpr (std::is_same_v<D, uint2_t> && std::is_same_v<F, half>) {
      static_assert(N % 8 == 0);
      Array<F, N> decoded;
      constexpr uint32_t magic = 0x64006400U;
      PRAGMA_UNROLL
      for (int i = 0; i < N; i += 8) {
        const uint32_t packed = (const uint16_t&)data[i];
        PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          const uint32_t lanes = (packed >> (j * 2)) & 0x0303U;
          uint32_t halves = __byte_perm(lanes, magic, 0x7170);
          (half2&)decoded[i + j * 2] =
              __hsub2((const half2&)halves, (const half2&)magic);
        }
      }
      return decoded;
    } else {
      return ConvertKvCache<D, F>::convert(data);
    }
  }

  template <class F, int Nf, int Mf, int K, class D, int Nd, int Md, class S,
            int Ns, int Ms, int Ks>
  __device__ static void apply(Array<F, Nf> (&frag)[K][Mf], int k,
                               Array<D, Nd> (&data)[K][Md],
                               Array<S, Ns> (&stat)[Ks][Ms], int div) {
    static_assert(Nf * Mf == Nd * Md);
    static_assert(Nd % Nf == 0 && Mf % Md == 0);

    auto& frag_k = reinterpret_cast<Array<F, Nd>(&)[Md]>(frag[k]);
    auto& stat_k = reinterpret_cast<Array<S, 1>(&)[Ns * Ms]>(stat[k / div]);
    auto& data_k = data[k];

    // static_assert(Nf != Nf);

    PRAGMA_UNROLL
    for (int m = 0; m < Md; ++m) {
      auto tmp = decode<F>(data_k[m]);
      PRAGMA_UNROLL
      for (int i = 0; i < Nd; i += 2) {
        dequant((Array<F, 2>&)tmp[i], stat_k[(m * Nd + i) / Nf]);
      }
      frag_k[m] = tmp;
    }
  }

  template <class F>
  __device__ static void dequant(Array<F, 2>& x, Array<uint32_t, 1> s) {
    Array<F, 2>& _s = (Array<F, 2>&)s;

    x[0] = __hfma(x[0], _s[0], _s[1]);
    x[1] = __hfma(x[1], _s[0], _s[1]);
  }

  __device__ static void dequant(Array<half_t, 2>& x, Array<uint8_t, 1> s) {
    // half_t s1 = __ushort_as_half(((uint16_t)s[0] + 15 - 127) << 10);
    // Adjusted in `AdjustUe8m0ScaleForHalf`
    half_t s1 = __ushort_as_half((uint16_t)s[0] << 10);
    x[0] = __hmul(x[0], s1);
    x[1] = __hmul(x[1], s1);
  }

  __device__ static void dequant(Array<half, 2>& x, Array<uint16_t, 1> s) {
    auto s1 = __ushort_as_half(s[0]);
    x[0] = __hmul(x[0], s1);
    x[1] = __hmul(x[1], s1);
  }
};

// High code bits travel with one group's affine coefficient pair. Metadata
// uses an aligned 64-bit carrier: low 32 bits are scale/min, high 32 bits are
// little-endian high codes. Tiles must start at a complete metadata group.
template<int LowBits, int HighBits, int GroupSize>
struct Transform_HMMA_SM70_BitPlane {
  template<class F, int Nf, int Mf, int K, class D, int Nd, int Md,
           class S, int Ns, int Ms, int Ks>
  __device__ static void apply(Array<F, Nf> (&frag)[K][Mf], int k,
                               Array<D, Nd> (&data)[K][Md],
                               Array<S, Ns> (&stat)[Ks][Ms], int div) {
    static_assert(std::is_same_v<F, half> && std::is_same_v<S, uint64_t>);
    static_assert(Nd == 8 && Nf == 8 && Mf == Md);
    static_assert(GroupSize * HighBits <= 32);
    auto& frag_k = reinterpret_cast<Array<F, Nd> (&)[Md]>(frag[k]);
    auto& stat_k = reinterpret_cast<Array<S, 1> (&)[Ns * Ms]>(stat[k / div]);
    const int base_k = (k * Nd) % GroupSize;
    PRAGMA_UNROLL
    for (int m = 0; m < Md; ++m) {
      const uint64_t metadata = stat_k[m][0];
      const uint32_t high = metadata >> 32;
      const uint32_t packed = LowBits == 2 ? (const uint16_t&)data[k][m]
                                           : (const uint32_t&)data[k][m];
      Array<F, Nd> decoded;
      constexpr uint32_t magic = 0x64006400U;
      Array<uint32_t, 1> coefficients;
      coefficients[0] = static_cast<uint32_t>(metadata);
      PRAGMA_UNROLL
      for (int i = 0; i < Nd; i += 2) {
        uint32_t halves;
        if constexpr (LowBits == 2) {
          const uint32_t lanes = (packed >> (i / 2 * 2)) & 0x0303U;
          halves = __byte_perm(lanes, magic, 0x7170);
        }
        else {
          halves = ((packed >> (i / 2 * 4)) & 0x000F000FU) | magic;
        }
        const uint32_t upper = high >> ((base_k + i) * HighBits);
        constexpr uint32_t mask = (1U << HighBits) - 1;
        const uint32_t pair = (upper & mask) | (((upper >> HighBits) & mask) << 16);
        halves |= pair << LowBits;
        // All combined codes are exact integers below 64. The FP16 1024
        // mantissa trick restores them without eight integer conversions.
        (half2&)decoded[i] = __hsub2((const half2&)halves, (const half2&)magic);
      }
      PRAGMA_UNROLL
      for (int i = 0; i < Nd; i += 2) {
        Transform_HMMA_SIMT_B::dequant((Array<F, 2>&)decoded[i], coefficients);
      }
      frag_k[m] = decoded;
    }
  }
};

// Nonlinear nibbles use the native U4 packing. Lookup and scale restore
// FP16 weights in registers before the shared mma884 loop.
template <int Table>
struct Transform_HMMA_SM70_Lut4 {
  static constexpr auto kQuantType =
      Table == 0 ? QuantType::kLut4IQ : QuantType::kLut4E2M1;

  __device__ static uint32_t iq_values(uint32_t nibbles) {
    // Four official IQ4_NL values + 128. Two lookups cover eight values
    // each; a third byte permutation selects their high index bits.
    const uint32_t selector = nibbles & 0x7777U;
    const uint32_t lo = __byte_perm(0x3F2D1801U, 0x766A5D4FU, selector);
    const uint32_t hi = __byte_perm(0xA6998D81U, 0xF1D9C5B5U, selector);
    return __byte_perm(lo, hi, ((nibbles & 0x8888U) >> 1) | 0x3210U);
  }

  template <class F, int Nf, int Mf, int K, class D, int Nd, int Md, class S,
            int Ns, int Ms, int Ks>
  __device__ static void apply(Array<F, Nf> (&frag)[K][Mf], int k,
                               Array<D, Nd> (&data)[K][Md],
                               Array<S, Ns> (&stat)[Ks][Ms], int div) {
    static_assert(std::is_same_v<D, uint4_t> && std::is_same_v<F, half>);
    static_assert(std::is_same_v<S, uint16_t> && Nd == 8 && Nf * Mf == Nd * Md);
    auto& dst = reinterpret_cast<Array<F, Nd> (&)[Md]>(frag[k]);
    auto& scales = reinterpret_cast<Array<S, 1> (&)[Ns * Ms]>(stat[k / div]);
    PRAGMA_UNROLL
    for (int m = 0; m < Md; ++m) {
      Array<F, Nd> decoded;
      if constexpr (Table == 0) {
        const uint32_t packed = (const uint32_t&)data[k][m];
        constexpr uint32_t magic = 0x64006400U;
        constexpr uint32_t bias = 0x64806480U;  // 1152 == 1024 + 128
        const uint32_t even = iq_values(packed);
        const uint32_t odd = iq_values(packed >> 16);
        PRAGMA_UNROLL
        for (int i = 0; i < Nd; i += 2) {
          const uint32_t selector = (i / 2) | ((i / 2 + 4) << 8);
          const uint32_t halves = magic |
              (__byte_perm(even, odd, selector) & 0x00FF00FFU);
          (half2&)decoded[i] = __hsub2((const half2&)halves, (const half2&)bias);
        }
      } else {
        decoded = ConvertKvCache<fp4_e2m1_t, F>::convert(
            (const Array<fp4_e2m1_t, Nd>&)data[k][m]);
      }
      PRAGMA_UNROLL
      for (int i = 0; i < Nd; i += 2) {
        uint32_t scale = scales[(m * Nd + i) / Nf][0];
        scale |= scale << 16;
        (half2&)decoded[i] = __hmul2((const half2&)decoded[i], (const half2&)scale);
      }
      dst[m] = decoded;
    }
  }
};

// Q3's exact centered form omits a redundant min and the unused 16 high bits.
// Preparation requires min == -4 * scale; otherwise the caller must fall back.
struct Transform_HMMA_SM70_CenteredBitPlane3 {
  static constexpr auto kQuantType = QuantType::kCenteredBitPlane3;

  template <class F, int Nf, int Mf, int K, class D, int Nd, int Md, class S,
            int Ns, int Ms, int Ks>
  __device__ static void apply(Array<F, Nf> (&frag)[K][Mf], int k,
                               Array<D, Nd> (&data)[K][Md],
                               Array<S, Ns> (&stat)[Ks][Ms], int div) {
    static_assert(std::is_same_v<D, uint2_t> && std::is_same_v<F, half>);
    static_assert(std::is_same_v<S, uint32_t> && Nd == 8 && Nf == 8 && Mf == Md);
    auto& dst = reinterpret_cast<Array<F, Nd> (&)[Md]>(frag[k]);
    auto& stats = reinterpret_cast<Array<S, 1> (&)[Ns * Ms]>(stat[k / div]);
    const int base_k = (k * Nd) % 16;
    PRAGMA_UNROLL
    for (int m = 0; m < Md; ++m) {
      const uint32_t metadata = stats[m][0];
      const uint32_t high = metadata >> 16;
      const uint32_t packed = (const uint16_t&)data[k][m];
      constexpr uint32_t magic = 0x64006400U;
      constexpr uint32_t bias = 0x64046404U;  // 1024 + center 4
      const uint32_t scale = (metadata & 65535U) | (metadata << 16);
      Array<F, Nd> decoded;
      PRAGMA_UNROLL
      for (int i = 0; i < Nd; i += 2) {
        const uint32_t lanes = (packed >> (i / 2 * 2)) & 0x0303U;
        uint32_t halves = __byte_perm(lanes, magic, 0x7170);
        const uint32_t upper = high >> (base_k + i);
        halves |= ((upper & 1U) | (((upper >> 1) & 1U) << 16)) << 2;
        const half2 centered = __hsub2((const half2&)halves, (const half2&)bias);
        (half2&)decoded[i] = __hmul2(centered, (const half2&)scale);
      }
      dst[m] = decoded;
    }
  }
};

// FP8 scales that have absorbed the E4M3 exponent-bias factor (256) during
// one-time weight preparation.
struct Transform_HMMA_SIMT_B_PrescaledE4M3 {
  template <class F, int Nf, int Mf, int K, class D, int Nd, int Md, class S,
            int Ns, int Ms, int Ks>
  __device__ static void apply(Array<F, Nf> (&frag)[K][Mf], int k,
                               Array<D, Nd> (&data)[K][Md],
                               Array<S, Ns> (&stat)[Ks][Ms], int div) {
    static_assert(std::is_same_v<D, fp8_e4m3_t>);
    static_assert(std::is_same_v<F, half>);
    static_assert(std::is_same_v<S, uint16_t>);
    static_assert(Nf * Mf == Nd * Md);
    static_assert(Nd % Nf == 0 && Mf % Md == 0);
    static_assert(Nd % 4 == 0);

    auto& frag_k = reinterpret_cast<Array<F, Nd>(&)[Md]>(frag[k]);
    auto& stat_k = reinterpret_cast<Array<S, 1>(&)[Ns * Ms]>(stat[k / div]);
    auto& data_k = data[k];

    PRAGMA_UNROLL
    for (int m = 0; m < Md; ++m) {
      Array<F, Nd> tmp;
      PRAGMA_UNROLL
      for (int i = 0; i < Nd; i += 4) {
        auto& src = reinterpret_cast<Array<fp8_e4m3_t, 4>&>(data_k[m][i]);
        reinterpret_cast<Array<half, 4>&>(tmp[i]) = cvt_f16x4_e4m3<false>(src);
      }
      PRAGMA_UNROLL
      for (int i = 0; i < Nd; i += 2) {
        auto scale = __ushort_as_half(stat_k[(m * Nd + i) / Nf][0]);
        tmp[i] = __hmul(tmp[i], scale);
        tmp[i + 1] = __hmul(tmp[i + 1], scale);
      }
      frag_k[m] = tmp;
    }
  }
};

// E2M1 scales absorb the exact 2^14 conversion factor at weight preparation.
// The packed half values below are E2M1 / 2^14, including signed subnormals.
// Preparation must reject scale magnitudes above 65504 / 16384.
struct Transform_HMMA_SIMT_B_PrescaledE2M1 {
  template <class F, int Nf, int Mf, int K, class D, int Nd, int Md, class S,
            int Ns, int Ms, int Ks>
  __device__ static void apply(Array<F, Nf> (&frag)[K][Mf], int k,
                               Array<D, Nd> (&data)[K][Md],
                               Array<S, Ns> (&stat)[Ks][Ms], int div) {
    static_assert(std::is_same_v<D, fp4_e2m1_t>);
    static_assert(std::is_same_v<F, half> && std::is_same_v<S, uint16_t>);
    static_assert(Nf * Mf == Nd * Md && Nd % 8 == 0);
    auto& dst = reinterpret_cast<Array<F, Nd>(&)[Md]>(frag[k]);
    auto& scales = reinterpret_cast<Array<S, 1>(&)[Ns * Ms]>(stat[k / div]);
    PRAGMA_UNROLL
    for (int m = 0; m < Md; ++m) {
      Array<F, Nd> tmp;
      PRAGMA_UNROLL
      for (int i = 0; i < Nd; i += 8) {
        const uint32_t x = reinterpret_cast<const uint32_t&>(data[k][m][i]);
        auto& words = reinterpret_cast<Array<uint32_t, 4>&>(tmp[i]);
        words[0] = ((x << 12) & 0x80008000U) | ((x << 9) & 0x0e000e00U);
        words[1] = ((x << 8) & 0x80008000U) | ((x << 5) & 0x0e000e00U);
        words[2] = ((x << 4) & 0x80008000U) | ((x << 1) & 0x0e000e00U);
        words[3] = (x & 0x80008000U) | ((x >> 3) & 0x0e000e00U);
      }
      PRAGMA_UNROLL
      for (int i = 0; i < Nd; i += 2) {
        uint32_t scale = scales[(m * Nd + i) / Nf][0];
        scale |= scale << 16;
        auto& word = reinterpret_cast<uint32_t&>(tmp[i]);
        asm("mul.f16x2 %0, %1, %2;" : "=r"(word) : "r"(word), "r"(scale));
      }
      dst[m] = tmp;
    }
  }
};

}  // namespace turbomind::gemm
