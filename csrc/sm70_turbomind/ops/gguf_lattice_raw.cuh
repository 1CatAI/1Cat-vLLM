// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Layout/formulas follow llama.cpp bed0a856606ee4a24a164066f73d2379447033f5
// ggml-common.h / ggml-quants.c (MIT). Shared codebooks retain their MIT
// license.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>
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

  // Grid entries are exact integers; expanding once per CTA removes
  // repeated byte unpacking and half-to-float conversion in vector decode.
  __device__ static void initialize_float(float* grid) {
    for (int i = threadIdx.x; i < kCodebookBytes; i += blockDim.x) {
      const uint32_t word = Codebook::word(i / 4);
      grid[i] = float(int((word >> (8 * (i % 4))) & 255) - 128);
    }
    __syncthreads();
  }

  __device__ static turbomind::Array<float, 8> float_grid_fragment(
      const uint8_t* block, int base, const float* grid) {
    const int octet = base / 8;
    const uint8_t high = block[66 + base / 32];
    const uint8_t signs = block[(Type == 21 ? 74 : 34) + octet];
    __align__(16) turbomind::Array<float, 8> values;
    if constexpr (Type == 21) {
      const int sub = octet % 4;
      const int first = block[2 + 2 * octet] | (((high >> (2 * sub)) & 1) << 8);
      const int second =
          block[3 + 2 * octet] | (((high >> (2 * sub + 1)) & 1) << 8);
      const float4 a = *reinterpret_cast<const float4*>(grid + first * 4);
      const float4 b = *reinterpret_cast<const float4*>(grid + second * 4);
      *reinterpret_cast<float4*>(&values[0]) = a;
      *reinterpret_cast<float4*>(&values[4]) = b;
    } else {
      const int index =
          block[2 + octet] | (((high >> (2 * (octet % 4))) & 3) << 8);
      const float4 a = *reinterpret_cast<const float4*>(grid + index * 8);
      const float4 b = *reinterpret_cast<const float4*>(grid + index * 8 + 4);
      *reinterpret_cast<float4*>(&values[0]) = a;
      *reinterpret_cast<float4*>(&values[4]) = b;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
      values[i] = __uint_as_float(__float_as_uint(values[i]) ^
                                  (((uint32_t{signs} >> i) & 1) << 31));
    return values;
  }

  // The biased byte grid consists of exact integers. This TurboMind PRMT
  // construction and half2 subtraction reconstruct them exactly: 1024+b
  // and 1152 are representable for every byte b, as is b-128. Only integer
  // unpacking uses half2; the original two-level scale and weight product
  // are still FP32. Sign restoration is an exact bit flip.
  __device__ static turbomind::Array<float, 8> table_values(uint64_t packed,
                                                            uint32_t signs,
                                                            float scale) {
    turbomind::Array<float, 8> result;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      constexpr uint32_t magic = 0x64006400U, bias = 0x64806480U;
      const uint32_t bytes = static_cast<uint32_t>(packed >> (i * 8));
      const uint32_t pair = __byte_perm(bytes, magic, 0x7170);
      half2 values = __hsub2((const half2&)pair, (const half2&)bias);
      const uint32_t mask =
          (((signs >> i) & 1) << 15) | (((signs >> (i + 1)) & 1) << 31);
      (uint32_t&)values ^= mask;
      const float2 exact = __half22float2(values);
      result[i] = scale * exact.x;
      result[i + 1] = scale * exact.y;
    }
    return result;
  }

  // Exact final FP16 operand formation, not an expanded coefficient. The
  // small coefficient times a grid integer is exactly representable in half
  // (IQ3_S <= 465, IQ2_S <= 1333/8). Original d has <= 11 significant bits,
  // so the reference FP32 product is exact too. One final half2 multiply
  // therefore matches FP32 dequantization followed by round-to-nearest half.
  // Float output and vector FMA still use the original FP32 scale formula.
  template <class Output>
  __device__ static turbomind::Array<Output, 8> table_fragment(uint64_t packed,
                                                               uint32_t signs,
                                                               float d,
                                                               int nibble) {
    if constexpr (std::is_same_v<Output, half>) {
      const float small =
          Type == 21 ? float(1 + 2 * nibble) : (0.5f + nibble) * 0.25f;
      const half2 factor = __float2half2_rn(small);
      const half2 base = __float2half2_rn(d);
      turbomind::Array<half, 8> result;
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        constexpr uint32_t magic = 0x64006400U, bias = 0x64806480U;
        const uint32_t bytes = static_cast<uint32_t>(packed >> (i * 8));
        const uint32_t pair = __byte_perm(bytes, magic, 0x7170);
        half2 values = __hsub2((const half2&)pair, (const half2&)bias);
        const uint32_t mask =
            (((signs >> i) & 1) << 15) | (((signs >> (i + 1)) & 1) << 31);
        (uint32_t&)values ^= mask;
        const half2 exact_factor = __hmul2(values, factor);
        (half2&)result[i] = __hmul2(exact_factor, base);
      }
      return result;
    } else {
      static_assert(std::is_same_v<Output, float>);
      const float scale =
          Type == 21 ? d * (1 + 2 * nibble) : (d * (0.5f + nibble)) * 0.25f;
      return table_values(packed, signs, scale);
    }
  }

  // Preserve the official multiplication order for both original scale levels.
  __device__ static float block_scale(const uint8_t* block, int base) {
    const float d = __half2float(*reinterpret_cast<const half*>(block));
    if constexpr (Type == 21) {
      const int nibble =
          (block[106 + base / 64] >> (4 * ((base / 32) % 2))) & 15;
      return d * (1 + 2 * nibble);
    } else {
      const int nibble =
          (block[74 + base / 32] >> (4 * ((base / 16) % 2))) & 15;
      return (d * (0.5f + nibble)) * 0.25f;
    }
  }

  // Eight consecutive K values. Both scales are multiplied in FP32 before
  // multiplying grid values. No FP16 expanded coefficient exists in storage.
  template <class Output = float, bool ApplyScale = true>
  __device__ static turbomind::Array<Output, 8> fragment(const uint8_t* block,
                                                         int base,
                                                         const uint8_t* grid) {
    const float d = __half2float(*reinterpret_cast<const half*>(block));
    const int octet = base / 8;
    const uint8_t high = block[66 + base / 32];
    const uint8_t signs = block[(Type == 21 ? 74 : 34) + octet];
    turbomind::Array<Output, 8> values;
    if constexpr (Type == 21) {
      const int nibble =
          (block[106 + base / 64] >> (4 * ((base / 32) % 2))) & 15;
      const int sub = octet % 4;
      const int first = block[2 + 2 * octet] | (((high >> (2 * sub)) & 1) << 8);
      const int second =
          block[3 + 2 * octet] | (((high >> (2 * sub + 1)) & 1) << 8);
      const uint32_t a = *reinterpret_cast<const uint32_t*>(grid + first * 4);
      const uint32_t b = *reinterpret_cast<const uint32_t*>(grid + second * 4);
      const uint64_t packed = a | (static_cast<uint64_t>(b) << 32);
      if constexpr (ApplyScale) {
        values = table_fragment<Output>(packed, signs, d, nibble);
      } else {
        static_assert(std::is_same_v<Output, float>);
        values = table_values(packed, signs, 1.f);
      }
    } else {
      const int nibble =
          (block[74 + base / 32] >> (4 * ((base / 16) % 2))) & 15;
      const int index =
          block[2 + octet] | (((high >> (2 * (octet % 4))) & 3) << 8);
      const uint64_t packed =
          *reinterpret_cast<const uint64_t*>(grid + index * 8);
      if constexpr (ApplyScale) {
        values = table_fragment<Output>(packed, signs, d, nibble);
      } else {
        static_assert(std::is_same_v<Output, float>);
        values = table_values(packed, signs, 1.f);
      }
    }
    return values;
  }
};
}  // namespace vllm::sm70_gguf
