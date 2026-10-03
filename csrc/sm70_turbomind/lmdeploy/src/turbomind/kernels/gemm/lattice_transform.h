// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include "src/turbomind/kernels/gemm/lattice_codebooks.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/core/array.h"
namespace turbomind::gemm {

template<int Type, int GroupSize>
struct Transform_HMMA_SM70_Lattice {
  using Codebook = LatticeCodebook<Type>;
  static constexpr int kCodebookBytes = Codebook::kBytes;
  static constexpr auto kQuantType = static_cast<QuantType>(8 + (
      Type == 16 ? 0 : Type == 17 ? 1 : Type == 18 ? 2 : Type == 19 ? 3 :
      Type == 21 ? 4 : Type == 22 ? 5 : 6));
  __device__ static void initialize(uint8_t* shared) {
    for (int i = threadIdx.x; i < kCodebookBytes; i += blockDim.x)
      shared[i] = Codebook::value(i);
    __syncthreads();
  }
  template<class F, int Nf, int Mf, int K, class D, int Nd, int Md,
           class S, int Ns, int Ms, int Ks>
  __device__ static void apply(Array<F,Nf> (&frag)[K][Mf], int k,
                              Array<D,Nd> (&data)[K][Md],
                              Array<S,Ns> (&stat)[Ks][Ms], int div,
                              const uint8_t* grid) {
    using Carrier = std::conditional_t<Codebook::kWidth == 4,uint4_t,uint2_t>;
    static_assert(std::is_same_v<D,Carrier> && std::is_same_v<F,half>);
    static_assert(Nd == 8 && Nf == 8 && Mf == Md);
    static_assert(sizeof(S) == (Codebook::kWidth == 4 ? 2 :
                  (Type == 19 || Type == 29 ? 2 : 4)));
    auto& dst = reinterpret_cast<Array<F,Nd> (&)[Md]>(frag[k]);
    auto& stats = reinterpret_cast<Array<S,1> (&)[Ns*Ms]>(stat[k/div]);
    const int base = (k * Nd) % GroupSize;
    PRAGMA_UNROLL
    for (int m = 0; m < Md; ++m) {
      const uint64_t metadata = stats[m][0];
      uint32_t packet;
      if constexpr (Codebook::kWidth == 4)
        packet = (const uint32_t&)data[k][m];
      else
        packet = (const uint16_t&)data[k][m];
      uint32_t scale = metadata & 65535U;
      scale |= scale << 16;
      // A lattice row is exactly four/eight consecutive biased bytes.
      // Fetch the row as aligned words once, then unpack FP16 pairs in
      // registers instead of repeatedly reading adjacent shared halfwords.
      uint64_t table_values;
      if constexpr (Codebook::kWidth == 8) {
        const int index = Type == 19 || Type == 29 ? packet & 2047 :
            (packet & 255) | (((metadata >> (16+2*(base/8))) & 3) << 8);
        table_values = *reinterpret_cast<const uint64_t*>(grid+index*8);
      } else {
        const int first = (packet & 255) | (((packet >> 24) & 1) << 8);
        const int second = ((packet >> 8) & 255) | (((packet >> 25) & 1) << 8);
        const uint32_t low = *reinterpret_cast<const uint32_t*>(grid+first*4);
        const uint32_t high = *reinterpret_cast<const uint32_t*>(grid+second*4);
        table_values = low | (static_cast<uint64_t>(high) << 32);
      }
      Array<F,8> decoded;
      PRAGMA_UNROLL
      for (int i = 0; i < 8; i += 2) {
        constexpr uint32_t magic = 0x64006400U;
        constexpr uint32_t bias = 0x64806480U;
        const uint32_t bytes = static_cast<uint32_t>(table_values >> (i*8));
        const uint32_t pair = __byte_perm(bytes,magic,0x7170);
        half2 values = __hsub2((const half2&)pair,(const half2&)bias);
        if constexpr (Type == 19 || Type == 29) {
          const half delta = __float2half((packet & 32768) ? -0.125f : 0.125f);
          values = __hadd2(values,__halves2half2(delta,delta));
        } else {
          const uint32_t signs = Codebook::kWidth == 8 ? packet >> 8 :
                                 packet >> 16;
          uint32_t sign_mask = ((signs >> i) & 1) << 15;
          sign_mask |= ((signs >> (i+1)) & 1) << 31;
          (uint32_t&)values ^= sign_mask;
        }
        (half2&)decoded[i] = __hmul2(values,(const half2&)scale);
      }
      dst[m] = decoded;
    }
  }
};
} // namespace turbomind::gemm
