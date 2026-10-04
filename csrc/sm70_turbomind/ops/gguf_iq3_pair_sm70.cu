// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "gguf_lattice_compact.cuh"
#include "src/turbomind/kernels/gemm/arch/mma_sm70.h"

#define VLLM_SM70_MMA_8N8K4(C, A0, A1, B0, B1)                      \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))
namespace {
// Reader only: original 26-bit packets stay tightly packed, without warp
// shuffles.
template <int CacheHint>
__device__ uint2 direct_packet(const uint8_t* tile, int width, int octet,
                               int col) {
  const uint32_t* words = reinterpret_cast<const uint32_t*>(tile) + octet * 26;
  const int bit = col * 26;
  const uint32_t low =
      CacheHint == 1 ? __ldcs(words + (bit >> 5)) : __ldcg(words + (bit >> 5));
  const uint32_t high = CacheHint == 1 ? __ldcs(words + (bit >> 5) + 1)
                                       : __ldcg(words + (bit >> 5) + 1);
  return make_uint2(low, high);
}

template <int SplitK, bool TwoChains, bool FullActivation = false,
          int CacheHint = 1, bool ExactM8 = false>
__global__ void iq3_pair_kernel(half* __restrict__ output,
                                const half* __restrict__ input,
                                const uint8_t* __restrict__ gate,
                                const uint8_t* __restrict__ up, int hidden,
                                int k, int m, float* scratch, int* counters,
                                int partitions) {
  constexpr int NAcc = TwoChains ? 2 : 1, RowTiles = 1;
  constexpr bool M1Only = false;
  using D = vllm::sm70_gguf::LatticeCompactDecoder<21>;
  // IQ3_S uses one scale nibble per four K8 octets. Each K16 pair
  // starts at an even octet, so both packets use the same coefficient
  // position. Reusing it lets the compiler share the coefficient work;
  // the packet still supplies its own indices and signs.
  __shared__ __align__(16) uint8_t grid[D::kCodebookBytes];
  D::initialize(grid);
  static_assert(RowTiles == 1 || RowTiles == 2,
                "QPN8 gated pair supports one or two 8-row tiles");
  static_assert(!M1Only || RowTiles == 1,
                "QPN8 gated M=1 specialization uses one row tile");
  __shared__ float partials[2][SplitK][M1Only ? 32 : RowTiles * 256];

  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int projection = warp_in_block / SplitK;
  const int warp = warp_in_block - projection * SplitK;
  const int hidden_tiles = hidden >> 5;
  const int tiles_n32 = hidden_tiles * 2;
  const int tile = blockIdx.x + projection * hidden_tiles;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const int group_begin = warp * groups_per_warp;
  const int col = quadpair * 8 + row;
  const uint8_t* source = projection ? up : gate;
  const uint8_t* macro = source + int64_t{blockIdx.x} * 32 * (k / 256) * 110;
  typename D::Parameters parameters{};
  int loaded_block = -1;

  float accum[RowTiles][NAcc][8];
#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        accum[row_tile][chain][i] = 0.0f;
      }
    }
  }
  const auto* first_tile = macro + int64_t{group_begin / 16} * 32 * 110;
  auto prefetched0 =
      direct_packet<CacheHint>(first_tile, 32, (group_begin % 16) * 2, col);
  auto prefetched1 =
      direct_packet<CacheHint>(first_tile, 32, (group_begin % 16) * 2 + 1, col);
  uint4 current_a0 = make_uint4(0, 0, 0, 0),
        current_a1 = make_uint4(0, 0, 0, 0);
  if constexpr (FullActivation) {
    if (ExactM8 || row < m) {
      const half* src = input + size_t{row} * k + group_begin * 16;
      current_a0 = *reinterpret_cast<const uint4*>(src);
      current_a1 = *reinterpret_cast<const uint4*>(src + 8);
    }
  }
  for (int base = group_begin; base < group_begin + groups_per_warp;
       base += 4) {
    const int block = base / 16;
    const auto* tile_data = macro + int64_t{block} * 32 * 110;
    if (block != loaded_block) {
      parameters = D::parameters<true>(tile_data, 32, col);
      loaded_block = block;
    }
    {
      const int group = base + 0;
      const int octet = (group % 16) * 2;

      auto w0 = prefetched0, w1 = prefetched1;
      uint2 next0 = make_uint2(0, 0), next1 = make_uint2(0, 0);
      uint4 next_a0 = make_uint4(0, 0, 0, 0), next_a1 = make_uint4(0, 0, 0, 0);
      if (group + 1 < group_begin + groups_per_warp) {
        const auto* next_tile = macro + int64_t{(group + 1) / 16} * 32 * 110;
        next0 = direct_packet<CacheHint>(next_tile, 32, ((group + 1) % 16) * 2,
                                         col);
        next1 = direct_packet<CacheHint>(next_tile, 32,
                                         ((group + 1) % 16) * 2 + 1, col);
        if constexpr (FullActivation) {
          if (ExactM8 || row < m) {
            const half* src = input + size_t{row} * k + (group + 1) * 16;
            next_a0 = *reinterpret_cast<const uint4*>(src);
            next_a1 = *reinterpret_cast<const uint4*>(src + 8);
          }
        }
      }
      const auto b0 = D::fragment<half>(
          parameters, __funnelshift_r(w0.x, w0.y, (col * 26) & 31) & 0x3ffffff,
          octet, grid);
      const auto b1 = D::fragment<half>(
          parameters, __funnelshift_r(w1.x, w1.y, (col * 26) & 31) & 0x3ffffff,
          octet, grid);
      half2 weights[8];
      *reinterpret_cast<uint4*>(weights) = *reinterpret_cast<const uint4*>(&b0);
      *reinterpret_cast<uint4*>(weights + 4) =
          *reinterpret_cast<const uint4*>(&b1);
      const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
      for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
        uint4 input01 = make_uint4(0, 0, 0, 0);
        uint4 input23 = make_uint4(0, 0, 0, 0);
        const int input_row_idx = row_tile * 8 + row;
        if constexpr (FullActivation) {
          input01 = current_a0;
          input23 = current_a1;
        } else {
          if (ExactM8 || input_row_idx < m) {
            const half* input_row = input + size_t{input_row_idx} * k;
            input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
            input23 =
                *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
          }
        }
        const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
        const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2],
                            b[3]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4],
                            b[5]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6],
                            b[7]);
      }
      prefetched0 = next0;
      prefetched1 = next1;
      if constexpr (FullActivation) {
        current_a0 = next_a0;
        current_a1 = next_a1;
      }
    }
    {
      const int group = base + 1;
      const int octet = (group % 16) * 2;

      auto w0 = prefetched0, w1 = prefetched1;
      uint2 next0 = make_uint2(0, 0), next1 = make_uint2(0, 0);
      uint4 next_a0 = make_uint4(0, 0, 0, 0), next_a1 = make_uint4(0, 0, 0, 0);
      if (group + 1 < group_begin + groups_per_warp) {
        const auto* next_tile = macro + int64_t{(group + 1) / 16} * 32 * 110;
        next0 = direct_packet<CacheHint>(next_tile, 32, ((group + 1) % 16) * 2,
                                         col);
        next1 = direct_packet<CacheHint>(next_tile, 32,
                                         ((group + 1) % 16) * 2 + 1, col);
        if constexpr (FullActivation) {
          if (ExactM8 || row < m) {
            const half* src = input + size_t{row} * k + (group + 1) * 16;
            next_a0 = *reinterpret_cast<const uint4*>(src);
            next_a1 = *reinterpret_cast<const uint4*>(src + 8);
          }
        }
      }
      const auto b0 = D::fragment<half>(
          parameters, __funnelshift_r(w0.x, w0.y, (col * 26) & 31) & 0x3ffffff,
          octet, grid);
      const auto b1 = D::fragment<half>(
          parameters, __funnelshift_r(w1.x, w1.y, (col * 26) & 31) & 0x3ffffff,
          octet, grid);
      half2 weights[8];
      *reinterpret_cast<uint4*>(weights) = *reinterpret_cast<const uint4*>(&b0);
      *reinterpret_cast<uint4*>(weights + 4) =
          *reinterpret_cast<const uint4*>(&b1);
      const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
      for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
        uint4 input01 = make_uint4(0, 0, 0, 0);
        uint4 input23 = make_uint4(0, 0, 0, 0);
        const int input_row_idx = row_tile * 8 + row;
        if constexpr (FullActivation) {
          input01 = current_a0;
          input23 = current_a1;
        } else {
          if (ExactM8 || input_row_idx < m) {
            const half* input_row = input + size_t{input_row_idx} * k;
            input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
            input23 =
                *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
          }
        }
        const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
        const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2],
                            b[3]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4],
                            b[5]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6],
                            b[7]);
      }
      prefetched0 = next0;
      prefetched1 = next1;
      if constexpr (FullActivation) {
        current_a0 = next_a0;
        current_a1 = next_a1;
      }
    }
    {
      const int group = base + 2;
      const int octet = (group % 16) * 2;

      auto w0 = prefetched0, w1 = prefetched1;
      uint2 next0 = make_uint2(0, 0), next1 = make_uint2(0, 0);
      uint4 next_a0 = make_uint4(0, 0, 0, 0), next_a1 = make_uint4(0, 0, 0, 0);
      if (group + 1 < group_begin + groups_per_warp) {
        const auto* next_tile = macro + int64_t{(group + 1) / 16} * 32 * 110;
        next0 = direct_packet<CacheHint>(next_tile, 32, ((group + 1) % 16) * 2,
                                         col);
        next1 = direct_packet<CacheHint>(next_tile, 32,
                                         ((group + 1) % 16) * 2 + 1, col);
        if constexpr (FullActivation) {
          if (ExactM8 || row < m) {
            const half* src = input + size_t{row} * k + (group + 1) * 16;
            next_a0 = *reinterpret_cast<const uint4*>(src);
            next_a1 = *reinterpret_cast<const uint4*>(src + 8);
          }
        }
      }
      const auto b0 = D::fragment<half>(
          parameters, __funnelshift_r(w0.x, w0.y, (col * 26) & 31) & 0x3ffffff,
          octet, grid);
      const auto b1 = D::fragment<half>(
          parameters, __funnelshift_r(w1.x, w1.y, (col * 26) & 31) & 0x3ffffff,
          octet, grid);
      half2 weights[8];
      *reinterpret_cast<uint4*>(weights) = *reinterpret_cast<const uint4*>(&b0);
      *reinterpret_cast<uint4*>(weights + 4) =
          *reinterpret_cast<const uint4*>(&b1);
      const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
      for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
        uint4 input01 = make_uint4(0, 0, 0, 0);
        uint4 input23 = make_uint4(0, 0, 0, 0);
        const int input_row_idx = row_tile * 8 + row;
        if constexpr (FullActivation) {
          input01 = current_a0;
          input23 = current_a1;
        } else {
          if (ExactM8 || input_row_idx < m) {
            const half* input_row = input + size_t{input_row_idx} * k;
            input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
            input23 =
                *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
          }
        }
        const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
        const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2],
                            b[3]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4],
                            b[5]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6],
                            b[7]);
      }
      prefetched0 = next0;
      prefetched1 = next1;
      if constexpr (FullActivation) {
        current_a0 = next_a0;
        current_a1 = next_a1;
      }
    }
    {
      const int group = base + 3;
      const int octet = (group % 16) * 2;

      auto w0 = prefetched0, w1 = prefetched1;
      uint2 next0 = make_uint2(0, 0), next1 = make_uint2(0, 0);
      uint4 next_a0 = make_uint4(0, 0, 0, 0), next_a1 = make_uint4(0, 0, 0, 0);
      if (group + 1 < group_begin + groups_per_warp) {
        const auto* next_tile = macro + int64_t{(group + 1) / 16} * 32 * 110;
        next0 = direct_packet<CacheHint>(next_tile, 32, ((group + 1) % 16) * 2,
                                         col);
        next1 = direct_packet<CacheHint>(next_tile, 32,
                                         ((group + 1) % 16) * 2 + 1, col);
        if constexpr (FullActivation) {
          if (ExactM8 || row < m) {
            const half* src = input + size_t{row} * k + (group + 1) * 16;
            next_a0 = *reinterpret_cast<const uint4*>(src);
            next_a1 = *reinterpret_cast<const uint4*>(src + 8);
          }
        }
      }
      const auto b0 = D::fragment<half>(
          parameters, __funnelshift_r(w0.x, w0.y, (col * 26) & 31) & 0x3ffffff,
          octet, grid);
      const auto b1 = D::fragment<half>(
          parameters, __funnelshift_r(w1.x, w1.y, (col * 26) & 31) & 0x3ffffff,
          octet, grid);
      half2 weights[8];
      *reinterpret_cast<uint4*>(weights) = *reinterpret_cast<const uint4*>(&b0);
      *reinterpret_cast<uint4*>(weights + 4) =
          *reinterpret_cast<const uint4*>(&b1);
      const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
      for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
        uint4 input01 = make_uint4(0, 0, 0, 0);
        uint4 input23 = make_uint4(0, 0, 0, 0);
        const int input_row_idx = row_tile * 8 + row;
        if constexpr (FullActivation) {
          input01 = current_a0;
          input23 = current_a1;
        } else {
          if (ExactM8 || input_row_idx < m) {
            const half* input_row = input + size_t{input_row_idx} * k;
            input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
            input23 =
                *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
          }
        }
        const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
        const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2],
                            b[3]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4],
                            b[5]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6],
                            b[7]);
      }
      prefetched0 = next0;
      prefetched1 = next1;
      if constexpr (FullActivation) {
        current_a0 = next_a0;
        current_a1 = next_a1;
      }
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        accum[row_tile][0][i] += accum[row_tile][chain][i];
      }
    }
  }
  if constexpr (M1Only) {
    if ((lane & 17) == 0) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
        for (int offset = 0; offset < 2; ++offset) {
          const int i = pair * 4 + offset;
          const int output_col =
              offset | (((lane >> 1) & 1) << 1) | (pair << 2);
          partials[projection][warp][quadpair * 8 + output_col] =
              accum[0][0][i];
        }
      }
    }
  } else {
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int output_row =
            row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int output_col =
            (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        partials[projection][warp][output_row * 32 + quadpair * 8 +
                                   output_col] = accum[row_tile][0][i];
      }
    }
  }
  __syncthreads();

  constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
  for (int element = threadIdx.x; element < kOutputElements;
       element += blockDim.x) {
    float gate = 0.0f;
    float up = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      gate += partials[0][k_warp][element];
      up += partials[1][k_warp][element];
    }
    if constexpr (M1Only) {
      const float g = __half2float(__float2half(gate));
      const half silu = __float2half(g / (1.f + expf(-g)));
      output[blockIdx.x * 32 + element] = __hmul(silu, __float2half(up));
    } else {
      const int output_row = element >> 5;
      const int output_col = element & 31;
      if (ExactM8 || output_row < m) {
        const float g = __half2float(__float2half(gate));
        const half silu = __float2half(g / (1.f + expf(-g)));
        output[static_cast<size_t>(output_row) * hidden + blockIdx.x * 32 +
               output_col] = __hmul(silu, __float2half(up));
      }
    }
  }
}

}  // namespace
void k64_chunks_gguf_iq3_pair_sm70_out(torch::Tensor out, torch::Tensor input,
                                       torch::Tensor gate, torch::Tensor up,
                                       int64_t split, bool prefetch,
                                       torch::Tensor scratch,
                                       torch::Tensor counters,
                                       int64_t partitions, bool staged) {
  TORCH_CHECK(
      out.is_cuda() && input.is_cuda() && gate.is_cuda() && up.is_cuda(),
      "IQ3 pair requires CUDA tensors");
  TORCH_CHECK(out.device() == input.device() &&
                  gate.device() == input.device() &&
                  up.device() == input.device(),
              "IQ3 pair device mismatch");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16 &&
                  gate.scalar_type() == torch::kUInt8 &&
                  up.scalar_type() == torch::kUInt8,
              "IQ3 pair dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  gate.is_contiguous() && up.is_contiguous(),
              "IQ3 pair requires contiguous tensors");
  TORCH_CHECK(
      input.dim() == 2 && out.dim() == 2 && input.size(0) == out.size(0),
      "IQ3 pair shape mismatch");
  const int m = input.size(0), n = out.size(1), k = input.size(1);
  TORCH_CHECK(m >= 1 && m <= 8 && n > 0 && n % 32 == 0 && k > 0 && k % 256 == 0,
              "IQ3 pair requires M1..8, N32, K256");
  TORCH_CHECK(gate.numel() >= int64_t{n} * (k / 256) * 110 &&
                  up.numel() == gate.numel(),
              "IQ3 pair storage mismatch");
  const c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(partitions >= 1 && partitions <= k / 256 &&
                  scratch.device() == input.device() &&
                  counters.device() == input.device() &&
                  scratch.scalar_type() == torch::kFloat32 &&
                  counters.scalar_type() == torch::kInt32 &&
                  scratch.is_contiguous() && counters.is_contiguous() &&
                  scratch.numel() >= int64_t{n / 32} * partitions * 512 &&
                  counters.numel() >= n / 32,
              "IQ3 pair workspace mismatch");
  TORCH_CHECK(partitions == 1 || partitions == 2,
              "Literal QPN uses one CTA per N32 tile");
  const auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(split == 8 && k % 512 == 0 && partitions == 1,
              "K64 probe uses split8/streaming loads");
  // Select the row specialization from the actual input shape.
  if (staged && m == 8)
    iq3_pair_kernel<8, false, true, 1, true><<<dim3(n / 32), 512, 0, stream>>>(
        (half*)out.data_ptr(), (const half*)input.data_ptr(),
        gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), n, k, m,
        scratch.data_ptr<float>(), counters.data_ptr<int>(), 1);
  else if (staged)
    iq3_pair_kernel<8, false, true, 1><<<dim3(n / 32), 512, 0, stream>>>(
        (half*)out.data_ptr(), (const half*)input.data_ptr(),
        gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), n, k, m,
        scratch.data_ptr<float>(), counters.data_ptr<int>(), 1);
  else
    iq3_pair_kernel<8, true, false, 1><<<dim3(n / 32), 512, 0, stream>>>(
        (half*)out.data_ptr(), (const half*)input.data_ptr(),
        gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), n, k, m,
        scratch.data_ptr<float>(), counters.data_ptr<int>(), 1);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef GGUF_IQ3_PAIR_RESEARCH
TORCH_LIBRARY(iq3_pair_k64_chunks_research, ops) {
  ops.def(
      "pair(Tensor(a!) out, Tensor input, Tensor gate, Tensor up, int split, "
      "bool prefetch, Tensor(b!) scratch, Tensor(c!) counters, int partitions, "
      "bool staged) "
      "-> ()");
  ops.impl("pair", torch::kCUDA, &k64_chunks_gguf_iq3_pair_sm70_out);
}
#endif
