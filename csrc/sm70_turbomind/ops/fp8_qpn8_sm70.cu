// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The QPN8 execution layout is derived from dnv2003/v100-skinny (MIT) and
// its block-scale adaptation in haohervchb/sglang-V100. Automatic operator
// dispatch is restricted to accepted SM70 tensor/layout contracts and can be
// disabled with VLLM_SM70_FP8_QPN8=0.
// See LICENSE.v100-skinny in this directory for the retained MIT notice.

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <mutex>

namespace {

__device__ __forceinline__ void fp8x8_to_half2x4(uint2 q, half2 out[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned b0 = (q.x >> (8 * i)) & 0xffu;
    const unsigned b1 = (q.y >> (8 * i)) & 0xffu;
    const unsigned h0 = ((b0 & 0x80u) << 8) | ((b0 & 0x7fu) << 7);
    const unsigned h1 = ((b1 & 0x80u) << 8) | ((b1 & 0x7fu) << 7);
    const unsigned packed = h0 | (h1 << 16);
    out[i] = *reinterpret_cast<const half2*>(&packed);
  }
}

__device__ __forceinline__ void fp8x8_to_half2x4_fast(uint2 q, half2 out[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x3f803f80u;
  unsigned permuted[4];
  permuted[0] = __byte_perm(q.x, q.y, 0x0400);
  permuted[1] = __byte_perm(q.x, q.y, 0x0501);
  permuted[2] = __byte_perm(q.x, q.y, 0x0602);
  permuted[3] = __byte_perm(q.x, q.y, 0x0703);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned value =
        ((permuted[i] << 8) & kSign) | ((permuted[i] << 7) & kExponentMantissa);
    out[i] = *reinterpret_cast<const half2*>(&value);
  }
}

constexpr int kQpn8PrepareThreads = 256;

__device__ __forceinline__ int qpn8_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ int qpn8_lane_from_col(int col) {
  return (col & 3) | (((col >> 3) & 3) << 2) | (((col >> 2) & 1) << 4);
}

__device__ __forceinline__ int qpn8_physical_k(int logical_k) {
  const int local = logical_k & 7;
  return (logical_k & 8) + (local >> 1) + ((local & 1) << 2);
}

__global__ void fp8_qpn8_prepack_sm70_kernel(
    uint8_t* __restrict__ codes, const uint8_t* __restrict__ qweight, int n,
    int k) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k;
  if (index >= numel) {
    return;
  }

  const int row = static_cast<int>(index / k);
  const int logical_k = static_cast<int>(index % k);
  const int tile = row >> 5;
  const int lane = qpn8_lane_from_col(row & 31);
  const int group = logical_k >> 4;
  const int physical_k = qpn8_physical_k(logical_k & 15);
  const int groups_k16 = k >> 4;
  const size_t packed_index =
      (((static_cast<size_t>(tile) * groups_k16 + group) * 32 + lane) * 16 +
       physical_k);
  codes[packed_index] = qweight[index];
}

__global__ void fp8_qpn8_scale_sm70_kernel(half* __restrict__ group_scales,
                                           const float* __restrict__ scales,
                                           int n_blocks, int k_blocks) {
  const int n_tiles = n_blocks * 4;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int numel = k_blocks * n_tiles;
  if (index >= numel) {
    return;
  }

  const int k_block = index / n_tiles;
  const int n_tile = index - k_block * n_tiles;
  group_scales[index] =
      __float2half(scales[(n_tile >> 2) * k_blocks + k_block] * 256.0f);
}

__global__ void fp8_qpn8_channel_scale_sm70_kernel(
    half* __restrict__ channel_scales, const float* __restrict__ scales,
    int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < n) {
    channel_scales[col] = __float2half(scales[col] * 256.0f);
  }
}

__global__ void fp8_qpn8_dequantize_sm70_kernel(
    half* __restrict__ output, const uint8_t* __restrict__ codes,
    const half* __restrict__ group_scales, int n, int k, bool channel_scales) {
  const size_t word_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t word_count = static_cast<size_t>(n) * k / 16;
  if (word_index >= word_count) {
    return;
  }

  const int groups_k16 = k >> 4;
  const int lane = static_cast<int>(word_index & 31);
  const size_t tile_word = word_index >> 5;
  const int group = static_cast<int>(tile_word % groups_k16);
  const int tile = static_cast<int>(tile_word / groups_k16);
  const int col = tile * 32 + qpn8_col_from_lane(lane);
  const int tiles_n32 = n >> 5;
  const half scale = channel_scales
                         ? group_scales[col]
                         : group_scales[(group >> 3) * tiles_n32 + tile];
  const half2 scale2 = __halves2half2(scale, scale);
  const uint4 packed = reinterpret_cast<const uint4*>(codes)[word_index];
  half2 weights[8];
  fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
  fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);

#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const half2 value = __hmul2(weights[pair], scale2);
    const int k_base = group * 16 + pair * 2;
    output[static_cast<size_t>(k_base) * n + col] = __low2half(value);
    output[static_cast<size_t>(k_base + 1) * n + col] = __high2half(value);
  }
}

__global__ void fp8_qpn8_silu_and_mul_sm70_kernel(
    half* __restrict__ output, const half* __restrict__ gate_up, int rows,
    int hidden) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const half* row_input = gate_up + static_cast<size_t>(row) * hidden * 2;
  half* row_output = output + static_cast<size_t>(row) * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float gate = __half2float(row_input[col]);
    const float silu = gate / (1.0f + __expf(-gate));
    row_output[col] = __hmul(__float2half(silu), row_input[hidden + col]);
  }
}

#define VLLM_SM70_MMA_8N8K4(C, A0, A1, B0, B1)                      \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

__global__ void fp8_qpn8_ba_split_copy_sm70_kernel(
    const half* __restrict__ qkvz, const half* __restrict__ ba,
    half* __restrict__ qkv, half* __restrict__ z, half* __restrict__ b,
    half* __restrict__ a, int m) {
  constexpr int kQkvN = 2560;
  constexpr int kZN = 1536;
  constexpr int kQkvzN = kQkvN + kZN;
  constexpr int kBAN = 24;
  constexpr int kOutputN = kQkvzN + kBAN;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m * kOutputN) {
    return;
  }
  const int row = index / kOutputN;
  const int col = index - row * kOutputN;
  if (col < kQkvN) {
    qkv[static_cast<size_t>(row) * kQkvN + col] =
        qkvz[static_cast<size_t>(row) * kQkvzN + col];
  } else if (col < kQkvzN) {
    z[static_cast<size_t>(row) * kZN + col - kQkvN] =
        qkvz[static_cast<size_t>(row) * kQkvzN + col];
  } else if (col < kQkvzN + kBAN / 2) {
    b[static_cast<size_t>(row) * (kBAN / 2) + col - kQkvzN] =
        ba[static_cast<size_t>(row) * kBAN + col - kQkvzN];
  } else {
    a[static_cast<size_t>(row) * (kBAN / 2) + col - kQkvzN - kBAN / 2] =
        ba[static_cast<size_t>(row) * kBAN + col - kQkvzN];
  }
}

template <int SplitK, int NAcc, bool FastDecoder, bool PrefetchCodes,
          bool M1Only = false, bool FusedBA = false, bool SplitOutputs = false>
__global__ void fp8_qpn8_sm70_kernel(
    const uint8_t* __restrict__ codes, const half* __restrict__ group_scales,
    const half* __restrict__ input, half* __restrict__ output,
    half* __restrict__ z_output, const half* __restrict__ ba_weight,
    half* __restrict__ ba_output, half* __restrict__ b_output,
    half* __restrict__ a_output, int ba_n, int qkv_n, int n, int k, int m,
    bool channel_scales) {
  __shared__ float partials[SplitK][M1Only ? 32 : 256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  if constexpr (FusedBA) {
    const int qpn_tiles = n >> 5;
    if (tile >= qpn_tiles) {
      constexpr int kBARowsPerBlock = 2;
      constexpr int kBAThreadsPerRow = 256;
      const int ba_block = tile - qpn_tiles;
      const int ba_group = threadIdx.x / kBAThreadsPerRow;
      const int ba_thread = threadIdx.x % kBAThreadsPerRow;
      const int ba_warp = ba_thread >> 5;
      const int ba_row = ba_block * kBARowsPerBlock + ba_group;
      float value = 0.0f;
      const half2* input2 = reinterpret_cast<const half2*>(input);
      const half2* weight2 = reinterpret_cast<const half2*>(
          ba_weight + static_cast<size_t>(ba_row) * k);
      for (int pair = ba_thread; pair < k / 2; pair += kBAThreadsPerRow) {
        const float2 x = __half22float2(__ldg(input2 + pair));
        const float2 weight = __half22float2(__ldg(weight2 + pair));
        value = fmaf(x.x, weight.x, value);
        value = fmaf(x.y, weight.y, value);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
      }
      if (lane == 0) {
        partials[ba_group][ba_warp] = value;
      }
      __syncthreads();
      if (ba_warp == 0) {
        value = lane < 8 ? partials[ba_group][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0 && ba_row < ba_n) {
          if constexpr (SplitOutputs) {
            if (ba_row < ba_n / 2) {
              b_output[ba_row] = __float2half(value);
            } else {
              a_output[ba_row - ba_n / 2] = __float2half(value);
            }
          } else {
            ba_output[ba_row] = __float2half(value);
          }
        }
      }
      return;
    }
  }
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const int group_begin = warp * groups_per_warp;
  const int tiles_n32 = n >> 5;
  const uint4* code_ptr = reinterpret_cast<const uint4*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half* scale_ptr = group_scales + tile;

  float accum[NAcc][8];
#pragma unroll
  for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[chain][i] = 0.0f;
    }
  }
  int loaded_scale_group = -1;
  half loaded_scale =
      channel_scales
          ? __ldg(group_scales + tile * 32 + qpn8_col_from_lane(lane))
          : __float2half(0.0f);
  uint4 prefetched = make_uint4(0, 0, 0, 0);
  if constexpr (PrefetchCodes) {
    prefetched = __ldcs(code_ptr + static_cast<size_t>(group_begin) * 32);
  }

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const int scale_group = group >> 3;
    if (!channel_scales && scale_group != loaded_scale_group) {
      loaded_scale =
          __ldg(scale_ptr + static_cast<size_t>(scale_group) * tiles_n32);
      loaded_scale_group = scale_group;
    }

    const uint4 packed =
        PrefetchCodes ? prefetched
                      : __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    uint4 next = make_uint4(0, 0, 0, 0);
    if constexpr (PrefetchCodes) {
      if (group + 1 < group_begin + groups_per_warp) {
        next = __ldcs(code_ptr + static_cast<size_t>(group + 1) * 32);
      }
    }
    half2 weights[8];
    if constexpr (FastDecoder) {
      fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
      fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);
    } else {
      fp8x8_to_half2x4(make_uint2(packed.x, packed.y), weights);
      fp8x8_to_half2x4(make_uint2(packed.z, packed.w), weights + 4);
    }

    const half2 scale2 = __halves2half2(loaded_scale, loaded_scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      weights[i] = __hmul2(weights[i], scale2);
    }

    uint4 input01 = make_uint4(0, 0, 0, 0);
    uint4 input23 = make_uint4(0, 0, 0, 0);
    if (row < m) {
      const half* input_row = input + static_cast<size_t>(row) * k;
      input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
      input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
    }

    const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
    VLLM_SM70_MMA_8N8K4(accum[0], a0[0], a0[1], b[0], b[1]);
    VLLM_SM70_MMA_8N8K4(accum[1 % NAcc], a0[2], a0[3], b[2], b[3]);
    VLLM_SM70_MMA_8N8K4(accum[2 % NAcc], a1[0], a1[1], b[4], b[5]);
    VLLM_SM70_MMA_8N8K4(accum[3 % NAcc], a1[2], a1[3], b[6], b[7]);
    if constexpr (PrefetchCodes) {
      prefetched = next;
    }
  }

#pragma unroll
  for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[0][i] += accum[chain][i];
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
          partials[warp][quadpair * 8 + output_col] = accum[0][i];
        }
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int output_row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int output_col =
          (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      partials[warp][output_row * 32 + quadpair * 8 + output_col] = accum[0][i];
    }
  }
  __syncthreads();

  constexpr int kOutputElements = M1Only ? 32 : 256;
  for (int element = threadIdx.x; element < kOutputElements;
       element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      value += partials[k_warp][element];
    }
    if constexpr (M1Only) {
      const int output_col = tile * 32 + element;
      if constexpr (SplitOutputs) {
        if (output_col < qkv_n) {
          output[output_col] = __float2half(value);
        } else {
          z_output[output_col - qkv_n] = __float2half(value);
        }
      } else {
        output[output_col] = __float2half(value);
      }
    } else {
      const int output_row = element >> 5;
      const int output_col = element & 31;
      if (output_row < m) {
        output[static_cast<size_t>(output_row) * n + tile * 32 + output_col] =
            __float2half(value);
      }
    }
  }
}

template <int SplitK, int NAcc, bool FastDecoder, bool PrefetchCodes,
          bool M1Only = false>
void launch_fp8_qpn8_sm70(const uint8_t* codes, const half* group_scales,
                          const half* input, half* output, int n, int k, int m,
                          bool channel_scales, cudaStream_t stream) {
  fp8_qpn8_sm70_kernel<SplitK, NAcc, FastDecoder, PrefetchCodes, M1Only>
      <<<(n / 32), (32 * SplitK), 0, stream>>>(
          codes, group_scales, input, output, nullptr, nullptr, nullptr,
          nullptr, nullptr, 0, n, n, k, m, channel_scales);
}

void launch_fp8_qpn8_ba_split_sm70(const uint8_t* codes,
                                   const half* group_scales, const half* input,
                                   half* qkv_output, half* z_output,
                                   const half* ba_weight, half* b_output,
                                   half* a_output, int ba_n, int qkv_n, int n,
                                   int k, bool channel_scales,
                                   cudaStream_t stream) {
  fp8_qpn8_sm70_kernel<16, 2, true, false, true, true, true>
      <<<(n / 32 + (ba_n + 1) / 2), 512, 0, stream>>>(
          codes, group_scales, input, qkv_output, z_output, ba_weight, nullptr,
          b_output, a_output, ba_n, qkv_n, n, k, 1, channel_scales);
}

template <int SplitK, int NAcc, bool FastDecoder, bool PrefetchCodes,
          bool M1Only = false>
__global__ void fp8_qpn8_gated_pair_sm70_kernel(
    const uint8_t* __restrict__ codes, const half* __restrict__ group_scales,
    const half* __restrict__ input, half* __restrict__ output, int hidden,
    int k, int m, bool channel_scales) {
  __shared__ float partials[2][SplitK][M1Only ? 32 : 256];

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
  const uint4* code_ptr = reinterpret_cast<const uint4*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half* scale_ptr = group_scales + tile;

  float accum[NAcc][8];
#pragma unroll
  for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[chain][i] = 0.0f;
    }
  }
  int loaded_scale_group = -1;
  half loaded_scale =
      channel_scales
          ? __ldg(group_scales + tile * 32 + qpn8_col_from_lane(lane))
          : __float2half(0.0f);
  uint4 prefetched = make_uint4(0, 0, 0, 0);
  if constexpr (PrefetchCodes) {
    prefetched = __ldcs(code_ptr + static_cast<size_t>(group_begin) * 32);
  }

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const int scale_group = group >> 3;
    if (!channel_scales && scale_group != loaded_scale_group) {
      loaded_scale =
          __ldg(scale_ptr + static_cast<size_t>(scale_group) * tiles_n32);
      loaded_scale_group = scale_group;
    }

    const uint4 packed =
        PrefetchCodes ? prefetched
                      : __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    uint4 next = make_uint4(0, 0, 0, 0);
    if constexpr (PrefetchCodes) {
      if (group + 1 < group_begin + groups_per_warp) {
        next = __ldcs(code_ptr + static_cast<size_t>(group + 1) * 32);
      }
    }
    half2 weights[8];
    if constexpr (FastDecoder) {
      fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
      fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);
    } else {
      fp8x8_to_half2x4(make_uint2(packed.x, packed.y), weights);
      fp8x8_to_half2x4(make_uint2(packed.z, packed.w), weights + 4);
    }
    const half2 scale2 = __halves2half2(loaded_scale, loaded_scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      weights[i] = __hmul2(weights[i], scale2);
    }

    uint4 input01 = make_uint4(0, 0, 0, 0);
    uint4 input23 = make_uint4(0, 0, 0, 0);
    if (row < m) {
      const half* input_row = input + static_cast<size_t>(row) * k;
      input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
      input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
    }
    const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
    VLLM_SM70_MMA_8N8K4(accum[0], a0[0], a0[1], b[0], b[1]);
    VLLM_SM70_MMA_8N8K4(accum[1 % NAcc], a0[2], a0[3], b[2], b[3]);
    VLLM_SM70_MMA_8N8K4(accum[2 % NAcc], a1[0], a1[1], b[4], b[5]);
    VLLM_SM70_MMA_8N8K4(accum[3 % NAcc], a1[2], a1[3], b[6], b[7]);
    if constexpr (PrefetchCodes) {
      prefetched = next;
    }
  }

#pragma unroll
  for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[0][i] += accum[chain][i];
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
          partials[projection][warp][quadpair * 8 + output_col] = accum[0][i];
        }
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int output_row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int output_col =
          (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      partials[projection][warp][output_row * 32 + quadpair * 8 + output_col] =
          accum[0][i];
    }
  }
  __syncthreads();

  constexpr int kOutputElements = M1Only ? 32 : 256;
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
      const float silu = gate / (1.0f + __expf(-gate));
      output[blockIdx.x * 32 + element] = __float2half(silu * up);
    } else {
      const int output_row = element >> 5;
      const int output_col = element & 31;
      if (output_row < m) {
        const float silu = gate / (1.0f + __expf(-gate));
        output[static_cast<size_t>(output_row) * hidden + blockIdx.x * 32 +
               output_col] = __float2half(silu * up);
      }
    }
  }
}

template <int SplitK, int NAcc, bool FastDecoder, bool PrefetchCodes,
          bool M1Only = false>
void launch_fp8_qpn8_gated_pair_sm70(const uint8_t* codes,
                                     const half* group_scales,
                                     const half* input, half* output,
                                     int hidden, int k, int m,
                                     bool channel_scales, cudaStream_t stream) {
  fp8_qpn8_gated_pair_sm70_kernel<SplitK, NAcc, FastDecoder, PrefetchCodes,
                                  M1Only>
      <<<(hidden / 32), (64 * SplitK), 0, stream>>>(
          codes, group_scales, input, output, hidden, k, m, channel_scales);
}

template <int SplitK, int NAcc>
__global__ void fp8_qpn8_split_cta_m1_stage1_sm70_kernel(
    const uint8_t* __restrict__ codes, const half* __restrict__ channel_scales,
    const half* __restrict__ input, float* __restrict__ partials, int n,
    int k) {
  const int lane = threadIdx.x;
  const int tile = blockIdx.x;
  const int split = blockIdx.y;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_split = groups_k16 / SplitK;
  const int group_begin = split * groups_per_split;
  const uint4* code_ptr = reinterpret_cast<const uint4*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half scale =
      __ldg(channel_scales + tile * 32 + qpn8_col_from_lane(lane));
  const half2 scale2 = __halves2half2(scale, scale);

  float accum[NAcc][8];
#pragma unroll
  for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      accum[chain][index] = 0.0f;
    }
  }

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_split;
       ++group) {
    const uint4 packed = __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    half2 weights[8];
    fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
    fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      weights[index] = __hmul2(weights[index], scale2);
    }

    uint4 input01 = make_uint4(0, 0, 0, 0);
    uint4 input23 = make_uint4(0, 0, 0, 0);
    if (row == 0) {
      input01 = *reinterpret_cast<const uint4*>(input + group * 16);
      input23 = *reinterpret_cast<const uint4*>(input + group * 16 + 8);
    }
    const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
    VLLM_SM70_MMA_8N8K4(accum[0], a0[0], a0[1], b[0], b[1]);
    VLLM_SM70_MMA_8N8K4(accum[1 % NAcc], a0[2], a0[3], b[2], b[3]);
    VLLM_SM70_MMA_8N8K4(accum[2 % NAcc], a1[0], a1[1], b[4], b[5]);
    VLLM_SM70_MMA_8N8K4(accum[3 % NAcc], a1[2], a1[3], b[6], b[7]);
  }

#pragma unroll
  for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      accum[0][index] += accum[chain][index];
    }
  }

  if ((lane & 17) == 0) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
      for (int offset = 0; offset < 2; ++offset) {
        const int index = pair * 4 + offset;
        const int output_col = offset | (((lane >> 1) & 1) << 1) | (pair << 2);
        const int column = tile * 32 + quadpair * 8 + output_col;
        partials[static_cast<size_t>(split) * n + column] = accum[0][index];
      }
    }
  }
}

template <int SplitK>
__global__ void fp8_qpn8_split_cta_m1_reduce_sm70_kernel(
    half* __restrict__ output, const float* __restrict__ partials, int n) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= n) {
    return;
  }
  float value = 0.0f;
#pragma unroll
  for (int split = 0; split < SplitK; ++split) {
    value += partials[static_cast<size_t>(split) * n + column];
  }
  output[column] = __float2half(value);
}

template <int SplitK>
__global__ void fp8_qpn8_hc_down_silu_reduce_sm70_kernel(
    half* __restrict__ lora, half* __restrict__ injection,
    const float* __restrict__ partials, int n) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= 324) {
    return;
  }
  float value = 0.0f;
#pragma unroll
  for (int split = 0; split < SplitK; ++split) {
    value += partials[static_cast<size_t>(split) * n + column];
  }
  const half rounded = __float2half(value);
  if (column < 320) {
    const float scaled = __half2float(rounded) * 0.25f;
    lora[column] = __float2half(scaled / (1.0f + __expf(-scaled)));
  } else {
    injection[column - 320] = rounded;
  }
}

template <int SplitK, int NAcc>
__global__ void fp8_qpn8_hc_up_gate_mix_sm70_kernel(
    const uint8_t* __restrict__ codes, const half* __restrict__ channel_scales,
    const half* __restrict__ lora, const half* __restrict__ xn,
    half* __restrict__ output) {
  constexpr int kHC = 4;
  constexpr int kHidden = 2560;
  constexpr int kK = 320;
  constexpr int kGroupsK16 = kK / 16;
  constexpr int kGroupsPerSplit = kGroupsK16 / SplitK;
  constexpr int kHiddenTiles = kHidden / 32;
  __shared__ float partials[kHC][SplitK][32];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int stream = warp / SplitK;
  const int split = warp - stream * SplitK;
  const int tile = blockIdx.x + stream * kHiddenTiles;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int group_begin = split * kGroupsPerSplit;
  const uint4* code_ptr = reinterpret_cast<const uint4*>(codes) +
                          static_cast<size_t>(tile) * kGroupsK16 * 32 + lane;
  const half scale =
      __ldg(channel_scales + tile * 32 + qpn8_col_from_lane(lane));
  const half2 scale2 = __halves2half2(scale, scale);

  float accum[NAcc][8];
#pragma unroll
  for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      accum[chain][index] = 0.0f;
    }
  }

#pragma unroll
  for (int group = group_begin; group < group_begin + kGroupsPerSplit;
       ++group) {
    const uint4 packed = __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    half2 weights[8];
    fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
    fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      weights[index] = __hmul2(weights[index], scale2);
    }

    uint4 input01 = make_uint4(0, 0, 0, 0);
    uint4 input23 = make_uint4(0, 0, 0, 0);
    if (row == 0) {
      input01 = *reinterpret_cast<const uint4*>(lora + group * 16);
      input23 = *reinterpret_cast<const uint4*>(lora + group * 16 + 8);
    }
    const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
    VLLM_SM70_MMA_8N8K4(accum[0], a0[0], a0[1], b[0], b[1]);
    VLLM_SM70_MMA_8N8K4(accum[1 % NAcc], a0[2], a0[3], b[2], b[3]);
    VLLM_SM70_MMA_8N8K4(accum[2 % NAcc], a1[0], a1[1], b[4], b[5]);
    VLLM_SM70_MMA_8N8K4(accum[3 % NAcc], a1[2], a1[3], b[6], b[7]);
  }

#pragma unroll
  for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      accum[0][index] += accum[chain][index];
    }
  }
  if ((lane & 17) == 0) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
      for (int offset = 0; offset < 2; ++offset) {
        const int index = pair * 4 + offset;
        const int output_col = offset | (((lane >> 1) & 1) << 1) | (pair << 2);
        partials[stream][split][quadpair * 8 + output_col] = accum[0][index];
      }
    }
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    const int inner = blockIdx.x * 32 + threadIdx.x;
    float mixed = 0.0f;
#pragma unroll
    for (int hc_stream = 0; hc_stream < kHC; ++hc_stream) {
      float gate = 0.0f;
#pragma unroll
      for (int k_split = 0; k_split < SplitK; ++k_split) {
        gate += partials[hc_stream][k_split][threadIdx.x];
      }
      gate = __half2float(__float2half(gate));
      const float normalized = __half2float(xn[hc_stream * kHidden + inner]);
      mixed += normalized / (1.0f + __expf(-gate));
    }
    output[inner] = __float2half(mixed * 0.25f);
  }
}

__global__ void fp8_qpn8_hc_down_transform_sm70_kernel(
    const half* __restrict__ down, half* __restrict__ lora,
    half* __restrict__ injection, int m, int down_stride) {
  constexpr int kLora = 320;
  constexpr int kInjection = 4;
  constexpr int kLogicalN = kLora + kInjection;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m * kLogicalN) {
    return;
  }
  const int row = index / kLogicalN;
  const int column = index - row * kLogicalN;
  const half value = down[static_cast<size_t>(row) * down_stride + column];
  if (column < kLora) {
    const float scaled = __half2float(value) * 0.25f;
    lora[static_cast<size_t>(row) * kLora + column] =
        __float2half(scaled / (1.0f + __expf(-scaled)));
  } else {
    injection[static_cast<size_t>(row) * kInjection + column - kLora] = value;
  }
}

__global__ void fp8_qpn8_hc_gate_mix_sm70_kernel(const half* __restrict__ xn,
                                                 const half* __restrict__ gate,
                                                 half* __restrict__ output,
                                                 int m) {
  constexpr int kHC = 4;
  constexpr int kHidden = 2560;
  constexpr int kN = kHC * kHidden;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m * kHidden) {
    return;
  }
  const int row = index / kHidden;
  const int inner = index - row * kHidden;
  float mixed = 0.0f;
#pragma unroll
  for (int stream = 0; stream < kHC; ++stream) {
    const size_t offset =
        static_cast<size_t>(row) * kN + stream * kHidden + inner;
    const float gate_value = __half2float(gate[offset]);
    const float normalized = __half2float(xn[offset]);
    mixed += normalized / (1.0f + __expf(-gate_value));
  }
  output[index] = __float2half(mixed * 0.25f);
}

}  // namespace

void fp8_qpn8_split_cta_m1_sm70_out(torch::Tensor out, torch::Tensor input,
                                    torch::Tensor codes,
                                    torch::Tensor channel_scales,
                                    torch::Tensor partials, int64_t split_k,
                                    int64_t accumulator_chains) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && codes.is_cuda() &&
                  channel_scales.is_cuda() && partials.is_cuda(),
              "fp8_qpn8_split_cta_m1_sm70_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8 &&
                  channel_scales.scalar_type() == torch::kFloat16 &&
                  partials.scalar_type() == torch::kFloat32,
              "fp8_qpn8_split_cta_m1_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  codes.is_contiguous() && channel_scales.is_contiguous() &&
                  partials.is_contiguous(),
              "fp8_qpn8_split_cta_m1_sm70_out: tensors must be contiguous");
  const int64_t n = out.size(1);
  const int64_t k = input.size(1);
  TORCH_CHECK(out.dim() == 2 && input.dim() == 2 && out.size(0) == 1 &&
                  input.size(0) == 1 && n > 0 && n % 32 == 0 && k > 0 &&
                  k % 16 == 0,
              "fp8_qpn8_split_cta_m1_sm70_out: shape mismatch");
  TORCH_CHECK(codes.numel() == n * k && channel_scales.numel() == n &&
                  partials.numel() >= split_k * n,
              "fp8_qpn8_split_cta_m1_sm70_out: workspace mismatch");
  TORCH_CHECK(split_k == 8 || split_k == 16 || split_k == 32,
              "fp8_qpn8_split_cta_m1_sm70_out: unsupported split_k");
  TORCH_CHECK((k / 16) % split_k == 0,
              "fp8_qpn8_split_cta_m1_sm70_out: invalid split_k");
  TORCH_CHECK(accumulator_chains == 1 || accumulator_chains == 2,
              "fp8_qpn8_split_cta_m1_sm70_out: invalid accumulator chains");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(n / 32),
                  static_cast<unsigned>(split_k));
  auto launch = [&](auto split, auto chains) {
    constexpr int kSplit = decltype(split)::value;
    constexpr int kChains = decltype(chains)::value;
    fp8_qpn8_split_cta_m1_stage1_sm70_kernel<kSplit, kChains>
        <<<grid, 32, 0, stream>>>(
            codes.data_ptr<uint8_t>(),
            reinterpret_cast<const half*>(channel_scales.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            partials.data_ptr<float>(), static_cast<int>(n),
            static_cast<int>(k));
    constexpr int kThreads = 256;
    fp8_qpn8_split_cta_m1_reduce_sm70_kernel<kSplit>
        <<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            partials.data_ptr<float>(), static_cast<int>(n));
  };

  if (split_k == 8 && accumulator_chains == 1) {
    launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
  } else if (split_k == 8) {
    launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 2>{});
  } else if (split_k == 16 && accumulator_chains == 1) {
    launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
  } else if (split_k == 16) {
    launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 2>{});
  } else if (accumulator_chains == 1) {
    launch(std::integral_constant<int, 32>{}, std::integral_constant<int, 1>{});
  } else {
    launch(std::integral_constant<int, 32>{}, std::integral_constant<int, 2>{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_qpn8_hc_down_silu_sm70_out(torch::Tensor lora, torch::Tensor injection,
                                    torch::Tensor input, torch::Tensor codes,
                                    torch::Tensor channel_scales,
                                    torch::Tensor partials) {
  TORCH_CHECK(lora.is_cuda() && injection.is_cuda() && input.is_cuda() &&
                  codes.is_cuda() && channel_scales.is_cuda() &&
                  partials.is_cuda(),
              "fp8_qpn8_hc_down_silu_sm70_out: tensors must be CUDA");
  TORCH_CHECK(lora.scalar_type() == torch::kFloat16 &&
                  injection.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8 &&
                  channel_scales.scalar_type() == torch::kFloat16 &&
                  partials.scalar_type() == torch::kFloat32,
              "fp8_qpn8_hc_down_silu_sm70_out: dtype mismatch");
  TORCH_CHECK(lora.is_contiguous() && injection.is_contiguous() &&
                  input.is_contiguous() && codes.is_contiguous() &&
                  channel_scales.is_contiguous() && partials.is_contiguous(),
              "fp8_qpn8_hc_down_silu_sm70_out: tensors must be contiguous");
  constexpr int64_t kN = 352;
  constexpr int64_t kK = 10240;
  constexpr int64_t kSplitK = 32;
  TORCH_CHECK(lora.numel() == 320 && injection.numel() == 4 &&
                  input.dim() == 2 && input.size(0) == 1 &&
                  input.size(1) == kK && codes.numel() >= kN * kK &&
                  channel_scales.numel() >= kN &&
                  partials.numel() >= kSplitK * kN,
              "fp8_qpn8_hc_down_silu_sm70_out: shape mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fp8_qpn8_split_cta_m1_stage1_sm70_kernel<32, 1>
      <<<dim3(kN / 32, kSplitK), 32, 0, stream>>>(
          codes.data_ptr<uint8_t>(),
          reinterpret_cast<const half*>(channel_scales.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          partials.data_ptr<float>(), kN, kK);
  constexpr int kThreads = 256;
  fp8_qpn8_hc_down_silu_reduce_sm70_kernel<32><<<2, kThreads, 0, stream>>>(
      reinterpret_cast<half*>(lora.data_ptr<at::Half>()),
      reinterpret_cast<half*>(injection.data_ptr<at::Half>()),
      partials.data_ptr<float>(), kN);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_qpn8_hc_up_gate_mix_sm70_out(torch::Tensor out, torch::Tensor lora,
                                      torch::Tensor xn, torch::Tensor codes,
                                      torch::Tensor channel_scales) {
  TORCH_CHECK(out.is_cuda() && lora.is_cuda() && xn.is_cuda() &&
                  codes.is_cuda() && channel_scales.is_cuda(),
              "fp8_qpn8_hc_up_gate_mix_sm70_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  lora.scalar_type() == torch::kFloat16 &&
                  xn.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8 &&
                  channel_scales.scalar_type() == torch::kFloat16,
              "fp8_qpn8_hc_up_gate_mix_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && lora.is_contiguous() &&
                  xn.is_contiguous() && codes.is_contiguous() &&
                  channel_scales.is_contiguous(),
              "fp8_qpn8_hc_up_gate_mix_sm70_out: tensors must be contiguous");
  constexpr int64_t kHC = 4;
  constexpr int64_t kHidden = 2560;
  constexpr int64_t kK = 320;
  constexpr int64_t kN = kHC * kHidden;
  TORCH_CHECK(out.numel() == kHidden && lora.numel() == kK &&
                  xn.numel() == kN && codes.numel() == kN * kK &&
                  channel_scales.numel() == kN,
              "fp8_qpn8_hc_up_gate_mix_sm70_out: shape mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(lora));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fp8_qpn8_hc_up_gate_mix_sm70_kernel<4, 2>
      <<<kHidden / 32, kHC * 4 * 32, 0, stream>>>(
          codes.data_ptr<uint8_t>(),
          reinterpret_cast<const half*>(channel_scales.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(lora.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(xn.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> fp8_qpn8_prepare_sm70(torch::Tensor qweight,
                                                 torch::Tensor scales) {
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda(),
              "fp8_qpn8_prepare_sm70: tensors must be CUDA tensors");
  TORCH_CHECK(qweight.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "fp8_qpn8_prepare_sm70: weight must be float8_e4m3fn");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
              "fp8_qpn8_prepare_sm70: scales must be float32");
  TORCH_CHECK(qweight.dim() == 2 && scales.dim() == 2,
              "fp8_qpn8_prepare_sm70: tensors must be 2D");
  TORCH_CHECK(qweight.is_contiguous() && scales.is_contiguous(),
              "fp8_qpn8_prepare_sm70: tensors must be contiguous");
  TORCH_CHECK(qweight.get_device() == scales.get_device(),
              "fp8_qpn8_prepare_sm70: tensors must share one device");

  const int64_t n = qweight.size(0);
  const int64_t k = qweight.size(1);
  const bool channel_scales = scales.size(0) == n && scales.size(1) == 1;
  const bool block_scales =
      scales.size(0) == n / 128 && scales.size(1) == k / 128;
  TORCH_CHECK(channel_scales ? (n > 0 && n % 32 == 0 && k > 0 && k % 16 == 0)
                             : (n > 0 && n % 128 == 0 && k > 0 && k % 128 == 0),
              "fp8_qpn8_prepare_sm70: channel scales require N%32=K%16=0; "
              "block scales require N%128=K%128=0");
  TORCH_CHECK(channel_scales || block_scales,
              "fp8_qpn8_prepare_sm70: expected channel scales [N, 1] or "
              "block scales [N/128, K/128]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto codes = torch::empty(
      {k, n},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kUInt8));
  auto group_scales = torch::empty(
      {channel_scales ? 1 : k / 128, channel_scales ? n : n / 32},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kFloat16));

  const int64_t weight_numel = n * k;
  const int weight_blocks = static_cast<int>(
      (weight_numel + kQpn8PrepareThreads - 1) / kQpn8PrepareThreads);
  fp8_qpn8_prepack_sm70_kernel<<<weight_blocks, kQpn8PrepareThreads, 0,
                                 stream>>>(
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(qweight.data_ptr()), static_cast<int>(n),
      static_cast<int>(k));

  const int64_t scale_numel = group_scales.numel();
  const int scale_blocks = static_cast<int>(
      (scale_numel + kQpn8PrepareThreads - 1) / kQpn8PrepareThreads);
  if (channel_scales) {
    fp8_qpn8_channel_scale_sm70_kernel<<<scale_blocks, kQpn8PrepareThreads, 0,
                                         stream>>>(
        reinterpret_cast<half*>(group_scales.data_ptr<at::Half>()),
        scales.data_ptr<float>(), static_cast<int>(n));
  } else {
    fp8_qpn8_scale_sm70_kernel<<<scale_blocks, kQpn8PrepareThreads, 0,
                                 stream>>>(
        reinterpret_cast<half*>(group_scales.data_ptr<at::Half>()),
        scales.data_ptr<float>(), static_cast<int>(n / 128),
        static_cast<int>(k / 128));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes, group_scales};
}

void fp8_qpn8_dequantize_sm70_out(torch::Tensor out, torch::Tensor codes,
                                  torch::Tensor group_scales) {
  TORCH_CHECK(out.is_cuda() && codes.is_cuda() && group_scales.is_cuda(),
              "fp8_qpn8_dequantize_sm70_out: tensors must be CUDA tensors");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8 &&
                  group_scales.scalar_type() == torch::kFloat16,
              "fp8_qpn8_dequantize_sm70_out: dtype mismatch");
  TORCH_CHECK(out.dim() == 2 && codes.dim() == 2 && group_scales.dim() == 2,
              "fp8_qpn8_dequantize_sm70_out: tensors must be 2D");
  TORCH_CHECK(out.is_contiguous() && codes.is_contiguous() &&
                  group_scales.is_contiguous(),
              "fp8_qpn8_dequantize_sm70_out: tensors must be contiguous");
  TORCH_CHECK(out.get_device() == codes.get_device() &&
                  out.get_device() == group_scales.get_device(),
              "fp8_qpn8_dequantize_sm70_out: tensors must share one device");

  const int64_t k = out.size(0);
  const int64_t n = out.size(1);
  const bool channel_scales =
      group_scales.size(0) == 1 && group_scales.size(1) == n;
  const bool block_scales =
      group_scales.size(0) == k / 128 && group_scales.size(1) == n / 32;
  TORCH_CHECK(channel_scales ? (n > 0 && n % 32 == 0 && k > 0 && k % 16 == 0)
                             : (n > 0 && n % 128 == 0 && k > 0 && k % 128 == 0),
              "fp8_qpn8_dequantize_sm70_out: channel scales require "
              "N%32=K%16=0; block scales require N%128=K%128=0");
  TORCH_CHECK(codes.numel() == k * n,
              "fp8_qpn8_dequantize_sm70_out: packed code size mismatch");
  TORCH_CHECK(channel_scales || block_scales,
              "fp8_qpn8_dequantize_sm70_out: scale shape mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  const int64_t word_count = k * n / 16;
  const int blocks = static_cast<int>((word_count + kQpn8PrepareThreads - 1) /
                                      kQpn8PrepareThreads);
  fp8_qpn8_dequantize_sm70_kernel<<<blocks, kQpn8PrepareThreads, 0,
                                    at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(group_scales.data_ptr<at::Half>()),
      static_cast<int>(n), static_cast<int>(k), channel_scales);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_qpn8_prefill_sm70_out(torch::Tensor out, int64_t dense_weight_ptr,
                               torch::Tensor input, torch::Tensor codes,
                               torch::Tensor group_scales, bool gated_silu) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda(),
              "fp8_qpn8_prefill_sm70_out: input and output must be CUDA");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16,
              "fp8_qpn8_prefill_sm70_out: input and output must be float16");
  TORCH_CHECK(input.dim() == 2 && out.dim() == 2 && dense_weight_ptr != 0,
              "fp8_qpn8_prefill_sm70_out: invalid input or workspace");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous(),
              "fp8_qpn8_prefill_sm70_out: tensors must be contiguous");

  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t n = codes.size(1);
  TORCH_CHECK(codes.dim() == 2 && codes.size(0) == k,
              "fp8_qpn8_prefill_sm70_out: code shape mismatch");
  TORCH_CHECK(out.size(0) == m && out.size(1) == (gated_silu ? n / 2 : n),
              "fp8_qpn8_prefill_sm70_out: output shape mismatch");

  auto dense_weight = torch::from_blob(
      reinterpret_cast<void*>(dense_weight_ptr), {k, n}, input.options());
  fp8_qpn8_dequantize_sm70_out(dense_weight, codes, group_scales);
  if (!gated_silu) {
    at::mm_out(out, input, dense_weight);
    return;
  }

  auto gate_up = at::mm(input, dense_weight);
  constexpr int kThreads = 256;
  fp8_qpn8_silu_and_mul_sm70_kernel<<<static_cast<int>(m), kThreads, 0,
                                      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate_up.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(n / 2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_qpn8_gemm_sm70_out(torch::Tensor out, torch::Tensor input,
                            torch::Tensor codes, torch::Tensor group_scales,
                            int64_t split_k, int64_t accumulator_chains,
                            bool fast_decoder, bool prefetch_codes) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda() && codes.is_cuda() &&
                  group_scales.is_cuda(),
              "fp8_qpn8_gemm_sm70_out: tensors must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16,
              "fp8_qpn8_gemm_sm70_out: input and output must be float16");
  TORCH_CHECK(codes.scalar_type() == torch::kUInt8,
              "fp8_qpn8_gemm_sm70_out: codes must be uint8");
  TORCH_CHECK(group_scales.scalar_type() == torch::kFloat16,
              "fp8_qpn8_gemm_sm70_out: group scales must be float16");
  TORCH_CHECK(input.dim() == 2 && out.dim() == 2 && group_scales.dim() == 2,
              "fp8_qpn8_gemm_sm70_out: expected 2D tensors");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous() &&
                  codes.is_contiguous() && group_scales.is_contiguous(),
              "fp8_qpn8_gemm_sm70_out: tensors must be contiguous");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == codes.get_device() &&
                  input.get_device() == group_scales.get_device(),
              "fp8_qpn8_gemm_sm70_out: tensors must share one device");

  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t n = out.size(1);
  TORCH_CHECK(m >= 1 && m <= 8, "fp8_qpn8_gemm_sm70_out: M must be in [1, 8]");
  TORCH_CHECK(out.size(0) == m, "fp8_qpn8_gemm_sm70_out: output M mismatch");
  TORCH_CHECK(n > 0 && n % 32 == 0,
              "fp8_qpn8_gemm_sm70_out: N must be a positive multiple of 32");
  const bool channel_scales =
      group_scales.size(0) == 1 && group_scales.size(1) == n;
  const bool block_scales =
      group_scales.size(0) == k / 128 && group_scales.size(1) == n / 32;
  TORCH_CHECK(channel_scales ? (k > 0 && k % 16 == 0) : (k > 0 && k % 128 == 0),
              "fp8_qpn8_gemm_sm70_out: channel scales require K%16=0; "
              "block scales require K%128=0");
  TORCH_CHECK(codes.numel() == n * k,
              "fp8_qpn8_gemm_sm70_out: packed code size mismatch");
  TORCH_CHECK(channel_scales || block_scales,
              "fp8_qpn8_gemm_sm70_out: scale shape mismatch");
  TORCH_CHECK(split_k == 4 || split_k == 8 || split_k == 12 || split_k == 16 ||
                  split_k == 32,
              "fp8_qpn8_gemm_sm70_out: unsupported split_k");
  TORCH_CHECK((k / 16) % split_k == 0,
              "fp8_qpn8_gemm_sm70_out: K/16 must be divisible by split_k");
  TORCH_CHECK(accumulator_chains == 1 || accumulator_chains == 2,
              "fp8_qpn8_gemm_sm70_out: accumulator_chains must be 1 or 2");
  TORCH_CHECK(!prefetch_codes || fast_decoder,
              "fp8_qpn8_gemm_sm70_out: prefetch experiment requires the "
              "fast decoder");
  const bool exact_shape_split = split_k == 12;
  TORCH_CHECK(!exact_shape_split ||
                  (accumulator_chains == 2 && fast_decoder && !prefetch_codes),
              "fp8_qpn8_gemm_sm70_out: split_k 12 requires the "
              "fast decoder, two accumulator chains, and no prefetch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* code_ptr = codes.data_ptr<uint8_t>();
  const auto* scale_ptr =
      reinterpret_cast<const half*>(group_scales.data_ptr<at::Half>());
  const auto* input_ptr =
      reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* output_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());

  // Keep the M=1 reduction order restricted to the two tuned output/down
  // projections. Extending it to the other split variants changed the frozen
  // random-sampling token stream without an end-to-end decode win.
  if (m == 1 && accumulator_chains == 2 && fast_decoder && !prefetch_codes &&
      (split_k == 12 || split_k == 16)) {
    if (split_k == 12) {
      launch_fp8_qpn8_sm70<12, 2, true, false, true>(
          code_ptr, scale_ptr, input_ptr, output_ptr, static_cast<int>(n),
          static_cast<int>(k), 1, channel_scales, stream);
    } else {
      launch_fp8_qpn8_sm70<16, 2, true, false, true>(
          code_ptr, scale_ptr, input_ptr, output_ptr, static_cast<int>(n),
          static_cast<int>(k), 1, channel_scales, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

#define VLLM_LAUNCH_QPN8(SPLIT, NACC, FAST, PREFETCH)                  \
  launch_fp8_qpn8_sm70<SPLIT, NACC, FAST, PREFETCH, false>(            \
      code_ptr, scale_ptr, input_ptr, output_ptr, static_cast<int>(n), \
      static_cast<int>(k), static_cast<int>(m), channel_scales, stream)

  if (prefetch_codes) {
    if (split_k == 4 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(4, 1, true, true);
    } else if (split_k == 4 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8(4, 2, true, true);
    } else if (split_k == 8 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(8, 1, true, true);
    } else if (split_k == 8 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8(8, 2, true, true);
    } else if (split_k == 16 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(16, 1, true, true);
    } else if (split_k == 16 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8(16, 2, true, true);
    } else if (split_k == 32 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(32, 1, true, true);
    } else {
      VLLM_LAUNCH_QPN8(32, 2, true, true);
    }
  } else if (fast_decoder) {
    if (split_k == 4 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(4, 1, true, false);
    } else if (split_k == 4 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8(4, 2, true, false);
    } else if (split_k == 8 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(8, 1, true, false);
    } else if (split_k == 8 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8(8, 2, true, false);
    } else if (split_k == 12) {
      VLLM_LAUNCH_QPN8(12, 2, true, false);
    } else if (split_k == 16 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(16, 1, true, false);
    } else if (split_k == 16 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8(16, 2, true, false);
    } else if (split_k == 32 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(32, 1, true, false);
    } else {
      VLLM_LAUNCH_QPN8(32, 2, true, false);
    }
  } else if (split_k == 4 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN8(4, 1, false, false);
  } else if (split_k == 4 && accumulator_chains == 2) {
    VLLM_LAUNCH_QPN8(4, 2, false, false);
  } else if (split_k == 8 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN8(8, 1, false, false);
  } else if (split_k == 8 && accumulator_chains == 2) {
    VLLM_LAUNCH_QPN8(8, 2, false, false);
  } else if (split_k == 16 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN8(16, 1, false, false);
  } else if (split_k == 16 && accumulator_chains == 2) {
    VLLM_LAUNCH_QPN8(16, 2, false, false);
  } else if (split_k == 32 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN8(32, 1, false, false);
  } else {
    VLLM_LAUNCH_QPN8(32, 2, false, false);
  }
#undef VLLM_LAUNCH_QPN8

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_qpn8_gemm_ba_split_sm70_out(torch::Tensor qkv_out, torch::Tensor z_out,
                                     torch::Tensor b_out, torch::Tensor a_out,
                                     torch::Tensor input, torch::Tensor codes,
                                     torch::Tensor group_scales,
                                     torch::Tensor ba_weight) {
  TORCH_CHECK(qkv_out.is_cuda() && z_out.is_cuda() && b_out.is_cuda() &&
                  a_out.is_cuda() && input.is_cuda() && codes.is_cuda() &&
                  group_scales.is_cuda() && ba_weight.is_cuda(),
              "fp8_qpn8_gemm_ba_split_sm70_out: tensors must be CUDA tensors");
  TORCH_CHECK(qkv_out.scalar_type() == torch::kFloat16 &&
                  z_out.scalar_type() == torch::kFloat16 &&
                  b_out.scalar_type() == torch::kFloat16 &&
                  a_out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  group_scales.scalar_type() == torch::kFloat16 &&
                  ba_weight.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8,
              "fp8_qpn8_gemm_ba_split_sm70_out: dtype mismatch");
  TORCH_CHECK(qkv_out.is_contiguous() && z_out.is_contiguous() &&
                  b_out.is_contiguous() && a_out.is_contiguous() &&
                  input.is_contiguous() && codes.is_contiguous() &&
                  group_scales.is_contiguous() && ba_weight.is_contiguous(),
              "fp8_qpn8_gemm_ba_split_sm70_out: tensors must be contiguous");
  TORCH_CHECK(qkv_out.get_device() == z_out.get_device() &&
                  qkv_out.get_device() == b_out.get_device() &&
                  qkv_out.get_device() == a_out.get_device() &&
                  qkv_out.get_device() == input.get_device() &&
                  qkv_out.get_device() == codes.get_device() &&
                  qkv_out.get_device() == group_scales.get_device() &&
                  qkv_out.get_device() == ba_weight.get_device(),
              "fp8_qpn8_gemm_ba_split_sm70_out: tensors must share one device");

  const int64_t k = input.size(1);
  constexpr int64_t n = 4096;
  constexpr int64_t qkv_n = 2560;
  constexpr int64_t z_n = n - qkv_n;
  constexpr int64_t ba_n = 24;
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) == 1 && (k == 2560 || k == 5120),
      "fp8_qpn8_gemm_ba_split_sm70_out: expected K=2560 or 5120");
  TORCH_CHECK(qkv_out.numel() == qkv_n && z_out.numel() == z_n &&
                  b_out.numel() == ba_n / 2 && a_out.numel() == ba_n / 2,
              "fp8_qpn8_gemm_ba_split_sm70_out: output shape mismatch");
  TORCH_CHECK(codes.dim() == 2 && codes.size(0) == k && codes.size(1) == n,
              "fp8_qpn8_gemm_ba_split_sm70_out: code shape mismatch");
  TORCH_CHECK(group_scales.dim() == 2 && group_scales.size(0) == 1 &&
                  group_scales.size(1) == n,
              "fp8_qpn8_gemm_ba_split_sm70_out: expected channel scales");
  TORCH_CHECK(ba_weight.dim() == 2 && ba_weight.size(0) == ba_n &&
                  ba_weight.size(1) == k,
              "fp8_qpn8_gemm_ba_split_sm70_out: b/a weight shape mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_fp8_qpn8_ba_split_sm70(
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(group_scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<half*>(qkv_out.data_ptr<at::Half>()),
      reinterpret_cast<half*>(z_out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(ba_weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(b_out.data_ptr<at::Half>()),
      reinterpret_cast<half*>(a_out.data_ptr<at::Half>()),
      static_cast<int>(ba_n), static_cast<int>(qkv_n), static_cast<int>(n),
      static_cast<int>(k), true, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_qpn8_dispatch_sm70_out(torch::Tensor out, int64_t dense_weight_ptr,
                                torch::Tensor input, torch::Tensor codes,
                                torch::Tensor group_scales, int64_t split_k,
                                int64_t accumulator_chains, bool prefetch_codes,
                                bool gated_silu);

void fp8_qpn8_dispatch_ba_split_sm70_out(
    torch::Tensor qkv_out, torch::Tensor z_out, torch::Tensor b_out,
    torch::Tensor a_out, torch::Tensor qkvz_staging, torch::Tensor ba_staging,
    int64_t dense_weight_ptr, torch::Tensor input, torch::Tensor codes,
    torch::Tensor group_scales, torch::Tensor ba_weight) {
  const int64_t k = input.size(1);
  constexpr int64_t n = 4096;
  constexpr int64_t qkv_n = 2560;
  constexpr int64_t z_n = n - qkv_n;
  constexpr int64_t ba_n = 24;
  const int64_t m = input.size(0);
  TORCH_CHECK(input.dim() == 2 && (k == 2560 || k == 5120) && m >= 1,
              "fp8_qpn8_dispatch_ba_split_sm70_out: bad input shape");
  TORCH_CHECK(qkv_out.is_cuda() && z_out.is_cuda() && b_out.is_cuda() &&
                  a_out.is_cuda() && qkvz_staging.is_cuda() &&
                  ba_staging.is_cuda() && input.is_cuda() && codes.is_cuda() &&
                  group_scales.is_cuda() && ba_weight.is_cuda(),
              "fp8_qpn8_dispatch_ba_split_sm70_out: tensors must be CUDA");
  TORCH_CHECK(qkv_out.scalar_type() == torch::kFloat16 &&
                  z_out.scalar_type() == torch::kFloat16 &&
                  b_out.scalar_type() == torch::kFloat16 &&
                  a_out.scalar_type() == torch::kFloat16 &&
                  qkvz_staging.scalar_type() == torch::kFloat16 &&
                  ba_staging.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  group_scales.scalar_type() == torch::kFloat16 &&
                  ba_weight.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8,
              "fp8_qpn8_dispatch_ba_split_sm70_out: dtype mismatch");
  TORCH_CHECK(
      qkv_out.is_contiguous() && z_out.is_contiguous() &&
          b_out.is_contiguous() && a_out.is_contiguous() &&
          qkvz_staging.is_contiguous() && ba_staging.is_contiguous() &&
          input.is_contiguous() && codes.is_contiguous() &&
          group_scales.is_contiguous() && ba_weight.is_contiguous(),
      "fp8_qpn8_dispatch_ba_split_sm70_out: tensors must be contiguous");
  TORCH_CHECK(
      qkv_out.numel() == m * qkv_n && z_out.numel() == m * z_n &&
          b_out.numel() == m * ba_n / 2 && a_out.numel() == m * ba_n / 2 &&
          qkvz_staging.numel() == m * n && ba_staging.numel() == m * ba_n,
      "fp8_qpn8_dispatch_ba_split_sm70_out: output shape mismatch");
  TORCH_CHECK(
      qkv_out.get_device() == z_out.get_device() &&
          qkv_out.get_device() == b_out.get_device() &&
          qkv_out.get_device() == a_out.get_device() &&
          qkv_out.get_device() == qkvz_staging.get_device() &&
          qkv_out.get_device() == ba_staging.get_device() &&
          qkv_out.get_device() == input.get_device() &&
          qkv_out.get_device() == codes.get_device() &&
          qkv_out.get_device() == group_scales.get_device() &&
          qkv_out.get_device() == ba_weight.get_device(),
      "fp8_qpn8_dispatch_ba_split_sm70_out: tensors must share one device");
  TORCH_CHECK(codes.dim() == 2 && codes.size(0) == k && codes.size(1) == n,
              "fp8_qpn8_dispatch_ba_split_sm70_out: code shape mismatch");
  TORCH_CHECK(group_scales.dim() == 2 && group_scales.size(0) == 1 &&
                  group_scales.size(1) == n,
              "fp8_qpn8_dispatch_ba_split_sm70_out: scale shape mismatch");
  TORCH_CHECK(ba_weight.dim() == 2 && ba_weight.size(0) == ba_n &&
                  ba_weight.size(1) == k,
              "fp8_qpn8_dispatch_ba_split_sm70_out: b/a weight mismatch");
  TORCH_CHECK(m <= 8 || dense_weight_ptr != 0,
              "fp8_qpn8_dispatch_ba_split_sm70_out: large-M requires the "
              "dense QPN8 workspace");

  if (m == 1) {
    static std::once_flag route_log_once;
    std::call_once(route_log_once, []() {
      std::fprintf(stderr,
                   "SM70 GDN QPN8 N4096 + FP16 b/a N24 split C++ route "
                   "enabled.\n");
      std::fflush(stderr);
    });
    fp8_qpn8_gemm_ba_split_sm70_out(qkv_out, z_out, b_out, a_out, input, codes,
                                    group_scales, ba_weight);
    return;
  }

  fp8_qpn8_dispatch_sm70_out(qkvz_staging, dense_weight_ptr, input, codes,
                             group_scales, 16, 2, false, false);
  at::mm_out(ba_staging, input, ba_weight.t());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  constexpr int threads = 256;
  const int64_t elements = m * (n + ba_n);
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  fp8_qpn8_ba_split_copy_sm70_kernel<<<blocks, threads, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(qkvz_staging.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(ba_staging.data_ptr<at::Half>()),
      reinterpret_cast<half*>(qkv_out.data_ptr<at::Half>()),
      reinterpret_cast<half*>(z_out.data_ptr<at::Half>()),
      reinterpret_cast<half*>(b_out.data_ptr<at::Half>()),
      reinterpret_cast<half*>(a_out.data_ptr<at::Half>()), static_cast<int>(m));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_qpn8_gated_pair_sm70_out(torch::Tensor out, torch::Tensor input,
                                  torch::Tensor codes,
                                  torch::Tensor group_scales, int64_t split_k,
                                  int64_t accumulator_chains, bool fast_decoder,
                                  bool prefetch_codes) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && codes.is_cuda() &&
                  group_scales.is_cuda(),
              "fp8_qpn8_gated_pair_sm70_out: tensors must be CUDA tensors");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  group_scales.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8,
              "fp8_qpn8_gated_pair_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  codes.is_contiguous() && group_scales.is_contiguous(),
              "fp8_qpn8_gated_pair_sm70_out: tensors must be contiguous");
  TORCH_CHECK(out.dim() == 2 && input.dim() == 2 && group_scales.dim() == 2,
              "fp8_qpn8_gated_pair_sm70_out: expected 2D tensors");
  TORCH_CHECK(out.get_device() == input.get_device() &&
                  out.get_device() == codes.get_device() &&
                  out.get_device() == group_scales.get_device(),
              "fp8_qpn8_gated_pair_sm70_out: tensors must share one device");

  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t hidden = out.size(1);
  const int64_t n = hidden * 2;
  TORCH_CHECK(m >= 1 && m <= 8 && out.size(0) == m,
              "fp8_qpn8_gated_pair_sm70_out: M must be in [1, 8]");
  const bool channel_scales =
      group_scales.size(0) == 1 && group_scales.size(1) == n;
  const bool block_scales =
      group_scales.size(0) == k / 128 && group_scales.size(1) == n / 32;
  TORCH_CHECK(
      hidden > 0 && hidden % 32 == 0 &&
          (channel_scales ? (k > 0 && k % 16 == 0) : (k > 0 && k % 128 == 0)),
      "fp8_qpn8_gated_pair_sm70_out: channel scales require K%16=0; "
      "block scales require K%128=0");
  TORCH_CHECK(codes.numel() == n * k,
              "fp8_qpn8_gated_pair_sm70_out: packed code size mismatch");
  TORCH_CHECK(channel_scales || block_scales,
              "fp8_qpn8_gated_pair_sm70_out: scale shape mismatch");
  TORCH_CHECK(split_k == 4 || split_k == 8 || split_k == 16,
              "fp8_qpn8_gated_pair_sm70_out: split_k must be 4, 8, or 16");
  TORCH_CHECK((k / 16) % split_k == 0,
              "fp8_qpn8_gated_pair_sm70_out: invalid split_k for K");
  TORCH_CHECK(
      accumulator_chains == 1 || accumulator_chains == 2,
      "fp8_qpn8_gated_pair_sm70_out: accumulator_chains must be 1 or 2");
  TORCH_CHECK(!prefetch_codes || fast_decoder,
              "fp8_qpn8_gated_pair_sm70_out: prefetch requires fast decoder");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* code_ptr = codes.data_ptr<uint8_t>();
  const auto* scale_ptr =
      reinterpret_cast<const half*>(group_scales.data_ptr<at::Half>());
  const auto* input_ptr =
      reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* output_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());

  // The frozen gate/up route is split-8. Preserve the generic reduction order
  // for split-4/16 rather than instantiating unaccepted M=1 variants.
  if (m == 1 && split_k == 8 && accumulator_chains == 2 && fast_decoder &&
      !prefetch_codes) {
    launch_fp8_qpn8_gated_pair_sm70<8, 2, true, false, true>(
        code_ptr, scale_ptr, input_ptr, output_ptr, static_cast<int>(hidden),
        static_cast<int>(k), 1, channel_scales, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

#define VLLM_LAUNCH_QPN8_GATED_PAIR(SPLIT, NACC, FAST, PREFETCH)            \
  launch_fp8_qpn8_gated_pair_sm70<SPLIT, NACC, FAST, PREFETCH, false>(      \
      code_ptr, scale_ptr, input_ptr, output_ptr, static_cast<int>(hidden), \
      static_cast<int>(k), static_cast<int>(m), channel_scales, stream)

  if (prefetch_codes) {
    if (split_k == 4 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8_GATED_PAIR(4, 1, true, true);
    } else if (split_k == 4 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8_GATED_PAIR(4, 2, true, true);
    } else if (split_k == 8 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8_GATED_PAIR(8, 1, true, true);
    } else if (split_k == 8 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8_GATED_PAIR(8, 2, true, true);
    } else if (split_k == 16 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8_GATED_PAIR(16, 1, true, true);
    } else {
      VLLM_LAUNCH_QPN8_GATED_PAIR(16, 2, true, true);
    }
  } else if (fast_decoder) {
    if (split_k == 4 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8_GATED_PAIR(4, 1, true, false);
    } else if (split_k == 4 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8_GATED_PAIR(4, 2, true, false);
    } else if (split_k == 8 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8_GATED_PAIR(8, 1, true, false);
    } else if (split_k == 8 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8_GATED_PAIR(8, 2, true, false);
    } else if (split_k == 16 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8_GATED_PAIR(16, 1, true, false);
    } else {
      VLLM_LAUNCH_QPN8_GATED_PAIR(16, 2, true, false);
    }
  } else if (split_k == 4 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN8_GATED_PAIR(4, 1, false, false);
  } else if (split_k == 4 && accumulator_chains == 2) {
    VLLM_LAUNCH_QPN8_GATED_PAIR(4, 2, false, false);
  } else if (split_k == 8 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN8_GATED_PAIR(8, 1, false, false);
  } else if (split_k == 8 && accumulator_chains == 2) {
    VLLM_LAUNCH_QPN8_GATED_PAIR(8, 2, false, false);
  } else if (split_k == 16 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN8_GATED_PAIR(16, 1, false, false);
  } else {
    VLLM_LAUNCH_QPN8_GATED_PAIR(16, 2, false, false);
  }
#undef VLLM_LAUNCH_QPN8_GATED_PAIR
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_qpn8_dispatch_sm70_out(torch::Tensor out, int64_t dense_weight_ptr,
                                torch::Tensor input, torch::Tensor codes,
                                torch::Tensor group_scales, int64_t split_k,
                                int64_t accumulator_chains, bool prefetch_codes,
                                bool gated_silu) {
  // Keep the M decision inside the opaque operator. AOTInductor compiles one
  // dynamic M=1..8192 range for this model, so a Python shape branch traced at
  // M=1/2 would otherwise be incorrectly reused by large-M prefill.
  if (input.size(0) <= 8) {
    if (gated_silu) {
      fp8_qpn8_gated_pair_sm70_out(out, input, codes, group_scales, split_k,
                                   accumulator_chains, true, prefetch_codes);
    } else {
      fp8_qpn8_gemm_sm70_out(out, input, codes, group_scales, split_k,
                             accumulator_chains, true, prefetch_codes);
    }
    return;
  }
  fp8_qpn8_prefill_sm70_out(out, dense_weight_ptr, input, codes, group_scales,
                            gated_silu);
}

void fp8_qpn8_hc_dispatch_sm70_out(
    torch::Tensor block_out, torch::Tensor injection_out,
    torch::Tensor down_staging, torch::Tensor lora_staging,
    torch::Tensor gate_staging, torch::Tensor partials,
    int64_t dense_weight_ptr, torch::Tensor xn, torch::Tensor down_codes,
    torch::Tensor down_scales, torch::Tensor up_codes,
    torch::Tensor up_scales) {
  constexpr int64_t kHC = 4;
  constexpr int64_t kHidden = 2560;
  constexpr int64_t kHyperHidden = kHC * kHidden;
  constexpr int64_t kLora = 320;
  constexpr int64_t kDownComputeN = 352;
  constexpr int64_t kDownSplit = 32;
  const int64_t m = xn.size(0);
  const int64_t kDownN = down_codes.size(1);
  TORCH_CHECK(m >= 1 && xn.dim() == 2 && xn.size(1) == kHyperHidden,
              "fp8_qpn8_hc_dispatch_sm70_out: expected xn [M, 10240]");
  TORCH_CHECK(block_out.dim() == 2 && block_out.size(0) == m &&
                  block_out.size(1) == kHidden && injection_out.dim() == 2 &&
                  injection_out.size(0) == m && injection_out.size(1) == kHC &&
                  down_staging.dim() == 2 && down_staging.size(0) == m &&
                  down_staging.size(1) == kDownN && lora_staging.dim() == 2 &&
                  lora_staging.size(0) == m && lora_staging.size(1) == kLora &&
                  gate_staging.dim() == 2 && gate_staging.size(0) == m &&
                  gate_staging.size(1) == kHyperHidden,
              "fp8_qpn8_hc_dispatch_sm70_out: staging shape mismatch");
  TORCH_CHECK(down_codes.numel() == kHyperHidden * kDownN &&
                  down_scales.numel() == kDownN &&
                  up_codes.numel() == kLora * kHyperHidden &&
                  up_scales.numel() == kHyperHidden &&
                  partials.scalar_type() == torch::kFloat32 &&
                  kDownN >= kDownComputeN && kDownN % 32 == 0 &&
                  partials.numel() >= kDownSplit * kDownComputeN,
              "fp8_qpn8_hc_dispatch_sm70_out: weight/workspace mismatch");
  TORCH_CHECK(block_out.scalar_type() == torch::kFloat16 &&
                  injection_out.scalar_type() == torch::kFloat16 &&
                  down_staging.scalar_type() == torch::kFloat16 &&
                  lora_staging.scalar_type() == torch::kFloat16 &&
                  gate_staging.scalar_type() == torch::kFloat16 &&
                  xn.scalar_type() == torch::kFloat16,
              "fp8_qpn8_hc_dispatch_sm70_out: activation dtype mismatch");

  if (m == 1) {
    fp8_qpn8_hc_down_silu_sm70_out(lora_staging, injection_out, xn, down_codes,
                                   down_scales, partials);
    fp8_qpn8_hc_up_gate_mix_sm70_out(block_out, lora_staging, xn, up_codes,
                                     up_scales);
    return;
  }

  fp8_qpn8_dispatch_sm70_out(down_staging, dense_weight_ptr, xn, down_codes,
                             down_scales, 32, 1, false, false);
  constexpr int kThreads = 256;
  const int down_elements = static_cast<int>(m * (kLora + kHC));
  fp8_qpn8_hc_down_transform_sm70_kernel<<<
      (down_elements + kThreads - 1) / kThreads, kThreads, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(down_staging.data_ptr<at::Half>()),
      reinterpret_cast<half*>(lora_staging.data_ptr<at::Half>()),
      reinterpret_cast<half*>(injection_out.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(kDownN));
  fp8_qpn8_dispatch_sm70_out(gate_staging, dense_weight_ptr, lora_staging,
                             up_codes, up_scales, 4, 2, false, false);
  const int block_elements = static_cast<int>(m * kHidden);
  fp8_qpn8_hc_gate_mix_sm70_kernel<<<(block_elements + kThreads - 1) / kThreads,
                                     kThreads, 0,
                                     at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(xn.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate_staging.data_ptr<at::Half>()),
      reinterpret_cast<half*>(block_out.data_ptr<at::Half>()),
      static_cast<int>(m));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef VLLM_QPN8_STANDALONE
  // Lets the exact same source file be compiled as an operator-race harness
  // before paying for a complete vLLM rebuild. Production builds register the
  // operator centrally in torch_bindings.cpp and do not define this macro.
  #ifdef VLLM_QPN8_STANDALONE_QWEN38_NAMESPACE
TORCH_LIBRARY_FRAGMENT(_C_qwen38, ops) {
  #else
TORCH_LIBRARY_FRAGMENT(_C, ops) {
  #endif
  ops.def("fp8_qpn8_prepare_sm70(Tensor qweight, Tensor scales) -> Tensor[]");
  ops.impl("fp8_qpn8_prepare_sm70", torch::kCUDA, &fp8_qpn8_prepare_sm70);
  ops.def(
      "fp8_qpn8_dequantize_sm70_out(Tensor(a!) out, Tensor codes, "
      "Tensor group_scales) -> ()");
  ops.impl("fp8_qpn8_dequantize_sm70_out", torch::kCUDA,
           &fp8_qpn8_dequantize_sm70_out);
  ops.def(
      "fp8_qpn8_prefill_sm70_out(Tensor(a!) out, int dense_weight_ptr, "
      "Tensor input, Tensor codes, Tensor group_scales, bool gated_silu) -> "
      "()");
  ops.impl("fp8_qpn8_prefill_sm70_out", torch::kCUDA,
           &fp8_qpn8_prefill_sm70_out);
  ops.def(
      "fp8_qpn8_dispatch_sm70_out(Tensor(a!) out, int dense_weight_ptr, "
      "Tensor input, Tensor codes, Tensor group_scales, int split_k, "
      "int accumulator_chains, bool prefetch_codes, bool gated_silu) -> ()");
  ops.impl("fp8_qpn8_dispatch_sm70_out", torch::kCUDA,
           &fp8_qpn8_dispatch_sm70_out);
  ops.def(
      "fp8_qpn8_gemm_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor group_scales, int split_k, int accumulator_chains, "
      "bool fast_decoder, bool prefetch_codes) -> ()");
  ops.impl("fp8_qpn8_gemm_sm70_out", torch::kCUDA, &fp8_qpn8_gemm_sm70_out);
  ops.def(
      "fp8_qpn8_gemm_ba_split_sm70_out(Tensor(a!) qkv_out, Tensor(b!) "
      "z_out, Tensor(c!) b_out, Tensor(d!) a_out, Tensor input, Tensor codes, "
      "Tensor group_scales, Tensor ba_weight) -> ()");
  ops.impl("fp8_qpn8_gemm_ba_split_sm70_out", torch::kCUDA,
           &fp8_qpn8_gemm_ba_split_sm70_out);
  ops.def(
      "fp8_qpn8_dispatch_ba_split_sm70_out(Tensor(a!) qkv_out, Tensor(b!) "
      "z_out, Tensor(c!) b_out, Tensor(d!) a_out, Tensor(e!) qkvz_staging, "
      "Tensor(f!) ba_staging, int dense_weight_ptr, Tensor input, Tensor "
      "codes, Tensor group_scales, Tensor ba_weight) -> ()");
  ops.impl("fp8_qpn8_dispatch_ba_split_sm70_out", torch::kCUDA,
           &fp8_qpn8_dispatch_ba_split_sm70_out);
  ops.def(
      "fp8_qpn8_gated_pair_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor codes, Tensor group_scales, int split_k, "
      "int accumulator_chains, bool fast_decoder, bool prefetch_codes) -> ()");
  ops.impl("fp8_qpn8_gated_pair_sm70_out", torch::kCUDA,
           &fp8_qpn8_gated_pair_sm70_out);
  ops.def(
      "fp8_qpn8_split_cta_m1_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor codes, Tensor channel_scales, Tensor partials, int split_k, "
      "int accumulator_chains) -> ()");
  ops.impl("fp8_qpn8_split_cta_m1_sm70_out", torch::kCUDA,
           &fp8_qpn8_split_cta_m1_sm70_out);
  ops.def(
      "fp8_qpn8_hc_down_silu_sm70_out(Tensor(a!) lora, Tensor(b!) "
      "injection, Tensor input, Tensor codes, Tensor channel_scales, "
      "Tensor partials) -> ()");
  ops.impl("fp8_qpn8_hc_down_silu_sm70_out", torch::kCUDA,
           &fp8_qpn8_hc_down_silu_sm70_out);
  ops.def(
      "fp8_qpn8_hc_up_gate_mix_sm70_out(Tensor(a!) out, Tensor lora, "
      "Tensor xn, Tensor codes, Tensor channel_scales) -> ()");
  ops.impl("fp8_qpn8_hc_up_gate_mix_sm70_out", torch::kCUDA,
           &fp8_qpn8_hc_up_gate_mix_sm70_out);
  ops.def(
      "fp8_qpn8_hc_dispatch_sm70_out(Tensor(a!) block_out, Tensor(b!) "
      "injection_out, Tensor(c!) down_staging, Tensor(d!) lora_staging, "
      "Tensor(e!) gate_staging, Tensor(f!) partials, int dense_weight_ptr, "
      "Tensor xn, Tensor down_codes, Tensor down_scales, Tensor up_codes, "
      "Tensor up_scales) -> ()");
  ops.impl("fp8_qpn8_hc_dispatch_sm70_out", torch::kCUDA,
           &fp8_qpn8_hc_dispatch_sm70_out);
}
#endif
