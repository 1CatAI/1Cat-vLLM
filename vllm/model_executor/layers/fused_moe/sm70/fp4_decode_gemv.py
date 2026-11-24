# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Shape-generic SM70 NVFP4 MoE decode kernels (CUDA JIT).

The fork's compiled QPN / direct-reduce NVFP4 ops are hard-wired to Qwen3.8
shapes (E512, K2560, I160, top-10). These kernels are the same quadpair
m8n8k4 GEMV design, generalized by shape, working directly on the
TurboMind-prepared expert banks of `ModelOptNvFp4SM70MoEMethod`
(`w13_tm_weight [E, K, N/8] int32`, `w13_tm_scales [E, K/16, N] fp16`).

- `w13_act`: W13 for the top-k routed experts of every token, with the
  softcapped SwiGLU fused in the epilogue (gate/up rounded to fp16 first,
  like the unfused W13 -> activation path).
- `w2_reduce`: W2 for the routed experts of a token plus the top-k weighted
  reduce (each route rounded to fp16, fp32 weighted sum in slot order).

No permutation / sort / expand: routes are addressed by `topk_ids` directly,
so decode is two (M < 4) or three (grouped) launches per visit for any M <= `MAX_TOKENS`.
CUDA-graph safe: no host sync, no data-dependent shapes.
"""

import os
from functools import lru_cache

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_TOKENS = min(
    16, int(os.environ.get("VLLM_INKLING2_SM70_MOE_FAST_MAX_TOKENS", "16"))
)
_PERSISTENT_TOKENS = 18  # NvFp4MoEWorkspace persistent buffers

_CPP = r"""
#include <torch/extension.h>
void w13_act(torch::Tensor out, torch::Tensor x, torch::Tensor weights,
             torch::Tensor scales, torch::Tensor ids, int64_t top_k,
             int64_t split_k, double gate_cap, double up_cap);
void w2_reduce(torch::Tensor out, torch::Tensor act, torch::Tensor weights,
               torch::Tensor scales, torch::Tensor ids,
               torch::Tensor topk_weights, int64_t split_k);
void w13_act_grouped(torch::Tensor out, torch::Tensor x, torch::Tensor weights,
                     torch::Tensor scales, torch::Tensor ids, int64_t top_k,
                     int64_t split_k, double gate_cap, double up_cap);
void w2_grouped(torch::Tensor out, torch::Tensor routed, torch::Tensor act,
                torch::Tensor weights, torch::Tensor scales, torch::Tensor ids,
                torch::Tensor topk_weights, int64_t split_k);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ void dequant_e2m1x8(unsigned packed, half2 scale,
                                               half2 out[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x0e000e00u;
  unsigned values[4];
  values[0] = ((packed << 12) & kSign) | ((packed << 9) & kExponentMantissa);
  values[1] = ((packed << 8) & kSign) | ((packed << 5) & kExponentMantissa);
  values[2] = ((packed << 4) & kSign) | ((packed << 1) & kExponentMantissa);
  values[3] = (packed & kSign) | ((packed >> 3) & kExponentMantissa);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    out[index] =
        __hmul2(*reinterpret_cast<const half2*>(&values[index]), scale);
  }
}

#define MMA_8N8K4(C, A0, A1, B0, B1)                                \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

__host__ __device__ constexpr int chunk_of(int ng) {
  // Largest divisor of ng that is <= 9: groups whose loads are hoisted together.
  for (int c = 9; c >= 1; --c)
    if (ng % c == 0) return c;
  return 1;
}

// One warp: 32 output columns (tile) of one expert, K groups [g0, g0+ng).
// Returns the per-lane accumulators of the m8n8k4 quadpair layout; only
// lanes with (lane & 17) == 0 hold the row-0 (token) result.
__device__ __forceinline__ void warp_gemv_tile(
    float accum[8], const half* __restrict__ input_row,
    const uint32_t* __restrict__ expert_weights,
    const half* __restrict__ expert_scales, int tile, int tiles_n32,
    int groups_k8, int g0, int ng, int lane) {
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col =
      ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
#pragma unroll
  for (int i = 0; i < 8; ++i) accum[i] = 0.0f;
#pragma unroll 4
  for (int group = g0; group < g0 + ng; ++group) {
    const size_t tile_group_base =
        (static_cast<size_t>(tile) * groups_k8 + group * 2) * 32 + packed_col;
    const unsigned packed0 = __ldcs(expert_weights + tile_group_base);
    const unsigned packed1 = __ldcs(expert_weights + tile_group_base + 32);
    const size_t scale_index =
        (static_cast<size_t>(group) * tiles_n32 + tile) * 32 + packed_col;
    const half scalar = __ldg(expert_scales + scale_index);
    const half2 scale =
        __hmul2(__halves2half2(scalar, scalar), __float2half2_rn(16384.0f));
    half2 decoded[8];
    dequant_e2m1x8(packed0, scale, decoded);
    dequant_e2m1x8(packed1, scale, decoded + 4);
    const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
    uint4 input01 = make_uint4(0, 0, 0, 0);
    uint4 input23 = make_uint4(0, 0, 0, 0);
    if (a_row == 0) {
      input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
      input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
    }
    const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
    MMA_8N8K4(accum, a0[0], a0[1], b[0], b[1]);
    MMA_8N8K4(accum, a0[2], a0[3], b[2], b[3]);
    MMA_8N8K4(accum, a1[0], a1[1], b[4], b[5]);
    MMA_8N8K4(accum, a1[2], a1[3], b[6], b[7]);
  }
}

__device__ __forceinline__ void store_partials(float* dst, const float accum[8],
                                               int lane) {
  if ((lane & 17) == 0) {
    const int quadpair = (lane >> 2) & 3;
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
      for (int offset = 0; offset < 2; ++offset) {
        const int index = pair * 4 + offset;
        const int local_col = offset | (((lane >> 1) & 1) << 1) | (pair << 2);
        dst[quadpair * 8 + local_col] = accum[index];
      }
    }
  }
}

// grid (I/32, routes); block 64*kSplitK: warps [0,S) gate tile, [S,2S) up tile.
template <int kSplitK>
__global__ void __launch_bounds__(64 * kSplitK)
    w13_act_kernel(const half* __restrict__ x, const uint32_t* __restrict__ w,
                   const half* __restrict__ sc,
                   const int32_t* __restrict__ ids, half* __restrict__ out,
                   int inter, int k, int num_experts, int top_k,
                   float gate_cap, float up_cap) {
  __shared__ float partials[2][kSplitK][32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int projection = warp / kSplitK;
  const int split = warp % kSplitK;
  const int route = blockIdx.y;
  const int tile = blockIdx.x + projection * (inter / 32);
  const int n = inter * 2;
  const int expert = __ldg(ids + route);
  float accum[8];
  const bool valid = expert >= 0 && expert < num_experts;
  if (valid) {
    const int groups_k16 = k >> 4;
    const int ng = groups_k16 / kSplitK;
    const size_t words_per_expert = static_cast<size_t>(k) * n / 8;
    const size_t scales_per_expert = static_cast<size_t>(groups_k16) * n;
    warp_gemv_tile(accum, x + static_cast<size_t>(route / top_k) * k,
                   w + static_cast<size_t>(expert) * words_per_expert,
                   sc + static_cast<size_t>(expert) * scales_per_expert, tile,
                   n >> 5, k >> 3, split * ng, ng, lane);
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) accum[i] = 0.0f;
  }
  store_partials(&partials[projection][split][0], accum, lane);
  __syncthreads();
  if (warp == 0) {
    float g = 0.0f, u = 0.0f;
#pragma unroll
    for (int s = 0; s < kSplitK; ++s) {
      g += partials[0][s][lane];
      u += partials[1][s][lane];
    }
    // fp16 W13 output boundary, then the softcapped SwiGLU in fp32.
    const float gh = __half2float(__float2half(g));
    const float uh = __half2float(__float2half(u));
    float gf = gh;
    if (gate_cap > 0.0f) gf = gate_cap * tanhf(gh / gate_cap);
    float uf = uh;
    if (up_cap > 0.0f) uf = up_cap * tanhf(uh / up_cap);
    const float sig = 1.0f / (1.0f + expf(-gh));
    out[static_cast<size_t>(route) * inter + blockIdx.x * 32 + lane] =
        __float2half(gf * sig * uf);
  }
}

// grid (N/32, tokens); block 32*top_k*kSplitK. Warp = (slot, split).
template <int kSplitK>
__global__ void __launch_bounds__(256 * kSplitK)
    w2_reduce_kernel(const half* __restrict__ act,
                     const uint32_t* __restrict__ w,
                     const half* __restrict__ sc,
                     const int32_t* __restrict__ ids,
                     const float* __restrict__ topk_w, half* __restrict__ out,
                     int n, int k, int num_experts, int top_k) {
  extern __shared__ float smem[];  // [top_k][kSplitK][32]
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slot = warp / kSplitK;
  const int split = warp % kSplitK;
  const int token = blockIdx.y;
  const int tile = blockIdx.x;
  const int route = token * top_k + slot;
  const int expert = __ldg(ids + route);
  float accum[8];
  if (expert >= 0 && expert < num_experts) {
    const int groups_k16 = k >> 4;
    const int ng = groups_k16 / kSplitK;
    const size_t words_per_expert = static_cast<size_t>(k) * n / 8;
    const size_t scales_per_expert = static_cast<size_t>(groups_k16) * n;
    warp_gemv_tile(accum, act + static_cast<size_t>(route) * k,
                   w + static_cast<size_t>(expert) * words_per_expert,
                   sc + static_cast<size_t>(expert) * scales_per_expert, tile,
                   n >> 5, k >> 3, split * ng, ng, lane);
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) accum[i] = 0.0f;
  }
  store_partials(smem + (slot * kSplitK + split) * 32, accum, lane);
  __syncthreads();
  if (warp == 0) {
    float weighted = 0.0f;
    for (int s = 0; s < top_k; ++s) {
      float v = 0.0f;
#pragma unroll
      for (int p = 0; p < kSplitK; ++p) v += smem[(s * kSplitK + p) * 32 + lane];
      // Per-route fp16 rounding before the weighted fp32 sum.
      weighted = fmaf(__ldg(topk_w + token * top_k + s),
                      __half2float(__float2half(v)), weighted);
    }
    out[static_cast<size_t>(token) * n + tile * 32 + lane] =
        __float2half(weighted);
  }
}

// ---- grouped (same-expert) kernels: each distinct expert is read once ----
//
// CTA (tile, route) is the "leader" iff `route` is the first occurrence of
// its expert in ids[0..R). A leader gathers every route sharing the expert
// (<= 16 rows, top-k ids are distinct per token and M <= 16) and feeds them
// as MMA rows (A rows = tokens), so the expert's weights are decoded and
// multiplied once for all of them. Non-leaders exit. Accumulation per row is
// identical to the per-route kernels, so results are bitwise equal.
constexpr int kMaxRoutes = 128;
constexpr int kMaxRows = 16;

struct GroupInfo {
  int nrows;
  int leader;
  int rows[kMaxRows];
};

// Called by all threads. Fills `info` (smem) for blockIdx.y's expert.
__device__ __forceinline__ void build_group(GroupInfo* info, int* sids,
                                            const int32_t* ids, int routes,
                                            int route, int num_experts) {
  for (int i = threadIdx.x; i < routes; i += blockDim.x) sids[i] = ids[i];
  __syncthreads();
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    const int expert = sids[route];
    int count = 0;
    for (int j0 = 0; j0 < routes; j0 += 32) {
      const int j = j0 + lane;
      const bool match = j < routes && sids[j] == expert;
      const unsigned mask = __ballot_sync(0xffffffffu, match);
      if (match) {
        const int pos = count + __popc(mask & ((1u << lane) - 1u));
        if (pos < kMaxRows) info->rows[pos] = j;
      }
      count += __popc(mask);
    }
    __syncwarp();
    if (lane == 0) {
      info->nrows = count < kMaxRows ? count : kMaxRows;
      info->leader = (expert >= 0 && expert < num_experts &&
                      info->rows[0] == route && count <= kMaxRows);
    }
  }
  __syncthreads();
}

// Accumulators for NCH row-chunks of 8 rows (A rows) against one 32-col tile.
template <int NG, int NCH>
__device__ __forceinline__ void warp_gemv_rows(
    float accum[NCH][8], const half* __restrict__ x, const int* rows,
    int nrows, int top_k, int k, const uint32_t* __restrict__ expert_weights,
    const half* __restrict__ expert_scales, int tile, int tiles_n32,
    int groups_k8, int g0, int lane) {
  constexpr int CH = chunk_of(NG);
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col =
      ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const half* in_row[NCH];
#pragma unroll
  for (int c = 0; c < NCH; ++c) {
    const int rr = c * 8 + a_row;
    in_row[c] = rr < nrows ? x + static_cast<size_t>(rows[rr] / top_k) * k
                           : nullptr;
#pragma unroll
    for (int i = 0; i < 8; ++i) accum[c][i] = 0.0f;
  }
#pragma unroll
  for (int c0 = 0; c0 < NG; c0 += CH) {
    unsigned p0[CH], p1[CH];
    half sc[CH];
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      const int group = g0 + c0 + j;
      const size_t base =
          (static_cast<size_t>(tile) * groups_k8 + group * 2) * 32 + packed_col;
      p0[j] = __ldcs(expert_weights + base);
      p1[j] = __ldcs(expert_weights + base + 32);
      sc[j] = __ldg(expert_scales +
                    (static_cast<size_t>(group) * tiles_n32 + tile) * 32 +
                    packed_col);
    }
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      const int group = g0 + c0 + j;
      const half2 scale = __hmul2(__halves2half2(sc[j], sc[j]),
                                  __float2half2_rn(16384.0f));
      half2 decoded[8];
      dequant_e2m1x8(p0[j], scale, decoded);
      dequant_e2m1x8(p1[j], scale, decoded + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        uint4 in01 = make_uint4(0, 0, 0, 0);
        uint4 in23 = make_uint4(0, 0, 0, 0);
        if (in_row[c] != nullptr) {
          in01 = *reinterpret_cast<const uint4*>(in_row[c] + group * 16);
          in23 = *reinterpret_cast<const uint4*>(in_row[c] + group * 16 + 8);
        }
        const unsigned* a0 = reinterpret_cast<const unsigned*>(&in01);
        const unsigned* a1 = reinterpret_cast<const unsigned*>(&in23);
        MMA_8N8K4(accum[c], a0[0], a0[1], b[0], b[1]);
        MMA_8N8K4(accum[c], a0[2], a0[3], b[2], b[3]);
        MMA_8N8K4(accum[c], a1[0], a1[1], b[4], b[5]);
        MMA_8N8K4(accum[c], a1[2], a1[3], b[6], b[7]);
      }
    }
  }
}

// part[row][32 cols] <- accumulators of every row (all lanes).
template <int NCH>
__device__ __forceinline__ void store_rows(float* part, const float accum[NCH][8],
                                           int lane) {
  const int t = (lane & 3) | ((lane >> 4) << 2);
  const int quadpair = (lane >> 2) & 3;
#pragma unroll
  for (int c = 0; c < NCH; ++c) {
#pragma unroll
    for (int idx = 0; idx < 8; ++idx) {
      const int row = c * 8 + ((t & 1) | (((idx >> 1) & 1) << 1) | ((t >> 2) << 2));
      const int col = quadpair * 8 +
                      ((idx & 1) | (((t >> 1) & 1) << 1) | (((idx >> 2) & 1) << 2));
      part[row * 32 + col] = accum[c][idx];
    }
  }
}

// grid (I/32, routes); block 64*S; dynamic smem: ids + info + [2S][16][32] f32.
template <int NG>
__global__ void __launch_bounds__(768) w13_act_grouped_kernel(
    const half* __restrict__ x, const uint32_t* __restrict__ w,
    const half* __restrict__ sc, const int32_t* __restrict__ ids,
    half* __restrict__ out, int routes, int inter, int k, int num_experts,
    int top_k, float gate_cap, float up_cap) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  int* sids = reinterpret_cast<int*>(smem_raw);
  GroupInfo* info = reinterpret_cast<GroupInfo*>(smem_raw + kMaxRoutes * 4);
  float* part = reinterpret_cast<float*>(smem_raw + kMaxRoutes * 4 + 128);
  const int S = blockDim.x >> 6;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int route = blockIdx.y;
  build_group(info, sids, ids, routes, route, num_experts);
  if (!info->leader) return;
  const int nrows = info->nrows;
  const int projection = warp / S;
  const int split = warp - projection * S;
  const int tile = blockIdx.x + projection * (inter / 32);
  const int n = inter * 2;
  const int expert = sids[route];
  const int groups_k16 = k >> 4;
  const uint32_t* ew = w + static_cast<size_t>(expert) * (static_cast<size_t>(k) * n / 8);
  const half* es = sc + static_cast<size_t>(expert) * (static_cast<size_t>(groups_k16) * n);
  float* my_part = part + (projection * S + split) * (kMaxRows * 32);
  if (nrows <= 8) {
    float accum[1][8];
    warp_gemv_rows<NG, 1>(accum, x, info->rows, nrows, top_k, k, ew, es, tile,
                          n >> 5, k >> 3, split * NG, lane);
    store_rows<1>(my_part, accum, lane);
  } else {
    float accum[2][8];
    warp_gemv_rows<NG, 2>(accum, x, info->rows, nrows, top_k, k, ew, es, tile,
                          n >> 5, k >> 3, split * NG, lane);
    store_rows<2>(my_part, accum, lane);
  }
  __syncthreads();
  for (int e = threadIdx.x; e < nrows * 32; e += blockDim.x) {
    const int row = e >> 5, col = e & 31;
    float g = 0.0f, u = 0.0f;
    for (int s = 0; s < S; ++s) {
      g += part[(0 * S + s) * (kMaxRows * 32) + row * 32 + col];
      u += part[(1 * S + s) * (kMaxRows * 32) + row * 32 + col];
    }
    const float gh = __half2float(__float2half(g));
    const float uh = __half2float(__float2half(u));
    float gf = gh;
    if (gate_cap > 0.0f) gf = gate_cap * tanhf(gh / gate_cap);
    float uf = uh;
    if (up_cap > 0.0f) uf = up_cap * tanhf(uh / up_cap);
    const float sig = 1.0f / (1.0f + expf(-gh));
    out[static_cast<size_t>(info->rows[row]) * inter + blockIdx.x * 32 + col] =
        __float2half(gf * sig * uf);
  }
}

// grid (N/32, routes); block 32*S (one warp per K split). Writes the fp16
// per-route outputs; token_reduce_kernel does the weighted sum.
template <int NG>
__global__ void __launch_bounds__(512) w2_grouped_kernel(
    const half* __restrict__ act, const uint32_t* __restrict__ w,
    const half* __restrict__ sc, const int32_t* __restrict__ ids,
    half* __restrict__ routed, int routes, int n, int k, int num_experts,
    int top_k) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  int* sids = reinterpret_cast<int*>(smem_raw);
  GroupInfo* info = reinterpret_cast<GroupInfo*>(smem_raw + kMaxRoutes * 4);
  float* part = reinterpret_cast<float*>(smem_raw + kMaxRoutes * 4 + 128);
  const int lane = threadIdx.x & 31;
  const int split = threadIdx.x >> 5;
  const int S = blockDim.x >> 5;
  const int route = blockIdx.y;
  build_group(info, sids, ids, routes, route, num_experts);
  if (!info->leader) return;
  const int nrows = info->nrows;
  const int tile = blockIdx.x;
  const int expert = sids[route];
  const int groups_k16 = k >> 4;
  const uint32_t* ew = w + static_cast<size_t>(expert) * (static_cast<size_t>(k) * n / 8);
  const half* es = sc + static_cast<size_t>(expert) * (static_cast<size_t>(groups_k16) * n);
  // Rows of w2's A operand are the routed activation rows themselves.
  // warp_gemv_rows indexes `x + (rows[rr] / top_k) * k`; pass top_k = 1.
  float* my_part = part + split * (kMaxRows * 32);
  if (nrows <= 8) {
    float accum[1][8];
    warp_gemv_rows<NG, 1>(accum, act, info->rows, nrows, 1, k, ew, es, tile,
                          n >> 5, k >> 3, split * NG, lane);
    store_rows<1>(my_part, accum, lane);
  } else {
    float accum[2][8];
    warp_gemv_rows<NG, 2>(accum, act, info->rows, nrows, 1, k, ew, es, tile,
                          n >> 5, k >> 3, split * NG, lane);
    store_rows<2>(my_part, accum, lane);
  }
  __syncthreads();
  for (int e = threadIdx.x; e < nrows * 32; e += blockDim.x) {
    const int row = e >> 5, col = e & 31;
    float v = 0.0f;
    for (int s = 0; s < S; ++s) v += part[s * (kMaxRows * 32) + row * 32 + col];
    routed[static_cast<size_t>(info->rows[row]) * n + tile * 32 + col] =
        __float2half(v);
  }
}

// out[t, n] = sum_s w[t, s] * routed[t * top_k + s, n], slot order, fp32 fma.
__global__ void token_reduce_kernel(const half* __restrict__ routed,
                                    const float* __restrict__ topk_w,
                                    half* __restrict__ out, int tokens, int n,
                                    int top_k) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= tokens * n) return;
  const int t = i / n, c = i - t * n;
  float weighted = 0.0f;
  for (int s = 0; s < top_k; ++s)
    weighted = fmaf(__ldg(topk_w + t * top_k + s),
                    __half2float(routed[(static_cast<size_t>(t) * top_k + s) * n + c]),
                    weighted);
  out[i] = __float2half(weighted);
}

}  // namespace

#define NG_CASES(X)                                                        \
  X(1) X(2) X(3) X(4) X(5) X(6) X(8) X(9) X(10) X(12) X(16) X(18) X(24) X(36) X(72)

void w13_act(torch::Tensor out, torch::Tensor x, torch::Tensor weights,
             torch::Tensor scales, torch::Tensor ids, int64_t top_k,
             int64_t split_k, double gate_cap, double up_cap) {
  TORCH_CHECK(out.is_cuda() && x.is_cuda() && weights.is_cuda() &&
              scales.is_cuda() && ids.is_cuda(), "w13_act: CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 &&
              out.scalar_type() == torch::kFloat16 &&
              weights.scalar_type() == torch::kInt32 &&
              scales.scalar_type() == torch::kFloat16 &&
              ids.scalar_type() == torch::kInt32, "w13_act: dtype");
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous() &&
              weights.is_contiguous() && scales.is_contiguous() &&
              ids.is_contiguous(), "w13_act: contiguous");
  const int64_t routes = ids.numel();
  const int64_t k = x.size(1);
  const int64_t inter = out.size(1);
  TORCH_CHECK(routes == x.size(0) * top_k && out.size(0) == routes &&
              inter % 32 == 0 && k % 16 == 0 && weights.dim() == 3 &&
              weights.size(1) == k && weights.size(2) * 8 == 2 * inter &&
              scales.size(1) == k / 16 && scales.size(2) == 2 * inter &&
              scales.size(0) == weights.size(0),
              "w13_act: shape contract");
  TORCH_CHECK(split_k > 0 && (k / 16) % split_k == 0, "w13_act: split_k");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const dim3 grid(inter / 32, routes);
  auto stream = at::cuda::getCurrentCUDAStream();
#define W13_CASE(S)                                                         \
  case S:                                                                   \
    w13_act_kernel<S><<<grid, 64 * S, 0, stream>>>(                         \
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),              \
        reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),     \
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),         \
        ids.data_ptr<int32_t>(),                                            \
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), (int)inter,      \
        (int)k, (int)weights.size(0), (int)top_k, (float)gate_cap,          \
        (float)up_cap);                                                     \
    break
  switch (split_k) {
    W13_CASE(1); W13_CASE(2); W13_CASE(3); W13_CASE(4); W13_CASE(6);
    W13_CASE(8); W13_CASE(9); W13_CASE(12); W13_CASE(16);
    default: TORCH_CHECK(false, "w13_act: unsupported split_k ", split_k);
  }
#undef W13_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void w2_reduce(torch::Tensor out, torch::Tensor act, torch::Tensor weights,
               torch::Tensor scales, torch::Tensor ids,
               torch::Tensor topk_weights, int64_t split_k) {
  TORCH_CHECK(out.is_cuda() && act.is_cuda() && weights.is_cuda() &&
              scales.is_cuda() && ids.is_cuda() && topk_weights.is_cuda(),
              "w2_reduce: CUDA tensors");
  TORCH_CHECK(act.scalar_type() == torch::kFloat16 &&
              out.scalar_type() == torch::kFloat16 &&
              weights.scalar_type() == torch::kInt32 &&
              scales.scalar_type() == torch::kFloat16 &&
              ids.scalar_type() == torch::kInt32 &&
              topk_weights.scalar_type() == torch::kFloat32, "w2_reduce: dtype");
  TORCH_CHECK(act.is_contiguous() && out.is_contiguous() &&
              weights.is_contiguous() && scales.is_contiguous() &&
              ids.is_contiguous() && topk_weights.is_contiguous(),
              "w2_reduce: contiguous");
  const int64_t tokens = out.size(0);
  const int64_t n = out.size(1);
  const int64_t k = act.size(1);
  const int64_t top_k = ids.numel() / tokens;
  TORCH_CHECK(top_k * tokens == ids.numel() && act.size(0) == ids.numel() &&
              topk_weights.numel() == ids.numel() && n % 32 == 0 &&
              k % 16 == 0 && weights.dim() == 3 && weights.size(1) == k &&
              weights.size(2) * 8 == n && scales.size(1) == k / 16 &&
              scales.size(2) == n && scales.size(0) == weights.size(0),
              "w2_reduce: shape contract");
  TORCH_CHECK(split_k > 0 && (k / 16) % split_k == 0 &&
              top_k * split_k * 32 <= 1024, "w2_reduce: split_k");
  const at::cuda::OptionalCUDAGuard guard(device_of(act));
  const dim3 grid(n / 32, tokens);
  const int threads = 32 * top_k * split_k;
  const size_t smem = top_k * split_k * 32 * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
#define W2_CASE(S)                                                          \
  case S:                                                                   \
    w2_reduce_kernel<S><<<grid, threads, smem, stream>>>(                   \
        reinterpret_cast<const half*>(act.data_ptr<at::Half>()),            \
        reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),     \
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),         \
        ids.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),            \
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), (int)n, (int)k,  \
        (int)weights.size(0), (int)top_k);                                  \
    break
  switch (split_k) {
    W2_CASE(1); W2_CASE(2); W2_CASE(3); W2_CASE(4);
    default: TORCH_CHECK(false, "w2_reduce: unsupported split_k ", split_k);
  }
#undef W2_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void w13_act_grouped(torch::Tensor out, torch::Tensor x, torch::Tensor weights,
                     torch::Tensor scales, torch::Tensor ids, int64_t top_k,
                     int64_t split_k, double gate_cap, double up_cap) {
  TORCH_CHECK(out.is_cuda() && x.is_cuda() && weights.is_cuda() &&
              scales.is_cuda() && ids.is_cuda(), "w13_act_grouped: CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 && out.scalar_type() == torch::kFloat16 &&
              weights.scalar_type() == torch::kInt32 &&
              scales.scalar_type() == torch::kFloat16 &&
              ids.scalar_type() == torch::kInt32, "w13_act_grouped: dtype");
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && weights.is_contiguous() &&
              scales.is_contiguous() && ids.is_contiguous(), "w13_act_grouped: contiguous");
  const int64_t routes = ids.numel(), k = x.size(1), inter = out.size(1);
  TORCH_CHECK(routes == x.size(0) * top_k && routes <= kMaxRoutes && x.size(0) <= kMaxRows &&
              out.size(0) == routes && inter % 32 == 0 && k % 16 == 0 &&
              weights.dim() == 3 && weights.size(1) == k && weights.size(2) * 8 == 2 * inter &&
              scales.size(1) == k / 16 && scales.size(2) == 2 * inter,
              "w13_act_grouped: shape contract");
  TORCH_CHECK(split_k > 0 && split_k <= 12 && (k / 16) % split_k == 0, "w13_act_grouped: split_k");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const int ng_total = static_cast<int>((k / 16) / split_k);
  const dim3 grid(inter / 32, routes);
  const size_t smem = kMaxRoutes * 4 + 128 + 2 * split_k * kMaxRows * 32 * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
#define W13G_CASE(S)                                                        \
  case S:                                                                   \
    cudaFuncSetAttribute(w13_act_grouped_kernel<S>,                         \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
    w13_act_grouped_kernel<S><<<grid, 64 * split_k, smem, stream>>>(        \
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),              \
        reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),     \
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),         \
        ids.data_ptr<int32_t>(), reinterpret_cast<half*>(out.data_ptr<at::Half>()), \
        (int)routes, (int)inter, (int)k, (int)weights.size(0), (int)top_k,  \
        (float)gate_cap, (float)up_cap);                                    \
    break;
  switch (ng_total) {
    NG_CASES(W13G_CASE)
    default: TORCH_CHECK(false, "w13_act_grouped: unsupported groups per warp ", ng_total);
  }
#undef W13G_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void w2_grouped(torch::Tensor out, torch::Tensor routed, torch::Tensor act,
                torch::Tensor weights, torch::Tensor scales, torch::Tensor ids,
                torch::Tensor topk_weights, int64_t split_k) {
  TORCH_CHECK(out.is_cuda() && routed.is_cuda() && act.is_cuda() && weights.is_cuda() &&
              scales.is_cuda() && ids.is_cuda() && topk_weights.is_cuda(), "w2_grouped: CUDA");
  TORCH_CHECK(act.scalar_type() == torch::kFloat16 && out.scalar_type() == torch::kFloat16 &&
              routed.scalar_type() == torch::kFloat16 &&
              weights.scalar_type() == torch::kInt32 && scales.scalar_type() == torch::kFloat16 &&
              ids.scalar_type() == torch::kInt32 && topk_weights.scalar_type() == torch::kFloat32,
              "w2_grouped: dtype");
  TORCH_CHECK(act.is_contiguous() && out.is_contiguous() && routed.is_contiguous() &&
              weights.is_contiguous() && scales.is_contiguous() && ids.is_contiguous() &&
              topk_weights.is_contiguous(), "w2_grouped: contiguous");
  const int64_t tokens = out.size(0), n = out.size(1), k = act.size(1);
  const int64_t routes = ids.numel();
  const int64_t top_k = routes / tokens;
  TORCH_CHECK(top_k * tokens == routes && routes <= kMaxRoutes && tokens <= kMaxRows &&
              act.size(0) == routes && routed.size(0) >= routes && routed.size(1) == n &&
              topk_weights.numel() == routes && n % 32 == 0 && k % 16 == 0 &&
              weights.dim() == 3 && weights.size(1) == k && weights.size(2) * 8 == n &&
              scales.size(1) == k / 16 && scales.size(2) == n, "w2_grouped: shape contract");
  TORCH_CHECK(split_k > 0 && split_k <= 16 && (k / 16) % split_k == 0, "w2_grouped: split_k");
  const at::cuda::OptionalCUDAGuard guard(device_of(act));
  const int ng_total = static_cast<int>((k / 16) / split_k);
  const dim3 grid(n / 32, routes);
  const size_t smem = kMaxRoutes * 4 + 128 + split_k * kMaxRows * 32 * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
#define W2G_CASE(S)                                                         \
  case S:                                                                   \
    cudaFuncSetAttribute(w2_grouped_kernel<S>,                              \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
    w2_grouped_kernel<S><<<grid, 32 * split_k, smem, stream>>>(             \
        reinterpret_cast<const half*>(act.data_ptr<at::Half>()),            \
        reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),     \
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),         \
        ids.data_ptr<int32_t>(), reinterpret_cast<half*>(routed.data_ptr<at::Half>()), \
        (int)routes, (int)n, (int)k, (int)weights.size(0), (int)top_k);     \
    break;
  switch (ng_total) {
    NG_CASES(W2G_CASE)
    default: TORCH_CHECK(false, "w2_grouped: unsupported groups per warp ", ng_total);
  }
#undef W2G_CASE
  token_reduce_kernel<<<(tokens * n + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const half*>(routed.data_ptr<at::Half>()),
      topk_weights.data_ptr<float>(), reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      (int)tokens, (int)n, (int)top_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load_inline

    build_dir = os.path.join(
        os.environ.get(
            "VLLM_INKLING2_SM70_EXT_DIR",
            os.path.expanduser("~/.cache/vllm/inkling2_sm70_ext"),
        ),
        "moe_fp4_decode",
    )
    os.makedirs(build_dir, exist_ok=True)
    return load_inline(
        name="sm70_nvfp4_decode_gemv",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["w13_act", "w2_reduce", "w13_act_grouped", "w2_grouped"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "-lineinfo"],
        build_directory=build_dir,
        verbose=bool(os.environ.get("VLLM_INKLING2_SM70_EXT_VERBOSE")),
    )


def w13_act(out, x, weights, scales, ids, top_k, split_k, gate_cap, up_cap):
    _ext().w13_act(out, x, weights, scales, ids, top_k, split_k, gate_cap, up_cap)


def w2_reduce(out, act, weights, scales, ids, topk_weights, split_k):
    _ext().w2_reduce(out, act, weights, scales, ids, topk_weights, split_k)


def w13_act_grouped(out, x, weights, scales, ids, top_k, split_k, gate_cap, up_cap):
    _ext().w13_act_grouped(
        out, x, weights, scales, ids, top_k, split_k, gate_cap, up_cap
    )


def w2_grouped(out, routed, act, weights, scales, ids, topk_weights, split_k):
    _ext().w2_grouped(out, routed, act, weights, scales, ids, topk_weights, split_k)


_W13_SPLITS = (16, 12, 9, 8, 6, 4, 3, 2, 1)
_W2_SPLITS = (4, 3, 2, 1)
_W13G_SPLITS = (12, 9, 8, 6, 4, 3, 2, 1)  # grouped W13: 64 * S threads <= 768
_W2G_SPLITS = (16, 12, 9, 8, 6, 4, 3, 2, 1)
# Instances compiled for the grouped kernels (groups per warp).
_NGS = frozenset((1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 24, 36, 72))
_GROUPED_MIN_TOKENS = int(
    os.environ.get("VLLM_INKLING2_SM70_MOE_FAST_GROUPED_MIN", "4")
)


def _largest_divisor(groups, cap, allowed, *, need_instance=False):
    for s in allowed:
        if s <= cap and groups % s == 0 and (not need_instance or groups // s in _NGS):
            return s
    return None


def _softcap_params(activation):
    """(gate_cap, up_cap) of a `partial(softcapped_swiglu_, ...)` hook, else None."""
    func = getattr(activation, "func", None)
    if getattr(func, "__name__", None) != "softcapped_swiglu_":
        return None
    kw = getattr(activation, "keywords", None) or {}
    if set(kw) != {"gate_cap", "up_cap"} or getattr(activation, "args", ()):
        return None
    return float(kw["gate_cap"]), float(kw["up_cap"])


class Fp4DecodeGemv:
    """Per-layer M <= MAX_TOKENS NVFP4 MoE route: W13 + softcapped SwiGLU, then
    W2 + weighted reduce. No sort/permute, graph safe. M < GROUPED_MIN launches two
    per-route kernels; larger M reads each distinct expert once (grouped)."""

    def __init__(self, gate_cap, up_cap, top_k, k13, k2):
        self.gate_cap, self.up_cap, self.top_k = gate_cap, up_cap, top_k
        self.g13, self.g2 = k13 // 16, k2 // 16

    @classmethod
    def create(cls, layer, *, raw_scale):
        if os.environ.get("VLLM_INKLING2_SM70_MOE_FAST", "1") == "0":
            return None
        caps = _softcap_params(getattr(layer, "sm70_moe_activation", None))
        if (
            caps is None
            or raw_scale
            or getattr(layer, "swiglu_limit", None) is not None
            or not torch.cuda.is_available()
            or torch.cuda.get_device_capability(layer.w13_tm_weight.device) != (7, 0)
        ):
            return None
        hidden = int(layer.sm70_nvfp4_hidden_size)
        inter = int(layer.sm70_nvfp4_intermediate_size)
        top_k = int(layer.sm70_nvfp4_top_k)
        if hidden % 32 or inter % 32 or top_k > 16 or top_k * MAX_TOKENS > 128:
            return None
        if layer.w13_tm_scales.dtype != torch.float16:
            return None
        self = cls(caps[0], caps[1], top_k, hidden, inter)
        for m in (1, 2, MAX_TOKENS):
            s13, s2 = self.splits(m)
            if s13 is None or s2 is None:
                return None
        _ext()  # compile / load now, never inside CUDA-graph capture
        logger.info_once(
            "SM70 NVFP4 decode GEMV MoE route enabled (hidden=%d, local_intermediate=%d, "
            "top_k=%d, M<=%d, grouped from M>=%d, softcap gate=%g up=%g).",
            hidden,
            inter,
            top_k,
            MAX_TOKENS,
            _GROUPED_MIN_TOKENS,
            caps[0],
            caps[1],
        )
        return self

    def grouped(self, tokens):
        return tokens >= _GROUPED_MIN_TOKENS

    def splits(self, tokens):
        """K-splits (warps along K) per stage; tuned on TP4 Inkling shapes."""
        if self.grouped(tokens):
            s13 = _largest_divisor(
                self.g13, 6 if tokens < 8 else 4, _W13G_SPLITS, need_instance=True
            )
            s2 = _largest_divisor(self.g2, 2, _W2G_SPLITS, need_instance=True)
        else:
            s13 = _largest_divisor(self.g13, 16 if tokens == 1 else 9, _W13_SPLITS)
            s2 = _largest_divisor(self.g2, 4 if tokens == 1 else 2, _W2_SPLITS)
            if s2 is not None and self.top_k * s2 * 32 > 1024:
                s2 = None
        return s13, s2

    def __call__(self, layer, x, topk_weights, topk_ids):
        tokens = x.shape[0]
        if (
            tokens > MAX_TOKENS
            or not x.is_contiguous()
            or not topk_ids.is_contiguous()
            or topk_ids.dtype != torch.int32
            or not topk_weights.is_contiguous()
        ):
            return None
        top_k, hidden = self.top_k, x.shape[1]
        inter = int(layer.sm70_nvfp4_intermediate_size)
        slots = tokens * top_k
        persistent = tokens <= _PERSISTENT_TOKENS
        if persistent:
            act = layer._nvfp4_sm70_intermediate[:slots]
            out = layer._nvfp4_sm70_output[:tokens]
        else:
            act = torch.empty(slots, inter, dtype=torch.float16, device=x.device)
            out = torch.empty(tokens, hidden, dtype=torch.float16, device=x.device)
        s13, s2 = self.splits(tokens)
        ids = topk_ids.view(-1)
        if self.grouped(tokens):
            routed = (
                layer._nvfp4_sm70_sorted_output[:slots]
                if persistent
                else torch.empty(slots, hidden, dtype=torch.float16, device=x.device)
            )
            w13_act_grouped(
                act,
                x,
                layer.w13_tm_weight,
                layer.w13_tm_scales,
                ids,
                top_k,
                s13,
                self.gate_cap,
                self.up_cap,
            )
            w2_grouped(
                out,
                routed,
                act,
                layer.w2_tm_weight,
                layer.w2_tm_scales,
                ids,
                topk_weights.view(-1),
                s2,
            )
        else:
            w13_act(
                act,
                x,
                layer.w13_tm_weight,
                layer.w13_tm_scales,
                ids,
                top_k,
                s13,
                self.gate_cap,
                self.up_cap,
            )
            w2_reduce(
                out,
                act,
                layer.w2_tm_weight,
                layer.w2_tm_scales,
                ids,
                topk_weights.view(-1),
                s2,
            )
        return out
