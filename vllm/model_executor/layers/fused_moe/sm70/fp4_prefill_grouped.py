# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Grouped SM70 NVFP4 MoE prefill route (CUDA JIT, M > 16 tokens per call).

Replaces the per-expert host loop of the TurboMind "dense" stage for custom
softcapped-SwiGLU contracts (Inkling2). Works directly on the TurboMind-prepared
banks of `ModelOptNvFp4SM70MoEMethod` (`w13_tm_weight [E, K, N/8] int32`,
`w13_tm_scales [E, K/16, N] fp16`, same for w2), with the same quadpair
`mma.sync.m8n8k4` f16 / fp32-accumulate fragment decode as `fp4_decode_gemv.py`.

Launches per visit (no host sync, sizes depend only on the token count):
  1. memset counts, 2. count tokens per expert, 3. scan + row-tile list,
  4. scatter slots to expert-sorted order,
  5. W13 + softcapped SwiGLU (gate/up tile pair in registers; fp16 rounding of
     gate/up before the activation, like the unfused path),
  6. W2 (per-route fp16 rounding, scattered to slot order),
  7. weighted reduce to token order (fp32 fma in slot order, deterministic).

A row tile is `8*NCH` rows of one expert (default 32). One warp owns a 32-column
W13 gate/up pair (or a 64-column W2 pair) and loops over K, so each decoded
weight tile is reused by all rows of the tile; the CTA's activation rows are
staged through shared memory in 96-wide K slabs (coalesced loads, conflict-free
fragment reads). Needs K % 96 == 0 for both stages (Inkling: 2304, 288 / 1152). Row results never depend on the other rows of
the tile or on their order, so the atomic scatter order does not change numerics.
"""

import os
from functools import lru_cache

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MIN_TOKENS = int(os.environ.get("VLLM_INKLING2_SM70_MOE_PREFILL_MIN_TOKENS", "17"))
# Largest token count per kernel round; bigger calls are split (bounds the
# [slots, hidden] routed buffer: 4096 tokens x 8 x 2304 x 2 B = 150 MB).
SUB_TOKENS = int(os.environ.get("VLLM_INKLING2_SM70_MOE_PREFILL_SUB_TOKENS", "4096"))
_NCH13 = int(os.environ.get("VLLM_INKLING2_SM70_MOE_PREFILL_NCH13", "4"))
_NCH2 = int(os.environ.get("VLLM_INKLING2_SM70_MOE_PREFILL_NCH2", "4"))

_CPP = r"""
#include <torch/extension.h>
void moe_prefill(torch::Tensor out, torch::Tensor x, torch::Tensor topk_w,
                 torch::Tensor ids, torch::Tensor w13_w, torch::Tensor w13_s,
                 torch::Tensor w2_w, torch::Tensor w2_s, torch::Tensor act,
                 torch::Tensor routed, torch::Tensor sorted_slot,
                 torch::Tensor meta, int64_t top_k, double gate_cap,
                 double up_cap, int64_t nch13, int64_t nch2);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef LB_BLOCKS
#define LB_BLOCKS 3
#endif

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

// Row / column of accumulator `idx` of this lane inside an 8-row chunk x 32-col tile
// (same mapping as `store_rows` of the decode kernels).
__device__ __forceinline__ void frag_pos(int lane, int idx, int& row, int& col) {
  const int t = (lane & 3) | ((lane >> 4) << 2);
  const int quadpair = (lane >> 2) & 3;
  row = (t & 1) | (((idx >> 1) & 1) << 1) | ((t >> 2) << 2);
  col = quadpair * 8 + ((idx & 1) | (((t >> 1) & 1) << 1) | (((idx >> 2) & 1) << 2));
}

// One K group (16) of one 32-col tile: loads + decode into b[8] half2.
struct BTile {
  unsigned p0, p1;
  half s;
};

__device__ __forceinline__ void load_b(BTile& b, const uint32_t* w, const half* s,
                                       int group) {
  b.p0 = __ldg(w + group * 64);
  b.p1 = __ldg(w + group * 64 + 32);
  b.s = __ldg(s);
}

__device__ __forceinline__ void decode_b(const BTile& b, half2 d[8]) {
  const half2 scale =
      __hmul2(__halves2half2(b.s, b.s), __float2half2_rn(16384.0f));
  dequant_e2m1x8(b.p0, scale, d);
  dequant_e2m1x8(b.p1, scale, d + 4);
}

__device__ __forceinline__ void load_a(uint4& a01, uint4& a23, const half* row,
                                       int group) {
  a01 = make_uint4(0, 0, 0, 0);
  a23 = make_uint4(0, 0, 0, 0);
  if (row != nullptr) {
    a01 = __ldg(reinterpret_cast<const uint4*>(row + group * 16));
    a23 = __ldg(reinterpret_cast<const uint4*>(row + group * 16 + 8));
  }
}

// ---------------------------------------------------------------- routing

__global__ void count_kernel(const int32_t* __restrict__ ids, int32_t* counts,
                             int slots, int num_experts) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < slots;
       i += gridDim.x * blockDim.x) {
    const int e = ids[i];
    if (static_cast<unsigned>(e) < static_cast<unsigned>(num_experts))
      atomicAdd(counts + e, 1);
  }
}

// One block. offsets[e] = exclusive scan of counts, tile list, cursor reset.
__global__ void scan_kernel(const int32_t* __restrict__ counts, int32_t* offsets,
                            int32_t* cursor, int32_t* tile_expert,
                            int32_t* tile_row0, int32_t* tile_nrows,
                            int32_t* tile_count, int num_experts, int bm) {
  extern __shared__ int sh[];  // [E] row offsets, [E] tile bases
  int* off = sh;
  int* tbase = sh + num_experts;
  if (threadIdx.x == 0) {
    int r = 0, t = 0;
    for (int e = 0; e < num_experts; ++e) {
      const int c = counts[e];
      off[e] = r;
      tbase[e] = t;
      offsets[e] = r;
      cursor[e] = 0;
      r += c;
      t += (c + bm - 1) / bm;
    }
    offsets[num_experts] = r;
    *tile_count = t;
  }
  __syncthreads();
  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
    const int c = counts[e];
    const int nt = (c + bm - 1) / bm;
    for (int j = 0; j < nt; ++j) {
      tile_expert[tbase[e] + j] = e;
      tile_row0[tbase[e] + j] = off[e] + j * bm;
      const int rem = c - j * bm;
      tile_nrows[tbase[e] + j] = rem < bm ? rem : bm;
    }
  }
}

__global__ void scatter_kernel(const int32_t* __restrict__ ids,
                               const int32_t* __restrict__ offsets,
                               int32_t* cursor, int32_t* sorted_slot, int slots,
                               int num_experts) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < slots;
       i += gridDim.x * blockDim.x) {
    const int e = ids[i];
    if (static_cast<unsigned>(e) < static_cast<unsigned>(num_experts)) {
      const int pos = offsets[e] + atomicAdd(cursor + e, 1);
      sorted_slot[pos] = i;
    }
  }
}

// ---------------------------------------------------------------- W13 + SwiGLU

// ---- A (activation rows) staged through shared memory ----------------------
// The MMA A fragment wants 8 distinct rows x 32 B per load; read straight from
// global that touches 8 cache lines per request and made the kernels L1/TEX
// bound. Instead the CTA loads a K slab of its rows with coalesced 16 B
// accesses (one global row = 12 consecutive lanes) into smem rows padded to
// 208 B, which makes the fragment reads (rows r*208 B + g*32 B) conflict free.
constexpr int KS = 96;                // K slab (halves); K % 96 == 0 (checked on host)
constexpr int APITCH = KS + 8;        // halves per smem row
constexpr int KGROUPS = KS / 16;      // 16-wide MMA K groups per slab
constexpr int CPR = KS / 8;           // 16 B chunks per row per slab

template <int NA>
__device__ __forceinline__ void a_load(uint4 (&r)[4], const half* __restrict__ src,
                                       const int* rowoff, int slab, int tid, int nthr) {
  constexpr int total = NA * 8 * CPR;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int id = tid + i * nthr;
    r[i] = make_uint4(0, 0, 0, 0);
    if (id < total) {
      const int row = id / CPR, q = id - row * CPR;
      const int off = rowoff[row];
      if (off >= 0)
        r[i] = __ldg(reinterpret_cast<const uint4*>(src + off + slab * KS + q * 8));
    }
  }
}

template <int NA>
__device__ __forceinline__ void a_store(half* sA, const uint4 (&r)[4], int tid, int nthr) {
  constexpr int total = NA * 8 * CPR;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int id = tid + i * nthr;
    if (id < total) {
      const int row = id / CPR, q = id - row * CPR;
      *reinterpret_cast<uint4*>(sA + row * APITCH + q * 8) = r[i];
    }
  }
}

template <int NCH>
struct SmemSizes {
  static constexpr int a_bytes = 2 * 8 * NCH * APITCH * 2;
  static constexpr int st13_bytes = 4 * 8 * NCH * 40 * 2;
  static constexpr int st2_bytes = 4 * 8 * NCH * 72 * 2;
  static constexpr int w13 = a_bytes > st13_bytes ? a_bytes : st13_bytes;
  static constexpr int w2 = a_bytes > st2_bytes ? a_bytes : st2_bytes;
};

// ---------------------------------------------------------------- W13 + SwiGLU

template <int NCH, int NA>
__device__ __forceinline__ void w13_body(
    half* smem_a, const int* rowoff, const half* __restrict__ x,
    const uint32_t* wg, const uint32_t* wu, const half* sg, const half* su,
    size_t sstride, int groups, int lane, int warp, int tid, int nthr, int nrows,
    int row0, int pair, int inter, float gate_cap, float up_cap,
    half* __restrict__ act) {
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  float ag[NA][8], au[NA][8];
#pragma unroll
  for (int c = 0; c < NA; ++c)
#pragma unroll
    for (int i = 0; i < 8; ++i) ag[c][i] = au[c][i] = 0.0f;
  const int nslabs = groups / KGROUPS;
  uint4 pf[4];
  a_load<NA>(pf, x, rowoff, 0, tid, nthr);
  a_store<NA>(smem_a, pf, tid, nthr);
  __syncthreads();
  BTile bg, bu;
  load_b(bg, wg, sg, 0);
  load_b(bu, wu, su, 0);
  for (int sl = 0; sl < nslabs; ++sl) {
    const bool more = sl + 1 < nslabs;
    if (more) a_load<NA>(pf, x, rowoff, sl + 1, tid, nthr);
    const half* sA = smem_a + (sl & 1) * (8 * NCH * APITCH);
#pragma unroll 1
    for (int gi = 0; gi < KGROUPS; ++gi) {
      const int g = sl * KGROUPS + gi;
      half2 dg[8], du[8];
      decode_b(bg, dg);
      decode_b(bu, du);
      if (g + 1 < groups) {
        load_b(bg, wg, sg + (g + 1) * sstride, g + 1);
        load_b(bu, wu, su + (g + 1) * sstride, g + 1);
      }
      uint4 a01[NA], a23[NA];
#pragma unroll
      for (int c = 0; c < NA; ++c) {
        const half* ar = sA + (c * 8 + a_row) * APITCH + gi * 16;
        a01[c] = *reinterpret_cast<const uint4*>(ar);
        a23[c] = *reinterpret_cast<const uint4*>(ar + 8);
      }
      const unsigned* bgw = reinterpret_cast<const unsigned*>(dg);
      const unsigned* buw = reinterpret_cast<const unsigned*>(du);
#pragma unroll
      for (int s = 0; s < 4; ++s) {
#pragma unroll
        for (int c = 0; c < NA; ++c) {
          const unsigned* aw = reinterpret_cast<const unsigned*>(s < 2 ? &a01[c] : &a23[c]);
          MMA_8N8K4(ag[c], aw[(s & 1) * 2], aw[(s & 1) * 2 + 1], bgw[2 * s], bgw[2 * s + 1]);
          MMA_8N8K4(au[c], aw[(s & 1) * 2], aw[(s & 1) * 2 + 1], buw[2 * s], buw[2 * s + 1]);
        }
      }
    }
    if (more) a_store<NA>(smem_a + ((sl + 1) & 1) * (8 * NCH * APITCH), pf, tid, nthr);
    __syncthreads();
  }

  // Epilogue (smem reused as the output stage): fp16 round gate/up, softcapped
  // SwiGLU in fp32, fp16 store.
  half* stage = smem_a + warp * (8 * NCH * 40);
#pragma unroll
  for (int c = 0; c < NA; ++c) {
#pragma unroll
    for (int idx = 0; idx < 8; ++idx) {
      int r, col;
      frag_pos(lane, idx, r, col);
      const float gh = __half2float(__float2half(ag[c][idx]));
      const float uh = __half2float(__float2half(au[c][idx]));
      float gf = gh;
      if (gate_cap > 0.0f) gf = gate_cap * tanhf(gh / gate_cap);
      float uf = uh;
      if (up_cap > 0.0f) uf = up_cap * tanhf(uh / up_cap);
      const float sig = 1.0f / (1.0f + expf(-gh));
      stage[(c * 8 + r) * 40 + col] = __float2half(gf * sig * uf);
    }
  }
  __syncwarp();
  for (int j = lane; j < 8 * NA * 4; j += 32) {
    const int r = j >> 2, q = j & 3;
    if (r < nrows)
      *reinterpret_cast<uint4*>(act + static_cast<size_t>(row0 + r) * inter +
                                pair * 32 + q * 8) =
          *reinterpret_cast<const uint4*>(&stage[r * 40 + q * 8]);
  }
}

// grid (inter/32/NW, max_tiles); block 32*NW. Warp = one gate/up tile pair.
template <int NCH>
__global__ void __launch_bounds__(128, LB_BLOCKS) w13_prefill_kernel(
    const half* __restrict__ x, const uint32_t* __restrict__ w,
    const half* __restrict__ sc, const int32_t* __restrict__ sorted_slot,
    const int32_t* __restrict__ tile_expert, const int32_t* __restrict__ tile_row0,
    const int32_t* __restrict__ tile_nrows, const int32_t* __restrict__ tile_count,
    half* __restrict__ act, int inter, int k, int top_k, float gate_cap,
    float up_cap) {
  if (static_cast<int>(blockIdx.y) >= *tile_count) return;
  __shared__ __align__(16) unsigned char smem_raw[SmemSizes<NCH>::w13];
  __shared__ int rowoff[8 * NCH];
  half* smem_a = reinterpret_cast<half*>(smem_raw);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int expert = tile_expert[blockIdx.y];
  const int row0 = tile_row0[blockIdx.y];
  const int nrows = tile_nrows[blockIdx.y];
  const int nthr = blockDim.x;
  const int pair = blockIdx.x * (nthr >> 5) + warp;
  const int n = inter * 2;
  const int tiles_n32 = n >> 5;
  const int groups = k >> 4;
  const int groups_k8 = k >> 3;
  const int packed_col =
      ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const uint32_t* ew = w + static_cast<size_t>(expert) * (static_cast<size_t>(k) * n / 8);
  const half* es = sc + static_cast<size_t>(expert) * (static_cast<size_t>(groups) * n);
  const int tile_g = pair, tile_u = pair + (inter >> 5);
  const uint32_t* wg = ew + static_cast<size_t>(tile_g) * groups_k8 * 32 + packed_col;
  const uint32_t* wu = ew + static_cast<size_t>(tile_u) * groups_k8 * 32 + packed_col;
  const half* sg = es + tile_g * 32 + packed_col;
  const half* su = es + tile_u * 32 + packed_col;
  const size_t sstride = static_cast<size_t>(tiles_n32) * 32;
  if (threadIdx.x < 8 * NCH)
    rowoff[threadIdx.x] =
        threadIdx.x < nrows ? (sorted_slot[row0 + threadIdx.x] / top_k) * k : -1;
  __syncthreads();

  const int nact = (nrows + 7) >> 3;
#define W13_CALL(NA)                                                              \
  w13_body<NCH, NA>(smem_a, rowoff, x, wg, wu, sg, su, sstride, groups, lane, warp, \
                    threadIdx.x, nthr, nrows, row0, pair, inter, gate_cap, up_cap, act)
  switch (nact) {
    case 1: W13_CALL(1); break;
    case 2: if (NCH >= 2) W13_CALL((NCH >= 2 ? 2 : 1)); break;
    case 3: if (NCH >= 3) W13_CALL((NCH >= 3 ? 3 : 1)); break;
    default: if (NCH >= 4) W13_CALL((NCH >= 4 ? 4 : 1)); break;
  }
#undef W13_CALL
}

// ---------------------------------------------------------------- W2

template <int NCH, int NA>
__device__ __forceinline__ void w2_body(
    half* smem_a, const int* rowoff, const half* __restrict__ act,
    const uint32_t* w0, const uint32_t* w1, const half* s0, const half* s1,
    size_t sstride, int groups, int lane, int warp, int tid, int nthr, int nrows,
    int row0, int pair, int n, const int32_t* __restrict__ sorted_slot,
    half* __restrict__ routed) {
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  float a0[NA][8], a1[NA][8];
#pragma unroll
  for (int c = 0; c < NA; ++c)
#pragma unroll
    for (int i = 0; i < 8; ++i) a0[c][i] = a1[c][i] = 0.0f;
  const int nslabs = groups / KGROUPS;
  uint4 pf[4];
  a_load<NA>(pf, act, rowoff, 0, tid, nthr);
  a_store<NA>(smem_a, pf, tid, nthr);
  __syncthreads();
  BTile b0, b1;
  load_b(b0, w0, s0, 0);
  load_b(b1, w1, s1, 0);
  for (int sl = 0; sl < nslabs; ++sl) {
    const bool more = sl + 1 < nslabs;
    if (more) a_load<NA>(pf, act, rowoff, sl + 1, tid, nthr);
    const half* sA = smem_a + (sl & 1) * (8 * NCH * APITCH);
#pragma unroll 1
    for (int gi = 0; gi < KGROUPS; ++gi) {
      const int g = sl * KGROUPS + gi;
      half2 d0[8], d1[8];
      decode_b(b0, d0);
      decode_b(b1, d1);
      if (g + 1 < groups) {
        load_b(b0, w0, s0 + (g + 1) * sstride, g + 1);
        load_b(b1, w1, s1 + (g + 1) * sstride, g + 1);
      }
      uint4 x01[NA], x23[NA];
#pragma unroll
      for (int c = 0; c < NA; ++c) {
        const half* ar = sA + (c * 8 + a_row) * APITCH + gi * 16;
        x01[c] = *reinterpret_cast<const uint4*>(ar);
        x23[c] = *reinterpret_cast<const uint4*>(ar + 8);
      }
      const unsigned* b0w = reinterpret_cast<const unsigned*>(d0);
      const unsigned* b1w = reinterpret_cast<const unsigned*>(d1);
#pragma unroll
      for (int s = 0; s < 4; ++s) {
#pragma unroll
        for (int c = 0; c < NA; ++c) {
          const unsigned* aw = reinterpret_cast<const unsigned*>(s < 2 ? &x01[c] : &x23[c]);
          MMA_8N8K4(a0[c], aw[(s & 1) * 2], aw[(s & 1) * 2 + 1], b0w[2 * s], b0w[2 * s + 1]);
          MMA_8N8K4(a1[c], aw[(s & 1) * 2], aw[(s & 1) * 2 + 1], b1w[2 * s], b1w[2 * s + 1]);
        }
      }
    }
    if (more) a_store<NA>(smem_a + ((sl + 1) & 1) * (8 * NCH * APITCH), pf, tid, nthr);
    __syncthreads();
  }
  half* stage = smem_a + warp * (8 * NCH * 72);
#pragma unroll
  for (int c = 0; c < NA; ++c) {
#pragma unroll
    for (int idx = 0; idx < 8; ++idx) {
      int r, col;
      frag_pos(lane, idx, r, col);
      stage[(c * 8 + r) * 72 + col] = __float2half(a0[c][idx]);
      stage[(c * 8 + r) * 72 + 32 + col] = __float2half(a1[c][idx]);
    }
  }
  __syncwarp();
  for (int j = lane; j < 8 * NA * 8; j += 32) {
    const int r = j >> 3, q = j & 7;
    if (r < nrows) {
      const int slot = sorted_slot[row0 + r];
      *reinterpret_cast<uint4*>(routed + static_cast<size_t>(slot) * n + pair * 64 +
                                q * 8) =
          *reinterpret_cast<const uint4*>(&stage[r * 72 + q * 8]);
    }
  }
}

// grid (n/64/NW, max_tiles); block 32*NW. Warp = two adjacent 32-col tiles.
template <int NCH>
__global__ void __launch_bounds__(128, LB_BLOCKS) w2_prefill_kernel(
    const half* __restrict__ act, const uint32_t* __restrict__ w,
    const half* __restrict__ sc, const int32_t* __restrict__ sorted_slot,
    const int32_t* __restrict__ tile_expert, const int32_t* __restrict__ tile_row0,
    const int32_t* __restrict__ tile_nrows, const int32_t* __restrict__ tile_count,
    half* __restrict__ routed, int n, int k) {
  if (static_cast<int>(blockIdx.y) >= *tile_count) return;
  __shared__ __align__(16) unsigned char smem_raw[SmemSizes<NCH>::w2];
  __shared__ int rowoff[8 * NCH];
  half* smem_a = reinterpret_cast<half*>(smem_raw);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int expert = tile_expert[blockIdx.y];
  const int row0 = tile_row0[blockIdx.y];
  const int nrows = tile_nrows[blockIdx.y];
  const int nthr = blockDim.x;
  const int pair = blockIdx.x * (nthr >> 5) + warp;
  const int tiles_n32 = n >> 5;
  const int groups = k >> 4;
  const int groups_k8 = k >> 3;
  const int packed_col =
      ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const uint32_t* ew = w + static_cast<size_t>(expert) * (static_cast<size_t>(k) * n / 8);
  const half* es = sc + static_cast<size_t>(expert) * (static_cast<size_t>(groups) * n);
  const int t0 = pair * 2, t1 = pair * 2 + 1;
  const uint32_t* w0 = ew + static_cast<size_t>(t0) * groups_k8 * 32 + packed_col;
  const uint32_t* w1 = ew + static_cast<size_t>(t1) * groups_k8 * 32 + packed_col;
  const half* s0 = es + t0 * 32 + packed_col;
  const half* s1 = es + t1 * 32 + packed_col;
  const size_t sstride = static_cast<size_t>(tiles_n32) * 32;
  if (threadIdx.x < 8 * NCH)
    rowoff[threadIdx.x] = threadIdx.x < nrows ? (row0 + threadIdx.x) * k : -1;
  __syncthreads();

  const int nact = (nrows + 7) >> 3;
#define W2_CALL(NA)                                                                \
  w2_body<NCH, NA>(smem_a, rowoff, act, w0, w1, s0, s1, sstride, groups, lane, warp, \
                   threadIdx.x, nthr, nrows, row0, pair, n, sorted_slot, routed)
  switch (nact) {
    case 1: W2_CALL(1); break;
    case 2: if (NCH >= 2) W2_CALL((NCH >= 2 ? 2 : 1)); break;
    case 3: if (NCH >= 3) W2_CALL((NCH >= 3 ? 3 : 1)); break;
    default: if (NCH >= 4) W2_CALL((NCH >= 4 ? 4 : 1)); break;
  }
#undef W2_CALL
}

// out[t, :] = sum_s w[t, s] * routed[t * top_k + s, :], slot order, fp32 fma.
// grid (tokens, ceil(n / 8 / blockDim)); 8 columns per thread.
__global__ void reduce_kernel(const half* __restrict__ routed,
                              const float* __restrict__ topk_w,
                              const int32_t* __restrict__ ids,
                              half* __restrict__ out, int n, int top_k,
                              int num_experts) {
  const int t = blockIdx.x;
  const int c8 = blockIdx.y * blockDim.x + threadIdx.x;
  if (c8 * 8 >= n) return;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
  for (int s = 0; s < top_k; ++s) {
    const int e = ids[t * top_k + s];
    if (static_cast<unsigned>(e) >= static_cast<unsigned>(num_experts)) continue;
    const float wt = __ldg(topk_w + t * top_k + s);
    const uint4 v = __ldg(reinterpret_cast<const uint4*>(
        routed + (static_cast<size_t>(t) * top_k + s) * n + c8 * 8));
    const half2* h = reinterpret_cast<const half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __half22float2(h[i]);
      acc[2 * i] = fmaf(wt, f.x, acc[2 * i]);
      acc[2 * i + 1] = fmaf(wt, f.y, acc[2 * i + 1]);
    }
  }
  uint4 o;
  half2* oh = reinterpret_cast<half2*>(&o);
#pragma unroll
  for (int i = 0; i < 4; ++i) oh[i] = __floats2half2_rn(acc[2 * i], acc[2 * i + 1]);
  *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * n + c8 * 8) = o;
}

int pick_nw(int pairs) {
  for (int nw = 4; nw >= 1; --nw)
    if (pairs % nw == 0) return nw;
  return 1;
}

}  // namespace

void moe_prefill(torch::Tensor out, torch::Tensor x, torch::Tensor topk_w,
                 torch::Tensor ids, torch::Tensor w13_w, torch::Tensor w13_s,
                 torch::Tensor w2_w, torch::Tensor w2_s, torch::Tensor act,
                 torch::Tensor routed, torch::Tensor sorted_slot,
                 torch::Tensor meta, int64_t top_k, double gate_cap,
                 double up_cap, int64_t nch13, int64_t nch2) {
  TORCH_CHECK(out.is_cuda() && x.is_cuda() && ids.is_cuda() && w13_w.is_cuda(),
              "moe_prefill: CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 && out.scalar_type() == torch::kFloat16 &&
              act.scalar_type() == torch::kFloat16 && routed.scalar_type() == torch::kFloat16 &&
              w13_w.scalar_type() == torch::kInt32 && w2_w.scalar_type() == torch::kInt32 &&
              w13_s.scalar_type() == torch::kFloat16 && w2_s.scalar_type() == torch::kFloat16 &&
              ids.scalar_type() == torch::kInt32 && topk_w.scalar_type() == torch::kFloat32 &&
              sorted_slot.scalar_type() == torch::kInt32 && meta.scalar_type() == torch::kInt32,
              "moe_prefill: dtype");
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && ids.is_contiguous() &&
              topk_w.is_contiguous() && w13_w.is_contiguous() && w13_s.is_contiguous() &&
              w2_w.is_contiguous() && w2_s.is_contiguous() && act.is_contiguous() &&
              routed.is_contiguous() && sorted_slot.is_contiguous() && meta.is_contiguous(),
              "moe_prefill: contiguous");
  const int64_t tokens = x.size(0), k = x.size(1), n = out.size(1);
  const int64_t slots = tokens * top_k;
  const int64_t experts = w13_w.size(0);
  const int64_t inter = act.size(1);
  TORCH_CHECK(ids.numel() == slots && topk_w.numel() == slots && out.size(0) == tokens &&
              n == k && k % 16 == 0 && inter % 32 == 0 && n % 64 == 0 &&
              act.size(0) >= slots && routed.size(0) >= slots && routed.size(1) == n &&
              sorted_slot.numel() >= slots && w13_w.dim() == 3 && w13_w.size(1) == k &&
              w13_w.size(2) * 8 == 2 * inter && w13_s.size(1) == k / 16 &&
              w13_s.size(2) == 2 * inter && w2_w.size(0) == experts &&
              w2_w.size(1) == inter && w2_w.size(2) * 8 == n && w2_s.size(1) == inter / 16 &&
              w2_s.size(2) == n && inter % 16 == 0,
              "moe_prefill: shape contract");
  TORCH_CHECK(k % 96 == 0 && inter % 96 == 0, "moe_prefill: K and I must be multiples of 96");
  TORCH_CHECK(nch13 == 1 || nch13 == 2 || nch13 == 4, "moe_prefill: nch13");
  TORCH_CHECK(nch2 == 1 || nch2 == 2 || nch2 == 4, "moe_prefill: nch2");
  const int bm = static_cast<int>(8 * (nch13 > nch2 ? nch13 : nch2));
  // Both stages share one tile list, so both use the same rows-per-tile.
  TORCH_CHECK(nch13 == nch2, "moe_prefill: nch13 must equal nch2 (shared tile list)");
  const int64_t max_tiles = (slots + bm - 1) / bm + experts;
  TORCH_CHECK(meta.numel() >= 3 * experts + 2 + 3 * max_tiles, "moe_prefill: meta size");
  int32_t* counts = meta.data_ptr<int32_t>();
  int32_t* cursor = counts + experts;
  int32_t* offsets = cursor + experts;          // experts + 1
  int32_t* tile_count = offsets + experts + 1;  // 1
  int32_t* tile_expert = tile_count + 1;
  int32_t* tile_row0 = tile_expert + max_tiles;
  int32_t* tile_nrows = tile_row0 + max_tiles;

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int route_blocks = static_cast<int>(std::min<int64_t>((slots + 255) / 256, 1024));
  cudaMemsetAsync(counts, 0, experts * sizeof(int32_t), stream);
  count_kernel<<<route_blocks, 256, 0, stream>>>(ids.data_ptr<int32_t>(), counts,
                                                 (int)slots, (int)experts);
  scan_kernel<<<1, 128, 2 * experts * sizeof(int), stream>>>(
      counts, offsets, cursor, tile_expert, tile_row0, tile_nrows, tile_count,
      (int)experts, bm);
  scatter_kernel<<<route_blocks, 256, 0, stream>>>(
      ids.data_ptr<int32_t>(), offsets, cursor, sorted_slot.data_ptr<int32_t>(),
      (int)slots, (int)experts);

  const half* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  half* actp = reinterpret_cast<half*>(act.data_ptr<at::Half>());
  half* routedp = reinterpret_cast<half*>(routed.data_ptr<at::Half>());
  const int nw13 = pick_nw(static_cast<int>(inter / 32));
  const int nw2 = pick_nw(static_cast<int>(n / 64));
  const dim3 g13(static_cast<unsigned>(inter / 32 / nw13), static_cast<unsigned>(max_tiles));
  const dim3 g2(static_cast<unsigned>(n / 64 / nw2), static_cast<unsigned>(max_tiles));
#define LAUNCH_STAGES(NCH)                                                          \
  w13_prefill_kernel<NCH><<<g13, 32 * nw13, 0, stream>>>(                          \
      xp, reinterpret_cast<const uint32_t*>(w13_w.data_ptr<int32_t>()),            \
      reinterpret_cast<const half*>(w13_s.data_ptr<at::Half>()),                   \
      sorted_slot.data_ptr<int32_t>(), tile_expert, tile_row0, tile_nrows,         \
      tile_count, actp, (int)inter, (int)k, (int)top_k, (float)gate_cap,           \
      (float)up_cap);                                                              \
  w2_prefill_kernel<NCH><<<g2, 32 * nw2, 0, stream>>>(                             \
      actp, reinterpret_cast<const uint32_t*>(w2_w.data_ptr<int32_t>()),           \
      reinterpret_cast<const half*>(w2_s.data_ptr<at::Half>()),                    \
      sorted_slot.data_ptr<int32_t>(), tile_expert, tile_row0, tile_nrows,         \
      tile_count, routedp, (int)n, (int)inter);
  switch (nch13) {
    case 1: LAUNCH_STAGES(1) break;
    case 2: LAUNCH_STAGES(2) break;
    default: LAUNCH_STAGES(4) break;
  }
#undef LAUNCH_STAGES
  const int rthreads = 128;
  const dim3 rgrid(static_cast<unsigned>(tokens),
                   static_cast<unsigned>((n / 8 + rthreads - 1) / rthreads));
  reduce_kernel<<<rgrid, rthreads, 0, stream>>>(
      routedp, topk_w.data_ptr<float>(), ids.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), (int)n, (int)top_k,
      (int)experts);
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
        "moe_fp4_prefill",
    )
    os.makedirs(build_dir, exist_ok=True)
    return load_inline(
        name="sm70_nvfp4_prefill_grouped",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["moe_prefill"],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_70,code=sm_70",
            "-lineinfo",
            "-DLB_BLOCKS=" + os.environ.get("VLLM_INKLING2_SM70_MOE_PREFILL_LB", "3"),
        ]
        + (["-Xptxas=-v"] if os.environ.get("VLLM_INKLING2_SM70_EXT_VERBOSE") else []),
        build_directory=build_dir,
        verbose=bool(os.environ.get("VLLM_INKLING2_SM70_EXT_VERBOSE")),
    )


def moe_prefill(
    out,
    x,
    topk_w,
    ids,
    w13_w,
    w13_s,
    w2_w,
    w2_s,
    act,
    routed,
    sorted_slot,
    meta,
    top_k,
    gate_cap,
    up_cap,
    nch13,
    nch2,
):
    _ext().moe_prefill(
        out,
        x,
        topk_w,
        ids,
        w13_w,
        w13_s,
        w2_w,
        w2_s,
        act,
        routed,
        sorted_slot,
        meta,
        top_k,
        gate_cap,
        up_cap,
        nch13,
        nch2,
    )


_WORKSPACES: dict = {}


def _workspace(device, sub_tokens, top_k, hidden, inter, experts, nch):
    """Per-device buffers shared by all layers (layers run one after another on
    the stream). Grown only when a bigger call arrives."""
    key = (device, top_k, hidden, inter, experts)
    slots = sub_tokens * top_k
    ws = _WORKSPACES.get(key)
    if ws is None or ws["slots"] < slots:
        bm = 8 * nch
        max_tiles = (slots + bm - 1) // bm + experts
        ws = {
            "slots": slots,
            "act": torch.empty(slots, inter, dtype=torch.float16, device=device),
            "routed": torch.empty(slots, hidden, dtype=torch.float16, device=device),
            "sorted": torch.empty(slots, dtype=torch.int32, device=device),
            "meta": torch.empty(
                3 * experts + 2 + 3 * max_tiles, dtype=torch.int32, device=device
            ),
        }
        _WORKSPACES[key] = ws
    return ws


def _softcap_params(activation):
    """(gate_cap, up_cap) of a `partial(softcapped_swiglu_, ...)` hook, else None."""
    func = getattr(activation, "func", None)
    if getattr(func, "__name__", None) != "softcapped_swiglu_":
        return None
    kw = getattr(activation, "keywords", None) or {}
    if set(kw) != {"gate_cap", "up_cap"} or getattr(activation, "args", ()):
        return None
    return float(kw["gate_cap"]), float(kw["up_cap"])


class Fp4PrefillGrouped:
    """Per-layer M >= MIN_TOKENS grouped NVFP4 MoE route (see module docstring)."""

    def __init__(self, gate_cap, up_cap, top_k, hidden, inter, experts):
        self.gate_cap, self.up_cap, self.top_k = gate_cap, up_cap, top_k
        self.hidden, self.inter, self.experts = hidden, inter, experts

    @classmethod
    def create(cls, layer, *, raw_scale):
        if os.environ.get("VLLM_INKLING2_SM70_MOE_PREFILL", "1") == "0":
            return None
        caps = _softcap_params(getattr(layer, "sm70_moe_activation", None))
        if (
            caps is None
            or raw_scale
            or getattr(layer, "swiglu_limit", None) is not None
            or not torch.cuda.is_available()
            or torch.cuda.get_device_capability(layer.w13_tm_weight.device) != (7, 0)
            or layer.w13_tm_scales.dtype != torch.float16
        ):
            return None
        hidden = int(layer.sm70_nvfp4_hidden_size)
        inter = int(layer.sm70_nvfp4_intermediate_size)
        top_k = int(layer.sm70_nvfp4_top_k)
        experts = int(layer.w13_tm_weight.shape[0])
        if hidden % 192 or inter % 96 or experts > 1024 or _NCH13 != _NCH2:
            return None
        self = cls(caps[0], caps[1], top_k, hidden, inter, experts)
        _ext()  # compile / load now, never lazily inside a request
        logger.info_once(
            "SM70 NVFP4 grouped prefill MoE route enabled (hidden=%d, "
            "local_intermediate=%d, experts=%d, top_k=%d, M>=%d, tile rows=%d, "
            "softcap gate=%g up=%g).",
            hidden,
            inter,
            experts,
            top_k,
            MIN_TOKENS,
            8 * _NCH13,
            caps[0],
            caps[1],
        )
        return self

    def __call__(self, layer, x, topk_weights, topk_ids):
        tokens = x.shape[0]
        if (
            tokens < MIN_TOKENS
            or x.dtype != torch.float16
            or not x.is_contiguous()
            or not topk_ids.is_contiguous()
            or topk_ids.dtype != torch.int32
            or not topk_weights.is_contiguous()
        ):
            return None
        top_k, hidden = self.top_k, self.hidden
        out = torch.empty(tokens, hidden, dtype=torch.float16, device=x.device)
        sub = min(tokens, SUB_TOKENS)
        ws = _workspace(x.device, sub, top_k, hidden, self.inter, self.experts, _NCH13)
        ids = topk_ids.view(-1)
        wts = topk_weights.view(-1)
        for t0 in range(0, tokens, sub):
            t1 = min(tokens, t0 + sub)
            moe_prefill(
                out[t0:t1],
                x[t0:t1],
                wts[t0 * top_k : t1 * top_k],
                ids[t0 * top_k : t1 * top_k],
                layer.w13_tm_weight,
                layer.w13_tm_scales,
                layer.w2_tm_weight,
                layer.w2_tm_scales,
                ws["act"],
                ws["routed"],
                ws["sorted"],
                ws["meta"],
                top_k,
                self.gate_cap,
                self.up_cap,
                _NCH13,
                _NCH2,
            )
        return out
