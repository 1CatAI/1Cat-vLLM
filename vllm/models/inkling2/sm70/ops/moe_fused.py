# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""One-launch Inkling2 NVFP4 MoE visit for SM70 decode (M <= 8 tokens).

Replaces the three launches of a decode visit (`router_fused_kernel`,
`w13_act*_kernel`, `w2_*_kernel` (+ `token_reduce_kernel`)) by ONE persistent
kernel: 80 CTAs x 1024 threads (one per SM), all co-resident, three phases separated by
two spin grid barriers (self-resetting atomic counters, CUDA-graph safe):

  R    router GEMV. CTA c owns expert rows c and c + 80 (9 warps per row).
       Arithmetic copied from `ops/router.py`: logits are bit-identical.
  --   barrier; every CTA runs the select for all M tokens itself (no second
       barrier): 4 warps per token, each files its 8 best keys, one warp
       ranks the 32 candidates (same keys, order and weights as the router's
       iterative arg-max). Then the expert table: distinct experts in
       ascending id, each with the routes that picked it.
  W13  jobs = distinct experts x tile pairs, dealt round-robin to the 80
       CTAs; 32 warps = 2 projections x 16
       K-splits; the MMA A rows are the tokens routed to the expert, so an
       expert is read once however many tokens picked it. Softcapped SwiGLU
       in the epilogue, fp16 activations to a global buffer.
  --   barrier
  W2   CTA (the first 72) = one 32-column output tile; warps = (distinct
       expert, K-split);
       the activations are first copied to shared memory in one round trip;
       per-route fp16 rounding, then the fp32 fma weighted sum in slot order.

The K-splits are the ones production uses at M = 1 (16 for W13; W2: 3 at
TP4, 4 at TP1) for every M, and every accumulation order is that of
`fp4_decode_gemv.py`'s per-route kernels. So the output is bit-identical to
the current path at M = 1, and a token's output does not depend on the number
of tokens in the step (the current path changes splits with M).

Data written by another CTA in the same launch (logits, activations) is read
with `ld.cg` / volatile loads: L1 is not coherent across SMs.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_CPP = r"""
#include <torch/extension.h>
void moe_fused(torch::Tensor x, torch::Tensor gate_w, torch::Tensor gate_b,
               torch::Tensor w13, torch::Tensor sc13, torch::Tensor w2, torch::Tensor sc2,
               torch::Tensor logits, torch::Tensor act, torch::Tensor counters,
               torch::Tensor w_out, torch::Tensor ids_out, torch::Tensor out,
               int64_t s2, double route_scale, double gate_cap, double up_cap, int64_t phases);
void moe_fused_dense(torch::Tensor x, torch::Tensor gate_w, torch::Tensor gate_b,
                     torch::Tensor w13, torch::Tensor w2,
                     torch::Tensor logits, torch::Tensor act, torch::Tensor counters,
                     torch::Tensor w_out, torch::Tensor ids_out, torch::Tensor out,
                     int64_t s2, double route_scale, double gate_cap, double up_cap, int64_t phases);
void moe_route(torch::Tensor x, torch::Tensor gate_w, torch::Tensor gate_b, torch::Tensor logits,
               torch::Tensor counters, torch::Tensor w_out, torch::Tensor ids_out, double route_scale);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kE = 128;                 // experts
constexpr int kTopK = 8;
constexpr int kH = 2304;                // hidden
constexpr int kMaxM = 8;                // tokens per launch (one MMA = 8 A rows)
constexpr int kMaxR = kMaxM * kTopK;    // routes
constexpr int kThreads = 1024;
constexpr int kWarps = 32;
constexpr int kGrid = 80;               // CTAs: one per SM of a V100 (all co-resident for the barriers)
constexpr int kTiles2 = kH / 32;        // 72 output tiles of W2: CTAs 72..79 only help in R and W13
constexpr int kRW = 9;                  // router warps per expert row (kH / 256)
constexpr int kS13 = 16;                // K-splits of W13 (warps per projection)
constexpr int kNG13 = (kH / 16) / kS13; // K groups of 16 per W13 warp
constexpr float kKeyFloor = -3.0e38f;    // lowest key of the select (see there)
constexpr int kChunk13 = VLLM_CHUNK13;  // W13: K groups of a warp whose loads are hoisted together
constexpr int kChunk2 = VLLM_CHUNK2;    // W2: same (the host checks that it divides the groups per warp)
constexpr size_t kMaxStage = 46 * 1024; // dynamic shared memory for staged activations

// ---- NVFP4 quadpair GEMV (same decode and MMA order as fp4_decode_gemv.py) ----

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

__device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

// One warp: 32 output columns (tile) of one expert over K groups [g0, g0 + ng),
// against up to 8 A rows. `row_ptr` is this lane's A row (lane -> row
// (lane & 3) + 4 * (lane >> 4)) or nullptr. kCg: the rows were written by other
// CTAs of this launch.
template <bool kCg, int kChunk>
__device__ __forceinline__ void gemv_rows(
    float accum[8], const half* row_ptr,
    const uint32_t* __restrict__ expert_weights,
    const half* __restrict__ expert_scales, int tile, int tiles_n32,
    int groups_k8, int g0, int ng, int lane) {
  const int packed_col =
      ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
#pragma unroll
  for (int i = 0; i < 8; ++i) accum[i] = 0.0f;
  for (int c0 = 0; c0 < ng; c0 += kChunk) {
    unsigned p0[kChunk], p1[kChunk];
    half sc[kChunk];
#pragma unroll
    for (int j = 0; j < kChunk; ++j) {
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
    for (int j = 0; j < kChunk; ++j) {
      const int group = g0 + c0 + j;
      const half2 scale = __hmul2(__halves2half2(sc[j], sc[j]),
                                  __float2half2_rn(16384.0f));
      half2 decoded[8];
      dequant_e2m1x8(p0[j], scale, decoded);
      dequant_e2m1x8(p1[j], scale, decoded + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
      uint4 in01 = make_uint4(0, 0, 0, 0);
      uint4 in23 = make_uint4(0, 0, 0, 0);
      if (row_ptr != nullptr) {
        const uint4* p = reinterpret_cast<const uint4*>(row_ptr + group * 16);
        if (kCg) {
          in01 = __ldcg(p);
          in23 = __ldcg(p + 1);
        } else {
          in01 = p[0];
          in23 = p[1];
        }
      }
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&in01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&in23);
      MMA_8N8K4(accum, a0[0], a0[1], b[0], b[1]);
      MMA_8N8K4(accum, a0[2], a0[3], b[2], b[3]);
      MMA_8N8K4(accum, a1[0], a1[1], b[4], b[5]);
      MMA_8N8K4(accum, a1[2], a1[3], b[6], b[7]);
    }
  }
}

// Same for resident fp16 experts (the dequantized MXFP8 layers): `wrow` is this
// lane's weight row (output column tile * 32 + packed_col), K contiguous; its
// halves are the MMA B operand as they are.
template <bool kCg, int kChunk>
__device__ __forceinline__ void gemv_rows_dense(
    float accum[8], const half* row_ptr, const half* __restrict__ wrow, int g0, int ng) {
#pragma unroll
  for (int i = 0; i < 8; ++i) accum[i] = 0.0f;
  for (int c0 = 0; c0 < ng; c0 += kChunk) {
    uint4 b01[kChunk], b23[kChunk];
#pragma unroll
    for (int j = 0; j < kChunk; ++j) {
      const uint4* wp = reinterpret_cast<const uint4*>(wrow + (g0 + c0 + j) * 16);
      b01[j] = __ldcs(wp);
      b23[j] = __ldcs(wp + 1);
    }
#pragma unroll
    for (int j = 0; j < kChunk; ++j) {
      const int group = g0 + c0 + j;
      uint4 in01 = make_uint4(0, 0, 0, 0);
      uint4 in23 = make_uint4(0, 0, 0, 0);
      if (row_ptr != nullptr) {
        const uint4* p = reinterpret_cast<const uint4*>(row_ptr + group * 16);
        if (kCg) {
          in01 = __ldcg(p);
          in23 = __ldcg(p + 1);
        } else {
          in01 = p[0];
          in23 = p[1];
        }
      }
      MMA_8N8K4(accum, in01.x, in01.y, b01[j].x, b01[j].y);
      MMA_8N8K4(accum, in01.z, in01.w, b01[j].z, b01[j].w);
      MMA_8N8K4(accum, in23.x, in23.y, b23[j].x, b23[j].y);
      MMA_8N8K4(accum, in23.z, in23.w, b23[j].z, b23[j].w);
    }
  }
}

// part[row][32 cols] <- the 8 rows of the warp's accumulators (all lanes).
__device__ __forceinline__ void store_rows(float* part, const float accum[8], int lane) {
  const int t = (lane & 3) | ((lane >> 4) << 2);
  const int quadpair = (lane >> 2) & 3;
#pragma unroll
  for (int idx = 0; idx < 8; ++idx) {
    const int row = (t & 1) | (((idx >> 1) & 1) << 1) | ((t >> 2) << 2);
    const int col = quadpair * 8 +
                    ((idx & 1) | (((t >> 1) & 1) << 1) | (((idx >> 2) & 1) << 2));
    part[row * 32 + col] = accum[idx];
  }
}

// Grid-wide barrier over the G co-resident CTAs. The counter cycles 0..2G-1
// (atomicInc wraps): a launch that finds it below G waits for >= G, the next
// one waits for the wrap below G. Never needs a reset, no host sync.
__device__ __forceinline__ void grid_barrier(unsigned* ctr, int G) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned old = atomicInc(ctr, 2u * G - 1u);
    const bool ph1 = old >= (unsigned)G;
    volatile unsigned* c = ctr;
    while (true) {
      const unsigned v = *c;
      if (ph1 ? (v < (unsigned)G) : (v >= (unsigned)G)) break;
    }
    __threadfence();
  }
  __syncthreads();
}

struct Tables {
  int ids[kMaxR];                 // [token][slot] expert, slot = rank of its key
  float w[kMaxR];                 // routing weights
  float sel[kMaxR];               // selected sigmoid scores
  unsigned mask[kE];              // per expert: bit m set = token m picked it
  unsigned words[kE / 32];        // presence bitmap of the experts
  int D;                          // distinct experts
  unsigned char dmap[kE];         // expert -> distinct index (ascending expert id)
  unsigned char dexp[kMaxR];      // expert of distinct index d
  unsigned char nrows[kMaxR];     // routes of distinct index d (<= M)
  unsigned char rows[kMaxR * 8];  // their route indices, token ascending
};

// kDense: experts are resident fp16 `w13 [E, 2I, H]` ([gate; up] rows) and
// `w2 [E, H, I]` (no scales) instead of the NVFP4 TurboMind banks.
template <bool kDense>
__global__ void __launch_bounds__(kThreads) moe_fused_kernel(
    const half* __restrict__ x, const float* __restrict__ gate_w,
    const float* __restrict__ gate_b,
    const void* __restrict__ w13v, const half* __restrict__ sc13,
    const void* __restrict__ w2v, const half* __restrict__ sc2,
    float* logits, half* act, unsigned* counters,
    float* __restrict__ w_out, int* __restrict__ ids_out, half* __restrict__ out,
    int M, int inter, int s2, float route_scale, float gate_cap, float up_cap, int phases,
    int staged, int route_only) {
  // Dynamic: the activations of all routes (W2's A operand), when they fit (`staged`).
  extern __shared__ __align__(16) unsigned char stage_raw[];
  __shared__ __align__(16) float part[kWarps * 256];  // [warp][8 rows][32 cols]
  __shared__ float routed[kMaxR * 32];        // fp16-rounded W2 output per route
  __shared__ float red[2][kRW][kMaxM];
  __shared__ Tables tb;
  const int c = blockIdx.x, G = gridDim.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  // ---------------- R: router GEMV, rows c and c + G ----------------
  {
    const int rr = warp / kRW;
    const int e = c + rr * G;
    if (rr < 2 && e < kE) {
      const int t = (warp - rr * kRW) * 32 + lane;   // 0..287, 8 columns each
      const float4* wp = reinterpret_cast<const float4*>(gate_w + (size_t)e * kH) + 2 * t;
      float4 w0 = __ldg(wp), w1 = __ldg(wp + 1);
      #pragma unroll
      for (int m = 0; m < kMaxM; m++) {
        if (m >= M) break;
        float acc = 0.f;
        if (m < M) {
          uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + (size_t)m * kH) + t);
          const __half2* xh = reinterpret_cast<const __half2*>(&xv);
          float2 a = __half22float2(xh[0]), b = __half22float2(xh[1]);
          float2 cc = __half22float2(xh[2]), d = __half22float2(xh[3]);
          acc = w0.x * a.x + w0.y * a.y + w0.z * b.x + w0.w * b.y +
                w1.x * cc.x + w1.y * cc.y + w1.z * d.x + w1.w * d.y;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) red[rr][warp - rr * kRW][m] = acc;
      }
    }
    __syncthreads();
    if (tid < 2 * kMaxM) {
      const int r2 = tid / kMaxM, m = tid - r2 * kMaxM;
      const int e2 = c + r2 * G;
      if (e2 < kE && m < M) {
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < kRW; w++) v += red[r2][w][m];
        logits[(size_t)m * kE + e2] = v;
      }
    }
  }
  grid_barrier(counters, G);
// Phase timing: `phases` = stop code. Compiled out unless the debug build is asked for.
#ifdef VLLM_MOE_FUSED_DEBUG
#define kStop(k) if (phases == (k)) return
#else
#define kStop(k)
#endif
  kStop(1);

  // ---------------- select + expert table (same in every CTA) ----------------
  // Top-8 by key = sigmoid(logit) + bias, ties to the lower expert id; the slot of
  // an expert is the rank of its key (what the router's iterative arg-max gives).
  // Token m owns warps 4m .. 4m + 3, one key per thread: every warp files its own 8
  // best keys (8 shuffle arg-max rounds, no sync), then one warp ranks the 32
  // candidates of the token. A 1024-thread CTA pays ~16 ns per instruction that all
  // of its warps run, so the work is kept on the few warps that need it.
  const int R = M * kTopK;
  const int m_ = warp >> 2, q_ = warp & 3, e_ = q_ * 32 + lane;
  float* cand_key = part;                    // [kMaxM][32]; `part` is free until W13
  float* cand_sc = part + kMaxM * 32;
  int* cand_id = reinterpret_cast<int*>(part + 2 * kMaxM * 32);
  if (tid < kE) tb.mask[tid] = 0u;
  if (m_ < M) {
    const float l = __ldcg(logits + (size_t)m_ * kE + e_);
    const float sc = 1.0f / (1.0f + expf(-l));
    float key = sc + gate_b[e_];
    // A row of a padded decode graph can hold NaN / inf. Comparisons must stay a total
    // order or a slot could be left unfiled (an expert id out of range): NaN and -inf
    // keys become the floor, a filed key becomes -inf below it. Finite rows are not
    // touched, and a garbage row only yields garbage for itself.
    if (!(key > kKeyFloor)) key = kKeyFloor;
    #pragma unroll
    for (int r = 0; r < kTopK; r++) {
      float mk = key; int me = lane;
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        float ok = __shfl_xor_sync(0xffffffffu, mk, o);
        int oe = __shfl_xor_sync(0xffffffffu, me, o);
        if (ok > mk || (ok == mk && oe < me)) { mk = ok; me = oe; }
      }
      if (me == lane) {   // this lane holds the warp's r-th best key
        const int slot = m_ * 32 + q_ * 8 + r;
        cand_key[slot] = key; cand_sc[slot] = sc; cand_id[slot] = e_;
        key = -INFINITY;
      }
    }
  }
  __syncthreads();
  kStop(2);
  if (m_ < M && q_ == 0) {   // lane = one of the token's 32 candidates
    const float k = cand_key[m_ * 32 + lane];
    const int id = cand_id[m_ * 32 + lane];
    int rank = 0;
    #pragma unroll 8
    for (int j = 0; j < 32; ++j) {
      const float kj = cand_key[m_ * 32 + j];
      const int ij = cand_id[m_ * 32 + j];
      rank += (kj > k || (kj == k && ij < id)) ? 1 : 0;
    }
    if (rank < kTopK) {
      tb.ids[m_ * kTopK + rank] = id;
      tb.sel[m_ * kTopK + rank] = cand_sc[m_ * 32 + lane];
      atomicOr(&tb.mask[id], 1u << m_);
    }
  }
  __syncthreads();
  kStop(3);
  if (tid < R) {
    const int m = tid >> 3;
    float sum = 0.f;
    #pragma unroll
    for (int k = 0; k < kTopK; k++) sum += tb.sel[m * kTopK + k];
    float den = sum + 1e-20f;
    tb.w[tid] = (tb.sel[tid] / den) * route_scale;
  }
  if (tid < kE) {
    const unsigned present = __ballot_sync(0xffffffffu, tb.mask[tid] != 0u);
    if (lane == 0) tb.words[warp] = present;
  }
  __syncthreads();
  kStop(4);
  if (tid < kE) {
    const unsigned mk = tb.mask[tid];
    if (mk != 0u) {
      int d = __popc(tb.words[warp] & ((1u << lane) - 1u));
      for (int w = 0; w < warp; ++w) d += __popc(tb.words[w]);
      tb.dmap[tid] = (unsigned char)d;
      tb.dexp[d] = (unsigned char)tid;
      tb.nrows[d] = (unsigned char)__popc(mk);
    }
  } else if (tid == kThreads - 1) {
    tb.D = __popc(tb.words[0]) + __popc(tb.words[1]) + __popc(tb.words[2]) + __popc(tb.words[3]);
  }
  __syncthreads();
  kStop(5);
  if (tid < R) {
    const int e = tb.ids[tid], m = tid >> 3;
    const int pos = __popc(tb.mask[e] & ((1u << m) - 1u));
    tb.rows[tb.dmap[e] * 8 + pos] = (unsigned char)tid;
    if (c == 0) { ids_out[tid] = e; w_out[tid] = tb.w[tid]; }
  }
  __syncthreads();
  const int D = tb.D;
  kStop(6);
  // Route-only launch (the MXFP8 layers' visit): ids / weights are written, nothing else.
  if (route_only) return;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col = ((lane >> 2) & 3) * 8 + a_row;

  // ---------------- W13 + softcapped SwiGLU ----------------
  {
    const int tiles13 = inter >> 5;
    const int n13 = inter * 2;
    const size_t words13 = static_cast<size_t>(kH) * n13 / 8;
    const size_t scales13 = static_cast<size_t>(kH >> 4) * n13;
    const int proj = warp / kS13, split = warp - proj * kS13;
    const int njobs = D * tiles13;
    for (int job = c; job < njobs; job += G) {
      const int d = job / tiles13, tp = job - d * tiles13;
      const int expert = tb.dexp[d], nrows = tb.nrows[d];
#ifdef VLLM_MOE_PF13
      {
        // The weights of this CTA's next job (nothing else needs them first): pull them toward L2
        // while this job computes. Per job: 2 projections x 288 lines of codes, 2 x 144 scale pieces.
        const int nj = job + G;
        if (!kDense && nj < njobs && tid < 864) {
          const int nd = nj / tiles13, ntp = nj - nd * tiles13;
          const int nexp = tb.dexp[nd];
          if (tid < 576) {
            const int pj = tid / 288, ln = tid - pj * 288;
            const uint32_t* wb = static_cast<const uint32_t*>(w13v) + nexp * words13 +
                (static_cast<size_t>(ntp + pj * tiles13) * (kH >> 3)) * 32;
            prefetch_l2(reinterpret_cast<const char*>(wb) + ln * 128);
          } else {
            const int sidx = tid - 576, pj = sidx / 144, g = sidx - pj * 144;
            prefetch_l2(sc13 + nexp * scales13 +
                        (static_cast<size_t>(g) * (n13 >> 5) + ntp + pj * tiles13) * 32);
          }
        }
      }
#endif
      const half* rp = nullptr;
      if (a_row < nrows) rp = x + static_cast<size_t>(tb.rows[d * 8 + a_row] >> 3) * kH;
      float accum[8];
      if (kDense) {
        const half* wrow = static_cast<const half*>(w13v) +
            (static_cast<size_t>(expert) * n13 + (tp + proj * tiles13) * 32 + packed_col) * kH;
        gemv_rows_dense<false, kChunk13>(accum, rp, wrow, split * kNG13, kNG13);
      } else {
        gemv_rows<false, kChunk13>(accum, rp, static_cast<const uint32_t*>(w13v) + expert * words13,
                         sc13 + expert * scales13, tp + proj * tiles13, n13 >> 5, kH >> 3,
                         split * kNG13, kNG13, lane);
      }
      store_rows(part + warp * 256, accum, lane);
      __syncthreads();
      kStop(7);
      if (tid < nrows * 32) {
        const int row = tid >> 5, col = tid & 31;
        float g = 0.0f, u = 0.0f;
#pragma unroll
        for (int s = 0; s < kS13; ++s) {
          g += part[s * 256 + row * 32 + col];
          u += part[(kS13 + s) * 256 + row * 32 + col];
        }
        // fp16 W13 output boundary, then the softcapped SwiGLU in fp32.
        const float gh = __half2float(__float2half(g));
        const float uh = __half2float(__float2half(u));
        float gf = gh;
        if (gate_cap > 0.0f) gf = gate_cap * tanhf(gh / gate_cap);
        float uf = uh;
        if (up_cap > 0.0f) uf = up_cap * tanhf(uh / up_cap);
        const float sig = 1.0f / (1.0f + expf(-gh));
        act[static_cast<size_t>(tb.rows[d * 8 + row]) * inter + tp * 32 + col] =
            __float2half(gf * sig * uf);
      }
      __syncthreads();
    }
  }
  kStop(8);
  grid_barrier(counters + 1, G);
  kStop(9);

  // ---------------- W2 + weighted reduce: output tile c ----------------
  {
    const int k2 = inter, g2 = inter >> 4;
    const int ng2 = g2 / s2, per_iter = kWarps / s2;
    const size_t words2 = static_cast<size_t>(k2) * kH / 8;
    const size_t scales2 = static_cast<size_t>(g2) * kH;
    const int di = warp / s2, split = warp - di * s2;
    const int tile = c;
    if (tile >= kTiles2) return;   // the whole CTA: nothing after W2
    const half* a_base = act;
    if (staged) {
      // One round trip for every activation this CTA will use; the GEMV then reads
      // its A operand from shared memory instead of ld.cg per K group.
      uint4* stage = reinterpret_cast<uint4*>(stage_raw);
      const int n16 = R * inter / 8;
      for (int i = tid; i < n16; i += kThreads)
        stage[i] = __ldcg(reinterpret_cast<const uint4*>(act) + i);
      __syncthreads();
      a_base = reinterpret_cast<const half*>(stage_raw);
    }
    kStop(10);
    for (int d0 = 0; d0 < D; d0 += per_iter) {
      const int d = d0 + di;
      if (di < per_iter && d < D) {
        const int expert = tb.dexp[d], nrows = tb.nrows[d];
        const half* rp = nullptr;
        if (a_row < nrows) rp = a_base + static_cast<size_t>(tb.rows[d * 8 + a_row]) * inter;
        float accum[8];
        if (kDense) {
          const half* wrow = static_cast<const half*>(w2v) +
              (static_cast<size_t>(expert) * kH + tile * 32 + packed_col) * k2;
          if (staged) gemv_rows_dense<false, kChunk2>(accum, rp, wrow, split * ng2, ng2);
          else gemv_rows_dense<true, kChunk2>(accum, rp, wrow, split * ng2, ng2);
        } else {
          const uint32_t* ew = static_cast<const uint32_t*>(w2v) + expert * words2;
          if (staged)
            gemv_rows<false, kChunk2>(accum, rp, ew, sc2 + expert * scales2, tile, kH >> 5, k2 >> 3,
                             split * ng2, ng2, lane);
          else
            gemv_rows<true, kChunk2>(accum, rp, ew, sc2 + expert * scales2, tile, kH >> 5, k2 >> 3,
                            split * ng2, ng2, lane);
        }
        store_rows(part + warp * 256, accum, lane);
      }
      __syncthreads();
      for (int e2 = tid; e2 < per_iter * 256; e2 += kThreads) {
        const int dj = e2 >> 8, row = (e2 >> 5) & 7, col = e2 & 31;
        const int dd = d0 + dj;
        if (dd < D && row < tb.nrows[dd]) {
          float v = 0.0f;
          for (int p = 0; p < s2; ++p) v += part[(dj * s2 + p) * 256 + row * 32 + col];
          // NVFP4: per-route fp16 rounding before the weighted fp32 sum (as the
          // per-route kernels). Dense: the route output stays fp32 (as `moe_gemv`).
          routed[tb.rows[dd * 8 + row] * 32 + col] = kDense ? v : __half2float(__float2half(v));
        }
      }
      __syncthreads();
    }
    kStop(11);
    if (tid < M * 32) {
      const int token = tid >> 5, col = tid & 31;
      float weighted = 0.0f;
      for (int s = 0; s < kTopK; ++s)
        weighted = fmaf(tb.w[token * kTopK + s], routed[(token * kTopK + s) * 32 + col], weighted);
      out[static_cast<size_t>(token) * kH + tile * 32 + col] = __float2half(weighted);
    }
  }
}

}  // namespace

void moe_fused(torch::Tensor x, torch::Tensor gate_w, torch::Tensor gate_b,
               torch::Tensor w13, torch::Tensor sc13, torch::Tensor w2, torch::Tensor sc2,
               torch::Tensor logits, torch::Tensor act, torch::Tensor counters,
               torch::Tensor w_out, torch::Tensor ids_out, torch::Tensor out,
               int64_t s2, double route_scale, double gate_cap, double up_cap, int64_t phases) {
  const int64_t M = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() &&
              x.dim() == 2 && x.size(1) == kH && M >= 1 && M <= kMaxM, "moe_fused: x");
  TORCH_CHECK(gate_w.scalar_type() == torch::kFloat32 && gate_w.is_contiguous() &&
              gate_w.dim() == 2 && gate_w.size(0) == kE && gate_w.size(1) == kH &&
              gate_b.scalar_type() == torch::kFloat32 && gate_b.is_contiguous() &&
              gate_b.numel() == kE, "moe_fused: gate");
  TORCH_CHECK(w13.scalar_type() == torch::kInt32 && w2.scalar_type() == torch::kInt32 &&
              sc13.scalar_type() == torch::kFloat16 && sc2.scalar_type() == torch::kFloat16 &&
              w13.is_contiguous() && w2.is_contiguous() && sc13.is_contiguous() &&
              sc2.is_contiguous() && w13.dim() == 3 && w2.dim() == 3 && sc13.dim() == 3 &&
              sc2.dim() == 3, "moe_fused: banks");
  const int64_t inter = w2.size(1);
  TORCH_CHECK(inter % 32 == 0 && w13.size(0) == kE && w13.size(1) == kH &&
              w13.size(2) * 8 == 2 * inter && sc13.size(0) == kE && sc13.size(1) == kH / 16 &&
              sc13.size(2) == 2 * inter && w2.size(0) == kE && w2.size(2) * 8 == kH &&
              sc2.size(0) == kE && sc2.size(1) == inter / 16 && sc2.size(2) == kH,
              "moe_fused: bank shapes");
  TORCH_CHECK(s2 >= 1 && s2 <= kWarps && (inter / 16) % s2 == 0 &&
              ((inter / 16) / s2) % kChunk2 == 0 && kNG13 % kChunk13 == 0, "moe_fused: s2");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && logits.is_contiguous() &&
              logits.numel() >= kMaxM * kE && act.scalar_type() == torch::kFloat16 &&
              act.is_contiguous() && act.numel() >= kMaxR * inter &&
              counters.scalar_type() == torch::kInt32 && counters.numel() >= 2,
              "moe_fused: scratch");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 && out.is_contiguous() && out.dim() == 2 &&
              out.size(0) == M && out.size(1) == kH && w_out.scalar_type() == torch::kFloat32 &&
              w_out.is_contiguous() && w_out.numel() == M * kTopK &&
              ids_out.scalar_type() == torch::kInt32 && ids_out.is_contiguous() &&
              ids_out.numel() == M * kTopK, "moe_fused: outputs");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  // Activations of the M * 8 routes staged in shared memory when they fit beside the
  // static 43 KB (V100: 96 KB per CTA with opt-in; 80 CTAs = one per SM, co-resident).
  size_t stage = static_cast<size_t>(M) * kTopK * inter * sizeof(half);
  if (stage > kMaxStage) stage = 0;
  cudaFuncSetAttribute(moe_fused_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)kMaxStage);
  moe_fused_kernel<false><<<kGrid, kThreads, stage, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()), gate_w.data_ptr<float>(),
      gate_b.data_ptr<float>(),
      static_cast<const void*>(w13.data_ptr<int32_t>()),
      reinterpret_cast<const half*>(sc13.data_ptr<at::Half>()),
      static_cast<const void*>(w2.data_ptr<int32_t>()),
      reinterpret_cast<const half*>(sc2.data_ptr<at::Half>()),
      logits.data_ptr<float>(), reinterpret_cast<half*>(act.data_ptr<at::Half>()),
      reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>()),
      w_out.data_ptr<float>(), ids_out.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), (int)M, (int)inter, (int)s2,
      (float)route_scale, (float)gate_cap, (float)up_cap, (int)phases, stage ? 1 : 0, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_fused_dense(torch::Tensor x, torch::Tensor gate_w, torch::Tensor gate_b,
                     torch::Tensor w13, torch::Tensor w2,
                     torch::Tensor logits, torch::Tensor act, torch::Tensor counters,
                     torch::Tensor w_out, torch::Tensor ids_out, torch::Tensor out,
                     int64_t s2, double route_scale, double gate_cap, double up_cap, int64_t phases) {
  const int64_t M = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() &&
              x.dim() == 2 && x.size(1) == kH && M >= 1 && M <= kMaxM, "moe_fused_dense: x");
  TORCH_CHECK(gate_w.scalar_type() == torch::kFloat32 && gate_w.is_contiguous() &&
              gate_w.dim() == 2 && gate_w.size(0) == kE && gate_w.size(1) == kH &&
              gate_b.scalar_type() == torch::kFloat32 && gate_b.is_contiguous() &&
              gate_b.numel() == kE, "moe_fused_dense: gate");
  TORCH_CHECK(w13.scalar_type() == torch::kFloat16 && w2.scalar_type() == torch::kFloat16 &&
              w13.is_contiguous() && w2.is_contiguous() && w13.dim() == 3 && w2.dim() == 3,
              "moe_fused_dense: experts");
  const int64_t inter = w2.size(2);
  TORCH_CHECK(inter % 32 == 0 && w13.size(0) == kE && w13.size(1) == 2 * inter &&
              w13.size(2) == kH && w2.size(0) == kE && w2.size(1) == kH,
              "moe_fused_dense: expert shapes");
  TORCH_CHECK(s2 >= 1 && s2 <= kWarps && (inter / 16) % s2 == 0 &&
              ((inter / 16) / s2) % kChunk2 == 0 && kNG13 % kChunk13 == 0, "moe_fused_dense: s2");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && logits.is_contiguous() &&
              logits.numel() >= kMaxM * kE && act.scalar_type() == torch::kFloat16 &&
              act.is_contiguous() && act.numel() >= kMaxR * inter &&
              counters.scalar_type() == torch::kInt32 && counters.numel() >= 2,
              "moe_fused_dense: scratch");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 && out.is_contiguous() && out.dim() == 2 &&
              out.size(0) == M && out.size(1) == kH && w_out.scalar_type() == torch::kFloat32 &&
              w_out.is_contiguous() && w_out.numel() == M * kTopK &&
              ids_out.scalar_type() == torch::kInt32 && ids_out.is_contiguous() &&
              ids_out.numel() == M * kTopK, "moe_fused_dense: outputs");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  // Activations of the M * 8 routes staged in shared memory when they fit beside the
  // static 43 KB (V100: 96 KB per CTA with opt-in; 80 CTAs = one per SM, co-resident).
  size_t stage = static_cast<size_t>(M) * kTopK * inter * sizeof(half);
  if (stage > kMaxStage) stage = 0;
  cudaFuncSetAttribute(moe_fused_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)kMaxStage);
  moe_fused_kernel<true><<<kGrid, kThreads, stage, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()), gate_w.data_ptr<float>(),
      gate_b.data_ptr<float>(),
      static_cast<const void*>(w13.data_ptr<at::Half>()), nullptr,
      static_cast<const void*>(w2.data_ptr<at::Half>()), nullptr,
      logits.data_ptr<float>(), reinterpret_cast<half*>(act.data_ptr<at::Half>()),
      reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>()),
      w_out.data_ptr<float>(), ids_out.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), (int)M, (int)inter, (int)s2,
      (float)route_scale, (float)gate_cap, (float)up_cap, (int)phases, stage ? 1 : 0, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Router + select only (the MXFP8 layers' visit): `w_out` [M, 8] fp32 / `ids_out` [M, 8] int32 are
// bit-identical to `router_fused`'s, in about two thirds of its time (select on 4 warps per token).
void moe_route(torch::Tensor x, torch::Tensor gate_w, torch::Tensor gate_b, torch::Tensor logits,
               torch::Tensor counters, torch::Tensor w_out, torch::Tensor ids_out, double route_scale) {
  const int64_t M = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() &&
              x.dim() == 2 && x.size(1) == kH && M >= 1 && M <= kMaxM, "moe_route: x");
  TORCH_CHECK(gate_w.scalar_type() == torch::kFloat32 && gate_w.is_contiguous() &&
              gate_w.dim() == 2 && gate_w.size(0) == kE && gate_w.size(1) == kH &&
              gate_b.scalar_type() == torch::kFloat32 && gate_b.is_contiguous() &&
              gate_b.numel() == kE, "moe_route: gate");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && logits.is_contiguous() &&
              logits.numel() >= kMaxM * kE && counters.scalar_type() == torch::kInt32 &&
              counters.numel() >= 2, "moe_route: scratch");
  TORCH_CHECK(w_out.scalar_type() == torch::kFloat32 && w_out.is_contiguous() &&
              w_out.numel() == M * kTopK && ids_out.scalar_type() == torch::kInt32 &&
              ids_out.is_contiguous() && ids_out.numel() == M * kTopK, "moe_route: outputs");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  moe_fused_kernel<false><<<kGrid, kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()), gate_w.data_ptr<float>(),
      gate_b.data_ptr<float>(), nullptr, nullptr, nullptr, nullptr,
      logits.data_ptr<float>(), nullptr, reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>()),
      w_out.data_ptr<float>(), ids_out.data_ptr<int32_t>(), nullptr, (int)M, 0, 1,
      (float)route_scale, 0.f, 0.f, 0, 0, 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

ENABLED = os.environ.get("VLLM_INKLING2_SM70_MOE_FUSED", "1") == "1"
# The same kernel on the resident fp16 experts of the MXFP8 layers.
DENSE_ENABLED = os.environ.get("VLLM_INKLING2_SM70_MOE_FUSED_DENSE", "0") == "1"
MAX_TOKENS = 8
# Debug build only (VLLM_INKLING2_SM70_MOE_FUSED_DEBUG=1, own extension): `_PHASES` = stop code.
_DEBUG = os.environ.get("VLLM_INKLING2_SM70_MOE_FUSED_DEBUG", "0") == "1"
_PHASES = 0
# VLLM_INKLING2_SM70_MOE_FUSED_PF=1: each CTA prefetches its next W13 job's weights to L2 while it works on this one
# (same bits; -2 us per visit at 4-5 tokens, no change at 1). Own extension dir.
_PF = int(os.environ.get("VLLM_INKLING2_SM70_MOE_FUSED_PF", "1")) & 1
_CHUNK13 = int(os.environ.get("VLLM_INKLING2_SM70_MOE_FUSED_CHUNK13", "3"))
_CHUNK2 = int(os.environ.get("VLLM_INKLING2_SM70_MOE_FUSED_CHUNK2", "3"))
_E, _H, _TOPK = 128, 2304, 8

_ext = None


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline

        build = os.path.join(
            os.environ.get(
                "VLLM_INKLING2_SM70_EXT_DIR",
                os.path.expanduser("~/.cache/vllm/inkling2_sm70_ext"),
            ),
            f"moe_fused_c{_CHUNK13}_{_CHUNK2}"
            + ("_pf" if _PF else "")
            + ("_dbg" if _DEBUG else ""),
        )
        os.makedirs(build, exist_ok=True)
        verbose = bool(os.environ.get("VLLM_INKLING2_SM70_EXT_VERBOSE"))
        _ext = load_inline(
            name=f"inkling2_sm70_moe_fused_c{_CHUNK13}_{_CHUNK2}"
            + ("_pf" if _PF else "")
            + ("_dbg" if _DEBUG else ""),
            cpp_sources=[_CPP],
            cuda_sources=[_CUDA],
            functions=["moe_fused", "moe_fused_dense", "moe_route"],
            build_directory=build,
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_70,code=sm_70",
                f"-DVLLM_CHUNK13={_CHUNK13}",
                f"-DVLLM_CHUNK2={_CHUNK2}",
            ]
            + (["-DVLLM_MOE_FUSED_DEBUG"] if _DEBUG else [])
            + (["-DVLLM_MOE_PF13"] if _PF else [])
            + (["-Xptxas=-v"] if verbose else []),
            verbose=verbose,
        )
    return _ext


class FusedState:
    """Per-layer scratch of the fused visit: barrier counters, router logits,
    activations, and the routing of the last call (ids / weights)."""

    def __init__(self, device: torch.device, inter: int) -> None:
        self.inter = inter
        self.s2: int | None = None
        self.ok: bool | None = None  # set by `ready` on first use
        self.counters = torch.zeros(2, dtype=torch.int32, device=device)
        self.logits = torch.zeros((MAX_TOKENS, _E), dtype=torch.float32, device=device)
        self.act = torch.zeros(
            (MAX_TOKENS * _TOPK, inter), dtype=torch.float16, device=device
        )
        self.weights = torch.zeros(
            (MAX_TOKENS, _TOPK), dtype=torch.float32, device=device
        )
        self.ids = torch.zeros((MAX_TOKENS, _TOPK), dtype=torch.int32, device=device)


def make_state(
    gate_weight: torch.Tensor, top_k: int, inter: int, dense: bool = False
) -> FusedState | None:
    """Scratch for one MoE layer, or None when the path is off / the shape is
    not the one the kernel is written for. Loads the extension, so call it at
    load time, never inside CUDA-graph capture. `dense`: an MXFP8 layer
    (resident fp16 experts)."""
    if (
        not (DENSE_ENABLED if dense else ENABLED)
        or top_k != _TOPK
        or gate_weight.dtype != torch.float32
        or tuple(gate_weight.shape) != (_E, _H)
        or not gate_weight.is_cuda
        or torch.cuda.get_device_capability(gate_weight.device) != (7, 0)
        # The kernel is 80 CTAs that must all be resident at once: one per SM.
        or torch.cuda.get_device_properties(gate_weight.device).multi_processor_count
        < 80
        or inter % 32
    ):
        return None
    _load()
    return FusedState(gate_weight.device, inter)


def ready(state: FusedState, experts) -> bool:
    """True when `experts` (the NVFP4 `FusedMoE` layer) runs the shape-generic
    decode GEMV route (`Fp4DecodeGemv`: softcapped SwiGLU, fp16 prepared
    scales) on banks of the expected shape. Checked once per layer."""
    if state.ok is None:
        fast = getattr(experts, "sm70_fp4_decode_gemv", None)
        ok = fast is not None
        if ok:
            inter = state.inter
            s13, s2 = fast.splits(1)  # the K-splits production uses at M = 1
            w13, w2 = experts.w13_tm_weight, experts.w2_tm_weight
            sc13, sc2 = experts.w13_tm_scales, experts.w2_tm_scales
            ok = (
                s13 == 16
                and s2 is not None
                and (inter // 16) % s2 == 0
                and (inter // 16 // s2) % _CHUNK2 == 0
                and tuple(w13.shape) == (_E, _H, 2 * inter // 8)
                and tuple(w2.shape) == (_E, inter, _H // 8)
                and tuple(sc13.shape) == (_E, _H // 16, 2 * inter)
                and tuple(sc2.shape) == (_E, inter // 16, _H)
                and w13.dtype == torch.int32
                and w2.dtype == torch.int32
                and sc13.dtype == torch.float16
                and sc2.dtype == torch.float16
                and w13.is_contiguous()
                and w2.is_contiguous()
                and sc13.is_contiguous()
                and sc2.is_contiguous()
            )
            if ok:
                state.s2 = int(s2)
                logger.info_once(
                    "Inkling2 SM70 one-launch NVFP4 MoE visit in use (M<=%d, local_intermediate=%d, "
                    "K-splits 16/%d, chunks %d/%d).",
                    MAX_TOKENS,
                    inter,
                    int(s2),
                    _CHUNK13,
                    _CHUNK2,
                )
        state.ok = ok
    return state.ok


def moe_fused(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    route_scale: float,
    experts,
    gate_cap: float,
    up_cap: float,
    state: FusedState,
) -> torch.Tensor:
    """Router + NVFP4 experts + weighted reduce of `x` fp16 `[M <= 8, 2304]`
    (contiguous): this rank's partial MoE output, fp16 `[M, 2304]`. The caller
    checks `ready`. The routing of the call is left in `state.weights[:M]` /
    `state.ids[:M]`."""
    m = x.shape[0]
    out = torch.empty((m, _H), dtype=torch.float16, device=x.device)
    _load().moe_fused(
        x,
        gate_weight,
        gate_bias,
        experts.w13_tm_weight,
        experts.w13_tm_scales,
        experts.w2_tm_weight,
        experts.w2_tm_scales,
        state.logits,
        state.act,
        state.counters,
        state.weights[:m],
        state.ids[:m],
        out,
        state.s2,
        route_scale,
        gate_cap,
        up_cap,
        _PHASES,
    )
    return out


def dense_ready(
    state: FusedState, w13_fp16: torch.Tensor | None, w2_fp16: torch.Tensor | None
) -> bool:
    """Checks the resident fp16 experts `w13 [E, 2I, H]` ([gate; up] rows) and
    `w2 [E, H, I]` once per layer and picks the W2 K-split."""
    if state.ok is None:
        inter = state.inter
        groups = inter // 16
        s2 = 4 if groups % 4 == 0 else 3
        state.ok = (
            w13_fp16 is not None
            and w2_fp16 is not None
            and w13_fp16.dtype == torch.float16
            and w2_fp16.dtype == torch.float16
            and tuple(w13_fp16.shape) == (_E, 2 * inter, _H)
            and tuple(w2_fp16.shape) == (_E, _H, inter)
            and w13_fp16.is_contiguous()
            and w2_fp16.is_contiguous()
            and groups % s2 == 0
            and (groups // s2) % _CHUNK2 == 0
        )
        if state.ok:
            state.s2 = s2
    return state.ok


def moe_fused_dense(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    route_scale: float,
    w13_fp16: torch.Tensor,
    w2_fp16: torch.Tensor,
    gate_cap: float,
    up_cap: float,
    state: FusedState,
) -> torch.Tensor:
    """`moe_fused` for an MXFP8 layer (resident fp16 experts). Same routing bits;
    the expert sums are tensor-core accumulations (fp32) instead of the fma
    chains of `ops.moe.moe_gemv`, so outputs agree to fp16 rounding, not bitwise."""
    m = x.shape[0]
    out = torch.empty((m, _H), dtype=torch.float16, device=x.device)
    _load().moe_fused_dense(
        x,
        gate_weight,
        gate_bias,
        w13_fp16,
        w2_fp16,
        state.logits,
        state.act,
        state.counters,
        state.weights[:m],
        state.ids[:m],
        out,
        state.s2,
        route_scale,
        gate_cap,
        up_cap,
        _PHASES,
    )
    return out


# ---- route-only launch: the router + select of the MXFP8 layers' visit ----
# VLLM_INKLING2_SM70_MXFP8_FUSE: MXFP8 visit as route (this kernel) + W13/SwiGLU + W2 + weighted sum
# (4 launches instead of 7, same bits). Off = the base path.
MXFP8_FUSE = os.environ.get("VLLM_INKLING2_SM70_MXFP8_FUSE", "1") == "1"


class RouteState:
    """Scratch of the route-only launch of one layer."""

    def __init__(self, device: torch.device) -> None:
        self.counters = torch.zeros(2, dtype=torch.int32, device=device)
        self.logits = torch.zeros((MAX_TOKENS, _E), dtype=torch.float32, device=device)


def make_route_state(gate_weight: torch.Tensor, top_k: int) -> RouteState | None:
    """Scratch for the route-only launch, or None (flag off, other shape or GPU). Loads the
    extension: call at load time, never inside CUDA-graph capture."""
    if (
        not MXFP8_FUSE
        or top_k != _TOPK
        or gate_weight.dtype != torch.float32
        or tuple(gate_weight.shape) != (_E, _H)
        or not gate_weight.is_cuda
        or torch.cuda.get_device_capability(gate_weight.device) != (7, 0)
        or torch.cuda.get_device_properties(gate_weight.device).multi_processor_count
        < 80
    ):
        return None
    _load()
    return RouteState(gate_weight.device)


def route(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    route_scale: float,
    state: RouteState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(weights float32 [M, 8], ids int32 [M, 8])` of `x` fp16 `[M <= 8, 2304]` (contiguous),
    bit-identical to `router.fused_route`."""
    m = x.shape[0]
    w_out = torch.empty((m, _TOPK), dtype=torch.float32, device=x.device)
    ids = torch.empty((m, _TOPK), dtype=torch.int32, device=x.device)
    _load().moe_route(
        x, gate_weight, gate_bias, state.logits, state.counters, w_out, ids, route_scale
    )
    return w_out, ids
