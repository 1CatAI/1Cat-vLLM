# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Inkling2 decode / verify attention for SM70 (CUDA JIT extension).

One launch replaces the q/r half of `attention_prologue`, the two memsets,
`_rel_attention_kernel` (MODE_PARTIAL) and `_rel_decode_combine_kernel`:

* q and r are L2-clipped (``q * q_scale``, ``r * r_scale``, optional tau) in
  the prologue of every CTA; the relative bias of key `k` for the token at
  position `pos` is ``fp16(r . rel_proj[:, pos - k])`` (0 past the extent),
  computed on the fly, so the sheared `rel_logits` buffer is never written.
* The GQA group (3 q heads) of a KV head and up to 4 query tokens form the
  rows of a CTA (12 rows max, 8 query tokens use two CTAs); no padding to 64
  rows.  QK^T runs lane-per-key (fp32 FMA), P.V lane-per-dim.
* Keys are split over `num_splits` CTAs x 4 warps in an interleaved order that
  depends only on the batch size, so the launch is CUDA-graph safe.  The last
  CTA of each (request, KV head, token group) merges the splits (log-sum-exp)
  and applies the sigmoid output gate and the sink: no combine launch, no
  zero-initialised buffers (the arrival counters reset themselves).

KV cache layout is the model's `(num_blocks, block_size, H_kv, 2 * D)` with K
and V of a head side by side, as written by `attention_prologue`.
"""

import os

import torch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define HD 128
#define NWARPS 4
#define NTHREADS (NWARPS * 32)
#define LOG2E_F 1.4426950408889634f
#define LN2_F 0.6931471805599453f

struct Params {
  const __half* q;        // [T, Hq, HD] raw projection (row stride stride_q)
  const __half* r;        // [T, Hq, 16]
  const __half* q_scale;  // [Hq, HD]
  const __half* r_scale;  // [Hq, 16]
  const float* tau;       // [T] or null
  const __half* rel_proj; // [16, E]
  const __half* kv;       // [nb, bs, Hkv, 2*HD]
  const int* block_table; // [R, max_blocks]
  const int* seq_lens;    // [R]
  const int* cu_q;        // [R + 1]
  const __half* gate;     // [T, Hq*HD]
  const __half* sinks;    // [T, Hq]
  __half* out;            // [T, Hq*HD]
  float* o_part;          // [S, T, Hq, HD]
  float* lse_part;        // [S, T, Hq]  (log2)
  int* counters;          // [(R * Hkv * ntg)]
  int64_t stride_q, stride_r, stride_bt, stride_gate, stride_sink;
  int64_t stride_blk, stride_slot, stride_h;
  int T, Hq, Hkv, ntg, S, page_shift, window, E;
  float scale_log2, q_thr, r_thr, eps;
};

__device__ __forceinline__ float ex2f(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float lg2f(float x) {
  float y;
  asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rsqrtf_approx(float x) {
  float y;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rnd16(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + ex2f(-x * LOG2E_F)); }
__device__ __forceinline__ void unpack2(uint32_t u, float& a, float& b) {
  __half2 h = *reinterpret_cast<__half2*>(&u);
  float2 f = __half22float2(h);
  a = f.x;
  b = f.y;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float clip_factor(float ss, float thr, float eps) {
  float norm = ss > 0.f ? __fmul_rn(ss, rsqrtf_approx(ss)) : 0.f;
  return fminf(__fdiv_rn(thr, __fadd_rn(norm, eps)), 1.f);
}

// TQ query tokens x G heads per CTA.
template <int TQ, int G, bool HAS_TAU>
__global__ void __launch_bounds__(NTHREADS) decode_kernel(const Params p) {
  constexpr int ROWS = TQ * G;
  constexpr int NACC = TQ == 1 ? 4 : 2;   // independent QK accumulators per row
  constexpr int VPRE = TQ <= 2 ? 32 : 8;  // V rows of a tile loaded up front
  extern __shared__ __align__(16) float smem[];
  // phase A
  float* q_s = smem;                        // [ROWS][HD]
  float* r_s = q_s + ROWS * HD;             // [ROWS][16]
  float* p_s = r_s + ROWS * 16;             // [NWARPS][ROWS][32]
  // phase B (aliases phase A)
  float* macc = smem;                       // [NWARPS][ROWS][HD]
  float* mm = macc + NWARPS * ROWS * HD;    // [NWARPS][ROWS]
  float* ml = mm + NWARPS * ROWS;           // [NWARPS][ROWS]
  constexpr int SM_A = ROWS * HD + ROWS * 16 + NWARPS * ROWS * 32;
  constexpr int SM_B = NWARPS * ROWS * HD + 2 * NWARPS * ROWS;
  float* g_s = smem + (SM_A > SM_B ? SM_A : SM_B);  // [ROWS][HD] gate, then [ROWS] sinks
  float* sk_s = g_s + ROWS * HD;
  __shared__ int s_last;
  __shared__ float s_ms[ROWS], s_den[ROWS];

  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int split = blockIdx.x;
  const int kvh = blockIdx.y / p.ntg;
  const int tg = blockIdx.y % p.ntg;
  const int req = blockIdx.z;
  const int q_start = p.cu_q[req];
  const int nq = p.cu_q[req + 1] - q_start;
  const int seqlen = p.seq_lens[req];
  const int tok0 = tg * TQ;
  if (nq <= tok0 || seqlen <= 0) return;
  const int ntok = min(TQ, nq - tok0);
  const int rows = ntok * G;
  const int pos0 = seqlen - nq + tok0;  // position of the group's first token

  // ---- key range -----------------------------------------------------------
  const int hi = pos0 + ntok;                                  // exclusive
  const int lo = p.window > 0 ? max(pos0 - p.window + 1, 0) : 0;
  const int tile0 = lo >> 5;
  const int ntiles = ((hi + 31) >> 5) - tile0;
  // Only the splits that own at least one tile take part (graph-safe: this is
  // decided on the device from the sequence length).
  const int active = min(p.S, (ntiles + NWARPS - 1) / NWARPS);
  if (split >= active) return;
  const int page_mask = (1 << p.page_shift) - 1;
  const int nworkers = p.S * NWARPS;
  const int64_t bt_row = (int64_t)req * p.stride_bt;

  auto lookup = [&](int tt, int& base, int& page) {
    base = (tile0 + tt) << 5;
    const int key = base + lane;
    page = -1;
    if (tt < ntiles && key >= lo && key < hi) page = p.block_table[bt_row + (key >> p.page_shift)];
  };

  int t = split * NWARPS + w;
  int base, page;
  lookup(t, base, page);
  uint4 kk[16];
  uint2 vv[VPRE];
  auto issue = [&]() {
    if (page >= 0) {
      const uint4* kp = reinterpret_cast<const uint4*>(
          p.kv + (int64_t)page * p.stride_blk + (int64_t)((base + lane) & page_mask) * p.stride_slot + (int64_t)kvh * p.stride_h);
#pragma unroll
      for (int c = 0; c < 16; c++) kk[c] = __ldg(kp + c);
    }
#pragma unroll
    for (int j = 0; j < VPRE; j++) {
      const int pg = __shfl_sync(0xffffffffu, page, j);
      vv[j] = make_uint2(0u, 0u);
      if (pg >= 0)
        vv[j] = __ldg(reinterpret_cast<const uint2*>(p.kv + (int64_t)pg * p.stride_blk + (int64_t)((base + j) & page_mask) * p.stride_slot +
                                                       (int64_t)kvh * p.stride_h + HD + lane * 4));
    }
  };
  issue();  // tile loads overlap the q / r clipping below

  // ---- q / r: clip, round, tau ------------------------------------------
  for (int r = w; r < ROWS; r += NWARPS) {
    if (r < rows) {
      const int token = q_start + tok0 + r / G;
      const int head = kvh * G + r % G;
      float tau = 1.f;
      if (HAS_TAU) tau = p.tau[token];
      {
        // gate / sink for the epilogue, staged now so no global load sits behind the combine
        const uint32_t* gp = reinterpret_cast<const uint32_t*>(p.gate + (int64_t)token * p.stride_gate + head * HD + lane * 4);
        float g0, g1, g2, g3;
        unpack2(gp[0], g0, g1); unpack2(gp[1], g2, g3);
        *reinterpret_cast<float4*>(g_s + r * HD + lane * 4) = make_float4(g0, g1, g2, g3);
        if (lane == 0) sk_s[r] = __half2float(p.sinks[(int64_t)token * p.stride_sink + head]);
      }
      {
        // q is a slice of the qkvr projection: rows are only 4-byte aligned
        const uint32_t* qp = reinterpret_cast<const uint32_t*>(p.q + (int64_t)token * p.stride_q + head * HD + lane * 4);
        uint2 qv = make_uint2(qp[0], qp[1]);
        uint2 sv = *reinterpret_cast<const uint2*>(p.q_scale + head * HD + lane * 4);
        float x[4], sc[4];
        unpack2(qv.x, x[0], x[1]); unpack2(qv.y, x[2], x[3]);
        unpack2(sv.x, sc[0], sc[1]); unpack2(sv.y, sc[2], sc[3]);
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) { x[i] = __fmul_rn(x[i], sc[i]); ss = __fadd_rn(ss, __fmul_rn(x[i], x[i])); }
        ss = warp_sum(ss);
        const float f = clip_factor(ss, p.q_thr, p.eps);
        float4 y;
        float* yp = reinterpret_cast<float*>(&y);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          float v = rnd16(__fmul_rn(x[i], f));
          if (HAS_TAU) v = rnd16(__fmul_rn(v, tau));
          yp[i] = v;
        }
        *reinterpret_cast<float4*>(q_s + r * HD + lane * 4) = y;
      }
      {
        float x = 0.f;
        if (lane < 16) {
          x = __half2float(p.r[(int64_t)token * p.stride_r + head * 16 + lane]);
          x = __fmul_rn(x, __half2float(p.r_scale[head * 16 + lane]));
        }
        float ss = __fmul_rn(x, x);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
        const float f = clip_factor(ss, p.r_thr, p.eps);
        float v = rnd16(__fmul_rn(x, f));
        if (HAS_TAU) v = rnd16(__fmul_rn(v, tau));
        if (lane < 16) r_s[r * 16 + lane] = v;
      }
    } else {
      *reinterpret_cast<float4*>(q_s + r * HD + lane * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (lane < 16) r_s[r * 16 + lane] = 0.f;
    }
  }
  __syncthreads();

  float m[ROWS], l[ROWS], acc[ROWS][4];
#pragma unroll
  for (int r = 0; r < ROWS; r++) {
    m[r] = -INFINITY;
    l[r] = 0.f;
    acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = 0.f;
  }
  float* my_p = p_s + w * ROWS * 32;

  while (t < ntiles) {
    const int key = base + lane;
    const bool kvalid = page >= 0;
    const int cur_base = base;
    // next tile's page id, in flight during this tile's math
    int base_n, page_n;
    lookup(t + nworkers, base_n, page_n);

    float s[ROWS];
    {
      float sa[ROWS][NACC];
#pragma unroll
      for (int r = 0; r < ROWS; r++)
#pragma unroll
        for (int a = 0; a < NACC; a++) sa[r][a] = 0.f;
      if (kvalid) {
#pragma unroll
        for (int c = 0; c < 16; c++) {
          float kf[8];
          unpack2(kk[c].x, kf[0], kf[1]); unpack2(kk[c].y, kf[2], kf[3]);
          unpack2(kk[c].z, kf[4], kf[5]); unpack2(kk[c].w, kf[6], kf[7]);
#pragma unroll
          for (int r = 0; r < ROWS; r++) {
            const float4 a = *reinterpret_cast<const float4*>(q_s + r * HD + c * 8);
            const float4 b = *reinterpret_cast<const float4*>(q_s + r * HD + c * 8 + 4);
            float v = sa[r][c % NACC];
            v = fmaf(a.x, kf[0], v); v = fmaf(a.y, kf[1], v); v = fmaf(a.z, kf[2], v); v = fmaf(a.w, kf[3], v);
            v = fmaf(b.x, kf[4], v); v = fmaf(b.y, kf[5], v); v = fmaf(b.z, kf[6], v); v = fmaf(b.w, kf[7], v);
            sa[r][c % NACC] = v;
          }
        }
      }
#pragma unroll
      for (int r = 0; r < ROWS; r++) {
        float v = sa[r][0];
#pragma unroll
        for (int a = 1; a < NACC; a++) v += sa[r][a];
        s[r] = v;
      }
    }
    // bias + mask (log2 domain)
#pragma unroll
    for (int tk = 0; tk < TQ; tk++) {
      const int dist = pos0 + tk - key;
      const bool vis = kvalid && tk < ntok && dist >= 0 && (p.window == 0 || dist < p.window);
      const bool inb = vis && dist < p.E;
      float pv[16];
      if (inb) {
#pragma unroll
        for (int d = 0; d < 16; d++) pv[d] = __half2float(p.rel_proj[d * p.E + dist]);
      } else {
#pragma unroll
        for (int d = 0; d < 16; d++) pv[d] = 0.f;
      }
#pragma unroll
      for (int g = 0; g < G; g++) {
        const int r = tk * G + g;
        float b = 0.f;
#pragma unroll
        for (int d4 = 0; d4 < 4; d4++) {
          const float4 rv = *reinterpret_cast<const float4*>(r_s + r * 16 + d4 * 4);
          b = fmaf(rv.x, pv[d4 * 4 + 0], b); b = fmaf(rv.y, pv[d4 * 4 + 1], b);
          b = fmaf(rv.z, pv[d4 * 4 + 2], b); b = fmaf(rv.w, pv[d4 * 4 + 3], b);
        }
        b = inb ? rnd16(b) : 0.f;
        s[r] = vis ? fmaf(s[r], p.scale_log2, b * LOG2E_F) : -INFINITY;
      }
    }
    // online softmax over the tile
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
      const float mt = warp_max(s[r]);
      const float mn = fmaxf(m[r], mt);
      const float ms = mn == -INFINITY ? 0.f : mn;
      const float alpha = ex2f(m[r] - ms);
      const float pr = ex2f(s[r] - ms);
      l[r] = l[r] * alpha + pr;
      m[r] = mn;
      acc[r][0] *= alpha; acc[r][1] *= alpha; acc[r][2] *= alpha; acc[r][3] *= alpha;
      my_p[r * 32 + lane] = pr;
    }
    __syncwarp();
    // P.V, lane owns dims [4 * lane, 4 * lane + 4)
#pragma unroll
    for (int j = 0; j < 32; j++) {
      uint2 vj;
      if (j < VPRE) {
        vj = vv[j];
      } else {
        vj = make_uint2(0u, 0u);
        const int pg = __shfl_sync(0xffffffffu, page, j);
        if (pg >= 0)
          vj = __ldg(reinterpret_cast<const uint2*>(p.kv + (int64_t)pg * p.stride_blk + (int64_t)((cur_base + j) & page_mask) * p.stride_slot +
                                                     (int64_t)kvh * p.stride_h + HD + lane * 4));
      }
      float vf[4];
      unpack2(vj.x, vf[0], vf[1]); unpack2(vj.y, vf[2], vf[3]);
#pragma unroll
      for (int r = 0; r < ROWS; r++) {
        const float pj = my_p[r * 32 + j];
        acc[r][0] = fmaf(pj, vf[0], acc[r][0]); acc[r][1] = fmaf(pj, vf[1], acc[r][1]);
        acc[r][2] = fmaf(pj, vf[2], acc[r][2]); acc[r][3] = fmaf(pj, vf[3], acc[r][3]);
      }
    }
    __syncwarp();
    t += nworkers;
    base = base_n;
    page = page_n;
    if (t < ntiles) issue();
  }

  // ---- merge the 4 warps ---------------------------------------------------
  __syncthreads();
#pragma unroll
  for (int r = 0; r < ROWS; r++) {
    const float lt = warp_sum(l[r]);
    if (lane == 0) { mm[w * ROWS + r] = m[r]; ml[w * ROWS + r] = lt; }
    *reinterpret_cast<float4*>(macc + (w * ROWS + r) * HD + lane * 4) = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
  }
  __syncthreads();
  const int td = threadIdx.x;  // output dim
  // per row (rolled): merge the warps, write this CTA's partial
#pragma unroll 1
  for (int r = 0; r < rows; r++) {
    float M = mm[r];
#pragma unroll
    for (int ww = 1; ww < NWARPS; ww++) M = fmaxf(M, mm[ww * ROWS + r]);
    const float Ms = M == -INFINITY ? 0.f : M;
    float L = 0.f, O = 0.f;
#pragma unroll
    for (int ww = 0; ww < NWARPS; ww++) {
      const float wt = ex2f(mm[ww * ROWS + r] - Ms);
      L = fmaf(ml[ww * ROWS + r], wt, L);
      O = fmaf(macc[(ww * ROWS + r) * HD + td], wt, O);
    }
    const int token = q_start + tok0 + r / G;
    const int head = kvh * G + r % G;
    const int idx = (split * p.T + token) * p.Hq + head;
    p.o_part[idx * HD + td] = L > 0.f ? O / L : 0.f;
    if (td == 0) p.lse_part[idx] = L > 0.f ? Ms + lg2f(L) : -INFINITY;
  }
  __threadfence();
  __syncthreads();
  const int cidx = (req * p.Hkv + kvh) * p.ntg + tg;
  if (threadIdx.x == 0) {
    const int prev = atomicAdd(p.counters + cidx, 1);
    s_last = (prev == active - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (threadIdx.x == 0) p.counters[cidx] = 0;

  // lse of every (row, split): one round trip into smem; the per-row weights
  // come from smem while the o loads of all (split, row) pairs are in flight.
  float* lse_sm = smem;                  // [rows][active]   (macc is dead)
  float* wt_sm = smem + ROWS * 128;      // [rows][active]
  const int part_stride = p.T * p.Hq;
  for (int i = threadIdx.x; i < rows * active; i += NTHREADS) {
    const int r = i / active, sp = i - r * active;
    lse_sm[r * 128 + sp] = __ldcg(p.lse_part + sp * part_stride + (q_start + tok0 + r / G) * p.Hq + kvh * G + r % G);
  }
  __syncthreads();
  if (threadIdx.x < rows) {
    const int r = threadIdx.x;
    float M = -INFINITY;
    for (int sp = 0; sp < active; sp++) M = fmaxf(M, lse_sm[r * 128 + sp]);
    const float Ms = M == -INFINITY ? 0.f : M;
    float den = 0.f;
    for (int sp = 0; sp < active; sp++) {
      const float wt = ex2f(lse_sm[r * 128 + sp] - Ms);
      wt_sm[r * 128 + sp] = wt;
      den += wt;
    }
    s_ms[r] = Ms;
    s_den[r] = den;
  }
  __syncthreads();
  float O[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; r++) O[r] = 0.f;
  const float* ob = p.o_part + ((q_start + tok0) * p.Hq + kvh * G) * HD + td;
  const int sp_stride = part_stride * HD;
#pragma unroll 4
  for (int sp = 0; sp < active; sp++) {
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
      if (r < rows) O[r] = fmaf(__ldcg(ob + sp * sp_stride + ((r / G) * p.Hq + r % G) * HD), wt_sm[r * 128 + sp], O[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; r++) {
    if (r < rows) {
      const float den = s_den[r];
      const float o = den > 0.f ? O[r] / den : 0.f;
      const float lse2 = den > 0.f ? s_ms[r] + lg2f(den) : -INFINITY;
      float y = o * sigmoidf_(g_s[r * HD + td]);
      y = y * sigmoidf_(lse2 * LN2_F - sk_s[r]);
      const int token = q_start + tok0 + r / G;
      const int head = kvh * G + r % G;
      p.out[token * (p.Hq * HD) + head * HD + td] = __float2half_rn(y);
    }
  }
}

template <int TQ, int G, bool HAS_TAU>
static void launch(const Params& p, int num_reqs, cudaStream_t stream) {
  constexpr int ROWS = TQ * G;
  const size_t smem_a = (size_t)(ROWS * HD + ROWS * 16 + NWARPS * ROWS * 32) * sizeof(float);
  const size_t smem_b = (size_t)(NWARPS * ROWS * HD + 2 * NWARPS * ROWS) * sizeof(float);
  size_t smem = smem_a > smem_b ? smem_a : smem_b;
  smem += (size_t)(ROWS * HD + ROWS) * sizeof(float);  // gate / sink staging
  dim3 grid(p.S, p.Hkv * p.ntg, num_reqs);
  decode_kernel<TQ, G, HAS_TAU><<<grid, NTHREADS, smem, stream>>>(p);
}

template <int G, bool HAS_TAU>
static void dispatch_tq(const Params& p, int tq, int num_reqs, cudaStream_t stream) {
  if (tq == 1) launch<1, G, HAS_TAU>(p, num_reqs, stream);
  else if (tq == 2) launch<2, G, HAS_TAU>(p, num_reqs, stream);
  else launch<4, G, HAS_TAU>(p, num_reqs, stream);
}

void decode_attention_cuda(
    torch::Tensor q, torch::Tensor r, torch::Tensor q_scale, torch::Tensor r_scale,
    c10::optional<torch::Tensor> tau, torch::Tensor rel_proj, torch::Tensor kv,
    torch::Tensor block_table, torch::Tensor seq_lens, torch::Tensor cu_q,
    torch::Tensor gate, torch::Tensor sinks, torch::Tensor out,
    torch::Tensor o_part, torch::Tensor lse_part, torch::Tensor counters,
    int64_t num_reqs, int64_t max_query_len, int64_t num_splits, int64_t window,
    double softmax_scale, double q_thr, double r_thr, double eps) {
  Params p;
  p.q = reinterpret_cast<const __half*>(q.data_ptr());
  p.r = reinterpret_cast<const __half*>(r.data_ptr());
  p.q_scale = reinterpret_cast<const __half*>(q_scale.data_ptr());
  p.r_scale = reinterpret_cast<const __half*>(r_scale.data_ptr());
  p.tau = tau.has_value() ? tau->data_ptr<float>() : nullptr;
  p.rel_proj = reinterpret_cast<const __half*>(rel_proj.data_ptr());
  p.kv = reinterpret_cast<const __half*>(kv.data_ptr());
  p.block_table = block_table.data_ptr<int>();
  p.seq_lens = seq_lens.data_ptr<int>();
  p.cu_q = cu_q.data_ptr<int>();
  p.gate = reinterpret_cast<const __half*>(gate.data_ptr());
  p.sinks = reinterpret_cast<const __half*>(sinks.data_ptr());
  p.out = reinterpret_cast<__half*>(out.data_ptr());
  p.o_part = o_part.data_ptr<float>();
  p.lse_part = lse_part.data_ptr<float>();
  p.counters = counters.data_ptr<int>();
  p.stride_q = q.stride(0);
  p.stride_r = r.stride(0);
  p.stride_bt = block_table.stride(0);
  p.stride_gate = gate.stride(0);
  p.stride_sink = sinks.stride(0);
  p.stride_blk = kv.stride(0);
  p.stride_slot = kv.stride(1);
  p.stride_h = kv.stride(2);
  p.T = q.size(0);
  p.Hq = q.size(1);
  p.Hkv = kv.size(2);
  const int G = p.Hq / p.Hkv;
  const int tq = max_query_len <= 1 ? 1 : max_query_len <= 2 ? 2 : 4;
  p.ntg = (max_query_len + tq - 1) / tq;
  p.S = (int)num_splits;
  int ps = 0;
  while ((1 << ps) < kv.size(1)) ps++;
  p.page_shift = ps;
  p.window = (int)window;
  p.E = rel_proj.size(1);
  p.scale_log2 = (float)softmax_scale * LOG2E_F;
  p.q_thr = (float)q_thr;
  p.r_thr = (float)r_thr;
  p.eps = (float)eps;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(G == 3, "decode_attention supports a GQA group of 3 only");
  if (p.tau) dispatch_tq<3, true>(p, tq, (int)num_reqs, stream);
  else dispatch_tq<3, false>(p, tq, (int)num_reqs, stream);
}
"""

_CPP_SRC = r"""
void decode_attention_cuda(
    torch::Tensor q, torch::Tensor r, torch::Tensor q_scale, torch::Tensor r_scale,
    c10::optional<torch::Tensor> tau, torch::Tensor rel_proj, torch::Tensor kv,
    torch::Tensor block_table, torch::Tensor seq_lens, torch::Tensor cu_q,
    torch::Tensor gate, torch::Tensor sinks, torch::Tensor out,
    torch::Tensor o_part, torch::Tensor lse_part, torch::Tensor counters,
    int64_t num_reqs, int64_t max_query_len, int64_t num_splits, int64_t window,
    double softmax_scale, double q_thr, double r_thr, double eps);
"""

_EXT = None
_COUNTERS: dict[int, torch.Tensor] = {}
_MAX_COUNTERS = 16384
_LOCAL_SPLITS = int(os.environ.get("INK_ATTN_LOCAL_SPLITS", "16"))
_GLOBAL_SPLITS = int(os.environ.get("INK_ATTN_GLOBAL_SPLITS", "64"))
_GLOBAL_CTAS = int(os.environ.get("INK_ATTN_GLOBAL_CTAS", "256"))


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        build_dir = os.environ.get(
            "VLLM_INKLING2_SM70_EXT_DIR",
            os.path.expanduser("~/.cache/vllm/inkling2_sm70_ext"),
        )
        build_dir = os.path.join(build_dir, "decode_attention")
        os.makedirs(build_dir, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
        _EXT = load_inline(
            name="inkling2_sm70_decode_attention",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["decode_attention_cuda"],
            build_directory=build_dir,
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                "-gencode=arch=compute_70,code=sm_70",
            ],
            verbose=False,
        )
    return _EXT


def _counters(device: torch.device) -> torch.Tensor:
    idx = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    buf = _COUNTERS.get(idx)
    if buf is None:
        buf = torch.zeros(_MAX_COUNTERS, dtype=torch.int32, device=device)
        _COUNTERS[idx] = buf
    return buf


_DUMMY: dict[int, torch.Tensor] = {}


def _dummy(device: torch.device) -> torch.Tensor:
    idx = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    buf = _DUMMY.get(idx)
    if buf is None:
        buf = torch.zeros(4, dtype=torch.float32, device=device)
        _DUMMY[idx] = buf
    return buf


def prepare_decode_attention(device: torch.device) -> None:
    """Build the extension and allocate the arrival counters (eagerly, so
    CUDA graph capture never does)."""
    _ext()
    _counters(device)
    _dummy(device)


def supports_decode_attention(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    d_rel: int,
    dtype: torch.dtype,
    max_query_len: int,
) -> bool:
    return (
        num_heads // num_kv_heads == 3
        and head_dim == 128
        and d_rel == 16
        and dtype == torch.float16
        and max_query_len <= 8
    )


def plan_decode_splits(
    is_local: bool,
    num_reqs: int,
    num_kv_heads: int,
    max_query_len: int,
    rel_extent: int = 1024,
) -> int:
    """KV split count (CTAs per request x KV head x token group). Depends on
    the batch only, so it holds for every replay of a CUDA graph."""
    ntg = 1 if max_query_len <= 4 else 2
    work = num_reqs * num_kv_heads * ntg
    if is_local:
        return max(1, min(_LOCAL_SPLITS, 128 // work))
    return max(1, min(_GLOBAL_SPLITS, _GLOBAL_CTAS // work))


def decode_attention(
    q: torch.Tensor,
    r: torch.Tensor,
    q_scale: torch.Tensor,
    r_scale: torch.Tensor,
    tau: torch.Tensor | None,
    rel_proj: torch.Tensor,
    kv_cache: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
    qk_clip: float,
    rel_clip: float,
    eps: float = 1e-6,
) -> None:
    """Gated `(T, H * D)` attention output of a decode / verify batch.

    `q` `(T, H, D)` and `r` `(T, H, d_rel)` are the raw (unclipped) projections,
    `rel_proj` the `(d_rel, extent)` relative projection, `tau` the per-token
    log scaling (global layers) or None. Rows from `num_actual_tokens` on are
    zeroed. `window` is 0 for global layers.
    """
    num_tokens, num_heads, head_dim = q.shape
    assert q.stride(2) == 1 and q.stride(1) == head_dim
    assert r.stride(2) == 1 and r.stride(1) == r.shape[2]
    assert rel_proj.is_contiguous() and kv_cache.stride(3) == 1
    assert gate.stride(1) == 1 and sinks.stride(1) == 1 and out.is_contiguous()
    num_reqs = seq_lens.shape[0]
    ntg = 1 if max_query_len <= 4 else 2
    assert num_reqs * kv_cache.shape[2] * ntg <= _MAX_COUNTERS
    assert num_splits <= 128
    if num_actual_tokens < num_tokens:
        out[num_actual_tokens:].zero_()
    o_part = torch.empty(
        (num_splits, num_tokens, num_heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    lse_part = torch.empty(
        (num_splits, num_tokens, num_heads), dtype=torch.float32, device=q.device
    )
    _ext().decode_attention_cuda(
        q,
        r,
        q_scale,
        r_scale,
        tau,
        rel_proj,
        kv_cache,
        block_table,
        seq_lens,
        cu_seqlens_q,
        gate,
        sinks,
        out,
        o_part,
        lse_part,
        _counters(q.device),
        num_reqs,
        max_query_len,
        num_splits,
        window,
        softmax_scale,
        qk_clip,
        rel_clip,
        eps,
    )


def kv_conv_prologue(
    k: torch.Tensor,
    v: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_dynamic: torch.Tensor | None,
    v_dynamic: torch.Tensor | None,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    *,
    clip_threshold: float,
    norm_eps: float,
    clip_eps: float = 1e-6,
) -> None:
    """The K/V half of `attention_prologue` (short convs, K clip, V norm, KV
    cache write) for decode batches: q/r are handled by `decode_attention`."""
    from vllm.triton_utils import triton

    from .sconv import _attn_prologue_kernel

    num_tokens = kv_slot_mapping.shape[0]
    num_heads, head_dim = key_cache.shape[2:]
    dynamic = k_dynamic is not None
    if dynamic:
        assert v_dynamic is not None
        width = k_weight.shape[0]
        dyn_strides = (k_dynamic.stride(0), v_dynamic.stride(0))
    else:
        width = k_weight.shape[1]
        dyn_strides = (0, 0)
    block_t = 4
    grid = (triton.cdiv(num_tokens, block_t), num_heads, 2)
    unused = k
    _attn_prologue_kernel[grid](
        k,
        v,
        k_weight,
        v_weight,
        k_dynamic if dynamic else k_weight,
        v_dynamic if dynamic else v_weight,
        cache,
        key_cache,
        value_cache,
        kv_slot_mapping,
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        query_start,
        num_tokens,
        k.stride(0),
        v.stride(0),
        k_weight.stride(0),
        k_weight.stride(1),
        *dyn_strides,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.shape[1],
        block_table.stride(0),
        block_table.shape[1],
        cache.shape[2],
        (width * (2 if dynamic else 1) + 1) ** -0.5,
        clip_threshold,
        clip_eps,
        norm_eps,
        unused,  # q_ptr .. seq_lens_ptr: unused without FUSE_QR
        unused,
        unused,
        unused,
        None,
        unused,
        unused,
        unused,
        unused,
        unused,
        0,
        0,
        clip_threshold,
        W=width,
        D=head_dim,
        DYNAMIC=dynamic,
        BT=block_t,
        HQ=1,
        D_REL=16,
        E=1024,
        SHIFT_W=1536,
        HAS_TAU=False,
        BLOCK_HQ=16,
        BLOCK_E=256,
        QR_TOKENS=1,
        FUSE_QR=False,
        num_warps=4,
        enable_fp_fusion=False,
    )
