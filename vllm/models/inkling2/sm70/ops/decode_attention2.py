# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Inkling2 decode / verify attention for SM70, second generation (CUDA JIT).

One launch per layer visit replaces `kv_conv_prologue` (K/V short conv, K clip,
V norm, KV cache and conv-state writes) and `decode_attention` (q/r clip,
relative bias, attention, gate and sink) for batches whose requests bring at
most 8 new tokens (1 = plain decode, 2..8 = the speculative verify step).

Why a rewrite: in the decode graph a small kernel costs about
`2 us + 6 ns x (SASS instructions on its straight-line path)`: every SM fetches
the code cold. The first kernel is 5,300 (1 token) to 10,200 (5..8 tokens)
unrolled instructions = 35 / 85 us in the live graph. This kernel is written for
short code (runtime loops over rows and chunks, no template per query length):

* grid `(S, H_kv, R)`, 256 threads; a CTA works on one 32-key tile at a time.
* Tile staging: thread = (key, 16-dim chunk) issues 4 x 16 B loads (K and V) and
  stores them to shared memory as fp32, so all loads of a tile are in flight at
  once and each K/V tile is read once for all query tokens of the request.
* Rows (query token x 3 q heads, at most 24) run on warp `row & 7`: QK with
  lane = key, softmax, then P.V with lane = 4 dims, all from shared memory.
* The tile that holds the new positions has exactly one owner CTA. Its warps
  (warp = new token, lane = 4 dims: the Triton prologue's layout and reduction
  order, so the KV cache and conv state come out bit-identical) run the short
  conv, write the KV cache and the conv state and stage the fresh K/V for the
  attention: no thread reads the new slots from the cache.
* Merge: every CTA writes its `(m, l, acc)` partial, a grid barrier per
  (request, KV head) follows (self-resetting counter, bounded spin), then CTA
  `i` combines output row `i` and applies the sigmoid gate and the sink.

The barrier needs every CTA of the grid resident at once: `S * H_kv * R <= 80`
(one CTA per SM); `supports()` refuses anything else and the caller keeps the
first-generation path.
"""

import os

import torch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

#define HD 128
#define GQ 3
#define NWC 8             // warps of the combine (fixed: same fp32 grouping for every NW)
#define TK 32
#define SPIN_MAX (1 << 22)
#define LOG2E_F 1.4426950408889634f
#define LN2_F 0.6931471805599453f

struct Params {
  const __half* q;         // [T, Hq, HD] raw projection (row stride stride_q)
  const __half* r;         // [T, Hq, 16]
  const __half* q_scale;   // [Hq, HD]
  const __half* r_scale;   // [Hq, 16]
  const float* tau;        // [T] or null
  const __half* rel_t;     // [E, 16] relative projection, transposed
  __half* kv;              // [nb, page, Hkv, 2 * HD]
  const int* block_table;  // [R, max_blocks]
  const int* seq_lens;     // [R]
  const int* cu_q;         // [R + 1]
  const __half* gate;      // [T, Hq * HD]
  const __half* sinks;     // [T, Hq]
  __half* out;             // [T, Hq * HD]
  float* o_part;           // [S, T, Hq, HD]
  float* ml_part;          // [S, T, Hq, 2]  (running max (log2), sum)
  int* counters;           // [R * Hkv]
  int* fault;              // [1] sticky barrier-timeout flag
  const __half* x[2];      // K / V projections [T, Hkv * HD] (row strides stride_x)
  const __half* w[2];      // conv taps: static [Hkv * HD, 4]; dynamic [4, Hkv * HD]
  const __half* dyn[2];    // dynamic per-token tap deltas [T, Hkv * 4] or null
  __half* cache;           // conv state [cb, Hkv, N, CP]
  const int64_t* pos;      // [T]
  const int64_t* kv_slot;  // [T]
  const int64_t* cslot;    // [T]
  const int* seq_idx;      // [T]
  const int* qstart;       // [T]
  const int* cbt;          // [R, cmax]
  int64_t stride_q, stride_r, stride_bt, stride_gate, stride_sink;
  int64_t stride_blk, stride_slot, stride_h;
  int64_t stride_x[2], stride_dyn[2], stride_w0;
  int64_t stride_c_blk, stride_c_h, stride_c_n, stride_cbt;
  int T, Hq, Hkv, S, page_shift, window, E, maxrows, cshift, cmax, stop, pf;
  float scale_log2, q_thr, r_thr, eps, conv_scale, norm_eps;
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
__device__ __forceinline__ int ld_volatile(const int* p) {
  int v;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
// Warp shuffles as one inline SHFL each (what Triton emits). nvcc lowers __shfl_*_sync on
// sm_70 to a call into a ~10-instruction helper per shuffle; every lane of the warp must
// execute these together (they do at every call site: the control flow around them is
// uniform per warp).
__device__ __forceinline__ float shx(float v, int o) {
  float r;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(r) : "f"(v), "r"(o));
  return r;
}
__device__ __forceinline__ float shi(float v, int l) {
  float r;
  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(r) : "f"(v), "r"(l));
  return r;
}
__device__ __forceinline__ float rnd16(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + ex2f(-x * LOG2E_F)); }
__device__ __forceinline__ void up2(uint32_t u, float& a, float& b) {
  __half2 h = *reinterpret_cast<__half2*>(&u);
  float2 f = __half22float2(h);
  a = f.x;
  b = f.y;
}
__device__ __forceinline__ float4 up4(const uint2 v) {
  float4 f;
  up2(v.x, f.x, f.y);
  up2(v.y, f.z, f.w);
  return f;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, shx(v, o));
  return v;
}
// Triton's reduction order: a + shuffled, xor 16, 8, 4, 2, 1.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, shx(v, o));
  return v;
}
__device__ __forceinline__ float clip_factor(float ss, float thr, float eps) {
  float norm = ss > 0.f ? __fmul_rn(ss, rsqrtf_approx(ss)) : 0.f;
  return fminf(__fdiv_rn(thr, __fadd_rn(norm, eps)), 1.f);
}

// NW warps per CTA (8 | 16), MS = most splits (CTAs) per (request, KV head): 40, or 80 with two CTAs per SM.
template <int NW, int MS>
__global__ void __launch_bounds__(NW * 32, (NW == 8) ? 2 : 1) attn2_kernel(const Params p) {
  constexpr int CSP = (MS + NWC - 1) / NWC;   // partials merged per warp in the combine
  constexpr int MAXW = (MS + 31) / 32;        // warps that hold the partial maxima
  extern __shared__ __align__(16) float smem[];
  const int MR = p.maxrows;
  float* K_s = smem;             // [32 quads][TK keys][4]
  float* V_s = K_s + TK * HD;    // [TK keys][HD]
  float* q_s = V_s + TK * HD;    // [MR][HD]
  float* macc = q_s + MR * HD;   // [MR][HD]
  float* r_s = macc + MR * HD;   // [MR][16]
  float* p_s = r_s + MR * 16;    // [MR][TK]
  float* m_s = p_s + MR * TK;    // [MR]
  float* l_s = m_s + MR;         // [MR]
  float* cm_s = l_s + MR;        // [MS] combine: partial max
  float* cl_s = cm_s + MS;       // [MS] combine: partial sum

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  static_assert(NWC * CSP >= MS, "combine fan-in");
  const int split = blockIdx.x, kvh = blockIdx.y, req = blockIdx.z;
  const int S = p.S;
  const int q_start = p.cu_q[req];
  const int nq = p.cu_q[req + 1] - q_start;
  const int seqlen = p.seq_lens[req];
  int rows = 0, active = 0, pos0 = 0, lo = 0, hi = 0, tile0 = 0, ntiles = 0;
  if (nq > 0 && seqlen > 0) {
    rows = min(nq * GQ, MR);
    pos0 = seqlen - nq;  // position of the request's first new token
    hi = seqlen;
    lo = p.window > 0 ? max(pos0 - p.window + 1, 0) : 0;
    tile0 = lo >> 5;
    ntiles = ((hi + 31) >> 5) - tile0;
    active = min(S, ntiles);
  }

  if (p.stop == 1) return;
  if (split < active) {
    const int page_mask = (1 << p.page_shift) - 1;
    const int64_t cmask = ((int64_t)1 << p.cshift) - 1;
    // K/V of a tile's cached keys: thread = (key, 16-dim chunk), 4 x 16 B loads kept in
    // registers (in flight during the q/r clip, the new-key conv and the previous tile's rows).
    uint4 k0, k1, v0, v1;
    // Tiles are numbered from the newest (u = 0). CTA 0 takes only that one (it also runs
    // the K/V conv of the new tokens, the longest path); CTAs 1.. share the older tiles.
    const int ustep = S > 1 ? S - 1 : 1;
    const int ulimit = (split == 0 && S > 1) ? 1 : ntiles;
    auto issue = [&](int u) {
      const int key = ((tile0 + ntiles - 1 - u) << 5) + (tid >> 3);
      k0 = make_uint4(0u, 0u, 0u, 0u);
      k1 = k0; v0 = k0; v1 = k0;
      if (tid < 256 && key >= lo && key < pos0) {
        const int page = p.block_table[(int64_t)req * p.stride_bt + (key >> p.page_shift)];
        const uint4* kp = reinterpret_cast<const uint4*>(
            p.kv + (int64_t)page * p.stride_blk + (int64_t)(key & page_mask) * p.stride_slot + (int64_t)kvh * p.stride_h + (tid & 7) * 16);
        k0 = __ldg(kp);
        k1 = __ldg(kp + 1);
        v0 = __ldg(kp + 16);
        v1 = __ldg(kp + 17);
      }
    };
    issue(split);
    // ATTN2_PREFETCH: the page of a tile is found 3 tiles ahead and its K/V sectors go to L2 two tiles before the
    // registers load them (the loop below is bound by the dependent block-table -> K/V load chain, not by the rows).
    auto page_of = [&](int u) -> int {
      if (u >= ulimit) return -1;
      const int key = ((tile0 + ntiles - 1 - u) << 5) + (tid >> 3);
      if (tid < 256 && key >= lo && key < pos0) return p.block_table[(int64_t)req * p.stride_bt + (key >> p.page_shift)];
      return -1;
    };
    auto issue_pg = [&](int u, int page) {
      k0 = make_uint4(0u, 0u, 0u, 0u);
      k1 = k0; v0 = k0; v1 = k0;
      if (page >= 0) {
        const int key = ((tile0 + ntiles - 1 - u) << 5) + (tid >> 3);
        const uint4* kp = reinterpret_cast<const uint4*>(
            p.kv + (int64_t)page * p.stride_blk + (int64_t)(key & page_mask) * p.stride_slot + (int64_t)kvh * p.stride_h + (tid & 7) * 16);
        k0 = __ldg(kp);
        k1 = __ldg(kp + 1);
        v0 = __ldg(kp + 16);
        v1 = __ldg(kp + 17);
      }
    };
    auto prefetch_pg = [&](int u, int page) {
      if (page >= 0) {
        const int key = ((tile0 + ntiles - 1 - u) << 5) + (tid >> 3);
        const char* kp = reinterpret_cast<const char*>(
            p.kv + (int64_t)page * p.stride_blk + (int64_t)(key & page_mask) * p.stride_slot + (int64_t)kvh * p.stride_h + (tid & 7) * 16);
        asm volatile("prefetch.global.L2 [%0];" ::"l"(kp));
        asm volatile("prefetch.global.L2 [%0];" ::"l"(kp + 256 * 2 / 2 * 1));
      }
    };
    int pg1 = -1, pg2 = -1, pg3 = -1;
    if (p.pf) {
      pg1 = page_of(split + ustep);
      pg2 = page_of(split + 2 * ustep);
      pg3 = page_of(split + 3 * ustep);
    }
    // ---- q / r of this warp's rows: scale, L2 clip, round (tau), stage ----
#pragma unroll 1
    for (int r = w; r < rows; r += NW) {
      const int tk = r / GQ;
      const int token = q_start + tk;
      const int head = kvh * GQ + (r - tk * GQ);
      float tau = 1.f;
      if (p.tau) tau = p.tau[token];
      {
        // q is a slice of the qkvr projection: rows are only 4-byte aligned
        const uint32_t* qp = reinterpret_cast<const uint32_t*>(p.q + (int64_t)token * p.stride_q + head * HD + lane * 4);
        float4 x = up4(make_uint2(qp[0], qp[1]));
        const float4 sc = up4(*reinterpret_cast<const uint2*>(p.q_scale + head * HD + lane * 4));
        x.x = __fmul_rn(x.x, sc.x); x.y = __fmul_rn(x.y, sc.y); x.z = __fmul_rn(x.z, sc.z); x.w = __fmul_rn(x.w, sc.w);
        float ss = __fmul_rn(x.x, x.x);
        ss = __fadd_rn(ss, __fmul_rn(x.y, x.y));
        ss = __fadd_rn(ss, __fmul_rn(x.z, x.z));
        ss = __fadd_rn(ss, __fmul_rn(x.w, x.w));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += shx(ss, o);
        const float f = clip_factor(ss, p.q_thr, p.eps);
        x.x = rnd16(__fmul_rn(x.x, f)); x.y = rnd16(__fmul_rn(x.y, f));
        x.z = rnd16(__fmul_rn(x.z, f)); x.w = rnd16(__fmul_rn(x.w, f));
        if (p.tau) {
          x.x = rnd16(__fmul_rn(x.x, tau)); x.y = rnd16(__fmul_rn(x.y, tau));
          x.z = rnd16(__fmul_rn(x.z, tau)); x.w = rnd16(__fmul_rn(x.w, tau));
        }
        *reinterpret_cast<float4*>(q_s + r * HD + lane * 4) = x;
        *reinterpret_cast<float4*>(macc + r * HD + lane * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      {
        float x = 0.f;
        if (lane < 16) {
          x = __half2float(p.r[(int64_t)token * p.stride_r + head * 16 + lane]);
          x = __fmul_rn(x, __half2float(p.r_scale[head * 16 + lane]));
        }
        float ss = __fmul_rn(x, x);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) ss = __fadd_rn(ss, shx(ss, o));
        const float f = clip_factor(ss, p.r_thr, p.eps);
        float v = rnd16(__fmul_rn(x, f));
        if (p.tau) v = rnd16(__fmul_rn(v, tau));
        if (lane < 16) r_s[r * 16 + lane] = v;
      }
      if (lane == 0) {
        m_s[r] = -INFINITY;
        l_s[r] = 0.f;
      }
    }

    if (p.stop == 2) return;
    const int interior_hi = (p.window == 0 && rows >= 12 && pos0 > p.E) ? ((pos0 - p.E) & ~31) : 0;
    const int u_wmma_start = interior_hi > 0 ? ((tile0 + ntiles) - (interior_hi >> 5)) : ntiles;
    const int u_scalar_limit = min(ulimit, u_wmma_start);
    int u = split;
#pragma unroll 1
    for (; u < u_scalar_limit; u += ustep) {
      const int base = (tile0 + ntiles - 1 - u) << 5;
      // ---- new keys of the tile: K/V short conv, K clip, V norm, cache writes.
      //      Work item i = (new key, K | V) runs on warp 7 - (i & 7) (the q/r clip of the
      //      first rows runs on the low warps meanwhile), lane = 4 dims. ----
      {
        const int nfirst = max(base, pos0) - pos0;
        const int nnew = min(base + TK, hi) - pos0 - nfirst;
#pragma unroll 1
        for (int i = NW - 1 - w; i < 2 * nnew; i += NW) {
          {
            const int n = nfirst + (i >> 1);
            const int s = i & 1;
            const int tok = q_start + n;
            const int kl = pos0 + n - base;
            const int64_t cpos = p.pos[tok];
            const int64_t cs = p.cslot[tok];
            const int64_t ks = p.kv_slot[tok];
            const bool cvalid = cs >= 0;
            const int qs = p.qstart[tok];
            const int64_t crow = (int64_t)p.seq_idx[tok] * p.stride_cbt;
            const int col = kvh * HD + lane * 4;
            const __half* x = p.x[s];
            const int64_t sx = p.stride_x[s];
            const uint32_t* xp = reinterpret_cast<const uint32_t*>(x + (int64_t)tok * sx + col);
            const uint2 curv = make_uint2(xp[0], xp[1]);
            uint2 tapv[3];
#pragma unroll
            for (int lag = 1; lag <= 3; lag++) {
              const int64_t src = cpos - lag;
              const int row = tok - lag;
              uint2 v = make_uint2(0u, 0u);
              if (cvalid && src >= 0) {
                if (row >= qs) {
                  const uint32_t* tp = reinterpret_cast<const uint32_t*>(x + (int64_t)row * sx + col);
                  v = make_uint2(tp[0], tp[1]);
                } else {
                  const int64_t lblk = min(src >> p.cshift, (int64_t)(p.cmax - 1));
                  const int64_t blk = p.cbt[crow + lblk];
                  v = *reinterpret_cast<const uint2*>(
                      p.cache + blk * p.stride_c_blk + (src & cmask) * p.stride_c_n + (int64_t)kvh * p.stride_c_h + s * HD + lane * 4);
                }
              }
              tapv[lag - 1] = v;
            }
            const float4 cur = up4(curv);
            const float4 c0 = cvalid ? cur : make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 t1 = up4(tapv[0]), t2 = up4(tapv[1]), t3 = up4(tapv[2]);
            float4 y;
            if (!p.dyn[0]) {
              // static taps [C][4], oldest first: fma chain from +0
              const uint4 wv = *reinterpret_cast<const uint4*>(p.w[s] + (int64_t)col * 4);
              const float4 wa = up4(make_uint2(wv.x, wv.y)), wb = up4(make_uint2(wv.z, wv.w));
              const uint4 wv2 = *reinterpret_cast<const uint4*>(p.w[s] + (int64_t)col * 4 + 8);
              const float4 wc = up4(make_uint2(wv2.x, wv2.y)), wd = up4(make_uint2(wv2.z, wv2.w));
              y.x = __fmaf_rn(c0.x, wa.w, __fmaf_rn(t1.x, wa.z, __fmaf_rn(t2.x, wa.y, __fmaf_rn(t3.x, wa.x, 0.f))));
              y.y = __fmaf_rn(c0.y, wb.w, __fmaf_rn(t1.y, wb.z, __fmaf_rn(t2.y, wb.y, __fmaf_rn(t3.y, wb.x, 0.f))));
              y.z = __fmaf_rn(c0.z, wc.w, __fmaf_rn(t1.z, wc.z, __fmaf_rn(t2.z, wc.y, __fmaf_rn(t3.z, wc.x, 0.f))));
              y.w = __fmaf_rn(c0.w, wd.w, __fmaf_rn(t1.w, wd.z, __fmaf_rn(t2.w, wd.y, __fmaf_rn(t3.w, wd.x, 0.f))));
            } else {
              // dynamic taps: (static [4][C] + per-token per-head delta), newest first
              const float4 d = up4(*reinterpret_cast<const uint2*>(p.dyn[s] + (int64_t)tok * p.stride_dyn[s] + kvh * 4));
              const __half* ws = p.w[s] + col;
              const float4 w0 = up4(*reinterpret_cast<const uint2*>(ws));
              const float4 w1 = up4(*reinterpret_cast<const uint2*>(ws + p.stride_w0));
              const float4 w2 = up4(*reinterpret_cast<const uint2*>(ws + 2 * p.stride_w0));
              const float4 w3 = up4(*reinterpret_cast<const uint2*>(ws + 3 * p.stride_w0));
#define DYN1(c) __fmaf_rn(t3.c, __fadd_rn(w3.c, d.w), __fmaf_rn(t2.c, __fadd_rn(w2.c, d.z), \
                __fmaf_rn(t1.c, __fadd_rn(w1.c, d.y), __fmul_rn(c0.c, __fadd_rn(w0.c, d.x)))))
              y.x = DYN1(x); y.y = DYN1(y); y.z = DYN1(z); y.w = DYN1(w);
#undef DYN1
            }
            y.x = rnd16(__fmul_rn(__fadd_rn(y.x, cur.x), p.conv_scale));
            y.y = rnd16(__fmul_rn(__fadd_rn(y.y, cur.y), p.conv_scale));
            y.z = rnd16(__fmul_rn(__fadd_rn(y.z, cur.z), p.conv_scale));
            y.w = rnd16(__fmul_rn(__fadd_rn(y.w, cur.w), p.conv_scale));
            float ss;
            if (!p.dyn[0]) {
              // static kernel: thread = 4 consecutive dims, then lanes xor 16, 8, 4, 2, 1
              ss = __fadd_rn(__fmul_rn(y.x, y.x), __fmul_rn(y.y, y.y));
              ss = __fadd_rn(ss, __fmul_rn(y.z, y.z));
              ss = __fadd_rn(ss, __fmul_rn(y.w, y.w));
              ss = warp_sum(ss);
            } else {
              // dynamic kernel: its layout is 1 dim x 8 lanes x 4 warps, tiled over 128 dims:
              // thread (warp g, slot j) sums dims 8g + j + 32k (k = 0..3), then slots xor 4, 2, 1,
              // then warps (0 + 2) + (1 + 3). Here dim d sits on lane d / 4, component d % 4.
              const int l8 = lane & 7;
              float4 sq = make_float4(__fmul_rn(y.x, y.x), __fmul_rn(y.y, y.y), __fmul_rn(y.z, y.z), __fmul_rn(y.w, y.w));
              float4 pq;
#define TSUM(c) pq.c = __fadd_rn(__fadd_rn(__fadd_rn(shi(sq.c, l8), \
                shi(sq.c, l8 + 8)), shi(sq.c, l8 + 16)), \
                shi(sq.c, l8 + 24)); \
                pq.c = __fadd_rn(pq.c, shx(pq.c, 1));
              TSUM(x) TSUM(y) TSUM(z) TSUM(w)
#undef TSUM
              ss = __fadd_rn(__fadd_rn(pq.x, pq.z), __fadd_rn(pq.y, pq.w));
              ss = __fadd_rn(ss, shx(ss, 4));
              ss = __fadd_rn(ss, shx(ss, 2));
            }
            const float f = s == 0 ? clip_factor(ss, p.q_thr, p.eps)
                                   : rsqrtf_approx(__fadd_rn(__fdiv_rn(ss, (float)HD), p.norm_eps));
            uint2 zv;
            *reinterpret_cast<__half2*>(&zv.x) = __floats2half2_rn(__fmul_rn(y.x, f), __fmul_rn(y.y, f));
            *reinterpret_cast<__half2*>(&zv.y) = __floats2half2_rn(__fmul_rn(y.z, f), __fmul_rn(y.w, f));
            if (ks >= 0)
              *reinterpret_cast<uint2*>(p.kv + (ks >> p.page_shift) * p.stride_blk + (ks & page_mask) * p.stride_slot +
                                        (int64_t)kvh * p.stride_h + s * HD + lane * 4) = zv;
            if (cvalid)
              *reinterpret_cast<uint2*>(p.cache + (cs >> p.cshift) * p.stride_c_blk + (cs & cmask) * p.stride_c_n +
                                        (int64_t)kvh * p.stride_c_h + s * HD + lane * 4) = curv;
            const float4 zf = up4(zv);
            if (s == 0) *reinterpret_cast<float4*>(K_s + (lane * TK + kl) * 4) = zf;
            else *reinterpret_cast<float4*>(V_s + kl * HD + lane * 4) = zf;
          }
        }
      }
      // ---- cached keys of the tile: registers -> shared memory (fp32) ----
      {
        const int kl = tid >> 3, sub = tid & 7;
        const int key = base + kl;
        if (tid < 256 && (key < pos0 || key >= hi)) {  // new keys were staged by their conv above
          float4* kd = reinterpret_cast<float4*>(K_s + (sub * 4 * TK + kl) * 4);
          float4* vd = reinterpret_cast<float4*>(V_s + kl * HD + sub * 16);
          kd[0] = up4(make_uint2(k0.x, k0.y));
          kd[TK] = up4(make_uint2(k0.z, k0.w));
          kd[2 * TK] = up4(make_uint2(k1.x, k1.y));
          kd[3 * TK] = up4(make_uint2(k1.z, k1.w));
          vd[0] = up4(make_uint2(v0.x, v0.y));
          vd[1] = up4(make_uint2(v0.z, v0.w));
          vd[2] = up4(make_uint2(v1.x, v1.y));
          vd[3] = up4(make_uint2(v1.z, v1.w));
        }
      }
      __syncthreads();
      if (p.stop == 3) return;
      if (p.pf) {
        if (u + ustep < u_scalar_limit) issue_pg(u + ustep, pg1);
        prefetch_pg(u + 3 * ustep, pg3);
        pg1 = pg2; pg2 = pg3; pg3 = page_of(u + 4 * ustep);
      } else if (u + ustep < u_scalar_limit) issue(u + ustep);  // next tile in flight during this tile's rows
      if (p.stop == 9) { __syncthreads(); continue; }  // tuning aid: tile loads + staging only
      // ---- rows of this warp: QK (lane = key), bias, softmax, P.V (lane = 4 dims) ----
#pragma unroll 1
      for (int r = w; r < rows; r += NW) {
        const int tk = r / GQ;
        const int key = base + lane;
        const int dist = pos0 + tk - key;
        const bool vis = key >= lo && key < hi && dist >= 0 && (p.window == 0 || dist < p.window);
        const bool inb = vis && dist < p.E;
        uint4 b0 = make_uint4(0u, 0u, 0u, 0u), b1 = b0;
        if (inb) {
          const uint4* bp = reinterpret_cast<const uint4*>(p.rel_t + (int64_t)dist * 16);
          b0 = __ldg(bp);
          b1 = __ldg(bp + 1);
        }
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        const float* qr = q_s + r * HD;
        const float* kr = K_s + lane * 4;
#pragma unroll 2
        for (int qd = 0; qd < HD / 4; qd++) {
          const float4 kq = *reinterpret_cast<const float4*>(kr + qd * (TK * 4));
          const float4 qq = *reinterpret_cast<const float4*>(qr + qd * 4);
          a0 = fmaf(kq.x, qq.x, a0);
          a1 = fmaf(kq.y, qq.y, a1);
          a2 = fmaf(kq.z, qq.z, a2);
          a3 = fmaf(kq.w, qq.w, a3);
        }
        float sc = (a0 + a1) + (a2 + a3);
        {
          const float4 r0 = *reinterpret_cast<const float4*>(r_s + r * 16);
          const float4 r1 = *reinterpret_cast<const float4*>(r_s + r * 16 + 4);
          const float4 r2 = *reinterpret_cast<const float4*>(r_s + r * 16 + 8);
          const float4 r3 = *reinterpret_cast<const float4*>(r_s + r * 16 + 12);
          const float4 p0 = up4(make_uint2(b0.x, b0.y)), p1 = up4(make_uint2(b0.z, b0.w));
          const float4 p2 = up4(make_uint2(b1.x, b1.y)), p3 = up4(make_uint2(b1.z, b1.w));
          float b = 0.f;
          b = fmaf(r0.x, p0.x, b); b = fmaf(r0.y, p0.y, b); b = fmaf(r0.z, p0.z, b); b = fmaf(r0.w, p0.w, b);
          b = fmaf(r1.x, p1.x, b); b = fmaf(r1.y, p1.y, b); b = fmaf(r1.z, p1.z, b); b = fmaf(r1.w, p1.w, b);
          b = fmaf(r2.x, p2.x, b); b = fmaf(r2.y, p2.y, b); b = fmaf(r2.z, p2.z, b); b = fmaf(r2.w, p2.w, b);
          b = fmaf(r3.x, p3.x, b); b = fmaf(r3.y, p3.y, b); b = fmaf(r3.z, p3.z, b); b = fmaf(r3.w, p3.w, b);
          b = inb ? rnd16(b) : 0.f;
          sc = vis ? fmaf(sc, p.scale_log2, b * LOG2E_F) : -INFINITY;
        }
        const float mo = m_s[r];
        const float mn = fmaxf(mo, warp_max(sc));
        const float ms = mn == -INFINITY ? 0.f : mn;
        const float alpha = ex2f(mo - ms);
        const float pr = ex2f(sc - ms);
        float lt = pr;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) lt += shx(lt, o);
        p_s[r * TK + lane] = pr;
        if (lane == 0) {
          m_s[r] = mn;
          l_s[r] = l_s[r] * alpha + lt;
        }
        __syncwarp();
        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
        const float* vr = V_s + lane * 4;
        const float* pp = p_s + r * TK;
#pragma unroll 2
        for (int j = 0; j < TK; j++) {
          const float4 vq = *reinterpret_cast<const float4*>(vr + j * HD);
          const float pj = pp[j];
          c0 = fmaf(pj, vq.x, c0);
          c1 = fmaf(pj, vq.y, c1);
          c2 = fmaf(pj, vq.z, c2);
          c3 = fmaf(pj, vq.w, c3);
        }
        float4* mp = reinterpret_cast<float4*>(macc + r * HD + lane * 4);
        float4 mv = *mp;
        mv.x = fmaf(mv.x, alpha, c0);
        mv.y = fmaf(mv.y, alpha, c1);
        mv.z = fmaf(mv.z, alpha, c2);
        mv.w = fmaf(mv.w, alpha, c3);
        *mp = mv;
      }
      __syncthreads();
    }


    // ---- WMMA tensor-core interior loop for global layers (dist >= E, no bias/mask/conv) ----
    if (u < ulimit) {
      using namespace nvcuda;
      constexpr int W_BM = 32;
      constexpr int W_LDK = HD + 8;  // 136 halves
      constexpr int W_LDS = TK + 4;  // 36 floats
      constexpr int W_LDP = TK + 8;  // 40 halves
      // Reuse K_s + V_s (32,768 bytes) for all WMMA staging -> zero extra shared memory!
      __half* w_sQ = reinterpret_cast<__half*>(K_s);  // 32 * 136 * 2 = 8,704 B (temp before key loop)
      // 1) Convert q_s [rows, 128] (fp32) -> w_sQ [32, 136] (fp16)
      for (int idx = tid; idx < W_BM * (HD / 4); idx += NW * 32) {
        const int r = idx >> 5;
        const int c4 = idx & 31;
        __half2 h0 = __floats2half2_rn(0.f, 0.f), h1 = h0;
        if (r < rows) {
          const float4 qv = *reinterpret_cast<const float4*>(q_s + r * HD + c4 * 4);
          h0 = __floats2half2_rn(qv.x, qv.y);
          h1 = __floats2half2_rn(qv.z, qv.w);
        }
        __half2* dst2 = reinterpret_cast<__half2*>(w_sQ + r * W_LDK + c4 * 4);
        dst2[0] = h0;
        dst2[1] = h1;
      }
      __syncthreads();

      // Load Q fragments into registers for warps 0..3 (wm = w>>1, wn = w&1)
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag[8];
      if (w < 4) {
        const int q_wm = w >> 1;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          wmma::load_matrix_sync(q_frag[k], w_sQ + q_wm * 16 * W_LDK + k * 16, W_LDK);
        }
      }
      __syncthreads();

      // 2) Copy existing macc [rows, 128] -> sO_tmp [32, 128] (in K_s, 16 KB) and load into o_frag[0..1] for w < 8
      float* sO_tmp = K_s;
      for (int idx = tid; idx < W_BM * (HD / 4); idx += NW * 32) {
        const int r = idx >> 5;
        const int c4 = idx & 31;
        float4 mv = (r < rows) ? *reinterpret_cast<const float4*>(macc + r * HD + c4 * 4)
                               : make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4*>(sO_tmp + r * HD + c4 * 4) = mv;
      }
      __syncthreads();

      wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[2];
      if (w < 8) {
        const int o_wm = w >> 2;
        const int o_wd = w & 3;
        wmma::load_matrix_sync(o_frag[0], sO_tmp + o_wm * 16 * HD + (o_wd * 32 + 0), HD, wmma::mem_row_major);
        wmma::load_matrix_sync(o_frag[1], sO_tmp + o_wm * 16 * HD + (o_wd * 32 + 16), HD, wmma::mem_row_major);
      }
      __syncthreads();

      // Layout inside K_s + V_s (32,768 B total):
      // sK_h: 8,704 B | sV_h: 8,704 B | sS_w: 4,608 B | sP_h: 2,560 B | sAlpha: 128 B => 24,704 B <= 32,768 B!
      __half* sK_h = reinterpret_cast<__half*>(K_s);
      __half* sV_h = sK_h + TK * W_LDK;
      float* sS_w = reinterpret_cast<float*>(sV_h + TK * W_LDK);
      __half* sP_h = reinterpret_cast<__half*>(sS_w + W_BM * W_LDS);
      float* sAlpha = reinterpret_cast<float*>(sP_h + W_BM * W_LDP);

      const int* bt_row = p.block_table + (int64_t)req * p.stride_bt;
      // Software-pipelined K/V loads: issue tile u+ustep into registers while WMMA runs on tile u!
      constexpr int NLD_KV = (TK * (HD / 8)) / (NW * 32);  // 1 for NW=16, 2 for NW=8
      uint4 pf_k[NLD_KV], pf_v[NLD_KV];
      auto load_kv_regs = [&](int uu) {
        const int base = (tile0 + ntiles - 1 - uu) << 5;
#pragma unroll
        for (int it = 0; it < NLD_KV; ++it) {
          const int idx = tid + it * (NW * 32);
          const int kl = idx >> 4;
          const int c8 = idx & 15;
          const int key = base + kl;
          const int page = bt_row[key >> p.page_shift];
          const __half* kp = p.kv + (int64_t)page * p.stride_blk + (int64_t)(key & page_mask) * p.stride_slot + (int64_t)kvh * p.stride_h;
          pf_k[it] = __ldg(reinterpret_cast<const uint4*>(kp + c8 * 8));
          pf_v[it] = __ldg(reinterpret_cast<const uint4*>(kp + HD + c8 * 8));
        }
      };
      load_kv_regs(u);
#pragma unroll 1
      for (; u < ulimit; u += ustep) {
        // Store prefetched K/V into shared memory
#pragma unroll
        for (int it = 0; it < NLD_KV; ++it) {
          const int idx = tid + it * (NW * 32);
          const int kl = idx >> 4;
          const int c8 = idx & 15;
          *reinterpret_cast<uint4*>(sK_h + kl * W_LDK + c8 * 8) = pf_k[it];
          *reinterpret_cast<uint4*>(sV_h + kl * W_LDK + c8 * 8) = pf_v[it];
        }
        __syncthreads();
        // Overlap next tile HBM loads with this tile WMMA + softmax!
        if (u + ustep < ulimit) load_kv_regs(u + ustep);

        // QK^T: warps 0..3 cover [32, 32] (4 tiles of 16x16)
        if (w < 4) {
          const int q_wm = w >> 1;
          const int q_wn = w & 1;
          wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
          wmma::fill_fragment(s_frag, 0.f);
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
            wmma::load_matrix_sync(k_frag, sK_h + q_wn * 16 * W_LDK + k * 16, W_LDK);
            wmma::mma_sync(s_frag, q_frag[k], k_frag, s_frag);
          }
          wmma::store_matrix_sync(sS_w + q_wm * 16 * W_LDS + q_wn * 16, s_frag, W_LDS, wmma::mem_row_major);
        }
        __syncthreads();

        // Online softmax: warps 0..7 handle 4 rows each (32 cols = 1 col/lane)
        if (w < 8) {
#pragma unroll
          for (int dr = 0; dr < 4; ++dr) {
            const int r = w * 4 + dr;
            if (r < rows) {
              const float sc = sS_w[r * W_LDS + lane] * p.scale_log2;
              const float mo = m_s[r];
              const float mn = fmaxf(mo, warp_max(sc));
              const float ms = (mn == -INFINITY) ? 0.f : mn;
              const float alpha = (mo == -INFINITY) ? 0.f : ex2f(mo - ms);
              const float pr = ex2f(sc - ms);
              float lt = pr;
#pragma unroll
              for (int o = 16; o > 0; o >>= 1) lt += shx(lt, o);
              sP_h[r * W_LDP + lane] = __float2half_rn(pr);
              if (lane == 0) {
                m_s[r] = mn;
                l_s[r] = l_s[r] * alpha + lt;
                sAlpha[r] = alpha;
              }
            } else {
              sP_h[r * W_LDP + lane] = __float2half_rn(0.f);
              if (lane == 0) sAlpha[r] = 1.f;
            }
          }
        }
        __syncthreads();

        // Rescale o_frag & P @ V: warps 0..7 (o_wm = w>>2, o_wd = w&3 -> 32 dims/warp)
        if (w < 8) {
          const int o_wm = w >> 2;
          const int o_wd = w & 3;
          const int base_r = o_wm * 16;
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int r = base_r + (lane & 1) + 2 * ((e >> 1) & 1) + 4 * ((lane >> 4) & 1) + 8 * ((lane >> 2) & 1);
            const float a = sAlpha[r];
            o_frag[0].x[e] *= a;
            o_frag[1].x[e] *= a;
          }
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
            wmma::load_matrix_sync(p_frag, sP_h + o_wm * 16 * W_LDP + k * 16, W_LDP);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v0_frag, v1_frag;
            wmma::load_matrix_sync(v0_frag, sV_h + k * 16 * W_LDK + (o_wd * 32 + 0), W_LDK);
            wmma::load_matrix_sync(v1_frag, sV_h + k * 16 * W_LDK + (o_wd * 32 + 16), W_LDK);
            wmma::mma_sync(o_frag[0], p_frag, v0_frag, o_frag[0]);
            wmma::mma_sync(o_frag[1], p_frag, v1_frag, o_frag[1]);
          }
        }
        __syncthreads();
      }

      // Write final o_frag back to sO_tmp (K_s) and copy [rows, 128] -> macc
      if (w < 8) {
        const int o_wm = w >> 2;
        const int o_wd = w & 3;
        wmma::store_matrix_sync(sO_tmp + o_wm * 16 * HD + (o_wd * 32 + 0), o_frag[0], HD, wmma::mem_row_major);
        wmma::store_matrix_sync(sO_tmp + o_wm * 16 * HD + (o_wd * 32 + 16), o_frag[1], HD, wmma::mem_row_major);
      }
      __syncthreads();
      for (int idx = tid; idx < rows * (HD / 4); idx += NW * 32) {
        const int r = idx >> 5;
        const int c4 = idx & 31;
        *reinterpret_cast<float4*>(macc + r * HD + c4 * 4) = *reinterpret_cast<const float4*>(sO_tmp + r * HD + c4 * 4);
      }
      __syncthreads();
    }

    if (p.stop == 4) return;
    // ---- this CTA's partial: raw accumulators + (max, sum) ----
#pragma unroll 1
    for (int r = w; r < rows; r += NW) {
      const int tk = r / GQ;
      const int64_t idx = ((int64_t)split * p.T + q_start + tk) * p.Hq + kvh * GQ + (r - tk * GQ);
      *reinterpret_cast<float4*>(p.o_part + idx * HD + lane * 4) = *reinterpret_cast<const float4*>(macc + r * HD + lane * 4);
      if (lane == 0) *reinterpret_cast<float2*>(p.ml_part + idx * 2) = make_float2(m_s[r], l_s[r]);
    }
    __threadfence();
  }

  // ---- arrival at the grid barrier of the S CTAs of this (request, KV head): the counter
  //      wraps to 0 with the S-th arrival, so it resets itself for the next launch ----
  __syncthreads();
  if (p.stop >= 1 && p.stop <= 5) return;
  int* ctr = p.counters + req * p.Hkv + kvh;
  if (tid == 0) atomicInc(reinterpret_cast<unsigned int*>(ctr), (unsigned int)(S - 1));

  // ---- combine: output row r is merged by CTA S - 1 - r: the high splits own no tile at the
  //      usual window, and CTA 0 (K/V conv, the last to arrive) gets a row only if rows >= S ----
  const int r0 = S - 1 - split;
  if (r0 >= rows) return;
  const int64_t part_stride = (int64_t)p.T * p.Hq;
  // pass 0 runs the combine once on whatever the partials hold while this CTA would only
  // wait: the code is then in the SM's instruction cache when the real pass starts.
#pragma unroll 1
  for (int pass = 0; pass < 2; pass++) {
   if (pass == 1) {
    if (tid == 0) {
      int it = 0;
      while (ld_volatile(ctr) != 0 && ++it < SPIN_MAX) {
      }
      if (it >= SPIN_MAX) *p.fault = 1;
    }
    __syncthreads();
    if (p.stop == 6) return;
   }
#pragma unroll 1
   for (int r = r0; r < rows; r += S) {
    const int tk = r / GQ;
    const int token = q_start + tk;
    const int head = kvh * GQ + (r - tk * GQ);
    const int64_t row_idx = (int64_t)token * p.Hq + head;
    {
      // partial accumulators of this warp's splits (w, w + 8, ...): all loads in flight,
      // together with the (max, sum) pairs that threads 0..active-1 fetch
      float4 pv[CSP];
#pragma unroll
      for (int j = 0; j < CSP; j++) {
        const int sp = w + NWC * j;
        pv[j] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (w < NWC && sp < active) pv[j] = __ldcg(reinterpret_cast<const float4*>(p.o_part + (sp * part_stride + row_idx) * HD + lane * 4));
      }
      float mx = -INFINITY;
      if (tid < active) {
        const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.ml_part + ((int64_t)tid * part_stride + row_idx) * 2));
        cm_s[tid] = ml.x;
        cl_s[tid] = ml.y;
        mx = ml.x;
      }
      if (w < MAXW) {  // active <= MS: the maxima sit on the first MAXW warps
        mx = warp_max(mx);
        if (lane == 0) K_s[NWC * HD + NWC + w] = mx;
      }
      __syncthreads();
      if (pass && p.stop == 7) return;
      float M = K_s[NWC * HD + NWC];
#pragma unroll
      for (int i = 1; i < MAXW; i++) M = fmaxf(M, K_s[NWC * HD + NWC + i]);
      const float Ms = M == -INFINITY ? 0.f : M;
      float4 O = make_float4(0.f, 0.f, 0.f, 0.f);
      float den = 0.f;
#pragma unroll
      for (int j = 0; j < CSP; j++) {
        const int sp = w + NWC * j;
        float wt = 0.f, lw = 0.f;
        if (w < NWC && sp < active) {
          wt = ex2f(cm_s[sp] - Ms);
          lw = cl_s[sp];
        }
        den = fmaf(lw, wt, den);
        O.x = fmaf(pv[j].x, wt, O.x);
        O.y = fmaf(pv[j].y, wt, O.y);
        O.z = fmaf(pv[j].z, wt, O.z);
        O.w = fmaf(pv[j].w, wt, O.w);
      }
      // K_s is free now: [NW][HD] warp sums, then [NW] denominators
      if (w < NWC) {
        *reinterpret_cast<float4*>(K_s + w * HD + lane * 4) = O;
        if (lane == 0) K_s[NWC * HD + w] = den;
      }
      __syncthreads();
      if (pass && p.stop == 8) return;
      if (w == 0) {
        den = 0.f;
        O = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 1
        for (int ww = 0; ww < NWC; ww++) {
          const float4 a = *reinterpret_cast<const float4*>(K_s + ww * HD + lane * 4);
          den += K_s[NWC * HD + ww];
          O.x += a.x; O.y += a.y; O.z += a.z; O.w += a.w;
        }
        const float lse2 = den > 0.f ? Ms + lg2f(den) : -INFINITY;
        const float sk = sigmoidf_(lse2 * LN2_F - __half2float(p.sinks[(int64_t)token * p.stride_sink + head]));
        // gate is a slice of the qkvr projection: rows are only 4-byte aligned
        const uint32_t* gp = reinterpret_cast<const uint32_t*>(p.gate + (int64_t)token * p.stride_gate + head * HD + lane * 4);
        const float4 gt = up4(make_uint2(gp[0], gp[1]));
        float4 y;
        y.x = (den > 0.f ? O.x / den : 0.f) * sigmoidf_(gt.x) * sk;
        y.y = (den > 0.f ? O.y / den : 0.f) * sigmoidf_(gt.y) * sk;
        y.z = (den > 0.f ? O.z / den : 0.f) * sigmoidf_(gt.z) * sk;
        y.w = (den > 0.f ? O.w / den : 0.f) * sigmoidf_(gt.w) * sk;
        uint2 yv;
        *reinterpret_cast<__half2*>(&yv.x) = __floats2half2_rn(y.x, y.y);
        *reinterpret_cast<__half2*>(&yv.y) = __floats2half2_rn(y.z, y.w);
        if (pass == 1) *reinterpret_cast<uint2*>(p.out + (int64_t)token * (p.Hq * HD) + head * HD + lane * 4) = yv;
      }
    }
    __syncthreads();
   }
  }
}

template <int NW, int MS>
static void launch_attn2(dim3 grid, const Params& p, size_t smem, cudaStream_t stream) {
  static size_t smem_set = 0;
  if (smem > smem_set) {
    TORCH_CHECK(cudaFuncSetAttribute(attn2_kernel<NW, MS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(96 * 1024)) == cudaSuccess);
    smem_set = 96 * 1024;
  }
  attn2_kernel<NW, MS><<<grid, NW * 32, smem, stream>>>(p);
}

#define CHK_HALF(t) TORCH_CHECK((t).scalar_type() == at::kHalf, #t " must be fp16")
#define CHK_INT(t) TORCH_CHECK((t).scalar_type() == at::kInt, #t " must be int32")
#define CHK_LONG(t) TORCH_CHECK((t).scalar_type() == at::kLong, #t " must be int64")

void decode_attention2_cuda(
    torch::Tensor q, torch::Tensor r, torch::Tensor q_scale, torch::Tensor r_scale,
    c10::optional<torch::Tensor> tau, torch::Tensor rel_t, torch::Tensor kv,
    torch::Tensor block_table, torch::Tensor seq_lens, torch::Tensor cu_q,
    torch::Tensor gate, torch::Tensor sinks, torch::Tensor out,
    torch::Tensor o_part, torch::Tensor ml_part, torch::Tensor counters, torch::Tensor fault,
    torch::Tensor kx, torch::Tensor vx, torch::Tensor kw, torch::Tensor vw,
    c10::optional<torch::Tensor> kdyn, c10::optional<torch::Tensor> vdyn,
    torch::Tensor cache, torch::Tensor pos, torch::Tensor kv_slot, torch::Tensor cslot,
    torch::Tensor seq_idx, torch::Tensor qstart, torch::Tensor cbt,
    int64_t num_reqs, int64_t max_query_len, int64_t num_splits, int64_t window,
    double softmax_scale, double q_thr, double r_thr, double eps, double conv_scale, double norm_eps, int64_t stop,
    int64_t num_warps, int64_t max_splits, int64_t prefetch) {
  const at::cuda::CUDAGuard guard(q.device());
  CHK_HALF(q); CHK_HALF(r); CHK_HALF(q_scale); CHK_HALF(r_scale); CHK_HALF(rel_t); CHK_HALF(kv);
  CHK_HALF(gate); CHK_HALF(sinks); CHK_HALF(out); CHK_HALF(kx); CHK_HALF(vx); CHK_HALF(kw); CHK_HALF(vw);
  CHK_HALF(cache);
  CHK_INT(block_table); CHK_INT(seq_lens); CHK_INT(cu_q); CHK_INT(seq_idx); CHK_INT(qstart); CHK_INT(cbt);
  CHK_INT(counters); CHK_INT(fault);
  CHK_LONG(pos); CHK_LONG(kv_slot); CHK_LONG(cslot);
  Params p;
  p.q = reinterpret_cast<const __half*>(q.data_ptr());
  p.r = reinterpret_cast<const __half*>(r.data_ptr());
  p.q_scale = reinterpret_cast<const __half*>(q_scale.data_ptr());
  p.r_scale = reinterpret_cast<const __half*>(r_scale.data_ptr());
  p.tau = tau.has_value() ? tau->data_ptr<float>() : nullptr;
  p.rel_t = reinterpret_cast<const __half*>(rel_t.data_ptr());
  p.kv = reinterpret_cast<__half*>(kv.data_ptr());
  p.block_table = block_table.data_ptr<int>();
  p.seq_lens = seq_lens.data_ptr<int>();
  p.cu_q = cu_q.data_ptr<int>();
  p.gate = reinterpret_cast<const __half*>(gate.data_ptr());
  p.sinks = reinterpret_cast<const __half*>(sinks.data_ptr());
  p.out = reinterpret_cast<__half*>(out.data_ptr());
  p.o_part = o_part.data_ptr<float>();
  p.ml_part = ml_part.data_ptr<float>();
  p.counters = counters.data_ptr<int>();
  p.fault = fault.data_ptr<int>();
  p.x[0] = reinterpret_cast<const __half*>(kx.data_ptr());
  p.x[1] = reinterpret_cast<const __half*>(vx.data_ptr());
  p.w[0] = reinterpret_cast<const __half*>(kw.data_ptr());
  p.w[1] = reinterpret_cast<const __half*>(vw.data_ptr());
  const bool dynamic = kdyn.has_value();
  TORCH_CHECK(dynamic == vdyn.has_value());
  p.dyn[0] = dynamic ? reinterpret_cast<const __half*>(kdyn->data_ptr()) : nullptr;
  p.dyn[1] = dynamic ? reinterpret_cast<const __half*>(vdyn->data_ptr()) : nullptr;
  p.cache = reinterpret_cast<__half*>(cache.data_ptr());
  p.pos = pos.data_ptr<int64_t>();
  p.kv_slot = kv_slot.data_ptr<int64_t>();
  p.cslot = cslot.data_ptr<int64_t>();
  p.seq_idx = seq_idx.data_ptr<int>();
  p.qstart = qstart.data_ptr<int>();
  p.cbt = cbt.data_ptr<int>();
  p.stride_q = q.stride(0);
  p.stride_r = r.stride(0);
  p.stride_bt = block_table.stride(0);
  p.stride_gate = gate.stride(0);
  p.stride_sink = sinks.stride(0);
  p.stride_blk = kv.stride(0);
  p.stride_slot = kv.stride(1);
  p.stride_h = kv.stride(2);
  p.stride_x[0] = kx.stride(0);
  p.stride_x[1] = vx.stride(0);
  p.stride_dyn[0] = dynamic ? kdyn->stride(0) : 0;
  p.stride_dyn[1] = dynamic ? vdyn->stride(0) : 0;
  p.stride_w0 = kw.stride(0);
  p.stride_c_blk = cache.stride(0);
  p.stride_c_h = cache.stride(1);
  p.stride_c_n = cache.stride(2);
  p.stride_cbt = cbt.stride(0);
  p.T = q.size(0);
  p.Hq = q.size(1);
  p.Hkv = kv.size(2);
  p.S = (int)num_splits;
  int ps = 0;
  while ((1 << ps) < kv.size(1)) ps++;
  int cs = 0;
  while ((1 << cs) < cache.size(2)) cs++;
  TORCH_CHECK((1 << ps) == kv.size(1) && (1 << cs) == cache.size(2), "page sizes must be powers of 2");
  p.page_shift = ps;
  p.cshift = cs;
  p.cmax = cbt.size(1);
  p.window = (int)window;
  p.E = rel_t.size(0);
  p.maxrows = (int)max_query_len * GQ;
  p.scale_log2 = (float)softmax_scale * LOG2E_F;
  p.q_thr = (float)q_thr;
  p.r_thr = (float)r_thr;
  p.eps = (float)eps;
  p.conv_scale = (float)conv_scale;
  p.norm_eps = (float)norm_eps;
  p.stop = (int)stop;
  p.pf = (int)prefetch;
  TORCH_CHECK(p.Hq == p.Hkv * GQ, "decode_attention2 supports a GQA group of 3 only");
  TORCH_CHECK(max_query_len >= 1 && max_query_len <= 8, "max_query_len must be 1..8");
  TORCH_CHECK((num_warps == 8 || num_warps == 16) && (max_splits == 40 || max_splits == 80) && !(num_warps == 16 && max_splits == 80),
              "bad kernel variant");
  const size_t smem = (size_t)(2 * TK * HD + p.maxrows * (2 * HD + 16 + TK + 2) + 2 * max_splits) * sizeof(float);
  // one CTA per SM, or two when the kernel is the 8-warp one and the shared memory of two fits
  const bool two = num_warps == 8 && smem <= 47 * 1024;
  const int64_t budget = two ? 160 : 80;
  TORCH_CHECK(p.S >= 1 && p.S <= max_splits && (int64_t)p.S * p.Hkv * num_reqs <= budget, "grid must fit the SMs (barrier)");
  TORCH_CHECK(max_splits == 40 || (num_warps == 8 && smem <= 47 * 1024) || (int64_t)p.S * p.Hkv * num_reqs <= 80,
              "80 splits need two CTAs per SM");
  TORCH_CHECK(kw.stride(dynamic ? 1 : 0) == (dynamic ? 1 : 4) && vw.stride(0) == kw.stride(0), "conv tap layout");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(p.S, p.Hkv, (unsigned)num_reqs);
  if (num_warps == 8 && max_splits == 40) launch_attn2<8, 40>(grid, p, smem, stream);
  else if (num_warps == 16) launch_attn2<16, 40>(grid, p, smem, stream);
  else launch_attn2<8, 80>(grid, p, smem, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP_SRC = r"""
void decode_attention2_cuda(
    torch::Tensor q, torch::Tensor r, torch::Tensor q_scale, torch::Tensor r_scale,
    c10::optional<torch::Tensor> tau, torch::Tensor rel_t, torch::Tensor kv,
    torch::Tensor block_table, torch::Tensor seq_lens, torch::Tensor cu_q,
    torch::Tensor gate, torch::Tensor sinks, torch::Tensor out,
    torch::Tensor o_part, torch::Tensor ml_part, torch::Tensor counters, torch::Tensor fault,
    torch::Tensor kx, torch::Tensor vx, torch::Tensor kw, torch::Tensor vw,
    c10::optional<torch::Tensor> kdyn, c10::optional<torch::Tensor> vdyn,
    torch::Tensor cache, torch::Tensor pos, torch::Tensor kv_slot, torch::Tensor cslot,
    torch::Tensor seq_idx, torch::Tensor qstart, torch::Tensor cbt,
    int64_t num_reqs, int64_t max_query_len, int64_t num_splits, int64_t window,
    double softmax_scale, double q_thr, double r_thr, double eps, double conv_scale, double norm_eps, int64_t stop,
    int64_t num_warps, int64_t max_splits, int64_t prefetch);
"""

MAX_CTAS = 80  # one CTA per SM (V100): all of them resident for the grid barrier
MAX_SPLITS = 40  # CTAs per (request, KV head): the combine merges 8 warps x 5 partials
# VLLM_INKLING2_SM70_ATTN2_2CTA=1: two CTAs per SM (160 resident) for query lengths <= 3 (shared memory <= 47 KB):
# up to 80 splits on global layers, twice the splits for batches
_TWO_CTA = os.environ.get("VLLM_INKLING2_SM70_ATTN2_2CTA", "1") == "1"
MAX_QUERY_LEN = 8
_STOP = int(
    os.environ.get("INK_ATTN2_STOP", "0")
)  # tuning aid: leave the kernel after stage N
# VLLM_INKLING2_SM70_ATTN2_WARPS: 8 = first build (256 threads), 16 = 512 threads: up to 16 rows (5 tokens) in one round
_NW = int(os.environ.get("VLLM_INKLING2_SM70_ATTN2_WARPS", "16"))
# VLLM_INKLING2_SM70_ATTN2_PREFETCH=1: block-table lookups 3 tiles ahead + L2 prefetch of K/V two tiles ahead
_PREFETCH = int(os.environ.get("VLLM_INKLING2_SM70_ATTN2_PREFETCH", "1") == "1")
assert _NW in (8, 16)

_EXT = None
_STATE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        build_dir = os.environ.get(
            "VLLM_INKLING2_SM70_EXT_DIR",
            os.path.expanduser("~/.cache/vllm/inkling2_sm70_ext"),
        )
        build_dir = os.path.join(build_dir, "decode_attention2")
        os.makedirs(build_dir, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
        _EXT = load_inline(
            name="inkling2_sm70_decode_attention2",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["decode_attention2_cuda"],
            build_directory=build_dir,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
            verbose=False,
        )
    return _EXT


def _state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    st = _STATE.get(idx)
    if st is None:
        st = (
            torch.zeros(MAX_CTAS, dtype=torch.int32, device=device),  # barrier counters
            torch.zeros(1, dtype=torch.int32, device=device),  # barrier timeout flag
        )
        _STATE[idx] = st
    return st


def prepare(device: torch.device) -> None:
    """Build the extension and allocate the barrier counters eagerly (never
    inside CUDA graph capture)."""
    _ext()
    _state(device)


def fault(device: torch.device) -> bool:
    """True if a grid barrier ever timed out on this device (host sync: tests only)."""
    return bool(_state(device)[1].item())


def supports(
    num_heads: int, num_kv_heads: int, head_dim: int, d_rel: int, dtype: torch.dtype
) -> bool:
    return (
        num_heads == 3 * num_kv_heads
        and head_dim == 128
        and d_rel == 16
        and dtype == torch.float16
        and num_kv_heads <= MAX_CTAS
    )


def plan_splits(
    num_reqs: int, num_kv_heads: int, max_query_len: int = 1, local: bool = True
) -> int:
    """CTAs per (request, KV head); 0 = the batch does not fit the barrier grid.
    Depends on the batch only (CUDA-graph safe)."""
    if _TWO_CTA and max_query_len <= (2 if _NW == 16 else 3):
        return min(40 if local else 80, 2 * MAX_CTAS // (num_reqs * num_kv_heads))
    return min(MAX_SPLITS, MAX_CTAS // (num_reqs * num_kv_heads))


def _aligned(t: torch.Tensor, n: int) -> bool:
    return t.data_ptr() % n == 0


def usable(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_dynamic: torch.Tensor | None,
    v_dynamic: torch.Tensor | None,
    kv_cache: torch.Tensor,
    conv_cache: torch.Tensor,
    positions: torch.Tensor,
    conv_slot_mapping: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
) -> bool:
    """Layout checks of one call (shapes, dtypes and alignments only: the
    result is the same for every replay of a CUDA graph)."""
    page, cn = kv_cache.shape[1], conv_cache.shape[2]
    ok = (
        q.dtype == torch.float16
        and k.dtype == torch.float16
        and v.dtype == torch.float16
        and kv_cache.dtype == torch.float16
        and conv_cache.dtype == torch.float16
        and positions.dtype == torch.int64
        and conv_slot_mapping.dtype == torch.int64
        and kv_slot_mapping.dtype == torch.int64
        and page & (page - 1) == 0
        and cn & (cn - 1) == 0
        and kv_cache.stride(3) == 1
        and conv_cache.stride(3) == 1
        and _aligned(kv_cache, 16)
        and all(s % 8 == 0 for s in kv_cache.stride()[:3])
        and _aligned(conv_cache, 8)
        and all(s % 4 == 0 for s in conv_cache.stride()[:3])
        and k.stride(1) == 1
        and v.stride(1) == 1
        and _aligned(k, 4)
        and _aligned(v, 4)
        and k.stride(0) % 2 == 0
        and v.stride(0) % 2 == 0
        and k_weight.dtype == torch.float16
        and v_weight.dtype == torch.float16
    )
    if not ok:
        return False
    if k_dynamic is None:
        return (
            k_weight.is_contiguous()
            and v_weight.is_contiguous()
            and k_weight.shape[1] == 4
            and _aligned(k_weight, 16)
            and _aligned(v_weight, 16)
        )
    assert v_dynamic is not None
    return (
        k_weight.shape[0] == 4
        and k_weight.stride(1) == 1
        and v_weight.stride() == k_weight.stride()
        and k_weight.stride(0) % 4 == 0
        and _aligned(k_weight, 8)
        and _aligned(v_weight, 8)
        and k_dynamic.dtype == torch.float16
        and v_dynamic.dtype == torch.float16
        and k_dynamic.stride(1) == 1
        and v_dynamic.stride(1) == 1
        and k_dynamic.stride(0) % 4 == 0
        and v_dynamic.stride(0) % 4 == 0
        and _aligned(k_dynamic, 8)
        and _aligned(v_dynamic, 8)
    )


def decode_attention2(
    q: torch.Tensor,
    r: torch.Tensor,
    q_scale: torch.Tensor,
    r_scale: torch.Tensor,
    tau: torch.Tensor | None,
    rel_t: torch.Tensor,
    kv_cache: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_dynamic: torch.Tensor | None,
    v_dynamic: torch.Tensor | None,
    conv_cache: torch.Tensor,
    positions: torch.Tensor,
    conv_block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    conv_slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
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
    norm_eps: float,
    eps: float = 1e-6,
) -> None:
    """K/V short convs + cache writes + gated attention output `(T, H * D)` of
    a decode / verify batch in one launch.

    `q` `(T, H, D)`, `r` `(T, H, d_rel)`: raw projections; `rel_t` the
    `(extent, d_rel)` transposed relative projection; `k`, `v` `(T, H_kv * D)`
    raw K/V projections; conv taps as for `kv_conv_prologue`. Rows from
    `num_actual_tokens` on are zeroed. `window` is 0 for global layers.
    """
    num_tokens, num_heads, head_dim = q.shape
    assert q.stride(2) == 1 and q.stride(1) == head_dim
    assert r.stride(2) == 1 and r.stride(1) == r.shape[2]
    assert rel_t.is_contiguous() and rel_t.shape[1] == 16 and rel_t.data_ptr() % 16 == 0
    assert gate.stride(1) == 1 and sinks.stride(1) == 1 and out.is_contiguous()
    num_reqs = seq_lens.shape[0]
    assert num_splits >= 1 and num_splits * kv_cache.shape[2] * num_reqs <= 2 * MAX_CTAS
    if num_actual_tokens < num_tokens:
        out[num_actual_tokens:].zero_()
    o_part = torch.empty(
        (num_splits, num_tokens, num_heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    ml_part = torch.empty(
        (num_splits, num_tokens, num_heads, 2), dtype=torch.float32, device=q.device
    )
    counters, fault_flag = _state(q.device)
    dynamic = k_dynamic is not None
    width = k_weight.shape[0] if dynamic else k_weight.shape[1]
    _ext().decode_attention2_cuda(
        q,
        r,
        q_scale,
        r_scale,
        tau,
        rel_t,
        kv_cache,
        block_table,
        seq_lens,
        cu_seqlens_q,
        gate,
        sinks,
        out,
        o_part,
        ml_part,
        counters,
        fault_flag,
        k,
        v,
        k_weight,
        v_weight,
        k_dynamic,
        v_dynamic,
        conv_cache,
        positions,
        kv_slot_mapping,
        conv_slot_mapping,
        seq_idx,
        query_start,
        conv_block_table,
        num_reqs,
        max_query_len,
        num_splits,
        window,
        softmax_scale,
        qk_clip,
        rel_clip,
        eps,
        (width * (2 if dynamic else 1) + 1) ** -0.5,
        norm_eps,
        _STOP,
        16 if (_NW == 16 and max_query_len >= 3) else 8,
        80 if num_splits > 40 else 40,
        _PREFETCH,
    )
