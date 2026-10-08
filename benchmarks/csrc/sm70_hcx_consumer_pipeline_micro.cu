// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// HCX: one-launch TP4 "all-reduce -> HC combine+RMSNorm -> HC down -> HC up"
// for Qwen3.8 Flash-Next on SM70 (M <= 8). Replaces cube_allreduce +
// _hc_combine_norm + down3 + up3.
//
// Each rank runs 80 CTAs; CTA i owns hidden columns [32i, 32i+32) of every HC
// stream.
//   1. Recursive-doubling all-reduce of the block output over r^1 and r^2
//   (FP32, symmetric adds,
//      so all ranks hold bit-identical sums), optionally adding a second local
//      partial first.
//   2. Combine (res + block * 2*sigmoid(inj/4)) and per-(row, stream) partial
//   sum of squares.
//   3. Intra-GPU flag barrier, rrms, gemma RMSNorm -> xn (kept in smem +
//   global).
//   4. HC down partial over the CTA's own 128 K rows (weights prefetched into
//   registers at
//      kernel start); intra-GPU barrier; deterministic split-K reduction; SiLU;
//      LL push of the rank's 80 lora (+4 injection) columns to r^1/r^2, forward
//      of r^1's columns to r^2.
//   5. HC up for the rank's 640 hidden columns (weights prefetched), gate-mix
//   with xn, LL push
//      of the hidden slice, forward, collect.
// Cross-GPU words are 8-byte {payload, 32-bit epoch tag}; only direct peers
// r^1, r^2 are written.
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
namespace consdmv {
#include "gguf_dmv13_core.cuh"
}  // namespace consdmv

namespace hcxcons {
using consdmv::body6;
using consdmv::Seg;
constexpr int HD = 2560, KD = 10240, NC = 80, LW = 336;

struct Args {
  const half* p0;
  const half* p1;
  const half* res;
  const half* inj;
  const half* nw;
  int nw_full;
  float eps;
  const uint4* wd;
  const uint4* wu;
  half* res_out;
  half* blk_out;
  half* inj_out;
  half* xn;
  float* sq;
  float* dpart;
  unsigned* bar;
  unsigned* seq;
  unsigned long long* dbg;
  uint2* ar[4];
  uint2* lora[4];
  uint2* hb[4];
  int rank;
  int M;
  // fused producer GEMV (OF >= 0): x [M, oldx], packed weight tiles in seg, K =
  // 128 * oG
  const half* ox;
  int oldx;
  int oG;
  Seg oseg;
  // optional gated RMSNorm on the producer input (GDN): x = core [M, K] fp16, z
  // [M, K] (row stride ldz), weight [128], per (token, 128-wide head)
  const half* gz;
  int ldz;
  const half* gw;
  float geps;
  half* gscr;  // [M][K] scratch for the normalized producer input
  consdmv::Segs consumer;
  float* consumer_partial;
  unsigned* consumer_ready;
};

__device__ __forceinline__ void mma(float (&d)[8], uint32_t a0, uint32_t a1,
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned ldv(const unsigned* p) {
  unsigned v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void st2(uint2* p, unsigned a, unsigned tag) {
  asm volatile("st.volatile.global.v2.u32 [%0], {%1,%2};" ::"l"(p), "r"(a),
               "r"(tag)
               : "memory");
}
__device__ __forceinline__ void st4(uint2* p, unsigned a, unsigned b,
                                    unsigned tag) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p),
               "r"(a), "r"(tag), "r"(b), "r"(tag)
               : "memory");
}
__device__ __forceinline__ unsigned poll2(const uint2* p, unsigned tag) {
  unsigned a, b;
  const long long t0 = clock64();
  do {
    asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];"
                 : "=r"(a), "=r"(b)
                 : "l"(p));
    if (clock64() - t0 > 4000000000LL) __trap();
  } while (b != tag);
  return a;
}
__device__ __forceinline__ uint2 poll4(const uint2* p, unsigned tag) {
  unsigned a, b, c, d;
  const long long t0 = clock64();
  do {
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                 : "l"(p));
    if (clock64() - t0 > 4000000000LL) __trap();
  } while (b != tag || d != tag);
  return make_uint2(a, c);
}
// Ordered non-coherent 16-byte load: volatile asm keeps it ahead of the later
// polling asm.
__device__ __forceinline__ uint4 ldgv(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}
__device__ __forceinline__ float4 ldcg4(const float4* p) {
  float4 r;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
               : "l"(p));
  return r;
}
__device__ __forceinline__ void pf_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}
__device__ __forceinline__ float sigm(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float divf(float a, float b) {
  float r;
  asm("div.full.f32 %0,%1,%2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ void grid_bar(unsigned* bar, unsigned target) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(bar, 1u);
    const long long t0 = clock64();
    while (ldv(bar) < target)
      if (clock64() - t0 > 4000000000LL) __trap();
    __threadfence();
  }
  __syncthreads();
}

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long v;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
  return v;
}
#define TS(k) \
  if (a.dbg && threadIdx.x == 0) a.dbg[blockIdx.x * 16 + (k)] = gtime()

// Load only the ready 32-column K fragment from the production canonical
// planes. The common decoder retains FP16 weight reconstruction and FP32 MMA.
template <int F>
__device__ __forceinline__ void consume_fragment(const consdmv::Seg& sg, int nt,
                                                 int kg, int ks, const half* xs,
                                                 int xstride, int rankchunk,
                                                 float (&acc)[8]) {
  const int lane = threadIdx.x & 31;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const size_t tg = static_cast<size_t>(nt) * 20 + kg;
  consdmv::Ld<F> w;
  w.c[ks][0] = __ldcs(sg.codes + (tg * 4 + ks) * 32 + lane);
  if constexpr (F == consdmv::LUT4) {
    // Keep the actual segment bank's 16-byte scale record. Adapt its
    // duplicated half scale to the common register-only LUT decoder.
    const unsigned sc = __ldcs(reinterpret_cast<const unsigned*>(sg.scale) +
                               (tg * 32 + lane) * 4 + ks);
    reinterpret_cast<unsigned*>(&w.sc)[ks >> 1] = (sc & 0xffffu)
                                                  << (16 * (ks & 1));
  } else {
    reinterpret_cast<unsigned*>(&w.sc)[ks] =
        __ldcs(reinterpret_cast<const unsigned*>(sg.scale) +
               (tg * 32 + lane) * 4 + ks);
  }
  if constexpr (F == consdmv::Q5K) {
    reinterpret_cast<unsigned*>(&w.hi[0])[ks] = __ldcs(
        reinterpret_cast<const unsigned*>(sg.high) + (tg * 32 + lane) * 4 + ks);
  }
  if constexpr (F == consdmv::Q6K) {
    const uint2 high =
        __ldcs(reinterpret_cast<const uint2*>(sg.high) +
               ((tg * 2 + (ks >> 1)) * 32 + lane) * 2 + (ks & 1));
    unsigned* dst = reinterpret_cast<unsigned*>(&w.hi[ks >> 1]) + 2 * (ks & 1);
    dst[0] = high.x;
    dst[1] = high.y;
  }
  uint32_t hw[16];
  consdmv::decode<F>(w, ks, hw, nullptr);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4 x = *reinterpret_cast<const uint4*>(xs + r * xstride +
                                                    rankchunk * 32 + j * 8);
    consdmv::mma(acc, x.x, x.y, hw[4 * j], hw[4 * j + 1]);
    consdmv::mma(acc, x.z, x.w, hw[4 * j + 2], hw[4 * j + 3]);
  }
}

__device__ __forceinline__ void consume_output(Args a, unsigned tag, half* xs) {
  const int i = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int group = i / 4, npart = i % 4;
  // Publish only after this CTA has materialized all four ranks' columns.
  __threadfence();
  __syncthreads();
  if (t == 0) atomicExch(a.consumer_ready + i, tag);
  __syncthreads();
  if (t < 4) {
    const long long start = clock64();
    while (ldv(a.consumer_ready + 4 * group + t) != tag)
      if (clock64() - start > 4000000000LL) __trap();
  }
  __syncthreads();
  // Independent groups require no global HC completion barrier.
  for (int q = t; q < 8 * 32; q += 256) {
    const int row = q / 32, c4 = q % 32;
    uint2 v = make_uint2(0, 0);
    if (row < a.M) {
      const int rankchunk = c4 / 8, k = 4 * (c4 % 8);
      const half* src = a.blk_out + row * HD + rankchunk * 640 + group * 32 + k;
      asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];"
                   : "=r"(v.x), "=r"(v.y)
                   : "l"(src)
                   : "memory");
    }
    *reinterpret_cast<uint2*>(xs + row * 136 + c4 * 4) = v;
  }
  __syncthreads();
  for (int tile = warp + 8 * npart; tile < a.consumer.main_tiles; tile += 32) {
    int si = 0;
    for (int j = 1; j < a.consumer.nseg; ++j)
      if (tile >= a.consumer.s[j].tile0) si = j;
    const auto& sg = a.consumer.s[si];
    const int nt = tile - sg.tile0;
    float acc[8] = {};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int kg = q * 5 + group / 4, ks = group % 4;
      if (sg.fmt == consdmv::Q6K)
        consume_fragment<consdmv::Q6K>(sg, nt, kg, ks, xs, 136, q, acc);
      else if (sg.fmt == consdmv::Q4K)
        consume_fragment<consdmv::Q4K>(sg, nt, kg, ks, xs, 136, q, acc);
      else if (sg.fmt == consdmv::Q5K)
        consume_fragment<consdmv::Q5K>(sg, nt, kg, ks, xs, 136, q, acc);
      else if (sg.fmt == consdmv::LUT4)
        consume_fragment<consdmv::LUT4>(sg, nt, kg, ks, xs, 136, q, acc);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int row = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = nt * 32 + ((lane >> 2) & 3) * 8 +
                      ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
      if (row < a.M && col < sg.n)
        __stcg(a.consumer_partial +
                   (static_cast<size_t>(group) * a.M + row) * a.consumer.hld +
                   sg.pad + col,
               acc[e]);
    }
  }
  __syncthreads();
}

__global__ void consumer_reduce(consdmv::Segs segs, const float* p, int M) {
  const int col = blockIdx.x * 256 + threadIdx.x;
  if (col >= segs.hld) return;
  int si = 0;
  for (int j = 1; j < segs.nseg; ++j)
    if (col >= segs.s[j].pad) si = j;
  const auto& sg = segs.s[si];
  for (int row = 0; row < M; ++row) {
    float sum = 0.f;
#pragma unroll
    for (int g = 0; g < 20; ++g)
      sum += __ldcg(p + (static_cast<size_t>(g) * M + row) * segs.hld + col);
    sg.out[row * sg.out_ld + col - sg.pad] = __float2half_rn(sum);
  }
}

// FULL: full-mesh NVLink (every pair direct). One-shot all-reduce and direct
// pushes to all three peers, no forwarding hop. Sums use the fixed order (v0 +
// v1) + (v2 + v3) on every rank, the same value the two-hop recursive doubling
// produces.
template <bool FULL, int OF, bool CONSUMER>
__global__ __launch_bounds__(256, 1) void hcx_kernel(Args a) {
  TS(0);
  const int i = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int M = a.M, rk = a.rank, p1 = rk ^ 1, p2 = rk ^ 2, p3 = rk ^ 3;
  const unsigned ep = ldv(a.seq), tag = ep + 1;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0), quad = (lane >> 2) & 3;
  // ---- 1. all-reduce of the block output for columns [32i, 32i + 32). The
  // partial is loaded and pushed before the weight prefetch is issued, so the
  // cross-GPU stores are not queued behind it.
  const int row = warp, col = 32 * i + lane;
  const bool act = row < M;
  float v = 0.f;
  if constexpr (OF >= 0) {
    // HC weights are needed only after the producer: warm them into L2 behind
    // the GEMV stream.
    if (warp < 6)
      pf_l2(reinterpret_cast<const char*>(a.wd + (i * 6 + warp) * 4 * 64) +
            lane * 128);
    if (warp >= 3)
      pf_l2(reinterpret_cast<const char*>(a.wu + (i * 5 + warp - 3) * 4 * 64) +
            lane * 128);
    extern __shared__ uint4 dsm[];
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int Kp = a.oG * 128;
    if (a.gz) {
      // rmsnorm_gated_exact (SiLU gate): (token, head) rows of 128 spread over
      // all CTAs, one warp per row, arithmetic identical to the production
      // kernel; published to a global scratch.
      const int heads = Kp / 128, nrows = M * heads;
      const uint2 wv = *reinterpret_cast<const uint2*>(a.gw + lane * 4);
      for (int rr = i * 8 + warp; rr < nrows; rr += NC * 8) {
        const int tok = rr / heads, hd = rr % heads;
        const uint2 xv = __ldg(reinterpret_cast<const uint2*>(
            a.ox + tok * a.oldx + hd * 128 + lane * 4));
        const uint2 zv = __ldg(reinterpret_cast<const uint2*>(
            a.gz + tok * a.ldz + hd * 128 + lane * 4));
        const half* xh = reinterpret_cast<const half*>(&xv);
        const half* zh = reinterpret_cast<const half*>(&zv);
        const half* wh = reinterpret_cast<const half*>(&wv);
        float values[4], gates[4], weights[4], sum = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          values[q] = __half2float(xh[q]);
          gates[q] = __half2float(zh[q]);
          weights[q] = __half2float(wh[q]);
          sum = __fadd_rn(sum, __fmul_rn(values[q], values[q]));
        }
        for (int off = 16; off > 0; off >>= 1)
          sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, off));
        sum = __shfl_sync(0xffffffffU, sum, 0);
        const float mean = __fmul_rn(sum, 1.0f / 128);
        const float inverse = rsqrtf(__fadd_rn(mean, a.geps));
        half outv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float normalized =
              __fmul_rn(__fmul_rn(values[q], inverse), weights[q]);
          const float denominator = __fadd_rn(1.0f, expf(-gates[q]));
          outv[q] =
              __float2half_rn(__fmul_rn(normalized, gates[q] / denominator));
        }
        __stcg(
            reinterpret_cast<uint2*>(a.gscr + tok * Kp + hd * 128 + lane * 4),
            *reinterpret_cast<uint2*>(outv));
      }
      grid_bar(a.bar, NC);
    }
    body6<OF, 8, 1>(a.oseg, i, true, warp, 0, 0, a.oG, 4 * a.oG, a.oG,
                    a.gz ? a.gscr : a.ox, a.gz ? Kp : a.oldx, M, dsm, nullptr,
                    acc, nullptr);
    __syncthreads();
    float* red = reinterpret_cast<float*>(dsm);
#pragma unroll
    for (int e = 0; e < 8; ++e) red[warp * 256 + lane * 8 + e] = acc[e];
    __syncthreads();
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) sum += red[k * 256 + t];
    __syncthreads();
    {
      const int lv = t >> 3, e = t & 7;
      const int token = (e & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
      const int c = ((lv >> 2) & 3) * 8 +
                    ((e & 1) | (((lv >> 1) & 1) << 1) | ((e >> 2) << 2));
      red[token * 32 + c] = __half2float(__float2half_rn(sum));
    }
    __syncthreads();
    if (act) v = red[row * 32 + lane];
  }
  if (act) {
    if constexpr (OF < 0) v = __half2float(a.p0[row * HD + col]);
    if (a.p1) v += __half2float(a.p1[row * HD + col]);
    if (FULL) {
      const int mine = (rk * NC + i) * 256 +
                       t;  // slot of source rank s: (s * NC + i) * 256 + t
      st2(a.ar[p1] + mine, __float_as_uint(v), tag);
      st2(a.ar[p2] + mine, __float_as_uint(v), tag);
      st2(a.ar[p3] + mine, __float_as_uint(v), tag);
    } else {
      st2(a.ar[p1] + i * 256 + t, __float_as_uint(v), tag);
    }
  }
  // Small operands of the combine are loaded before the weight prefetch as
  // well.
  half resv[4], nwv[4];
  float g[4];
  if (act) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      resv[b] = a.res[row * KD + b * HD + col];
      nwv[b] = a.nw[a.nw_full ? b * HD + col : col];
      g[b] = __half2float(a.inj[row * 4 + b]);
    }
    asm volatile("" ::"h"(__half_as_ushort(resv[0])),
                 "h"(__half_as_ushort(resv[1])), "h"(__half_as_ushort(resv[2])),
                 "h"(__half_as_ushort(resv[3])), "h"(__half_as_ushort(nwv[0])),
                 "h"(__half_as_ushort(nwv[3])), "f"(g[0]), "f"(g[3]));
#pragma unroll
    for (int b = 0; b < 4; ++b) g[b] = 2.0f * sigm(g[b] / 4.0f);
  }
  // ---- weight prefetch (independent of every exchange)
  uint4 dlo[4], dhi[4], ulo[4], uhi[4];
  if (warp < 6) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint4* p = a.wd + ((i * 6 + warp) * 4 + s) * 64;
      dlo[s] = ldgv(p + lane);
      dhi[s] = ldgv(p + 32 + lane);
    }
  }
  // Up weights: L2 prefetch now (one 128-byte line per lane: 5 warps x 4 s x 1
  // KB), registers later.
  if (warp >= 3) {
    const char* p =
        reinterpret_cast<const char*>(a.wu + (i * 5 + warp - 3) * 4 * 64);
    pf_l2(p + lane * 128);
  }
  if (act) {
    if (FULL) {
      float x[4];
#pragma unroll
      for (int s = 0; s < 4; ++s)
        x[s] = s == rk ? v
                       : __uint_as_float(
                             poll2(a.ar[rk] + (s * NC + i) * 256 + t, tag));
      v = (x[0] + x[1]) + (x[2] + x[3]);
    } else {
      v = v + __uint_as_float(poll2(a.ar[rk] + i * 256 + t, tag));
      const int slot = (NC + i) * 256 + t;
      st2(a.ar[p2] + slot, __float_as_uint(v), tag);
      v = v + __uint_as_float(poll2(a.ar[rk] + slot, tag));
    }
  }
  __shared__ __align__(16) half xs[8][128 + 8];
  __shared__ __align__(16) half ls[8][320 + 8];
  __shared__ float red[6][8][32];
  __shared__ __align__(16) half tout[8][8];
  __shared__ float rr[8][4];
  TS(1);
  // ---- 2. combine; z = fp16(out * (1 + w)) per stream feeds HC down before
  // rrms is known
  float o[4];
  {
    const float B = __half2float(__float2half_rn(v));
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const half h = __float2half_rn(__half2float(resv[b]) + B * g[b]);
      o[b] = __half2float(h);
      if (act) a.res_out[row * KD + b * HD + col] = h;
      xs[row][b * 32 + lane] =
          act ? __float2half_rn(o[b] + o[b] * __half2float(nwv[b]))
              : __float2half_rn(0.f);
      float q = act ? o[b] * o[b] : 0.f;
#pragma unroll
      for (int s = 16; s > 0; s >>= 1) q += __shfl_xor_sync(0xffffffff, q, s);
      if (lane == 0 && act) a.sq[(i * 8 + row) * 4 + b] = q;
    }
  }
  __syncthreads();
  TS(2);
  // ---- 3. per-stream HC down partials over this CTA's 128 K rows (stream b =
  // k_local / 32)
  if (warp < 6) {
    const int tile = warp >> 1, kh = warp & 1;
    float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
          acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int k = (kh * 4 + s) * 16;
      const uint4 x0 = *reinterpret_cast<const uint4*>(&xs[r8][k]);
      const uint4 x1 = *reinterpret_cast<const uint4*>(&xs[r8][k + 8]);
      float (&acc)[8] = s < 2 ? acc0 : acc1;
      mma(acc, x0.x, x0.y, dlo[s].x, dlo[s].y);
      mma(acc, x0.z, x0.w, dlo[s].z, dlo[s].w);
      mma(acc, x1.x, x1.y, dhi[s].x, dhi[s].y);
      mma(acc, x1.z, x1.w, dhi[s].z, dhi[s].w);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int rw = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int n = tile * 32 + quad * 8 +
                    ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
      if (rw < M) {
        *reinterpret_cast<float2*>(a.dpart + ((i * 8 + rw) * 96 + n) * 4 +
                                   2 * kh) = make_float2(acc0[e], acc1[e]);
      }
    }
  }
  TS(3);
  const unsigned gb = (OF >= 0 && a.gz) ? NC : 0;
  grid_bar(a.bar, gb + NC);
  TS(4);
  // The split-K partials of this warp's first output do not depend on rrms:
  // load them now so the two L2 round trips (sq for rrms, dpart for the
  // reduction) overlap.
  auto valid = [](int q, int n) { return n < 80 || (q == 3 && n < 84); };
  const int oi0 = i + NC * warp;
  const bool has0 = oi0 < M * 96 && valid(rk, oi0 % 96);
  float4 pre[3];
#pragma unroll
  for (int j = 0; j < 3; ++j)
    pre[j] = has0 && lane + 32 * j < NC
                 ? ldcg4(reinterpret_cast<const float4*>(a.dpart) +
                         ((lane + 32 * j) * 8 + oi0 / 96) * 96 + oi0 % 96)
                 : make_float4(0.f, 0.f, 0.f, 0.f);
  // ---- 4. rrms; xn for the up mix; split-K + stream reduction lora = sum_b
  // rrms_b * P_b
  if (act) {
    float4 qs[3];
#pragma unroll
    for (int j = 0; j < 3; ++j)
      qs[j] = lane + 32 * j < NC
                  ? __ldcg(reinterpret_cast<const float4*>(a.sq) +
                           (lane + 32 * j) * 8 + row)
                  : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      auto comp = [&](const float4& f) {
        return b == 0 ? f.x : b == 1 ? f.y : b == 2 ? f.z : f.w;
      };
      float q = (comp(qs[0]) + comp(qs[1])) + comp(qs[2]);
#pragma unroll
      for (int s = 16; s > 0; s >>= 1) q += __shfl_xor_sync(0xffffffff, q, s);
      const float rrms = rsqrtf(q / HD + a.eps);
      if (lane == 0) rr[row][b] = rrms;
      float y = o[b] * rrms;
      y += y * __half2float(nwv[b]);
      a.xn[row * KD + b * HD + col] = __float2half_rn(y);
    }
  }
  __syncthreads();
  TS(9);
  auto gcol = [](int q, int n) { return n < 80 ? q * 80 + n : 320 + n - 80; };
  for (int oi = i + NC * warp; oi < M * 96; oi += NC * 8) {
    const int rw = oi / 96, n = oi % 96;
    if (!valid(rk, n)) continue;
    float pb[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 3; ++j)
      if (lane + 32 * j < NC) {
        const float4 pv =
            oi == oi0 ? pre[j]
                      : __ldcg(reinterpret_cast<const float4*>(a.dpart) +
                               ((lane + 32 * j) * 8 + rw) * 96 + n);
        pb[0] += pv.x;
        pb[1] += pv.y;
        pb[2] += pv.z;
        pb[3] += pv.w;
      }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1)
#pragma unroll
      for (int b = 0; b < 4; ++b)
        pb[b] += __shfl_xor_sync(0xffffffff, pb[b], s);
    float q = 0.f;
#pragma unroll
    for (int b = 0; b < 4; ++b) q = fmaf(rr[rw][b], pb[b], q);
    half hv = __float2half_rn(q);
    if (n < 80) {
      const float xx = divf(__half2float(hv), 4.0f);
      hv = __float2half_rn(xx * sigm(xx));
    }
    if (oi == oi0) TS(10);
    const int pos = rw * LW + gcol(rk, n);
    const unsigned w = __half_as_ushort(hv);
    if (lane == 0) st2(a.lora[rk] + pos, w, tag);
    if (lane == 1) st2(a.lora[p1] + pos, w, tag);
    if (lane == 2) st2(a.lora[p2] + pos, w, tag);
    if (FULL && lane == 3) st2(a.lora[p3] + pos, w, tag);
  }
  TS(5);
  if (warp >= 3) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint4* p = a.wu + ((i * 5 + warp - 3) * 4 + s) * 64;
      ulo[s] = ldgv(p + lane);
      uhi[s] = ldgv(p + 32 + lane);
    }
  }
  // ---- 5. HC up: collect the full lora, then gate-mix the rank's hidden slice
  // Each CTA polls a 1/80 share of the lora + injection words, then one
  // intra-GPU barrier, then every CTA bulk-reads the (now complete) buffer.
  // Avoids 80 CTAs spinning on the same lines. The poll pass also forwards
  // r^1's columns to r^2 (r^2's diagonal).
  for (int idx = i * 256 + t; idx < M * 324; idx += NC * 256) {
    const int pos = (idx / 324) * LW + idx % 324, c = idx % 324;
    const unsigned w = poll2(a.lora[rk] + pos, tag);
    if (!FULL && ((c >= 80 * p1 && c < 80 * p1 + 80) || (p1 == 3 && c >= 320)))
      st2(a.lora[p2] + pos, w, tag);
  }
  grid_bar(a.bar, gb + 2 * NC);
  for (int idx = t; idx < M * 320; idx += 256) {
    const int rw = idx / 320, c = idx % 320;
    ls[rw][c] = __ushort_as_half(
        static_cast<unsigned short>(__ldcg(a.lora[rk] + rw * LW + c).x));
  }
  for (int idx = t; idx < (8 - M) * 320; idx += 256)
    ls[M + idx / 320][idx % 320] = __float2half_rn(0.f);
  if (i == 0 && t < M * 4)
    a.inj_out[t] = __ushort_as_half(static_cast<unsigned short>(
        __ldcg(a.lora[rk] + (t >> 2) * LW + 320 + (t & 3)).x));
  __syncthreads();
  TS(6);
  if (warp >= 3) {
    const int uw = warp - 3;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int k = (uw * 4 + s) * 16;
      const uint4 x0 = *reinterpret_cast<const uint4*>(&ls[r8][k]);
      const uint4 x1 = *reinterpret_cast<const uint4*>(&ls[r8][k + 8]);
      mma(acc, x0.x, x0.y, ulo[s].x, ulo[s].y);
      mma(acc, x0.z, x0.w, ulo[s].z, ulo[s].w);
      mma(acc, x1.x, x1.y, uhi[s].x, uhi[s].y);
      mma(acc, x1.z, x1.w, uhi[s].z, uhi[s].w);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) red[uw][e][lane] = acc[e];
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float acc = red[0][e][lane];
#pragma unroll
      for (int q = 1; q < 5; ++q) acc += red[q][e][lane];
      const int rw = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int hh = (e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2);
      const int h = 640 * rk + 8 * i + hh;
      const float sg = sigm(__half2float(__float2half_rn(acc)));
      const float xv =
          rw < M ? __half2float(__ldcg(a.xn + rw * KD + quad * HD + h)) : 0.f;
      float mixed = 0.f;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int src = (lane & ~12) | (b << 2);
        mixed = fmaf(__shfl_sync(0xffffffff, sg, src),
                     __shfl_sync(0xffffffff, xv, src), mixed);
      }
      if (quad == 0) tout[rw][hh] = __float2half_rn(divf(mixed, 4.0f));
    }
  }
  __syncthreads();
  TS(7);
  // push own 8 hidden columns (two 8-byte LL words per 4 halves) to r^1, r^2
  // and the local output
  if (t < 2 * M) {
    const int rw = t >> 1, h4 = (t & 1) * 4, h = 640 * rk + 8 * i + h4;
    const uint2 d = *reinterpret_cast<const uint2*>(&tout[rw][h4]);
    st4(a.hb[p1] + (rw * HD + h) / 2, d.x, d.y, tag);
    st4(a.hb[p2] + (rw * HD + h) / 2, d.x, d.y, tag);
    if (FULL) st4(a.hb[p3] + (rw * HD + h) / 2, d.x, d.y, tag);
    *reinterpret_cast<uint2*>(a.blk_out + rw * HD + h) = d;
  }
  if (FULL) {
    if (t >= 32 && t < 32 + 6 * M) {  // collect the three peer slices
      const int u = t - 32, k = u % 3,
                src = k == 0   ? p1
                      : k == 1 ? p2
                               : p3,
                rw = (u / 3) >> 1;
      const int h = 640 * src + 8 * i + ((u / 3) & 1) * 4;
      const uint2 d = poll4(a.hb[rk] + (rw * HD + h) / 2, tag);
      *reinterpret_cast<uint2*>(a.blk_out + rw * HD + h) = d;
    }
  } else {
    if (t >= 32 && t < 32 + 2 * M) {  // forward r^1's slice to r^2
      const int u = t - 32, rw = u >> 1, h = 640 * p1 + 8 * i + (u & 1) * 4;
      const uint2 d = poll4(a.hb[rk] + (rw * HD + h) / 2, tag);
      st4(a.hb[p2] + (rw * HD + h) / 2, d.x, d.y, tag);
      *reinterpret_cast<uint2*>(a.blk_out + rw * HD + h) = d;
    }
    if (t >= 64 && t < 64 + 4 * M) {  // collect r^2 and r^3 slices
      const int u = t - 64, src = (u & 2) ? p3 : p2, rw = u >> 2,
                h = 640 * src + 8 * i + (u & 1) * 4;
      const uint2 d = poll4(a.hb[rk] + (rw * HD + h) / 2, tag);
      *reinterpret_cast<uint2*>(a.blk_out + rw * HD + h) = d;
    }
  }
  __syncthreads();
  TS(8);
  if constexpr (CONSUMER) consume_output(a, tag, &xs[0][0]);
  if (t == 0) {
    __threadfence();
    if (atomicAdd(a.bar + 1, 1u) == NC - 1) {
      a.bar[0] = 0;
      a.bar[1] = 0;
      __threadfence();
      atomicExch(a.seq, ep + 1);
    }
  }
}
}  // namespace hcxcons

void hcx_consumer_out(
    torch::Tensor p0, std::optional<torch::Tensor> p1, torch::Tensor res,
    torch::Tensor inj, torch::Tensor nw, double eps, torch::Tensor wd,
    torch::Tensor wu, torch::Tensor res_out, torch::Tensor blk_out,
    torch::Tensor inj_out, torch::Tensor xn, torch::Tensor sq,
    torch::Tensor dpart, torch::Tensor bar, torch::Tensor seq,
    std::vector<int64_t> ar, std::vector<int64_t> lora, std::vector<int64_t> hb,
    int64_t rank, std::optional<torch::Tensor> dbg, int64_t full,
    std::optional<torch::Tensor> ox, std::optional<torch::Tensor> ocodes,
    std::optional<torch::Tensor> ohigh, std::optional<torch::Tensor> oscale,
    int64_t ofmt, std::optional<torch::Tensor> gz,
    std::optional<torch::Tensor> gw, double geps,
    std::optional<torch::Tensor> gscr, std::vector<torch::Tensor> codes,
    std::vector<torch::Tensor> high, std::vector<torch::Tensor> scale,
    std::vector<torch::Tensor> outs, std::vector<int64_t> fmts,
    std::vector<int64_t> ns, torch::Tensor cpartial, torch::Tensor cready,
    bool consume) {
  const c10::cuda::CUDAGuard guard(p0.device());
  hcxcons::Args a{};
  a.p0 = reinterpret_cast<const half*>(p0.data_ptr());
  a.p1 = p1 ? reinterpret_cast<const half*>(p1->data_ptr()) : nullptr;
  a.res = reinterpret_cast<const half*>(res.data_ptr());
  a.inj = reinterpret_cast<const half*>(inj.data_ptr());
  a.nw = reinterpret_cast<const half*>(nw.data_ptr());
  a.nw_full = nw.numel() == hcxcons::KD;
  a.eps = static_cast<float>(eps);
  a.wd = reinterpret_cast<const uint4*>(wd.data_ptr());
  a.wu = reinterpret_cast<const uint4*>(wu.data_ptr());
  a.res_out = reinterpret_cast<half*>(res_out.data_ptr());
  a.blk_out = reinterpret_cast<half*>(blk_out.data_ptr());
  a.inj_out = reinterpret_cast<half*>(inj_out.data_ptr());
  a.xn = reinterpret_cast<half*>(xn.data_ptr());
  a.sq = sq.data_ptr<float>();
  a.dpart = dpart.data_ptr<float>();
  a.bar = reinterpret_cast<unsigned*>(bar.data_ptr<int>());
  a.seq = reinterpret_cast<unsigned*>(seq.data_ptr<int>());
  for (int q = 0; q < 4; ++q) {
    a.ar[q] = reinterpret_cast<uint2*>(ar[q]);
    a.lora[q] = reinterpret_cast<uint2*>(lora[q]);
    a.hb[q] = reinterpret_cast<uint2*>(hb[q]);
  }
  a.rank = static_cast<int>(rank);
  a.dbg = dbg ? reinterpret_cast<unsigned long long*>(dbg->data_ptr<int64_t>())
              : nullptr;
  a.M = static_cast<int>(p0.size(0));
  TORCH_CHECK(a.M >= 1 && a.M <= 8, "HCX: M1..8");
  auto st = at::cuda::getCurrentCUDAStream();
  if (ox) {
    a.ox = reinterpret_cast<const half*>(ox->data_ptr());
    a.oldx = static_cast<int>(ox->stride(0));
    TORCH_CHECK(ox->size(1) % 128 == 0 && ox->size(0) == a.M);
    a.oG = static_cast<int>(ox->size(1) / 128);
    a.oseg.codes = reinterpret_cast<const uint4*>(ocodes->data_ptr());
    a.oseg.high = ohigh->numel()
                      ? reinterpret_cast<const uint4*>(ohigh->data_ptr())
                      : nullptr;
    a.oseg.scale = reinterpret_cast<const uint4*>(oscale->data_ptr());
    a.oseg.n = hcxcons::HD;
    a.oseg.fmt = static_cast<int>(ofmt);
  }
  if (gz) {
    TORCH_CHECK(ox && gw && gw->numel() == 128 && gz->stride(1) == 1 &&
                ox->size(1) % 128 == 0);
    a.gz = reinterpret_cast<const half*>(gz->data_ptr());
    a.ldz = static_cast<int>(gz->stride(0));
    a.gw = reinterpret_cast<const half*>(gw->data_ptr());
    a.geps = static_cast<float>(geps);
    TORCH_CHECK(gscr && gscr->numel() >= ox->numel());
    a.gscr = reinterpret_cast<half*>(gscr->data_ptr());
  }

  TORCH_CHECK(!ox, "research consumer has no fused producer");
  TORCH_CHECK(codes.size() > 0 && codes.size() <= 4);
  a.consumer.nseg = static_cast<int>(codes.size());
  int nsum = 0, tiles = 0;
  for (int j = 0; j < a.consumer.nseg; ++j) {
    TORCH_CHECK(fmts[j] == consdmv::Q4K || fmts[j] == consdmv::Q5K ||
                fmts[j] == consdmv::Q6K || fmts[j] == consdmv::LUT4);
    TORCH_CHECK(ns[j] % 32 == 0);
    auto& s = a.consumer.s[j];
    s.codes = reinterpret_cast<const uint4*>(codes[j].data_ptr());
    s.high = high[j].numel()
                 ? reinterpret_cast<const uint4*>(high[j].data_ptr())
                 : nullptr;
    s.scale = reinterpret_cast<const uint4*>(scale[j].data_ptr());
    s.out = reinterpret_cast<half*>(outs[j].data_ptr());
    s.out_ld = outs[j].stride(0);
    s.n = ns[j];
    s.fmt = fmts[j];
    s.tile0 = tiles;
    s.pad = nsum;
    tiles += ns[j] / 32;
    nsum += ns[j];
  }
  a.consumer.main_tiles = tiles;
  a.consumer.hld = nsum;
  TORCH_CHECK(cpartial.numel() >= 20 * a.M * nsum && cready.numel() >= 80);
  a.consumer_partial = cpartial.data_ptr<float>();
  a.consumer_ready = reinterpret_cast<unsigned*>(cready.data_ptr<int>());
  if (consume) {
    if (full)
      hcxcons::hcx_kernel<true, -1, true><<<80, 256, 0, st>>>(a);
    else
      hcxcons::hcx_kernel<false, -1, true><<<80, 256, 0, st>>>(a);
    hcxcons::consumer_reduce<<<(nsum + 255) / 256, 256, 0, st>>>(
        a.consumer, a.consumer_partial, a.M);
  } else {
    if (full)
      hcxcons::hcx_kernel<true, -1, false><<<80, 256, 0, st>>>(a);
    else
      hcxcons::hcx_kernel<false, -1, false><<<80, 256, 0, st>>>(a);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY(round14_hcx_consumer, m) {
  m.def(
      "run(Tensor p0, Tensor? p1, Tensor res, Tensor inj, Tensor nw, float "
      "eps, Tensor wd, Tensor wu, Tensor(a!) res_out, Tensor(b!) blk_out, "
      "Tensor(c!) inj_out, Tensor(d!) xn, Tensor(e!) sq, Tensor(f!) dpart, "
      "Tensor(g!) bar, Tensor(h!) seq, int[] ar, int[] lora, int[] hb, int "
      "rank, Tensor? dbg, int full, Tensor? ox, Tensor? ocodes, Tensor? ohigh, "
      "Tensor? oscale, int ofmt, Tensor? gz, Tensor? gw, float geps, "
      "Tensor(i!)? gscr, Tensor[] codes, Tensor[] high, Tensor[] scale, "
      "Tensor[] outs, int[] fmts, int[] ns, Tensor(j!) cpartial, Tensor(k!) "
      "cready, bool consume) -> ()");
  m.impl("run", torch::kCUDA, &hcx_consumer_out);
}
