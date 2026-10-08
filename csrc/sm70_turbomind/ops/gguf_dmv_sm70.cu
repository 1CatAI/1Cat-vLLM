// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Based on the supplied dmv11.cu shared-activation projection implementation.
// Small-M (M<=8) GGUF dense projections on SM70.
// One launch serves several same-input projections ("segments"). Weights are
// reconstructed to FP16 (scale*code+min, as the canonical TurboMind path) and
// multiplied with mma.m8n8k4 FP32 accumulation. Split-K across CTAs reduces
// deterministically in the same launch (last CTA sums partials in order).
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <vector>
#include <utility>
#ifndef DMV_IQ4_PRMT
  #define DMV_IQ4_PRMT 1
#endif
#ifndef DMV_FULLK
  #define DMV_FULLK 1
#endif
#ifndef DMV_PROBE
  #define DMV_PROBE 0
#endif

namespace {

enum Fmt {
  Q4K = 0,
  Q5K = 1,
  Q6K = 2,
  LUT4 = 3,
  Q8 = 4,
  IQ3S = 5,
  IQ3X = 6,
  LUT6 = 7
};
constexpr bool is_iq3(int f) { return f == IQ3S || f == IQ3X; }
constexpr bool is_iq2(int f) { return f >= 7 && f <= 9; }
constexpr int TAB_VECS =
    192;  // IQ3_S grid (512 words) then IQ3_XXS grid (256 words)
constexpr int MAXSEG = 4;

struct Seg {
  const uint4* codes;
  const uint4* high;
  const uint4* scale;
  half* out;
  long long out_ld;
  int n;
  int fmt;
  int tile0;
  int pad;
};
struct Segs {
  Seg s[MAXSEG];
  int nseg;
  const uint4* tab;  // grid tables for IQ3 formats
  // Optional small FP16 projection (e.g. GDN a/b) computed by one extra CTA.
  const half* ab_w;  // [ab_n, K] row-major
  half* ab_out;      // [M, ab_ld]
  int ab_n;
  int ab_ld;
  int main_tiles;
  int pair;    // segs 0/1 = gate/up of equal n; TN must be 2
  half* hout;  // [M, hld] silu(gate) * up
  int hld;
  bool gdn_heads;
  const half* sgate;  // optional: out = fp16(fp16(acc) * sigmoid(sgate[token]))
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

__device__ __forceinline__ uint32_t lop_or(uint32_t a, uint32_t mask,
                                           uint32_t c) {
  uint32_t r;  // (a & mask) | c
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(mask), "r"(c));
  return r;
}

__device__ __forceinline__ half2 h2(uint32_t v) {
  return *reinterpret_cast<half2*>(&v);
}
__device__ __forceinline__ uint32_t u32(half2 v) {
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ half2 lo2(uint32_t v) {
  return h2((v & 0xffffu) | (v << 16));
}
__device__ __forceinline__ half2 hi2(uint32_t v) {
  return h2((v >> 16) | (v & 0xffff0000u));
}
template <int S>
__device__ __forceinline__ uint32_t shf(uint32_t x) {
  if constexpr (S >= 0)
    return x >> S;
  else
    return x << (-S);
}

template <int FMT>
struct CW {
  static constexpr int v = FMT == Q8 ? 2 : 1;
};

template <int FMT>
struct Ld {
  uint4 c[4][CW<FMT>::v];
  uint4 sc;
  uint4 hi[FMT == Q6K ? 2 : 1];
};

template <int FMT>
__device__ __forceinline__ void load(Ld<FMT>& L, const Seg& s, int t, int g,
                                     int S, int G, int lane) {
  const size_t tg = static_cast<size_t>(t) * G + g;
  if constexpr (is_iq3(FMT)) {
#pragma unroll
    for (int j = 0; j < 3; ++j)
      L.c[j][0] = __ldcs(s.codes + (tg * 3 + j) * 32 + lane);
    if constexpr (FMT == IQ3S) {
      const uint2 m =
          __ldcs(reinterpret_cast<const uint2*>(s.scale) + tg * 32 + lane);
      L.sc.x = m.x;
      L.sc.y = m.y;
    } else {
      L.sc.y = __ldcs(reinterpret_cast<const unsigned int*>(s.scale) + tg * 32 +
                      lane);
    }
    return;
  }
  if constexpr (FMT == LUT4 || FMT == LUT6) {
    const uint2 m =
        __ldcs(reinterpret_cast<const uint2*>(s.scale) + tg * 32 + lane);
    L.sc.x = m.x;
    L.sc.y = m.y;
  } else {
    L.sc = __ldcs(s.scale + tg * 32 + lane);
  }
  if constexpr (FMT == Q5K) L.hi[0] = __ldcs(s.high + tg * 32 + lane);
  if constexpr (FMT == Q6K) {
    L.hi[0] = __ldcs(s.high + (tg * 2) * 32 + lane);
    L.hi[1] = __ldcs(s.high + (tg * 2 + 1) * 32 + lane);
  }
  const size_t Sp = static_cast<size_t>(G) * 4;
#pragma unroll
  for (int st = 0; st < 4; ++st) {
    const int sidx = g * 4 + st;
    if (DMV_FULLK || sidx < S) {
#pragma unroll
      for (int j = 0; j < CW<FMT>::v; ++j)
        L.c[st][j] = __ldcs(
            s.codes +
            ((static_cast<size_t>(t) * Sp + sidx) * CW<FMT>::v + j) * 32 +
            lane);
    }
  }
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// Decode one 32-weight step into 16 half2 in k order.
template <int FMT, bool ExactLattice = false>
__device__ __forceinline__ void decode(const Ld<FMT>& L, int st,
                                       uint32_t (&hw)[16], const half2* lut) {
  const uint32_t MAGIC = 0x64006400u;
  if constexpr (is_iq3(FMT)) {
    const uint32_t* grid =
        reinterpret_cast<const uint32_t*>(lut) + (FMT == IQ3X ? 512 : 0);
    const uint4& I = L.c[st >> 1][0];
    const uint32_t w0 = (st & 1) ? I.z : I.x, w1 = (st & 1) ? I.w : I.y;
    // Sign word: bit p = weight 2p, bit 16+p = weight 2p+1 of this step.
    const uint32_t sg = word(L.c[2][0], st);
    const uint32_t dsw = L.sc.y;
    const uint32_t hs = FMT == IQ3S ? ((L.sc.x >> (8 * st)) & 255u) << 8 : 0u;
    const uint32_t nib = (dsw >> (16 + 4 * st)) & 15u;
    const float sf =
        __half2float(
            __ushort_as_half(static_cast<unsigned short>(dsw & 0xffffu))) *
        static_cast<float>(1 + 2 * nib) * (FMT == IQ3X ? 0.25f : 1.0f);
    const half2 s2 = __float2half2_rn(sf);
    const half2 nb2 = __float2half2_rn(-1024.0f * __half2float(__low2half(s2)));
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      uint32_t idx;
      asm("prmt.b32 %0, %1, 0, %2;"
          : "=r"(idx)
          : "r"(q < 4 ? w0 : w1), "r"(0x4440u | (q & 3)));
      if constexpr (FMT == IQ3S) idx = lop_or(hs >> q, 0x100u, idx);
#if DMV_PROBE == 2
      const uint32_t gw = idx * 0x01010101u;
#else
      const uint32_t gw = grid[idx];
#endif
      uint32_t a, b;
      asm("prmt.b32 %0, %1, %2, 0x4140;" : "=r"(a) : "r"(gw), "r"(0x64646464u));
      asm("prmt.b32 %0, %1, %2, 0x4342;" : "=r"(b) : "r"(gw), "r"(0x64646464u));
      uint32_t va, vb;
      if constexpr (ExactLattice) {
        // Preserve the original reader's one final operand rounding. The
        // integer grid times the local scale is exactly representable in half.
        const half2 small = __float2half2_rn(static_cast<float>(1 + 2 * nib) *
                                             (FMT == IQ3X ? 0.25f : 1.0f));
        const half2 original_d = lo2(dsw);
        va =
            u32(__hmul2(__hmul2(__hsub2(h2(a), h2(MAGIC)), small), original_d));
        vb =
            u32(__hmul2(__hmul2(__hsub2(h2(b), h2(MAGIC)), small), original_d));
      } else {
        va = u32(__hfma2(h2(a), s2, nb2));
        vb = u32(__hfma2(h2(b), s2, nb2));
      }
      uint32_t ra, rb;  // r = v ^ ((sg << (15 - p)) & 0x80008000)
      asm("lop3.b32 %0, %1, %2, %3, 0x78;"
          : "=r"(ra)
          : "r"(va), "r"(sg << (15 - 2 * q)), "r"(0x80008000u));
      asm("lop3.b32 %0, %1, %2, %3, 0x78;"
          : "=r"(rb)
          : "r"(vb), "r"(sg << (14 - 2 * q)), "r"(0x80008000u));
      hw[2 * q] = ra;
      hw[2 * q + 1] = rb;
    }
    return;
  }
  const uint32_t sc = FMT == LUT6 ? 0 : word(L.sc, st);
  if constexpr (FMT == Q4K || FMT == Q5K) {
    const half2 s2 = lo2(sc), m2 = hi2(sc);
    uint32_t hb = 0;
    if constexpr (FMT == Q5K) hb = word(L.hi[0], st);
    const uint32_t cr[4] = {L.c[st][0].x, L.c[st][0].y, L.c[st][0].z,
                            L.c[st][0].w};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t extra = MAGIC;
        if constexpr (FMT == Q5K) {
          uint32_t sh;
          const int s = 4 * c + j - 4;
          sh = s >= 0 ? (hb >> (s >= 0 ? s : 0)) : (hb << (s < 0 ? -s : 0));
          extra = lop_or(sh, 0x00100010u, MAGIC);
        }
        const uint32_t h = lop_or(cr[c] >> (4 * j), 0x000f000fu, extra);
        hw[4 * c + j] = u32(__hfma2(__hsub2(h2(h), h2(MAGIC)), s2, m2));
      }
    }
  } else if constexpr (FMT == Q6K) {
    const half2 s0 = lo2(sc), s1 = hi2(sc);
    const uint4& hv = L.hi[st >> 1];
    const uint32_t hx = (st & 1) ? hv.z : hv.x;
    const uint32_t hy = (st & 1) ? hv.w : hv.y;
    const uint32_t cr[4] = {L.c[st][0].x, L.c[st][0].y, L.c[st][0].z,
                            L.c[st][0].w};
    const half2 bias = h2(0x64206420u);  // 1056 = 1024 + 32
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint32_t hb = c < 2 ? hx : hy;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int s = 2 * (4 * (c & 1) + j) - 4;
        const uint32_t sh =
            s >= 0 ? (hb >> (s >= 0 ? s : 0)) : (hb << (s < 0 ? -s : 0));
        const uint32_t extra = lop_or(sh, 0x00300030u, MAGIC);
        const uint32_t h = lop_or(cr[c] >> (4 * j), 0x000f000fu, extra);
        hw[4 * c + j] = u32(__hmul2(__hsub2(h2(h), bias), c < 2 ? s0 : s1));
      }
    }
  } else if constexpr (FMT == LUT4 || FMT == LUT6) {
    // compact plane: word0 = s0 | s1 << 16, word1 = s2 | s3 << 16
    const uint32_t sw = (st & 2) ? L.sc.y : L.sc.x;
    uint32_t s2u;
    asm("prmt.b32 %0, %1, 0, %2;"
        : "=r"(s2u)
        : "r"(sw), "r"((st & 1) ? 0x3232u : 0x1010u));
    const half2 s4 = h2(s2u);
    const uint32_t cr[4] = {L.c[st][0].x, L.c[st][0].y, L.c[st][0].z,
                            L.c[st][0].w};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const unsigned nibble = (L.sc.y >> (8 * st + (c < 2 ? 0 : 4))) & 15u;
      const half2 s2 =
          FMT == LUT6 ? __float2half2_rn((0.5f + nibble) * 0.25f) : s4;
#if DMV_IQ4_PRMT
      // kIQ4 + 128 as bytes; nibble bit 3 picks the upper table half.
      const uint32_t T0 = FMT == LUT6 ? 0x88786755u : 0x3F2D1801u;
      const uint32_t T1 = FMT == LUT6 ? 0x8080ab99u : 0x766A5D4Fu;
      const uint32_t T2 = 0xA6998D81u, T3 = 0xF1D9C5B5u;
      const half2 b1152 = h2(0x64806480u);
      const uint32_t q = cr[c];
      const uint32_t qm = q & 0x77777777u;
      uint32_t sel;
      asm("lop3.b32 %0, %1, %2, %3, 0xEA;"
          : "=r"(sel)
          : "r"(q >> 1), "r"(0x44444444u), "r"(0x32103210u));
  #pragma unroll
      for (int i = 0; i < 2; ++i) {
        uint32_t lo, hi, v, a, b;
        asm("prmt.b32 %0, %1, %2, %3;"
            : "=r"(lo)
            : "r"(T0), "r"(T1), "r"(qm >> (16 * i)));
        if constexpr (FMT == LUT6) {
          v = lo;
        } else {
          asm("prmt.b32 %0, %1, %2, %3;"
              : "=r"(hi)
              : "r"(T2), "r"(T3), "r"(qm >> (16 * i)));
          asm("prmt.b32 %0, %1, %2, %3;"
              : "=r"(v)
              : "r"(lo), "r"(hi), "r"(sel >> (16 * i)));
        }
        asm("prmt.b32 %0, %1, %2, 0x4140;"
            : "=r"(a)
            : "r"(v), "r"(0x64646464u));
        asm("prmt.b32 %0, %1, %2, 0x4342;"
            : "=r"(b)
            : "r"(v), "r"(0x64646464u));
        half2 va = __hmul2(__hsub2(h2(a), b1152), s2);
        half2 vb = __hmul2(__hsub2(h2(b), b1152), s2);
        if constexpr (FMT == LUT6) {
          // small*grid is exactly representable in FP16 (max1333/8).
          // Retain original d; only the final multiply rounds a weight.
          va = __hmul2(va, lo2(L.sc.x));
          vb = __hmul2(vb, lo2(L.sc.x));
        }
        hw[4 * c + 2 * i] = u32(va);
        hw[4 * c + 2 * i + 1] = u32(vb);
      }
#else
  #pragma unroll
      for (int b = 0; b < 4; ++b)
        hw[4 * c + b] = u32(__hmul2(lut[(cr[c] >> (8 * b)) & 0xff], s2));
#endif
    }
  } else {  // Q8: int8 codes, (x ^ 0x80) + 1024 trick, minus 1152.
    const half2 s2 = lo2(sc);
    const half2 bias = h2(0x64806480u);  // 1152
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint32_t v = word(L.c[st][c >> 2], c & 3) ^ 0x80808080u;
      uint32_t a, b;
      asm("prmt.b32 %0, %1, %2, 0x4140;" : "=r"(a) : "r"(v), "r"(0x64646464u));
      asm("prmt.b32 %0, %1, %2, 0x4342;" : "=r"(b) : "r"(v), "r"(0x64646464u));
      hw[2 * c] = u32(__hmul2(__hsub2(h2(a), bias), s2));
      hw[2 * c + 1] = u32(__hmul2(__hsub2(h2(b), bias), s2));
    }
  }
}

__device__ __constant__ int8_t kIQ4[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

template <int FMT, int KW, int TN, bool LegacyIq2 = false>
__device__ __forceinline__ void body6(
    const Seg& sg, int t, bool on, int kslot, int pid, int g0, int g1, int S,
    int G, const half* __restrict__ x, int ldx, int M, uint4* xs, half2* lut,
    float (&acc)[8], const uint4* __restrict__ tab, bool gdn_heads, bool pair) {
  constexpr int NT = 32 * KW * TN, PARTS = 4 / TN;
  const int lane = threadIdx.x % 32;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  Ld<FMT> A;
  int g = g0 + kslot;
  int end = g1, stride = KW;
  if constexpr (LegacyIq2) {
    if (pair) {
      const int part = (g1 - g0) / KW;
      g = g0 + kslot * part;
      end = g + part;
      stride = 1;
    }
  }
  uint4* slot = xs + kslot * 256;
  uint4 X[PARTS];
  // Three value heads per key head: TP4 has four groups, TP2 has eight.
  const int head_shift = G == 24 ? 3 : 2;
  const int head_mask = (1 << head_shift) - 1;
  auto xload = [&](int gg) {
#pragma unroll
    for (int jj = 0; jj < PARTS; ++jj) {
      const int idx = lane + 32 * (pid + TN * jj), c = idx >> 3, row = idx & 7;
      X[jj] = make_uint4(0, 0, 0, 0);
      if (row < M)
        X[jj] = __ldg(reinterpret_cast<const uint4*>(
            x + row * ldx +
            (gdn_heads ? (gg & head_mask) * 3 + (gg >> head_shift) : gg) * 128 +
            c * 8));
    }
  };
  auto xstore = [&](int b) {
#pragma unroll
    for (int jj = 0; jj < PARTS; ++jj)
      slot[b * 128 + lane + 32 * (pid + TN * jj)] = X[jj];
  };
  auto xsync = [&]() {
    if constexpr (TN > 1)
      asm volatile("bar.sync %0, %1;" ::"r"(kslot + 1), "r"(TN * 32)
                   : "memory");
    else
      __syncwarp();
  };
  if (g < end) {
    if (on) load<FMT>(A, sg, t, g, S, G, lane);
    xload(g);
  }
  static_assert(DMV_IQ4_PRMT,
                "pair mode needs the shared-memory-free IQ4 decode");
  if (tab != nullptr) {
    uint4* t4 = reinterpret_cast<uint4*>(lut);
    for (int i = threadIdx.x; i < TAB_VECS; i += NT) t4[i] = __ldg(tab + i);
  }
  __syncthreads();
  if (g >= end) return;
  xstore(0);
  xsync();
  int cur = 0;
  while (true) {
    Ld<FMT> B;
    const int gn = g + stride;
    const bool more = gn < end;
    if (more) {
      if (on) load<FMT>(B, sg, t, gn, S, G, lane);
      xload(gn);
    }
    if (on) {
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        const uint4* xp = slot + cur * 128 + (st * 4) * 8 + r;
        uint4 xa[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) xa[j] = xp[j * 8];
        uint32_t hw[16];
        decode<FMT, LegacyIq2>(A, st, hw, lut);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          mma(acc, xa[j].x, xa[j].y, hw[4 * j], hw[4 * j + 1]);
          mma(acc, xa[j].z, xa[j].w, hw[4 * j + 2], hw[4 * j + 3]);
        }
      }
    }
    if (!more) break;
    cur ^= 1;
    xstore(cur);
    xsync();
    A = B;
    g = gn;
  }
}

__device__ __forceinline__ void write_out(const Seg& sg, int t, int v,
                                          float val, int M,
                                          const half* sgate = nullptr) {
  const int lv = v >> 3, i = v & 7;
  const int token = (i & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
  const int col = (i & 1) | (((lv >> 1) & 1) << 1) | ((i >> 2) << 2);
  const int row = t * 32 + ((lv >> 2) & 3) * 8 + col;
  if (token < M && row < sg.n) {
    if (sgate != nullptr) {
      const float gate =
          1.0f / (1.0f + __expf(-__half2float(__ldcg(sgate + token))));
      val = __half2float(__float2half_rn(val)) * gate;
    }
    sg.out[token * sg.out_ld + row] = __float2half_rn(val);
  }
}

// Extra CTAs: one warp per row, out[m, r] = sum_k x[m, k] * w[r, k] in FP32.
__device__ __forceinline__ void ab_row(const Segs& segs,
                                       const half* __restrict__ x, int ldx,
                                       int M, int K, int row) {
  const int lane = threadIdx.x % 32;
  float acc[8] = {};
  for (int k = lane * 8; k < K; k += 256) {
    const uint4 wv =
        __ldcs(reinterpret_cast<const uint4*>(segs.ab_w + (size_t)row * K + k));
    const half2* wh = reinterpret_cast<const half2*>(&wv);
    float2 wf[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wf[j] = __half22float2(wh[j]);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      if (m < M) {
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(x + m * ldx + k));
        const half2* h = reinterpret_cast<const half2*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __half22float2(h[j]);
          acc[m] = fmaf(f.x, wf[j].x, acc[m]);
          acc[m] = fmaf(f.y, wf[j].y, acc[m]);
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < 8; ++m) {
    float v = acc[m];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0 && m < M)
      segs.ab_out[(size_t)m * segs.ab_ld + row] = __float2half_rn(v);
  }
}

// A separate diagnostic specialization records one timestamp pair per warp.
// The ordinary specialization contains no timer instructions or stores.
template <bool Enabled>
struct WarpTimer {
  __device__ __forceinline__ WarpTimer(uint64_t*, int) {}
};

template <>
struct WarpTimer<true> {
  uint64_t* slot;
  __device__ __forceinline__ WarpTimer(uint64_t* timestamps, int warps)
      : slot(nullptr) {
    if (threadIdx.x % 32 == 0) {
      const size_t block =
          static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
      slot = timestamps + (block * warps + threadIdx.x / 32) * 2;
      uint64_t now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)::"memory");
      slot[0] = now;
    }
  }
  __device__ __forceinline__ ~WarpTimer() {
    if (slot != nullptr) {
      uint64_t now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)::"memory");
      slot[1] = now;
    }
  }
};

template <int KW, int TN, int FA, int FB, int FC = -1, bool Clocked = false>
__device__ __forceinline__ void dense_tile(const Segs& segs,
                                           const half* __restrict__ x, int ldx,
                                           int M, int K, int S, int G,
                                           int split, float* ws, int* cnt,
                                           uint64_t* timestamps, int tg,
                                           int sp) {
  constexpr int W = KW * TN;
  extern __shared__ uint4 smem[];
  __shared__ int last;
  if (segs.ab_n > 0 && tg >= segs.main_tiles) {
    const int row = (tg - segs.main_tiles) * W + threadIdx.x / 32;
    if (sp == 0 && row < segs.ab_n) ab_row(segs, x, ldx, M, K, row);
    return;
  }
  int si = 0;
#pragma unroll
  for (int i = 1; i < MAXSEG; ++i)
    if (i < segs.nseg && tg >= segs.s[i].tile0) si = i;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int tin = warp % TN, kslot = warp / TN;
  const bool pair = segs.pair != 0;
  constexpr int PP = TN / 2 > 0 ? TN / 2 : 1;
  const Seg& sg = pair ? segs.s[tin / PP] : segs.s[si];
  const int t = pair ? tg * PP + tin % PP : (tg - sg.tile0) * TN + tin;
  const bool on = t * 32 < sg.n;
  const int gps = (G + split - 1) / split;
  const int g0 = min(G, sp * gps), g1 = min(G, g0 + gps);
  uint4* xs = smem;
  half2* lut = reinterpret_cast<half2*>(smem + KW * 256);
  float acc[8] = {};
  constexpr bool legacy_iq2 = KW == 8 && TN == 2 && (FA == LUT6 || FB == LUT6);
  if (FA == FB || sg.fmt == FA)
    body6<FA, KW, TN, legacy_iq2>(sg, t, on, kslot, tin, g0, warp < W ? g1 : 0,
                                  S, G, x, ldx, M, xs, lut, acc, segs.tab,
                                  segs.gdn_heads, pair);
  else if (FC < 0 || sg.fmt == FB)
    body6<FB, KW, TN, legacy_iq2>(sg, t, on, kslot, tin, g0, warp < W ? g1 : 0,
                                  S, G, x, ldx, M, xs, lut, acc, segs.tab,
                                  segs.gdn_heads, pair);
  else {
    if constexpr (FC >= 0)
      body6<FC, KW, TN, false>(sg, t, on, kslot, tin, g0, warp < W ? g1 : 0, S,
                               G, x, ldx, M, xs, lut, acc, segs.tab,
                               segs.gdn_heads, pair);
  }
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int e = 0; e < 8; ++e) red[warp * 256 + lane * 8 + e] = acc[e];
  __syncthreads();
  if (pair) {
    auto write_h = [&](int v, float g, float u) {
      const int e = v & 255, pt = tg * PP + (v >> 8);
      const int lv = e >> 3, i = e & 7;
      const int token = (i & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
      const int col = (i & 1) | (((lv >> 1) & 1) << 1) | ((i >> 2) << 2);
      const int row = pt * 32 + ((lv >> 2) & 3) * 8 + col;
      if (token < M && row < segs.s[0].n) {
        const float gf = __half2float(__float2half_rn(g));
        const float uf = __half2float(__float2half_rn(u));
        if constexpr (legacy_iq2) {
          const half activated = __float2half_rn(gf / (1.0f + expf(-gf)));
          segs.hout[(size_t)token * segs.hld + row] =
              __hmul(activated, __float2half_rn(uf));
        } else {
          segs.hout[(size_t)token * segs.hld + row] =
              __float2half_rn(gf / (1.0f + __expf(-gf)) * uf);
        }
      }
    };
    for (int v = threadIdx.x; v < PP * 256; v += 32 * W) {
      const int pt = v >> 8, e = v & 255;
      float g = 0.f, u = 0.f;
#pragma unroll
      for (int k = 0; k < KW; ++k) {
        g += red[(k * TN + pt) * 256 + e];
        u += red[(k * TN + PP + pt) * 256 + e];
      }
      if (split == 1)
        write_h(v, g, u);
      else {
        __stcg(ws + (static_cast<size_t>(tg) * split + sp) * PP * 512 + v, g);
        __stcg(ws + (static_cast<size_t>(tg) * split + sp) * PP * 512 +
                   PP * 256 + v,
               u);
      }
    }
    if (split == 1) return;
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      last = atomicAdd(cnt + tg, 1) == split - 1;
      if (last) __threadfence();
    }
    __syncthreads();
    if (!last) return;
    for (int v = threadIdx.x; v < PP * 256; v += 32 * W) {
      float g = 0.f, u = 0.f;
      for (int p = 0; p < split; ++p) {
        g += __ldcg(ws + (static_cast<size_t>(tg) * split + p) * PP * 512 + v);
        u += __ldcg(ws + (static_cast<size_t>(tg) * split + p) * PP * 512 +
                    PP * 256 + v);
      }
      write_h(v, g, u);
    }
    if (threadIdx.x == 0) cnt[tg] = 0;
    return;
  }
  const int tbase = (tg - sg.tile0) * TN;
  for (int v = threadIdx.x; v < TN * 256; v += 32 * W) {
    const int i = v / 256, e = v % 256;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < KW; ++k) s += red[(k * TN + i) * 256 + e];
    if (split == 1)
      write_out(sg, tbase + i, e, s, M, segs.sgate);
    else
      __stcg(ws + (static_cast<size_t>(tg) * split + sp) * TN * 256 + v, s);
  }
  if (split == 1) return;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(cnt + tg, 1) == split - 1;
    if (last) __threadfence();
  }
  __syncthreads();
  if (!last) return;
  for (int v = threadIdx.x; v < TN * 256; v += 32 * W) {
    float pv[8];
#pragma unroll
    for (int p = 0; p < 8; ++p)
      pv[p] = p < split
                  ? __ldcg(ws +
                           (static_cast<size_t>(tg) * split + p) * TN * 256 + v)
                  : 0.f;
    float s = 0.f;
#pragma unroll
    for (int p = 0; p < 8; ++p) s += pv[p];
    write_out(sg, tbase + v / 256, v % 256, s, M, segs.sgate);
  }
  if (threadIdx.x == 0) cnt[tg] = 0;
}

template <int KW, int TN, int FA, int FB, int FC = -1, bool Clocked = false>
__global__ void __launch_bounds__(32 * KW * TN)
    dense_mv(Segs segs, const half* __restrict__ x, int ldx, int M, int K,
             int S, int G, int split, float* ws, int* cnt,
             uint64_t* timestamps) {
  constexpr int W = KW * TN;
  WarpTimer<Clocked> timer(timestamps, W);
  extern __shared__ uint4 smem[];
  __shared__ int last;
  if (segs.ab_n > 0 && blockIdx.x >= segs.main_tiles) {
    const int row = (blockIdx.x - segs.main_tiles) * W + threadIdx.x / 32;
    if (blockIdx.y == 0 && row < segs.ab_n) ab_row(segs, x, ldx, M, K, row);
    return;
  }
  const int tg = blockIdx.x, sp = blockIdx.y;
  int si = 0;
#pragma unroll
  for (int i = 1; i < MAXSEG; ++i)
    if (i < segs.nseg && tg >= segs.s[i].tile0) si = i;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int tin = warp % TN, kslot = warp / TN;
  const bool pair = segs.pair != 0;
  constexpr int PP = TN / 2 > 0 ? TN / 2 : 1;
  const Seg& sg = pair ? segs.s[tin / PP] : segs.s[si];
  const int t = pair ? tg * PP + tin % PP : (tg - sg.tile0) * TN + tin;
  const bool on = t * 32 < sg.n;
  const int gps = (G + split - 1) / split;
  const int g0 = min(G, sp * gps), g1 = min(G, g0 + gps);
  uint4* xs = smem;
  half2* lut = reinterpret_cast<half2*>(smem + KW * 256);
  float acc[8] = {};
  constexpr bool legacy_iq2 = KW == 8 && TN == 2 && (FA == LUT6 || FB == LUT6);
  if (FA == FB || sg.fmt == FA)
    body6<FA, KW, TN, legacy_iq2>(sg, t, on, kslot, tin, g0, g1, S, G, x, ldx,
                                  M, xs, lut, acc, segs.tab, segs.gdn_heads,
                                  pair);
  else if (FC < 0 || sg.fmt == FB)
    body6<FB, KW, TN, legacy_iq2>(sg, t, on, kslot, tin, g0, g1, S, G, x, ldx,
                                  M, xs, lut, acc, segs.tab, segs.gdn_heads,
                                  pair);
  else {
    if constexpr (FC >= 0)
      body6<FC, KW, TN, false>(sg, t, on, kslot, tin, g0, g1, S, G, x, ldx, M,
                               xs, lut, acc, segs.tab, segs.gdn_heads, pair);
  }
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int e = 0; e < 8; ++e) red[warp * 256 + lane * 8 + e] = acc[e];
  __syncthreads();
  if (pair) {
    auto write_h = [&](int v, float g, float u) {
      const int e = v & 255, pt = tg * PP + (v >> 8);
      const int lv = e >> 3, i = e & 7;
      const int token = (i & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
      const int col = (i & 1) | (((lv >> 1) & 1) << 1) | ((i >> 2) << 2);
      const int row = pt * 32 + ((lv >> 2) & 3) * 8 + col;
      if (token < M && row < segs.s[0].n) {
        const float gf = __half2float(__float2half_rn(g));
        const float uf = __half2float(__float2half_rn(u));
        if constexpr (legacy_iq2) {
          const half activated = __float2half_rn(gf / (1.0f + expf(-gf)));
          segs.hout[(size_t)token * segs.hld + row] =
              __hmul(activated, __float2half_rn(uf));
        } else {
          segs.hout[(size_t)token * segs.hld + row] =
              __float2half_rn(gf / (1.0f + __expf(-gf)) * uf);
        }
      }
    };
    for (int v = threadIdx.x; v < PP * 256; v += 32 * W) {
      const int pt = v >> 8, e = v & 255;
      float g = 0.f, u = 0.f;
#pragma unroll
      for (int k = 0; k < KW; ++k) {
        g += red[(k * TN + pt) * 256 + e];
        u += red[(k * TN + PP + pt) * 256 + e];
      }
      if (split == 1)
        write_h(v, g, u);
      else {
        __stcg(ws + (static_cast<size_t>(tg) * split + sp) * PP * 512 + v, g);
        __stcg(ws + (static_cast<size_t>(tg) * split + sp) * PP * 512 +
                   PP * 256 + v,
               u);
      }
    }
    if (split == 1) return;
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      last = atomicAdd(cnt + tg, 1) == split - 1;
      if (last) __threadfence();
    }
    __syncthreads();
    if (!last) return;
    for (int v = threadIdx.x; v < PP * 256; v += 32 * W) {
      float g = 0.f, u = 0.f;
      for (int p = 0; p < split; ++p) {
        g += __ldcg(ws + (static_cast<size_t>(tg) * split + p) * PP * 512 + v);
        u += __ldcg(ws + (static_cast<size_t>(tg) * split + p) * PP * 512 +
                    PP * 256 + v);
      }
      write_h(v, g, u);
    }
    if (threadIdx.x == 0) cnt[tg] = 0;
    return;
  }
  const int tbase = (tg - sg.tile0) * TN;
  for (int v = threadIdx.x; v < TN * 256; v += 32 * W) {
    const int i = v / 256, e = v % 256;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < KW; ++k) s += red[(k * TN + i) * 256 + e];
    if (split == 1)
      write_out(sg, tbase + i, e, s, M, segs.sgate);
    else
      __stcg(ws + (static_cast<size_t>(tg) * split + sp) * TN * 256 + v, s);
  }
  if (split == 1) return;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(cnt + tg, 1) == split - 1;
    if (last) __threadfence();
  }
  __syncthreads();
  if (!last) return;
  for (int v = threadIdx.x; v < TN * 256; v += 32 * W) {
    float pv[8];
#pragma unroll
    for (int p = 0; p < 8; ++p)
      pv[p] = p < split
                  ? __ldcg(ws +
                           (static_cast<size_t>(tg) * split + p) * TN * 256 + v)
                  : 0.f;
    float s = 0.f;
#pragma unroll
    for (int p = 0; p < 8; ++p) s += pv[p];
    write_out(sg, tbase + v / 256, v % 256, s, M, segs.sgate);
  }
  if (threadIdx.x == 0) cnt[tg] = 0;
}

template <int KW, int TN, int FA, int FB, int FC = -1, bool Clocked = false>
void launch(const Segs& segs, const half* x, int ldx, int M, int K, int S,
            int G, int split, float* ws, int* cnt, int tiles, cudaStream_t st,
            uint64_t* timestamps) {
  constexpr int W = KW * TN;
  const size_t xs_bytes = static_cast<size_t>(KW) * 256 * 16 + TAB_VECS * 16;
  const size_t red_bytes = static_cast<size_t>(W) * 256 * 4;
  const size_t smem = xs_bytes > red_bytes ? xs_bytes : red_bytes;
  static bool init = false;
  if (!init) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        dense_mv<KW, TN, FA, FB, FC, Clocked>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 90 * 1024));
    init = true;
  }
  dense_mv<KW, TN, FA, FB, FC, Clocked>
      <<<dim3(tiles, split), 32 * W, smem, st>>>(segs, x, ldx, M, K, S, G,
                                                 split, ws, cnt, timestamps);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "dense_mv launch: ", cudaGetErrorString(e));
}

}  // namespace

#include "gguf_projection_collective_pipeline_sm70.cuh"

// codes/high/scale: per segment planes; out: per segment [M, n] views.
template <bool Clocked>
void gguf_dmv_dispatch(torch::Tensor x, std::vector<torch::Tensor> codes,
                       std::vector<torch::Tensor> high,
                       std::vector<torch::Tensor> scale,
                       std::vector<torch::Tensor> out, std::vector<int64_t> fmt,
                       std::vector<int64_t> n, int64_t K, int64_t split,
                       int64_t warps, torch::Tensor ws, torch::Tensor cnt,
                       int64_t tp,
                       std::optional<torch::Tensor> sgate = std::nullopt,
                       std::optional<torch::Tensor> tab = std::nullopt,
                       std::optional<torch::Tensor> ab_w = std::nullopt,
                       std::optional<torch::Tensor> ab_out = std::nullopt,
                       std::optional<torch::Tensor> pair_out = std::nullopt,
                       bool gdn_heads = false,
                       torch::Tensor clock_data = torch::Tensor{}) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
                  x.stride(1) == 1 && x.size(0) == 8 && x.size(1) == K,
              "DMV requires eight FP16 rows with contiguous K");
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(
      K > 0 && K % 128 == 0 && split >= 1 && split <= 8 && split <= K / 128,
      "invalid DMV split geometry");
  TORCH_CHECK(!codes.empty() && codes.size() <= MAXSEG &&
                  high.size() == codes.size() && scale.size() == codes.size() &&
                  out.size() == codes.size() && fmt.size() == codes.size() &&
                  n.size() == codes.size(),
              "inconsistent DMV segments");
  auto storage = [&](const torch::Tensor& v, at::ScalarType dtype) {
    TORCH_CHECK(v.device() == x.device() && v.scalar_type() == dtype &&
                    v.is_contiguous(),
                "invalid DMV storage");
  };
  storage(ws, at::kFloat);
  storage(cnt, at::kInt);
  TORCH_CHECK(tp > 0, "invalid DMV N tile count");
  const int64_t groups = K / 128;
  bool needs_table = false;
  int64_t main_tiles = 0;
  for (size_t i = 0; i < codes.size(); ++i) {
    TORCH_CHECK(
        fmt[i] == Q4K || fmt[i] == LUT4 || is_iq2(fmt[i]) || is_iq3(fmt[i]),
        "unqualified DMV format");
    TORCH_CHECK(n[i] > 0 && n[i] % 32 == 0, "DMV N must align to 32");
    storage(codes[i], at::kByte);
    storage(scale[i], at::kByte);
    storage(high[i], at::kByte);
    const int64_t tiles = n[i] / 32;
    TORCH_CHECK(
        codes[i].numel() == tiles * groups * 512 * (is_iq3(fmt[i]) ? 3 : 4) &&
            high[i].numel() == (is_iq2(fmt[i]) ? 2 * 6561 : 0) &&
            scale[i].numel() == tiles * groups *
                                    (fmt[i] == IQ3X  ? 128
                                     : fmt[i] == Q4K ? 512
                                                     : 256),
        "DMV plane size mismatch");
    TORCH_CHECK(out[i].device() == x.device() &&
                    out[i].scalar_type() == at::kHalf && out[i].dim() == 2 &&
                    out[i].size(0) == 8 && out[i].size(1) == n[i] &&
                    out[i].stride(1) == 1,
                "invalid DMV output view");
    main_tiles += (n[i] + 32 * tp - 1) / (32 * tp);
    needs_table |= is_iq3(fmt[i]);
  }
  TORCH_CHECK((warps == 4 && (tp == 1 || tp == 2 || tp == 4)) ||
                  (tp == 2 && (warps == 2 || warps == 6 || warps == 8)) ||
                  (tp == 4 && warps == 2),
              "unqualified DMV tile configuration");
  if (needs_table) {
    TORCH_CHECK(tab.has_value(), "IQ3 codebook is missing");
    storage(*tab, at::kByte);
    TORCH_CHECK(tab->numel() == TAB_VECS * 16, "invalid IQ3 codebook");
  }
  if (pair_out && pair_out->numel()) {
    TORCH_CHECK(codes.size() == 2 && n[0] == n[1] && tp % 2 == 0 &&
                    (n[0] / 32) % (tp / 2) == 0,
                "invalid DMV pair geometry");
    storage(*pair_out, at::kHalf);
    TORCH_CHECK(pair_out->sizes() == out[0].sizes(), "invalid DMV pair output");
    main_tiles = n[0] / 32 / (tp / 2);
  }
  TORCH_CHECK(
      ws.numel() >= main_tiles * split * tp * 256 && cnt.numel() >= main_tiles,
      "DMV scratch is too small");
  TORCH_CHECK(!sgate.has_value(), "DMV sigmoid gate is not qualified");
  if (ab_w && ab_w->numel()) {
    TORCH_CHECK(ab_out.has_value(), "floating output is missing");
    storage(*ab_w, at::kHalf);
    TORCH_CHECK(ab_w->dim() == 2 && ab_w->size(1) == K &&
                    ab_out->device() == x.device() &&
                    ab_out->scalar_type() == at::kHalf && ab_out->dim() == 2 &&
                    ab_out->size(0) == 8 && ab_out->size(1) == ab_w->size(0) &&
                    ab_out->stride(1) == 1,
                "invalid DMV floating operands");
  }
  TORCH_CHECK(!gdn_heads || K == 1536 || K == 3072,
              "GDN head tiling requires K1536 or K3072");
  Segs segs{};
  segs.gdn_heads = gdn_heads;
  if (pair_out && pair_out->numel()) {
    segs.pair = 1;
    segs.hout = reinterpret_cast<half*>(pair_out->data_ptr());
    segs.hld = static_cast<int>(pair_out->stride(0));
  }
  if (ab_w && ab_w->numel()) {
    segs.ab_w = reinterpret_cast<const half*>(ab_w->data_ptr());
    segs.ab_out = reinterpret_cast<half*>(ab_out->data_ptr());
    segs.ab_n = static_cast<int>(ab_w->size(0));
    segs.ab_ld = static_cast<int>(ab_out->stride(0));
  }
  segs.tab = tab && tab->numel()
                 ? reinterpret_cast<const uint4*>(tab->data_ptr())
                 : nullptr;
  segs.sgate = sgate && sgate->numel()
                   ? reinterpret_cast<const half*>(sgate->data_ptr())
                   : nullptr;
  segs.nseg = static_cast<int>(codes.size());
  TORCH_CHECK(segs.nseg <= MAXSEG);
  int tiles = 0;
  for (int i = 0; i < segs.nseg; ++i) {
    Seg& s = segs.s[i];
    s.codes = reinterpret_cast<const uint4*>(codes[i].data_ptr());
    s.high = high[i].numel()
                 ? reinterpret_cast<const uint4*>(high[i].data_ptr())
                 : nullptr;
    s.scale = reinterpret_cast<const uint4*>(scale[i].data_ptr());
    s.out = reinterpret_cast<half*>(out[i].data_ptr());
    s.out_ld = out[i].stride(0);
    s.n = static_cast<int>(n[i]);
    s.fmt = is_iq2(fmt[i]) ? LUT6 : static_cast<int>(fmt[i]);
    s.tile0 = tiles;
    tiles += (s.n + 32 * tp - 1) / (32 * tp);
  }
  const int M = static_cast<int>(x.size(0));
  TORCH_CHECK(M <= 8);
  TORCH_CHECK(!DMV_FULLK || K % 128 == 0, "K must be a multiple of 128");
  const int S = static_cast<int>((K + 31) / 32);
  const int G = (S + 3) / 4;
  auto st = at::cuda::getCurrentCUDAStream();
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
  const int ldx = static_cast<int>(x.stride(0));
  float* wsp = ws.data_ptr<float>();
  int* cp = cnt.data_ptr<int>();
  TORCH_CHECK(split <= 8);
  if (segs.pair) {
    TORCH_CHECK(segs.nseg == 2 && n[0] == n[1] && tp % 2 == 0,
                "pair mode: gate/up of equal n, even tp");
    TORCH_CHECK((n[0] / 32) % (tp / 2) == 0,
                "pair mode: tiles must divide by tp/2");
    tiles = static_cast<int>(n[0]) / 32 / (tp / 2);
    if (warps == 8 && (is_iq2(fmt[0]) || is_iq2(fmt[1])))
      TORCH_CHECK(split == 1 && (K / 128) % warps == 0,
                  "IQ2 pair requires complete contiguous warp partitions");
  }
  segs.main_tiles = tiles;
  if (segs.ab_n > 0) tiles += (segs.ab_n + warps * tp - 1) / (warps * tp);
  uint64_t* timestamps = nullptr;
  if constexpr (Clocked) {
    storage(clock_data, at::kLong);
    TORCH_CHECK(clock_data.dim() == 3 && clock_data.size(0) >= tiles * split &&
                    clock_data.size(1) == warps * tp && clock_data.size(2) == 2,
                "DMV timer storage requires [blocks, warps, 2] int64 values");
    timestamps = reinterpret_cast<uint64_t*>(clock_data.data_ptr<int64_t>());
  }
  // Only three measured-reader families are needed by the remaining
  // three-format attention inputs. Keep the existing pair specializations.
  int family_mask = 0;
  for (int i = 0; i < segs.nseg; ++i) family_mask |= 1 << segs.s[i].fmt;
  if (__builtin_popcount(static_cast<unsigned int>(family_mask)) == 3) {
    TORCH_CHECK(!segs.pair && warps == 4 && tp == 2,
                "three-format DMV requires single outputs with KW4/TN2");
    constexpr int kBase = (1 << Q4K) | (1 << LUT4);
    if (family_mask == (kBase | (1 << IQ3S)))
      return launch<4, 2, Q4K, LUT4, IQ3S, Clocked>(
          segs, xp, ldx, M, K, S, G, split, wsp, cp, tiles, st, timestamps);
    if (family_mask == (kBase | (1 << IQ3X)))
      return launch<4, 2, Q4K, LUT4, IQ3X, Clocked>(
          segs, xp, ldx, M, K, S, G, split, wsp, cp, tiles, st, timestamps);
    TORCH_CHECK(false, "unqualified three-format DMV family set");
  }
  int fa = segs.s[0].fmt, fb = fa;
  for (int i = 1; i < segs.nseg; ++i)
    if (segs.s[i].fmt != fa) fb = segs.s[i].fmt;
  for (int i = 0; i < segs.nseg; ++i)
    TORCH_CHECK(segs.s[i].fmt == fa || segs.s[i].fmt == fb,
                "at most two formats per launch");
#define CFG(A, B, X, Y)                                                      \
  if (warps == A && tp == B && fa == X && fb == Y)                           \
    return launch<A, B, X, Y, -1, Clocked>(segs, xp, ldx, M, K, S, G, split, \
                                           wsp, cp, tiles, st, timestamps);
#define FMTS(A, B)       \
  CFG(A, B, Q4K, Q4K);   \
  CFG(A, B, LUT4, LUT4); \
  CFG(A, B, IQ3S, IQ3S); \
  CFG(A, B, IQ3X, IQ3X); \
  CFG(A, B, IQ3S, IQ3X); \
  CFG(A, B, IQ3X, IQ3S); \
  CFG(A, B, LUT4, IQ3S); \
  CFG(A, B, IQ3S, LUT4); \
  CFG(A, B, LUT4, IQ3X); \
  CFG(A, B, IQ3X, LUT4); \
  CFG(A, B, Q4K, IQ3S);  \
  CFG(A, B, IQ3S, Q4K);  \
  CFG(A, B, Q4K, IQ3X);  \
  CFG(A, B, IQ3X, Q4K);  \
  CFG(A, B, Q4K, LUT4);  \
  CFG(A, B, LUT4, Q4K);
#define IQ2_FMTS(A, B)   \
  CFG(A, B, LUT6, LUT6); \
  CFG(A, B, LUT6, IQ3S); \
  CFG(A, B, IQ3S, LUT6); \
  CFG(A, B, LUT6, IQ3X); \
  CFG(A, B, IQ3X, LUT6); \
  CFG(A, B, LUT6, LUT4); \
  CFG(A, B, LUT4, LUT6);
  IQ2_FMTS(4, 2);
  IQ2_FMTS(4, 4);
  IQ2_FMTS(8, 2);
#undef IQ2_FMTS
  FMTS(4, 1);
  FMTS(2, 2);
  FMTS(4, 2);
  FMTS(8, 2);
  FMTS(4, 4);
  FMTS(6, 2);
  FMTS(2, 4);
#undef FMTS
#undef CFG
  TORCH_CHECK(false, "unsupported warps/tp");
}

void gguf_dmv_sm70_out(torch::Tensor x, std::vector<torch::Tensor> codes,
                       std::vector<torch::Tensor> high,
                       std::vector<torch::Tensor> scale,
                       std::vector<torch::Tensor> out, std::vector<int64_t> fmt,
                       std::vector<int64_t> n, int64_t K, int64_t split,
                       int64_t warps, torch::Tensor ws, torch::Tensor cnt,
                       int64_t tp, std::optional<torch::Tensor> sgate,
                       std::optional<torch::Tensor> tab,
                       std::optional<torch::Tensor> ab_w,
                       std::optional<torch::Tensor> ab_out,
                       std::optional<torch::Tensor> pair_out, bool gdn_heads) {
  gguf_dmv_dispatch<false>(
      std::move(x), std::move(codes), std::move(high), std::move(scale),
      std::move(out), std::move(fmt), std::move(n), K, split, warps,
      std::move(ws), std::move(cnt), tp, std::move(sgate), std::move(tab),
      std::move(ab_w), std::move(ab_out), std::move(pair_out), gdn_heads);
}

void gguf_dmv_sm70_clocked_out(
    torch::Tensor clock_data, torch::Tensor x, std::vector<torch::Tensor> codes,
    std::vector<torch::Tensor> high, std::vector<torch::Tensor> scale,
    std::vector<torch::Tensor> out, std::vector<int64_t> fmt,
    std::vector<int64_t> n, int64_t K, int64_t split, int64_t warps,
    torch::Tensor ws, torch::Tensor cnt, int64_t tp,
    std::optional<torch::Tensor> sgate, std::optional<torch::Tensor> tab,
    std::optional<torch::Tensor> ab_w, std::optional<torch::Tensor> ab_out,
    std::optional<torch::Tensor> pair_out, bool gdn_heads) {
  gguf_dmv_dispatch<true>(
      std::move(x), std::move(codes), std::move(high), std::move(scale),
      std::move(out), std::move(fmt), std::move(n), K, split, warps,
      std::move(ws), std::move(cnt), tp, std::move(sgate), std::move(tab),
      std::move(ab_w), std::move(ab_out), std::move(pair_out), gdn_heads,
      std::move(clock_data));
}

namespace {
__global__ void dmv_restore(const uint4* codes, const uint8_t* scale,
                            uint16_t* weight, void* stats, int fmt, int k,
                            int n) {
  const int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= int64_t{k / 32} * n) return;
  const int col = i % n, step = i / n;
  const int t = col / 32, g = step / 4, st = step % 4;
  const int lane = (col % 4) | (((col % 32) / 8) << 2) | ((col & 4) << 2);
  const int64_t tg = int64_t{t} * (k / 128) + g;
  if (fmt == LUT4) {
    const uint4 v = codes[((int64_t{t} * (k / 32) + step) * 32 + lane)];
    uint32_t* dst = reinterpret_cast<uint32_t*>(weight);
    const int64_t offset = (int64_t{t} * (k / 8) + step * 4) * 32 + col % 32;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      uint32_t src = word(v, c), result = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        result |= ((src >> (4 * j)) & 15) << (4 * (j / 2 + (j % 2) * 4));
      dst[offset + c * 32] = result;
    }
    const uint16_t* coef = reinterpret_cast<const uint16_t*>(scale);
    static_cast<uint16_t*>(stats)[i] = coef[(tg * 32 + lane) * 4 + st];
    return;
  }
  const uint4 v = codes[(tg * 3 + st / 2) * 32 + lane];
  const uint32_t idx[2] = {st % 2 ? v.z : v.x, st % 2 ? v.w : v.y};
  const uint32_t sg = word(codes[(tg * 3 + 2) * 32 + lane], st);
  const uint32_t* meta = reinterpret_cast<const uint32_t*>(scale);
  const int64_t mp = (tg * 32 + lane) * (fmt == IQ3S ? 2 : 1);
  const uint32_t dsw = meta[mp + (fmt == IQ3S ? 1 : 0)];
  const uint32_t high = fmt == IQ3S ? (meta[mp] >> (8 * st)) & 255u : 0;
  const uint32_t nib = (dsw >> (16 + 4 * st)) & 15;
  const float d = __half2float(__ushort_as_half(dsw & 65535));
  const uint16_t coef = __half_as_ushort(
      __float2half_rn(d * float(1 + 2 * nib) * (fmt == IQ3X ? 0.25f : 1.f)));
  uint32_t natural = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    natural |= ((sg >> j) & 1) << (2 * j);
    natural |= ((sg >> (16 + j)) & 1) << (2 * j + 1);
  }
  static_cast<uint64_t*>(stats)[i] =
      uint64_t{coef} | (uint64_t{natural} << 16) | (uint64_t{high} << 48);
  const int64_t offset = (int64_t{t} * (k / 8) + step * 4) * 32 + col % 32;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    weight[offset + j * 32] = (idx[j / 2] >> (16 * (j % 2))) & 65535;
}
}  // namespace

void gguf_dmv_restore_sm70_out(torch::Tensor weight, torch::Tensor stats,
                               torch::Tensor codes, torch::Tensor scale,
                               int64_t fmt, int64_t k, int64_t n) {
  TORCH_CHECK(fmt == LUT4 || is_iq3(fmt), "unsupported DMV restore format");
  TORCH_CHECK(k > 0 && k % 128 == 0 && n > 0 && n % 32 == 0,
              "invalid DMV restore geometry");
  TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == at::kByte &&
                  codes.is_contiguous(),
              "invalid DMV restore codes");
  const c10::cuda::CUDAGuard guard(codes.device());
  TORCH_CHECK(
      scale.device() == codes.device() && scale.scalar_type() == at::kByte &&
          scale.is_contiguous() &&
          codes.numel() == n / 32 * (k / 128) * 512 * (fmt == LUT4 ? 4 : 3) &&
          scale.numel() == n / 32 * (k / 128) * (fmt == IQ3X ? 128 : 256),
      "invalid DMV restore planes");
  TORCH_CHECK(weight.device() == codes.device() &&
                  weight.scalar_type() == at::kInt && weight.is_contiguous() &&
                  weight.numel() == n * k / (fmt == LUT4 ? 8 : 16),
              "invalid DMV transient weight");
  TORCH_CHECK(
      stats.device() == codes.device() && stats.is_contiguous() &&
          stats.scalar_type() == (fmt == LUT4 ? at::kShort : at::kLong) &&
          stats.numel() == n * (k / 32),
      "invalid DMV transient stats");
  const int64_t count = n * (k / 32);
  dmv_restore<<<(count + 255) / 256, 256, 0,
                at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(codes.data_ptr()),
      scale.data_ptr<uint8_t>(), reinterpret_cast<uint16_t*>(weight.data_ptr()),
      stats.data_ptr(), fmt, k, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
// Reverse signed nibbles into the original u2 index/sign packets. Only the
// fallback path reads the magnitude table; M8 needs no shared codebook.
__global__ void dmv_restore_iq2(const uint4* codes, const uint2* meta,
                                const uint16_t* reverse, uint16_t* weight,
                                uint32_t* stats, int kind, int k, int n) {
  const int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= int64_t{k / 32} * n) return;
  const int col = i % n, step = i / n;
  const int t = col / 32, g = step / 4, st = step % 4;
  const int lane = (col % 4) | (((col % 32) / 8) << 2) | ((col & 4) << 2);
  const int64_t tg = int64_t{t} * (k / 128) + g;
  const uint4 v = codes[(int64_t{t} * (k / 32) + step) * 32 + lane];
  const uint2 m = meta[tg * 32 + lane];
  const float d = __half2float(__ushort_as_half(m.x & 65535));
  uint32_t high[2] = {0, 0};
  const int64_t offset = (int64_t{t} * (k / 8) + step * 4) * 32 + col % 32;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint32_t q = word(v, c);
    uint32_t key = 0, factor = 1, sign = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t digit = (q >> (4 * j)) & 15;
      const uint32_t magnitude = digit < 3 ? 2 - digit : digit - 3;
      key += magnitude * factor;
      factor *= 3;
      sign |= static_cast<uint32_t>(digit < 3) << j;
    }
    const uint32_t index = reverse[key];
    weight[offset + c * 32] = (index & 255) | (sign << 8);
    const int half = kind == 16 ? 0 : c / 2;
    const int shift = 16 + 2 * (kind == 16 ? c : c % 2);
    high[half] |= (index >> 8) << shift;
  }
  const int groups = kind == 16 ? 1 : 2;
#pragma unroll
  for (int h = 0; h < groups; ++h) {
    const uint32_t nib = (m.y >> (8 * st + 4 * h)) & 15;
    const uint32_t coefficient = __half_as_ushort(
        __float2half_rn(d * (0.5f + static_cast<float>(nib)) * 0.25f));
    stats[(int64_t{step} * groups + h) * n + col] = coefficient | high[h];
  }
}
}  // namespace

void gguf_dmv_restore_iq2_sm70_out(torch::Tensor weight, torch::Tensor stats,
                                   torch::Tensor codes, torch::Tensor meta,
                                   torch::Tensor reverse, int64_t kind,
                                   int64_t k, int64_t n) {
  TORCH_CHECK(kind == 16 || kind == 17 || kind == 22, "invalid IQ2 type");
  TORCH_CHECK(k > 0 && k % 128 == 0 && n > 0 && n % 32 == 0,
              "invalid IQ2 restore geometry");
  TORCH_CHECK(codes.is_cuda(), "IQ2 planes must be CUDA tensors");
  const c10::cuda::CUDAGuard guard(codes.device());
  auto storage = [&](const torch::Tensor& tensor, at::ScalarType dtype,
                     int64_t count) {
    TORCH_CHECK(tensor.device() == codes.device() &&
                    tensor.scalar_type() == dtype && tensor.is_contiguous() &&
                    tensor.numel() == count,
                "invalid IQ2 restore storage");
  };
  const int group = kind == 16 ? 32 : 16;
  storage(codes, at::kByte, n * k / 2);
  storage(meta, at::kByte, n * k / 16);
  storage(reverse, at::kByte, 2 * 6561);
  storage(weight, at::kInt, n * k / 16);
  storage(stats, at::kInt, n * k / group);
  const int64_t count = n * (k / 32);
  dmv_restore_iq2<<<(count + 255) / 256, 256, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(codes.data_ptr()),
      reinterpret_cast<const uint2*>(meta.data_ptr()),
      reinterpret_cast<const uint16_t*>(reverse.data_ptr()),
      reinterpret_cast<uint16_t*>(weight.data_ptr()),
      reinterpret_cast<uint32_t*>(stats.data_ptr()), kind, k, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_dmv_allreduce_norm_sm70_out(
    torch::Tensor input, torch::Tensor codes, torch::Tensor scales,
    torch::Tensor table, torch::Tensor partial,
    const std::vector<int64_t>& pointers, int64_t rank, torch::Tensor residual,
    torch::Tensor weight, torch::Tensor normalized, torch::Tensor residual_out,
    int64_t format, double epsilon) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
                  input.is_contiguous() &&
                  input.sizes() == at::IntArrayRef({8, 4352}),
              "GGUF projection collective requires M8/K4352");
  const c10::cuda::CUDAGuard guard(input.device());
  auto storage = [&](const torch::Tensor& tensor, at::ScalarType dtype) {
    TORCH_CHECK(tensor.device() == input.device() && tensor.is_contiguous() &&
                    tensor.scalar_type() == dtype,
                "invalid GGUF collective operand");
  };
  storage(codes, at::kByte);
  storage(scales, at::kByte);
  storage(table, at::kByte);
  storage(partial, at::kHalf);
  storage(normalized, at::kHalf);
  storage(residual, at::kFloat);
  storage(residual_out, at::kFloat);
  TORCH_CHECK(weight.device() == input.device() && weight.is_contiguous() &&
                  (weight.scalar_type() == at::kFloat ||
                   weight.scalar_type() == at::kHalf),
              "norm weight must be contiguous FP16 or FP32");
  TORCH_CHECK(pointers.size() == 4 && rank >= 0 && rank < 4 &&
                  (format == IQ3S || format == LUT4 || format == Q4K) &&
                  epsilon > 0.0,
              "unqualified GGUF collective");
  TORCH_CHECK(partial.sizes() == at::IntArrayRef({8, 5120}) &&
                  normalized.sizes() == partial.sizes() &&
                  residual.sizes() == partial.sizes() &&
                  residual_out.sizes() == partial.sizes() &&
                  weight.sizes() == at::IntArrayRef({5120}),
              "invalid GGUF collective output/norm geometry");
  TORCH_CHECK(codes.numel() == 160 * 34 * 512 * (format == IQ3S ? 3 : 4) &&
                  scales.numel() == 160 * 34 * (format == Q4K ? 512 : 256) &&
                  (format != IQ3S || table.numel() == TAB_VECS * 16),
              "invalid GGUF collective planes");
  Segs down{};
  down.nseg = 1;
  down.tab = format == IQ3S ? reinterpret_cast<const uint4*>(table.data_ptr())
                            : nullptr;
  auto& segment = down.s[0];
  segment.codes = reinterpret_cast<const uint4*>(codes.data_ptr());
  segment.scale = reinterpret_cast<const uint4*>(scales.data_ptr());
  segment.out = reinterpret_cast<half*>(partial.data_ptr());
  segment.out_ld = segment.n = 5120;
  segment.fmt = static_cast<int>(format);
  vllm::RankData buffers{};
  for (int peer = 0; peer < 4; ++peer) {
    TORCH_CHECK(pointers[peer] != 0, "missing GGUF peer packet buffer");
    buffers.ptrs[peer] = reinterpret_cast<void*>(pointers[peer]);
  }
  auto x = reinterpret_cast<const half*>(input.data_ptr());
  auto out = reinterpret_cast<half*>(normalized.data_ptr());
  const float eps = static_cast<float>(epsilon);
  auto dispatch = [&](auto* weight_tag) {
    using Weight = std::remove_pointer_t<decltype(weight_tag)>;
#define PIPE(F)                                                               \
  run_pipeline<F, Weight>(down, x, buffers, static_cast<int>(rank),           \
                          residual.data_ptr<float>(),                         \
                          reinterpret_cast<const Weight*>(weight.data_ptr()), \
                          out, residual_out.data_ptr<float>(), eps, true)
    if (format == IQ3S) {
      PIPE(IQ3S);
    } else if (format == LUT4) {
      PIPE(LUT4);
    } else {
      PIPE(Q4K);
    }
#undef PIPE
  };
  if (weight.scalar_type() == at::kHalf)
    dispatch(static_cast<half*>(nullptr));
  else
    dispatch(static_cast<float*>(nullptr));
}
