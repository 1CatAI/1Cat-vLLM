// EXPERT_FAST gate/up (VLLM_V41_EXPERT_FAST=1): gate_up_fast<Type>, a
// bit-identical replacement for v41k::gate_up_em5<Type> in plan mode (the
// kernel gate_up7 and gate_up7p launch).
//
// Same 512 threads per (expert, 32-row slab) and the same thread -> work map:
// row = slab * 32 + warp * 2 + lane / 16, lane l = lane % 16 takes groups l +
// 16 i (i = 0..9) of the 160 K groups, block (l / 8) + 2 i, sub-block l % 8.
// The grid is persistent (one CTA per SM walks the 18 nu units) instead of
// one CTA per unit. Per member c the gate
// and up partials are the same serial chain FFMA(FMUL(FMUL(d, xd), I2F(s0 m0
// + s1 m1)), 0.125, acc) from +0 over i, the same xor 8, 4, 2, 1 butterfly,
// and the epilogue (half rounding, clamp 10, silu with expf, x up x prob,
// half rounding, q8_warp_aligned) is the old code. Only the decode differs:
// PRMT selectors over the bank-replicated fast table (v41k_fast.cuh, proved
// equal to Group3 for every code), filled once per CTA while the first
// weights are in flight, and weights loaded two steps ahead.
//
// Included by v41k.cu after v41k::load_q8 and v41k::q8_warp_aligned.
#pragma once
#include "v41k_fast.cuh"

#if defined(__CUDACC__)
namespace v41k_fast {

// One sub-block (32 weights) of one row for step i = block (l / 8) + 2 i.
template <int Type>
struct GuRaw;
template <>
struct GuRaw<29> {  // IQ1_M, 56-byte blocks; a step advances 112 bytes
  struct Ptr {
    const uint8_t *qsp, *qhp, *scp;
    __device__ __forceinline__ Ptr(const uint8_t* blk, int sub)
        : qsp(blk + 4 * sub), qhp(blk + 32 + 2 * sub), scp(blk + 48) {}
    __device__ __forceinline__ void pin() {
      pin_ptr(qsp);
      pin_ptr(qhp);
      pin_ptr(scp);
    }
  };
  uint32_t qs, qh, s01, s23;
  __device__ __forceinline__ void load(const Ptr& p, int i) {
    qs = __ldg(reinterpret_cast<const uint32_t*>(p.qsp + 112 * i));
    qh = ldg16(p.qhp + 112 * i);
    const uint2 sc = __ldg(reinterpret_cast<const uint2*>(p.scp + 112 * i));
    s01 = sc.x;
    s23 = sc.y;
  }
  int sub;
  __device__ __forceinline__ void decode(const uint32_t* book, uint32_t,
                                         uint32_t, uint32_t, uint32_t (&w)[8],
                                         float& d, int& m0, int& m1) const {
    dec29(qs, qh, s01, s23, sub, book, w, m0, m1);
    d = __half2float(__ushort_as_half(iq1m_d_bits(s01, s23)));
  }
};
// IQ2_XXS / IQ2_XS: the 8 sub-block bytes at block + 2 + 8 sub as three
// aligned words. Rows start 4-byte aligned (checked by the host) and BB = 2
// (mod 4), so block l / 8 + 2 i starts 2 (l < 8) or 0 bytes into a word: the
// funnel shift is 16 or 0 bits, fixed per lane.
template <int BB>
struct GuPtrIQ2 {
  const uint32_t* pw;  // aligned word holding the sub-block's first byte
  const uint8_t* dp;   // block start (fp16 d)
  __device__ __forceinline__ GuPtrIQ2(const uint8_t* blk, int sub) : dp(blk) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(blk + 2 + 8 * sub);
    pw = reinterpret_cast<const uint32_t*>(x & ~uintptr_t(3));
  }
  __device__ __forceinline__ void pin() {
    pin_ptr(pw);
    pin_ptr(dp);
  }
};
template <>
struct GuRaw<16> {  // IQ2_XXS, 66-byte blocks; a step advances 132 bytes
  using Ptr = GuPtrIQ2<66>;
  uint32_t x0, x1, x2, dh;
  __device__ __forceinline__ void load(const Ptr& p, int i) {
    x0 = __ldg(p.pw + 33 * i);
    x1 = __ldg(p.pw + 33 * i + 1);
    x2 = __ldg(p.pw + 33 * i + 2);
    dh = ldg16(p.dp + 132 * i);
  }
  __device__ __forceinline__ void decode(const uint32_t* book, uint32_t lg,
                                         uint32_t ls, uint32_t sh,
                                         uint32_t (&w)[8], float& d, int& m0,
                                         int& m1) const {
    dec16(__funnelshift_r(x0, x1, sh), __funnelshift_r(x1, x2, sh), book, lg,
          ls, w, m0, m1);
    d = __half2float(__ushort_as_half((unsigned short)dh));
  }
};
template <>
struct GuRaw<17> {  // IQ2_XS, 74-byte blocks; a step advances 148 bytes
  struct Ptr : GuPtrIQ2<74> {
    const uint8_t* scp;  // scale byte of the sub-block
    __device__ __forceinline__ Ptr(const uint8_t* blk, int sub)
        : GuPtrIQ2<74>(blk, sub), scp(blk + 66 + sub) {}
    __device__ __forceinline__ void pin() {
      GuPtrIQ2<74>::pin();
      pin_ptr(scp);
    }
  };
  uint32_t x0, x1, x2, dh, sc;
  __device__ __forceinline__ void load(const Ptr& p, int i) {
    x0 = __ldg(p.pw + 37 * i);
    x1 = __ldg(p.pw + 37 * i + 1);
    x2 = __ldg(p.pw + 37 * i + 2);
    dh = ldg16(p.dp + 148 * i);
    sc = __ldg(p.scp + 148 * i);
  }
  __device__ __forceinline__ void decode(const uint32_t* book, uint32_t,
                                         uint32_t ls, uint32_t sh,
                                         uint32_t (&w)[8], float& d, int& m0,
                                         int& m1) const {
    dec17(__funnelshift_r(x0, x1, sh), __funnelshift_r(x1, x2, sh), sc, book,
          ls, w, m0, m1);
    d = __half2float(__ushort_as_half((unsigned short)dh));
  }
};

// Per-unit plan entries: expert, member count, member Q8 row offsets.
struct GuUnit {
  int u, e, cnt, tok[6];  // tok[c] = (route / 6) * 160
};
__device__ __forceinline__ GuUnit gu_unit(const int* __restrict__ plan,
                                          int unit) {
  GuUnit g;
  g.u = unit / 18;
  g.e = __ldg(plan + 1 + g.u);
  g.cnt = __ldg(plan + 37 + g.u);
#pragma unroll
  for (int c = 0; c < 6; ++c)
    g.tok[c] = c < g.cnt ? __ldg(plan + 73 + 6 * g.u + c) / 6 * 160 : 0;
  return g;
}

// Persistent: gridDim.x CTAs (one per SM) walk units = (distinct expert u,
// 32-row slab) round-robin, unit = blockIdx.x + k gridDim.x < 18 nu. The
// table is filled once per CTA. Weights are loaded two steps ahead (even /
// odd raw sets); the last two steps of a unit load the next unit's first two.
template <int Type>
__global__ void __launch_bounds__(512, 1)
    gate_up_fast(int8_t* __restrict__ hq, float* __restrict__ hd,
                 const int8_t* __restrict__ xq, const float* __restrict__ xd,
                 const float* __restrict__ prob,
                 const uint8_t* __restrict__ gate,
                 const uint8_t* __restrict__ up, int row_stride,
                 const uint32_t* __restrict__ fbook,
                 const int* __restrict__ plan) {
  using FB = FastBook<Type>;
  using Raw = GuRaw<Type>;
  using Ptr = typename Raw::Ptr;
  constexpr int STEPS = 10, SLABS = 18;
  __shared__ __align__(16) uint32_t book[FB::kSmemWords];
  __shared__ float red[2][6][2][32];
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int units = SLABS * __ldg(plan);
  int unit = blockIdx.x;
  if (unit >= units) return;
  const int l = lane & 15, sub = l & 7;
  const int lrow = warp * 2 + lane / 16;
  const int boff = (l >> 3) * FB::kBlock;
  auto rows = [&](int un, int e) {
    return (int64_t(e) * 576 + (un % SLABS) * 32 + lrow) * row_stride + boff;
  };
  GuUnit cur = gu_unit(plan, unit);
  Ptr pg(gate + rows(unit, cur.e), sub), pu(up + rows(unit, cur.e), sub);
  pg.pin();
  pu.pin();
  Raw ga, ua, gb, ub;  // steps 2k / 2k + 1
  if constexpr (Type == 29) ga.sub = ua.sub = gb.sub = ub.sub = sub;
  ga.load(pg, 0);  // in flight while the table is filled
  ua.load(pu, 0);
  gb.load(pg, 1);
  ub.load(pu, 1);
  uint32_t lg = uint32_t(lane) * 4, ls = uint32_t(lane & 15) * 4;
  uint32_t sh = l < 8 ? 16 : 0;
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+r"(lg), "+r"(ls), "+r"(sh));
#endif
  fill_book<Type>(book, fbook, t, 512);
  __syncthreads();
  int buf = 0;
  while (true) {
    const int nunit = unit + int(gridDim.x);
    const bool more = nunit < units;
    const GuUnit nxt = more ? gu_unit(plan, nunit) : cur;
    float gs[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float us[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    auto step = [&](Raw& rg, Raw& ru, int i, int ahead) {
      uint32_t wg[8], wu[8];
      float dg, du;
      int mg0, mg1, mu0, mu1;
      rg.decode(book, lg, ls, sh, wg, dg, mg0, mg1);
      ru.decode(book, lg, ls, sh, wu, du, mu0, mu1);
      rg.load(pg, ahead);
      ru.load(pu, ahead);
      const int group = l + 16 * i;
      {
        int q[8];
        float d;
        v41k::load_q8(xq, xd, cur.tok[0] + group, q, d);
        gs[0] = fast_dot(wg, q, dg, d, mg0, mg1, gs[0]);
        us[0] = fast_dot(wu, q, du, d, mu0, mu1, us[0]);
      }
#pragma unroll
      for (int c = 1; c < 6; ++c) {
        if (c >= cur.cnt) break;  // block-uniform
        int q[8];
        float d;
        v41k::load_q8(xq, xd, cur.tok[c] + group, q, d);
        gs[c] = fast_dot(wg, q, dg, d, mg0, mg1, gs[c]);
        us[c] = fast_dot(wu, q, du, d, mu0, mu1, us[c]);
      }
    };
#pragma unroll 1
    for (int k = 0; k < STEPS / 2; ++k) {
      int ahead = 2 * k + 2;
      if (k == STEPS / 2 - 1) {  // steps 8, 9: load the next unit's 0, 1
        ahead = 0;
        if (more) {
          pg = Ptr(gate + rows(nunit, nxt.e), sub);
          pu = Ptr(up + rows(nunit, nxt.e), sub);
          pg.pin();
          pu.pin();
        }  // else: harmless reload of this unit's steps 0, 1
      }
      step(ga, ua, 2 * k, ahead);
      step(gb, ub, 2 * k + 1, ahead + 1);
    }
#pragma unroll
    for (int c = 0; c < 6; ++c) {
      if (c >= cur.cnt) break;
#pragma unroll
      for (int o = 8; o; o >>= 1) {
        gs[c] += __shfl_xor_sync(0xffffffffU, gs[c], o);
        us[c] += __shfl_xor_sync(0xffffffffU, us[c], o);
      }
      if (lane % 16 == 0) {
        red[buf][c][0][lrow] = gs[c];
        red[buf][c][1][lrow] = us[c];
      }
    }
    // red[buf] is rewritten two units later, after the next barrier, which
    // every thread reaches only after this unit's epilogue.
    __syncthreads();
    // Epilogue: v41k::gate_up_em5's, verbatim.
    if (warp < cur.cnt) {
      const int route = __ldg(plan + 73 + 6 * cur.u + warp);
      float g16 = __half2float(__float2half_rn(red[buf][warp][0][lane]));
      float u16 = __half2float(__float2half_rn(red[buf][warp][1][lane]));
      g16 = g16 > 10.f ? 10.f : g16;
      u16 = u16 > 10.f ? 10.f : (u16 < -10.f ? -10.f : u16);
      const float act = __fdiv_rn(g16, __fadd_rn(1.f, expf(-g16)));
      const float value = __half2float(
          __float2half_rn(__fmul_rn(__fmul_rn(act, u16), prob[route])));
      v41k::q8_warp_aligned(hq, hd, int64_t(route) * 18 + unit % SLABS,
                            value);
    }
    if (!more) break;
    unit = nunit;
    cur = nxt;
    buf ^= 1;
  }
}

}  // namespace v41k_fast
#endif  // __CUDACC__
