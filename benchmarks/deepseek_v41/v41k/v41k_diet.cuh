// EXPERT_DIET (VLLM_V41_EXPERT_DIET, default 0 = off): instruction-diet
// variants of the EXPERT_FAST kernels. New kernels beside the old ones; the
// EXPERT_FAST kernels (v41k_fast.cuh, v41k_fast_gu.cuh) are not touched.
//
// gate_up_diet<Type> is bitwise identical to v41k_fast::gate_up_fast<Type>
// (and so to v41k::gate_up_em5<Type> in plan mode): same 512 threads, same
// persistent grid and unit walk, same thread -> (row, K group) map, same
// integer weights, same per-lane serial chain FFMA(FMUL(FMUL(d, xd),
// I2F(S)), 0.125, acc) over i = 0..9, same xor 8, 4, 2, 1 butterfly, same
// epilogue (verbatim). What changes is how many instructions each step costs:
//
//  * Activations of all M tokens (M <= 6, 160 Q8 groups each) are copied
//    once per CTA into shared memory as 48-byte records (32 q bytes, the fp32
//    scale, 12 bytes pad), so a member's activation is 3 LDS at
//    [record + immediate] instead of an IMAD.WIDE address pair and 3 LDG
//    behind its `c < cnt` branch. The next member's record is read before
//    the current member's dots, so the (short, fixed) LDS latency overlaps
//    math instead of the L1 / L2 latency stalling every member in turn.
//    48-byte records keep the LDS.128 of 8 consecutive groups conflict-free.
//  * Weight addresses: one 32-bit row offset per thread, advanced once per
//    two steps, plus per-lane 64-bit field bases: each field address is one
//    IMAD.WIDE.U32, the odd step an immediate. The old code rebuilt every
//    64-bit field address per step (IMAD + sign extension + IADD3/IADD3.X).
//  * IQ1_M: the octet indices come from 2 PRMT over (qh | (qh >> 4) << 16)
//    (old: PRMT, SHL, 2 LOP3, 2 PRMT); the fp16 block scale bits from PRMT,
//    2 SHF, 1 LOP3 (old: ~9 shifts / masks / LEA.HI / IADD).
//  * IQ2_XXS (m0 == m1): one dp4a chain over the 8 words and one IMUL,
//    s * m, which is the same integer as s0 * m + s1 * m.
// All integer arithmetic is exact, so only the float ops matter for the
// bits, and those are the old ones in the old order.
//
// The host requires gate / up to be smaller than 2 GB (32-bit offsets) and
// M <= 6 (activation records <= 46080 bytes of dynamic shared memory).
//
// Included by v41k.cu after v41k_fast_gu.cuh (and by the host emulator).
#pragma once
#include "v41k_fast.cuh"
#include "v41k_fast_gu.cuh"

namespace v41k_diet {
using v41k_fast::prmt;

// ---- decode pieces (host + device, checked exhaustively on the CPU) -------

// (a & m) | (b & ~m): one LOP3.
V41KF_HD uint32_t lop_sel(uint32_t a, uint32_t b, uint32_t m) {
  return (a & m) | (b & ~m);
}

// IQ1_M fp16 block scale bits (low 16 bits; the high 16 are garbage): the
// top nibbles of the four 16-bit scale words, word 0 lowest. Bytes 1, 5, 3, 7
// of (s23:s01) hold them in their high nibbles; y >> 4 brings words 0 and 2
// to nibbles 0 and 2, y >> 16 words 1 and 3 to nibbles 1 and 3.
V41KF_HD uint32_t d29_bits(uint32_t s01, uint32_t s23) {
  const uint32_t y = prmt(s01, s23, 0x7351);
  return lop_sel(y >> 4, y >> 16, 0x0F0F0F0Fu);
}

// IQ1_M octet table indices: a holds octets 0 / 1, b octets 2 / 3, each in a
// 16-bit half as (qs byte | qh nibble << 8) in bits 0..11 (bits 12..15 are
// garbage, masked by the offset). qh < 65536.
V41KF_HD void idx29(uint32_t qs, uint32_t qh, uint32_t& a, uint32_t& b) {
  const uint32_t h = qh + ((qh >> 4) << 16);  // qh0, qh1, (qh >> 4) bytes
  a = prmt(qs, h, 0x6140);                    // qs0, qh0, qs1, (qh >> 4)0
  b = prmt(qs, h, 0x7352);                    // qs2, qh1, qs3, (qh >> 4)1
}

// IQ1_M sub-block multipliers: f6 = 6 bits of the scale word at bit sh
// (sh = 16 ((sub >> 1) & 1) + 6 (sub & 1)), hi = sub >= 4; one = 1 (an
// opaque register on the device, so each multiplier is SHF + one LOP3).
V41KF_HD void m29(uint32_t s01, uint32_t s23, bool hi, uint32_t sh,
                  uint32_t one, int& m0, int& m1) {
  const uint32_t f6 = (hi ? s23 : s01) >> sh;
  m0 = int(((f6 << 1) & 14) | one);  // 2 (f6 & 7) + 1
  m1 = int(((f6 >> 2) & 14) | one);  // 2 ((f6 >> 3) & 7) + 1
}

// The IQ1_M octet words from a / b (idx29) over the CTA's selector table:
// v41k_fast::dec29's lookup, unchanged.
V41KF_HD void dec29_words(uint32_t a, uint32_t b, const uint32_t* book,
                          uint32_t (&w)[8]) {
  using FB = v41k_fast::FastBook<29>;
#pragma unroll
  for (int o = 0; o < 4; ++o) {
    const uint32_t x = o < 2 ? a : b;
    const uint32_t off = ((o & 1) ? x >> 14 : x << 2) & 0x3FFCu;
    const uint32_t sel = v41k_fast::lds_byte_off(book, off);
    w[2 * o] = prmt(FB::kPos, FB::kNeg, sel);
    w[2 * o + 1] = prmt(FB::kPos, FB::kNeg, sel >> 16);
  }
}

// fast_dot with S = s0 m0 + s1 m1 (two chains) or, for equal multipliers,
// S = (s0 + s1) m from one chain: the same integer. Float ops = fast_dot's.
template <bool OneM>
V41KF_HD float dot(const uint32_t (&w)[8], const int (&q)[8], float d,
                   float xd, int m0, int m1, float acc) {
  int S;
  if constexpr (OneM) {
    int s = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) s = v41k_fast::dp4a(w[i], q[i], s);
    S = s * m0;
    (void)m1;
  } else {
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) s0 = v41k_fast::dp4a(w[i], q[i], s0);
#pragma unroll
    for (int i = 4; i < 8; ++i) s1 = v41k_fast::dp4a(w[i], q[i], s1);
    S = s0 * m0 + s1 * m1;
  }
#if defined(__CUDA_ARCH__)
  return __fmaf_rn(__fmul_rn(__fmul_rn(d, xd), __int2float_rn(S)), 0.125f,
                   acc);
#else
  const volatile float a = d * xd;
  const volatile float b = a * float(S);
  return __builtin_fmaf(b, 0.125f, acc);
#endif
}

#if defined(__CUDACC__)

// base + off. Mad = true: one IMAD.WIDE.U32 (one: an opaque register holding
// 1, so ptxas keeps the multiply-add instead of rebuilding a 64-bit add; it
// still splits some into IADD3 / IMAD.X pairs). That saves ~32 instructions
// per 2 steps for IQ1_M but costs registers: IQ2_XS uses the plain add (with
// the multiply-add it reaches 128 registers and spills 4 bytes).
template <bool Mad = true>
__device__ __forceinline__ const uint8_t* at(const uint8_t* base, uint32_t off,
                                             uint32_t one) {
#if defined(__CUDA_ARCH__)
  if constexpr (Mad) {
    const uint8_t* p;
    asm("mad.wide.u32 %0, %1, %2, %3;"
        : "=l"(p)
        : "r"(off), "r"(one), "l"(base));
    return p;
  }
#endif
  (void)one;
  return base + off;
}

// Raw fields of one sub-block (32 weights) of one row, step i: block
// (l / 8) + 2 i. Bases are per lane and matrix (block l / 8 of row 0 plus the
// lane's field offset); a step adds STEP bytes; off = the row offset.
template <int Type>
struct DietRaw;
template <>
struct DietRaw<29> {  // IQ1_M, 56-byte blocks
  static constexpr int STEP = 112;
  struct Base {
    const uint8_t *q, *h, *s;
    __device__ __forceinline__ Base(const uint8_t* m, int boff, int sub)
        : q(m + boff + 4 * sub), h(m + boff + 32 + 2 * sub), s(m + boff + 48) {}
  };
  struct Addr {
    const uint8_t *q, *h, *s;
    __device__ __forceinline__ Addr(const Base& b, uint32_t off, uint32_t one)
        : q(at(b.q, off, one)), h(at(b.h, off, one)), s(at(b.s, off, one)) {}
  };
  uint32_t qs, qh, s01, s23;
  __device__ __forceinline__ void load(const Addr& a, int imm) {
    qs = v41k_fast::ldw32(a.q + imm);
    qh = v41k_fast::ldw16(a.h + imm);
    const uint2 sc = v41k_fast::ldw64(a.s + imm);
    s01 = sc.x;
    s23 = sc.y;
  }
  struct Lane {  // per-lane decode constants
    uint32_t sh, one;
    bool hi;
    __device__ __forceinline__ Lane(int sub, int, uint32_t o)
        : sh(uint32_t(((sub >> 1) & 1) * 16 + (sub & 1) * 6)),
          one(o),
          hi(sub >= 4) {}
  };
  __device__ __forceinline__ void decode(const uint32_t* book, const Lane& ln,
                                         uint32_t, uint32_t, uint32_t (&w)[8],
                                         float& d, int& m0, int& m1) const {
    uint32_t a, b;
    idx29(qs, qh, a, b);
    dec29_words(a, b, book, w);
    m29(s01, s23, ln.hi, ln.sh, ln.one, m0, m1);
    d = __half2float(__ushort_as_half((unsigned short)d29_bits(s01, s23)));
  }
};
// IQ2_XXS / IQ2_XS: the sub-block's 8 bytes as three aligned words (rows
// are 4-byte aligned and the block start is 2 (l < 8) or 0 (mod 4), as in
// v41k_fast::GuPtrIQ2), and the fp16 d at the block start.
template <int BB>
__device__ __forceinline__ int diet_wofs(int boff, int sub) {
  return (boff + 2 + 8 * sub) & ~3;
}
template <>
struct DietRaw<16> {  // IQ2_XXS, 66-byte blocks
  static constexpr int STEP = 132;
  struct Base {
    const uint8_t *w, *d;
    __device__ __forceinline__ Base(const uint8_t* m, int boff, int sub)
        : w(m + diet_wofs<66>(boff, sub)), d(m + boff) {}
  };
  struct Addr {
    const uint8_t *w, *d;
    __device__ __forceinline__ Addr(const Base& b, uint32_t off, uint32_t one)
        : w(at(b.w, off, one)), d(at(b.d, off, one)) {}
  };
  uint32_t x0, x1, x2, dh;
  __device__ __forceinline__ void load(const Addr& a, int imm) {
    x0 = v41k_fast::ldw32(a.w + imm);
    x1 = v41k_fast::ldw32(a.w + imm + 4);
    x2 = v41k_fast::ldw32(a.w + imm + 8);
    dh = v41k_fast::ldw16(a.d + imm);
  }
  struct Lane {
    uint32_t sh;
    __device__ __forceinline__ Lane(int, int l, uint32_t) : sh(l < 8 ? 16 : 0) {}
  };
  __device__ __forceinline__ void decode(const uint32_t* book, const Lane& ln,
                                         uint32_t lg, uint32_t ls,
                                         uint32_t (&w)[8], float& d, int& m0,
                                         int& m1) const {
    v41k_fast::dec16(__funnelshift_r(x0, x1, ln.sh),
                     __funnelshift_r(x1, x2, ln.sh), book, lg, ls, w, m0, m1);
    d = __half2float(__ushort_as_half((unsigned short)dh));
  }
};
template <>
struct DietRaw<17> {  // IQ2_XS, 74-byte blocks
  static constexpr int STEP = 148;
  struct Base {
    const uint8_t *w, *d, *s;
    __device__ __forceinline__ Base(const uint8_t* m, int boff, int sub)
        : w(m + diet_wofs<74>(boff, sub)), d(m + boff), s(m + boff + 66 + sub) {}
  };
  struct Addr {
    const uint8_t *w, *d, *s;
    __device__ __forceinline__ Addr(const Base& b, uint32_t off, uint32_t one)
        : w(at<false>(b.w, off, one)),
          d(at<false>(b.d, off, one)),
          s(at<false>(b.s, off, one)) {}
  };
  uint32_t x0, x1, x2, dh, sc;
  __device__ __forceinline__ void load(const Addr& a, int imm) {
    x0 = v41k_fast::ldw32(a.w + imm);
    x1 = v41k_fast::ldw32(a.w + imm + 4);
    x2 = v41k_fast::ldw32(a.w + imm + 8);
    dh = v41k_fast::ldw16(a.d + imm);
    sc = v41k_fast::ldw8(a.s + imm);
  }
  struct Lane {
    uint32_t sh;
    __device__ __forceinline__ Lane(int, int l, uint32_t) : sh(l < 8 ? 16 : 0) {}
  };
  __device__ __forceinline__ void decode(const uint32_t* book, const Lane& ln,
                                         uint32_t, uint32_t ls,
                                         uint32_t (&w)[8], float& d, int& m0,
                                         int& m1) const {
    v41k_fast::dec17(__funnelshift_r(x0, x1, ln.sh),
                     __funnelshift_r(x1, x2, ln.sh), sc, book, ls, w, m0, m1);
    d = __half2float(__ushort_as_half((unsigned short)dh));
  }
};

// Activation records in shared memory: record r = token * 160 + group at
// byte 48 r of the record area: q bytes 0..31, scale at 32.
constexpr int kRec = 48, kGuGroups = 160;
template <int Type>
struct GuDiet {
  static constexpr int kBookBytes = v41k_fast::FastBook<Type>::kSmemWords * 4;
  // dynamic shared memory for M tokens: table + records
  static constexpr int smem(int tokens) {
    return kBookBytes + tokens * kGuGroups * kRec;
  }
};

__device__ __forceinline__ void lda(const uint8_t* sm, uint32_t a, int (&q)[8],
                                    float& d) {
  const int4 x = *reinterpret_cast<const int4*>(sm + a);
  const int4 y = *reinterpret_cast<const int4*>(sm + a + 16);
  q[0] = x.x;
  q[1] = x.y;
  q[2] = x.z;
  q[3] = x.w;
  q[4] = y.x;
  q[5] = y.y;
  q[6] = y.z;
  q[7] = y.w;
  d = *reinterpret_cast<const float*>(sm + a + 32);
}

// Same contract and launch shape as v41k_fast::gate_up_fast<Type> (grid =
// min(SMs, 18 routes), 512 threads), plus tokens = routes / 6 and dynamic
// shared memory GuDiet<Type>::smem(tokens).
template <int Type>
__global__ void __launch_bounds__(512, 1)
    gate_up_diet(int8_t* __restrict__ hq, float* __restrict__ hd,
                 const int8_t* __restrict__ xq, const float* __restrict__ xd,
                 const float* __restrict__ prob,
                 const uint8_t* __restrict__ gate,
                 const uint8_t* __restrict__ up, int row_stride,
                 const uint32_t* __restrict__ fbook,
                 const int* __restrict__ plan, int tokens) {
  using FB = v41k_fast::FastBook<Type>;
  using Raw = DietRaw<Type>;
  using Base = typename Raw::Base;
  using Addr = typename Raw::Addr;
  constexpr int STEPS = 10, SLABS = 18, STEP = Raw::STEP;
  constexpr int ACT0 = GuDiet<Type>::kBookBytes;
  constexpr bool kOneM = Type == 16;
#if defined(V41K_HOST_EMU)  // host emulator: CTAs run one at a time
  alignas(16) static uint8_t dsm[GuDiet<Type>::smem(6)];
#else
  extern __shared__ __align__(16) uint8_t dsm[];
#endif
  __shared__ float red[2][6][2][32];
  uint32_t* const book = reinterpret_cast<uint32_t*>(dsm);
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int units = SLABS * __ldg(plan);
  int unit = blockIdx.x;
  if (unit >= units) return;
  const int l = lane & 15, sub = l & 7;
  const int lrow = warp * 2 + lane / 16;
  const int boff = (l >> 3) * FB::kBlock;
  // row offset (bytes) of the lane's row in unit un of expert e
  auto rows = [&](int un, int e) {
    return uint32_t((e * 576 + (un % SLABS) * 32 + lrow) * row_stride);
  };
  uint32_t one = 1;
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+r"(one));
#endif
  const Base bg(gate, boff, sub), bu(up, boff, sub);
  v41k_fast::GuUnit cur = v41k_fast::gu_unit(plan, unit);
  uint32_t uoff = rows(unit, cur.e);
  Raw ga, ua, gb, ub;  // steps 2k / 2k + 1
  {
    const Addr ag(bg, uoff, one), au(bu, uoff, one);
    ga.load(ag, 0);  // in flight while the table and records are filled
    ua.load(au, 0);
    gb.load(ag, STEP);
    ub.load(au, STEP);
  }
  // Activation records: at most 960 (M = 6), so two per thread.
  {
    const int nrec = tokens * kGuGroups;
    int4 a0 = {0, 0, 0, 0}, b0 = a0, a1 = a0, b1 = a0;
    float d0 = 0.f, d1 = 0.f;
    const int r0 = t, r1 = t + 512;
    if (r0 < nrec) {
      const int4* p = reinterpret_cast<const int4*>(xq) + 2 * r0;
      a0 = __ldg(p);
      b0 = __ldg(p + 1);
      d0 = __ldg(xd + r0);
    }
    if (r1 < nrec) {
      const int4* p = reinterpret_cast<const int4*>(xq) + 2 * r1;
      a1 = __ldg(p);
      b1 = __ldg(p + 1);
      d1 = __ldg(xd + r1);
    }
    v41k_fast::fill_book<Type>(book, fbook, t, 512);
    if (r0 < nrec) {
      uint8_t* s = dsm + ACT0 + kRec * r0;
      *reinterpret_cast<int4*>(s) = a0;
      *reinterpret_cast<int4*>(s + 16) = b0;
      *reinterpret_cast<float*>(s + 32) = d0;
    }
    if (r1 < nrec) {
      uint8_t* s = dsm + ACT0 + kRec * r1;
      *reinterpret_cast<int4*>(s) = a1;
      *reinterpret_cast<int4*>(s + 16) = b1;
      *reinterpret_cast<float*>(s + 32) = d1;
    }
  }
  uint32_t lg = uint32_t(lane) * 4, ls = uint32_t(lane & 15) * 4;
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+r"(lg), "+r"(ls));
#endif
  const typename Raw::Lane ln(sub, l, one);
  __syncthreads();
  int buf = 0;
  while (true) {
    const int nunit = unit + int(gridDim.x);
    const bool more = nunit < units;
    const v41k_fast::GuUnit nxt = more ? v41k_fast::gu_unit(plan, nunit) : cur;
    const uint32_t noff = more ? rows(nunit, nxt.e) : uoff;
    // member record offsets (+ the lane's group l); tok[c] = token * 160
    uint32_t mr[6];
#pragma unroll
    for (int c = 0; c < 6; ++c)
      mr[c] = uint32_t(ACT0 + kRec * (cur.tok[c] + l));
    const int cnt = cur.cnt;
    float gs[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float us[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    // step 2k + odd: decode (rg, ru), load step 2k + odd + 2 from (ag, au)
    auto step = [&](Raw& rg, Raw& ru, uint32_t so, const Addr& ag,
                    const Addr& au, int imm) {
      int q[2][8];  // members c (q[c & 1]) and c + 1: no copies once unrolled
      float qd[2] = {0.f, 0.f};
      lda(dsm, mr[0] + so, q[0], qd[0]);
      uint32_t wg[8], wu[8];
      float dg, du;
      int mg0, mg1, mu0, mu1;
      rg.decode(book, ln, lg, ls, wg, dg, mg0, mg1);
      ru.decode(book, ln, lg, ls, wu, du, mu0, mu1);
      rg.load(ag, imm);
      ru.load(au, imm);
#pragma unroll
      for (int c = 0; c < 6; ++c) {
        if (c >= cnt) break;  // block-uniform; cnt >= 1
        const int a = c & 1, b = a ^ 1;
        if (c + 1 < 6 && c + 1 < cnt)
          lda(dsm, mr[c + 1 < 6 ? c + 1 : 0] + so, q[b], qd[b]);
        gs[c] = dot<kOneM>(wg, q[a], dg, qd[a], mg0, mg1, gs[c]);
        us[c] = dot<kOneM>(wu, q[a], du, qd[a], mu0, mu1, us[c]);
      }
    };
    uint32_t off = uoff;
#pragma unroll 1
    for (int k = 0; k < STEPS / 2; ++k) {
      // loads of steps 2k + 2, 2k + 3; the last pair loads the next unit's
      // steps 0, 1 (or reloads this unit's when there is none: harmless)
      off = k == STEPS / 2 - 1 ? noff : off + 2 * STEP;
      const Addr ag(bg, off, one), au(bu, off, one);
      const uint32_t so = uint32_t(k) * (2 * 16 * kRec);
      step(ga, ua, so, ag, au, 0);
      step(gb, ub, so + 16 * kRec, ag, au, STEP);
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
    uoff = noff;
    buf ^= 1;
  }
}

// ---- down_rows_diet -------------------------------------------------------
// v41k_fast::down_rows_fast<Type, M> rolled. The old kernel unrolls 9 K
// blocks x 6 members: a ~40 KB loop at every M, 3x the ~12 KB L0
// instruction cache (no_instruction was its top non-memory stall at M = 6),
// and holds all 9 raw blocks of the next unit in registers. Here:
//  * the K blocks run as 3 iterations of 3 (#pragma unroll 1), with a
//    3-slot raw register ring: slot s holds block 3 i + s and, once decoded,
//    is reloaded with block 3 (i + 1) + s, or with block s of the next unit
//    in the last iteration (lookahead 3 blocks instead of 9); the next unit's
//    rows are prefetched into L2 when it is claimed, so those loads see L2
//    rather than DRAM latency;
//  * member 0 runs inline (register accumulator, activation loaded before
//    the decode); members 1..cnt-1 run in a rolled loop whose accumulators
//    live in shared memory (one float per lane and member: an exact store and
//    reload of the same serial chain), the next member's activation loaded
//    before the current member's dot;
//  * IQ1_M decodes with the gate_up_diet pieces (exact integers, checked).
// Unit walk (dynamic claim one ahead), lane map (row lane / 8, sub lane % 8,
// group sub + 8 j), per-(row, sub) serial chain over j = 0..8, xor 1, 2, 4
// butterfly, half rounding and the class-ordered epilogue are the old ones,
// so the output is bitwise identical.
//
// SA = true also copies the gathered activations (4 rank slots) into shared
// memory once per CTA: every activation load is then an LDS (no L1 prefetch).
// Only instantiated where it fits (dn_sa_fits).
//
// The host requires down to be smaller than 2 GB (32-bit row offsets).
template <int Type, int M, bool SA>
struct DnDiet {
  static constexpr int ROUTES = 6 * M;
  static constexpr int RS = (ROUTES * 648 + 255) / 256 * 256;
  static constexpr int QB = ROUTES * 576;
  static constexpr int kBookBytes = v41k_fast::FastBook<Type>::kSmemWords * 4;
  static constexpr int kAccBytes = v41k_fast::kDnWarps * 5 * 32 * 4;
  static constexpr int kActBytes = SA ? 3 * RS + ROUTES * 648 : 0;
  static constexpr int kSmem = kBookBytes + kAccBytes + kActBytes;
};
// dynamic + ~2.5 KB static + 1 KB reserve within the 96 KB carveout
template <int Type, int M>
constexpr bool dn_sa_fits() {
  return DnDiet<Type, M, true>::kSmem + 4 * 1024 <= 96 * 1024;
}

template <bool SA>
__device__ __forceinline__ void dn_act(const uint8_t* pq, const float* px,
                                       int (&q)[8], float& d) {
  if constexpr (SA) {  // shared memory: plain loads (LDS.128 x 2, LDS)
    const int4 a = *reinterpret_cast<const int4*>(pq);
    const int4 b = *reinterpret_cast<const int4*>(pq + 16);
    q[0] = a.x;
    q[1] = a.y;
    q[2] = a.z;
    q[3] = a.w;
    q[4] = b.x;
    q[5] = b.y;
    q[6] = b.z;
    q[7] = b.w;
    d = *px;
  } else {
    v41k_fast::load_act(pq, px, q, d);
  }
}

__device__ __forceinline__ void prefetch_l2(const void* p) {
#if defined(__CUDA_ARCH__)
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
#else
  (void)p;
#endif
}

// One raw K block of the lane's (row, sub): load from the block at byte
// offset o (= row offset + BB j) of the down tensor, then decode.
template <int Type>
struct DnSlot;
struct DnLane {  // per-lane constants
  const uint8_t* down;
  uint32_t sub, lg, ls, sh29, one;
  bool hi;
};
template <>
struct DnSlot<29> {  // IQ1_M, 56-byte blocks
  static constexpr int BB = 56;
  uint32_t qs, qh, s01, s23;
  __device__ __forceinline__ void load(const DnLane& ln, uint32_t o) {
    const uint8_t* const b = ln.down + o;
    qs = v41k_fast::ldw32(b + 4 * ln.sub);
    qh = v41k_fast::ldw16(b + 32 + 2 * ln.sub);
    const uint2 sc = v41k_fast::ldw64(b + 48);
    s01 = sc.x;
    s23 = sc.y;
  }
  __device__ __forceinline__ void decode(const uint32_t* book,
                                         const DnLane& ln, uint32_t (&w)[8],
                                         float& d, int& m0, int& m1) const {
    uint32_t a, b;
    idx29(qs, qh, a, b);
    dec29_words(a, b, book, w);
    m29(s01, s23, ln.hi, ln.sh29, ln.one, m0, m1);
    d = __half2float(__ushort_as_half((unsigned short)d29_bits(s01, s23)));
  }
};
template <int BB>  // IQ2: the 8 sub-block bytes at 2 + 8 sub as 3 aligned words
struct DnSlotIQ2 {
  uint32_t x0, x1, x2, dh, sh;
  __device__ __forceinline__ void load_iq2(const DnLane& ln, uint32_t o) {
    const uint32_t w = o + 2 + 8 * ln.sub;  // down is 8-byte aligned
    sh = (w & 3) * 8;
    const uint8_t* const p = ln.down + (w & ~3u);
    x0 = v41k_fast::ldw32(p);
    x1 = v41k_fast::ldw32(p + 4);
    x2 = v41k_fast::ldw32(p + 8);
    dh = v41k_fast::ldw16(ln.down + o);
  }
};
template <>
struct DnSlot<16> : DnSlotIQ2<66> {
  static constexpr int BB = 66;
  __device__ __forceinline__ void load(const DnLane& ln, uint32_t o) {
    load_iq2(ln, o);
  }
  __device__ __forceinline__ void decode(const uint32_t* book,
                                         const DnLane& ln, uint32_t (&w)[8],
                                         float& d, int& m0, int& m1) const {
    v41k_fast::dec16(__funnelshift_r(x0, x1, sh), __funnelshift_r(x1, x2, sh),
                     book, ln.lg, ln.ls, w, m0, m1);
    d = __half2float(__ushort_as_half((unsigned short)dh));
  }
};
template <>
struct DnSlot<17> : DnSlotIQ2<74> {
  static constexpr int BB = 74;
  uint32_t sc;
  __device__ __forceinline__ void load(const DnLane& ln, uint32_t o) {
    load_iq2(ln, o);
    sc = v41k_fast::ldw8(ln.down + o + 66 + ln.sub);
  }
  __device__ __forceinline__ void decode(const uint32_t* book,
                                         const DnLane& ln, uint32_t (&w)[8],
                                         float& d, int& m0, int& m1) const {
    v41k_fast::dec17(__funnelshift_r(x0, x1, sh), __funnelshift_r(x1, x2, sh),
                     sc, book, ln.ls, w, m0, m1);
    d = __half2float(__ushort_as_half((unsigned short)dh));
  }
};

// Same contract and launch shape as v41k_fast::down_rows_fast<Type, M>
// (grid rows_local / 16, 512 threads), plus DnDiet<Type, M, SA>::kSmem bytes
// of dynamic shared memory.
template <int Type, int M, bool SA>
__global__ void __launch_bounds__(32 * v41k_fast::kDnWarps, 1)
    down_rows_diet(half* __restrict__ out,
                   const uint8_t* __restrict__ gathered,
                   const uint8_t* __restrict__ down, int row_stride,
                   int rows_local, int row_base,
                   const uint32_t* __restrict__ fbook, int classes,
                   const int* __restrict__ plan) {
  using namespace v41k_fast;
  using D = DnDiet<Type, M, SA>;
  using Slot = DnSlot<Type>;
  constexpr int LG = 18, N = 5120, TOP_K = 6, BB = Slot::BB;
  constexpr int RS = D::RS, QB = D::QB;
#if defined(V41K_HOST_EMU)
  alignas(16) static uint8_t dsm[D::kSmem];
#else
  extern __shared__ __align__(16) uint8_t dsm[];
#endif
  uint32_t* const book = reinterpret_cast<uint32_t*>(dsm);
  float* const s_p = reinterpret_cast<float*>(dsm + D::kBookBytes);
  uint8_t* const s_act = dsm + D::kBookBytes + D::kAccBytes;
  __shared__ int s_e[36], s_cnt[36], s_route[36][6], s_key[36], s_nu, s_next;
  __shared__ half s_v[36][kDnRows];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int r4 = lane >> 3, sub = lane & 7;
  const int row0 = blockIdx.x * kDnRows;

  if (t < 36) {
    s_e[t] = __ldg(plan + 1 + t);
    s_cnt[t] = __ldg(plan + 37 + t);
    s_key[t] = -1;
  } else if (t >= 64 && t < 64 + 216) {
    s_route[(t - 64) / 6][(t - 64) % 6] = __ldg(plan + 73 + (t - 64));
  } else if (t == 40) {
    s_nu = __ldg(plan);
    s_next = kDnWarps;
  }
  // byte offset of the lane's row of unit un (expert e) in down
  auto rowoff = [&](int un, int e) {
    return uint32_t(e * rows_local + row0 + (un & 3) * 4 + r4) *
           uint32_t(row_stride);
  };
  uint32_t one = 1, lg = uint32_t(lane) * 4, ls = uint32_t(lane & 15) * 4;
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+r"(one), "+r"(lg), "+r"(ls));
#endif
  const DnLane ln{down,
                  uint32_t(sub),
                  lg,
                  ls,
                  uint32_t(((sub >> 1) & 1) * 16 + (sub & 1) * 6),
                  one,
                  sub >= 4};
  // L2 prefetch offset in a row: 128 sub, clamped to the row's last byte
  const uint32_t pf_off = 128 * sub < 9 * BB - 1 ? 128 * sub : 9 * BB - 1;
  const int units = 4 * __ldg(plan);
  int u = warp;
  uint32_t ro = 0;
  Slot sl[3];
  if (u < units) {  // first unit's blocks 0..2 in flight while the table fills
    const int q0 = u >> 2;
    if constexpr (!SA) {
      const int c0 = __ldg(plan + 37 + q0);
#pragma unroll
      for (int c = 0; c < 6; ++c) {
        if (c >= c0) break;
        dn_prefetch_route<RS, QB>(gathered, __ldg(plan + 73 + 6 * q0 + c),
                                  lane);
      }
    }
    ro = rowoff(u, __ldg(plan + 1 + q0));
#pragma unroll
    for (int s = 0; s < 3; ++s) sl[s].load(ln, ro + BB * s);
    prefetch_l2(down + ro + pf_off);
  }
  if constexpr (SA) {
    constexpr int n16 = D::kActBytes / 16;
    const int4* src = reinterpret_cast<const int4*>(gathered);
    int4* dst = reinterpret_cast<int4*>(s_act);
    for (int i = t; i < n16; i += 32 * kDnWarps) dst[i] = __ldg(src + i);
  }
  fill_book<Type>(book, fbook, t, 32 * kDnWarps);
  __syncthreads();
  const int nu = s_nu;
  if (warp == kDnWarps - 1) {
    for (int i = lane; i < 216; i += 32) {
      const int q = i / 6, c = i % 6;
      if (q < nu && c < s_cnt[q])
        s_key[s_route[q][c]] = (q % classes) << 16 | q << 3 | c;
    }
  }
  const uint8_t* const abase = SA ? s_act : gathered;
  const uint8_t* const gq = abase + 32 * sub;
  const float* const gd = reinterpret_cast<const float*>(abase + QB) + sub;
  float* const myp = s_p + warp * (5 * 32) + lane;  // member c: myp[32 (c - 1)]
  while (u < units) {
    int nxt = 0;
    if (lane == 0) nxt = atomicAdd(&s_next, 1);
    nxt = __shfl_sync(0xffffffffU, nxt, 0);
    const bool more = nxt < units;
    uint32_t nro = ro;
    if (more) {
      const int qn = nxt >> 2;
      if constexpr (!SA) {
        const int cn = s_cnt[qn];
#pragma unroll
        for (int c = 0; c < 6; ++c) {
          if (c >= cn) break;
          dn_prefetch_route<RS, QB>(gathered, s_route[qn][c], lane);
        }
      }
      nro = rowoff(nxt, s_e[qn]);
      // the next unit's 4 rows into L2: 8 lanes per row, lines 128 sub
      // (clamped to the row's last byte) cover every line of the row
      prefetch_l2(down + nro + pf_off);
    }
    const int q = u >> 2, slab = u & 3, cnt = s_cnt[q];
    const int rt0 = s_route[q][0];
    const uint8_t* const pq0 = gq + rt0 * (LG * 32);
    const float* const pd0 = gd + rt0 * LG;
#pragma unroll 1
    for (int c = 1; c < cnt; ++c) myp[32 * (c - 1)] = 0.f;
    float p0 = 0.f;
#pragma unroll 1
    for (int i = 0; i < 3; ++i) {
      const bool reload = i < 2 || more;
      const uint32_t lo = i < 2 ? ro + uint32_t(3 * BB) * uint32_t(i + 1) : nro;
#pragma unroll
      for (int s = 0; s < 3; ++s) {
        const int j = 3 * i + s;
        const int rank = int(uint32_t(sub + 8 * j) / 18u);
        const int qo = 256 * j + rank * (RS - LG * 32);
        const int dof = 8 * j + rank * ((RS - LG * 4) / 4);
        int qv[8];
        float xd;
        dn_act<SA>(pq0 + qo, pd0 + dof, qv, xd);
        uint32_t w[8];
        float d;
        int m0, m1;
        sl[s].decode(book, ln, w, d, m0, m1);
        if (reload) sl[s].load(ln, lo + BB * s);
        p0 = fast_dot(w, qv, d, xd, m0, m1, p0);
        if (cnt > 1) {
          int qc[8];
          float xc;
          int r = s_route[q][1];
          dn_act<SA>(gq + r * (LG * 32) + qo, gd + r * LG + dof, qc, xc);
#pragma unroll 1
          for (int c = 1; c < cnt; ++c) {
            int qn[8];
            float xn = 0.f;
            if (c + 1 < cnt) {  // next member's activation first
              r = s_route[q][c + 1];
              dn_act<SA>(gq + r * (LG * 32) + qo, gd + r * LG + dof, qn, xn);
            }
            float* const pc = myp + 32 * (c - 1);
            *pc = fast_dot(w, qc, d, xc, m0, m1, *pc);
#pragma unroll
            for (int k = 0; k < 8; ++k) qc[k] = qn[k];
            xc = xn;
          }
        }
      }
    }
#pragma unroll
    for (int c = 0; c < 6; ++c) {
      if (c >= cnt) break;
      float v = c == 0 ? p0 : myp[32 * (c - 1)];
      v += __shfl_xor_sync(0xffffffffU, v, 1);
      v += __shfl_xor_sync(0xffffffffU, v, 2);
      v += __shfl_xor_sync(0xffffffffU, v, 4);
      if (sub == 0) s_v[s_route[q][c]][slab * 4 + r4] = __float2half_rn(v);
    }
    ro = nro;
    u = nxt;
  }
  __syncthreads();
  if (t < kDnRows * M) {  // the old epilogue, verbatim
    const int k = t / kDnRows, rl = t % kDnRows;
    int key[TOP_K];
#pragma unroll
    for (int i = 0; i < TOP_K; ++i) key[i] = s_key[k * TOP_K + i];
    float total = 0.f, acc = 0.f;
    int last = -1, cls = -1;
#pragma unroll
    for (int n = 0; n < TOP_K; ++n) {
      int best = 0x7fffffff, bi = 0;
#pragma unroll
      for (int i = 0; i < TOP_K; ++i)
        if (key[i] > last && key[i] < best) {
          best = key[i];
          bi = i;
        }
      if (best != 0x7fffffff) {
        last = best;
        if ((best >> 16) != cls) {
          if (cls >= 0) total += acc;
          acc = 0.f;
          cls = best >> 16;
        }
        acc += __half2float(s_v[k * TOP_K + bi][rl]);
      }
    }
    if (cls >= 0) total += acc;
    out[k * N + row_base + row0 + rl] = __float2half_rn(total);
#pragma unroll
    for (int o = 0; o < 4; ++o)
      if (o * rows_local != row_base)
        out[k * N + o * rows_local + row0 + rl] = __float2half_rn(0.f);
  }
}

#endif  // __CUDACC__
}  // namespace v41k_diet
