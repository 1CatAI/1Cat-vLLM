// V4.1 routed experts, decode M = 1..6: EXPERT_FAST kernels
// (VLLM_V41_EXPERT_FAST=1). New ops beside the old ones; old kernels untouched.
//
// Decode. The old path looks every 8-weight octet up in a byte codebook
// (uint2 per entry, 16 KB for IQ1_M) and then applies signs or the IQ1 delta
// with __vsub4/__vadd4 (emulated on sm_70, 4-5 instructions per word). Here
// every codebook entry is one 32-bit PRMT selector word: nibble i of the word
// picks byte i of the octet out of two constant registers POS/NEG,
//   nibble = code (0..2) | negate_or_delta << 2,
// so an octet is 1 LDS + 2 PRMT. IQ2_XXS/IQ2_XS magnitudes {8, 25, 43} are
// codes 0..2, POS = {8, 25, 43}, NEG = {-8, -25, -43}, and the 7-bit sign
// index (+ parity) is a second selector word OR-ed in. IQ1_M values
// {-1, 0, 1} x 8 + delta are codes 0..2 with POS = {-7, 1, 9} (delta +1) and
// NEG = {-9, -1, 7} (delta -1); the delta bit is folded into a 4096-entry
// table indexed by (qh nibble << 8 | qs byte). The decoded int8 weights are
// exactly the old ones (checked exhaustively by expert_fast_check.py and the
// host test), so the integer dots are unchanged.
//
// Bank conflicts. The IQ2 tables are replicated across banks in shared
// memory: word (i, r) at i * R + r, lane reads r = lane % R (grid R = 32 for
// IQ2_XXS, 16 otherwise; signs R = 16), so a warp lookup is 1-2 wavefronts
// instead of ~3.5 random ones. IQ1_M (4096 words) is not replicated.
//
// Numerics (down_rows_fast): bitwise identical to down_rows<T, NW, 8>
// (ROWS_TILE rows = 4, NW = classes). Each (row, sub s) partial is the same
// serial chain over the K blocks j = 0..8 (group g = s + 8 j), the 8 subs are
// summed by the same xor 1, 2, 4 butterfly, rounded to half, accumulated per
// token in the old warp-class order (class = q % NW, then q, then member c),
// and the NW class partials are added in ascending class order from +0.
// Empty classes are skipped: they contributed +0 to a total that starts at
// +0 and therefore can never be -0, so the result is unchanged. Only the
// assignment of rows/experts to warps and CTAs differs.
#pragma once
#include <cstdint>

#if defined(__CUDACC__)
  #define V41KF_HD __host__ __device__ __forceinline__
#else
  #define V41KF_HD inline
#endif

namespace v41k_fast {

V41KF_HD uint32_t prmt(uint32_t a, uint32_t b, uint32_t c) {
#if defined(__CUDA_ARCH__)
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
#else
  const uint64_t x = (uint64_t(b) << 32) | a;
  uint32_t d = 0;
  for (int n = 0; n < 4; ++n) {
    const uint32_t s = (c >> (4 * n)) & 15;
    uint32_t v = uint32_t(x >> (8 * (s & 7))) & 255;
    if (s & 8) v = (v & 128) ? 255 : 0;
    d |= v << (8 * n);
  }
  return d;
#endif
}

V41KF_HD int dp4a(uint32_t a, int b, int c) {
#if defined(__CUDA_ARCH__)
  return __dp4a(int(a), b, c);
#else
  for (int i = 0; i < 4; ++i)
    c += int(int8_t(a >> (8 * i))) * int(int8_t(uint32_t(b) >> (8 * i)));
  return c;
#endif
}

template <int Type>
struct FastBook;
template <>
struct FastBook<16> {  // IQ2_XXS
  static constexpr int kBlock = 66, kGrid = 256, kGridR = 32, kSigns = 128,
                       kSignR = 16;
  static constexpr int kWords = kGrid + kSigns;  // global table words
  static constexpr int kSignBase = kGrid * kGridR;
  static constexpr int kSmemWords = kSignBase + kSigns * kSignR;  // 40 KB
  static constexpr uint32_t kPos = 0x002B1908u, kNeg = 0x00D5E7F8u;
};
template <>
struct FastBook<17> {  // IQ2_XS
  static constexpr int kBlock = 74, kGrid = 512, kGridR = 16, kSigns = 128,
                       kSignR = 16;
  static constexpr int kWords = kGrid + kSigns;
  static constexpr int kSignBase = kGrid * kGridR;
  static constexpr int kSmemWords = kSignBase + kSigns * kSignR;  // 40 KB
  static constexpr uint32_t kPos = 0x002B1908u, kNeg = 0x00D5E7F8u;
};
template <>
struct FastBook<29> {  // IQ1_M
  static constexpr int kBlock = 56, kGrid = 4096, kGridR = 1, kSigns = 0,
                       kSignR = 0;
  static constexpr int kWords = kGrid;
  static constexpr int kSignBase = kGrid;
  static constexpr int kSmemWords = kGrid;  // 16 KB
  static constexpr uint32_t kPos = 0x000901F9u, kNeg = 0x0007FFF7u;
};

// Global fast-table word i. value(k) = byte k of LatticeCodebook<Type>
// (biased +128: IQ2 magnitudes 136/153/171, IQ1 values 127/128/129).
template <int Type, class Value>
V41KF_HD uint32_t fast_word(int i, Value value) {
  using FB = FastBook<Type>;
  if constexpr (Type == 29) {
    const int index = i & 2047;
    uint32_t sel = 0;
    for (int b = 0; b < 8; ++b)
      sel |= uint32_t(int(value(8 * index + b)) - 127) << (4 * b);
    return (i >> 11) & 1 ? sel | 0x44444444u : sel;
  } else {
    if (i < FB::kGrid) {
      uint32_t sel = 0;
      for (int b = 0; b < 8; ++b) {
        const int v = value(8 * i + b);
        sel |= uint32_t(v == 136 ? 0 : v == 153 ? 1 : 2) << (4 * b);
      }
      return sel;
    }
    const int s7 = i - FB::kGrid;
    int parity = 0;
    for (int b = 0; b < 7; ++b) parity ^= (s7 >> b) & 1;
    const int s8 = s7 | parity << 7;
    uint32_t w = 0;
    for (int b = 0; b < 8; ++b) w |= uint32_t((s8 >> b) & 1) << (4 * b + 2);
    return w;
  }
}

// ---- one 32-weight group (sub-block) of an original block -----------------
// book: the CTA's shared table (FastBook<Type>::kSmemWords words).
V41KF_HD uint32_t lds_byte_off(const uint32_t* book, uint32_t off) {
  return *reinterpret_cast<const uint32_t*>(
      reinterpret_cast<const char*>(book) + off);
}

// Lane byte offsets into the replicated tables: lg = 4 (lane % 32) for the
// IQ2_XXS grid (R = 32), ls = 4 (lane % 16) for the R = 16 tables.
// IQ2_XXS: w0 = 4 grid bytes, w1 = 4 x 7 sign bits + 4-bit scale.
V41KF_HD void dec16(uint32_t w0, uint32_t w1, const uint32_t* book,
                    uint32_t lg, uint32_t ls, uint32_t (&w)[8], int& m0,
                    int& m1) {
  using FB = FastBook<16>;
#pragma unroll
  for (int o = 0; o < 4; ++o) {
    const uint32_t go = (o == 0 ? w0 << 7 : w0 >> (8 * o - 7)) & 0x7F80u;
    const uint32_t so = (o == 0 ? w1 << 6 : w1 >> (7 * o - 6)) & 0x1FC0u;
    const uint32_t sel = lds_byte_off(book, go | lg) |
                         lds_byte_off(book, FB::kSignBase * 4 + (so | ls));
    w[2 * o] = prmt(FB::kPos, FB::kNeg, sel);
    w[2 * o + 1] = prmt(FB::kPos, FB::kNeg, sel >> 16);
  }
  m0 = m1 = 1 + 2 * int(w1 >> 28);
}

// IQ2_XS: four 16-bit codes (9-bit grid index, 7 sign bits); sc = scale byte.
V41KF_HD void dec17(uint32_t w0, uint32_t w1, uint32_t sc,
                    const uint32_t* book, uint32_t ls, uint32_t (&w)[8],
                    int& m0, int& m1) {
  using FB = FastBook<17>;
#pragma unroll
  for (int o = 0; o < 4; ++o) {
    const uint32_t x = o < 2 ? w0 : w1;
    const uint32_t go = ((o & 1) ? x >> 10 : x << 6) & 0x7FC0u;
    const uint32_t so = ((o & 1) ? x >> 19 : x >> 3) & 0x1FC0u;
    const uint32_t sel = lds_byte_off(book, go | ls) |
                         lds_byte_off(book, FB::kSignBase * 4 + (so | ls));
    w[2 * o] = prmt(FB::kPos, FB::kNeg, sel);
    w[2 * o + 1] = prmt(FB::kPos, FB::kNeg, sel >> 16);
  }
  m0 = 1 + 2 * int(sc & 15);
  m1 = 1 + 2 * int((sc >> 4) & 15);
}

// IQ1_M: qs = 4 index bytes, qh = 4 nibbles (3 index bits + delta), s01/s23 =
// the block's four 16-bit scale words; sub = the sub-block (0..7).
V41KF_HD uint16_t iq1m_d_bits(uint32_t s01, uint32_t s23) {
  return uint16_t(((s01 & 0xffff) >> 12) | (((s01 >> 16) >> 8) & 0xf0) |
                  (((s23 & 0xffff) >> 4) & 0xf00) | ((s23 >> 16) & 0xf000));
}
V41KF_HD void dec29(uint32_t qs, uint32_t qh, uint32_t s01, uint32_t s23,
                    int sub, const uint32_t* book, uint32_t (&w)[8], int& m0,
                    int& m1) {
  using FB = FastBook<29>;
  const uint32_t t = prmt(qh, 0, 0x4140);  // bytes qh0, 0, qh1, 0
  const uint32_t hs = (t & 0x000F000Fu) | ((t << 4) & 0x0F000F00u);
  const uint32_t i01 = prmt(qs, hs, 0x5140), i23 = prmt(qs, hs, 0x7362);
#pragma unroll
  for (int o = 0; o < 4; ++o) {
    const uint32_t x = o < 2 ? i01 : i23;
    const uint32_t off = ((o & 1) ? x >> 14 : x << 2) & 0x3FFCu;
    const uint32_t sel = lds_byte_off(book, off);
    w[2 * o] = prmt(FB::kPos, FB::kNeg, sel);
    w[2 * o + 1] = prmt(FB::kPos, FB::kNeg, sel >> 16);
  }
  const uint32_t sw = sub < 4 ? s01 : s23;
  const int sh = ((sub >> 1) & 1) * 16 + (sub & 1) * 6;
  const uint32_t f6 = (sw >> sh) & 63;
  m0 = 2 * int(f6 & 7) + 1;
  m1 = 2 * int(f6 >> 3) + 1;
}

// The old Group3::apply followed by `dot += ...`, which nvcc contracts to
// FMUL, FMUL, FFMA(x, 0.125, dot) (checked in the old SASS); written out
// with explicit roundings so the new kernels cannot contract differently.
V41KF_HD float fast_dot(const uint32_t (&w)[8], const int (&q)[8], float d,
                        float xd, int m0, int m1, float acc) {
  int s0 = 0, s1 = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) s0 = dp4a(w[i], q[i], s0);
#pragma unroll
  for (int i = 4; i < 8; ++i) s1 = dp4a(w[i], q[i], s1);
#if defined(__CUDA_ARCH__)
  return __fmaf_rn(__fmul_rn(__fmul_rn(d, xd), __int2float_rn(s0 * m0 + s1 * m1)),
                   0.125f, acc);
#else
  const volatile float a = d * xd;
  const volatile float b = a * float(s0 * m0 + s1 * m1);
  return __builtin_fmaf(b, 0.125f, acc);
#endif
}

#if defined(__CUDACC__)

template <int Type>
__global__ void build_fast_table(uint32_t* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= FastBook<Type>::kWords) return;
  out[i] = fast_word<Type>(i, [](int k) -> int {
    return turbomind::gemm::LatticeCodebook<Type>::value(k);
  });
}

// Global table -> replicated shared table.
template <int Type>
__device__ __forceinline__ void fill_book(uint32_t* book,
                                          const uint32_t* __restrict__ g,
                                          int t, int nt) {
  using FB = FastBook<Type>;
  for (int i = t; i < FB::kSmemWords; i += nt) {
    int src = i;
    if constexpr (Type != 29)
      src = i < FB::kSignBase ? i / FB::kGridR
                              : FB::kGrid + (i - FB::kSignBase) / FB::kSignR;
    book[i] = __ldg(g + src);
  }
}

__device__ __forceinline__ uint32_t ldg16(const uint8_t* p) {
  return __ldg(reinterpret_cast<const unsigned short*>(p));
}

// Raw fields of the 9 K blocks of one (row, sub) of a 2304-wide down row,
// loaded block by block so the next unit's loads can replace the current
// unit's as soon as each block is decoded. Ptr holds the per-row base
// pointers, so every block load is [base + immediate]; pin() makes them
// opaque so ptxas keeps them in registers instead of rebuilding them from
// the row offset at every block.
template <class T>
__device__ __forceinline__ void pin_ptr(T*& p) {
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+l"(p));
#endif
}
template <int Type>
struct DnRaw;
template <int BB>  // IQ2_XXS (66) / IQ2_XS (74): 8 sub-block bytes at 2 + 8 s
struct DnPtrIQ2 {
  const uint8_t* row;
  const uint32_t *pe, *po;  // word-aligned sub-block base, even / odd blocks
  uint32_t a;               // byte misalignment of block 0's sub-block
  __device__ __forceinline__ DnPtrIQ2(const uint8_t* r, int sub) : row(r) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(r + 2 + 8 * sub);
    pe = reinterpret_cast<const uint32_t*>(x & ~uintptr_t(3));
    po = reinterpret_cast<const uint32_t*>((x + BB) & ~uintptr_t(3));
    a = uint32_t(x) & 3;
  }
  // 8 bytes of block j as three aligned words; BB j = 2 j (mod 4).
  __device__ __forceinline__ const uint32_t* words(int j) const {
    return (j & 1 ? po : pe) + (BB / 2) * (j >> 1);
  }
  __device__ __forceinline__ void pin() {
    pin_ptr(row);
    pin_ptr(pe);
    pin_ptr(po);
  }
};
template <>
struct DnRaw<16> {
  using Ptr = DnPtrIQ2<66>;
  uint32_t x[9][3], dh[9], a;
  __device__ __forceinline__ void set(const Ptr& p) { a = p.a; }
  __device__ __forceinline__ void load(int j, const Ptr& p) {
    const uint32_t* q = p.words(j);
    x[j][0] = __ldg(q);
    x[j][1] = __ldg(q + 1);
    x[j][2] = __ldg(q + 2);
    dh[j] = ldg16(p.row + 66 * j);
  }
  __device__ __forceinline__ void decode(int j, const uint32_t* book,
                                         uint32_t lg, uint32_t ls, int,
                                         uint32_t (&w)[8], float& d, int& m0,
                                         int& m1) const {
    const uint32_t sh = ((a + 2 * j) & 3) * 8;
    const uint32_t w0 = __funnelshift_r(x[j][0], x[j][1], sh);
    const uint32_t w1 = __funnelshift_r(x[j][1], x[j][2], sh);
    dec16(w0, w1, book, lg, ls, w, m0, m1);
    d = __half2float(__ushort_as_half((unsigned short)dh[j]));
  }
};
template <>
struct DnRaw<17> {
  struct Ptr : DnPtrIQ2<74> {
    const uint8_t* scp;  // scale byte of sub-block s of block 0
    __device__ __forceinline__ Ptr(const uint8_t* r, int sub)
        : DnPtrIQ2<74>(r, sub), scp(r + 66 + sub) {}
    __device__ __forceinline__ void pin() {
      DnPtrIQ2<74>::pin();
      pin_ptr(scp);
    }
  };
  uint32_t x[9][3], dh[9], sc[9], a;
  __device__ __forceinline__ void set(const Ptr& p) { a = p.a; }
  __device__ __forceinline__ void load(int j, const Ptr& p) {
    const uint32_t* q = p.words(j);
    x[j][0] = __ldg(q);
    x[j][1] = __ldg(q + 1);
    x[j][2] = __ldg(q + 2);
    dh[j] = ldg16(p.row + 74 * j);
    sc[j] = __ldg(p.scp + 74 * j);
  }
  __device__ __forceinline__ void decode(int j, const uint32_t* book,
                                         uint32_t, uint32_t ls, int,
                                         uint32_t (&w)[8], float& d, int& m0,
                                         int& m1) const {
    const uint32_t sh = ((a + 2 * j) & 3) * 8;
    const uint32_t w0 = __funnelshift_r(x[j][0], x[j][1], sh);
    const uint32_t w1 = __funnelshift_r(x[j][1], x[j][2], sh);
    dec17(w0, w1, sc[j], book, ls, w, m0, m1);
    d = __half2float(__ushort_as_half((unsigned short)dh[j]));
  }
};
template <>
struct DnRaw<29> {
  struct Ptr {
    const uint8_t *row, *qsp, *qhp;
    __device__ __forceinline__ Ptr(const uint8_t* r, int sub)
        : row(r), qsp(r + 4 * sub), qhp(r + 32 + 2 * sub) {}
    __device__ __forceinline__ void pin() {
      pin_ptr(row);
      pin_ptr(qsp);
      pin_ptr(qhp);
    }
  };
  uint32_t qs[9], qh[9], s01[9], s23[9];
  __device__ __forceinline__ void set(const Ptr&) {}
  __device__ __forceinline__ void load(int j, const Ptr& p) {
    qs[j] = __ldg(reinterpret_cast<const uint32_t*>(p.qsp + 56 * j));
    qh[j] = ldg16(p.qhp + 56 * j);
    const uint2 sc = __ldg(reinterpret_cast<const uint2*>(p.row + 56 * j + 48));
    s01[j] = sc.x;
    s23[j] = sc.y;
  }
  __device__ __forceinline__ void decode(int j, const uint32_t* book, uint32_t,
                                         uint32_t, int sub, uint32_t (&w)[8],
                                         float& d, int& m0, int& m1) const {
    dec29(qs[j], qh[j], s01[j], s23[j], sub, book, w, m0, m1);
    d = __half2float(__ushort_as_half(iq1m_d_bits(s01[j], s23[j])));
  }
};

__device__ __forceinline__ void load_act(const uint8_t* pq, const float* px,
                                         int (&q)[8], float& d) {
  const int4* p = reinterpret_cast<const int4*>(pq);
  const int4 a = __ldg(p), b = __ldg(p + 1);
  q[0] = a.x;
  q[1] = a.y;
  q[2] = a.z;
  q[3] = a.w;
  q[4] = b.x;
  q[5] = b.y;
  q[6] = b.z;
  q[7] = b.w;
  d = __ldg(px);
}

// Gathered activations of group g = sub + 8 j (K block j of a lane): rank
// block g / 18, group g % 18. With the rank stride RS a compile-time constant
// the offsets from the per-route base (gathered + rt * 576 + 32 sub for hq,
// gathered + QB + rt * 72 + 4 sub for hd) are immediates, except for j = 2, 4,
// 6, where lanes with sub >= j fall in the next rank block.
__device__ __forceinline__ int act_rank(int j, int sub) {
  constexpr int lo[9] = {0, 0, 0, 1, 1, 2, 2, 3, 3};
  return lo[j] + int((j == 2 || j == 4 || j == 6) && sub >= j);
}

// Row-split down, same contract as down_rows (see v41k.cu) for M tokens x
// top-6 routes: rows [row_base, row_base + rows_local) of out are written,
// the other ranks' rows zeroed. 16 rows per CTA (grid = rows_local / 16 = 80,
// one per SM), 16 warps. A unit is (distinct expert q, 4-row slab): one warp,
// lane = (row r = lane / 8, sub s = lane % 8), K block j = 0..8 -> group
// s + 8 j. Each block is decoded once; member 0 of the expert always runs,
// members 1..5 behind a warp-uniform early exit. Units are claimed
// dynamically one ahead, and the next unit's raw blocks are loaded into the
// registers of the current unit's blocks right after each is decoded.
constexpr int kDnRows = 16, kDnWarps = 16;

template <int Type, int M>
__global__ void __launch_bounds__(32 * kDnWarps, 1)
    down_rows_fast(half* __restrict__ out, const uint8_t* __restrict__ gathered,
                   const uint8_t* __restrict__ down, int row_stride,
                   int rows_local, int row_base,
                   const uint32_t* __restrict__ fbook, int classes,
                   const int* __restrict__ plan) {
  using FB = FastBook<Type>;
  constexpr int LG = 18, N = 5120, TOP_K = 6, ROUTES = M * TOP_K;
  constexpr int RS = (ROUTES * 648 + 255) / 256 * 256, QB = ROUTES * 576;
  __shared__ __align__(16) uint32_t book[FB::kSmemWords];
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
  // First unit's weights in flight while the table is built.
  const int units = 4 * __ldg(plan);
  int u = warp;
  DnRaw<Type> raw;
  using Ptr = typename DnRaw<Type>::Ptr;
  if (u < units) {
    const Ptr p0(down + (int64_t(__ldg(plan + 1 + (u >> 2))) * rows_local +
                         row0 + (u & 3) * 4 + r4) *
                            row_stride,
                 sub);
    raw.set(p0);
#pragma unroll
    for (int j = 0; j < 9; ++j) raw.load(j, p0);
  }
  // Lane byte offsets into the replicated tables, kept in registers (opaque
  // to the compiler, which would otherwise re-mask the lane per lookup).
  uint32_t lg = uint32_t(lane) * 4, ls = uint32_t(lane & 15) * 4;
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+r"(lg), "+r"(ls));
#endif
  fill_book<Type>(book, fbook, t, 32 * kDnWarps);
  __syncthreads();
  const int nu = s_nu;
  if (warp == kDnWarps - 1) {
    // Old accumulation order of token k: class q % classes, then q, then c.
    for (int i = lane; i < 216; i += 32) {
      const int q = i / 6, c = i % 6;
      if (q < nu && c < s_cnt[q])
        s_key[s_route[q][c]] = (q % classes) << 16 | q << 3 | c;
    }
  }
  const uint8_t* const gq = gathered + 32 * sub;
  const float* const gd = reinterpret_cast<const float*>(gathered + QB) + sub;
  while (u < units) {
    int nxt = 0;
    if (lane == 0) nxt = atomicAdd(&s_next, 1);
    nxt = __shfl_sync(0xffffffffU, nxt, 0);
    const bool more = nxt < units;
    Ptr np(down + (more ? (int64_t(s_e[nxt >> 2]) * rows_local + row0 +
                           (nxt & 3) * 4 + r4) *
                              row_stride
                        : 0),
           sub);
    np.pin();
    const int q = u >> 2, slab = u & 3, cnt = s_cnt[q];
    int rt[6];
#pragma unroll
    for (int c = 0; c < 6; ++c) rt[c] = s_route[q][c];
    const uint8_t* const pq0 = gq + rt[0] * (LG * 32);
    const float* const pd0 = gd + rt[0] * LG;
    float p[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 9; ++j) {
      uint32_t w[8];
      float d;
      int m0, m1;
      raw.decode(j, book, lg, ls, sub, w, d, m0, m1);
      if (more) raw.load(j, np);
      const int rank = act_rank(j, sub);  // (sub + 8 j) / 18
      const int qo = 256 * j + rank * (RS - LG * 32);
      const int dof = 8 * j + rank * ((RS - LG * 4) / 4);
      {
        int qv[8];
        float xd;
        load_act(pq0 + qo, pd0 + dof, qv, xd);
        p[0] = fast_dot(w, qv, d, xd, m0, m1, p[0]);
      }
#pragma unroll
      for (int c = 1; c < 6; ++c) {
        if (c >= cnt) break;
        int qv[8];
        float xd;
        load_act(gq + rt[c] * (LG * 32) + qo, gd + rt[c] * LG + dof, qv, xd);
        p[c] = fast_dot(w, qv, d, xd, m0, m1, p[c]);
      }
    }
#pragma unroll
    for (int c = 0; c < 6; ++c) {
      if (c >= cnt) break;
      p[c] += __shfl_xor_sync(0xffffffffU, p[c], 1);
      p[c] += __shfl_xor_sync(0xffffffffU, p[c], 2);
      p[c] += __shfl_xor_sync(0xffffffffU, p[c], 4);
      if (sub == 0) s_v[rt[c]][slab * 4 + r4] = __float2half_rn(p[c]);
    }
    if (more) raw.set(np);
    u = nxt;
  }
  __syncthreads();
  if (t < kDnRows * M) {
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
}  // namespace v41k_fast
