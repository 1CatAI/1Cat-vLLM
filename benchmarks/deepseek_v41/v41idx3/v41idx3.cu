// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// v41idx3: the decode index scorer of DeepSeek-V4.1 on SM70 with its dot products on the tensor cores, BITWISE
// equal to v41idx::index_scores_tiled (#1372). Selected by VLLM_V41_INDEX_TC=1 together with
// VLLM_V41_INDEX_TILED=1 (vllm/models/deepseek_v41/index_tiled.py). Host proof: test_v41idx3_host.py beside this
// file; GPU check: check_v41idx3.py.
//
// Contract (v41idx.cu's, unchanged):
//   q   fp16 [tokens * 8, 128] (token-major rows), w fp16 [tokens * 8]
//   keys uint8 [capacity, 68]: 64 B E2M1 nibbles (even element low) + 4 biased power-of-two scale exponents, one
//        per 32 elements
//   positions int64 [>= tokens]; end = min((positions[tokens - 1] + 1) / ratio, capacity), capacity = keys.size(0)
//   out fp16, row stride out.size(1): out[t, n] for n < end only =
//     fp16( sum_h fp16( relu( fp16( q[t,h] . k[n] ) ) * w[t,h] ) )
//   with q . k accumulated as ONE fmaf chain over d = 0..127 in order, from 0.f, the key values rounded to fp16
//   first, and the head sum in h order. Columns n >= end are never written.
//
// How the bits can be the serving scorer's although the tensor cores add otherwise (four products at a time,
// truncating): when every product of a dot product is a multiple of one power of two G and every partial sum is
// below 2^23 G, NO addition rounds or truncates in either chain, so both give the real sum. The engine quantizes q
// and the keys to E2M1 values times a power of two per 32 dimensions, which makes that the usual case (2026-11-18:
// 100.00 % of 83,280,384 dot products of one decode step after a 127,677-token prompt). The kernel does not assume
// it: a GUARD
// decides per (token, key), in integers, and a cell it does not pass is recomputed by the serving chain itself.
//
//   GUARD. Row r of q: over its 32-dimension groups that are not all zero, Eq = the exponent of the largest
//   magnitude, lq = the exponent of the lowest set bit; a row holding an infinity, a NaN or a subnormal is not
//   taken. A[token] = max over its 8 rows of (max_g Eq - min_g lq). Key n: ek = its four scale bytes - 127; taken
//   only if all four are in 114..140 (then every value is zero or a NORMAL fp16, at most 6 * 2^ek, a multiple of
//   2^(ek - 1)); B[n] = max ek - min ek. A cell passes the FIRST LOOK if A + B <= 11 (one add and one compare a
//   cell). A cell that does not gets a SECOND LOOK per head: the same rule for that row and that key alone,
//   max_g(Eq + ek) - min_g(lq + ek) <= 11 over the row's nonzero groups (the first look is this made coarser
//   twice: the token's worst row, and the key's bytes whatever the row's zero groups). A head that passes keeps
//   its tensor-core value; a head that does not is recomputed by the serving chain. Then for every passing row and
//   group g with a nonzero product: the products are multiples of G = 2^(min_g(lq + ek) - 1) and every partial sum
//   of either chain is at most 384 * sum_g 2^(Eq + ek) < 2^(max_g(Eq + ek) + 11) <= 2^23 G. The measured adder of
//   the card (A5/g0z/ps/RZ/p24: five terms truncated toward zero at 2^(Emax - 23), summed, rounded toward zero to
//   24 bits; a probe of single mma instructions on one V100, 2026-11-18) then loses nothing: its truncation unit
//   is at most G / 2.
//   The serving epilogue's relu is `s > 0.f ? s : 0.f`, so a zero's sign does not reach the output.
//
//   LAYOUT (measured by the same probe: 0 of 64 outputs off): on sm_70 each quad pair of a warp (the 8 lanes with
//   one value of (lane >> 2) & 3) is its own 8 x 8 x 4 product. Thread slot s = (lane & 3) + 4 * (lane >> 4)
//   holds A's row s and B's column s, k in register order; output (m, n) is in slot (m&1)|(n&2)|(m&4), element
//   (n&1)|(m&2)|(n&4). Here: a warp is one token, A's 8 rows are its 8 heads, a quad pair's 8 columns are 8 keys;
//   a block of 64 keys is two accumulator sets of four quad pairs.
//
//   variant bit 0  NOGUARD  measuring only: no guard, no fallback (the output is not trusted)
//   variant bit 1  FORCE    every cell goes through the fallback (the serving chain): proves that path bitwise
//   variant bits 2-3 CTAS   the grid is (2 + this field) CTAs per SM
//
// counters: int64 [>= 4]; a call adds to counters[0] the number of (token, key) cells the first look did not
// pass, to counters[1] the number of their heads recomputed by the serving chain, and, if it has a live key, 1 to
// counters[2] and tokens * end (the cells it writes) to counters[3].
//
// The stage, guard, fragment, epilogue and fallback bodies are __host__ __device__: v41idx3_cpu_ops.cu runs them
// on the CPU with the CTA schedule and with the mma replaced by the measured adder (tc_cell below, itself checked
// against the bits the card recorded).

#include <cstdint>
#include <cstring>
#include <cmath>
#include <climits>
#include <cuda_fp16.h>

#if !defined(V41IDX_HOST_TEST) && !defined(V41IDX3_KERNEL_ONLY)
  #include <mutex>
  #include <torch/all.h>
  #include <ATen/cuda/CUDAContext.h>
  #include <c10/cuda/CUDAGuard.h>
  #include <c10/cuda/CUDAException.h>
#endif

namespace v41idx3 {

constexpr int KB = 64;         // keys per block
constexpr int HEADS = 8;       // A's rows: the token's heads
constexpr int QS = 132;        // q row stride in halves (264 B: 8 rows fall on 16 different banks)
constexpr int KS = 132;        // key row stride in halves
constexpr int KEY_BYTES = 68;
constexpr int MAX_TOKENS = 8;
constexpr int MAX_THREADS = 32 * MAX_TOKENS;
#ifdef V41IDX3_MUTANT_LIMIT
constexpr int GUARD_LIMIT = 40;  // host-test mutant only: passes cells whose chains can round (must be caught)
#else
constexpr int GUARD_LIMIT = 11;
#endif
constexpr int BYTE_LO = 114, BYTE_HI = 140;
constexpr int NOT_TAKEN = 100;   // A of a token or B of a key the guard does not take
constexpr int8_t GROUP_ZERO = -128, GROUP_BAD = 127;

__host__ __device__ constexpr int smem_bytes(int tokens) {
  return tokens * HEADS * QS * 2    // q, fp16
         + tokens * HEADS * 4       // w, fp32
         + KB * KS * 2              // the block's keys, fp16
         + tokens * KB * HEADS * 2  // per token [key][head]: fp16( relu(fp16(dot)) * w )
         + tokens * HEADS * 4 * 2   // per row and group: Eq, lq
         + MAX_TOKENS * 4           // A per token
         + KB                       // B per key
         + KB * 4;                  // the keys' scale bytes
}

struct Smem {
  __half* qs;
  float* ws;
  __half* ks;
  __half* hs;
  int8_t* ge;
  int8_t* gl;
  int* at;
  uint8_t* kb;
  uint8_t* ksc;
};
__host__ __device__ inline Smem carve(unsigned char* base, int tokens) {
  const int rows = tokens * HEADS;
  Smem s;
  s.qs = reinterpret_cast<__half*>(base);
  s.ws = reinterpret_cast<float*>(base + rows * QS * 2);
  s.ks = reinterpret_cast<__half*>(s.ws + rows);
  s.hs = s.ks + KB * KS;
  s.ge = reinterpret_cast<int8_t*>(s.hs + tokens * KB * HEADS);
  s.gl = s.ge + rows * 4;
  s.at = reinterpret_cast<int*>(s.gl + rows * 4);
  s.kb = reinterpret_cast<uint8_t*>(s.at + MAX_TOKENS);
  s.ksc = s.kb + KB;
  return s;
}

__host__ __device__ inline float bits_float(uint32_t u) {
#ifdef __CUDA_ARCH__
  return __uint_as_float(u);
#else
  float f;
  memcpy(&f, &u, 4);
  return f;
#endif
}

__host__ __device__ inline uint32_t half_bits(__half h) {
  unsigned short u;
  memcpy(&u, &h, 2);
  return u;
}

__host__ __device__ inline int ctz32(uint32_t x) {
#ifdef __CUDA_ARCH__
  return __clz(__brev(x));
#else
  return __builtin_ctz(x);
#endif
}

// v41idx.cu's live_end: stale positions cannot extend the key allocation.
__host__ __device__ inline int64_t live_end(int64_t position, int ratio, int64_t capacity) {
  const int64_t end = (position + 1) / ratio;
  return end < capacity ? end : capacity;
}

// v41idx.cu's e2m1 (the reference's v41k_e2m1), operation for operation.
__host__ __device__ inline float e2m1(int code) {
  const int m = code & 7;
  const float mag = m < 2 ? 0.5f * m
                          : bits_float(uint32_t(((m >> 1) + 126) << 23) |
                                       uint32_t((m & 1) << 22));
  return (code & 8) ? -mag : mag;
}

// q rows -> shared rows, the fp16 input itself; w -> ws (fp32, exactly __half2float of the input).
__host__ __device__ inline void stage_q(int tid, int nt, const __half* q, const __half* w, int rows, __half* qs,
                                        float* ws) {
  for (int i = tid; i < rows * 128; i += nt) {
    const int r = i / 128, d = i % 128;
    qs[r * QS + d] = q[r * 128 + d];
  }
  for (int r = tid; r < rows; r += nt) ws[r] = __half2float(w[r]);
}

// GUARD, q side: one 32-dimension group of one row.
__host__ __device__ inline void scan_group(const __half* q, int r, int g, int8_t* ge, int8_t* gl) {
  int emax = GROUP_ZERO, lmin = 127;
  bool bad = false;
  for (int i = 0; i < 32; ++i) {
    const uint32_t b = half_bits(q[r * 128 + g * 32 + i]);
    const int e = int((b >> 10) & 31u);
    const uint32_t m = b & 1023u;
    if (e == 0) {
      if (m) bad = true;  // a subnormal: not taken
      continue;           // a zero: no product
    }
    if (e == 31) {
      bad = true;  // an infinity or a NaN
      continue;
    }
    const int E = e - 15;
    const int l = E - 10 + ctz32(m | 1024u);
    emax = E > emax ? E : emax;
    lmin = l < lmin ? l : lmin;
  }
  ge[r * 4 + g] = int8_t(bad ? GROUP_BAD : emax);
  gl[r * 4 + g] = int8_t(lmin);
}

// GUARD, q side: A of one token from its rows' groups.
__host__ __device__ inline int token_a(int t, const int8_t* ge, const int8_t* gl) {
  int a = 0;
  for (int h = 0; h < HEADS; ++h) {
    int emax = GROUP_ZERO, lmin = 127;
    for (int g = 0; g < 4; ++g) {
      const int e = ge[(t * HEADS + h) * 4 + g];
      if (e == GROUP_BAD) return NOT_TAKEN;
      if (e == GROUP_ZERO) continue;
      const int l = gl[(t * HEADS + h) * 4 + g];
      emax = e > emax ? e : emax;
      lmin = l < lmin ? l : lmin;
    }
    if (emax == GROUP_ZERO) continue;  // a row of zeros: every product is zero
    a = emax - lmin > a ? emax - lmin : a;
  }
  return a;
}

// v41idx.cu's stage_keys, operation for operation: keys [n0, n0 + KB) into ks (fp16), zeros past end.
__host__ __device__ inline void stage_keys(int tid, int nt, const uint8_t* keys, int64_t n0, int64_t end,
                                           __half* ks) {
  for (int i = tid; i < KB * 16; i += nt) {
    const int k = i / 16, wd = i % 16;
    const int64_t n = n0 + k;
    __half h[8];
    if (n < end) {
      const uint8_t* row = keys + n * KEY_BYTES;
      uint32_t word;
#ifdef __CUDA_ARCH__
      word = *reinterpret_cast<const uint32_t*>(row + 4 * wd);
#else
      memcpy(&word, row + 4 * wd, 4);
#endif
      const float scale = bits_float(uint32_t(row[64 + wd / 4]) << 23);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int byte = int((word >> (8 * c)) & 255u);
        const float lo = e2m1(byte & 15) * scale;
        const float hi = e2m1((byte >> 4) & 15) * scale;
        h[2 * c] = __float2half_rn(lo);
        h[2 * c + 1] = __float2half_rn(hi);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 8; ++c) h[c] = __float2half_rn(0.f);
    }
    __half* dst = ks + k * KS + 8 * wd;
#ifdef __CUDA_ARCH__
    uint2 a, b;
    memcpy(&a, h, 8);
    memcpy(&b, h + 4, 8);
    reinterpret_cast<uint2*>(dst)[0] = a;
    reinterpret_cast<uint2*>(dst)[1] = b;
#else
    memcpy(dst, h, 16);
#endif
  }
}

// GUARD, key side: B of the block's keys from their scale bytes, and the bytes themselves for the second look
// (a key past end is staged as zeros: B = 0, and it is never written).
__host__ __device__ inline void stage_kb(int tid, int nt, const uint8_t* keys, int64_t n0, int64_t end,
                                         uint8_t* kb, uint8_t* ksc) {
  for (int k = tid; k < KB; k += nt) {
    const int64_t n = n0 + k;
    int b = 0;
    if (n < end) {
      const uint8_t* sc = keys + n * KEY_BYTES + 64;
#pragma unroll
      for (int g = 0; g < 4; ++g) ksc[k * 4 + g] = sc[g];
      int lo = sc[0], hi = sc[0];
#pragma unroll
      for (int g = 1; g < 4; ++g) {
        lo = sc[g] < lo ? sc[g] : lo;
        hi = sc[g] > hi ? sc[g] : hi;
      }
#ifdef V41IDX3_MUTANT_RANGE
      b = hi - lo;  // host-test mutant only: the scale bytes' range is not checked (must be caught by the count)
#else
      b = (lo >= BYTE_LO && hi <= BYTE_HI) ? hi - lo : NOT_TAKEN;
#endif
    }
    kb[k] = uint8_t(b);
  }
}

// One thread's operand registers of one mma: four consecutive halves, k in register order.
struct Frag {
  uint32_t x, y;
};
__host__ __device__ inline Frag load_frag(const __half* p) {
  Frag f;
#ifdef __CUDA_ARCH__
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  f.x = raw.x;
  f.y = raw.y;
#else
  memcpy(&f, p, 8);
#endif
  return f;
}
__host__ __device__ inline int slot_of(int lane) { return (lane & 3) + ((lane & 16) ? 4 : 0); }
__host__ __device__ inline int quad_of(int lane) { return (lane >> 2) & 3; }
// A: head `slot` of token t, dimensions d..d+3.
__host__ __device__ inline Frag frag_a(const __half* qs, int t, int lane, int d) {
  return load_frag(qs + (t * HEADS + slot_of(lane)) * QS + d);
}
// B: key (set * 4 + quad) * 8 + slot of the block, dimensions d..d+3.
__host__ __device__ inline Frag frag_b(const __half* ks, int set, int lane, int d) {
  return load_frag(ks + ((set * 4 + quad_of(lane)) * 8 + slot_of(lane)) * KS + d);
}

// The thread's 16 accumulators -> the token's [key][head] table: the serving epilogue's per-head operations.
__host__ __device__ inline void heads_out(int t, int lane, const float acc[2][8], const float* ws, __half* hs) {
  const int s = slot_of(lane), qd = quad_of(lane);
#pragma unroll
  for (int set = 0; set < 2; ++set)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int m = (s & 1) | (e & 2) | (s & 4);  // head
      const int n = (e & 1) | (s & 2) | (e & 4);  // key within the quad pair's eight
      const int kk = (set * 4 + qd) * 8 + n;
      float sv = __half2float(__float2half_rn(acc[set][e]));
      sv = sv > 0.f ? sv : 0.f;
      hs[(t * KB + kk) * HEADS + m] = __float2half_rn(sv * ws[t * HEADS + m]);
    }
}

// The fallback: one head's term of one (token, key) cell by the serving chain on the staged values (v41idx.cu's
// tile for one head and key): fp16( relu( fp16( q . k ) ) * w ) as fp32.
__host__ __device__ inline float serving_head(int row, int kk, const __half* qs, const __half* ks, const float* ws) {
  const __half* qp = qs + row * QS;
  const __half* kp = ks + kk * KS;
  float acc = 0.f;
#ifdef V41IDX3_MUTANT_FALLBACK
  for (int d = 127; d >= 0; --d)  // host-test mutant only: the chain backwards (must be caught)
#else
  for (int d = 0; d < 128; ++d)
#endif
    acc = fmaf(__half2float(qp[d]), __half2float(kp[d]), acc);
  float s = __half2float(__float2half_rn(acc));
  s = s > 0.f ? s : 0.f;
  return __half2float(__float2half_rn(s * ws[row]));
}

// GUARD, the second look, for one row and one key of a cell the first look did not pass: the same rule without
// the two coarsenings (the token's worst row; a key's bytes whatever the row's zero groups).
__host__ __device__ inline bool pair_passes(const Smem& s, int row, int kk) {
  if (s.kb[kk] == NOT_TAKEN) return false;  // a scale byte out of range
  int top = INT_MIN, low = INT_MAX;
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const int e = s.ge[row * 4 + g];
    if (e == GROUP_BAD) return false;
    if (e == GROUP_ZERO) continue;
    const int ek = int(s.ksc[kk * 4 + g]) - 127;
    const int l = int(s.gl[row * 4 + g]) + ek;
    top = e + ek > top ? e + ek : top;
    low = l < low ? l : low;
  }
  return top == INT_MIN || top - low <= GUARD_LIMIT;
}

struct Fell {
  int cells;  // cells the first look did not pass
  int heads;  // heads of those cells recomputed by the serving chain
};

// One thread writes two keys of its token: the head sum in h order (the serving epilogue's). In a cell the first
// look does not pass, a head the second look does not pass either is the serving chain's; the others stay.
template <bool GUARD, bool FORCE>
__host__ __device__ inline Fell cells_out(int t, int lane, const Smem& s, int64_t n0, int64_t end, __half* out,
                                          int64_t n_out) {
  Fell fell = {0, 0};
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int kk = lane + 32 * j;
    const int64_t n = n0 + kk;
    if (n >= end) continue;
    const __half* hp = s.hs + (t * KB + kk) * HEADS;
    float total = 0.f;
    if (FORCE || (GUARD && s.at[t] + int(s.kb[kk]) > GUARD_LIMIT)) {
      ++fell.cells;
      for (int h = 0; h < HEADS; ++h) {
        if (!FORCE && pair_passes(s, t * HEADS + h, kk)) {
          total += __half2float(hp[h]);
        } else {
          total += serving_head(t * HEADS + h, kk, s.qs, s.ks, s.ws);
          ++fell.heads;
        }
      }
    } else {
#pragma unroll
      for (int h = 0; h < HEADS; ++h) total += __half2float(hp[h]);
    }
    out[int64_t(t) * n_out + n] = __float2half_rn(total);
  }
  return fell;
}

// The card's adder for one output cell of one mma (model A5/g0z/ps/RZ/p24), in exact double arithmetic: every
// quantity is an integer multiple of a power of two far inside 53 bits. Host only; v41idx3_cpu_ops.cu exposes it
// so that a test compares it with the 6,500 single instructions and 32 chains a V100 recorded.
inline float tc_cell(float c, const float a[4], const float b[4]) {
  double p[4];
  int emax = INT_MIN;
  // An infinity or a NaN among the operands: the model was not measured there. The scorer never uses such a
  // cell (the guard does not take its row or key); the host returns a NaN for it.
  if (!std::isfinite(c)) return NAN;
  for (int i = 0; i < 4; ++i)
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return NAN;
  for (int i = 0; i < 4; ++i) {
    p[i] = double(a[i]) * double(b[i]);
    if (p[i] != 0.0) {
      const int e = std::ilogb(a[i]) + std::ilogb(b[i]);
      emax = e > emax ? e : emax;
    }
  }
  if (c != 0.f) {
    const int e = std::ilogb(c);
    emax = e > emax ? e : emax;
  }
  if (emax == INT_MIN) return 0.f;
  const double quantum = std::ldexp(1.0, emax - 23);
  double total = std::trunc(double(c) / quantum);
  for (int i = 0; i < 4; ++i) total += std::trunc(p[i] / quantum);
  total *= quantum;
  if (total == 0.0) return 0.f;
  const int e = std::ilogb(total);
  const double ulp = std::ldexp(1.0, (e > -126 ? e : -126) - 23);
  return float(std::trunc(total / ulp) * ulp);
}

#ifndef V41IDX_HOST_TEST
__device__ __forceinline__ void mma(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

template <bool GUARD, bool FORCE>
__global__ void __launch_bounds__(MAX_THREADS)
    index_scores_tc(const __half* __restrict__ q, const __half* __restrict__ w, const uint8_t* __restrict__ keys,
                    const int64_t* __restrict__ positions, __half* __restrict__ out, int tokens, int64_t n_out,
                    int ratio, int64_t capacity, unsigned long long* __restrict__ fell_total) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int64_t end = live_end(positions[tokens - 1], ratio, capacity);
  int64_t n0 = int64_t(blockIdx.x) * KB;
  if (n0 >= end) return;
  const int rows = tokens * HEADS;
  const Smem s = carve(smem, tokens);
  const int tid = threadIdx.x, nt = blockDim.x;  // nt = 32 * tokens = 4 * rows
  const int t = tid >> 5, lane = tid & 31;
  stage_q(tid, nt, q, w, rows, s.qs, s.ws);
  if (GUARD) scan_group(q, tid >> 2, tid & 3, s.ge, s.gl);
  __syncthreads();
  if (GUARD && lane == 0) s.at[t] = token_a(t, s.ge, s.gl);
  Fell fell = {0, 0};
  for (; n0 < end; n0 += int64_t(gridDim.x) * KB) {
    stage_keys(tid, nt, keys, n0, end, s.ks);
    if (GUARD) stage_kb(tid, nt, keys, n0, end, s.kb, s.ksc);
    __syncthreads();
    float acc[2][8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[0][e] = acc[1][e] = 0.f;
    if (!FORCE) {
#pragma unroll 4
      for (int d = 0; d < 128; d += 4) {
        const Frag a = frag_a(s.qs, t, lane, d);
        const Frag b0 = frag_b(s.ks, 0, lane, d);
        const Frag b1 = frag_b(s.ks, 1, lane, d);
        mma(acc[0], a.x, a.y, b0.x, b0.y);
        mma(acc[1], a.x, a.y, b1.x, b1.y);
      }
      heads_out(t, lane, acc, s.ws, s.hs);
      __syncwarp();
    }
    const Fell f = cells_out<GUARD, FORCE>(t, lane, s, n0, end, out, n_out);
    fell.cells += f.cells;
    fell.heads += f.heads;
    __syncthreads();
  }
  if (fell.cells) {
    atomicAdd(fell_total, static_cast<unsigned long long>(fell.cells));
    atomicAdd(fell_total + 1, static_cast<unsigned long long>(fell.heads));
  }
  if (blockIdx.x == 0 && tid == 0) {  // once a call that has a live key (CTA 0 has one whenever end > 0)
    atomicAdd(fell_total + 2, 1ull);
    atomicAdd(fell_total + 3, static_cast<unsigned long long>(tokens) * static_cast<unsigned long long>(end));
  }
}

#ifdef V41IDX3_KERNEL_ONLY
// Compile-only instantiation (no torch): nvcc -c -DV41IDX3_KERNEL_ONLY ... -Xptxas -v prints each kernel's
// registers and spills on a machine with no card.
template __global__ void index_scores_tc<true, false>(const __half*, const __half*, const uint8_t*, const int64_t*,
                                                       __half*, int, int64_t, int, int64_t, unsigned long long*);
template __global__ void index_scores_tc<false, false>(const __half*, const __half*, const uint8_t*, const int64_t*,
                                                        __half*, int, int64_t, int, int64_t, unsigned long long*);
template __global__ void index_scores_tc<true, true>(const __half*, const __half*, const uint8_t*, const int64_t*,
                                                      __half*, int, int64_t, int, int64_t, unsigned long long*);
#endif
#endif

}  // namespace v41idx3

#if !defined(V41IDX_HOST_TEST) && !defined(V41IDX3_KERNEL_ONLY)
namespace {

int sm_count(int device) {
  static std::mutex mu;
  static int counts[64] = {0};
  std::lock_guard<std::mutex> lock(mu);
  TORCH_CHECK(device >= 0 && device < 64, "v41idx3: device index out of range");
  if (counts[device] == 0) {
    int c = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&c, cudaDevAttrMultiProcessorCount, device));
    counts[device] = c;
  }
  return counts[device];
}

template <bool GUARD, bool FORCE>
void launch(torch::Tensor& out, torch::Tensor& q, torch::Tensor& w, torch::Tensor& keys, torch::Tensor& positions,
            int ratio, int ctas, torch::Tensor& counters) {
  const int device = out.get_device();
  const int tokens = int(out.size(0));
  C10_CUDA_CHECK(cudaFuncSetAttribute(v41idx3::index_scores_tc<GUARD, FORCE>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      v41idx3::smem_bytes(v41idx3::MAX_TOKENS)));
  const int64_t blocks = (keys.size(0) + v41idx3::KB - 1) / v41idx3::KB;
  const int grid = int(std::min<int64_t>(blocks, int64_t(ctas) * sm_count(device)));
  auto stream = at::cuda::getCurrentCUDAStream();
  v41idx3::index_scores_tc<GUARD, FORCE><<<grid, 32 * tokens, v41idx3::smem_bytes(tokens), stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()), reinterpret_cast<const __half*>(w.data_ptr()),
      keys.data_ptr<uint8_t>(), positions.data_ptr<int64_t>(), reinterpret_cast<__half*>(out.data_ptr()), tokens,
      out.size(1), ratio, keys.size(0), reinterpret_cast<unsigned long long*>(counters.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void index_scores_tc(torch::Tensor out, torch::Tensor q, torch::Tensor w, torch::Tensor keys,
                     torch::Tensor positions, int64_t heads, int64_t ratio, int64_t variant,
                     torch::Tensor counters) {
  const int64_t tokens = out.dim() == 2 ? out.size(0) : -1;
  TORCH_CHECK(out.is_cuda() && q.device() == out.device() && w.device() == out.device() &&
                  keys.device() == out.device() && positions.device() == out.device() &&
                  counters.device() == out.device(),
              "v41idx3: all tensors must be on one CUDA device");
  TORCH_CHECK(out.dim() == 2 && out.scalar_type() == torch::kHalf && out.stride(1) == 1 &&
                  out.stride(0) == out.size(1),
              "v41idx3: out must be fp16 [tokens, n_out] with row stride n_out");
  TORCH_CHECK(heads == 8, "v41idx3: 8 index heads per rank only (a tile's 8 rows), got ", heads);
  TORCH_CHECK(tokens >= 1 && tokens <= v41idx3::MAX_TOKENS, "v41idx3: need 1 <= tokens <= 8");
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.is_contiguous() && q.dim() == 2 &&
                  q.size(0) == tokens * heads && q.size(1) == 128,
              "v41idx3: q must be contiguous fp16 [tokens * 8, 128]");
  TORCH_CHECK(w.scalar_type() == torch::kHalf && w.is_contiguous() && w.numel() == tokens * heads,
              "v41idx3: w must be contiguous fp16 [tokens * 8]");
  TORCH_CHECK(keys.scalar_type() == torch::kUInt8 && keys.is_contiguous() && keys.dim() == 2 &&
                  keys.size(1) == 68 && keys.size(0) > 0,
              "v41idx3: keys must be contiguous uint8 [capacity > 0, 68]");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 && positions.is_contiguous() &&
                  positions.numel() >= tokens,
              "v41idx3: positions must be contiguous int64 with >= tokens entries");
  TORCH_CHECK(counters.scalar_type() == torch::kInt64 && counters.is_contiguous() && counters.numel() >= 4,
              "v41idx3: counters must be contiguous int64 with >= 4 entries");
  TORCH_CHECK(out.size(1) >= keys.size(0), "v41idx3: output width must cover key capacity");
  TORCH_CHECK(ratio >= 1, "v41idx3: ratio must be >= 1");
  TORCH_CHECK(variant >= 0 && variant < 16 && (variant & 3) != 3,
              "v41idx3: variant must be 0..15 and not NOGUARD with FORCE, got ", variant);
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const int ctas = 2 + int((variant >> 2) & 3);
  if (variant & 2) launch<true, true>(out, q, w, keys, positions, int(ratio), ctas, counters);
  else if (variant & 1) launch<false, false>(out, q, w, keys, positions, int(ratio), ctas, counters);
  else launch<true, false>(out, q, w, keys, positions, int(ratio), ctas, counters);
}

int64_t version() { return 1; }

}  // namespace

TORCH_LIBRARY(v41idx3, m) {
  m.def(
      "index_scores_tc(Tensor(a!) out, Tensor q, Tensor w, Tensor keys, Tensor positions, int heads, int ratio, "
      "int variant, Tensor(b!) counters) -> ()",
      &index_scores_tc);
  m.def("version() -> int", &version);
}
// Shape-only capture/fake implementation (as v41idx's). out and counters are mutated in place.
namespace {
void index_scores_tc_meta(torch::Tensor out, torch::Tensor q, torch::Tensor w, torch::Tensor keys,
                          torch::Tensor positions, int64_t heads, int64_t ratio, int64_t variant,
                          torch::Tensor counters) {}
}  // namespace
TORCH_LIBRARY_IMPL(v41idx3, Meta, m) { m.impl("index_scores_tc", &index_scores_tc_meta); }
#endif
