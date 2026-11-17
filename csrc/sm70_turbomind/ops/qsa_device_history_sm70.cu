// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <algorithm>
#include <type_traits>
// Direct device E4M3/FP16 history: FP16 QK operands, FP32 QK/PV,
// probability and merge. No protected-hot ownership or staging is required.
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace device_qsa {
constexpr int D = 256, H = 6, C = 64, PAD = 264;
struct Args {
  const half* q;
  const unsigned char* history;
  const float* scales;
  const int* indices;
  const int* table;
  const int* requests;
  const int64_t* positions;
  const int* lengths;
  const half* gate;
  const half* hot;          // [hot_tokens, 2, D] FP16 or encoded E4M3 bytes
  const float* hot_scales;  // [hot_tokens, 2], encoded hot storage only
  const int* resolved;      // [M, width] hot token, -1 host, -2 invalid
  const int32_t* page_slots;
  const int32_t* tags;
  int hot_pages;
  int64_t page_slots_len;
  half* out;
  float* partial;
  float* state;
  int M, blocks, page, width, table_width, num_requests, splits, tiles;
  // rows_fp8 over one window of a history staged in parts: write the
  // window-normalized FP32 rows to `partial` and (max, sum) to `state`.
  int window;
  // merge over one window (FP16 history): window-normalized FP32 rows
  // and per-head (max, sum) instead of the gated output.
  float* wout;
  float* wstate;
  int64_t qs0, qs1, is0, ts0, os0, os1, gs0, gs1, rs0;
};

__device__ __forceinline__ float fast_exp2(float x) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
  return result;
}

__device__ __forceinline__ float e4m3(unsigned c) {
  const unsigned sign = (c & 128u) << 24;
  const unsigned magnitude = c & 127u;
  if (magnitude == 127u) return __uint_as_float(sign | 0x7fc00000u);
  const unsigned normal = (magnitude << 20) + (120u << 23);
  const float subnormal = float(magnitude & 7u) * 0x1p-9f;
  const unsigned bits = magnitude < 8u ? __float_as_uint(subnormal) : normal;
  return __uint_as_float(bits | sign);
}

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

// B row-major: lane holds B[k = lane & 3][n = (lane & 16 ? 4 : 0) + i] of its
// quad pair's 8 columns (i = 0..3, pairs in b0, b1); A and D as in mma().
__device__ __forceinline__ void mma_rr(float (&d)[8], uint32_t a0, uint32_t a1,
                                       uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// lut: the 256 E4M3 code values (e4m3(code)), in shared memory.
__device__ __forceinline__ uint4 decode8_raw(const uint2 packed, float scale,
                                             const float* lut) {
  uint4 result;
  unsigned* words = reinterpret_cast<unsigned*>(&result);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned p = j < 2 ? packed.x : packed.y;
    const unsigned code0 = (p >> (16 * (j & 1))) & 255u;
    const unsigned code1 = (p >> (16 * (j & 1) + 8)) & 255u;
    const half lo = __float2half_rn(__fmul_rn(lut[code0], scale));
    const half hi = __float2half_rn(__fmul_rn(lut[code1], scale));
    words[j] =
        unsigned(__half_as_ushort(lo)) | (unsigned(__half_as_ushort(hi)) << 16);
  }
  return result;
}

// E4M3 code -> its exact value without a table: sign and the 7 magnitude bits
// placed as FP16 bits give value * 2^-8 for normals and subnormals alike
// (code 0x7f, NaN in E4M3, decodes as 480). No shared-memory lookups, so
// no bank conflicts on random codes; identical to e4m3() otherwise.
__device__ __forceinline__ float e4m3_bits(unsigned c) {
  const unsigned short h =
      static_cast<unsigned short>(((c & 0x80u) << 8) | ((c & 0x7fu) << 7));
  return __half2float(__ushort_as_half(h)) * 256.f;
}

__device__ __forceinline__ uint4 decode8_bits(const uint2 packed, float scale) {
  uint4 result;
  unsigned* words = reinterpret_cast<unsigned*>(&result);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned p = j < 2 ? packed.x : packed.y;
    const unsigned code0 = (p >> (16 * (j & 1))) & 255u;
    const unsigned code1 = (p >> (16 * (j & 1) + 8)) & 255u;
    const half lo = __float2half_rn(__fmul_rn(e4m3_bits(code0), scale));
    const half hi = __float2half_rn(__fmul_rn(e4m3_bits(code1), scale));
    words[j] =
        unsigned(__half_as_ushort(lo)) | (unsigned(__half_as_ushort(hi)) << 16);
  }
  return result;
}

__device__ __forceinline__ float e4m3_exact(unsigned c, bool hot_src) {
  if (!hot_src && (c & 0x7fu) == 0x7fu)
    return __uint_as_float(((c & 0x80u) << 24) | 0x7fc00000u);
  return e4m3_bits(c);
}

__device__ __forceinline__ uint4 decode8_exact(const uint2 packed, float scale,
                                               bool hot_src) {
  uint4 result;
  unsigned* words = reinterpret_cast<unsigned*>(&result);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned p = j < 2 ? packed.x : packed.y;
    const unsigned code0 = (p >> (16 * (j & 1))) & 255u;
    const unsigned code1 = (p >> (16 * (j & 1) + 8)) & 255u;
    const half lo =
        __float2half_rn(__fmul_rn(e4m3_exact(code0, hot_src), scale));
    const half hi =
        __float2half_rn(__fmul_rn(e4m3_exact(code1, hot_src), scale));
    words[j] =
        unsigned(__half_as_ushort(lo)) | (unsigned(__half_as_ushort(hi)) << 16);
  }
  return result;
}

// Two E4M3 codes (bytes `lo`, `hi` of `word`, selected by __byte_perm) as an
// FP16 pair holding value * 2^-8 exactly: sign to bit 15, the 7 magnitude
// bits to bits 7..13. Normals and subnormals alike; no rounding.
__device__ __forceinline__ uint32_t e4m3x2_half(uint32_t word,
                                                uint32_t selector) {
  const uint32_t x = __byte_perm(word, 0u, selector);  // [c0, 0, c1, 0]
  return ((x << 8) & 0x80008000u) | ((x << 7) & 0x3f803f80u);
}

// Eight E4M3 codes as four FP16 pairs of value * 2^-8 (unscaled).
__device__ __forceinline__ uint4 decode8_unscaled(const uint2 packed) {
  return make_uint4(
      e4m3x2_half(packed.x, 0x4140u), e4m3x2_half(packed.x, 0x4342u),
      e4m3x2_half(packed.y, 0x4140u), e4m3x2_half(packed.y, 0x4342u));
}

// Bit-identical fast path for encoded hot storage (no 0x7f NaN codes): decode
// 2 codes at a time via __byte_perm and fold 2^8 into scale256 = scale * 256.f.
__device__ __forceinline__ uint4 decode8_hot_fast(const uint2 packed,
                                                  float scale256) {
  const uint32_t h0 = e4m3x2_half(packed.x, 0x4140u);
  const uint32_t h1 = e4m3x2_half(packed.x, 0x4342u);
  const uint32_t h2 = e4m3x2_half(packed.y, 0x4140u);
  const uint32_t h3 = e4m3x2_half(packed.y, 0x4342u);
  const float2 f0 = __half22float2(*reinterpret_cast<const half2*>(&h0));
  const float2 f1 = __half22float2(*reinterpret_cast<const half2*>(&h1));
  const float2 f2 = __half22float2(*reinterpret_cast<const half2*>(&h2));
  const float2 f3 = __half22float2(*reinterpret_cast<const half2*>(&h3));
  const half2 r0 =
      __floats2half2_rn(__fmul_rn(f0.x, scale256), __fmul_rn(f0.y, scale256));
  const half2 r1 =
      __floats2half2_rn(__fmul_rn(f1.x, scale256), __fmul_rn(f1.y, scale256));
  const half2 r2 =
      __floats2half2_rn(__fmul_rn(f2.x, scale256), __fmul_rn(f2.y, scale256));
  const half2 r3 =
      __floats2half2_rn(__fmul_rn(f3.x, scale256), __fmul_rn(f3.y, scale256));
  return make_uint4(*reinterpret_cast<const uint32_t*>(&r0),
                    *reinterpret_cast<const uint32_t*>(&r1),
                    *reinterpret_cast<const uint32_t*>(&r2),
                    *reinterpret_cast<const uint32_t*>(&r3));
}

__device__ __forceinline__ uint4 decode8(const unsigned char* source,
                                         float scale, const float* lut) {
  return decode8_raw(__ldg(reinterpret_cast<const uint2*>(source)), scale, lut);
}

// Read one 8-value chunk of token K (kv=0) or V (kv=1): from the protected
// hot cache when resolved there, else from the authoritative history, which
// may be pinned host memory mapped into the device address space (UVA).
template <bool FP8, bool HOT, bool HOT_FP8 = false>
__device__ __forceinline__ uint4 load_kv(const Args& a, int64_t slot, int hot,
                                         int kv, int dim, const float* lut) {
  if (slot < 0) return make_uint4(0, 0, 0, 0);
  if constexpr (HOT) {
    if (hot >= 0) {
      if constexpr (HOT_FP8) {
        const auto* source = reinterpret_cast<const unsigned char*>(a.hot) +
                             (int64_t(hot) * 2 + kv) * D + dim;
        // Match the software bit decoder used to materialize FP16 hot values,
        // including the FP16 rounding before QK/PV arithmetic.
        return decode8_bits(__ldg(reinterpret_cast<const uint2*>(source)),
                            __ldg(a.hot_scales + int64_t(hot) * 2 + kv));
      } else {
        return __ldg(reinterpret_cast<const uint4*>(
            a.hot + (int64_t(hot) * 2 + kv) * D + dim));
      }
    }
  }
  const int block = slot / a.page, off = slot % a.page;
  const int64_t element = ((int64_t(block) * 2 + kv) * a.page + off) * D + dim;
  if constexpr (FP8)
    return decode8(a.history + element, __ldg(a.scales + slot * 2 + kv), lut);
  else
    return __ldg(reinterpret_cast<const uint4*>(
        reinterpret_cast<const half*>(a.history) + element));
}

template <bool FP8, bool HOT, bool HOT_FP8 = false>
__global__ __launch_bounds__(256) void partial(Args a) {
  const int row = blockIdx.y, split = blockIdx.x;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  __shared__ __align__(16) half query[8][PAD];
  __shared__ __align__(16) half kv[C][PAD];
  __shared__ float scores[4][8][C];
  __shared__ int64_t slots[C];
  __shared__ int hots[C];
  __shared__ float lut[256];
  if constexpr (FP8) lut[threadIdx.x] = e4m3(threadIdx.x);
  __shared__ float alpha[8];
  const int req = a.requests[row];
  const int64_t position = a.positions[row];
  const int length = req >= 0 && req < a.num_requests ? a.lengths[req] : 0;
  const int head = t / 32, d8 = (t % 32) * 8;
  const uint4 qv = head < H ? __ldg(reinterpret_cast<const uint4*>(
                                  a.q + row * a.qs0 + head * a.qs1 + d8))
                            : make_uint4(0, 0, 0, 0);
  *reinterpret_cast<uint4*>(&query[head][d8]) = qv;
  const int nt = warp & 1, kw = warp >> 1;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int nr =
      nt * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  constexpr int kChunks = C * 32 / 256;
  using Raw = std::conditional_t<FP8, uint2, uint4>;
  // Running softmax state of head `warp` (FP32), and this thread's
  // dimension t of every head's unnormalized output. With one tile per CTA
  // this is exactly the single-tile arithmetic.
  float running_max = __int_as_float(0xff800000), running_sum = 0.f;
  float result[H] = {};
  for (int tile = 0; tile < a.tiles; ++tile) {
    const int base = (split * a.tiles + tile) * C;
    if (base >= a.width) break;
    __syncthreads();  // the previous tile is done with slots, kv and scores
    if (t < C) {
      const int col = base + t;
      const int logical = col < a.width ? a.indices[row * a.is0 + col] : -1;
      int64_t slot = -1;
      if (req >= 0 && req < a.num_requests && logical >= 0 &&
          logical <= position && logical < length &&
          logical / a.page < a.table_width) {
        const int physical = a.table[req * a.ts0 + logical / a.page];
        if (physical >= 0 && physical < a.blocks)
          slot = int64_t(physical) * a.page + logical % a.page;
      }
      slots[t] = slot;
      int hot = HOT && slot >= 0 ? a.resolved[row * a.rs0 + col] : -1;
      if (HOT && hot == -1 && slot >= 0 && a.page_slots != nullptr &&
          a.tags != nullptr) {
        const int64_t p4 = slot >> 2;
        if (p4 < a.page_slots_len) {
          const int32_t reg = a.page_slots[p4];
          if (reg >= 0 && reg < a.hot_pages &&
              a.tags[reg] == static_cast<int32_t>(p4)) {
            hot = (reg << 2) | static_cast<int>(slot & 3);
          }
        }
      }
      hots[t] = hot;
    }
    __syncthreads();
    for (int v = t; v < C * 32; v += 256) {
      const int token = v / 32, dim = (v % 32) * 8;
      *reinterpret_cast<uint4*>(&kv[token][dim]) =
          load_kv<FP8, HOT, HOT_FP8>(a, slots[token], hots[token], 0, dim, lut);
    }
    __syncthreads();
    // Issue V loads ahead of QK and softmax so global/PCIe latency overlaps
    // tensor-core QK and score reduction (including FP16 hot-cache readers).
    Raw raw[kChunks];
    float raw_scale[kChunks];
    if constexpr (!HOT || !FP8) {
#pragma unroll
      for (int i = 0; i < kChunks; ++i) {
        const int v = t + i * 256, token = v / 32, dim = (v % 32) * 8;
        const int64_t slot = slots[token];
        raw[i] = Raw{};
        raw_scale[i] = 0.f;
        if (slot >= 0) {
          if constexpr (HOT && !FP8) {
            const int hot = hots[token];
            if (hot >= 0) {
              raw[i] = __ldg(reinterpret_cast<const uint4*>(
                  a.hot + (int64_t(hot) * 2 + 1) * D + dim));
            } else {
              const int block = slot / a.page, off = slot % a.page;
              const int64_t element =
                  ((int64_t(block) * 2 + 1) * a.page + off) * D + dim;
              raw[i] = __ldg(reinterpret_cast<const uint4*>(
                  reinterpret_cast<const half*>(a.history) + element));
            }
          } else {
            const int block = slot / a.page, off = slot % a.page;
            const int64_t element =
                ((int64_t(block) * 2 + 1) * a.page + off) * D + dim;
            if constexpr (FP8) {
              raw[i] =
                  __ldg(reinterpret_cast<const uint2*>(a.history + element));
              raw_scale[i] = __ldg(a.scales + slot * 2 + 1);
            } else {
              raw[i] = __ldg(reinterpret_cast<const uint4*>(
                  reinterpret_cast<const half*>(a.history) + element));
            }
          }
        }
      }
    }
    float acc[8] = {};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int k = kw * 64 + s * 16;
      const uint4 q0 = *reinterpret_cast<const uint4*>(&query[r8][k]);
      const uint4 q1 = *reinterpret_cast<const uint4*>(&query[r8][k + 8]);
      const uint4 k0 = *reinterpret_cast<const uint4*>(&kv[nr][k]);
      const uint4 k1 = *reinterpret_cast<const uint4*>(&kv[nr][k + 8]);
      mma(acc, q0.x, q0.y, k0.x, k0.y);
      mma(acc, q0.z, q0.w, k0.z, k0.w);
      mma(acc, q1.x, q1.y, k1.x, k1.y);
      mma(acc, q1.z, q1.w, k1.z, k1.w);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = nt * 32 + ((lane >> 2) & 3) * 8 +
                      ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
      scores[kw][h][col] = acc[e];
    }
    __syncthreads();
    for (int v = t; v < 8 * C; v += 256) {
      const int h = v / C, k = v % C;
      const float sum =
          ((scores[0][h][k] + scores[1][h][k]) + scores[2][h][k]) +
          scores[3][h][k];
      scores[0][h][k] =
          slots[k] >= 0 ? sum * (0.0625f * 1.4426950408889634f) : -1e20f;
    }
    __syncthreads();
    float maximum = fmaxf(scores[0][warp][lane], scores[0][warp][lane + 32]);
#pragma unroll
    for (int s = 16; s; s >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
    maximum = fmaxf(running_max, maximum);
    const float p0 =
        slots[lane] >= 0 ? fast_exp2(scores[0][warp][lane] - maximum) : 0.f;
    const float p1 = slots[lane + 32] >= 0
                         ? fast_exp2(scores[0][warp][lane + 32] - maximum)
                         : 0.f;
    scores[0][warp][lane] = p0;
    scores[0][warp][lane + 32] = p1;
    float sum = p0 + p1;
#pragma unroll
    for (int s = 16; s; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
    const float scale = fast_exp2(running_max - maximum);
    running_sum = __fmaf_rn(running_sum, scale, sum);
    running_max = maximum;
    if (lane == 0) alpha[warp] = scale;
    __syncthreads();
    // The shared key tile is retired. Reuse its storage for values, keeping
    // FP32 probabilities and PV FMAs instead of rounding probabilities to
    // half.
    if constexpr (HOT && FP8) {
      for (int v = t; v < C * 32; v += 256) {
        const int token = v / 32, dim = (v % 32) * 8;
        *reinterpret_cast<uint4*>(&kv[token][dim]) = load_kv<FP8, HOT, HOT_FP8>(
            a, slots[token], hots[token], 1, dim, lut);
      }
    } else {
#pragma unroll
      for (int i = 0; i < kChunks; ++i) {
        const int v = t + i * 256, token = v / 32, dim = (v % 32) * 8;
        uint4 value = make_uint4(0, 0, 0, 0);
        if (slots[token] >= 0) {
          if constexpr (FP8)
            value = decode8_raw(raw[i], raw_scale[i], lut);
          else
            value = raw[i];
        }
        *reinterpret_cast<uint4*>(&kv[token][dim]) = value;
      }
    }
    __syncthreads();
    // Thread t owns dimension t of every head: one value load serves all
    // six heads. Each output keeps its sequential FP32 FMA chain over k.
#pragma unroll
    for (int h = 0; h < H; ++h) result[h] *= alpha[h];
#pragma unroll 4
    for (int k = 0; k < C; ++k) {
      const float value = __half2float(kv[k][t]);
#pragma unroll
      for (int h = 0; h < H; ++h)
        result[h] = __fmaf_rn(scores[0][h][k], value, result[h]);
    }
  }
  if (lane == 0 && warp < H) {
    a.state[((row * H + warp) * a.splits + split) * 2] = running_max;
    a.state[((row * H + warp) * a.splits + split) * 2 + 1] = running_sum;
  }
#pragma unroll
  for (int h = 0; h < H; ++h)
    a.partial[((row * H + h) * a.splits + split) * D + t] = result[h];
}

// One (row, head): combine the split states and partial outputs. Shared
// scratch: weights/denom_parts[64], scalars[2].
__device__ __forceinline__ void merge_head(const Args& a, int row, int head,
                                           int t, float* weights,
                                           float* denom_parts, float* scalars) {
  const int64_t state_base = int64_t(row * H + head) * a.splits * 2;
  if (t < 32) {
    const int s0 = t, s1 = t + 32;
    const float m0 = s0 < a.splits && a.state[state_base + s0 * 2 + 1] > 0
                         ? a.state[state_base + s0 * 2]
                         : -1e20f;
    const float m1 = s1 < a.splits && a.state[state_base + s1 * 2 + 1] > 0
                         ? a.state[state_base + s1 * 2]
                         : -1e20f;
    float maximum = fmaxf(m0, m1);
#pragma unroll
    for (int s = 16; s; s >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
    if (!t) scalars[0] = maximum;
  }
  __syncthreads();
  if (t < 64) {
    const bool valid = t < a.splits && a.state[state_base + t * 2 + 1] > 0;
    const float weight =
        valid ? fast_exp2(a.state[state_base + t * 2] - scalars[0]) : 0.f;
    weights[t] = weight;
    denom_parts[t] = valid ? weight * a.state[state_base + t * 2 + 1] : 0.f;
  }
  __syncthreads();
  if (t < 32) {
    float sum = denom_parts[t] + denom_parts[t + 32];
#pragma unroll
    for (int s = 16; s; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
    if (!t) scalars[1] = sum;
  }
  __syncthreads();
  float numerator = 0.f;
  for (int s = 0; s < a.splits; ++s)
    numerator = __fmaf_rn(weights[s],
                          a.partial[((row * H + head) * a.splits + s) * D + t],
                          numerator);
  const float denominator = scalars[1];
  float result = denominator > 0 ? __fdiv_rn(numerator, denominator) : 0.f;
  if (a.wout) {
    a.wout[int64_t(row * H + head) * D + t] = result;
    if (!t) {
      a.wstate[int64_t(row * H + head) * 2] = scalars[0];
      a.wstate[int64_t(row * H + head) * 2 + 1] = denominator;
    }
    return;
  }
  if (a.gate) {
    result = __half2float(__float2half_rn(result));
    const float gate = __half2float(a.gate[row * a.gs0 + head * a.gs1 + t]);
    result *= __fdiv_rn(1.f, 1.f + fast_exp2(-gate * 1.4426950408889634f));
  }
  a.out[row * a.os0 + head * a.os1 + t] = __float2half_rn(result);
}

// Prefill over staged E4M3 history (one 64-key tile per CTA, no hot cache):
// the K and V tiles stay E4M3 in shared memory and are decoded where they are
// used, with the same arithmetic as `partial` (half(e4m3 * scale), FP32
// QK/PV). Half the tile footprint fits three CTAs per SM.
constexpr int KP = 272;  // E4M3 tile row stride (bytes)
__global__ __launch_bounds__(256, 3) void partial_fp8_lite(Args a) {
  const int row = blockIdx.y, split = blockIdx.x;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  __shared__ __align__(16) half query[8][PAD];
  __shared__ __align__(16) unsigned char tile[C][KP];
  __shared__ float scores[4][8][C];
  __shared__ int64_t slots[C];
  __shared__ float kscale[C], vscale[C];
  __shared__ float lut[256];
  lut[t] = e4m3(t);
  const int req = a.requests[row];
  const int64_t position = a.positions[row];
  const int length = req >= 0 && req < a.num_requests ? a.lengths[req] : 0;
  if (t < C) {
    const int col = split * C + t;
    const int logical = col < a.width ? a.indices[row * a.is0 + col] : -1;
    int64_t slot = -1;
    if (req >= 0 && req < a.num_requests && logical >= 0 &&
        logical <= position && logical < length &&
        logical / a.page < a.table_width) {
      const int physical = a.table[req * a.ts0 + logical / a.page];
      if (physical >= 0 && physical < a.blocks)
        slot = int64_t(physical) * a.page + logical % a.page;
    }
    slots[t] = slot;
    kscale[t] = slot >= 0 ? __ldg(a.scales + slot * 2) : 0.f;
    vscale[t] = slot >= 0 ? __ldg(a.scales + slot * 2 + 1) : 0.f;
  }
  const int head = t / 32, d8 = (t % 32) * 8;
  const uint4 qv = head < H ? __ldg(reinterpret_cast<const uint4*>(
                                  a.q + row * a.qs0 + head * a.qs1 + d8))
                            : make_uint4(0, 0, 0, 0);
  *reinterpret_cast<uint4*>(&query[head][d8]) = qv;
  __syncthreads();
  constexpr int kChunks = C * 32 / 256;
  uint2 vraw[kChunks];
#pragma unroll
  for (int i = 0; i < kChunks; ++i) {
    const int v = t + i * 256, token = v / 32, dim = (v % 32) * 8;
    const int64_t slot = slots[token];
    uint2 kraw = make_uint2(0, 0);
    vraw[i] = make_uint2(0, 0);
    if (slot >= 0) {
      const int block = slot / a.page, off = slot % a.page;
      const int64_t base = (int64_t(block) * 2 * a.page + off) * D + dim;
      kraw = __ldg(reinterpret_cast<const uint2*>(a.history + base));
      vraw[i] = __ldg(reinterpret_cast<const uint2*>(a.history + base +
                                                     int64_t(a.page) * D));
    }
    *reinterpret_cast<uint2*>(&tile[token][dim]) = kraw;
  }
  __syncthreads();
  const int nt = warp & 1, kw = warp >> 1;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int nr =
      nt * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const float ks = kscale[nr];
  float acc[8] = {};
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int k = kw * 64 + s * 16;
    const uint4 q0 = *reinterpret_cast<const uint4*>(&query[r8][k]);
    const uint4 q1 = *reinterpret_cast<const uint4*>(&query[r8][k + 8]);
    const uint4 k0 =
        decode8_raw(*reinterpret_cast<const uint2*>(&tile[nr][k]), ks, lut);
    const uint4 k1 =
        decode8_raw(*reinterpret_cast<const uint2*>(&tile[nr][k + 8]), ks, lut);
    mma(acc, q0.x, q0.y, k0.x, k0.y);
    mma(acc, q0.z, q0.w, k0.z, k0.w);
    mma(acc, q1.x, q1.y, k1.x, k1.y);
    mma(acc, q1.z, q1.w, k1.z, k1.w);
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = nt * 32 + ((lane >> 2) & 3) * 8 +
                    ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
    scores[kw][h][col] = acc[e];
  }
  __syncthreads();
  for (int v = t; v < 8 * C; v += 256) {
    const int h = v / C, k = v % C;
    const float sum = ((scores[0][h][k] + scores[1][h][k]) + scores[2][h][k]) +
                      scores[3][h][k];
    scores[0][h][k] =
        slots[k] >= 0 ? sum * (0.0625f * 1.4426950408889634f) : -1e20f;
  }
  __syncthreads();
  float maximum = fmaxf(scores[0][warp][lane], scores[0][warp][lane + 32]);
#pragma unroll
  for (int s = 16; s; s >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
  const float p0 =
      slots[lane] >= 0 ? fast_exp2(scores[0][warp][lane] - maximum) : 0.f;
  const float p1 = slots[lane + 32] >= 0
                       ? fast_exp2(scores[0][warp][lane + 32] - maximum)
                       : 0.f;
  scores[0][warp][lane] = p0;
  scores[0][warp][lane + 32] = p1;
  float sum = p0 + p1;
#pragma unroll
  for (int s = 16; s; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
  if (lane == 0 && warp < H) {
    a.state[((row * H + warp) * a.splits + split) * 2] = maximum;
    a.state[((row * H + warp) * a.splits + split) * 2 + 1] = sum;
  }
  __syncthreads();  // the K tile is retired
#pragma unroll
  for (int i = 0; i < kChunks; ++i) {
    const int v = t + i * 256, token = v / 32, dim = (v % 32) * 8;
    *reinterpret_cast<uint2*>(&tile[token][dim]) = vraw[i];
  }
  __syncthreads();
  float result[H] = {};
#pragma unroll 4
  for (int k = 0; k < C; ++k) {
    const float value =
        __half2float(__float2half_rn(__fmul_rn(lut[tile[k][t]], vscale[k])));
#pragma unroll
    for (int h = 0; h < H; ++h)
      result[h] = __fmaf_rn(scores[0][h][k], value, result[h]);
  }
#pragma unroll
  for (int h = 0; h < H; ++h)
    a.partial[((row * H + h) * a.splits + split) * D + t] = result[h];
}

// Decode/verify over encoded E4M3 hot cache + mapped host history: keep K/V
// tiles as raw E4M3 bytes in shared memory (half the footprint -> up to 3 CTAs
// per SM instead of 1), preload per-token K/V scales once in `t < C`, prefetch
// V bytes in registers before QK MMA + softmax, and decode with conflict-free
// bit arithmetic matching `partial<true, true, true>` bit-for-bit.
__global__ __launch_bounds__(256, 3) void partial_hot_fp8_lite(Args a) {
  const int row = blockIdx.y, split = blockIdx.x;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  __shared__ __align__(16) half query[8][PAD];
  __shared__ __align__(16) unsigned char tile[C][KP];
  __shared__ __align__(16) float scores[4][8][C];
  __shared__ int64_t slots[C];
  __shared__ int hots[C];
  __shared__ __align__(16) float kscale[C], vscale[C];
  __shared__ float alpha[8];
  const int req = a.requests[row];
  const int64_t position = a.positions[row];
  const int length = req >= 0 && req < a.num_requests ? a.lengths[req] : 0;
  const int head = t / 32, d8 = (t % 32) * 8;
  const uint4 qv = head < H ? __ldg(reinterpret_cast<const uint4*>(
                                  a.q + row * a.qs0 + head * a.qs1 + d8))
                            : make_uint4(0, 0, 0, 0);
  *reinterpret_cast<uint4*>(&query[head][d8]) = qv;
  const int nt = warp & 1, kw = warp >> 1;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int nr =
      nt * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  constexpr int kChunks = C * 16 / 256;
  const bool kv_aligned16 = ((reinterpret_cast<uintptr_t>(a.hot) |
                              reinterpret_cast<uintptr_t>(a.history)) &
                             15u) == 0u;
  float running_max = __int_as_float(0xff800000), running_sum = 0.f;
  float result[H] = {};
  for (int tl = 0; tl < a.tiles; ++tl) {
    const int base = (split * a.tiles + tl) * C;
    if (base >= a.width) break;
    __syncthreads();
    bool my_hot_or_inv = true;
    if (t < C) {
      const int col = base + t;
      const int logical = col < a.width ? a.indices[row * a.is0 + col] : -1;
      int64_t slot = -1;
      if (req >= 0 && req < a.num_requests && logical >= 0 &&
          logical <= position && logical < length &&
          logical / a.page < a.table_width) {
        const int physical = a.table[req * a.ts0 + logical / a.page];
        if (physical >= 0 && physical < a.blocks)
          slot = int64_t(physical) * a.page + logical % a.page;
      }
      int hot = slot >= 0 ? a.resolved[row * a.rs0 + col] : -1;
      if (hot == -1 && slot >= 0 && a.page_slots != nullptr &&
          a.tags != nullptr) {
        const int64_t p4 = slot >> 2;
        if (p4 < a.page_slots_len) {
          const int32_t reg = a.page_slots[p4];
          if (reg >= 0 && reg < a.hot_pages &&
              a.tags[reg] == static_cast<int32_t>(p4)) {
            hot = (reg << 2) | static_cast<int>(slot & 3);
          }
        }
      }
      slots[t] = slot;
      hots[t] = hot;
      my_hot_or_inv = (slot < 0) || (hot >= 0);
      if (slot < 0) {
        kscale[t] = 0.f;
        vscale[t] = 0.f;
      } else if (hot >= 0) {
        const float2 hs = __ldg(
            reinterpret_cast<const float2*>(a.hot_scales + int64_t(hot) * 2));
        kscale[t] = hs.x;
        vscale[t] = hs.y;
      } else {
        const float2 ms =
            __ldg(reinterpret_cast<const float2*>(a.scales + slot * 2));
        kscale[t] = ms.x;
        vscale[t] = ms.y;
      }
    }
    const bool all_hot = __syncthreads_and(my_hot_or_inv);
    uint4 vraw[kChunks];
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
      const int v = t + i * 256, token = v >> 4, dim = (v & 15) << 4;
      const int64_t slot = slots[token];
      uint4 kraw = make_uint4(0, 0, 0, 0);
      vraw[i] = make_uint4(0, 0, 0, 0);
      if (slot >= 0) {
        const int hot = hots[token];
        const unsigned char* src_k;
        const unsigned char* src_v;
        if (all_hot || hot >= 0) {
          src_k = reinterpret_cast<const unsigned char*>(a.hot) +
                  int64_t(hot) * 2 * D + dim;
          src_v = src_k + D;
        } else {
          const int block = slot / a.page, off = slot % a.page;
          src_k = a.history + (int64_t(block) * 2 * a.page + off) * D + dim;
          src_v = src_k + int64_t(a.page) * D;
        }
        if (kv_aligned16) {
          kraw = __ldg(reinterpret_cast<const uint4*>(src_k));
          vraw[i] = __ldg(reinterpret_cast<const uint4*>(src_v));
        } else {
          const uint2 kl = __ldg(reinterpret_cast<const uint2*>(src_k));
          const uint2 kh = __ldg(reinterpret_cast<const uint2*>(src_k + 8));
          const uint2 vl = __ldg(reinterpret_cast<const uint2*>(src_v));
          const uint2 vh = __ldg(reinterpret_cast<const uint2*>(src_v + 8));
          kraw = make_uint4(kl.x, kl.y, kh.x, kh.y);
          vraw[i] = make_uint4(vl.x, vl.y, vh.x, vh.y);
        }
      }
      *reinterpret_cast<uint2*>(&tile[token][dim]) = make_uint2(kraw.x, kraw.y);
      *reinterpret_cast<uint2*>(&tile[token][dim + 8]) =
          make_uint2(kraw.z, kraw.w);
    }
    __syncthreads();
    float acc[8] = {};
    if (all_hot) {
      const float ks256 = kscale[nr] * 256.f;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int k = kw * 64 + s * 16;
        const uint4 q0 = *reinterpret_cast<const uint4*>(&query[r8][k]);
        const uint4 q1 = *reinterpret_cast<const uint4*>(&query[r8][k + 8]);
        const uint4 k0 = decode8_hot_fast(
            *reinterpret_cast<const uint2*>(&tile[nr][k]), ks256);
        const uint4 k1 = decode8_hot_fast(
            *reinterpret_cast<const uint2*>(&tile[nr][k + 8]), ks256);
        mma(acc, q0.x, q0.y, k0.x, k0.y);
        mma(acc, q0.z, q0.w, k0.z, k0.w);
        mma(acc, q1.x, q1.y, k1.x, k1.y);
        mma(acc, q1.z, q1.w, k1.z, k1.w);
      }
    } else {
      const float ks = kscale[nr];
      const bool hot_nr = hots[nr] >= 0;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int k = kw * 64 + s * 16;
        const uint4 q0 = *reinterpret_cast<const uint4*>(&query[r8][k]);
        const uint4 q1 = *reinterpret_cast<const uint4*>(&query[r8][k + 8]);
        const uint4 k0 = decode8_exact(
            *reinterpret_cast<const uint2*>(&tile[nr][k]), ks, hot_nr);
        const uint4 k1 = decode8_exact(
            *reinterpret_cast<const uint2*>(&tile[nr][k + 8]), ks, hot_nr);
        mma(acc, q0.x, q0.y, k0.x, k0.y);
        mma(acc, q0.z, q0.w, k0.z, k0.w);
        mma(acc, q1.x, q1.y, k1.x, k1.y);
        mma(acc, q1.z, q1.w, k1.z, k1.w);
      }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = nt * 32 + ((lane >> 2) & 3) * 8 +
                      ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
      scores[kw][h][col] = acc[e];
    }
    __syncthreads();
    for (int v = t; v < 8 * C; v += 256) {
      const int h = v / C, k = v % C;
      const float sum =
          ((scores[0][h][k] + scores[1][h][k]) + scores[2][h][k]) +
          scores[3][h][k];
      scores[0][h][k] =
          slots[k] >= 0 ? sum * (0.0625f * 1.4426950408889634f) : -1e20f;
    }
    __syncthreads();
    float maximum = fmaxf(scores[0][warp][lane], scores[0][warp][lane + 32]);
#pragma unroll
    for (int s = 16; s; s >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
    maximum = fmaxf(running_max, maximum);
    const float p0 =
        slots[lane] >= 0 ? fast_exp2(scores[0][warp][lane] - maximum) : 0.f;
    const float p1 = slots[lane + 32] >= 0
                         ? fast_exp2(scores[0][warp][lane + 32] - maximum)
                         : 0.f;
    scores[0][warp][lane] = p0;
    scores[0][warp][lane + 32] = p1;
    float sum = p0 + p1;
#pragma unroll
    for (int s = 16; s; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
    const float scale = fast_exp2(running_max - maximum);
    running_sum = __fmaf_rn(running_sum, scale, sum);
    running_max = maximum;
    if (lane == 0) alpha[warp] = scale;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
      const int v = t + i * 256, token = v >> 4, dim = (v & 15) << 4;
      *reinterpret_cast<uint2*>(&tile[token][dim]) =
          make_uint2(vraw[i].x, vraw[i].y);
      *reinterpret_cast<uint2*>(&tile[token][dim + 8]) =
          make_uint2(vraw[i].z, vraw[i].w);
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < H; ++h) result[h] *= alpha[h];
    if (all_hot) {
#pragma unroll 4
      for (int k = 0; k < C; k += 2) {
        const float2 vs = *reinterpret_cast<const float2*>(&vscale[k]);
        const uint32_t c01 =
            unsigned(tile[k][t]) | (unsigned(tile[k + 1][t]) << 8);
        const uint32_t h01 = e4m3x2_half(c01, 0x4140u);
        const float2 d01 =
            __half22float2(*reinterpret_cast<const half2*>(&h01));
        const float v0 =
            __half2float(__float2half_rn(__fmul_rn(d01.x, vs.x * 256.f)));
        const float v1 =
            __half2float(__float2half_rn(__fmul_rn(d01.y, vs.y * 256.f)));
#pragma unroll
        for (int h = 0; h < H; ++h) {
          const float2 sc = *reinterpret_cast<const float2*>(&scores[0][h][k]);
          result[h] = __fmaf_rn(sc.x, v0, result[h]);
          result[h] = __fmaf_rn(sc.y, v1, result[h]);
        }
      }
    } else {
#pragma unroll 4
      for (int k = 0; k < C; ++k) {
        const float dec = e4m3_exact(tile[k][t], hots[k] >= 0);
        const float value =
            __half2float(__float2half_rn(__fmul_rn(dec, vscale[k])));
#pragma unroll
        for (int h = 0; h < H; ++h)
          result[h] = __fmaf_rn(scores[0][h][k], value, result[h]);
      }
    }
  }
  if (lane == 0 && warp < H) {
    a.state[((row * H + warp) * a.splits + split) * 2] = running_max;
    a.state[((row * H + warp) * a.splits + split) * 2 + 1] = running_sum;
  }
#pragma unroll
  for (int h = 0; h < H; ++h)
    a.partial[((row * H + h) * a.splits + split) * D + t] = result[h];
}

// Prefill over staged E4M3 history, one CTA per query row: every 64-key tile
// of the row with an online softmax and the gated output written directly
// (no partial/state workspace, no merge pass). Software-pipelined because a
// few CTAs per SM cannot hide latency for each other: tile j + 2 is located
// in three stages spread over tile j (index, table, scales), tile j + 1's K
// is loaded once tile j's K is in shared memory and its V once tile j's V
// is, and tiles without a valid key are skipped. K and V tiles decode to
// exact unscaled FP16 (a few bit operations, no table, no rounding); the
// per-key K scale multiplies the FP32 score and the per-key V scale the PV
// weight. QK on tensor cores with FP32 accumulation and softmax; PV in FP32
// FMAs, or with MMA_PV on tensor cores (FP16 weights, FP32 accumulation).
template <int MIN_BLOCKS, bool MMA_PV>
__global__ __launch_bounds__(256, MIN_BLOCKS) void rows_fp8(Args a) {
  const int row = blockIdx.x;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  __shared__ __align__(16) half query[8][PAD];
  // Padded rows: the QK K reads (8-row groups), the PV V reads (4 rows) and
  // the PV probability reads (8 head rows) would otherwise share banks.
  constexpr int RKP = 296, SP = C + 4;
  __shared__ __align__(16) unsigned char tile[C][RKP];
  __shared__ __align__(16) float scores[4][8][SP];
  __shared__ int64_t slots[3][C];
  __shared__ float kscale[3][C], vscale[3][C];
  __shared__ float alpha[8], denom[8];
  const int req = a.requests[row];
  const int64_t position = a.positions[row];
  const bool live = req >= 0 && req < a.num_requests;
  const int length = live ? a.lengths[req] : 0;
  const int ntiles = (a.width + C - 1) / C;
  // Locate stages (threads t < C, key t of the located tile).
  int logical = -1;
  int64_t found = -1;
  float found_k = 0.f, found_v = 0.f;
  auto stage_index = [&](int j) {
    const int col = j * C + t;
    logical = t < C && j < ntiles && col < a.width
                  ? a.indices[row * a.is0 + col]
                  : -1;
  };
  auto stage_table = [&]() {
    found = -1;
    if (live && logical >= 0 && logical <= position && logical < length &&
        logical / a.page < a.table_width) {
      const int physical = a.table[req * a.ts0 + logical / a.page];
      if (physical >= 0 && physical < a.blocks)
        found = int64_t(physical) * a.page + logical % a.page;
    }
  };
  // Scales times 2^8: the tiles decode to value * 2^-8.
  auto stage_scales = [&]() {
    found_k = found >= 0 ? __ldg(a.scales + found * 2) * 256.f : 0.f;
    found_v = found >= 0 ? __ldg(a.scales + found * 2 + 1) * 256.f : 0.f;
  };
  auto stage_store = [&](int b) {
    if (t < C) {
      slots[b][t] = found;
      kscale[b][t] = found_k;
      vscale[b][t] = found_v;
    }
    return t < C && found >= 0;
  };
  constexpr int kChunks = C * 32 / 256;
  uint2 kraw[kChunks], vraw[kChunks];
  auto load = [&](uint2(&raw)[kChunks], int b, int kv) {
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
      const int v = t + i * 256, token = v / 32, dim = (v % 32) * 8;
      const int64_t slot = slots[b][token];
      raw[i] = make_uint2(0, 0);
      if (slot >= 0) {
        const int block = slot / a.page, off = slot % a.page;
        raw[i] = __ldg(reinterpret_cast<const uint2*>(
            a.history + (int64_t(block) * 2 * a.page + off) * D + dim +
            (kv ? int64_t(a.page) * D : 0)));
      }
    }
  };
  auto store = [&](const uint2(&raw)[kChunks]) {
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
      const int v = t + i * 256, token = v / 32, dim = (v % 32) * 8;
      *reinterpret_cast<uint2*>(&tile[token][dim]) = raw[i];
    }
  };
  {
    const int head = t / 32, d8 = (t % 32) * 8;
    const uint4 qv = head < H ? __ldg(reinterpret_cast<const uint4*>(
                                    a.q + row * a.qs0 + head * a.qs1 + d8))
                              : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(&query[head][d8]) = qv;
  }
  // Prologue: tiles 0 and 1 located, tile 0's K/V in flight.
  stage_index(0);
  stage_table();
  stage_scales();
  int count0 = __syncthreads_count(stage_store(0));
  stage_index(1);
  stage_table();
  stage_scales();
  int count1 = __syncthreads_count(stage_store(1));
  if (count0) {
    load(kraw, 0, 0);
    load(vraw, 0, 1);
  }
  const int nt = warp & 1, kw = warp >> 1;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int nr =
      nt * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  float running_max = -1e20f, running_sum = 0.f;  // head `warp`
  float result[H] = {};
  float pv[8] = {};  // MMA_PV: element e is head pv_head(e), dim pv_dim(e)
  const int c8 = r8;
  const int pv_base = warp * 32 + ((lane >> 2) & 3) * 8;
  auto pv_head = [&](int e) {
    return (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
  };
  auto pv_dim = [&](int e) {
    return pv_base + ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
  };
  for (int j = 0; j < ntiles; ++j) {
    const int b = j % 3, b1 = (j + 1) % 3, b2 = (j + 2) % 3;
    const bool current = count0 > 0;
    stage_index(j + 2);
    if (current) store(kraw);
    // Tile j + 1's K overlaps the rest of this tile.
    if (count1) load(kraw, b1, 0);
    __syncthreads();  // K tile published
    if (current) {
      float acc[8] = {};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int k = kw * 64 + s * 16;
        const uint4 q0 = *reinterpret_cast<const uint4*>(&query[r8][k]);
        const uint4 q1 = *reinterpret_cast<const uint4*>(&query[r8][k + 8]);
        const uint4 k0 =
            decode8_unscaled(*reinterpret_cast<const uint2*>(&tile[nr][k]));
        const uint4 k1 =
            decode8_unscaled(*reinterpret_cast<const uint2*>(&tile[nr][k + 8]));
        mma(acc, q0.x, q0.y, k0.x, k0.y);
        mma(acc, q0.z, q0.w, k0.z, k0.w);
        mma(acc, q1.x, q1.y, k1.x, k1.y);
        mma(acc, q1.z, q1.w, k1.z, k1.w);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int col = nt * 32 + ((lane >> 2) & 3) * 8 +
                        ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
        scores[kw][h][col] = acc[e];
      }
    }
    stage_table();
    __syncthreads();  // QK partials published, K tile retired
    if (current) {
      store(vraw);
      // Warp `warp` owns head row `warp`: reduce the four K-quarter
      // partials of its keys, then the online-softmax update.
      float s0 = ((scores[0][warp][lane] + scores[1][warp][lane]) +
                  scores[2][warp][lane]) +
                 scores[3][warp][lane];
      float s1 = ((scores[0][warp][lane + 32] + scores[1][warp][lane + 32]) +
                  scores[2][warp][lane + 32]) +
                 scores[3][warp][lane + 32];
      const bool v0 = slots[b][lane] >= 0, v1 = slots[b][lane + 32] >= 0;
      // Per-key K scale after the dot product (unscaled K operands).
      s0 = v0 ? s0 * kscale[b][lane] * (0.0625f * 1.4426950408889634f) : -1e20f;
      s1 = v1 ? s1 * kscale[b][lane + 32] * (0.0625f * 1.4426950408889634f)
              : -1e20f;
      float maximum = fmaxf(s0, s1);
#pragma unroll
      for (int s = 16; s; s >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
      maximum = fmaxf(maximum, running_max);
      const float p0 = v0 ? fast_exp2(s0 - maximum) : 0.f;
      const float p1 = v1 ? fast_exp2(s1 - maximum) : 0.f;
      // PV weights carry the per-key V scale (unscaled V operands).
      scores[0][warp][lane] = p0 * vscale[b][lane];
      scores[0][warp][lane + 32] = p1 * vscale[b][lane + 32];
      float sum = p0 + p1;
#pragma unroll
      for (int s = 16; s; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
      const float scale = fast_exp2(running_max - maximum);
      running_sum = running_sum * scale + sum;
      running_max = maximum;
      if (lane == 0) alpha[warp] = scale;
    }
    // Tile j + 1's V overlaps this tile's PV and the next QK.
    if (count1) load(vraw, b1, 1);
    stage_scales();
    __syncthreads();  // V tile, probabilities and alpha published
    if (current) {
      if constexpr (MMA_PV) {
#pragma unroll
        for (int e = 0; e < 8; ++e) pv[e] *= alpha[pv_head(e)];
        // Row-major V operand: 4 consecutive dims of key k + (lane & 3).
        const int vk = lane & 3, vn = pv_base + ((lane & 16) ? 4 : 0);
#pragma unroll 4
        for (int k = 0; k < C; k += 4) {
          const float4 p = *reinterpret_cast<const float4*>(&scores[0][c8][k]);
          const half2 p01 = __floats2half2_rn(p.x, p.y);
          const half2 p23 = __floats2half2_rn(p.z, p.w);
          const uint32_t codes =
              *reinterpret_cast<const uint32_t*>(&tile[k + vk][vn]);
          mma_rr(pv, *reinterpret_cast<const uint32_t*>(&p01),
                 *reinterpret_cast<const uint32_t*>(&p23),
                 e4m3x2_half(codes, 0x4140u), e4m3x2_half(codes, 0x4342u));
        }
      } else {
#pragma unroll
        for (int h = 0; h < H; ++h) result[h] *= alpha[h];
#pragma unroll 4
        for (int k = 0; k < C; ++k) {
          const float value = e4m3_bits(tile[k][t]) * (1.f / 256.f);
#pragma unroll
          for (int h = 0; h < H; ++h)
            result[h] = __fmaf_rn(scores[0][h][k], value, result[h]);
        }
      }
    }
    // Publishes tile j + 2's location; tile/scores free for the next tile.
    const int count2 = __syncthreads_count(stage_store(b2));
    count0 = count1;
    count1 = count2;
  }
  if (lane == 0) denom[warp] = running_sum;
  if (a.window && lane == 0 && warp < H) {
    a.state[(int64_t(row) * H + warp) * 2] = running_max;
    a.state[(int64_t(row) * H + warp) * 2 + 1] = running_sum;
  }
  __syncthreads();
  auto emit = [&](int h, int dim, float sum) {
    float value = denom[h] > 0 ? __fdiv_rn(sum, denom[h]) : 0.f;
    if (a.window) {
      a.partial[(int64_t(row) * H + h) * D + dim] = value;
      return;
    }
    if (a.gate) {
      value = __half2float(__float2half_rn(value));
      const float gate = __half2float(a.gate[row * a.gs0 + h * a.gs1 + dim]);
      value *= __fdiv_rn(1.f, 1.f + fast_exp2(-gate * 1.4426950408889634f));
    }
    a.out[row * a.os0 + h * a.os1 + dim] = __float2half_rn(value);
  };
  if constexpr (MMA_PV) {
#pragma unroll
    for (int e = 0; e < 8; ++e)
      if (pv_head(e) < H) emit(pv_head(e), pv_dim(e), pv[e]);
  } else {
#pragma unroll
    for (int h = 0; h < H; ++h) emit(h, t, result[h]);
  }
}

__global__ void merge(Args a) {
  __shared__ float weights[64], denom_parts[64], scalars[2];
  merge_head(a, blockIdx.x / H, blockIdx.x % H, threadIdx.x, weights,
             denom_parts, scalars);
}
}  // namespace device_qsa

void device_qsa_out(
    torch::Tensor q, torch::Tensor history, torch::Tensor scales,
    torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
    torch::Tensor positions, torch::Tensor lengths, torch::Tensor out,
    std::optional<torch::Tensor> gate, torch::Tensor partial,
    torch::Tensor state, std::optional<torch::Tensor> hot = std::nullopt,
    std::optional<torch::Tensor> resolved = std::nullopt, int64_t tiles = 1,
    bool prefill = false, int64_t whole_rows = 0,
    std::optional<torch::Tensor> hot_scales = std::nullopt, bool window = false,
    std::optional<torch::Tensor> window_out = std::nullopt,
    std::optional<torch::Tensor> window_state = std::nullopt,
    std::optional<torch::Tensor> page_slots = std::nullopt,
    std::optional<torch::Tensor> tags = std::nullopt) {
  const c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 3 &&
              q.size(0) > 0 && (whole_rows || q.size(0) <= 64) &&
              q.size(1) == 6 && q.size(2) == 256 && q.stride(2) == 1);
  TORCH_CHECK((history.scalar_type() == at::kByte ||
               history.scalar_type() == at::kHalf) &&
              history.dim() == 5 && history.size(1) == 2 &&
              history.size(3) == 1 && history.size(4) == 256 &&
              history.is_contiguous());
  TORCH_CHECK(scales.scalar_type() == at::kFloat && scales.is_contiguous() &&
              scales.numel() == (history.scalar_type() == at::kByte
                                     ? history.size(0) * history.size(2) * 2
                                     : 2));
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.dim() == 2 &&
              indices.size(0) == q.size(0) && indices.stride(1) == 1 &&
              indices.size(1) > 0 && indices.size(1) <= 4096);
  TORCH_CHECK(table.scalar_type() == at::kInt && table.dim() == 2 &&
              table.stride(1) == 1 && requests.scalar_type() == at::kInt &&
              positions.scalar_type() == at::kLong &&
              lengths.scalar_type() == at::kInt);
  TORCH_CHECK(requests.numel() == q.size(0) && positions.numel() == q.size(0) &&
              lengths.numel() == table.size(0) && requests.is_contiguous() &&
              positions.is_contiguous() && lengths.is_contiguous());
  TORCH_CHECK(
      out.sizes() == q.sizes() && out.scalar_type() == at::kHalf &&
      out.stride(2) == 1 &&
      (!gate || (gate->sizes() == q.sizes() &&
                 gate->scalar_type() == at::kHalf && gate->stride(2) == 1)));
  for (const auto& tensor : {history, scales, indices, table, requests,
                             positions, lengths, out, partial, state})
    TORCH_CHECK(tensor.device() == q.device());
  if (gate) TORCH_CHECK(gate->device() == q.device());
  TORCH_CHECK(hot.has_value() == resolved.has_value());
  TORCH_CHECK(!hot_scales || (hot && history.scalar_type() == at::kByte),
              "Encoded hot storage requires E4M3 authoritative history");
  if (hot) {
    TORCH_CHECK(hot->is_cuda() && hot->device() == q.device() &&
                hot->scalar_type() == (hot_scales ? at::kByte : at::kHalf) &&
                hot->dim() == 3 && hot->size(1) == 2 && hot->size(2) == 256 &&
                hot->is_contiguous());
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(hot->data_ptr()) % (hot_scales ? 8 : 16) ==
            0,
        "QSA hot storage requires aligned vector loads");
    TORCH_CHECK(resolved->device() == q.device() &&
                resolved->scalar_type() == at::kInt && resolved->dim() == 2 &&
                resolved->size(0) >= q.size(0) &&
                resolved->size(1) >= indices.size(1) &&
                resolved->stride(1) == 1);
    if (hot_scales) {
      TORCH_CHECK(hot_scales->device() == q.device() &&
                  hot_scales->scalar_type() == at::kFloat &&
                  hot_scales->dim() == 2 &&
                  hot_scales->size(0) == hot->size(0) &&
                  hot_scales->size(1) == 2 && hot_scales->is_contiguous());
    }
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
              q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0);
  auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 0);
  device_qsa::Args a{};
  a.q = reinterpret_cast<const half*>(q.data_ptr());
  a.history = reinterpret_cast<const unsigned char*>(history.data_ptr());
  a.scales = scales.data_ptr<float>();
  a.indices = indices.data_ptr<int>();
  a.table = table.data_ptr<int>();
  a.requests = requests.data_ptr<int>();
  a.positions = positions.data_ptr<int64_t>();
  a.lengths = lengths.data_ptr<int>();
  a.gate = gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr;
  a.hot = hot ? reinterpret_cast<const half*>(hot->data_ptr()) : nullptr;
  a.hot_scales = hot_scales ? hot_scales->data_ptr<float>() : nullptr;
  a.resolved = resolved ? resolved->data_ptr<int>() : nullptr;
  a.rs0 = resolved ? resolved->stride(0) : 0;
  if (hot && page_slots && tags && page_slots->is_cuda() &&
      page_slots->device() == q.device() &&
      page_slots->scalar_type() == at::kInt && page_slots->is_contiguous() &&
      tags->is_cuda() && tags->device() == q.device() &&
      tags->scalar_type() == at::kInt && tags->is_contiguous() &&
      tags->numel() == hot->size(0) / 4) {
    a.page_slots = page_slots->data_ptr<int32_t>();
    a.tags = tags->data_ptr<int32_t>();
    a.hot_pages = static_cast<int>(tags->numel());
    a.page_slots_len = page_slots->numel();
  }
  a.out = reinterpret_cast<half*>(out.data_ptr());
  a.partial = partial.data_ptr<float>();
  a.state = state.data_ptr<float>();
  a.window = window ? 1 : 0;
  TORCH_CHECK(window_out.has_value() == window_state.has_value());
  if (window_out) {
    TORCH_CHECK(
        !whole_rows && window_out->scalar_type() == at::kFloat &&
            window_state->scalar_type() == at::kFloat &&
            window_out->is_contiguous() && window_state->is_contiguous() &&
            window_out->numel() == q.size(0) * 6 * 256 &&
            window_state->numel() == q.size(0) * 6 * 2,
        "QSA window merge needs FP32 [M, 6, 256] rows and [M, 6, 2] state");
  }
  a.wout = window_out ? window_out->data_ptr<float>() : nullptr;
  a.wstate = window_state ? window_state->data_ptr<float>() : nullptr;
  TORCH_CHECK(
      !window ||
          (whole_rows && partial.scalar_type() == at::kFloat &&
           state.scalar_type() == at::kFloat && partial.is_contiguous() &&
           state.is_contiguous() && partial.numel() >= q.size(0) * 6 * 256 &&
           state.numel() >= q.size(0) * 6 * 2),
      "QSA window rows need FP32 [M, 6, 256] partial and [M, 6, 2] state");
  a.M = q.size(0);
  a.blocks = history.size(0);
  a.page = history.size(2);
  a.width = indices.size(1);
  a.table_width = table.size(1);
  a.num_requests = table.size(0);
  // Keys per CTA: `tiles` 64-key tiles with an online softmax (1: the
  // single-tile arithmetic of the decode readers).
  a.tiles = static_cast<int>(tiles);
  TORCH_CHECK(a.tiles >= 1);
  a.splits = (a.width + 64 * a.tiles - 1) / (64 * a.tiles);
  TORCH_CHECK(whole_rows || (partial.scalar_type() == at::kFloat &&
                             state.scalar_type() == at::kFloat &&
                             partial.is_contiguous() && state.is_contiguous() &&
                             partial.numel() >= a.M * 6 * a.splits * 256 &&
                             state.numel() >= a.M * 6 * a.splits * 2));
  a.qs0 = q.stride(0);
  a.qs1 = q.stride(1);
  a.is0 = indices.stride(0);
  a.ts0 = table.stride(0);
  a.os0 = out.stride(0);
  a.os1 = out.stride(1);
  a.gs0 = gate ? gate->stride(0) : 0;
  a.gs1 = gate ? gate->stride(1) : 0;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(a.splits, a.M);
  const bool fp8 = history.scalar_type() == at::kByte;
  if (whole_rows) {
    // One CTA per row, no workspace.
    // whole_rows: CTAs per SM (2 or 3), +10 for tensor-core PV.
    TORCH_CHECK(fp8 && !hot);
    switch (whole_rows) {
      case 2:
        device_qsa::rows_fp8<2, false><<<a.M, 256, 0, stream>>>(a);
        break;
      case 3:
        device_qsa::rows_fp8<3, false><<<a.M, 256, 0, stream>>>(a);
        break;
      case 12:
        device_qsa::rows_fp8<2, true><<<a.M, 256, 0, stream>>>(a);
        break;
      case 13:
        device_qsa::rows_fp8<3, true><<<a.M, 256, 0, stream>>>(a);
        break;
      default:
        TORCH_CHECK(false, "invalid whole_rows ", whole_rows);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (prefill && fp8 && !hot && a.tiles == 1) {
    device_qsa::partial_fp8_lite<<<grid, 256, 0, stream>>>(a);
  } else if (fp8 && hot_scales) {
    static int configured_dev = -1;
    const int cur_dev = q.device().index();
    if (configured_dev != cur_dev) {
      cudaFuncSetAttribute(device_qsa::partial_hot_fp8_lite,
                           cudaFuncAttributePreferredSharedMemoryCarveout, 100);
      configured_dev = cur_dev;
    }
    device_qsa::partial_hot_fp8_lite<<<grid, 256, 0, stream>>>(a);
  } else if (fp8 && hot) {
    device_qsa::partial<true, true><<<grid, 256, 0, stream>>>(a);
  } else if (fp8) {
    device_qsa::partial<true, false><<<grid, 256, 0, stream>>>(a);
  } else if (hot) {
    device_qsa::partial<false, true><<<grid, 256, 0, stream>>>(a);
  } else {
    device_qsa::partial<false, false><<<grid, 256, 0, stream>>>(a);
  }
  device_qsa::merge<<<a.M * 6, 256, 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// Copy a contiguous run of authoritative host pages with the copy engine.
// History is a CUDA-addressable view of pinned storage; using its host alias
// avoids treating that view as an ordinary device-to-device tensor copy.
// Return false without writes for actual device history (the gather fallback).
bool stage_host_run(torch::Tensor history, torch::Tensor scales,
                    torch::Tensor out, torch::Tensor out_scales,
                    int64_t source_page, int64_t target_page, int64_t pages) {
  TORCH_CHECK(history.is_cuda() && out.is_cuda() &&
                  history.device() == out.device() && history.is_contiguous() &&
                  out.is_contiguous(),
              "QSA stage history must have contiguous same-device views");
  TORCH_CHECK(history.dim() == 5 && out.dim() == 5 && history.size(1) == 2 &&
                  history.size(2) > 0 && history.size(2) % 4 == 0 &&
                  history.size(3) == 1 && history.size(4) == 256 &&
                  history.sizes().slice(1) == out.sizes().slice(1) &&
                  history.scalar_type() == out.scalar_type() &&
                  (history.scalar_type() == at::kByte ||
                   history.scalar_type() == at::kHalf),
              "QSA stage history requires matching page-major E4M3/FP16 views");
  TORCH_CHECK(source_page >= 0 && source_page <= history.size(0) &&
                  target_page >= 0 && target_page <= out.size(0) &&
                  pages >= 0 && pages <= history.size(0) - source_page &&
                  pages <= out.size(0) - target_page,
              "QSA stage page range is outside its source or destination");
  const bool fp8 = history.scalar_type() == at::kByte;
  const int64_t scale_stride = 2 * history.size(2);
  if (fp8) {
    TORCH_CHECK(scales.is_cuda() && out_scales.is_cuda() &&
                    scales.device() == out.device() &&
                    out_scales.device() == out.device() &&
                    scales.scalar_type() == at::kFloat &&
                    out_scales.scalar_type() == at::kFloat &&
                    scales.is_contiguous() && out_scales.is_contiguous() &&
                    scales.numel() >= history.size(0) * scale_stride &&
                    out_scales.numel() >= out.size(0) * scale_stride,
                "QSA stage scales must cover the page-major FP8 views");
  }
  if (!pages) return true;
  const c10::cuda::CUDAGuard guard(out.device());
  cudaPointerAttributes source{}, destination{}, source_scales{}, dest_scales{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&source, history.data_ptr()));
  if (source.type != cudaMemoryTypeHost || source.hostPointer == nullptr)
    return false;
  C10_CUDA_CHECK(cudaPointerGetAttributes(&destination, out.data_ptr()));
  TORCH_CHECK(destination.type == cudaMemoryTypeDevice,
              "QSA stage destination must be device memory");
  if (fp8) {
    C10_CUDA_CHECK(cudaPointerGetAttributes(&source_scales, scales.data_ptr()));
    C10_CUDA_CHECK(
        cudaPointerGetAttributes(&dest_scales, out_scales.data_ptr()));
    TORCH_CHECK(source_scales.type == cudaMemoryTypeHost &&
                    source_scales.hostPointer != nullptr &&
                    dest_scales.type == cudaMemoryTypeDevice,
                "QSA stage requires host scales and device scale storage");
  }
  // Validate both copies before issuing either. Preserve the caller's stream
  // and existing buffer-free / completion events; no allocation or host wait.
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t page_bytes = history.stride(0) * history.element_size();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      static_cast<char*>(out.data_ptr()) + target_page * page_bytes,
      static_cast<const char*>(source.hostPointer) + source_page * page_bytes,
      pages * page_bytes, cudaMemcpyHostToDevice, stream));
  if (fp8) {
    C10_CUDA_CHECK(cudaMemcpyAsync(
        out_scales.data_ptr<float>() + target_page * scale_stride,
        static_cast<const float*>(source_scales.hostPointer) +
            source_page * scale_stride,
        pages * scale_stride * sizeof(float), cudaMemcpyHostToDevice, stream));
  }
  return true;
}

namespace host_prep {

__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(half v) { return __half2float(v); }

__device__ __forceinline__ uint8_t fp32_to_e4m3fn_rne(float x) {
  const uint32_t xbits = __float_as_uint(x);
  const uint32_t sign = (xbits >> 24) & 0x80u;
  const float value = fminf(fabsf(x), 448.0f);
  if (value == 0.0f) return static_cast<uint8_t>(sign);
  if (value < 0x1p-6f) {
    int sub;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(sub) : "f"(value * 512.0f));
    sub = min(max(sub, 0), 8);
    return static_cast<uint8_t>(sign | static_cast<uint32_t>(sub));
  }
  const uint32_t vbits = __float_as_uint(value);
  const uint32_t rounded = (vbits + 0x7ffffu + ((vbits >> 20) & 1u)) >> 20;
  const int normal = min(max(static_cast<int>(rounded) - 960, 8), 126);
  return static_cast<uint8_t>(sign | static_cast<uint32_t>(normal));
}

template <bool FP8, bool HOT_FP8>
__device__ __forceinline__ void write_host_hot_kv(
    const uint4& val_u4, int64_t slot, int kv, int lane, int blocks,
    int page_size, int hot_pages, int64_t page_slots_len, void* history_raw,
    float* scales, const int32_t* tags, const int32_t* page_slots,
    void* hot_raw, float* hot_scales) {
  if (slot < 0 || slot >= static_cast<int64_t>(blocks) * page_size) return;
  const int64_t block = slot / page_size, tok = slot % page_size;
  const int64_t hist_elem =
      ((block * 2 + kv) * page_size + tok) * 256 + lane * 8;
  const int64_t page = slot >> 2;
  const int32_t resident =
      (page >= 0 && page < page_slots_len) ? page_slots[page] : -1;
  const bool update_hot = (resident >= 0 && resident < hot_pages) &&
                          (tags[resident] == static_cast<int32_t>(page));
  const int64_t hot_elem =
      ((static_cast<int64_t>(resident) * 4 + (slot & 3)) * 2 + kv) * 256 +
      lane * 8;

  if constexpr (FP8) {
    const half* vh = reinterpret_cast<const half*>(&val_u4);
    float vf[8];
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      vf[j] = __half2float(vh[j]);
      m = fmaxf(m, fabsf(vf[j]));
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
    const float scale = fmaxf(__fdiv_rn(m, 448.0f), 0x1p-126f);
    if (lane == 0) scales[slot * 2 + kv] = scale;

    uint2 packed;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&packed);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      bytes[j] = fp32_to_e4m3fn_rne(__fdiv_rn(vf[j], scale));
    *reinterpret_cast<uint2*>(static_cast<uint8_t*>(history_raw) + hist_elem) =
        packed;
    if (update_hot) {
      if constexpr (HOT_FP8) {
        *reinterpret_cast<uint2*>(static_cast<uint8_t*>(hot_raw) + hot_elem) =
            packed;
        if (lane == 0) hot_scales[resident * 8 + (slot & 3) * 2 + kv] = scale;
      } else {
        *reinterpret_cast<uint4*>(static_cast<half*>(hot_raw) + hot_elem) =
            device_qsa::decode8_bits(packed, scale);
      }
    }
  } else {
    *reinterpret_cast<uint4*>(static_cast<half*>(history_raw) + hist_elem) =
        val_u4;
    if (update_hot) {
      *reinterpret_cast<uint4*>(static_cast<half*>(hot_raw) + hot_elem) =
          val_u4;
    }
  }
}

template <typename CS, typename PosT, typename SlotT, bool FP8, bool HOT_FP8>
__global__ void prep_write_kernel(
    const half* __restrict__ qkv, int64_t ld_qkv, const PosT* __restrict__ pos,
    const CS* __restrict__ cos_sin, int rot, const half* __restrict__ qw,
    const half* __restrict__ kw, float eps, half* __restrict__ query_out,
    half* __restrict__ key_out, const SlotT* __restrict__ slots,
    void* __restrict__ history_raw, float* __restrict__ scales,
    const int32_t* __restrict__ tags, const int32_t* __restrict__ page_slots,
    void* __restrict__ hot_raw, float* __restrict__ hot_scales,
    int32_t* __restrict__ epoch, int HQ, int blocks, int page_size,
    int hot_pages, int64_t page_slots_len) {
  const int t = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (warp > HQ + 1) return;

  if (warp == HQ + 1) {
    const int64_t sl = static_cast<int64_t>(slots[t]);
    const uint4 v = *reinterpret_cast<const uint4*>(
        qkv + static_cast<int64_t>(t) * ld_qkv + HQ * 512 + 256 + lane * 8);
    write_host_hot_kv<FP8, HOT_FP8>(
        v, sl, 1, lane, blocks, page_size, hot_pages, page_slots_len,
        history_raw, scales, tags, page_slots, hot_raw, hot_scales);
    return;
  }

  const bool is_k = (warp == HQ);
  const half* src = qkv + static_cast<int64_t>(t) * ld_qkv +
                    (is_k ? HQ * 512 : warp * 512) + lane * 8;
  const uint4 raw = *reinterpret_cast<const uint4*>(src);
  const half* xh = reinterpret_cast<const half*>(&raw);
  const half* w = (is_k ? kw : qw) + lane * 8;
  const uint4 wraw = *reinterpret_cast<const uint4*>(w);
  const half* wh = reinterpret_cast<const half*>(&wraw);
  float x[8], ss = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j] = __half2float(xh[j]);
    ss += x[j] * x[j];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(ss * (1.0f / 256.0f) + eps);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j] = __half2float(
        __float2half_rn(x[j] * inv * (1.0f + __half2float(wh[j]))));
  }
  const int half_rot = rot / 2, d0 = lane * 8;
  float px[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int off = half_rot / 8;
    const float up = __shfl_down_sync(0xffffffffu, x[j], off);
    const float dn = __shfl_up_sync(0xffffffffu, x[j], off);
    px[j] = (d0 < half_rot) ? up : dn;
  }
  if (d0 < rot) {
    const int64_t p = static_cast<int64_t>(pos[t]);
    const CS* cs = cos_sin + p * rot;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int d = d0 + j;
      const bool first = d < half_rot;
      const int i = first ? d : d - half_rot;
      const float c = to_f32(cs[i]), s = to_f32(cs[half_rot + i]);
      x[j] = first ? x[j] * c - px[j] * s : x[j] * c + px[j] * s;
    }
  }
  uint4 outv;
  half* oh = reinterpret_cast<half*>(&outv);
#pragma unroll
  for (int j = 0; j < 8; ++j) oh[j] = __float2half_rn(x[j]);
  if (!is_k) {
    *reinterpret_cast<uint4*>(
        query_out + (static_cast<int64_t>(t) * HQ + warp) * 256 + lane * 8) =
        outv;
  } else {
    *reinterpret_cast<uint4*>(key_out + static_cast<int64_t>(t) * 256 +
                              lane * 8) = outv;
    if (t == 0 && lane == 0) atomicAdd(epoch, 1);
    const int64_t sl = static_cast<int64_t>(slots[t]);
    write_host_hot_kv<FP8, HOT_FP8>(
        outv, sl, 0, lane, blocks, page_size, hot_pages, page_slots_len,
        history_raw, scales, tags, page_slots, hot_raw, hot_scales);
  }
}

}  // namespace host_prep

void prep_write_out(torch::Tensor qkv, torch::Tensor pos, torch::Tensor cos_sin,
                    torch::Tensor qw, torch::Tensor kw, double eps,
                    torch::Tensor query_out, torch::Tensor key_out,
                    torch::Tensor slots, torch::Tensor history,
                    torch::Tensor scales, torch::Tensor tags,
                    torch::Tensor page_slots, torch::Tensor hot,
                    std::optional<torch::Tensor> hot_scales,
                    torch::Tensor epoch) {
  const c10::cuda::CUDAGuard guard(qkv.device());
  const int64_t M = qkv.size(0);
  if (M == 0) return;
  const int HQ = static_cast<int>(query_out.size(1));
  const int rot = static_cast<int>(cos_sin.size(1));
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kHalf &&
              qkv.dim() == 2 && M <= 64 && qkv.stride(1) == 1 &&
              qkv.size(1) >= HQ * 512 + 512);
  TORCH_CHECK(HQ >= 1 && HQ <= 24 && query_out.is_cuda() &&
              query_out.scalar_type() == at::kHalf &&
              query_out.sizes() == at::IntArrayRef({M, HQ, 256}) &&
              query_out.is_contiguous() && key_out.is_cuda() &&
              key_out.scalar_type() == at::kHalf &&
              key_out.sizes() == at::IntArrayRef({M, 1, 256}) &&
              key_out.is_contiguous());
  TORCH_CHECK(rot > 0 && rot <= 256 && rot % 16 == 0 &&
              cos_sin.is_contiguous() &&
              (cos_sin.scalar_type() == at::kFloat ||
               cos_sin.scalar_type() == at::kHalf));
  TORCH_CHECK(qw.scalar_type() == at::kHalf && qw.numel() == 256 &&
              qw.is_contiguous() && kw.scalar_type() == at::kHalf &&
              kw.numel() == 256 && kw.is_contiguous());
  TORCH_CHECK(
      pos.is_contiguous() && pos.numel() >= M &&
      (pos.scalar_type() == at::kLong || pos.scalar_type() == at::kInt));
  TORCH_CHECK(
      slots.is_contiguous() && slots.numel() >= M &&
      (slots.scalar_type() == at::kLong || slots.scalar_type() == at::kInt));
  const bool fp8 = history.scalar_type() == at::kByte;
  const bool hot_fp8 = hot_scales.has_value() && hot_scales->defined() &&
                       hot_scales->numel() > 0;
  TORCH_CHECK((fp8 || history.scalar_type() == at::kHalf) &&
              history.dim() == 5 && history.size(1) == 2 &&
              history.size(3) == 1 && history.size(4) == 256 &&
              history.is_contiguous() && scales.scalar_type() == at::kFloat &&
              scales.is_contiguous());
  TORCH_CHECK(!hot_fp8 || fp8);
  TORCH_CHECK(hot.is_cuda() &&
              hot.scalar_type() == (hot_fp8 ? at::kByte : at::kHalf) &&
              hot.dim() == 3 && hot.size(1) == 2 && hot.size(2) == 256 &&
              hot.is_contiguous());
  TORCH_CHECK(tags.scalar_type() == at::kInt && tags.is_contiguous() &&
              page_slots.scalar_type() == at::kInt &&
              page_slots.is_contiguous() && epoch.scalar_type() == at::kInt &&
              epoch.is_contiguous() && epoch.numel() >= 1);
  auto st = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>(history.size(0));
  const int page_size = static_cast<int>(history.size(2));
  const int hot_pages = static_cast<int>(tags.numel());
  const int64_t page_slots_len = page_slots.numel();
  float* hs_ptr = hot_fp8 ? hot_scales->data_ptr<float>() : nullptr;

#define LAUNCH_PREP_WRITE(CS_T, CS_P, POS_T, SLOT_T, IS_FP8, IS_HOT_FP8) \
  host_prep::prep_write_kernel<CS_T, POS_T, SLOT_T, IS_FP8, IS_HOT_FP8>  \
      <<<static_cast<int>(M), 32 * (HQ + 2), 0, st>>>(                   \
          reinterpret_cast<const half*>(qkv.data_ptr()), qkv.stride(0),  \
          pos.data_ptr<POS_T>(), CS_P, rot,                              \
          reinterpret_cast<const half*>(qw.data_ptr()),                  \
          reinterpret_cast<const half*>(kw.data_ptr()),                  \
          static_cast<float>(eps),                                       \
          reinterpret_cast<half*>(query_out.data_ptr()),                 \
          reinterpret_cast<half*>(key_out.data_ptr()),                   \
          slots.data_ptr<SLOT_T>(), history.data_ptr(),                  \
          scales.data_ptr<float>(), tags.data_ptr<int32_t>(),            \
          page_slots.data_ptr<int32_t>(), hot.data_ptr(), hs_ptr,        \
          epoch.data_ptr<int32_t>(), HQ, blocks, page_size, hot_pages,   \
          page_slots_len)

#define DISPATCH_QUANTS(CS_T, CS_P, POS_T, SLOT_T)              \
  if (fp8 && hot_fp8) {                                         \
    LAUNCH_PREP_WRITE(CS_T, CS_P, POS_T, SLOT_T, true, true);   \
  } else if (fp8) {                                             \
    LAUNCH_PREP_WRITE(CS_T, CS_P, POS_T, SLOT_T, true, false);  \
  } else {                                                      \
    LAUNCH_PREP_WRITE(CS_T, CS_P, POS_T, SLOT_T, false, false); \
  }

#define DISPATCH_SLOTS(CS_T, CS_P, POS_T)       \
  if (slots.scalar_type() == at::kLong) {       \
    DISPATCH_QUANTS(CS_T, CS_P, POS_T, int64_t) \
  } else {                                      \
    DISPATCH_QUANTS(CS_T, CS_P, POS_T, int32_t) \
  }

#define DISPATCH_POS(CS_T, CS_P)        \
  if (pos.scalar_type() == at::kLong) { \
    DISPATCH_SLOTS(CS_T, CS_P, int64_t) \
  } else {                              \
    DISPATCH_SLOTS(CS_T, CS_P, int32_t) \
  }

  if (cos_sin.scalar_type() == at::kFloat) {
    DISPATCH_POS(float, cos_sin.data_ptr<float>())
  } else {
    DISPATCH_POS(half, reinterpret_cast<const half*>(cos_sin.data_ptr()))
  }
#undef DISPATCH_POS
#undef DISPATCH_SLOTS
#undef DISPATCH_QUANTS
#undef LAUNCH_PREP_WRITE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace hot_resolve {

template <typename PosT>
__global__ __launch_bounds__(256) void protect_hot_kernel(
    const int32_t* __restrict__ indices, const int32_t* __restrict__ table,
    const int32_t* __restrict__ requests, const PosT* __restrict__ positions,
    const int32_t* __restrict__ lengths, const int32_t* __restrict__ tags,
    const int32_t* __restrict__ page_slots, int32_t* __restrict__ stamps,
    const int32_t* __restrict__ epoch, int64_t* __restrict__ virtual_positions,
    int32_t* __restrict__ virtual_lengths, int32_t* __restrict__ resolved,
    int64_t* __restrict__ stats, int32_t* __restrict__ miss_queue, int width,
    int num_tiles, int table_width, int64_t table_stride, int64_t index_stride,
    int64_t resolved_stride, int num_requests, int num_blocks, int page_size,
    int hot_pages) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * 256 + threadIdx.x;
  const int32_t req = requests[row];
  const int64_t pos = static_cast<int64_t>(positions[row]);
  const bool valid_req = (req >= 0 && req < num_requests);
  const int64_t len = valid_req ? static_cast<int64_t>(lengths[req]) : 0;

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int64_t visible = max(min(pos + 1, len), static_cast<int64_t>(0));
    const int64_t selected_count =
        min(visible / 4, static_cast<int64_t>(width / 4)) * 4 +
        min(visible % 4, static_cast<int64_t>(width % 4));
    virtual_positions[row] = selected_count - 1;
    virtual_lengths[row] = static_cast<int32_t>(selected_count);
  }

  int32_t out_val = -2;
  if (col < width) {
    const int32_t logical =
        indices[static_cast<int64_t>(row) * index_stride + col];
    bool valid = valid_req && logical >= 0 &&
                 static_cast<int64_t>(logical) <= pos &&
                 static_cast<int64_t>(logical) < len &&
                 (logical / page_size) < table_width;
    int32_t blk = -1;
    if (valid) {
      blk = table[static_cast<int64_t>(req) * table_stride +
                  (logical / page_size)];
      valid = (blk >= 0 && blk < num_blocks);
    }
    if (valid) {
      const int32_t token = blk * page_size + (logical % page_size);
      const int32_t page = token >> 2;
      const int32_t reg = page_slots[page];
      if (reg >= 0 && reg < hot_pages && tags[reg] == page) {
        stamps[reg] = *epoch;
        out_val = (reg << 2) | (token & 3);
      } else {
        out_val = -1;
      }
    }
    resolved[static_cast<int64_t>(row) * resolved_stride + col] = out_val;
  }

  const uint32_t miss32 =
      __ballot_sync(0xffffffffu, col < width && out_val == -1);
  const uint32_t hit32 =
      __ballot_sync(0xffffffffu, col < width && out_val >= 0);
  const int half_shift = threadIdx.x & 16;
  const uint32_t miss16 = (miss32 >> half_shift) & 0xffffu;
  const uint32_t hit16 = (hit32 >> half_shift) & 0xffffu;
  const int tile = blockIdx.x * 16 + (threadIdx.x >> 4);
  if (tile < num_tiles && (threadIdx.x & 15) == 0) {
    if (miss16 == 0u) {
      if (hit16 != 0u) {
        const int64_t counter =
            (static_cast<int64_t>(row) * num_tiles + tile) * 3;
        stats[counter] += __popc(hit16);
      }
    } else {
      const int idx = atomicAdd(&miss_queue[0], 1);
      miss_queue[2 + idx] = (row << 16) | tile;
    }
  }
}

template <typename PosT, bool FP8, bool HOT_FP8>
__global__ __launch_bounds__(128) void install_hot_misses_kernel(
    const void* __restrict__ history_raw, const float* __restrict__ scales,
    void* __restrict__ hot_raw, float* __restrict__ hot_scales,
    int32_t* __restrict__ tags, int32_t* __restrict__ stamps,
    const int32_t* __restrict__ epoch, int32_t* __restrict__ hands,
    int32_t* __restrict__ page_slots, int64_t* __restrict__ stats,
    int32_t* __restrict__ miss_queue, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ table, const int32_t* __restrict__ requests,
    const PosT* __restrict__ positions, const int32_t* __restrict__ lengths,
    int32_t* __restrict__ resolved, int width, int num_tiles, int table_width,
    int64_t table_stride, int64_t index_stride, int64_t resolved_stride,
    int num_requests, int num_blocks, int page_size, int hot_pages,
    int num_hands) {
  const int total_misses =
      *reinterpret_cast<const volatile int32_t*>(&miss_queue[0]);
  if (blockIdx.x >= total_misses) return;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int32_t epoch_val = *epoch;

  for (int item = blockIdx.x; item < total_misses; item += gridDim.x) {
    const int32_t packed = miss_queue[2 + item];
    const int row = packed >> 16;
    const int tile = packed & 0xffff;
    const int col_base = tile * 16 + warp * 4;
    const int col = col_base + lane;

    const int32_t req = requests[row];
    const int64_t pos = static_cast<int64_t>(positions[row]);
    const bool valid_req = (req >= 0 && req < num_requests);
    const int64_t len = valid_req ? static_cast<int64_t>(lengths[req]) : 0;

    int32_t init_res = -2;
    bool valid = false;
    int32_t page = 0;
    int32_t off4 = 0;
    if (lane < 4 && col < width) {
      init_res = resolved[static_cast<int64_t>(row) * resolved_stride + col];
      const int32_t logical =
          indices[static_cast<int64_t>(row) * index_stride + col];
      valid = valid_req && logical >= 0 &&
              static_cast<int64_t>(logical) <= pos &&
              static_cast<int64_t>(logical) < len &&
              (logical / page_size) < table_width;
      if (valid) {
        const int32_t blk = table[static_cast<int64_t>(req) * table_stride +
                                  (logical / page_size)];
        if (blk >= 0 && blk < num_blocks) {
          const int32_t tok = blk * page_size + (logical % page_size);
          page = tok >> 2;
          off4 = tok & 3;
        } else {
          valid = false;
        }
      }
    }

    const uint32_t group_miss_mask =
        __ballot_sync(0xffffffffu, lane < 4 && init_res == -1);
    int32_t grp_hits = 0, grp_misses = 0, grp_contended = 0;

    if (group_miss_mask == 0u) {
      grp_hits = __popc(__ballot_sync(0xffffffffu, lane < 4 && init_res >= 0));
    } else {
      const int32_t first = __shfl_sync(0xffffffffu, page, 0);
      const int32_t v0 = __shfl_sync(0xffffffffu, valid ? 1 : 0, 0);
      const bool lane_ok = !valid || (page == first && off4 == lane);
      const bool aligned =
          (v0 != 0) &&
          (__ballot_sync(0xffffffffu, lane >= 4 || lane_ok) == 0xffffffffu);

      if (!aligned) {
        grp_hits =
            __popc(__ballot_sync(0xffffffffu, lane < 4 && init_res >= 0));
        grp_misses =
            __popc(__ballot_sync(0xffffffffu, lane < 4 && init_res == -1));
      } else {
        int32_t slot = -1;
        int hit_flag = 0, claim_flag = 0;
        if (lane == 0) {
          const int32_t reg = atomicAdd(&page_slots[first], 0);
          if (reg >= 0 && reg < hot_pages) {
            const int32_t t =
                *reinterpret_cast<const volatile int32_t*>(&tags[reg]);
            if (t == first && atomicCAS(&page_slots[first], reg, reg) == reg) {
              stamps[reg] = epoch_val;
              if (*reinterpret_cast<const volatile int32_t*>(&tags[reg]) ==
                      first &&
                  *reinterpret_cast<const volatile int32_t*>(
                      &page_slots[first]) == reg) {
                slot = reg;
                hit_flag = 1;
              }
            }
          }
          if (!hit_flag) {
            claim_flag = (atomicCAS(&page_slots[first], -1, -2) == -1) ? 1 : 0;
          }
        }
        const bool hit = __shfl_sync(0xffffffffu, hit_flag, 0) != 0;
        const bool claim = __shfl_sync(0xffffffffu, claim_flag, 0) != 0;
        slot = __shfl_sync(0xffffffffu, slot, 0);

        bool install = false;
        int32_t old_tag = -1;
        if (claim) {
          const uint32_t base_hand = static_cast<uint32_t>(
              ((static_cast<uint32_t>(first) * 2654435761u) >> 16) %
              static_cast<uint32_t>(num_hands));
          const uint32_t span = static_cast<uint32_t>(hot_pages / num_hands);
          const int probes_per_hand =
              max(1, min(8, static_cast<int>((span + 15u) >> 4)));
          const int hand_tries = min(num_hands, 8);
          for (int h = 0; h < hand_tries && !install; ++h) {
            const uint32_t lane_hand = (base_hand + static_cast<uint32_t>(h)) %
                                       static_cast<uint32_t>(num_hands);
            const uint32_t base = lane_hand * span;
            for (int probe_iter = 0; probe_iter < probes_per_hand && !install;
                 ++probe_iter) {
              uint32_t hand = 0;
              if (lane == 0) {
                hand = atomicAdd(reinterpret_cast<uint32_t*>(&hands[lane_hand]),
                                 16u);
              }
              hand = __shfl_sync(0xffffffffu, hand, 0);
              int32_t age = -1;
              if (lane < 16) {
                const uint32_t cand =
                    base + ((hand + static_cast<uint32_t>(lane)) % span);
                const int32_t st =
                    *reinterpret_cast<const volatile int32_t*>(&stamps[cand]);
                age = (st != epoch_val) ? (epoch_val - st) : -1;
              }
              int32_t best_age = age;
#pragma unroll
              for (int off = 8; off >= 1; off >>= 1) {
                const int32_t other =
                    __shfl_xor_sync(0xffffffffu, best_age, off);
                if (other > best_age) best_age = other;
              }
              best_age = __shfl_sync(0xffffffffu, best_age, 0);
              if (best_age >= 0) {
                const uint32_t rot = (static_cast<uint32_t>(first) * 5u) & 15u;
                const uint32_t best_mask =
                    __ballot_sync(0xffffffffu, lane < 16 && age == best_age);
                const uint32_t other_mask =
                    __ballot_sync(0xffffffffu, lane < 16 && age >= 0) &
                    ~best_mask;
                for (int pass = 0; pass < 2 && !install; ++pass) {
                  const uint32_t raw_mask = pass == 0 ? best_mask : other_mask;
                  uint32_t rem =
                      ((raw_mask >> rot) | (raw_mask << (16u - rot))) & 0xffffu;
                  while (rem != 0u && !install) {
                    const uint32_t bit = static_cast<uint32_t>(__ffs(rem) - 1);
                    rem &= (rem - 1u);
                    const uint32_t chosen = (bit + rot) & 15u;
                    const int32_t chosen_slot =
                        static_cast<int32_t>(base + ((hand + chosen) % span));
                    int won = 0;
                    if (lane == 0) {
                      const int32_t obs =
                          *reinterpret_cast<const volatile int32_t*>(
                              &stamps[chosen_slot]);
                      if (obs != epoch_val &&
                          atomicCAS(&stamps[chosen_slot], obs, epoch_val) ==
                              obs) {
                        won = 1;
                        slot = chosen_slot;
                        old_tag = *reinterpret_cast<const volatile int32_t*>(
                            &tags[chosen_slot]);
                        if (old_tag >= 0) {
                          atomicCAS(&page_slots[old_tag], chosen_slot, -1);
                        }
                      }
                    }
                    if (__shfl_sync(0xffffffffu, won, 0)) {
                      slot = __shfl_sync(0xffffffffu, slot, 0);
                      old_tag = __shfl_sync(0xffffffffu, old_tag, 0);
                      install = true;
                    }
                  }
                }
              }
            }
          }
        }

        if (install) {
          const int32_t phys_blk = (first * 4) / page_size;
          const int32_t phys_off = (first * 4) % page_size;
          if constexpr (HOT_FP8) {
            const auto* src_bytes =
                static_cast<const unsigned char*>(history_raw);
            auto* dst_bytes = static_cast<unsigned char*>(hot_raw);
            if (((reinterpret_cast<uintptr_t>(src_bytes) |
                  reinterpret_cast<uintptr_t>(dst_bytes)) &
                 15u) == 0u) {
              const int tok_sub = lane >> 4;
              const int sub16 = lane & 15;
#pragma unroll
              for (int tp = 0; tp < 2; ++tp) {
                const int tok_idx = tp * 2 + tok_sub;
#pragma unroll
                for (int kv = 0; kv < 2; ++kv) {
                  const unsigned char* s_ptr =
                      src_bytes +
                      ((static_cast<int64_t>(phys_blk) * 2 + kv) * page_size +
                       phys_off + tok_idx) *
                          256;
                  unsigned char* d_ptr =
                      dst_bytes +
                      (static_cast<int64_t>(slot * 4 + tok_idx) * 2 + kv) * 256;
                  reinterpret_cast<uint4*>(d_ptr)[sub16] =
                      reinterpret_cast<const uint4*>(s_ptr)[sub16];
                }
              }
            } else {
              const int tok_idx = lane >> 3;
              const int sub8 = lane & 7;
#pragma unroll
              for (int kv = 0; kv < 2; ++kv) {
                const unsigned char* s_ptr =
                    src_bytes +
                    ((static_cast<int64_t>(phys_blk) * 2 + kv) * page_size +
                     phys_off + tok_idx) *
                        256;
                unsigned char* d_ptr =
                    dst_bytes +
                    (static_cast<int64_t>(slot * 4 + tok_idx) * 2 + kv) * 256;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                  reinterpret_cast<uint2*>(d_ptr)[sub8 + j * 8] =
                      reinterpret_cast<const uint2*>(s_ptr)[sub8 + j * 8];
                }
              }
            }
            if (lane < 8) {
              hot_scales[slot * 8 + lane] = scales[first * 8 + lane];
            }
          } else if constexpr (FP8) {
            const auto* src_bytes =
                static_cast<const unsigned char*>(history_raw);
            auto* dst_half = static_cast<half*>(hot_raw);
#pragma unroll
            for (int tok_idx = 0; tok_idx < 4; ++tok_idx) {
#pragma unroll
              for (int kv = 0; kv < 2; ++kv) {
                const unsigned char* s_ptr =
                    src_bytes +
                    ((static_cast<int64_t>(phys_blk) * 2 + kv) * page_size +
                     phys_off + tok_idx) *
                        256;
                half* d_ptr =
                    dst_half +
                    (static_cast<int64_t>(slot * 4 + tok_idx) * 2 + kv) * 256;
                const uint2 packed_codes =
                    reinterpret_cast<const uint2*>(s_ptr)[lane];
                const float sc = scales[(first * 4 + tok_idx) * 2 + kv];
                reinterpret_cast<uint4*>(d_ptr)[lane] =
                    device_qsa::decode8_bits(packed_codes, sc);
              }
            }
          } else {
            const auto* src_half = static_cast<const half*>(history_raw);
            auto* dst_half = static_cast<half*>(hot_raw);
#pragma unroll
            for (int tok_idx = 0; tok_idx < 4; ++tok_idx) {
#pragma unroll
              for (int kv = 0; kv < 2; ++kv) {
                const half* s_ptr =
                    src_half +
                    ((static_cast<int64_t>(phys_blk) * 2 + kv) * page_size +
                     phys_off + tok_idx) *
                        256;
                half* d_ptr =
                    dst_half +
                    (static_cast<int64_t>(slot * 4 + tok_idx) * 2 + kv) * 256;
                reinterpret_cast<uint4*>(d_ptr)[lane] =
                    reinterpret_cast<const uint4*>(s_ptr)[lane];
              }
            }
          }
        }

        __syncwarp();
        if (lane == 0) {
          if (install) {
            __threadfence();
            atomicCAS(&tags[slot], old_tag, first);
            __threadfence();
            atomicCAS(&page_slots[first], -2, slot);
          } else if (claim) {
            atomicCAS(&page_slots[first], -2, -1);
          }
        }
        __syncwarp();

        if (lane < 4 && col < width) {
          resolved[static_cast<int64_t>(row) * resolved_stride + col] =
              valid ? ((hit || install) ? ((slot << 2) | lane) : -1) : -2;
        }
        const int valid_cnt = __popc(
            __ballot_sync(0xffffffffu, lane < 4 && col < width && valid));
        grp_hits = hit ? valid_cnt : 0;
        grp_misses = hit ? 0 : valid_cnt;
        grp_contended = (!hit && !install) ? 1 : 0;
      }
    }

    if (lane == 0) {
      const int64_t base_counter =
          (static_cast<int64_t>(row) * num_tiles + tile) * 3;
      if (grp_hits != 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&stats[base_counter]),
                  static_cast<unsigned long long>(grp_hits));
      }
      if (grp_misses != 0) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(&stats[base_counter + 1]),
            static_cast<unsigned long long>(grp_misses));
      }
      if (grp_contended != 0) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(&stats[base_counter + 2]),
            static_cast<unsigned long long>(grp_contended));
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const int active_blocks = min(total_misses, static_cast<int>(gridDim.x));
    const int done = atomicAdd(&miss_queue[1], 1) + 1;
    if (done == active_blocks) {
      miss_queue[1] = 0;
      __threadfence();
      *reinterpret_cast<volatile int32_t*>(&miss_queue[0]) = 0;
    }
  }
}

template <bool FP8, bool HOT_FP8>
__global__ __launch_bounds__(128) void warm_hot_staged_kernel(
    const void* __restrict__ staged_raw,
    const float* __restrict__ staged_scales, void* __restrict__ hot_raw,
    float* __restrict__ hot_scales, int32_t* __restrict__ tags,
    int32_t* __restrict__ stamps, const int32_t* __restrict__ epoch,
    int32_t* __restrict__ hands, int32_t* __restrict__ page_slots,
    const int32_t* __restrict__ table, const int32_t* __restrict__ staged_table,
    const int32_t* __restrict__ lengths, int num_requests, int table_width,
    int64_t table_stride, int64_t staged_stride, int num_blocks,
    int staged_blocks, int page_size, int hot_pages, int num_hands,
    int per_req_budget) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int global_warp = blockIdx.x * 4 + warp;
  const int total_warps = gridDim.x * 4;
  const int total_items = num_requests * per_req_budget;
  const int32_t epoch_val = *epoch;
  constexpr int kSinkPages = 64;

  for (int item = global_warp; item < total_items; item += total_warps) {
    const int req = item / per_req_budget;
    const int idx_in_budget = item % per_req_budget;
    const int full_p4 = lengths[req] / 4;
    if (idx_in_budget >= full_p4) continue;

    int lp4 = idx_in_budget;
    if (full_p4 > per_req_budget && idx_in_budget >= kSinkPages) {
      lp4 = (full_p4 - (per_req_budget - kSinkPages)) +
            (idx_in_budget - kSinkPages);
    }
    const int logical0 = lp4 * 4;
    const int blk_col = logical0 / page_size;
    const int off_in_page = logical0 % page_size;
    if (blk_col >= table_width) continue;

    const int32_t phys_blk =
        table[static_cast<int64_t>(req) * table_stride + blk_col];
    const int32_t stg_blk =
        staged_table[static_cast<int64_t>(req) * staged_stride + blk_col];
    if (phys_blk < 0 || phys_blk >= num_blocks || stg_blk < 0 ||
        stg_blk >= staged_blocks) {
      continue;
    }

    const int32_t first = (phys_blk * page_size + off_in_page) >> 2;
    int32_t slot = -1;
    int hit_flag = 0, claim_flag = 0;
    if (lane == 0) {
      const int32_t reg = atomicAdd(&page_slots[first], 0);
      if (reg >= 0 && reg < hot_pages &&
          *reinterpret_cast<const volatile int32_t*>(&tags[reg]) == first) {
        stamps[reg] = epoch_val;
        hit_flag = 1;
      } else if (reg == -1) {
        claim_flag = (atomicCAS(&page_slots[first], -1, -2) == -1) ? 1 : 0;
      }
    }
    if (__shfl_sync(0xffffffffu, hit_flag, 0) != 0) continue;
    const bool claim = __shfl_sync(0xffffffffu, claim_flag, 0) != 0;
    if (!claim) continue;

    bool install = false;
    int32_t old_tag = -1;
    const uint32_t base_hand = static_cast<uint32_t>(
        ((static_cast<uint32_t>(first) * 2654435761u) >> 16) %
        static_cast<uint32_t>(num_hands));
    const uint32_t span = static_cast<uint32_t>(hot_pages / num_hands);
    const int probes_per_hand =
        max(1, min(8, static_cast<int>((span + 15u) >> 4)));
    const int hand_tries = min(num_hands, 8);
    for (int h = 0; h < hand_tries && !install; ++h) {
      const uint32_t lane_hand = (base_hand + static_cast<uint32_t>(h)) %
                                 static_cast<uint32_t>(num_hands);
      const uint32_t base = lane_hand * span;
      for (int probe_iter = 0; probe_iter < probes_per_hand && !install;
           ++probe_iter) {
        uint32_t hand = 0;
        if (lane == 0) {
          hand = atomicAdd(reinterpret_cast<uint32_t*>(&hands[lane_hand]), 16u);
        }
        hand = __shfl_sync(0xffffffffu, hand, 0);
        int32_t age = -1;
        if (lane < 16) {
          const uint32_t cand =
              base + ((hand + static_cast<uint32_t>(lane)) % span);
          const int32_t st =
              *reinterpret_cast<const volatile int32_t*>(&stamps[cand]);
          age = (st != epoch_val) ? (epoch_val - st) : -1;
        }
        int32_t best_age = age;
#pragma unroll
        for (int off = 8; off >= 1; off >>= 1) {
          const int32_t other = __shfl_xor_sync(0xffffffffu, best_age, off);
          if (other > best_age) best_age = other;
        }
        best_age = __shfl_sync(0xffffffffu, best_age, 0);
        if (best_age >= 0) {
          const uint32_t rot = (static_cast<uint32_t>(first) * 5u) & 15u;
          const uint32_t best_mask =
              __ballot_sync(0xffffffffu, lane < 16 && age == best_age);
          const uint32_t other_mask =
              __ballot_sync(0xffffffffu, lane < 16 && age >= 0) & ~best_mask;
          for (int pass = 0; pass < 2 && !install; ++pass) {
            const uint32_t raw_mask = pass == 0 ? best_mask : other_mask;
            uint32_t rem =
                ((raw_mask >> rot) | (raw_mask << (16u - rot))) & 0xffffu;
            while (rem != 0u && !install) {
              const uint32_t bit = static_cast<uint32_t>(__ffs(rem) - 1);
              rem &= (rem - 1u);
              const uint32_t chosen = (bit + rot) & 15u;
              const int32_t chosen_slot =
                  static_cast<int32_t>(base + ((hand + chosen) % span));
              int won = 0;
              if (lane == 0) {
                const int32_t obs = *reinterpret_cast<const volatile int32_t*>(
                    &stamps[chosen_slot]);
                if (obs != epoch_val &&
                    atomicCAS(&stamps[chosen_slot], obs, epoch_val) == obs) {
                  won = 1;
                  slot = chosen_slot;
                  old_tag = *reinterpret_cast<const volatile int32_t*>(
                      &tags[chosen_slot]);
                  if (old_tag >= 0) {
                    atomicCAS(&page_slots[old_tag], chosen_slot, -1);
                  }
                }
              }
              if (__shfl_sync(0xffffffffu, won, 0)) {
                slot = __shfl_sync(0xffffffffu, slot, 0);
                old_tag = __shfl_sync(0xffffffffu, old_tag, 0);
                install = true;
              }
            }
          }
        }
      }
    }

    if (install) {
      const int32_t stg_first = (stg_blk * page_size + off_in_page) >> 2;
      if constexpr (HOT_FP8) {
        const auto* src_bytes = static_cast<const unsigned char*>(staged_raw);
        auto* dst_bytes = static_cast<unsigned char*>(hot_raw);
        const int tok_idx = lane >> 3;
        const int sub8 = lane & 7;
#pragma unroll
        for (int kv = 0; kv < 2; ++kv) {
          const unsigned char* s_ptr =
              src_bytes +
              ((static_cast<int64_t>(stg_blk) * 2 + kv) * page_size +
               off_in_page + tok_idx) *
                  256;
          unsigned char* d_ptr =
              dst_bytes +
              (static_cast<int64_t>(slot * 4 + tok_idx) * 2 + kv) * 256;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            reinterpret_cast<uint2*>(d_ptr)[sub8 + j * 8] =
                reinterpret_cast<const uint2*>(s_ptr)[sub8 + j * 8];
          }
        }
        if (lane < 8) {
          hot_scales[slot * 8 + lane] = staged_scales[stg_first * 8 + lane];
        }
      } else if constexpr (FP8) {
        const auto* src_bytes = static_cast<const unsigned char*>(staged_raw);
        auto* dst_half = static_cast<half*>(hot_raw);
#pragma unroll
        for (int tok_idx = 0; tok_idx < 4; ++tok_idx) {
#pragma unroll
          for (int kv = 0; kv < 2; ++kv) {
            const unsigned char* s_ptr =
                src_bytes +
                ((static_cast<int64_t>(stg_blk) * 2 + kv) * page_size +
                 off_in_page + tok_idx) *
                    256;
            half* d_ptr =
                dst_half +
                (static_cast<int64_t>(slot * 4 + tok_idx) * 2 + kv) * 256;
            const uint2 packed_codes =
                reinterpret_cast<const uint2*>(s_ptr)[lane];
            const float sc = staged_scales[(stg_first * 4 + tok_idx) * 2 + kv];
            reinterpret_cast<uint4*>(d_ptr)[lane] =
                device_qsa::decode8_bits(packed_codes, sc);
          }
        }
      } else {
        const auto* src_half = static_cast<const half*>(staged_raw);
        auto* dst_half = static_cast<half*>(hot_raw);
#pragma unroll
        for (int tok_idx = 0; tok_idx < 4; ++tok_idx) {
#pragma unroll
          for (int kv = 0; kv < 2; ++kv) {
            const half* s_ptr =
                src_half +
                ((static_cast<int64_t>(stg_blk) * 2 + kv) * page_size +
                 off_in_page + tok_idx) *
                    256;
            half* d_ptr =
                dst_half +
                (static_cast<int64_t>(slot * 4 + tok_idx) * 2 + kv) * 256;
            reinterpret_cast<uint4*>(d_ptr)[lane] =
                reinterpret_cast<const uint4*>(s_ptr)[lane];
          }
        }
      }
    }

    __syncwarp();
    if (lane == 0) {
      if (install) {
        __threadfence();
        atomicCAS(&tags[slot], old_tag, first);
        __threadfence();
        atomicCAS(&page_slots[first], -2, slot);
      } else {
        atomicCAS(&page_slots[first], -2, -1);
      }
    }
    __syncwarp();
  }
}

}  // namespace hot_resolve

void resolve_hot_out(torch::Tensor history, torch::Tensor scales,
                     torch::Tensor hot, std::optional<torch::Tensor> hot_scales,
                     torch::Tensor tags, torch::Tensor stamps,
                     torch::Tensor epoch, torch::Tensor hands,
                     torch::Tensor page_slots, torch::Tensor stats,
                     torch::Tensor miss_queue, torch::Tensor indices,
                     torch::Tensor table, torch::Tensor requests,
                     torch::Tensor positions, torch::Tensor lengths,
                     torch::Tensor virtual_positions,
                     torch::Tensor virtual_lengths, torch::Tensor resolved) {
  const int rows = indices.size(0);
  const int width = indices.size(1);
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(indices.device());
  const bool fp8 = history.scalar_type() == at::kByte;
  const bool hot_fp8 = hot_scales.has_value() && hot_scales->numel() > 0;
  TORCH_CHECK((fp8 || history.scalar_type() == at::kHalf) &&
              history.dim() == 5 && history.size(1) == 2 &&
              history.size(2) % 4 == 0 && history.size(3) == 1 &&
              history.size(4) == 256 && history.is_contiguous());
  TORCH_CHECK(!hot_fp8 || fp8,
              "Encoded hot storage requires E4M3 authoritative history");
  const int num_blocks = history.size(0);
  const int page_size = history.size(2);
  const int hot_pages = hot.size(0) / 4;
  const int num_hands = hands.numel();
  const int num_tiles = (width + 15) / 16;
  const int table_width = table.size(1);
  const int num_requests = table.size(0);
  TORCH_CHECK(miss_queue.numel() >= rows * num_tiles + 2 &&
              stats.numel() >= rows * num_tiles * 3);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 protect_grid(std::max(1, (width + 255) / 256), rows);
  const dim3 install_grid(std::min(rows * std::max(1, num_tiles), 160));
  float* hs_ptr = hot_fp8 ? hot_scales->data_ptr<float>() : nullptr;

  auto launch = [&](auto pos_tag) {
    using PosT = decltype(pos_tag);
    hot_resolve::protect_hot_kernel<PosT><<<protect_grid, 256, 0, stream>>>(
        indices.data_ptr<int32_t>(), table.data_ptr<int32_t>(),
        requests.data_ptr<int32_t>(), positions.data_ptr<PosT>(),
        lengths.data_ptr<int32_t>(), tags.data_ptr<int32_t>(),
        page_slots.data_ptr<int32_t>(), stamps.data_ptr<int32_t>(),
        epoch.data_ptr<int32_t>(), virtual_positions.data_ptr<int64_t>(),
        virtual_lengths.data_ptr<int32_t>(), resolved.data_ptr<int32_t>(),
        stats.data_ptr<int64_t>(), miss_queue.data_ptr<int32_t>(), width,
        num_tiles, table_width, table.stride(0), indices.stride(0),
        resolved.stride(0), num_requests, num_blocks, page_size, hot_pages);
    if (hot_fp8) {
      hot_resolve::install_hot_misses_kernel<PosT, true, true>
          <<<install_grid, 128, 0, stream>>>(
              history.data_ptr(), scales.data_ptr<float>(), hot.data_ptr(),
              hs_ptr, tags.data_ptr<int32_t>(), stamps.data_ptr<int32_t>(),
              epoch.data_ptr<int32_t>(), hands.data_ptr<int32_t>(),
              page_slots.data_ptr<int32_t>(), stats.data_ptr<int64_t>(),
              miss_queue.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
              table.data_ptr<int32_t>(), requests.data_ptr<int32_t>(),
              positions.data_ptr<PosT>(), lengths.data_ptr<int32_t>(),
              resolved.data_ptr<int32_t>(), width, num_tiles, table_width,
              table.stride(0), indices.stride(0), resolved.stride(0),
              num_requests, num_blocks, page_size, hot_pages, num_hands);
    } else if (fp8) {
      hot_resolve::install_hot_misses_kernel<PosT, true, false>
          <<<install_grid, 128, 0, stream>>>(
              history.data_ptr(), scales.data_ptr<float>(), hot.data_ptr(),
              nullptr, tags.data_ptr<int32_t>(), stamps.data_ptr<int32_t>(),
              epoch.data_ptr<int32_t>(), hands.data_ptr<int32_t>(),
              page_slots.data_ptr<int32_t>(), stats.data_ptr<int64_t>(),
              miss_queue.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
              table.data_ptr<int32_t>(), requests.data_ptr<int32_t>(),
              positions.data_ptr<PosT>(), lengths.data_ptr<int32_t>(),
              resolved.data_ptr<int32_t>(), width, num_tiles, table_width,
              table.stride(0), indices.stride(0), resolved.stride(0),
              num_requests, num_blocks, page_size, hot_pages, num_hands);
    } else {
      hot_resolve::install_hot_misses_kernel<PosT, false, false>
          <<<install_grid, 128, 0, stream>>>(
              history.data_ptr(), scales.data_ptr<float>(), hot.data_ptr(),
              nullptr, tags.data_ptr<int32_t>(), stamps.data_ptr<int32_t>(),
              epoch.data_ptr<int32_t>(), hands.data_ptr<int32_t>(),
              page_slots.data_ptr<int32_t>(), stats.data_ptr<int64_t>(),
              miss_queue.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
              table.data_ptr<int32_t>(), requests.data_ptr<int32_t>(),
              positions.data_ptr<PosT>(), lengths.data_ptr<int32_t>(),
              resolved.data_ptr<int32_t>(), width, num_tiles, table_width,
              table.stride(0), indices.stride(0), resolved.stride(0),
              num_requests, num_blocks, page_size, hot_pages, num_hands);
    }
  };
  if (positions.scalar_type() == at::kLong) {
    launch(int64_t{});
  } else {
    TORCH_CHECK(positions.scalar_type() == at::kInt);
    launch(int32_t{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void warm_hot_staged_out(torch::Tensor staged_history,
                         torch::Tensor staged_scales, torch::Tensor hot,
                         std::optional<torch::Tensor> hot_scales,
                         torch::Tensor tags, torch::Tensor stamps,
                         torch::Tensor epoch, torch::Tensor hands,
                         torch::Tensor page_slots, torch::Tensor table,
                         torch::Tensor staged_table, torch::Tensor lengths,
                         int64_t num_blocks) {
  const int num_requests = table.size(0);
  const int staged_blocks = staged_history.size(0);
  if (num_requests <= 0 || staged_blocks <= 0) return;
  const c10::cuda::CUDAGuard guard(hot.device());
  const bool fp8 = staged_history.scalar_type() == at::kByte;
  const bool hot_fp8 = hot_scales.has_value() && hot_scales->numel() > 0;
  const int page_size = staged_history.size(2);
  const int hot_pages = hot.size(0) / 4;
  if (hot_pages <= 0) return;
  const int per_req_budget =
      std::min(staged_blocks * (page_size / 4),
               std::max(16, hot_pages / std::max(1, num_requests)));
  const int total_items = num_requests * per_req_budget;
  if (total_items <= 0) return;
  const dim3 grid(std::min((total_items + 3) / 4, 160));
  const auto stream = at::cuda::getCurrentCUDAStream();
  float* hs_ptr = hot_fp8 ? hot_scales->data_ptr<float>() : nullptr;
  if (hot_fp8) {
    hot_resolve::warm_hot_staged_kernel<true, true><<<grid, 128, 0, stream>>>(
        staged_history.data_ptr(), staged_scales.data_ptr<float>(),
        hot.data_ptr(), hs_ptr, tags.data_ptr<int32_t>(),
        stamps.data_ptr<int32_t>(), epoch.data_ptr<int32_t>(),
        hands.data_ptr<int32_t>(), page_slots.data_ptr<int32_t>(),
        table.data_ptr<int32_t>(), staged_table.data_ptr<int32_t>(),
        lengths.data_ptr<int32_t>(), num_requests, table.size(1),
        table.stride(0), staged_table.stride(0), static_cast<int>(num_blocks),
        staged_blocks, page_size, hot_pages, hands.numel(), per_req_budget);
  } else if (fp8) {
    hot_resolve::warm_hot_staged_kernel<true, false><<<grid, 128, 0, stream>>>(
        staged_history.data_ptr(), staged_scales.data_ptr<float>(),
        hot.data_ptr(), nullptr, tags.data_ptr<int32_t>(),
        stamps.data_ptr<int32_t>(), epoch.data_ptr<int32_t>(),
        hands.data_ptr<int32_t>(), page_slots.data_ptr<int32_t>(),
        table.data_ptr<int32_t>(), staged_table.data_ptr<int32_t>(),
        lengths.data_ptr<int32_t>(), num_requests, table.size(1),
        table.stride(0), staged_table.stride(0), static_cast<int>(num_blocks),
        staged_blocks, page_size, hot_pages, hands.numel(), per_req_budget);
  } else {
    hot_resolve::warm_hot_staged_kernel<false, false><<<grid, 128, 0, stream>>>(
        staged_history.data_ptr(), staged_scales.data_ptr<float>(),
        hot.data_ptr(), nullptr, tags.data_ptr<int32_t>(),
        stamps.data_ptr<int32_t>(), epoch.data_ptr<int32_t>(),
        hands.data_ptr<int32_t>(), page_slots.data_ptr<int32_t>(),
        table.data_ptr<int32_t>(), staged_table.data_ptr<int32_t>(),
        lengths.data_ptr<int32_t>(), num_requests, table.size(1),
        table.stride(0), staged_table.stride(0), static_cast<int>(num_blocks),
        staged_blocks, page_size, hot_pages, hands.numel(), per_req_budget);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(vllm_sm70_qsa_device, m) {
  m.def(
      "stage_host_run(Tensor history, Tensor scales, Tensor(a!) out, "
      "Tensor(b!) out_scales, int source_page, int target_page, int pages) "
      "-> bool");
  m.impl("stage_host_run", torch::kCUDA, &stage_host_run);
  m.def(
      "run(Tensor q, Tensor history, Tensor scales, Tensor indices, Tensor "
      "table, Tensor requests, Tensor positions, Tensor lengths, Tensor(a!) "
      "out, Tensor? gate, Tensor(b!) partial, Tensor(c!) state) -> ()");
  m.impl("run", torch::kCUDA,
         [](torch::Tensor q, torch::Tensor history, torch::Tensor scales,
            torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
            torch::Tensor positions, torch::Tensor lengths, torch::Tensor out,
            std::optional<torch::Tensor> gate, torch::Tensor partial,
            torch::Tensor state) {
           device_qsa_out(q, history, scales, indices, table, requests,
                          positions, lengths, out, gate, partial, state);
         });
  // Prefill over device (staged) history: every row in `step`-row tiles that
  // reuse one partial/state workspace, launched from C++ instead of a Python
  // loop. Same kernels and arithmetic as `run` on each tile.
  m.def(
      "run_tiled(Tensor q, Tensor history, Tensor scales, Tensor indices, "
      "Tensor table, Tensor requests, Tensor positions, Tensor lengths, "
      "Tensor(a!) out, Tensor? gate, Tensor(b!) partial, Tensor(c!) state, "
      "int step, int tiles=1) -> ()");
  m.impl("run_tiled", torch::kCUDA,
         [](torch::Tensor q, torch::Tensor history, torch::Tensor scales,
            torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
            torch::Tensor positions, torch::Tensor lengths, torch::Tensor out,
            std::optional<torch::Tensor> gate, torch::Tensor partial,
            torch::Tensor state, int64_t step, int64_t tiles) {
           TORCH_CHECK(step > 0 && step <= 64);
           const int64_t rows = q.size(0);
           TORCH_CHECK(indices.size(0) == rows && requests.size(0) == rows &&
                       positions.size(0) == rows && out.size(0) == rows &&
                       (!gate || gate->size(0) == rows));
           for (int64_t start = 0; start < rows; start += step) {
             const int64_t n = std::min(step, rows - start);
             std::optional<torch::Tensor> g;
             if (gate) g = gate->narrow(0, start, n);
             device_qsa_out(q.narrow(0, start, n), history, scales,
                            indices.narrow(0, start, n), table,
                            requests.narrow(0, start, n),
                            positions.narrow(0, start, n), lengths,
                            out.narrow(0, start, n), g, partial, state,
                            std::nullopt, std::nullopt, tiles, true);
           }
         });
  // Prefill over staged E4M3 history: one CTA per row over all of its keys,
  // gated output written directly (no workspace, no merge).
  m.def(
      "run_rows(Tensor q, Tensor history, Tensor scales, Tensor indices, "
      "Tensor table, Tensor requests, Tensor positions, Tensor lengths, "
      "Tensor(a!) out, Tensor? gate, int occupancy=3) -> ()");
  m.impl("run_rows", torch::kCUDA,
         [](torch::Tensor q, torch::Tensor history, torch::Tensor scales,
            torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
            torch::Tensor positions, torch::Tensor lengths, torch::Tensor out,
            std::optional<torch::Tensor> gate, int64_t occupancy) {
           const auto empty = at::empty({0}, q.options().dtype(at::kFloat));
           device_qsa_out(q, history, scales, indices, table, requests,
                          positions, lengths, out, gate, empty, empty,
                          std::nullopt, std::nullopt, 1, true, occupancy);
         });
  // One window of a staged-in-parts history: window-normalized FP32 rows
  // and per-head (max, sum); qsa_merge_windows combines the windows.
  m.def(
      "run_rows_window(Tensor q, Tensor history, Tensor scales, Tensor "
      "indices, "
      "Tensor table, Tensor requests, Tensor positions, Tensor lengths, "
      "Tensor(a!) partial, Tensor(b!) state, int occupancy=12) -> ()");
  m.impl("run_rows_window", torch::kCUDA,
         [](torch::Tensor q, torch::Tensor history, torch::Tensor scales,
            torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
            torch::Tensor positions, torch::Tensor lengths,
            torch::Tensor partial, torch::Tensor state, int64_t occupancy) {
           // Window rows write only partial/state; `q` stands in for out.
           device_qsa_out(q, history, scales, indices, table, requests,
                          positions, lengths, q, std::nullopt, partial, state,
                          std::nullopt, std::nullopt, 1, true, occupancy,
                          std::nullopt, true);
         });
  // run_tiled over one window of a staged-in-parts history (FP16 draft):
  // window-normalized FP32 rows and per-head (max, sum) for
  // qsa_merge_windows instead of the gated output.
  m.def(
      "run_tiled_window(Tensor q, Tensor history, Tensor scales, "
      "Tensor indices, Tensor table, Tensor requests, Tensor positions, "
      "Tensor lengths, Tensor(a!) partial, Tensor(b!) state, "
      "Tensor(c!) window_out, Tensor(d!) window_state, int step, "
      "int tiles=1) -> ()");
  m.impl(
      "run_tiled_window", torch::kCUDA,
      [](torch::Tensor q, torch::Tensor history, torch::Tensor scales,
         torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
         torch::Tensor positions, torch::Tensor lengths, torch::Tensor partial,
         torch::Tensor state, torch::Tensor window_out,
         torch::Tensor window_state, int64_t step, int64_t tiles) {
        TORCH_CHECK(step > 0 && step <= 64);
        const int64_t rows = q.size(0);
        TORCH_CHECK(indices.size(0) == rows && requests.size(0) == rows &&
                    positions.size(0) == rows && window_out.size(0) == rows &&
                    window_state.size(0) == rows);
        for (int64_t start = 0; start < rows; start += step) {
          const int64_t n = std::min(step, rows - start);
          // Window rows write only window_out/state; `q` stands in for out.
          device_qsa_out(
              q.narrow(0, start, n), history, scales,
              indices.narrow(0, start, n), table, requests.narrow(0, start, n),
              positions.narrow(0, start, n), lengths, q.narrow(0, start, n),
              std::nullopt, partial, state, std::nullopt, std::nullopt, tiles,
              true, 0, std::nullopt, false, window_out.narrow(0, start, n),
              window_state.narrow(0, start, n));
        }
      });
  // Host-authoritative history with a protected device hot cache: resolved
  // tokens read the hot copy, the rest read mapped host history directly.
  m.def(
      "run_hot(Tensor q, Tensor history, Tensor scales, Tensor indices, Tensor "
      "table, Tensor requests, Tensor positions, Tensor lengths, Tensor(a!) "
      "out, Tensor? gate, Tensor(b!) partial, Tensor(c!) state, Tensor hot, "
      "Tensor resolved, int tiles=1, Tensor? page_slots=None, "
      "Tensor? tags=None) -> ()");
  m.impl("run_hot", torch::kCUDA,
         [](torch::Tensor q, torch::Tensor history, torch::Tensor scales,
            torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
            torch::Tensor positions, torch::Tensor lengths, torch::Tensor out,
            std::optional<torch::Tensor> gate, torch::Tensor partial,
            torch::Tensor state, torch::Tensor hot, torch::Tensor resolved,
            int64_t tiles, std::optional<torch::Tensor> page_slots,
            std::optional<torch::Tensor> tags) {
           device_qsa_out(q, history, scales, indices, table, requests,
                          positions, lengths, out, gate, partial, state, hot,
                          resolved, tiles, false, 0, std::nullopt, false,
                          std::nullopt, std::nullopt, page_slots, tags);
         });
  m.def(
      "run_hot_fp8(Tensor q, Tensor history, Tensor scales, Tensor indices, "
      "Tensor table, Tensor requests, Tensor positions, Tensor lengths, "
      "Tensor(a!) out, Tensor? gate, Tensor(b!) partial, Tensor(c!) state, "
      "Tensor hot, Tensor resolved, Tensor hot_scales, int tiles=1, "
      "Tensor? page_slots=None, Tensor? tags=None) -> ()");
  m.impl("run_hot_fp8", torch::kCUDA,
         [](torch::Tensor q, torch::Tensor history, torch::Tensor scales,
            torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
            torch::Tensor positions, torch::Tensor lengths, torch::Tensor out,
            std::optional<torch::Tensor> gate, torch::Tensor partial,
            torch::Tensor state, torch::Tensor hot, torch::Tensor resolved,
            torch::Tensor hot_scales, int64_t tiles,
            std::optional<torch::Tensor> page_slots,
            std::optional<torch::Tensor> tags) {
           device_qsa_out(q, history, scales, indices, table, requests,
                          positions, lengths, out, gate, partial, state, hot,
                          resolved, tiles, false, 0, hot_scales, false,
                          std::nullopt, std::nullopt, page_slots, tags);
         });
  m.def(
      "prep_write(Tensor qkv, Tensor pos, Tensor cos_sin, Tensor qw, Tensor "
      "kw, "
      "float eps, Tensor(a!) query_out, Tensor(b!) key_out, Tensor slots, "
      "Tensor(c!) history, Tensor(d!) scales, Tensor tags, Tensor page_slots, "
      "Tensor(e!) hot, Tensor(f!)? hot_scales, Tensor(g!) epoch) -> ()");
  m.impl("prep_write", torch::kCUDA, &prep_write_out);
  m.def(
      "resolve_hot(Tensor history, Tensor scales, Tensor(a!) hot, "
      "Tensor(b!)? hot_scales, Tensor(c!) tags, Tensor(d!) stamps, "
      "Tensor epoch, Tensor(e!) hands, Tensor(f!) page_slots, "
      "Tensor(g!) stats, Tensor(h!) miss_queue, Tensor indices, "
      "Tensor table, Tensor requests, Tensor positions, Tensor lengths, "
      "Tensor(i!) virtual_positions, Tensor(j!) virtual_lengths, "
      "Tensor(k!) resolved) -> ()");
  m.impl("resolve_hot", torch::kCUDA, &resolve_hot_out);
  m.def(
      "warm_hot_staged(Tensor staged_history, Tensor staged_scales, "
      "Tensor(a!) hot, Tensor(b!)? hot_scales, Tensor(c!) tags, "
      "Tensor(d!) stamps, Tensor epoch, Tensor(e!) hands, "
      "Tensor(f!) page_slots, Tensor table, Tensor staged_table, "
      "Tensor lengths, int num_blocks) -> ()");
  m.impl("warm_hot_staged", torch::kCUDA, &warm_hot_staged_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
