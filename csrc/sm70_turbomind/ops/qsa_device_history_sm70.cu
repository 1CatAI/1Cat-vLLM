#include <algorithm>
#include <type_traits>
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
  const half* hot;      // [hot_tokens, 2, D] FP16, or null
  const int* resolved;  // [M, width] hot token, -1 host, -2 invalid
  half* out;
  float* partial;
  float* state;
  int M, blocks, page, width, table_width, num_requests, splits, tiles;
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

__device__ __forceinline__ uint4 decode8(const unsigned char* source,
                                         float scale, const float* lut) {
  return decode8_raw(__ldg(reinterpret_cast<const uint2*>(source)), scale, lut);
}

// Read one 8-value chunk of token K (kv=0) or V (kv=1): from the protected
// hot cache when resolved there, else from the authoritative history, which
// may be pinned host memory mapped into the device address space (UVA).
template <bool FP8, bool HOT>
__device__ __forceinline__ uint4 load_kv(const Args& a, int64_t slot, int hot,
                                         int kv, int dim, const float* lut) {
  if (slot < 0) return make_uint4(0, 0, 0, 0);
  if constexpr (HOT) {
    if (hot >= 0)
      return __ldg(reinterpret_cast<const uint4*>(
          a.hot + (int64_t(hot) * 2 + kv) * D + dim));
  }
  const int block = slot / a.page, off = slot % a.page;
  const int64_t element = ((int64_t(block) * 2 + kv) * a.page + off) * D + dim;
  if constexpr (FP8)
    return decode8(a.history + element, __ldg(a.scales + slot * 2 + kv), lut);
  else
    return __ldg(reinterpret_cast<const uint4*>(
        reinterpret_cast<const half*>(a.history) + element));
}

template <bool FP8, bool HOT>
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
      hots[t] = HOT && slot >= 0 ? a.resolved[row * a.rs0 + col] : -1;
    }
    __syncthreads();
    for (int v = t; v < C * 32; v += 256) {
      const int token = v / 32, dim = (v % 32) * 8;
      *reinterpret_cast<uint4*>(&kv[token][dim]) =
          load_kv<FP8, HOT>(a, slots[token], hots[token], 0, dim, lut);
    }
    __syncthreads();
    // Without the hot cache, issue this thread's V loads now so their
    // latency overlaps QK and the softmax; they are decoded into the
    // retired K tile.
    Raw raw[kChunks];
    float raw_scale[kChunks];
    if constexpr (!HOT) {
#pragma unroll
      for (int i = 0; i < kChunks; ++i) {
        const int v = t + i * 256, token = v / 32, dim = (v % 32) * 8;
        const int64_t slot = slots[token];
        raw[i] = Raw{};
        raw_scale[i] = 0.f;
        if (slot >= 0) {
          const int block = slot / a.page, off = slot % a.page;
          const int64_t element =
              ((int64_t(block) * 2 + 1) * a.page + off) * D + dim;
          if constexpr (FP8) {
            raw[i] = __ldg(reinterpret_cast<const uint2*>(a.history + element));
            raw_scale[i] = __ldg(a.scales + slot * 2 + 1);
          } else {
            raw[i] = __ldg(reinterpret_cast<const uint4*>(
                reinterpret_cast<const half*>(a.history) + element));
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
    if constexpr (HOT) {
      for (int v = t; v < C * 32; v += 256) {
        const int token = v / 32, dim = (v % 32) * 8;
        *reinterpret_cast<uint4*>(&kv[token][dim]) =
            load_kv<FP8, HOT>(a, slots[token], hots[token], 1, dim, lut);
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
  __syncthreads();
  auto emit = [&](int h, int dim, float sum) {
    float value = denom[h] > 0 ? __fdiv_rn(sum, denom[h]) : 0.f;
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

void device_qsa_out(torch::Tensor q, torch::Tensor history,
                    torch::Tensor scales, torch::Tensor indices,
                    torch::Tensor table, torch::Tensor requests,
                    torch::Tensor positions, torch::Tensor lengths,
                    torch::Tensor out, std::optional<torch::Tensor> gate,
                    torch::Tensor partial, torch::Tensor state,
                    std::optional<torch::Tensor> hot = std::nullopt,
                    std::optional<torch::Tensor> resolved = std::nullopt,
                    int64_t tiles = 1, bool prefill = false,
                    int64_t whole_rows = 0) {
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
  if (hot) {
    TORCH_CHECK(hot->is_cuda() && hot->device() == q.device() &&
                hot->scalar_type() == at::kHalf && hot->dim() == 3 &&
                hot->size(1) == 2 && hot->size(2) == 256 &&
                hot->is_contiguous());
    TORCH_CHECK(resolved->device() == q.device() &&
                resolved->scalar_type() == at::kInt && resolved->dim() == 2 &&
                resolved->size(0) >= q.size(0) &&
                resolved->size(1) >= indices.size(1) &&
                resolved->stride(1) == 1);
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
  a.resolved = resolved ? resolved->data_ptr<int>() : nullptr;
  a.rs0 = resolved ? resolved->stride(0) : 0;
  a.out = reinterpret_cast<half*>(out.data_ptr());
  a.partial = partial.data_ptr<float>();
  a.state = state.data_ptr<float>();
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
  if (prefill && fp8 && !hot && a.tiles == 1)
    device_qsa::partial_fp8_lite<<<grid, 256, 0, stream>>>(a);
  else if (fp8 && hot)
    device_qsa::partial<true, true><<<grid, 256, 0, stream>>>(a);
  else if (fp8)
    device_qsa::partial<true, false><<<grid, 256, 0, stream>>>(a);
  else if (hot)
    device_qsa::partial<false, true><<<grid, 256, 0, stream>>>(a);
  else
    device_qsa::partial<false, false><<<grid, 256, 0, stream>>>(a);
  device_qsa::merge<<<a.M * 6, 256, 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY(vllm_sm70_qsa_device, m) {
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
  // Host-authoritative history with a protected device hot cache: resolved
  // tokens read the hot copy, the rest read mapped host history directly.
  m.def(
      "run_hot(Tensor q, Tensor history, Tensor scales, Tensor indices, Tensor "
      "table, Tensor requests, Tensor positions, Tensor lengths, Tensor(a!) "
      "out, Tensor? gate, Tensor(b!) partial, Tensor(c!) state, Tensor hot, "
      "Tensor resolved) -> ()");
  m.impl("run_hot", torch::kCUDA,
         [](torch::Tensor q, torch::Tensor history, torch::Tensor scales,
            torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
            torch::Tensor positions, torch::Tensor lengths, torch::Tensor out,
            std::optional<torch::Tensor> gate, torch::Tensor partial,
            torch::Tensor state, torch::Tensor hot, torch::Tensor resolved) {
           device_qsa_out(q, history, scales, indices, table, requests,
                          positions, lengths, out, gate, partial, state, hot,
                          resolved);
         });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
