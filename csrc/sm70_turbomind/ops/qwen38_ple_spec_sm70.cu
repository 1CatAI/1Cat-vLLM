// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// MTP4 PLE convolution: rollback, four dilated taps, SiLU, state commit.
// One CTA row per speculative request; rows past the last request are zero.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
__device__ half as_half(half x) { return x; }
__device__ half as_half(float x) { return __float2half_rn(x); }
__device__ void store_half(half* ptr, half x) { *ptr = x; }
__device__ void store_half(float* ptr, half x) { *ptr = __half2float(x); }
template <class State>
__global__ void ple_conv(const half* x, State* state, const half* weight,
                         const int* ids, const int* starts, const int* accepted,
                         half* out, int rows, int hidden, int64_t s0,
                         int64_t s1, int64_t s2) {
  int c = blockIdx.x * 128 + threadIdx.x;
  if (c >= hidden) return;
  const int j = blockIdx.y, first = starts[j];
  const int count = max(0, min(5, min(starts[j + 1] - first, rows - first)));
  const int sid = ids[j], rollback = sid ? max(0, min(4, accepted[j] - 1)) : 0;
  x += static_cast<int64_t>(first) * hidden;
  half h[14];
#pragma unroll
  for (int t = 0; t < 9; ++t)
    h[t] = sid ? as_half(state[sid * s0 + c * s1 + (rollback + t) * s2])
               : __float2half_rn(0.f);
#pragma unroll
  for (int t = 0; t < 5; ++t)
    h[t + 9] = t < count ? x[t * hidden + c] : __float2half_rn(0.f);
  half w[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) w[k] = weight[c * 4 + k];
#pragma unroll
  for (int r = 0; r < 5; ++r) {
    float v = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
      v = __fmaf_rn(__half2float(h[r + k * 3]), __half2float(w[k]), v);
    v = __half2float(__float2half_rn(v));
    if (r < count)
      out[(first + r) * hidden + c] =
          __float2half_rn(v / __fadd_rn(1.f, expf(-v)));
  }
  if (j == gridDim.y - 1)
    for (int r = max(0, first + count); r < rows; ++r)
      out[r * hidden + c] = __float2half_rn(0.f);
  if (sid) {
#pragma unroll
    for (int t = 0; t < 13; ++t)
      if (t < 8 + count)
        store_half(state + sid * s0 + c * s1 + t * s2, h[t + 1]);
  }
}

template <class State>
__global__ void ple_conv_add(const half* x, State* state, const half* weight,
                             const int* ids, const int* starts,
                             const int* accepted, half* residual_inout,
                             int rows, int hidden, int64_t s0, int64_t s1,
                             int64_t s2) {
  int c = blockIdx.x * 128 + threadIdx.x;
  if (c >= hidden) return;
  const int j = blockIdx.y, first = starts[j];
  const int next_first = starts[j + 1];
  const int count = max(0, min(5, min(next_first - first, rows - first)));
  const int sid = ids[j], rollback = sid ? max(0, min(4, accepted[j] - 1)) : 0;
  if (j == 0) {
    for (int r = 0; r < min(first, rows); ++r) {
      residual_inout[r * hidden + c] =
          __hadd(residual_inout[r * hidden + c], __float2half_rn(0.f));
    }
  }
  x += static_cast<int64_t>(first) * hidden;
  half h[14];
#pragma unroll
  for (int t = 0; t < 9; ++t)
    h[t] = sid ? as_half(state[sid * s0 + c * s1 + (rollback + t) * s2])
               : __float2half_rn(0.f);
#pragma unroll
  for (int t = 0; t < 5; ++t)
    h[t + 9] = t < count ? x[t * hidden + c] : __float2half_rn(0.f);
  half w[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) w[k] = weight[c * 4 + k];
#pragma unroll
  for (int r = 0; r < 5; ++r) {
    float v = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
      v = __fmaf_rn(__half2float(h[r + k * 3]), __half2float(w[k]), v);
    v = __half2float(__float2half_rn(v));
    if (r < count) {
      const half conv_val = __float2half_rn(v / __fadd_rn(1.f, expf(-v)));
      const int rr = first + r;
      residual_inout[rr * hidden + c] =
          __hadd(residual_inout[rr * hidden + c], conv_val);
    }
  }
  const int seg_end = min(next_first, rows);
  for (int rr = max(0, first + count); rr < seg_end; ++rr) {
    residual_inout[rr * hidden + c] =
        __hadd(residual_inout[rr * hidden + c], __float2half_rn(0.f));
  }
  if (j == gridDim.y - 1) {
    for (int rr = max(0, seg_end); rr < rows; ++rr) {
      residual_inout[rr * hidden + c] =
          __hadd(residual_inout[rr * hidden + c], __float2half_rn(0.f));
    }
  }
  if (sid) {
#pragma unroll
    for (int t = 0; t < 13; ++t)
      if (t < 8 + count)
        store_half(state + sid * s0 + c * s1 + t * s2, h[t + 1]);
  }
}

constexpr int kPleLLRowWords = 12800;  // 10240 key + 2560 value
constexpr int kPlePageRows = 64;

struct PlePeer {
  uint32_t* ll[4];
};

__device__ __forceinline__ void ple_mma(float (&d)[8], uint32_t a0, uint32_t a1,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned ple_ld_vol(const unsigned* p) {
  unsigned v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ void ple_st_vol(unsigned* p, unsigned v) {
  asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(p), "r"(v));
}

__device__ __forceinline__ void ple_st_vol4(uint32_t* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

__device__ __forceinline__ uint4 ple_poll4(const uint32_t* p, uint32_t tag) {
  uint4 v;
  const long long t0 = clock64();
  do {
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    if (clock64() - t0 > 4000000000LL) __trap();
  } while ((v.x >> 16) != tag || (v.y >> 16) != tag || (v.z >> 16) != tag ||
           (v.w >> 16) != tag);
  return v;
}

__device__ __forceinline__ uint32_t ple_pk(half v, uint32_t tag) {
  return (tag << 16) | __half_as_ushort(v);
}

__device__ __forceinline__ uint32_t ple_tag_of(unsigned seq) {
  return seq % 65535u + 1u;
}

__device__ __forceinline__ int ple_gcol_base(int rk, int tile) {
  return tile < 80 ? (rk * 2560 + tile * 32)
                   : (10240 + rk * 640 + (tile - 80) * 32);
}

__device__ __forceinline__ half decode_ple_e4m3_byte(uint8_t raw, float scale) {
  const uint32_t sign = (raw >> 7) & 1u;
  const uint32_t exp = (raw >> 3) & 0x0fu;
  const uint32_t man = raw & 0x07u;
  if (exp == 0x0fu && man == 0x07u) {
    return __ushort_as_half(0x7e00u);
  }
  const float mag = (exp == 0u) ? (static_cast<float>(man) * 0.001953125f)
                                : (static_cast<float>(8u + man) *
                                   __uint_as_float((exp + 117u) << 23));
  const float signed_val = __uint_as_float(__float_as_uint(mag) | (sign << 31));
  return __float2half_rn(signed_val * scale);
}

template <bool IsFp8Bytes>
__device__ __forceinline__ void load_ple_emb_group(
    half (&emb_s)[8][1032], const void* __restrict__ emb_ptr, float scale,
    int first_row, int gm) {
  const int tid = threadIdx.x;  // 0..127; 128 * 8 = 1024 cols per row
  const int col8 = tid * 8;
  for (int rr = 0; rr < gm; ++rr) {
    if constexpr (IsFp8Bytes) {
      const auto* src = static_cast<const uint8_t*>(emb_ptr) +
                        static_cast<size_t>(first_row + rr) * 1024 + col8;
      const uint2 raw8 = *reinterpret_cast<const uint2*>(src);
      const auto* bytes = reinterpret_cast<const uint8_t*>(&raw8);
      half decoded[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        decoded[j] = decode_ple_e4m3_byte(bytes[j], scale);
      *reinterpret_cast<uint4*>(&emb_s[rr][col8]) =
          *reinterpret_cast<const uint4*>(decoded);
    } else {
      const auto* src = static_cast<const half*>(emb_ptr) +
                        static_cast<size_t>(first_row + rr) * 1024 + col8;
      *reinterpret_cast<uint4*>(&emb_s[rr][col8]) =
          *reinterpret_cast<const uint4*>(src);
    }
  }
}

template <bool IsFp8Bytes, bool ScaleIsHalf>
__global__ __launch_bounds__(128, 4) void ple_kv_proj_tp4_kernel(
    const void* __restrict__ emb_ptr, const void* __restrict__ scale_ptr,
    const half* __restrict__ packed_kv, unsigned* __restrict__ cnt,
    unsigned* __restrict__ dirty_m, PlePeer pr, unsigned* __restrict__ seq,
    int rk, int total_m) {
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int tile = blockIdx.x;  // 0..99
  const unsigned next_seq = ple_ld_vol(seq) + 1u;
  const unsigned page = next_seq & 1u;
  const size_t page_off =
      static_cast<size_t>(page) * kPlePageRows * kPleLLRowWords;
  for (int i = 0; i < 4; ++i) {
    if (pr.ll[i]) pr.ll[i] += page_off;
  }

  const unsigned dirty = ple_ld_vol(dirty_m + page);
  if (static_cast<int>(dirty) > total_m) {
    const int clear_rows = min(kPlePageRows, static_cast<int>(dirty)) - total_m;
    const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
    for (int idx = tid; idx < clear_rows * 32; idx += 128) {
      const int row = total_m + (idx >> 5);
      const int sub = idx & 31;
      const int src_rk = sub >> 3;
      const int c4 = (sub & 7) * 4;
      *reinterpret_cast<uint4*>(pr.ll[rk] +
                                static_cast<size_t>(row) * kPleLLRowWords +
                                ple_gcol_base(src_rk, tile) + c4) = zero4;
    }
  }

  float scale = 1.0f;
  if constexpr (IsFp8Bytes) {
    scale = ScaleIsHalf ? __half2float(*static_cast<const half*>(scale_ptr))
                        : *static_cast<const float*>(scale_ptr);
  }

  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int col = quad * 8 + r;
  const half* w = packed_kv + static_cast<size_t>(tile) * 1024 * 32;
  uint4 wlo[16], whi[16];
#pragma unroll
  for (int g = 0; g < 16; ++g) {
    const int kg = warp * 16 + g;
    wlo[g] = __ldg(reinterpret_cast<const uint4*>(w + (kg * 64 + col) * 8));
    whi[g] =
        __ldg(reinterpret_cast<const uint4*>(w + (kg * 64 + 32 + col) * 8));
  }

  __shared__ __align__(16) half emb_s[8][1032];
  __shared__ float red[4][8][32];
  __shared__ __align__(16) uint32_t tout[8][32];

  const int groups_m = (total_m + 7) / 8;
  const uint32_t tag = ple_tag_of(next_seq);
  const int p1 = rk ^ 1, p2 = rk ^ 2, p3 = rk ^ 3;
  const bool full_mesh = (pr.ll[p3] != nullptr);
  const int gbase = ple_gcol_base(rk, tile);

  for (int grp = 0; grp < groups_m; ++grp) {
    const int first_row = grp * 8;
    const int gm = min(8, total_m - first_row);
    if (grp > 0) __syncthreads();
    load_ple_emb_group<IsFp8Bytes>(emb_s, emb_ptr, scale, first_row, gm);
    __syncthreads();

    float acc[8] = {};
#pragma unroll
    for (int g = 0; g < 16; ++g) {
      const int k = (warp * 16 + g) * 16;
      uint4 a = make_uint4(0u, 0u, 0u, 0u), b = a;
      if (r < gm) {
        a = *reinterpret_cast<const uint4*>(&emb_s[r][k]);
        b = *reinterpret_cast<const uint4*>(&emb_s[r][k + 8]);
      }
      ple_mma(acc, a.x, a.y, wlo[g].x, wlo[g].y);
      ple_mma(acc, a.z, a.w, wlo[g].z, wlo[g].w);
      ple_mma(acc, b.x, b.y, whi[g].x, whi[g].y);
      ple_mma(acc, b.z, b.w, whi[g].z, whi[g].w);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) red[warp][i][lane] = acc[i];
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float sum = red[0][i][lane] + red[1][i][lane] + red[2][i][lane] +
                          red[3][i][lane];
        const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int nn =
            quad * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
        if (rr < gm) tout[rr][nn] = ple_pk(__float2half_rn(sum), tag);
      }
    }
    __syncthreads();
    if (tid < gm * 8) {
      const int rr = tid >> 3;
      const int c4 = (tid & 7) * 4;
      const uint4 pkt = *reinterpret_cast<const uint4*>(&tout[rr][c4]);
      const size_t dst_off =
          static_cast<size_t>(first_row + rr) * kPleLLRowWords + gbase + c4;
      ple_st_vol4(pr.ll[rk] + dst_off, pkt);
      ple_st_vol4(pr.ll[p1] + dst_off, pkt);
      ple_st_vol4(pr.ll[p2] + dst_off, pkt);
      if (full_mesh) ple_st_vol4(pr.ll[p3] + dst_off, pkt);
    }
  }

  if (!full_mesh) {
    const int p1_base = ple_gcol_base(p1, tile);
    for (int idx = tid; idx < total_m * 8; idx += 128) {
      const int row = idx >> 3;
      const int c4 = (idx & 7) * 4;
      const size_t off =
          static_cast<size_t>(row) * kPleLLRowWords + p1_base + c4;
      ple_st_vol4(pr.ll[p2] + off, ple_poll4(pr.ll[rk] + off, tag));
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0 && atomicAdd(cnt, 1u) == 99u) {
    *cnt = 0u;
    ple_st_vol(dirty_m + page, static_cast<unsigned>(total_m));
    ple_st_vol(seq, next_seq);
  }
}

template <bool IsFp8Bytes, bool ScaleIsHalf>
__global__ __launch_bounds__(128, 4) void ple_kv_proj_local_kernel(
    half* __restrict__ key_out, half* __restrict__ val_out,
    const void* __restrict__ emb_ptr, const void* __restrict__ scale_ptr,
    const half* __restrict__ key_w, const half* __restrict__ val_w,
    int total_m) {
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int tile = blockIdx.x;  // 0..319 key, 320..399 val
  const bool is_key = tile < 320;
  const int local_tile = is_key ? tile : (tile - 320);
  const int out_stride = is_key ? 10240 : 2560;
  half* out_base = is_key ? key_out : val_out;
  const half* w_base = is_key ? key_w : val_w;

  float scale = 1.0f;
  if constexpr (IsFp8Bytes) {
    scale = ScaleIsHalf ? __half2float(*static_cast<const half*>(scale_ptr))
                        : *static_cast<const float*>(scale_ptr);
  }

  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int col = local_tile * 32 + quad * 8 + r;
  const half* wrow = w_base + static_cast<size_t>(col) * 1024;
  uint4 wlo[16], whi[16];
#pragma unroll
  for (int g = 0; g < 16; ++g) {
    const int k = (warp * 16 + g) * 16;
    wlo[g] = __ldg(reinterpret_cast<const uint4*>(wrow + k));
    whi[g] = __ldg(reinterpret_cast<const uint4*>(wrow + k + 8));
  }

  __shared__ __align__(16) half emb_s[8][1032];
  __shared__ float red[4][8][32];

  const int groups_m = (total_m + 7) / 8;
  for (int grp = 0; grp < groups_m; ++grp) {
    const int first_row = grp * 8;
    const int gm = min(8, total_m - first_row);
    if (grp > 0) __syncthreads();
    load_ple_emb_group<IsFp8Bytes>(emb_s, emb_ptr, scale, first_row, gm);
    __syncthreads();

    float acc[8] = {};
#pragma unroll
    for (int g = 0; g < 16; ++g) {
      const int k = (warp * 16 + g) * 16;
      uint4 a = make_uint4(0u, 0u, 0u, 0u), b = a;
      if (r < gm) {
        a = *reinterpret_cast<const uint4*>(&emb_s[r][k]);
        b = *reinterpret_cast<const uint4*>(&emb_s[r][k + 8]);
      }
      ple_mma(acc, a.x, a.y, wlo[g].x, wlo[g].y);
      ple_mma(acc, a.z, a.w, wlo[g].z, wlo[g].w);
      ple_mma(acc, b.x, b.y, whi[g].x, whi[g].y);
      ple_mma(acc, b.z, b.w, whi[g].z, whi[g].w);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) red[warp][i][lane] = acc[i];
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float sum = red[0][i][lane] + red[1][i][lane] + red[2][i][lane] +
                          red[3][i][lane];
        const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int cc = local_tile * 32 + quad * 8 +
                       ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
        if (rr < gm) {
          out_base[static_cast<size_t>(first_row + rr) * out_stride + cc] =
              __float2half_rn(sum);
        }
      }
    }
  }
}

__device__ __forceinline__ float ple_warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

template <bool FromLL>
__global__ __launch_bounds__(320, 3) void ple_gate_norm_kernel(
    half* __restrict__ gated_value_out, half* __restrict__ normalized_out,
    const half* __restrict__ hidden_states, const half* __restrict__ key_in,
    const half* __restrict__ val_in, const uint32_t* __restrict__ ll_ple,
    const unsigned* __restrict__ ple_seq, const half* __restrict__ norm_key_w,
    const half* __restrict__ norm_query_w, const half* __restrict__ norm_conv_w,
    float eps) {
  const int b = blockIdx.x;    // 0..3
  const int row = blockIdx.y;  // 0..M-1
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int c0 = tid * 8;
  const size_t hb_off = (static_cast<size_t>(row) * 4 + b) * 2560 + c0;
  const size_t w_off = static_cast<size_t>(b) * 2560 + c0;

  const uint4 qv = *reinterpret_cast<const uint4*>(hidden_states + hb_off);
  const uint4 nkw_v = *reinterpret_cast<const uint4*>(norm_key_w + w_off);
  const uint4 nqw_v = *reinterpret_cast<const uint4*>(norm_query_w + w_off);
  const uint4 ncw_v = *reinterpret_cast<const uint4*>(norm_conv_w + w_off);

  half kh[8], vh[8];
  if constexpr (FromLL) {
    const unsigned dseq = ple_ld_vol(ple_seq);
    const uint32_t tag = ple_tag_of(dseq);
    const uint32_t* row_ll =
        ll_ple +
        static_cast<size_t>(dseq & 1u) * kPlePageRows * kPleLLRowWords +
        static_cast<size_t>(row) * kPleLLRowWords;
    const uint4 k0 = ple_poll4(row_ll + b * 2560 + c0, tag);
    const uint4 k1 = ple_poll4(row_ll + b * 2560 + c0 + 4, tag);
    const uint4 v0 = ple_poll4(row_ll + 10240 + c0, tag);
    const uint4 v1 = ple_poll4(row_ll + 10240 + c0 + 4, tag);
    kh[0] = __ushort_as_half(k0.x & 0xffffu);
    kh[1] = __ushort_as_half(k0.y & 0xffffu);
    kh[2] = __ushort_as_half(k0.z & 0xffffu);
    kh[3] = __ushort_as_half(k0.w & 0xffffu);
    kh[4] = __ushort_as_half(k1.x & 0xffffu);
    kh[5] = __ushort_as_half(k1.y & 0xffffu);
    kh[6] = __ushort_as_half(k1.z & 0xffffu);
    kh[7] = __ushort_as_half(k1.w & 0xffffu);
    vh[0] = __ushort_as_half(v0.x & 0xffffu);
    vh[1] = __ushort_as_half(v0.y & 0xffffu);
    vh[2] = __ushort_as_half(v0.z & 0xffffu);
    vh[3] = __ushort_as_half(v0.w & 0xffffu);
    vh[4] = __ushort_as_half(v1.x & 0xffffu);
    vh[5] = __ushort_as_half(v1.y & 0xffffu);
    vh[6] = __ushort_as_half(v1.z & 0xffffu);
    vh[7] = __ushort_as_half(v1.w & 0xffffu);
  } else {
    *reinterpret_cast<uint4*>(kh) =
        *reinterpret_cast<const uint4*>(key_in + hb_off);
    *reinterpret_cast<uint4*>(vh) = *reinterpret_cast<const uint4*>(
        val_in + static_cast<size_t>(row) * 2560 + c0);
  }

  const auto* qh = reinterpret_cast<const half*>(&qv);
  const auto* nkw = reinterpret_cast<const half*>(&nkw_v);
  const auto* nqw = reinterpret_cast<const half*>(&nqw_v);
  const auto* ncw = reinterpret_cast<const half*>(&ncw_v);

  float kf[8], qf[8];
  float ksq = 0.0f, qsq = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    kf[j] = __half2float(kh[j]);
    qf[j] = __half2float(qh[j]);
    ksq = __fmaf_rn(kf[j], kf[j], ksq);
    qsq = __fmaf_rn(qf[j], qf[j], qsq);
  }
  ksq = ple_warp_sum(ksq);
  qsq = ple_warp_sum(qsq);

  __shared__ float s_red[2][10];
  __shared__ float s_scalars[3];
  if (lane == 0) {
    s_red[0][warp] = ksq;
    s_red[1][warp] = qsq;
  }
  __syncthreads();
  if (tid == 0) {
    float tk = 0.0f, tq = 0.0f;
#pragma unroll
    for (int w = 0; w < 10; ++w) {
      tk += s_red[0][w];
      tq += s_red[1][w];
    }
    s_scalars[0] = rsqrtf(tk * (1.0f / 2560.0f) + eps);
    s_scalars[1] = rsqrtf(tq * (1.0f / 2560.0f) + eps);
  }
  __syncthreads();

  const float inv_k = s_scalars[0];
  const float inv_q = s_scalars[1];
  float dot = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const half kn =
        __float2half_rn((kf[j] * inv_k) * (1.0f + __half2float(nkw[j])));
    const half qn =
        __float2half_rn((qf[j] * inv_q) * (1.0f + __half2float(nqw[j])));
    dot += __half2float(__hmul(kn, qn));
  }
  dot = ple_warp_sum(dot);
  if (lane == 0) s_red[0][warp] = dot;
  __syncthreads();
  if (tid == 0) {
    float tdot = 0.0f;
#pragma unroll
    for (int w = 0; w < 10; ++w) tdot += s_red[0][w];
    const float g0 = __half2float(__float2half_rn(
        __half2float(__float2half_rn(tdot)) / 50.59644256269407f));
    const float sign_g =
        isnan(g0) ? g0 : (g0 > 0.0f ? 1.0f : (g0 < 0.0f ? -1.0f : 0.0f));
    const half min_clamp = __ushort_as_half(0x0011u);  // float16(1e-6)
    const float abs_clamped = fmaxf(fabsf(g0), __half2float(min_clamp));
    const float sqrt_g = __half2float(__float2half_rn(sqrtf(abs_clamped)));
    const float arg = __half2float(__float2half_rn(sign_g * sqrt_g));
    s_scalars[2] = __half2float(__float2half_rn(1.0f / (1.0f + expf(-arg))));
  }
  __syncthreads();

  const half gate_h = __float2half_rn(s_scalars[2]);
  half gv_h[8];
  float gv_f[8];
  float gvsq = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    gv_h[j] = __hmul(gate_h, vh[j]);
    gv_f[j] = __half2float(gv_h[j]);
    gvsq = __fmaf_rn(gv_f[j], gv_f[j], gvsq);
  }
  *reinterpret_cast<uint4*>(gated_value_out + hb_off) =
      *reinterpret_cast<const uint4*>(gv_h);

  gvsq = ple_warp_sum(gvsq);
  if (lane == 0) s_red[0][warp] = gvsq;
  __syncthreads();
  if (tid == 0) {
    float tgv = 0.0f;
#pragma unroll
    for (int w = 0; w < 10; ++w) tgv += s_red[0][w];
    s_scalars[0] = rsqrtf(tgv * (1.0f / 2560.0f) + eps);
  }
  __syncthreads();

  const float inv_gv = s_scalars[0];
  half norm_h[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    norm_h[j] =
        __float2half_rn((gv_f[j] * inv_gv) * (1.0f + __half2float(ncw[j])));
  }
  *reinterpret_cast<uint4*>(normalized_out + hb_off) =
      *reinterpret_cast<const uint4*>(norm_h);
}

void ple_spec_conv(torch::Tensor output, torch::Tensor state, torch::Tensor x,
                   torch::Tensor weight, torch::Tensor ids,
                   torch::Tensor starts, torch::Tensor accepted) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(1) == 10240 &&
                  x.size(0) >= 1 && x.size(0) <= 40,
              "SM70 PLE MTP4 requires M1..40 and H10240");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  for (const auto& t : {output, state, x, weight, ids, starts, accepted})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device(),
                "PLE MTP4 tensors must share a CUDA device");
  for (const auto& t : {output, x, weight})
    TORCH_CHECK(t.scalar_type() == at::kHalf && t.is_contiguous(),
                "PLE MTP4 input, weight and output must be contiguous FP16");
  for (const auto& t : {ids, starts, accepted})
    TORCH_CHECK(
        t.scalar_type() == at::kInt && t.dim() == 1 && t.is_contiguous(),
        "PLE MTP4 metadata must be contiguous int32 vectors");
  TORCH_CHECK(ids.numel() >= 1 && ids.numel() <= 8 &&
                  starts.numel() >= ids.numel() + 1 &&
                  accepted.numel() >= ids.numel(),
              "PLE MTP4 fast path requires 1..8 speculative requests");
  TORCH_CHECK(state.dim() == 3 && state.size(0) > 0 && state.size(1) == 10240 &&
                  state.size(2) == 13 &&
                  (state.scalar_type() == at::kHalf ||
                   state.scalar_type() == at::kFloat),
              "PLE MTP4 requires FP16/FP32 [states,10240,13] cache");
  TORCH_CHECK((state.stride(2) == 1 && state.stride(1) >= 13) ||
                  (state.stride(1) == 1 && state.stride(2) >= 10240),
              "Unsupported PLE cache layout");
  TORCH_CHECK(
      state.stride(0) >= 10239 * state.stride(1) + 12 * state.stride(2) + 1,
      "PLE cache rows must not overlap");
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({10240, 4}) &&
                  output.sizes() == x.sizes(),
              "Invalid PLE MTP4 weight/output geometry");
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(T)                                                            \
  ple_conv<T><<<dim3(80, ids.numel()), 128, 0, stream>>>(                    \
      reinterpret_cast<const half*>(x.data_ptr()),                           \
      reinterpret_cast<T*>(state.data_ptr()),                                \
      reinterpret_cast<const half*>(weight.data_ptr()), ids.data_ptr<int>(), \
      starts.data_ptr<int>(), accepted.data_ptr<int>(),                      \
      reinterpret_cast<half*>(output.data_ptr()), x.size(0), 10240,          \
      state.stride(0), state.stride(1), state.stride(2))
  if (state.scalar_type() == at::kHalf) {
    LAUNCH(half);
  } else {
    LAUNCH(float);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ple_spec_conv_add(torch::Tensor residual_inout, torch::Tensor state,
                       torch::Tensor x, torch::Tensor weight, torch::Tensor ids,
                       torch::Tensor starts, torch::Tensor accepted) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(1) == 10240 &&
                  x.size(0) >= 1 && x.size(0) <= 40,
              "SM70 PLE MTP4 add requires M1..40 and H10240");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  for (const auto& t :
       {residual_inout, state, x, weight, ids, starts, accepted})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device(),
                "PLE MTP4 add tensors must share a CUDA device");
  for (const auto& t : {residual_inout, x, weight})
    TORCH_CHECK(t.scalar_type() == at::kHalf && t.is_contiguous(),
                "PLE MTP4 add input, weight and residual must be FP16");
  for (const auto& t : {ids, starts, accepted})
    TORCH_CHECK(
        t.scalar_type() == at::kInt && t.dim() == 1 && t.is_contiguous(),
        "PLE MTP4 add metadata must be contiguous int32 vectors");
  TORCH_CHECK(ids.numel() >= 1 && ids.numel() <= 8 &&
                  starts.numel() >= ids.numel() + 1 &&
                  accepted.numel() >= ids.numel(),
              "PLE MTP4 add fast path requires 1..8 speculative requests");
  TORCH_CHECK(state.dim() == 3 && state.size(0) > 0 && state.size(1) == 10240 &&
                  state.size(2) == 13 &&
                  (state.scalar_type() == at::kHalf ||
                   state.scalar_type() == at::kFloat),
              "PLE MTP4 add requires FP16/FP32 [states,10240,13] cache");
  TORCH_CHECK((state.stride(2) == 1 && state.stride(1) >= 13) ||
                  (state.stride(1) == 1 && state.stride(2) >= 10240),
              "Unsupported PLE cache layout");
  TORCH_CHECK(
      state.stride(0) >= 10239 * state.stride(1) + 12 * state.stride(2) + 1,
      "PLE cache rows must not overlap");
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({10240, 4}) &&
                  residual_inout.sizes() == x.sizes(),
              "Invalid PLE MTP4 add weight/residual geometry");
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_ADD(T)                                                        \
  ple_conv_add<T><<<dim3(80, ids.numel()), 128, 0, stream>>>(                \
      reinterpret_cast<const half*>(x.data_ptr()),                           \
      reinterpret_cast<T*>(state.data_ptr()),                                \
      reinterpret_cast<const half*>(weight.data_ptr()), ids.data_ptr<int>(), \
      starts.data_ptr<int>(), accepted.data_ptr<int>(),                      \
      reinterpret_cast<half*>(residual_inout.data_ptr()), x.size(0), 10240,  \
      state.stride(0), state.stride(1), state.stride(2))
  if (state.scalar_type() == at::kHalf) {
    LAUNCH_ADD(half);
  } else {
    LAUNCH_ADD(float);
  }
#undef LAUNCH_ADD
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ple_kv_proj_tp4(torch::Tensor embeddings, torch::Tensor scale,
                     torch::Tensor packed_kv, torch::Tensor cnt,
                     torch::Tensor dirty_m, c10::ArrayRef<int64_t> ll_ptrs,
                     torch::Tensor seq, int64_t logical_rank) {
  TORCH_CHECK(embeddings.is_cuda() && embeddings.dim() == 2 &&
                  embeddings.size(0) >= 1 &&
                  embeddings.size(0) <= kPlePageRows &&
                  embeddings.size(1) == 1024 && embeddings.is_contiguous(),
              "PLE TP4 KV proj requires contiguous CUDA [1..64, 1024]");
  const bool is_bytes = embeddings.scalar_type() == at::kByte;
  TORCH_CHECK(is_bytes || embeddings.scalar_type() == at::kHalf,
              "PLE TP4 KV proj requires uint8 or float16 embeddings");
  const c10::cuda::CUDAGuard guard(embeddings.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(embeddings.data_ptr()) %
                      (is_bytes ? 8 : 16) ==
                  0,
              "PLE embeddings must be aligned");
  if (is_bytes) {
    TORCH_CHECK(scale.is_cuda() && scale.device() == embeddings.device() &&
                    scale.numel() == 1 &&
                    (scale.scalar_type() == at::kHalf ||
                     scale.scalar_type() == at::kFloat),
                "PLE uint8 embeddings require 1-element FP16/FP32 scale");
  }
  TORCH_CHECK(
      packed_kv.is_cuda() && packed_kv.device() == embeddings.device() &&
          packed_kv.scalar_type() == at::kHalf && packed_kv.is_contiguous() &&
          packed_kv.sizes() == at::IntArrayRef({100, 64, 2, 32, 8}) &&
          reinterpret_cast<uintptr_t>(packed_kv.data_ptr()) % 16 == 0,
      "Invalid packed_kv for PLE TP4 KV proj");
  for (const auto& t : {cnt, dirty_m, seq}) {
    TORCH_CHECK(t.is_cuda() && t.device() == embeddings.device() &&
                    t.scalar_type() == at::kInt && t.is_contiguous(),
                "PLE TP4 KV proj counters must be contiguous int32");
  }
  TORCH_CHECK(cnt.numel() >= 1 && dirty_m.numel() >= 2 && seq.numel() >= 1 &&
                  ll_ptrs.size() == 4 && logical_rank >= 0 && logical_rank < 4,
              "Invalid PLE TP4 LL arguments");
  PlePeer pr{};
  for (int i = 0; i < 4; ++i) {
    pr.ll[i] = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(ll_ptrs[i]));
  }
  const int rk = static_cast<int>(logical_rank);
  TORCH_CHECK(pr.ll[rk] != nullptr && pr.ll[rk ^ 1] != nullptr &&
                  pr.ll[rk ^ 2] != nullptr,
              "PLE TP4 LL requires local and 2 direct peer pointers");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const void* s_ptr = is_bytes ? scale.data_ptr() : nullptr;
  const int m = static_cast<int>(embeddings.size(0));
  const auto* kv_ptr = reinterpret_cast<const half*>(packed_kv.data_ptr());
  auto* cnt_ptr = reinterpret_cast<unsigned*>(cnt.data_ptr<int>());
  auto* dirty_ptr = reinterpret_cast<unsigned*>(dirty_m.data_ptr<int>());
  auto* seq_ptr = reinterpret_cast<unsigned*>(seq.data_ptr<int>());
  if (is_bytes) {
    if (scale.scalar_type() == at::kHalf) {
      ple_kv_proj_tp4_kernel<true, true>
          <<<100, 128, 0, stream>>>(embeddings.data_ptr(), s_ptr, kv_ptr,
                                    cnt_ptr, dirty_ptr, pr, seq_ptr, rk, m);
    } else {
      ple_kv_proj_tp4_kernel<true, false>
          <<<100, 128, 0, stream>>>(embeddings.data_ptr(), s_ptr, kv_ptr,
                                    cnt_ptr, dirty_ptr, pr, seq_ptr, rk, m);
    }
  } else {
    ple_kv_proj_tp4_kernel<false, false>
        <<<100, 128, 0, stream>>>(embeddings.data_ptr(), nullptr, kv_ptr,
                                  cnt_ptr, dirty_ptr, pr, seq_ptr, rk, m);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ple_kv_proj_local(torch::Tensor key_out, torch::Tensor val_out,
                       torch::Tensor embeddings, torch::Tensor scale,
                       torch::Tensor key_w, torch::Tensor val_w) {
  TORCH_CHECK(embeddings.is_cuda() && embeddings.dim() == 2 &&
                  embeddings.size(0) >= 1 && embeddings.size(0) <= 64 &&
                  embeddings.size(1) == 1024 && embeddings.is_contiguous(),
              "PLE local KV proj requires contiguous CUDA [1..64, 1024]");
  const bool is_bytes = embeddings.scalar_type() == at::kByte;
  TORCH_CHECK(is_bytes || embeddings.scalar_type() == at::kHalf,
              "PLE local KV proj requires uint8 or float16 embeddings");
  const c10::cuda::CUDAGuard guard(embeddings.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(embeddings.data_ptr()) %
                      (is_bytes ? 8 : 16) ==
                  0,
              "PLE embeddings must be aligned");
  if (is_bytes) {
    TORCH_CHECK(scale.is_cuda() && scale.device() == embeddings.device() &&
                    scale.numel() == 1 &&
                    (scale.scalar_type() == at::kHalf ||
                     scale.scalar_type() == at::kFloat),
                "PLE uint8 embeddings require 1-element FP16/FP32 scale");
  }
  const int m = static_cast<int>(embeddings.size(0));
  for (const auto& t : {key_out, val_out, key_w, val_w}) {
    TORCH_CHECK(t.is_cuda() && t.device() == embeddings.device() &&
                    t.scalar_type() == at::kHalf && t.is_contiguous() &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "PLE local KV proj tensors must be aligned FP16");
  }
  TORCH_CHECK(key_out.sizes() == at::IntArrayRef({m, 10240}) &&
                  val_out.sizes() == at::IntArrayRef({m, 2560}) &&
                  key_w.sizes() == at::IntArrayRef({10240, 1024}) &&
                  val_w.sizes() == at::IntArrayRef({2560, 1024}),
              "Invalid PLE local KV proj shapes");
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto* ko = reinterpret_cast<half*>(key_out.data_ptr());
  auto* vo = reinterpret_cast<half*>(val_out.data_ptr());
  const auto* kw = reinterpret_cast<const half*>(key_w.data_ptr());
  const auto* vw = reinterpret_cast<const half*>(val_w.data_ptr());
  if (is_bytes) {
    if (scale.scalar_type() == at::kHalf) {
      ple_kv_proj_local_kernel<true, true><<<400, 128, 0, stream>>>(
          ko, vo, embeddings.data_ptr(), scale.data_ptr(), kw, vw, m);
    } else {
      ple_kv_proj_local_kernel<true, false><<<400, 128, 0, stream>>>(
          ko, vo, embeddings.data_ptr(), scale.data_ptr(), kw, vw, m);
    }
  } else {
    ple_kv_proj_local_kernel<false, false><<<400, 128, 0, stream>>>(
        ko, vo, embeddings.data_ptr(), nullptr, kw, vw, m);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ple_gate_norm(torch::Tensor gated_value_out, torch::Tensor normalized_out,
                   torch::Tensor hidden_states, torch::Tensor key,
                   torch::Tensor value, int64_t ll_ple_ptr,
                   torch::Tensor ple_seq, torch::Tensor norm_key_w,
                   torch::Tensor norm_query_w, torch::Tensor norm_conv_w,
                   double eps) {
  TORCH_CHECK(hidden_states.is_cuda() && hidden_states.dim() == 2 &&
                  hidden_states.size(0) >= 1 &&
                  hidden_states.size(0) <= 32768 &&
                  hidden_states.size(1) == 10240,
              "PLE gate+norm requires [1..32768, 10240] hidden_states");
  const c10::cuda::CUDAGuard guard(hidden_states.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  for (const auto& t : {gated_value_out, normalized_out, hidden_states,
                        norm_key_w, norm_query_w, norm_conv_w}) {
    TORCH_CHECK(t.is_cuda() && t.device() == hidden_states.device() &&
                    t.scalar_type() == at::kHalf && t.is_contiguous() &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "PLE gate+norm requires aligned contiguous FP16 tensors");
  }
  const int64_t m = hidden_states.size(0);
  TORCH_CHECK(gated_value_out.sizes() == at::IntArrayRef({m, 10240}) &&
                  normalized_out.sizes() == at::IntArrayRef({m, 10240}) &&
                  norm_key_w.sizes() == at::IntArrayRef({10240}) &&
                  norm_query_w.sizes() == at::IntArrayRef({10240}) &&
                  norm_conv_w.sizes() == at::IntArrayRef({10240}),
              "Invalid PLE gate+norm shapes");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(4, static_cast<unsigned>(m));
  if (ll_ple_ptr != 0) {
    TORCH_CHECK(m <= kPlePageRows && ple_seq.is_cuda() &&
                    ple_seq.device() == hidden_states.device() &&
                    ple_seq.scalar_type() == at::kInt && ple_seq.numel() >= 1,
                "Invalid PLE gate+norm LL sequence tensor");
    ple_gate_norm_kernel<true><<<grid, 320, 0, stream>>>(
        reinterpret_cast<half*>(gated_value_out.data_ptr()),
        reinterpret_cast<half*>(normalized_out.data_ptr()),
        reinterpret_cast<const half*>(hidden_states.data_ptr()), nullptr,
        nullptr,
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(ll_ple_ptr)),
        reinterpret_cast<const unsigned*>(ple_seq.data_ptr<int>()),
        reinterpret_cast<const half*>(norm_key_w.data_ptr()),
        reinterpret_cast<const half*>(norm_query_w.data_ptr()),
        reinterpret_cast<const half*>(norm_conv_w.data_ptr()),
        static_cast<float>(eps));
  } else {
    for (const auto& t : {key, value}) {
      TORCH_CHECK(t.is_cuda() && t.device() == hidden_states.device() &&
                      t.scalar_type() == at::kHalf && t.is_contiguous() &&
                      reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                  "PLE gate+norm local key/value must be aligned FP16");
    }
    TORCH_CHECK(key.sizes() == at::IntArrayRef({m, 10240}) &&
                    value.sizes() == at::IntArrayRef({m, 2560}),
                "Invalid PLE gate+norm local key/value shapes");
    ple_gate_norm_kernel<false><<<grid, 320, 0, stream>>>(
        reinterpret_cast<half*>(gated_value_out.data_ptr()),
        reinterpret_cast<half*>(normalized_out.data_ptr()),
        reinterpret_cast<const half*>(hidden_states.data_ptr()),
        reinterpret_cast<const half*>(key.data_ptr()),
        reinterpret_cast<const half*>(value.data_ptr()), nullptr, nullptr,
        reinterpret_cast<const half*>(norm_key_w.data_ptr()),
        reinterpret_cast<const half*>(norm_query_w.data_ptr()),
        reinterpret_cast<const half*>(norm_conv_w.data_ptr()),
        static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "qwen38_ple_spec_sm70_out(Tensor(a!) output, Tensor(b!) state, Tensor x, "
      "Tensor weight, Tensor ids, Tensor starts, Tensor accepted) -> ()");
  m.def(
      "qwen38_ple_spec_add_sm70_out(Tensor(a!) residual_inout, Tensor(b!) "
      "state, Tensor x, Tensor weight, Tensor ids, Tensor starts, Tensor "
      "accepted) -> ()");
  m.def(
      "qwen38_ple_kv_proj_tp4_sm70_out(Tensor embeddings, Tensor scale, "
      "Tensor packed_kv, Tensor(a!) cnt, Tensor(b!) dirty_m, int[] ll_ptrs, "
      "Tensor(c!) seq, int logical_rank) -> ()");
  m.def(
      "qwen38_ple_kv_proj_local_sm70_out(Tensor(a!) key_out, Tensor(b!) "
      "val_out, Tensor embeddings, Tensor scale, Tensor key_w, Tensor val_w) "
      "-> ()");
  m.def(
      "qwen38_ple_gate_norm_sm70_out(Tensor(a!) gated_value_out, Tensor(b!) "
      "normalized_out, Tensor hidden_states, Tensor key, Tensor value, int "
      "ll_ple_ptr, Tensor ple_seq, Tensor norm_key_w, Tensor norm_query_w, "
      "Tensor norm_conv_w, float eps) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen38_ple_spec_sm70_out", &ple_spec_conv);
  m.impl("qwen38_ple_spec_add_sm70_out", &ple_spec_conv_add);
  m.impl("qwen38_ple_kv_proj_tp4_sm70_out", &ple_kv_proj_tp4);
  m.impl("qwen38_ple_kv_proj_local_sm70_out", &ple_kv_proj_local);
  m.impl("qwen38_ple_gate_norm_sm70_out", &ple_gate_norm);
}
