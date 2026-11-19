// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Reuse each compressed key across a request's speculative queries. One CTA
// row per request; each request's rows are contiguous (sorted token_to_req).
// Provides a standalone L2-resident histogram + single-launch multi-CTA exact
// lexicographic Top-512 path for SM70 long-context decode.
#include <algorithm>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <torch/library.h>
#include <torch/types.h>
#include <pybind11/pybind11.h>

namespace {

constexpr int kTopK = 512;
constexpr int kHistBins = 512;
constexpr int kSingleCTAMaxLen = 3584;
constexpr int kMaxTieCap = 8192;
constexpr int kTopKThreads = 256;

#define QSA_MMA(C, A0, A1, B0, B1)                                  \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
  if (value == 0.0f) value = 0.0f;
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Strictly monotone mapping from all 2^32 ordered FP32 keys into [0, 491].
// Dedicated bin 32 for 0.0f and 400 fine bins (6.25% width) across
// [2^-10, 2^15), covering all active positive QSA scores.
__device__ __forceinline__ uint32_t coarse_bin_from_ordered(uint32_t key) {
  if (key < 0x80000000u) return key >> 26;
  if (key == 0x80000000u) return 32u;
  if (key < 0xba800000u) return 33u + ((key - 0x80000000u) >> 25);
  if (key < 0xc7000000u) return 63u + ((key - 0xba800000u) >> 19);
  return 463u + ((key - 0xc7000000u) >> 25);
}

__device__ __forceinline__ uint32_t ldcg_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ int32_t ldcg_s32(const int32_t* p) {
  int32_t v;
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

template <typename Position>
__global__ void mqa_multiquery(
    const half* __restrict__ q, const half* __restrict__ cache,
    const int* __restrict__ table, const int* __restrict__ requests,
    const Position* __restrict__ positions, const int* __restrict__ lengths,
    float* __restrict__ scores, int* __restrict__ visible, int m, int columns,
    int pages, int page_size, int page_shift, int page_mask, int table_width,
    int sq0, int sq1, int sq2, int sc0, int sc1, int sc3, int st0, int st1,
    float divisor, int compress) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int req = blockIdx.y, nreq = gridDim.y;
  const long long length = lengths[req];

  __shared__ int s_first[2], s_count[2], s_end[2];
  __shared__ int s_row_vis[64];
  if (warp < 2) {
    const int row = threadIdx.x;
    int my_first = m, my_count = 0, my_end = 0;
    if (row < m) {
      const int owner = requests[row];
      if (owner != req) {
        s_row_vis[row] = 0;
        if (req == 0 && blockIdx.x == 0 && (owner < 0 || owner >= nreq))
          visible[row] = 0;
      } else {
        my_first = row;
        my_count = 1;
        const long long prefix = positions[row] + 1;
        const long long causal = prefix < length ? prefix : length;
        const int n = causal / compress;
        my_end = max(0, min(n, columns));
        s_row_vis[row] = my_end;
        if (blockIdx.x == 0) visible[row] = n;
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      my_first = min(my_first, __shfl_down_sync(0xffffffff, my_first, off));
      my_count += __shfl_down_sync(0xffffffff, my_count, off);
      my_end = max(my_end, __shfl_down_sync(0xffffffff, my_end, off));
    }
    if (lane == 0) {
      s_first[warp] = my_first;
      s_count[warp] = my_count;
      s_end[warp] = my_end;
    }
  }
  __syncthreads();
  const int first = min(s_first[0], s_first[1]);
  const int count = s_count[0] + s_count[1];
  const int end = max(s_end[0], s_end[1]);
  if (!count || blockIdx.x * 32 >= end) return;

  table += static_cast<int64_t>(req) * st0;
  const int qhead = quad * 8 + r;
  const bool fast_aligned =
      (sc3 == 1 && sq2 == 1 && sc0 % 8 == 0 && sc1 % 8 == 0 && sq0 % 8 == 0 &&
       sq1 % 8 == 0 &&
       (reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(cache)) %
               16 ==
           0);

  __shared__ __align__(16) half s_q[32][136];
  __shared__ __align__(16) half s_k[4][8][136];

  if (fast_aligned && count <= 8) {
    const int total_q_vec = count * 4 * 16;
    for (int idx = threadIdx.x; idx < total_q_vec; idx += 128) {
      const int qh_all = idx >> 4;
      const int v8 = (idx & 15) * 8;
      const int qr = qh_all >> 2;
      const int qh = qh_all & 3;
      const half* query =
          q + static_cast<int64_t>(first + qr) * sq0 + qh * sq1 + v8;
      *reinterpret_cast<uint4*>(&s_q[qh_all][v8]) =
          __ldg(reinterpret_cast<const uint4*>(query));
    }
    __syncthreads();
  }

  const int key_sub = lane >> 2;  // 0..7
  const int vec_sub = lane & 3;   // 0..3
  auto load_tile_keys = [&](int bx, uint4 kv_out[4]) {
    const int col_k = bx * 32 + warp * 8 + key_sub;
    const int lp =
        page_shift >= 0 ? (col_k >> page_shift) : (col_k / page_size);
    const int po = page_shift >= 0 ? (col_k & page_mask) : (col_k % page_size);
    const int page_k =
        (col_k < end && lp < table_width) ? __ldg(&table[lp * st1]) : -1;
    if (col_k < end && page_k >= 0 && page_k < pages) {
      const half* k_ptr = cache + static_cast<int64_t>(page_k) * sc0 + po * sc1;
#pragma unroll
      for (int p4 = 0; p4 < 4; ++p4) {
        kv_out[p4] = __ldg(
            reinterpret_cast<const uint4*>(k_ptr + (p4 * 4 + vec_sub) * 8));
      }
    } else {
#pragma unroll
      for (int p4 = 0; p4 < 4; ++p4) kv_out[p4] = make_uint4(0, 0, 0, 0);
    }
  };

  uint4 kv[4];
  if (fast_aligned) {
    load_tile_keys(blockIdx.x, kv);
  }

  for (int blk_x = blockIdx.x; blk_x * 32 < end; blk_x += gridDim.x) {
    const int column_base = blk_x * 32 + warp * 8;
    const int col = column_base + r;
    const int col_lp =
        page_shift >= 0 ? (col >> page_shift) : (col / page_size);
    const int col_po = page_shift >= 0 ? (col & page_mask) : (col % page_size);
    const int page =
        (col < end && col_lp < table_width) ? __ldg(&table[col_lp * st1]) : -1;
    const bool key_valid = col < end && page >= 0 && page < pages;

    uint4 kv_next[4];
    const int next_blk_x = blk_x + gridDim.x;
    if (fast_aligned) {
#pragma unroll
      for (int p4 = 0; p4 < 4; ++p4) {
        *reinterpret_cast<uint4*>(&s_k[warp][key_sub][(p4 * 4 + vec_sub) * 8]) =
            kv[p4];
      }
      __syncwarp();
      if (next_blk_x * 32 < end) {
        load_tile_keys(next_blk_x, kv_next);
      }
    }

    for (int t0 = 0; t0 < count; t0 += 8) {
      const int tm = min(8, count - t0), qrow = qhead / 4;
      const half* const qt = q + static_cast<int64_t>(first + t0) * sq0;
      float acc[8] = {};
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        uint4 a = {}, b = {}, lo = {}, hi = {};
        half* av = reinterpret_cast<half*>(&a);
        half* bv = reinterpret_cast<half*>(&b);
        half* lv = reinterpret_cast<half*>(&lo);
        half* hv = reinterpret_cast<half*>(&hi);
        if (fast_aligned) {
          a = *reinterpret_cast<const uint4*>(&s_k[warp][r][g * 16]);
          b = *reinterpret_cast<const uint4*>(&s_k[warp][r][g * 16 + 8]);
          if (count <= 8) {
            if (qrow < tm) {
              lo = *reinterpret_cast<const uint4*>(&s_q[qhead][g * 16]);
              hi = *reinterpret_cast<const uint4*>(&s_q[qhead][g * 16 + 8]);
            }
          } else if (qrow < tm) {
            const half* query = qt + qrow * sq0 + (qhead % 4) * sq1 + g * 16;
            lo = __ldg(reinterpret_cast<const uint4*>(query));
            hi = __ldg(reinterpret_cast<const uint4*>(query + 8));
          }
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            if (key_valid) {
              const half* k =
                  cache + static_cast<int64_t>(page) * sc0 + col_po * sc1;
              av[j] = k[(g * 16 + j) * sc3];
              bv[j] = k[(g * 16 + 8 + j) * sc3];
            }
            if (qrow < tm) {
              const half* query = qt + qrow * sq0 + (qhead % 4) * sq1;
              lv[j] = query[(g * 16 + j) * sq2];
              hv[j] = query[(g * 16 + 8 + j) * sq2];
            }
          }
        }
        QSA_MMA(acc, a.x, a.y, lo.x, lo.y);
        QSA_MMA(acc, a.z, a.w, lo.z, lo.w);
        QSA_MMA(acc, b.x, b.y, hi.x, hi.y);
        QSA_MMA(acc, b.z, b.w, hi.z, hi.w);
      }
#pragma unroll
      for (int p = 0; p < 2; ++p) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int i = p * 4 + k * 2;
          const float h0 = fmaxf(acc[i], 0.f), h1 = fmaxf(acc[i + 1], 0.f);
          const float h2 = __shfl_xor_sync(0xffffffff, h0, 2);
          const float h3 = __shfl_xor_sync(0xffffffff, h1, 2);
          const float sum = __fadd_rn(__fadd_rn(__fadd_rn(h0, h1), h2), h3);
          const int local = quad * 2 + p, row = first + t0 + local;
          const int column =
              column_base + k * 2 + ((lane & 16) ? 4 : 0) + (lane & 1);
          const int key_lane = (lane & ~3) | (k * 2 + (lane & 1));
          const bool valid =
              __shfl_sync(0xffffffff, static_cast<int>(key_valid), key_lane);
          if ((lane & 2) == 0 && local < tm && column < s_row_vis[row]) {
            scores[static_cast<int64_t>(row) * columns + column] =
                valid ? sum / divisor : -CUDART_INF_F;
          }
        }
      }
    }  // row tile

    if (fast_aligned && next_blk_x * 32 < end) {
      __syncwarp();
#pragma unroll
      for (int p4 = 0; p4 < 4; ++p4) kv[p4] = kv_next[p4];
    }
  }  // persistent column tile
}

#undef QSA_MMA

// Ultra-lean L2-resident coarse histogram pass: 2 KB smem/CTA, float4 loads,
// single non-zero flush per CTA. Skips rows with length <= 512 completely.
__global__ __launch_bounds__(kTopKThreads) void qsa_coarse_hist_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ visible,
    uint32_t* __restrict__ ws_hist, int columns) {
  const int part = blockIdx.x;
  const int parts = gridDim.x;
  const int row = blockIdx.y;
  const uint32_t tx = threadIdx.x;

  const int32_t raw_len = visible[row];
  const uint32_t length = raw_len > 0 ? min(static_cast<uint32_t>(raw_len),
                                            static_cast<uint32_t>(columns))
                                      : 0u;
  if (length <= static_cast<uint32_t>(kSingleCTAMaxLen)) return;

  __shared__ uint32_t s_hist[kHistBins];
  s_hist[tx] = 0u;
  s_hist[tx + kTopKThreads] = 0u;
  __syncthreads();

  const float* row_scores = scores + static_cast<int64_t>(row) * columns;
  const uint32_t chunk = ((length + static_cast<uint32_t>(parts) - 1u) /
                              static_cast<uint32_t>(parts) +
                          3u) &
                         ~3u;
  const uint32_t start = min(static_cast<uint32_t>(part) * chunk, length);
  const uint32_t end = min(start + chunk, length);

  if (start < end) {
    const bool aligned4 =
        ((reinterpret_cast<uintptr_t>(row_scores) & 15u) == 0u);
    if (aligned4) {
      const uint32_t vec_end = max(start, end & ~3u);
      for (uint32_t base = start + tx * 4u; base < vec_end;
           base += kTopKThreads * 4u) {
        const float4 v = *reinterpret_cast<const float4*>(row_scores + base);
        atomicAdd(&s_hist[coarse_bin_from_ordered(ordered_float_bits(v.x))],
                  1u);
        atomicAdd(&s_hist[coarse_bin_from_ordered(ordered_float_bits(v.y))],
                  1u);
        atomicAdd(&s_hist[coarse_bin_from_ordered(ordered_float_bits(v.z))],
                  1u);
        atomicAdd(&s_hist[coarse_bin_from_ordered(ordered_float_bits(v.w))],
                  1u);
      }
      const uint32_t tail_col = vec_end + tx;
      if (tail_col < end) {
        atomicAdd(&s_hist[coarse_bin_from_ordered(
                      ordered_float_bits(row_scores[tail_col]))],
                  1u);
      }
    } else {
      for (uint32_t col = start + tx; col < end; col += kTopKThreads) {
        atomicAdd(&s_hist[coarse_bin_from_ordered(
                      ordered_float_bits(row_scores[col]))],
                  1u);
      }
    }
  }
  __syncthreads();

  uint32_t* row_hist = ws_hist + static_cast<int64_t>(row) * kHistBins;
  const uint32_t c0 = s_hist[tx];
  const uint32_t c1 = s_hist[tx + kTopKThreads];
  if (parts == 1) {
    row_hist[tx] = c0;
    row_hist[tx + kTopKThreads] = c1;
  } else {
    if (c0) atomicAdd(&row_hist[tx], c0);
    if (c1) atomicAdd(&row_hist[tx + kTopKThreads], c1);
  }
}

struct MultiCTATopKShared {
  using ScanU32 = cub::BlockScan<uint32_t, kTopKThreads>;
  using ScanU64 = cub::BlockScan<uint64_t, kTopKThreads>;
  using SortU64 = cub::BlockRadixSort<uint64_t, kTopKThreads, 2>;
  using SortI32 = cub::BlockRadixSort<int32_t, kTopKThreads, 2>;

  union {
    typename ScanU32::TempStorage scan_u32;
    typename ScanU64::TempStorage scan_u64;
    typename SortU64::TempStorage sort_u64;
    typename SortI32::TempStorage sort_i32;
  } cub_tmp;

  uint32_t rad_hist[512];
  int32_t selected[kTopK];
  uint32_t thresh_bin;
  uint32_t greater_total;
  uint32_t thresh_count;
  uint32_t is_last;
  uint32_t prefix;
  uint32_t remaining;
  uint32_t sel_cursor;
  uint32_t eq_count;
  uint32_t idx_prefix;
  uint32_t idx_remaining;
  uint32_t greater_seen;
  uint32_t equal_seen;
  uint32_t chunk_greater_base;
  uint32_t chunk_equal_base;
};

__global__ __launch_bounds__(kTopKThreads) void qsa_multicta_topk_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ visible,
    int32_t* __restrict__ output, uint32_t* __restrict__ ws_hist,
    int32_t* __restrict__ ws_greater, uint32_t* __restrict__ ws_tie_keys,
    int32_t* __restrict__ ws_tie_idx, uint32_t* __restrict__ ws_meta,
    int columns) {
  const int part = blockIdx.x;
  const int parts = gridDim.x;
  const int row = blockIdx.y;
  const uint32_t tx = threadIdx.x;
  const int lane = tx & 31u;

  const int32_t raw_len = visible[row];
  const uint32_t length = raw_len > 0 ? min(static_cast<uint32_t>(raw_len),
                                            static_cast<uint32_t>(columns))
                                      : 0u;

  uint32_t* row_hist = ws_hist + static_cast<int64_t>(row) * kHistBins;
  uint32_t* row_meta = ws_meta + static_cast<int64_t>(row) * 8;
  int32_t* row_out = output + static_cast<int64_t>(row) * kTopK;

  if (length <= static_cast<uint32_t>(kTopK)) {
    if (part == 0) {
      for (uint32_t i = tx; i < static_cast<uint32_t>(kTopK);
           i += kTopKThreads) {
        row_out[i] = i < length ? static_cast<int32_t>(i) : -1;
      }
    }
    return;
  }

  const bool single_cta =
      (parts == 1) || (length <= static_cast<uint32_t>(kSingleCTAMaxLen));
  if (single_cta && part != 0) return;

  __shared__ MultiCTATopKShared s;

  const float* row_scores = scores + static_cast<int64_t>(row) * columns;

  // Step 1: Reverse-order suffix scan over 512 coarse bins (2 bins/thread).
  // For short rows (<= 3584, e.g. <= 14K context), build the 512-bin histogram
  // right here in shared memory with zero cross-CTA traffic or fences.
  uint32_t c0, c1;
  const uint32_t b0 = (kHistBins - 1u) - tx * 2u;
  const uint32_t b1 = b0 - 1u;
  if (length <= static_cast<uint32_t>(kSingleCTAMaxLen)) {
    s.rad_hist[tx] = 0u;
    s.rad_hist[tx + kTopKThreads] = 0u;
    __syncthreads();
    const bool aligned4 =
        ((reinterpret_cast<uintptr_t>(row_scores) & 15u) == 0u);
    if (aligned4) {
      const uint32_t vec_end = length & ~3u;
      for (uint32_t base = tx * 4u; base < vec_end; base += kTopKThreads * 4u) {
        const float4 v = *reinterpret_cast<const float4*>(row_scores + base);
        atomicAdd(&s.rad_hist[coarse_bin_from_ordered(ordered_float_bits(v.x))],
                  1u);
        atomicAdd(&s.rad_hist[coarse_bin_from_ordered(ordered_float_bits(v.y))],
                  1u);
        atomicAdd(&s.rad_hist[coarse_bin_from_ordered(ordered_float_bits(v.z))],
                  1u);
        atomicAdd(&s.rad_hist[coarse_bin_from_ordered(ordered_float_bits(v.w))],
                  1u);
      }
      if (vec_end + tx < length) {
        atomicAdd(&s.rad_hist[coarse_bin_from_ordered(
                      ordered_float_bits(row_scores[vec_end + tx]))],
                  1u);
      }
    } else {
      for (uint32_t col = tx; col < length; col += kTopKThreads) {
        atomicAdd(&s.rad_hist[coarse_bin_from_ordered(
                      ordered_float_bits(row_scores[col]))],
                  1u);
      }
    }
    __syncthreads();
    c0 = s.rad_hist[b0];
    c1 = s.rad_hist[b1];
  } else {
    c0 = row_hist[b0];
    c1 = row_hist[b1];
  }
  const uint32_t pair_sum = c0 + c1;
  uint32_t pair_prefix = 0u;
  MultiCTATopKShared::ScanU32(s.cub_tmp.scan_u32)
      .ExclusiveSum(pair_sum, pair_prefix);
  if (pair_prefix < static_cast<uint32_t>(kTopK) &&
      pair_prefix + pair_sum >= static_cast<uint32_t>(kTopK)) {
    if (pair_prefix + c0 >= static_cast<uint32_t>(kTopK)) {
      s.thresh_bin = b0;
      s.greater_total = pair_prefix;
      s.thresh_count = c0;
    } else {
      s.thresh_bin = b1;
      s.greater_total = pair_prefix + c0;
      s.thresh_count = c1;
    }
  }
  __syncthreads();

  const uint32_t thresh_bin = s.thresh_bin;
  const uint32_t greater_total = s.greater_total;
  const uint32_t thresh_count = s.thresh_count;
  const uint32_t remaining = static_cast<uint32_t>(kTopK) - greater_total;
  const bool collect_ties = (thresh_count <= static_cast<uint32_t>(kMaxTieCap));

  int32_t* row_greater = ws_greater + static_cast<int64_t>(row) * kTopK;
  uint32_t* row_tie_keys = ws_tie_keys + static_cast<int64_t>(row) * kMaxTieCap;
  int32_t* row_tie_idx = ws_tie_idx + static_cast<int64_t>(row) * kMaxTieCap;

  // Step 2: Parallel single-pass scan across active CTAs (float4 vectorized).
  const uint32_t eff_parts = single_cta ? 1u : static_cast<uint32_t>(parts);
  const uint32_t chunk = ((length + eff_parts - 1u) / eff_parts + 3u) & ~3u;
  const uint32_t start =
      single_cta ? 0u : min(static_cast<uint32_t>(part) * chunk, length);
  const uint32_t end = single_cta ? length : min(start + chunk, length);

  if (single_cta && tx == 0) {
    s.greater_seen = 0u;
    s.equal_seen = 0u;
  }
  if (single_cta) __syncthreads();

  auto emit_candidate = [&](bool is_gt, bool is_tie, uint32_t key,
                            uint32_t col) {
    const uint32_t mask_g = __ballot_sync(0xffffffff, is_gt);
    if (mask_g) {
      int base = 0;
      if (lane == 0) {
        base = static_cast<int>(
            single_cta ? atomicAdd(&s.greater_seen,
                                   static_cast<uint32_t>(__popc(mask_g)))
                       : atomicAdd(&row_meta[0],
                                   static_cast<uint32_t>(__popc(mask_g))));
      }
      base = __shfl_sync(0xffffffff, base, 0);
      if (is_gt) {
        const int pos = base + __popc(mask_g & ((1u << lane) - 1u));
        if (single_cta) {
          s.selected[pos] = static_cast<int32_t>(col);
        } else {
          row_greater[pos] = static_cast<int32_t>(col);
        }
      }
    }
    if (collect_ties) {
      const uint32_t mask_t = __ballot_sync(0xffffffff, is_tie);
      if (mask_t) {
        int base = 0;
        if (lane == 0) {
          base = static_cast<int>(
              single_cta ? atomicAdd(&s.equal_seen,
                                     static_cast<uint32_t>(__popc(mask_t)))
                         : atomicAdd(&row_meta[1],
                                     static_cast<uint32_t>(__popc(mask_t))));
        }
        base = __shfl_sync(0xffffffff, base, 0);
        if (is_tie) {
          const int pos = base + __popc(mask_t & ((1u << lane) - 1u));
          row_tie_keys[pos] = key;
          row_tie_idx[pos] = static_cast<int32_t>(col);
        }
      }
    }
  };

  if (start < end) {
    const bool aligned4 =
        ((reinterpret_cast<uintptr_t>(row_scores) & 15u) == 0u);
    if (aligned4) {
      const uint32_t vec_end = max(start, end & ~3u);
      for (uint32_t tile = start; tile < vec_end; tile += kTopKThreads * 4u) {
        const uint32_t base = tile + tx * 4u;
        const bool valid4 = (base < vec_end);
        float vals[4] = {0.f, 0.f, 0.f, 0.f};
        if (valid4) {
          const float4 v = *reinterpret_cast<const float4*>(row_scores + base);
          vals[0] = v.x;
          vals[1] = v.y;
          vals[2] = v.z;
          vals[3] = v.w;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t key = 0u, bin = 0u;
          if (valid4) {
            key = ordered_float_bits(vals[j]);
            bin = coarse_bin_from_ordered(key);
          }
          emit_candidate(valid4 && (bin > thresh_bin),
                         valid4 && (bin == thresh_bin), key, base + j);
        }
      }
      const uint32_t tail_col = vec_end + tx;
      const bool tail_valid = tail_col < end;
      uint32_t tail_key = 0u;
      uint32_t tail_bin = 0u;
      if (tail_valid) {
        tail_key = ordered_float_bits(row_scores[tail_col]);
        tail_bin = coarse_bin_from_ordered(tail_key);
      }
      if (__any_sync(0xffffffff, tail_valid)) {
        emit_candidate(tail_valid && tail_bin > thresh_bin,
                       tail_valid && tail_bin == thresh_bin, tail_key,
                       tail_col);
      }
    } else {
      for (uint32_t tile = start; tile < end; tile += kTopKThreads) {
        const uint32_t col = tile + tx;
        const bool valid = col < end;
        uint32_t key = 0u, bin = 0u;
        if (valid) {
          key = ordered_float_bits(row_scores[col]);
          bin = coarse_bin_from_ordered(key);
        }
        emit_candidate(valid && (bin > thresh_bin),
                       valid && (bin == thresh_bin), key, col);
      }
    }
  }

  // Step 3: Device-wide fence and last-CTA handoff (skipped in single_cta).
  __syncthreads();
  if (!single_cta) {
    if (tx == 0) {
      __threadfence();
      const uint32_t prev = atomicAdd(&row_meta[2], 1u);
      s.is_last = (prev == static_cast<uint32_t>(parts - 1)) ? 1u : 0u;
    }
    __syncthreads();
    if (!s.is_last) return;

    // Step 4: Load strictly-greater indices via L2-coherent loads.
    for (uint32_t i = tx; i < greater_total; i += kTopKThreads) {
      s.selected[i] = ldcg_s32(&row_greater[i]);
    }
    __syncthreads();
  }

  auto load_tie_k = [&](uint32_t i) -> uint32_t {
    return single_cta ? row_tie_keys[i] : ldcg_u32(&row_tie_keys[i]);
  };
  auto load_tie_i = [&](uint32_t i) -> int32_t {
    return single_cta ? row_tie_idx[i] : ldcg_s32(&row_tie_idx[i]);
  };

  const int idx_bits = 32 - __clz(length - 1u);

  if (collect_ties) {
    if (thresh_count == remaining) {
      for (uint32_t i = tx; i < thresh_count; i += kTopKThreads) {
        s.selected[greater_total + i] = load_tie_i(i);
      }
      __syncthreads();
    } else if (thresh_count <= static_cast<uint32_t>(kTopK)) {
      // Pack only the differing bits of key + (~idx) so CUB runs ~half as many
      // radix passes (e.g. 35 bits at 250K vs 64 bits).
      const int key_bits = (thresh_bin >= 63u && thresh_bin <= 462u)
                               ? 19
                               : (thresh_bin == 32u ? 1 : 26);
      const uint32_t key_mask = (1u << key_bits) - 1u;
      const uint32_t idx_mask =
          idx_bits == 32 ? 0xffffffffu : ((1u << idx_bits) - 1u);
      const int total_bits = min(64, key_bits + idx_bits + 1);
      uint64_t sort_keys[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint32_t pos = tx * 2u + j;
        if (pos < thresh_count) {
          const uint32_t k = load_tie_k(pos) & key_mask;
          const uint32_t inv_idx =
              (~static_cast<uint32_t>(load_tie_i(pos))) & idx_mask;
          // Bit (key_bits + idx_bits) = 1 guarantees all real items beat 0ULL.
          sort_keys[j] = (1ULL << (key_bits + idx_bits)) |
                         (static_cast<uint64_t>(k) << idx_bits) | inv_idx;
        } else {
          sort_keys[j] = 0ULL;
        }
      }
      MultiCTATopKShared::SortU64(s.cub_tmp.sort_u64)
          .SortDescending(sort_keys, 0, total_bits);
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint32_t rank = tx * 2u + j;
        if (rank < remaining) {
          const uint32_t inv_idx =
              static_cast<uint32_t>(sort_keys[j]) & idx_mask;
          const int32_t idx = static_cast<int32_t>((~inv_idx) & idx_mask);
          s.selected[greater_total + rank] = idx;
        }
      }
      __syncthreads();
    } else {
      if (tx == 0) {
        s.prefix = 0u;
        s.remaining = remaining;
      }
      __syncthreads();

#pragma unroll
      for (int pass = 0; pass < 4; ++pass) {
        s.rad_hist[tx] = 0u;
        __syncthreads();
        const int shift = 24 - pass * 8;
        const uint32_t prefix = s.prefix;
        const uint32_t prefix_mask =
            pass == 0 ? 0u : (~uint32_t{0} << (shift + 8));
        for (uint32_t i = tx; i < thresh_count; i += kTopKThreads) {
          const uint32_t k = load_tie_k(i);
          if ((k & prefix_mask) == prefix) {
            atomicAdd(&s.rad_hist[(k >> shift) & 0xffu], 1u);
          }
        }
        __syncthreads();
        if (tx == 0) {
          uint32_t rem = s.remaining;
          for (int bin = 255; bin >= 0; --bin) {
            const uint32_t c = s.rad_hist[bin];
            if (rem > c) {
              rem -= c;
            } else {
              s.prefix |= static_cast<uint32_t>(bin) << shift;
              s.remaining = rem;
              break;
            }
          }
        }
        __syncthreads();
      }

      const uint32_t pivot_key = s.prefix;
      const uint32_t rem_ties = s.remaining;
      if (tx == 0) {
        s.sel_cursor = greater_total;
        s.eq_count = 0u;
      }
      __syncthreads();

      for (uint32_t i = tx; i < thresh_count; i += kTopKThreads) {
        const uint32_t k = load_tie_k(i);
        if (k > pivot_key) {
          const uint32_t dst = atomicAdd(&s.sel_cursor, 1u);
          s.selected[dst] = load_tie_i(i);
        } else if (k == pivot_key) {
          atomicAdd(&s.eq_count, 1u);
        }
      }
      __syncthreads();

      if (s.eq_count == rem_ties) {
        for (uint32_t i = tx; i < thresh_count; i += kTopKThreads) {
          if (load_tie_k(i) == pivot_key) {
            const uint32_t dst = atomicAdd(&s.sel_cursor, 1u);
            s.selected[dst] = load_tie_i(i);
          }
        }
        __syncthreads();
      } else {
        if (tx == 0) {
          s.idx_prefix = 0u;
          s.idx_remaining = rem_ties;
        }
        __syncthreads();
        const int start_pass =
            length <= 65536u ? 2 : (length <= 16777216u ? 1 : 0);
        for (int pass = start_pass; pass < 4; ++pass) {
          s.rad_hist[tx] = 0u;
          __syncthreads();
          const int shift = 24 - pass * 8;
          const uint32_t iprefix = s.idx_prefix;
          const uint32_t imask =
              pass == start_pass ? 0u : (~uint32_t{0} << (shift + 8));
          for (uint32_t i = tx; i < thresh_count; i += kTopKThreads) {
            if (load_tie_k(i) == pivot_key) {
              const uint32_t idx = static_cast<uint32_t>(load_tie_i(i));
              if ((idx & imask) == iprefix) {
                atomicAdd(&s.rad_hist[(idx >> shift) & 0xffu], 1u);
              }
            }
          }
          __syncthreads();
          if (tx == 0) {
            uint32_t rem = s.idx_remaining;
            for (int bin = 0; bin < 256; ++bin) {
              const uint32_t c = s.rad_hist[bin];
              if (rem > c) {
                rem -= c;
              } else {
                s.idx_prefix |= static_cast<uint32_t>(bin) << shift;
                s.idx_remaining = rem;
                break;
              }
            }
          }
          __syncthreads();
        }
        const uint32_t pivot_idx = s.idx_prefix;
        for (uint32_t i = tx; i < thresh_count; i += kTopKThreads) {
          if (load_tie_k(i) == pivot_key) {
            const uint32_t idx = static_cast<uint32_t>(load_tie_i(i));
            if (idx <= pivot_idx) {
              const uint32_t dst = atomicAdd(&s.sel_cursor, 1u);
              s.selected[dst] = static_cast<int32_t>(idx);
            }
          }
        }
        __syncthreads();
      }
    }
  } else {
    if (tx == 0) {
      s.prefix = 0u;
      s.remaining = remaining;
    }
    __syncthreads();

#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
      s.rad_hist[tx] = 0u;
      __syncthreads();
      const int shift = 24 - pass * 8;
      const uint32_t prefix = s.prefix;
      const uint32_t prefix_mask =
          pass == 0 ? 0u : (~uint32_t{0} << (shift + 8));
      for (uint32_t col = tx; col < length; col += kTopKThreads) {
        const uint32_t key = ordered_float_bits(row_scores[col]);
        if (coarse_bin_from_ordered(key) == thresh_bin &&
            (key & prefix_mask) == prefix) {
          atomicAdd(&s.rad_hist[(key >> shift) & 0xffu], 1u);
        }
      }
      __syncthreads();
      if (tx == 0) {
        uint32_t rem = s.remaining;
        for (int bin = 255; bin >= 0; --bin) {
          const uint32_t c = s.rad_hist[bin];
          if (rem > c) {
            rem -= c;
          } else {
            s.prefix |= static_cast<uint32_t>(bin) << shift;
            s.remaining = rem;
            break;
          }
        }
      }
      __syncthreads();
    }

    const uint32_t pivot_key = s.prefix;
    const uint32_t rem_ties = s.remaining;
    if (tx == 0) {
      s.greater_seen = 0u;
      s.equal_seen = 0u;
    }
    __syncthreads();

    for (uint32_t base = 0; base < length; base += kTopKThreads) {
      const uint32_t col = base + tx;
      uint32_t gt = 0u, eq = 0u;
      if (col < length) {
        const uint32_t key = ordered_float_bits(row_scores[col]);
        if (coarse_bin_from_ordered(key) == thresh_bin) {
          gt = key > pivot_key ? 1u : 0u;
          eq = key == pivot_key ? 1u : 0u;
        }
      }
      const uint64_t counts = (static_cast<uint64_t>(gt) << 32) | eq;
      uint64_t prefix_counts = 0ULL;
      uint64_t aggregate_counts = 0ULL;
      MultiCTATopKShared::ScanU64(s.cub_tmp.scan_u64)
          .ExclusiveSum(counts, prefix_counts, aggregate_counts);
      __syncthreads();
      if (tx == 0) {
        s.chunk_greater_base = s.greater_seen;
        s.chunk_equal_base = s.equal_seen;
        s.greater_seen += static_cast<uint32_t>(aggregate_counts >> 32);
        s.equal_seen += static_cast<uint32_t>(aggregate_counts);
      }
      __syncthreads();
      const uint32_t gt_before =
          s.chunk_greater_base + static_cast<uint32_t>(prefix_counts >> 32);
      const uint32_t eq_before =
          s.chunk_equal_base + static_cast<uint32_t>(prefix_counts);
      if (gt || (eq && eq_before < rem_ties)) {
        const uint32_t offset =
            greater_total + gt_before + min(eq_before, rem_ties);
        s.selected[offset] = static_cast<int32_t>(col);
      }
      __syncthreads();
    }
  }

  // Step 5: Sort the 512 selected indices ascending over only the active bits.
  int32_t final_idx[2];
  final_idx[0] = s.selected[tx * 2u];
  final_idx[1] = s.selected[tx * 2u + 1u];
  __syncthreads();
  MultiCTATopKShared::SortI32(s.cub_tmp.sort_i32).Sort(final_idx, 0, idx_bits);
  row_out[tx * 2u] = final_idx[0];
  row_out[tx * 2u + 1u] = final_idx[1];

  // Step 6: Self-clean per-row histogram & counters when multi-CTA ran.
  if (length > static_cast<uint32_t>(kSingleCTAMaxLen)) {
    row_hist[tx] = 0u;
    row_hist[tx + 256u] = 0u;
    if (!single_cta && tx < 8u) row_meta[tx] = 0u;
  }
}

inline void compute_page_shift(int page_size, int& page_shift, int& page_mask) {
  if (page_size > 0 && (page_size & (page_size - 1)) == 0) {
    page_shift = __builtin_ctz(page_size);
    page_mask = page_size - 1;
  } else {
    page_shift = -1;
    page_mask = 0;
  }
}

template <typename KernelFunc>
inline void configure_max_smem_once(KernelFunc fn) {
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    configured = true;
  }
}

void validate_mqa_inputs(const torch::Tensor& scores,
                         const torch::Tensor& visible, const torch::Tensor& q,
                         const torch::Tensor& cache, const torch::Tensor& table,
                         const torch::Tensor& requests,
                         const torch::Tensor& positions,
                         const torch::Tensor& lengths, double divisor,
                         int64_t compress) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 3 &&
                  q.size(0) > 0 && q.size(0) <= 64 && q.size(1) == 4 &&
                  q.size(2) == 128 && table.dim() == 2 && table.size(0) >= 1 &&
                  table.size(0) <= 65535,
              "Shared-key MQA requires FP16 M1..64/H4/D128");
  const int m = q.size(0);
  TORCH_CHECK(cache.dim() == 4 && cache.size(2) == 1 && cache.size(3) == 128 &&
                  cache.scalar_type() == at::kHalf &&
                  scores.scalar_type() == at::kFloat &&
                  scores.is_contiguous() && scores.size(0) == m &&
                  divisor > 0 && compress > 0,
              "Invalid MQA cache, scores, or scale");
  for (const auto& t : {table, requests, lengths, visible})
    TORCH_CHECK(t.device() == q.device() && t.scalar_type() == at::kInt,
                "MQA metadata must use same-device int32");
  TORCH_CHECK(scores.device() == q.device() && cache.device() == q.device() &&
                  requests.numel() >= m && positions.numel() >= m &&
                  visible.numel() >= m && lengths.numel() >= table.size(0),
              "MQA tensor extent mismatch");
  TORCH_CHECK(positions.device() == q.device() && positions.is_contiguous() &&
                  (positions.scalar_type() == at::kLong ||
                   positions.scalar_type() == at::kInt),
              "Positions must use contiguous same-device int32/int64");
  TORCH_CHECK(requests.is_contiguous() && lengths.is_contiguous() &&
                  visible.is_contiguous(),
              "MQA row metadata must be contiguous");
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
}

void mqa_out(torch::Tensor scores, torch::Tensor visible, torch::Tensor q,
             torch::Tensor cache, torch::Tensor table, torch::Tensor requests,
             torch::Tensor positions, torch::Tensor lengths, double divisor,
             int64_t compress) {
  const c10::cuda::CUDAGuard guard(q.device());
  validate_mqa_inputs(scores, visible, q, cache, table, requests, positions,
                      lengths, divisor, compress);
  const int m = q.size(0), columns = scores.size(1);
  if (!columns) return;
  int page_shift = -1, page_mask = 0;
  compute_page_shift(cache.size(1), page_shift, page_mask);
  const int nreq = static_cast<int>(table.size(0));
  const int total_col_tiles = (columns + 31) / 32;
  const int max_col_ctas = std::max(16, 400 / nreq);
  const dim3 grid(std::min(total_col_tiles, max_col_ctas), nreq);
  if (positions.scalar_type() == at::kLong) {
    configure_max_smem_once(mqa_multiquery<int64_t>);
    mqa_multiquery<int64_t><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(q.data_ptr()),
        reinterpret_cast<const half*>(cache.data_ptr()), table.data_ptr<int>(),
        requests.data_ptr<int>(), positions.data_ptr<int64_t>(),
        lengths.data_ptr<int>(), scores.data_ptr<float>(),
        visible.data_ptr<int>(), m, columns, cache.size(0), cache.size(1),
        page_shift, page_mask, table.size(1), q.stride(0), q.stride(1),
        q.stride(2), cache.stride(0), cache.stride(1), cache.stride(3),
        table.stride(0), table.stride(1), divisor, compress);
  } else {
    configure_max_smem_once(mqa_multiquery<int32_t>);
    mqa_multiquery<int32_t><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(q.data_ptr()),
        reinterpret_cast<const half*>(cache.data_ptr()), table.data_ptr<int>(),
        requests.data_ptr<int>(), positions.data_ptr<int>(),
        lengths.data_ptr<int>(), scores.data_ptr<float>(),
        visible.data_ptr<int>(), m, columns, cache.size(0), cache.size(1),
        page_shift, page_mask, table.size(1), q.stride(0), q.stride(1),
        q.stride(2), cache.stride(0), cache.stride(1), cache.stride(3),
        table.stride(0), table.stride(1), divisor, compress);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mqa_topk_out(torch::Tensor output, torch::Tensor scores,
                  torch::Tensor visible, torch::Tensor ws_hist,
                  torch::Tensor ws_greater, torch::Tensor ws_tie_keys,
                  torch::Tensor ws_tie_idx, torch::Tensor ws_meta,
                  torch::Tensor q, torch::Tensor cache, torch::Tensor table,
                  torch::Tensor requests, torch::Tensor positions,
                  torch::Tensor lengths, double divisor, int64_t compress) {
  const c10::cuda::CUDAGuard guard(q.device());
  validate_mqa_inputs(scores, visible, q, cache, table, requests, positions,
                      lengths, divisor, compress);
  const int m = q.size(0), columns = scores.size(1);
  TORCH_CHECK(output.device() == q.device() && output.is_contiguous() &&
                  output.scalar_type() == at::kInt && output.dim() == 2 &&
                  output.size(0) == m && output.size(1) == kTopK,
              "QSA fused topk output must be contiguous int32 [M, 512]");
  TORCH_CHECK(
      ws_hist.device() == q.device() && ws_hist.is_contiguous() &&
          ws_hist.scalar_type() == at::kInt &&
          ws_hist.numel() >= m * kHistBins &&
          ws_greater.device() == q.device() && ws_greater.is_contiguous() &&
          ws_greater.scalar_type() == at::kInt &&
          ws_greater.numel() >= m * kTopK &&
          ws_tie_keys.device() == q.device() && ws_tie_keys.is_contiguous() &&
          ws_tie_keys.scalar_type() == at::kInt &&
          ws_tie_keys.numel() >= m * kMaxTieCap &&
          ws_tie_idx.device() == q.device() && ws_tie_idx.is_contiguous() &&
          ws_tie_idx.scalar_type() == at::kInt &&
          ws_tie_idx.numel() >= m * kMaxTieCap &&
          ws_meta.device() == q.device() && ws_meta.is_contiguous() &&
          ws_meta.scalar_type() == at::kInt && ws_meta.numel() >= m * 8,
      "QSA fused topk workspace mismatch");
  if (!columns) {
    output.fill_(-1);
    visible.zero_();
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  int page_shift = -1, page_mask = 0;
  compute_page_shift(cache.size(1), page_shift, page_mask);
  const int nreq = static_cast<int>(table.size(0));
  const int total_col_tiles = (columns + 31) / 32;
  const int max_col_ctas = std::max(16, 400 / nreq);
  const dim3 score_grid(std::min(total_col_tiles, max_col_ctas), nreq);

  if (positions.scalar_type() == at::kLong) {
    configure_max_smem_once(mqa_multiquery<int64_t>);
    mqa_multiquery<int64_t><<<score_grid, 128, 0, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr()),
        reinterpret_cast<const half*>(cache.data_ptr()), table.data_ptr<int>(),
        requests.data_ptr<int>(), positions.data_ptr<int64_t>(),
        lengths.data_ptr<int>(), scores.data_ptr<float>(),
        visible.data_ptr<int>(), m, columns, cache.size(0), cache.size(1),
        page_shift, page_mask, table.size(1), q.stride(0), q.stride(1),
        q.stride(2), cache.stride(0), cache.stride(1), cache.stride(3),
        table.stride(0), table.stride(1), divisor, compress);
  } else {
    configure_max_smem_once(mqa_multiquery<int32_t>);
    mqa_multiquery<int32_t><<<score_grid, 128, 0, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr()),
        reinterpret_cast<const half*>(cache.data_ptr()), table.data_ptr<int>(),
        requests.data_ptr<int>(), positions.data_ptr<int>(),
        lengths.data_ptr<int>(), scores.data_ptr<float>(),
        visible.data_ptr<int>(), m, columns, cache.size(0), cache.size(1),
        page_shift, page_mask, table.size(1), q.stride(0), q.stride(1),
        q.stride(2), cache.stride(0), cache.stride(1), cache.stride(3),
        table.stride(0), table.stride(1), divisor, compress);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  uint32_t* hist_ptr = reinterpret_cast<uint32_t*>(ws_hist.data_ptr<int32_t>());
  int32_t* greater_ptr = ws_greater.data_ptr<int32_t>();
  uint32_t* tie_keys_ptr =
      reinterpret_cast<uint32_t*>(ws_tie_keys.data_ptr<int32_t>());
  int32_t* tie_idx_ptr = ws_tie_idx.data_ptr<int32_t>();
  uint32_t* meta_ptr = reinterpret_cast<uint32_t*>(ws_meta.data_ptr<int32_t>());

  const int max_parts_by_cols = std::max(1, (columns + 2047) / 2048);
  const int target_parts = std::max(1, 160 / m);
  const int parts = std::min(max_parts_by_cols, std::min(target_parts, 32));
  const dim3 topk_grid(parts, m);

  qsa_coarse_hist_kernel<<<topk_grid, kTopKThreads, 0, stream>>>(
      scores.data_ptr<float>(), visible.data_ptr<int32_t>(), hist_ptr, columns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  qsa_multicta_topk_kernel<<<topk_grid, kTopKThreads, 0, stream>>>(
      scores.data_ptr<float>(), visible.data_ptr<int32_t>(),
      output.data_ptr<int32_t>(), hist_ptr, greater_ptr, tie_keys_ptr,
      tie_idx_ptr, meta_ptr, columns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY(vllm_sm70_qsa_indexer, m) {
  m.def(
      "run(Tensor(a!) scores, Tensor(b!) visible, Tensor q, Tensor cache, "
      "Tensor table, Tensor requests, Tensor positions, Tensor lengths, "
      "float divisor, int compress) -> ()");
  m.impl("run", torch::kCUDA, &mqa_out);
  m.def(
      "run_topk(Tensor(a!) output, Tensor(b!) scores, Tensor(c!) visible, "
      "Tensor(d!) ws_hist, Tensor(e!) ws_greater, Tensor(f!) ws_tie_keys, "
      "Tensor(g!) ws_tie_idx, Tensor(h!) ws_meta, Tensor q, Tensor cache, "
      "Tensor table, Tensor requests, Tensor positions, Tensor lengths, "
      "float divisor, int compress) -> ()");
  m.impl("run_topk", torch::kCUDA, &mqa_topk_out);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
