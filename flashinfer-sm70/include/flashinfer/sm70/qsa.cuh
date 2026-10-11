// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from 1Cat-vLLM qsa_indexer_sm70.cu at be3722bc095b9792bf9833068595f0b1ef70ad37.
// Group speculative queries by request on device, then reuse each key tile in
// Volta m8n8k4 MMA across up to eight rows (four indexer heads per row).
#pragma once
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace flashinfer::sm70 {
template <typename Position>
__global__ void group_qsa_rows(const int* requests, const Position* positions, const int* lengths,
                               int* rows, int* counts, int* visible, int m, int row_capacity,
                               int request_count, int compress) {
  const int req = blockIdx.x, row = threadIdx.x;
  if (row >= m) return;
  const int request = requests[row];
  if (req == 0) {
    const long long length = request >= 0 && request < request_count ? lengths[request] : 0;
    const long long prefix = static_cast<long long>(positions[row]) + 1;
    visible[row] = static_cast<int>((prefix < length ? prefix : length) / compress);
  }
  // The row ordering is stable, independent of scheduling and cache addresses.
  if (request == req) {
    int rank = 0;
    for (int previous = 0; previous < row; ++previous) rank += requests[previous] == req;
    rows[req * row_capacity + rank] = row;
  }
  if (row == 0) {
    int count = 0;
    for (int i = 0; i < m; ++i) count += requests[i] == req;
    counts[req] = count;
  }
}

#define QSA_MMA(C, A0, A1, B0, B1)                                                          \
  asm volatile(                                                                             \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "                                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                                        \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), "+f"(C[5]), "+f"(C[6]), \
        "+f"(C[7])                                                                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <typename Position, bool Grouped>
__global__ void mqa_multiquery(const half* q, const half* cache, const int* table,
                               const int* grouped_rows, const int* counts, const int* requests,
                               int* visible, int num_rows, const Position* positions,
                               const int* lengths, float* scores, int row_capacity, int columns,
                               int64_t score_stride, int pages, int page_size, int table_width,
                               int64_t sq0, int64_t sq1, int64_t sq2, int64_t sc0, int64_t sc1,
                               int64_t sc3, int64_t st0, int64_t st1, float divisor, int compress) {
  const int groups_per_request = (row_capacity + 7) / 8;
  const int request = Grouped ? blockIdx.y / groups_per_request : 0;
  const int group_start = Grouped ? (blockIdx.y % groups_per_request) * 8 : 0;
  const int m = Grouped ? min(8, counts[request] - group_start) : num_rows;
  if (m <= 0) return;
  const int* rows = grouped_rows + request * row_capacity + group_start;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int column_base = blockIdx.x * 32 + warp * 8;
  const int col = column_base + r;
  int end = 0;
  for (int row = 0; row < m; ++row) {
    const int logical_row = Grouped ? rows[row] : row;
    const long long prefix = static_cast<long long>(positions[logical_row]) + 1;
    const long long length = Grouped || requests[row] == 0 ? lengths[request] : 0;
    const long long causal = prefix < length ? prefix : length;
    const int n = static_cast<int>(causal / compress);
    if (!Grouped && blockIdx.x == 0 && threadIdx.x == row) visible[row] = n;
    end = max(end, min(n, columns));
  }
  if (blockIdx.x * 32 >= end) return;
  int page = col < end && col / page_size < table_width
                 ? table[static_cast<int64_t>(request) * st0 + (col / page_size) * st1]
                 : -1;
  const bool key_valid = col < end && page >= 0 && page < pages;
  const int qhead = quad * 8 + r, qrow = qhead / 4;
  float acc[8] = {};
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    // General strides include transposed paged storage and sliced Q views.
    uint4 a = {}, b = {}, lo = {}, hi = {};
    half* av = reinterpret_cast<half*>(&a);
    half* bv = reinterpret_cast<half*>(&b);
    half* lv = reinterpret_cast<half*>(&lo);
    half* hv = reinterpret_cast<half*>(&hi);
    if (sc3 == 1 && sq2 == 1 && sc0 % 8 == 0 && sc1 % 8 == 0 && sq0 % 8 == 0 && sq1 % 8 == 0 &&
        (reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(cache)) % 16 == 0) {
      if (key_valid) {
        const half* k = cache + static_cast<int64_t>(page) * sc0 + (col % page_size) * sc1 + g * 16;
        a = *reinterpret_cast<const uint4*>(k);
        b = *reinterpret_cast<const uint4*>(k + 8);
      }
      if (qrow < m) {
        const half* query = q + (Grouped ? rows[qrow] : qrow) * sq0 + (qhead % 4) * sq1 + g * 16;
        lo = *reinterpret_cast<const uint4*>(query);
        hi = *reinterpret_cast<const uint4*>(query + 8);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (key_valid) {
          const half* k = cache + static_cast<int64_t>(page) * sc0 + (col % page_size) * sc1;
          av[j] = k[(g * 16 + j) * sc3];
          bv[j] = k[(g * 16 + 8 + j) * sc3];
        }
        if (qrow < m) {
          const half* query = q + (Grouped ? rows[qrow] : qrow) * sq0 + (qhead % 4) * sq1;
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
      const int row = quad * 2 + p;
      const int column = column_base + k * 2 + ((lane & 16) ? 4 : 0) + (lane & 1);
      // Keys for this output row reside in the lane with r==key-column.
      const int key_lane = (lane & ~3) | (k * 2 + (lane & 1));
      const bool valid = __shfl_sync(0xffffffff, static_cast<int>(key_valid), key_lane);
      if ((lane & 2) == 0 && row < m && (Grouped || requests[row] == 0) && column < columns &&
          column < (static_cast<long long>(positions[Grouped ? rows[row] : row]) + 1) / compress &&
          column < lengths[request] / compress) {
        scores[static_cast<int64_t>(Grouped ? rows[row] : row) * score_stride + column] =
            valid ? sum / divisor : -CUDART_INF_F;
      }
    }
  }
}

#undef QSA_MMA

}  // namespace flashinfer::sm70
