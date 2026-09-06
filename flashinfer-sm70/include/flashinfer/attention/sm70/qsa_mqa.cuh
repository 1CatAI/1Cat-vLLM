// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Schedule algorithm adapted from FlashInfer (Apache-2.0), copyright 2025
// FlashInfer team, and TensorRT-LLM, copyright 2026 NVIDIA CORPORATION.
// Pinned source: 6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f,
// flashinfer/attn_scores/kernels/schedule_kernel.py.
// Native Volta implementation: no CuTe, TMA, FP8 MMA, host length readback,
// global spin barrier, or request-capacity-sized launch grid.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

#include <cstdint>

namespace flashinfer::attention::sm70 {

constexpr int kMQATile = 64;
constexpr int kMQAMaxRows = 64;

struct MQAParams {
  const half* q;
  const half* k;
  const int32_t* table;
  const int32_t* requests;
  const void* positions;
  const int32_t* lengths;
  int32_t* visible;
  int32_t* schedule;
  float* logits;
  int32_t* task_visits;
  int rows, heads, columns, pages, page_size, table_width, num_requests;
  int ratio, workers;
  bool positions64;
  float divisor;
  int64_t q_row, q_head, k_page, k_token, table_row, out_row;
};

// Recomputed on every call/replay into instance-owned storage. In particular,
// empty rows produce duplicate prefixes: upper_bound (<=), not lower_bound.
__global__ void PlanMQA(MQAParams p) {
  __shared__ int prefix[kMQAMaxRows];
  const int lane = threadIdx.x;
  int carry = 0;
  for (int base = 0; base < kMQAMaxRows; base += 32) {
    const int row = base + lane;
    int visible = 0;
    if (row < p.rows) {
      const int request = p.requests[row];
      const int64_t pos = p.positions64
                              ? static_cast<const int64_t*>(p.positions)[row]
                              : static_cast<const int32_t*>(p.positions)[row];
      if (request >= 0 && request < p.num_requests && pos >= 0) {
        // floor(min(pos+1, len)/ratio) equals the minimum of both floors.
        // Clamp before adding one: INT64_MAX positions cannot overflow, and
        // the bounded covered-token count needs only a 32-bit division.
        const int length = max(0, p.lengths[request]);
        const int covered = int(min(pos, int64_t(length) - 1) + 1);
        const int64_t capacity = int64_t(p.page_size) * p.table_width;
        visible = int(
            min(int64_t(covered / p.ratio), min(int64_t(p.columns), capacity)));
      }
      p.visible[row] = visible;
    }
    int count = (visible + kMQATile - 1) / kMQATile;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
      const int other = __shfl_up_sync(0xffffffff, count, offset);
      if (lane >= offset) count += other;
    }
    count += carry;
    prefix[row] = count;
    carry = __shfl_sync(0xffffffff, count, 31);
  }
  __syncwarp();
  const int per_worker = carry / p.workers;
  const int extra = carry % p.workers;
  for (int worker = lane; worker <= p.workers; worker += 32) {
    const int start = worker * per_worker + min(worker, extra);
    int lo = 0, hi = p.rows;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (prefix[mid] <= start)
        lo = mid + 1;
      else
        hi = mid;
    }
    p.schedule[2 * worker] = lo;
    p.schedule[2 * worker + 1] = start - (lo ? prefix[lo - 1] : 0);
  }
}

// QSA is sum_h relu(dot(K, Q_h)) / divisor, not softmax attention.
// FP16 operands, FP32 tensor-core accumulation and a padded 16-head reduction.
// One CTA consumes a balanced contiguous range of *live* (row, KV tile) tasks.
template <int Dim, bool Audit = false>
__global__ void ScoreMQA(MQAParams p) {
  using namespace nvcuda;
  // Eight-half skew keeps WMMA's shared-memory rows out of the same banks.
  constexpr int LD = Dim + 8;
  __shared__ __align__(32) half query[16 * LD];
  __shared__ __align__(32) half keys[kMQATile * LD];
  __shared__ __align__(32) float scores[kMQATile * 16];
  __shared__ int64_t key_offsets[kMQATile];
  const int tid = threadIdx.x, warp = tid / 32;
  int row = p.schedule[2 * blockIdx.x];
  int tile = p.schedule[2 * blockIdx.x + 1];
  const int end_row = p.schedule[2 * (blockIdx.x + 1)];
  const int end_tile = p.schedule[2 * (blockIdx.x + 1) + 1];
  while (row < p.rows &&
         (row < end_row || (row == end_row && tile < end_tile))) {
    const int visible = p.visible[row];
    const int tile_end =
        row == end_row ? end_tile : (visible + kMQATile - 1) / kMQATile;
    if (tile < tile_end) {
      for (int i = tid; i < 16 * (Dim / 8); i += 128) {
        const int h = i / (Dim / 8), d = (i % (Dim / 8)) * 8;
        const int4 value =
            h < p.heads ? *reinterpret_cast<const int4*>(
                              p.q + int64_t(row) * p.q_row + h * p.q_head + d)
                        : make_int4(0, 0, 0, 0);
        *reinterpret_cast<int4*>(query + h * LD + d) = value;
      }
      const int request = p.requests[row];
      __syncthreads();
      for (; tile < tile_end; ++tile) {
        if constexpr (Audit) {
          if (tid == 0)
            atomicAdd(p.task_visits +
                          row * ((p.columns + kMQATile - 1) / kMQATile) + tile,
                      1);
        }
        if (tid < kMQATile) {
          const int col = tile * kMQATile + tid;
          int physical = -1;
          if (col < visible)
            physical =
                p.table[int64_t(request) * p.table_row + col / p.page_size];
          key_offsets[tid] = physical >= 0 && physical < p.pages
                                 ? int64_t(physical) * p.k_page +
                                       (col % p.page_size) * p.k_token
                                 : -1;
        }
        __syncthreads();
        for (int i = tid; i < kMQATile * (Dim / 8); i += 128) {
          const int local_col = i / (Dim / 8), d = (i % (Dim / 8)) * 8;
          const int64_t offset = key_offsets[local_col];
          const int4 value =
              offset >= 0 ? *reinterpret_cast<const int4*>(p.k + offset + d)
                          : make_int4(0, 0, 0, 0);
          *reinterpret_cast<int4*>(keys + local_col * LD + d) = value;
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::fill_fragment(c, 0.f);
#pragma unroll
        for (int d = 0; d < Dim; d += 16) {
          wmma::load_matrix_sync(a, keys + warp * 16 * LD + d, LD);
          wmma::load_matrix_sync(b, query + d, LD);
          wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(scores + warp * 16 * 16, c, 16,
                                wmma::mem_row_major);
        __syncthreads();
        if (tid < kMQATile) {
          const int col = tile * kMQATile + tid;
          if (col < visible) {
            float values[16];
#pragma unroll
            for (int h = 0; h < 16; ++h)
              values[h] = h < p.heads ? fmaxf(scores[tid * 16 + h], 0.f) : 0.f;
#pragma unroll
            for (int step = 8; step; step /= 2)
#pragma unroll
              for (int h = 0; h < step; ++h) values[h] += values[h + step];
            p.logits[int64_t(row) * p.out_row + col] =
                key_offsets[tid] >= 0 ? values[0] / p.divisor : -CUDART_INF_F;
          }
        }
        __syncthreads();
      }
    }
    ++row;
    tile = 0;
  }
}

}  // namespace flashinfer::attention::sm70
