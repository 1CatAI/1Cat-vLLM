// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Numerical-compatibility experiment, not an admitted runtime default.
// Reuse the pinned FlashInfer virtual-page preparation and FP32 cascade.
// The Volta partial kernel preserves production's 16-token tile partition,
// FP16 probabilities before PV, and unrounded FP32 normalization state.
#include <flashinfer/attention/sm70/qsa_decode.cuh>
#include <math_constants.h>
#include <mma.h>

namespace flashinfer::attention::sm70 {

template <int Group>
__global__ void QSAWMMACompatiblePartial(QSAParams p, int selected,
                                         int splits) {
  using namespace nvcuda;
  constexpr int D = 256, LD = D + 8, LP = 24;
  __shared__ __align__(32) half query[16 * LD];
  __shared__ __align__(32) half keys[16 * LD];
  __shared__ __align__(32) half values[16 * LD];
  __shared__ __align__(32) half probabilities[16 * LP];
  __shared__ __align__(32) float scores[16 * 16];
  // Reuse one output tile per warp; never reserve a padded 16 x D result.
  __shared__ __align__(32) float output[4 * 16 * 16];
  __shared__ float maximum[16], denominator[16], alpha[16];
  __shared__ int64_t offsets[16];
  const int tid = threadIdx.x, warp = tid / 32;
  const int row = blockIdx.x / splits, split = blockIdx.x % splits;
  const int kv_head = blockIdx.y;
  const int first_head = kv_head * Group;
  for (int i = tid; i < 16 * (D / 8); i += 128) {
    const int h = i / (D / 8), d = i % (D / 8) * 8;
    const int4 q = h < Group ? *reinterpret_cast<const int4*>(
                                   p.q + int64_t(row) * p.q_stride_n +
                                   (first_head + h) * p.q_stride_h + d)
                             : make_int4(0, 0, 0, 0);
    *reinterpret_cast<int4*>(query + h * LD + d) = q;
  }
  if (tid < 16) {
    maximum[tid] = -1e20f;
    denominator[tid] = 0.f;
  }
  // Discover the accumulator fragment's row layout using the supported WMMA
  // load API, rather than depending on undocumented lane/register mappings.
  for (int i = tid; i < 256; i += 128) scores[i] = float(i / 16);
  __syncthreads();
  using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
  Acc row_map, accum[4];
  wmma::load_matrix_sync(row_map, scores, 16, wmma::mem_row_major);
#pragma unroll
  for (int n = 0; n < 4; ++n) wmma::fill_fragment(accum[n], 0.f);
  __syncthreads();
  const int tiles = (selected + 15) / 16;
  const int start = split * tiles / splits;
  const int end = (split + 1) * tiles / splits;
  for (int tile = start; tile < end; ++tile) {
    if (tid < 16) {
      const int index = tile * 16 + tid;
      offsets[tid] =
          index < selected
              ? p.paged_kv.offsets[size_t(row) * p.paged_kv.width + index]
              : -1;
    }
    __syncthreads();
    for (int i = tid; i < 16 * (D / 8); i += 128) {
      const int n = i / (D / 8), d = i % (D / 8) * 8;
      const int64_t offset = offsets[n];
      int4 k = make_int4(0, 0, 0, 0), v = k;
      if (offset >= 0) {
        const int64_t address = offset + kv_head * p.paged_kv.head_stride + d;
        k = *reinterpret_cast<const int4*>(p.paged_kv.k_data.data + address);
        v = *reinterpret_cast<const int4*>(p.paged_kv.v_data.data + address);
      }
      *reinterpret_cast<int4*>(keys + n * LD + d) = k;
      *reinterpret_cast<int4*>(values + n * LD + d) = v;
    }
    __syncthreads();
    if (warp == 0) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
      Acc c;
      wmma::fill_fragment(c, 0.f);
#pragma unroll
      for (int d = 0; d < D; d += 16) {
        wmma::load_matrix_sync(a, query + d, LD);
        wmma::load_matrix_sync(b, keys + d, LD);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(scores, c, 16, wmma::mem_row_major);
    }
    __syncthreads();
    if (tid < 16) {
      float s[16], ps[16];
      float next_max = maximum[tid];
#pragma unroll
      for (int n = 0; n < 16; ++n) {
        s[n] = offsets[n] >= 0
                   ? scores[tid * 16 + n] * (1.4426950408889634f / 16.f)
                   : -1e20f;
        next_max = fmaxf(next_max, s[n]);
      }
      const float scale = exp2f(maximum[tid] - next_max);
#pragma unroll
      for (int n = 0; n < 16; ++n) {
        ps[n] = offsets[n] >= 0 ? exp2f(s[n] - next_max) : 0.f;
        probabilities[tid * LP + n] = __float2half_rn(ps[n]);
      }
#pragma unroll
      for (int step = 8; step > 0; step /= 2)
#pragma unroll
        for (int n = 0; n < step; ++n) ps[n] += ps[n + step];
      denominator[tid] = denominator[tid] * scale + ps[0];
      maximum[tid] = next_max;
      alpha[tid] = scale;
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::load_matrix_sync(a, probabilities, LP);
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
      for (int e = 0; e < Acc::num_elements; ++e)
        accum[n].x[e] *= alpha[int(row_map.x[e])];
      wmma::load_matrix_sync(b, values + warp * 64 + n * 16, LD);
      wmma::mma_sync(accum[n], a, b, accum[n]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int n = 0; n < 4; ++n) {
#pragma unroll
    for (int e = 0; e < Acc::num_elements; ++e) {
      const float norm = denominator[int(row_map.x[e])];
      accum[n].x[e] = norm > 0.f ? accum[n].x[e] / norm : 0.f;
    }
    wmma::store_matrix_sync(output + warp * 256, accum[n], 16,
                            wmma::mem_row_major);
    __syncwarp();
    for (int i = tid % 32; i < Group * 16; i += 32) {
      const int h = i / 16, d = warp * 64 + n * 16 + i % 16;
      p.o[(size_t(blockIdx.x) * p.num_qo_heads + first_head + h) * D + d] =
          output[warp * 256 + i];
    }
    __syncwarp();
  }
  if (tid < Group) {
    const float norm = denominator[tid];
    p.lse[size_t(blockIdx.x) * p.num_qo_heads + first_head + tid] =
        norm > 0.f ? maximum[tid] + log2f(norm) : -CUDART_INF_F;
  }
}

template <int Group>
void LaunchQSAWMMACompatible(QSAParams p, half* output, int rows, int splits,
                             int selected, cudaStream_t stream) {
  QSAWMMACompatiblePartial<Group>
      <<<dim3(rows * splits, p.paged_kv.num_heads), 128, 0, stream>>>(
          p, selected, splits);
  MergeStatesKernel<8, float, half>
      <<<rows, dim3(32, p.num_qo_heads), 0, stream>>>(
          p.o, p.lse, output, nullptr, splits, p.num_qo_heads, 256);
}

}  // namespace flashinfer::attention::sm70
