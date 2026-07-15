// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Standalone SM70 split-KV experiment. The BM32 attention body intentionally
// retains the accepted ALL_P=true, PAIR_SCRATCH=true implementation shape.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kM = 32;
constexpr int kM64 = 64;
constexpr int kD = 256;
constexpr int kBlockN = 128;
constexpr int kPanelM = 16;
constexpr int kPanelN = 16;
constexpr int kPanelK = 16;
constexpr int kSoftmaxPanelN = 32;
constexpr int kSoftmaxPanelsPerBlock = kBlockN / kSoftmaxPanelN;
constexpr int kQKWarps = kBlockN / kPanelN;
constexpr int kBm32Threads = 512;
constexpr int kMergeThreads = 256;
constexpr int kSplitParts = 3;
constexpr int kQPanelElements = kPanelM * kD;
constexpr int kQElements = kM * kD;
constexpr int kM64QElements = kM64 * kD;
constexpr int kPPanelElements = kPanelM * kSoftmaxPanelN;
constexpr int kOutputElements = kM * kD;
constexpr int kM64OutputElements = kM64 * kD;
constexpr float kNegativeInfinity = -1e30f;
constexpr float kComparisonAbsTolerance = 2e-2f;

static_assert(kM == 2 * kPanelM);
static_assert(kM64 == 4 * kPanelM);
static_assert(kBlockN == kQKWarps * kPanelN);
static_assert(kSoftmaxPanelN == 2 * kPanelK);
static_assert(kBm32Threads / 32 == 2 * kQKWarps);
static_assert(kMergeThreads == kM * 8);

using MatrixAFragment =
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kPanelM, kPanelN,
                          kPanelK, __half, nvcuda::wmma::row_major>;
using QKMatrixBFragment =
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kPanelM, kPanelN,
                          kPanelK, __half, nvcuda::wmma::col_major>;
using PVMatrixBFragment =
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kPanelM, kPanelN,
                          kPanelK, __half, nvcuda::wmma::row_major>;
using AccumulatorFragment =
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kPanelM, kPanelN,
                          kPanelK, float>;

// This is the accepted ALL_P=true, PAIR_SCRATCH=true BM32 storage layout.
struct alignas(16) BM32PairScratchShared {
  __half query[kQElements];
  float score[kM * kBlockN];
  __half probability_top[kSoftmaxPanelsPerBlock][kPPanelElements];
  __half probability_bottom[kSoftmaxPanelsPerBlock][kPPanelElements];
  float row_max[kM];
  float row_sum[kM];
  float row_exp_diff[kSoftmaxPanelsPerBlock][kM];
  int block_index;
};

static_assert(sizeof(BM32PairScratchShared) == 41744,
              "all-P shared layout changed unexpectedly");
static_assert(sizeof(BM32PairScratchShared) <= 48 * 1024,
              "BM32 pair-scratch storage exceeds the 48 KiB gate");

struct Args {
  int device = 0;
  int groups = 96;
  int nblocks = 96;
  int warmup = 20;
  int rounds = 100;
  int launches_per_sample = 8;
  bool profile_only = false;
  bool smoke_only = false;
  std::string profile_kernel = "all";
  std::string pattern = "random";
};

struct TimingSummary {
  double median_us = 0.0;
  double p90_us = 0.0;
  double mean_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;
};

struct PairSummary {
  int count = 0;
  int split_faster = 0;
  int baseline_faster = 0;
  int ties = 0;
  double split_minus_baseline_median_us = 0.0;
  double split_minus_baseline_mean_us = 0.0;
};

struct KernelResources {
  int registers_per_thread = 0;
  size_t static_shared_bytes = 0;
  size_t dynamic_shared_bytes = 0;
  size_t local_bytes_per_thread = 0;
  int active_ctas_per_sm = 0;
  int threads_per_cta = 0;
  int warps_per_cta = 0;
  int resident_warps = 0;
};

struct Comparison {
  bool all_finite = true;
  bool bitwise_equal = true;
  int64_t bitwise_mismatch_elements = 0;
  int64_t mismatch_elements_at_tolerance = 0;
  float max_abs_error = 0.0f;
  int64_t first_mismatch_element = -1;
  uint16_t first_baseline_bits = 0;
  uint16_t first_split_bits = 0;
  float first_baseline_value = 0.0f;
  float first_split_value = 0.0f;
};

void check_cuda(cudaError_t status, const char* expression, const char* file,
                int line) {
  if (status == cudaSuccess) {
    return;
  }
  std::cerr << "CUDA failure at " << file << ':' << line << " for "
            << expression << ": " << cudaGetErrorString(status) << '\n';
  std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) \
  check_cuda((expression), #expression, __FILE__, __LINE__)

__device__ __forceinline__ int swizzled_row_slot(int row) {
  return (row & 3) | ((row & 8) >> 1) | ((row & 4) << 1);
}

__device__ __forceinline__ int swizzled_matrix_a_offset(int row, int column) {
  const int tile = column / kPanelK;
  const int within_tile = column - tile * kPanelK;
  const int plane = within_tile >> 3;
  const int inner = within_tile & 7;
  return tile * kPanelM * kPanelK + plane * kPanelM * 8 +
         swizzled_row_slot(row) * 8 + inner;
}

__device__ __forceinline__ void stage_swizzled_q_panel(
    const __half* __restrict__ source, __half* __restrict__ destination,
    int thread, int thread_count) {
  constexpr int kHalfPerUint4 = sizeof(uint4) / sizeof(__half);
  constexpr int kQueryVectors = kQPanelElements / kHalfPerUint4;
  constexpr int kVectorsPerRow = kD / kHalfPerUint4;
  const uint4* source_vectors = reinterpret_cast<const uint4*>(source);
  uint4* destination_vectors = reinterpret_cast<uint4*>(destination);

  for (int index = thread; index < kQueryVectors; index += thread_count) {
    const int row = index / kVectorsPerRow;
    const int vector_column = index % kVectorsPerRow;
    const int k_tile = vector_column >> 1;
    const int plane = vector_column & 1;
    const int slot = swizzled_row_slot(row);
    destination_vectors[k_tile * (2 * kPanelM) + plane * kPanelM + slot] =
        __ldg(source_vectors + index);
  }
}

__device__ __forceinline__ void load_swizzled_matrix_a_fragment(
    MatrixAFragment& fragment, const __half* __restrict__ matrix,
    int k_offset) {
  const int lane = threadIdx.x & 31;
  const int row = (lane & 3) + ((lane >> 4) & 1) * 4 +
                  ((lane >> 2) & 1) * 8;
  const int slot = swizzled_row_slot(row);
  const int tile_offset = (k_offset / kPanelK) * kPanelM * kPanelK;
  uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
      matrix + tile_offset + slot * 8));
  uint32_t* words = reinterpret_cast<uint32_t*>(&fragment);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]),
                 "=r"(words[3])
               : "r"(address)
               : "memory");
  address += kPanelM * 8 * sizeof(__half);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(words[4]), "=r"(words[5]), "=r"(words[6]),
                 "=r"(words[7])
               : "r"(address)
               : "memory");
}

__device__ __forceinline__ int accumulator_fragment_row(int lane,
                                                         int element) {
  const int row_base =
      (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  return row_base + ((element >> 1) & 1) * 2;
}

__device__ __forceinline__ int accumulator_fragment_column(int lane,
                                                            int element) {
  const int column_base = ((lane >> 1) & 1) * 2 + ((lane >> 3) & 1) * 8;
  return column_base + (element & 1) + ((element >> 2) & 1) * 4;
}

__device__ __forceinline__ void qk_pair_accumulate(
    const __half* __restrict__ shared_query_top,
    const __half* __restrict__ shared_query_bottom,
    const __half* __restrict__ key_tile, AccumulatorFragment& top_accumulator,
    AccumulatorFragment& bottom_accumulator) {
  MatrixAFragment a_fragment;
  QKMatrixBFragment b_fragment;
  nvcuda::wmma::fill_fragment(top_accumulator, 0.0f);
  nvcuda::wmma::fill_fragment(bottom_accumulator, 0.0f);

#pragma unroll
  for (int k_offset = 0; k_offset < kD; k_offset += kPanelK) {
    nvcuda::wmma::load_matrix_sync(b_fragment, key_tile + k_offset, kD);
    load_swizzled_matrix_a_fragment(a_fragment, shared_query_top, k_offset);
    nvcuda::wmma::mma_sync(top_accumulator, a_fragment, b_fragment,
                           top_accumulator);
    load_swizzled_matrix_a_fragment(a_fragment, shared_query_bottom,
                                    k_offset);
    nvcuda::wmma::mma_sync(bottom_accumulator, a_fragment, b_fragment,
                           bottom_accumulator);
  }
}

__device__ __forceinline__ float make_probability_row(
    const float* __restrict__ score_row, __half* __restrict__ probability,
    int probability_row, float* __restrict__ row_max,
    float* __restrict__ row_sum, int state_row, int panel) {
  const int lane = threadIdx.x & 31;
  const float* panel_score = score_row + panel * kSoftmaxPanelN;
  const float4* score_vectors = reinterpret_cast<const float4*>(panel_score);
  float thread_max = kNegativeInfinity;
  if (lane < kSoftmaxPanelN / 4) {
    const float4 values = score_vectors[lane];
    thread_max = fmaxf(fmaxf(values.x, values.y), fmaxf(values.z, values.w));
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_max = fmaxf(thread_max,
                        __shfl_down_sync(0xffffffffU, thread_max, offset));
  }
  const float panel_max = __shfl_sync(0xffffffffU, thread_max, 0);
  const float old_max = row_max[state_row];
  const float new_max = fmaxf(old_max, panel_max);
  const float exp_diff = __expf(old_max - new_max);

  float thread_sum = 0.0f;
  if (lane < kSoftmaxPanelN / 4) {
    float4 values = score_vectors[lane];
    values.x = __expf(fmaxf(values.x - new_max, -80.0f));
    values.y = __expf(fmaxf(values.y - new_max, -80.0f));
    values.z = __expf(fmaxf(values.z - new_max, -80.0f));
    values.w = __expf(fmaxf(values.w - new_max, -80.0f));
    thread_sum = (values.x + values.y) + (values.z + values.w);

    const int column = lane * 4;
    __half2* first_pair = reinterpret_cast<__half2*>(
        probability + swizzled_matrix_a_offset(probability_row, column));
    __half2* second_pair = reinterpret_cast<__half2*>(
        probability + swizzled_matrix_a_offset(probability_row, column + 2));
    *first_pair = __float22half2_rn(make_float2(values.x, values.y));
    *second_pair = __float22half2_rn(make_float2(values.z, values.w));
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_sum += __shfl_down_sync(0xffffffffU, thread_sum, offset);
  }
  const float panel_sum = __shfl_sync(0xffffffffU, thread_sum, 0);
  if (lane == 0) {
    row_sum[state_row] = exp_diff * row_sum[state_row] + panel_sum;
    row_max[state_row] = new_max;
  }
  return exp_diff;
}

__device__ __forceinline__ void scale_accumulator_two_rows(
    AccumulatorFragment& accumulator, float first_row_scale,
    float second_row_scale) {
  accumulator.x[0] *= first_row_scale;
  accumulator.x[1] *= first_row_scale;
  accumulator.x[2] *= second_row_scale;
  accumulator.x[3] *= second_row_scale;
  accumulator.x[4] *= first_row_scale;
  accumulator.x[5] *= first_row_scale;
  accumulator.x[6] *= second_row_scale;
  accumulator.x[7] *= second_row_scale;
}

__device__ __forceinline__ void scale_phase_reuse_accumulators(
    AccumulatorFragment& accumulator_top,
    AccumulatorFragment& accumulator_bottom,
    const float* __restrict__ row_exp_diff) {
  const int row = accumulator_fragment_row(threadIdx.x & 31, 0);
  scale_accumulator_two_rows(accumulator_top, row_exp_diff[row],
                             row_exp_diff[row + 2]);
  scale_accumulator_two_rows(accumulator_bottom,
                             row_exp_diff[kPanelM + row],
                             row_exp_diff[kPanelM + row + 2]);
}

__device__ __forceinline__ void update_phase_reuse_pv_panel(
    const __half* __restrict__ probability_top,
    const __half* __restrict__ probability_bottom,
    const __half* __restrict__ value_panel, int d_offset,
    AccumulatorFragment& accumulator_top,
    AccumulatorFragment& accumulator_bottom) {
  {
    MatrixAFragment a_fragment;
    PVMatrixBFragment b_fragment;
    nvcuda::wmma::load_matrix_sync(b_fragment, value_panel + d_offset, kD);
    load_swizzled_matrix_a_fragment(a_fragment, probability_top, 0);
    nvcuda::wmma::mma_sync(accumulator_top, a_fragment, b_fragment,
                           accumulator_top);
    load_swizzled_matrix_a_fragment(a_fragment, probability_bottom, 0);
    nvcuda::wmma::mma_sync(accumulator_bottom, a_fragment, b_fragment,
                           accumulator_bottom);
  }
  {
    MatrixAFragment a_fragment;
    PVMatrixBFragment b_fragment;
    nvcuda::wmma::load_matrix_sync(
        b_fragment, value_panel + kPanelK * kD + d_offset, kD);
    load_swizzled_matrix_a_fragment(a_fragment, probability_top, kPanelK);
    nvcuda::wmma::mma_sync(accumulator_top, a_fragment, b_fragment,
                           accumulator_top);
    load_swizzled_matrix_a_fragment(a_fragment, probability_bottom, kPanelK);
    nvcuda::wmma::mma_sync(accumulator_bottom, a_fragment, b_fragment,
                           accumulator_bottom);
  }
}

__device__ __forceinline__ void store_accumulator_output(
    const AccumulatorFragment& accumulator, __half* __restrict__ output,
    const float* __restrict__ row_sum, int row_offset, int d_offset) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int element = 0; element < accumulator.num_elements; ++element) {
    const int row = row_offset + accumulator_fragment_row(lane, element);
    const int column = d_offset + accumulator_fragment_column(lane, element);
    const float inverse_sum = 1.0f / fmaxf(row_sum[row], 1e-24f);
    output[row * kD + column] =
        __float2half_rn(accumulator.x[element] * inverse_sum);
  }
}

__device__ __forceinline__ void store_unnormalized_accumulator(
    const AccumulatorFragment& accumulator, float* __restrict__ output,
    int row_offset, int d_offset) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int element = 0; element < accumulator.num_elements; ++element) {
    const int row = row_offset + accumulator_fragment_row(lane, element);
    const int column = d_offset + accumulator_fragment_column(lane, element);
    output[row * kD + column] = accumulator.x[element];
  }
}

__device__ __forceinline__ void spill_qk_warp_pv_accumulators(
    float* __restrict__ shared_score, AccumulatorFragment& accumulator_top,
    AccumulatorFragment& accumulator_bottom) {
  const int warp = threadIdx.x >> 5;
  if (warp >= kQKWarps) {
    return;
  }
  const int lane = threadIdx.x & 31;
  const int warp_pair = warp >> 1;
  const int warp_in_pair = warp & 1;
  const int pair_column = warp_pair * 32 + (lane & 15) * 2;
  const int lane_row = lane >> 4;

#pragma unroll
  for (int element_pair = 0; element_pair < 4; ++element_pair) {
    const int top_offset =
        (warp_in_pair * 16 + element_pair * 2 + lane_row) * kBlockN +
        pair_column;
    const uint32_t top_address = static_cast<uint32_t>(
        __cvta_generic_to_shared(shared_score + top_offset));
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};"
                 :
                 : "r"(top_address),
                   "r"(__float_as_uint(
                       accumulator_top.x[2 * element_pair])),
                   "r"(__float_as_uint(
                       accumulator_top.x[2 * element_pair + 1]))
                 : "memory");
    asm volatile("st.shared.v2.u32 [%0+4096], {%1, %2};"
                 :
                 : "r"(top_address),
                   "r"(__float_as_uint(
                       accumulator_bottom.x[2 * element_pair])),
                   "r"(__float_as_uint(
                       accumulator_bottom.x[2 * element_pair + 1]))
                 : "memory");
  }
}

__device__ __forceinline__ void reload_qk_warp_pv_accumulators(
    const float* __restrict__ shared_score, AccumulatorFragment& accumulator_top,
    AccumulatorFragment& accumulator_bottom) {
  const int warp = threadIdx.x >> 5;
  if (warp >= kQKWarps) {
    return;
  }
  const int lane = threadIdx.x & 31;
  const int warp_pair = warp >> 1;
  const int warp_in_pair = warp & 1;
  const int pair_column = warp_pair * 32 + (lane & 15) * 2;
  const int lane_row = lane >> 4;

#pragma unroll
  for (int element_pair = 0; element_pair < 4; ++element_pair) {
    const int top_offset =
        (warp_in_pair * 16 + element_pair * 2 + lane_row) * kBlockN +
        pair_column;
    const uint32_t top_address = static_cast<uint32_t>(
        __cvta_generic_to_shared(shared_score + top_offset));
    uint32_t first_word;
    uint32_t second_word;
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                 : "=r"(first_word), "=r"(second_word)
                 : "r"(top_address)
                 : "memory");
    accumulator_top.x[2 * element_pair] = __uint_as_float(first_word);
    accumulator_top.x[2 * element_pair + 1] = __uint_as_float(second_word);
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2+4096];"
                 : "=r"(first_word), "=r"(second_word)
                 : "r"(top_address)
                 : "memory");
    accumulator_bottom.x[2 * element_pair] = __uint_as_float(first_word);
    accumulator_bottom.x[2 * element_pair + 1] =
        __uint_as_float(second_word);
  }
}

__device__ __forceinline__ void sync_qk_warp_pair(int warp_pair) {
  const int barrier_id = warp_pair + 1;
  asm volatile("bar.sync %0, 64;" : : "r"(barrier_id) : "memory");
}

extern "C" __global__ __launch_bounds__(kBm32Threads, 2)
void sm70_native_bm32_splitkv_unsplit_baseline(
    const __half* __restrict__ query, const __half* __restrict__ key,
    const __half* __restrict__ value, __half* __restrict__ output, int groups,
    int nblocks) {
  __shared__ __align__(16) BM32PairScratchShared shared;

  const int m32_index = blockIdx.x;
  const int group = m32_index >> 1;
  if (group >= groups) {
    return;
  }
  const int m32_panel = m32_index & 1;
  const __half* query_group =
      query + static_cast<int64_t>(group) * kM64QElements +
      m32_panel * kQElements;
  stage_swizzled_q_panel(query_group, shared.query, threadIdx.x,
                         kBm32Threads);
  stage_swizzled_q_panel(query_group + kQPanelElements,
                         shared.query + kQPanelElements, threadIdx.x,
                         kBm32Threads);
  if (threadIdx.x < kM) {
    shared.row_max[threadIdx.x] = kNegativeInfinity;
    shared.row_sum[threadIdx.x] = 0.0f;
  }
  if (threadIdx.x == 0) {
    shared.block_index = 0;
  }
  __syncthreads();

  AccumulatorFragment accumulator_top;
  AccumulatorFragment accumulator_bottom;
  nvcuda::wmma::fill_fragment(accumulator_top, 0.0f);
  nvcuda::wmma::fill_fragment(accumulator_bottom, 0.0f);

  for (;;) {
    if (shared.block_index >= nblocks) {
      break;
    }
    const int qk_warp = threadIdx.x >> 5;
    if (qk_warp < kQKWarps) {
      const int n_offset = qk_warp * kPanelN;
      spill_qk_warp_pv_accumulators(shared.score, accumulator_top,
                                    accumulator_bottom);
      asm volatile("" ::: "memory");
      AccumulatorFragment qk_top;
      AccumulatorFragment qk_bottom;
      qk_pair_accumulate(shared.query, shared.query + kQPanelElements,
                         key +
                             (shared.block_index * kBlockN + n_offset) * kD,
                         qk_top, qk_bottom);
      asm volatile("" ::: "memory");
      reload_qk_warp_pv_accumulators(shared.score, accumulator_top,
                                     accumulator_bottom);
      sync_qk_warp_pair(qk_warp >> 1);
      nvcuda::wmma::store_matrix_sync(shared.score + n_offset, qk_top,
                                      kBlockN,
                                      nvcuda::wmma::mem_row_major);
      nvcuda::wmma::store_matrix_sync(
          shared.score + kPanelM * kBlockN + n_offset, qk_bottom, kBlockN,
          nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

#pragma unroll
    for (int panel = 0; panel < kSoftmaxPanelsPerBlock; ++panel) {
      const int phase_warp = threadIdx.x >> 5;
      const float top_exp_diff = make_probability_row(
          shared.score + phase_warp * kBlockN,
          shared.probability_top[panel], phase_warp, shared.row_max,
          shared.row_sum, phase_warp, panel);
      const float bottom_exp_diff = make_probability_row(
          shared.score + (kPanelM + phase_warp) * kBlockN,
          shared.probability_bottom[panel], phase_warp, shared.row_max,
          shared.row_sum, kPanelM + phase_warp, panel);
      if ((threadIdx.x & 31) == 0) {
        shared.row_exp_diff[panel][phase_warp] = top_exp_diff;
        shared.row_exp_diff[panel][kPanelM + phase_warp] = bottom_exp_diff;
      }
      __syncwarp();
    }
    __syncthreads();

    const int d_offset = (threadIdx.x >> 5) * kPanelN;
#pragma unroll
    for (int panel = 0; panel < kSoftmaxPanelsPerBlock; ++panel) {
      if (shared.block_index != 0 || panel != 0) {
        scale_phase_reuse_accumulators(accumulator_top, accumulator_bottom,
                                       shared.row_exp_diff[panel]);
      }
      update_phase_reuse_pv_panel(
          shared.probability_top[panel], shared.probability_bottom[panel],
          value +
              (shared.block_index * kBlockN + panel * kSoftmaxPanelN) * kD,
          d_offset, accumulator_top, accumulator_bottom);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      ++shared.block_index;
    }
    __syncthreads();
  }

  __half* output_m32 =
      output + static_cast<int64_t>(m32_index) * kOutputElements;
  const int d_offset = (threadIdx.x >> 5) * kPanelN;
  store_accumulator_output(accumulator_top, output_m32, shared.row_sum, 0,
                           d_offset);
  store_accumulator_output(accumulator_bottom, output_m32, shared.row_sum,
                           kPanelM, d_offset);
}

extern "C" __global__ __launch_bounds__(kBm32Threads, 2)
void sm70_native_bm32_splitkv_partial(
    const __half* __restrict__ query, const __half* __restrict__ key,
    const __half* __restrict__ value, float* __restrict__ partial_accumulator,
    float* __restrict__ partial_row_max, float* __restrict__ partial_row_sum,
    int groups, int nblocks) {
  __shared__ __align__(16) BM32PairScratchShared shared;

  const int m32_index = blockIdx.x;
  const int split_index = blockIdx.y;
  const int group = m32_index >> 1;
  if (group >= groups) {
    return;
  }
  const int m32_panel = m32_index & 1;
  const int split_begin_block = split_index * nblocks / kSplitParts;
  const int split_end_block = (split_index + 1) * nblocks / kSplitParts;
  const __half* query_group =
      query + static_cast<int64_t>(group) * kM64QElements +
      m32_panel * kQElements;
  stage_swizzled_q_panel(query_group, shared.query, threadIdx.x,
                         kBm32Threads);
  stage_swizzled_q_panel(query_group + kQPanelElements,
                         shared.query + kQPanelElements, threadIdx.x,
                         kBm32Threads);
  if (threadIdx.x < kM) {
    shared.row_max[threadIdx.x] = kNegativeInfinity;
    shared.row_sum[threadIdx.x] = 0.0f;
  }
  if (threadIdx.x == 0) {
    shared.block_index = split_begin_block;
  }
  __syncthreads();

  AccumulatorFragment accumulator_top;
  AccumulatorFragment accumulator_bottom;
  nvcuda::wmma::fill_fragment(accumulator_top, 0.0f);
  nvcuda::wmma::fill_fragment(accumulator_bottom, 0.0f);

  for (;;) {
    if (shared.block_index >= split_end_block) {
      break;
    }
    const int qk_warp = threadIdx.x >> 5;
    if (qk_warp < kQKWarps) {
      const int n_offset = qk_warp * kPanelN;
      spill_qk_warp_pv_accumulators(shared.score, accumulator_top,
                                    accumulator_bottom);
      asm volatile("" ::: "memory");
      AccumulatorFragment qk_top;
      AccumulatorFragment qk_bottom;
      qk_pair_accumulate(shared.query, shared.query + kQPanelElements,
                         key +
                             (shared.block_index * kBlockN + n_offset) * kD,
                         qk_top, qk_bottom);
      asm volatile("" ::: "memory");
      reload_qk_warp_pv_accumulators(shared.score, accumulator_top,
                                     accumulator_bottom);
      sync_qk_warp_pair(qk_warp >> 1);
      nvcuda::wmma::store_matrix_sync(shared.score + n_offset, qk_top,
                                      kBlockN,
                                      nvcuda::wmma::mem_row_major);
      nvcuda::wmma::store_matrix_sync(
          shared.score + kPanelM * kBlockN + n_offset, qk_bottom, kBlockN,
          nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

#pragma unroll
    for (int panel = 0; panel < kSoftmaxPanelsPerBlock; ++panel) {
      const int phase_warp = threadIdx.x >> 5;
      const float top_exp_diff = make_probability_row(
          shared.score + phase_warp * kBlockN,
          shared.probability_top[panel], phase_warp, shared.row_max,
          shared.row_sum, phase_warp, panel);
      const float bottom_exp_diff = make_probability_row(
          shared.score + (kPanelM + phase_warp) * kBlockN,
          shared.probability_bottom[panel], phase_warp, shared.row_max,
          shared.row_sum, kPanelM + phase_warp, panel);
      if ((threadIdx.x & 31) == 0) {
        shared.row_exp_diff[panel][phase_warp] = top_exp_diff;
        shared.row_exp_diff[panel][kPanelM + phase_warp] = bottom_exp_diff;
      }
      __syncwarp();
    }
    __syncthreads();

    const int d_offset = (threadIdx.x >> 5) * kPanelN;
#pragma unroll
    for (int panel = 0; panel < kSoftmaxPanelsPerBlock; ++panel) {
      if (shared.block_index != split_begin_block || panel != 0) {
        scale_phase_reuse_accumulators(accumulator_top, accumulator_bottom,
                                       shared.row_exp_diff[panel]);
      }
      update_phase_reuse_pv_panel(
          shared.probability_top[panel], shared.probability_bottom[panel],
          value +
              (shared.block_index * kBlockN + panel * kSoftmaxPanelN) * kD,
          d_offset, accumulator_top, accumulator_bottom);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      ++shared.block_index;
    }
    __syncthreads();
  }

  const int64_t partial_slot =
      static_cast<int64_t>(m32_index) * kSplitParts + split_index;
  float* partial_output =
      partial_accumulator + partial_slot * kOutputElements;
  const int d_offset = (threadIdx.x >> 5) * kPanelN;
  store_unnormalized_accumulator(accumulator_top, partial_output, 0,
                                 d_offset);
  store_unnormalized_accumulator(accumulator_bottom, partial_output,
                                 kPanelM, d_offset);
  if (threadIdx.x < kM) {
    partial_row_max[partial_slot * kM + threadIdx.x] =
        shared.row_max[threadIdx.x];
    partial_row_sum[partial_slot * kM + threadIdx.x] =
        shared.row_sum[threadIdx.x];
  }
}

extern "C" __global__ __launch_bounds__(kMergeThreads)
void sm70_native_bm32_splitkv_merge(
    const float* __restrict__ partial_accumulator,
    const float* __restrict__ partial_row_max,
    const float* __restrict__ partial_row_sum, __half* __restrict__ output,
    int groups) {
  const int m32_index = blockIdx.x;
  if ((m32_index >> 1) >= groups) {
    return;
  }
  const int row = threadIdx.x >> 3;
  const int lane_in_row = threadIdx.x & 7;
  const int64_t first_slot = static_cast<int64_t>(m32_index) * kSplitParts;
  const int64_t stats_base = first_slot * kM + row;
  const float max_0 = partial_row_max[stats_base];
  const float max_1 = partial_row_max[stats_base + kM];
  const float max_2 = partial_row_max[stats_base + 2 * kM];
  const float merged_max = fmaxf(max_0, fmaxf(max_1, max_2));
  const float weight_0 = __expf(max_0 - merged_max);
  const float weight_1 = __expf(max_1 - merged_max);
  const float weight_2 = __expf(max_2 - merged_max);
  const float denominator =
      weight_0 * partial_row_sum[stats_base] +
      weight_1 * partial_row_sum[stats_base + kM] +
      weight_2 * partial_row_sum[stats_base + 2 * kM];
  const float inverse_denominator = 1.0f / fmaxf(denominator, 1e-24f);
  const int64_t output_base = static_cast<int64_t>(m32_index) * kOutputElements;
  const int64_t partial_base_0 = first_slot * kOutputElements + row * kD;
  const int64_t partial_base_1 =
      (first_slot + 1) * kOutputElements + row * kD;
  const int64_t partial_base_2 =
      (first_slot + 2) * kOutputElements + row * kD;

#pragma unroll
  for (int column = lane_in_row; column < kD; column += 8) {
    const float numerator =
        weight_0 * partial_accumulator[partial_base_0 + column] +
        weight_1 * partial_accumulator[partial_base_1 + column] +
        weight_2 * partial_accumulator[partial_base_2 + column];
    output[output_base + row * kD + column] =
        __float2half_rn(numerator * inverse_denominator);
  }
}

uint32_t next_random(uint32_t* state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

float random_half_value(uint32_t* state) {
  const int value = static_cast<int>(next_random(state) & 0x7ffu) - 1024;
  return static_cast<float>(value) / 2048.0f;
}

float alternating_half_value(size_t index, uint32_t salt) {
  const int magnitude =
      1 + static_cast<int>((index * 17u + salt * 29u) % 127u);
  const float value = static_cast<float>(magnitude) / 512.0f;
  return ((index + salt) & 1u) == 0 ? value : -value;
}

void fill_input(std::vector<__half>* values, const std::string& pattern,
                uint32_t* random_state, uint32_t salt) {
  for (size_t index = 0; index < values->size(); ++index) {
    const float value = pattern == "random"
                            ? random_half_value(random_state)
                            : alternating_half_value(index, salt);
    (*values)[index] = __float2half_rn(value);
  }
}

uint16_t half_bits(const __half* values, size_t index) {
  uint16_t bits = 0;
  std::memcpy(&bits, values + index, sizeof(bits));
  return bits;
}

Comparison compare_outputs(const std::vector<__half>& baseline,
                           const std::vector<__half>& split) {
  Comparison result;
  for (size_t index = 0; index < baseline.size(); ++index) {
    const uint16_t baseline_bits = half_bits(baseline.data(), index);
    const uint16_t split_bits = half_bits(split.data(), index);
    const float baseline_value = __half2float(baseline[index]);
    const float split_value = __half2float(split[index]);
    const bool finite = std::isfinite(baseline_value) && std::isfinite(split_value);
    if (!finite) {
      result.all_finite = false;
      result.max_abs_error = std::numeric_limits<float>::infinity();
    } else {
      const float absolute_error = std::fabs(baseline_value - split_value);
      result.max_abs_error = std::max(result.max_abs_error, absolute_error);
      result.mismatch_elements_at_tolerance +=
          absolute_error > kComparisonAbsTolerance;
    }
    if (baseline_bits != split_bits) {
      result.bitwise_equal = false;
      ++result.bitwise_mismatch_elements;
      if (result.first_mismatch_element < 0) {
        result.first_mismatch_element = static_cast<int64_t>(index);
        result.first_baseline_bits = baseline_bits;
        result.first_split_bits = split_bits;
        result.first_baseline_value = baseline_value;
        result.first_split_value = split_value;
      }
    }
  }
  return result;
}

TimingSummary summarize(const std::vector<double>& samples) {
  std::vector<double> ordered = samples;
  std::sort(ordered.begin(), ordered.end());
  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  const size_t middle = ordered.size() / 2;
  const double median = ordered.size() % 2 == 0
                            ? (ordered[middle - 1] + ordered[middle]) / 2.0
                            : ordered[middle];
  return {median,
          ordered[static_cast<size_t>(0.9 * (ordered.size() - 1))],
          sum / static_cast<double>(samples.size()), ordered.front(),
          ordered.back()};
}

PairSummary summarize_pairs(const std::vector<double>& baseline,
                            const std::vector<double>& split) {
  std::vector<double> deltas;
  deltas.reserve(baseline.size());
  PairSummary result;
  result.count = static_cast<int>(baseline.size());
  for (size_t index = 0; index < baseline.size(); ++index) {
    const double delta = split[index] - baseline[index];
    deltas.push_back(delta);
    if (split[index] < baseline[index]) {
      ++result.split_faster;
    } else if (baseline[index] < split[index]) {
      ++result.baseline_faster;
    } else {
      ++result.ties;
    }
  }
  const TimingSummary summary = summarize(deltas);
  result.split_minus_baseline_median_us = summary.median_us;
  result.split_minus_baseline_mean_us = summary.mean_us;
  return result;
}

template <typename Kernel>
KernelResources query_resources(Kernel kernel, int threads_per_cta) {
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int active_ctas = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_ctas, kernel, threads_per_cta, 0));
  KernelResources result;
  result.registers_per_thread = attributes.numRegs;
  result.static_shared_bytes = attributes.sharedSizeBytes;
  result.dynamic_shared_bytes = 0;
  result.local_bytes_per_thread = attributes.localSizeBytes;
  result.active_ctas_per_sm = active_ctas;
  result.threads_per_cta = threads_per_cta;
  result.warps_per_cta = threads_per_cta / 32;
  result.resident_warps = result.active_ctas_per_sm * result.warps_per_cta;
  return result;
}

bool bm32_resource_gate(const KernelResources& resources) {
  return resources.threads_per_cta == kBm32Threads &&
         resources.registers_per_thread == 64 &&
         resources.static_shared_bytes == sizeof(BM32PairScratchShared) &&
         resources.dynamic_shared_bytes == 0 &&
         resources.local_bytes_per_thread == 0 &&
         resources.active_ctas_per_sm == 2;
}

void print_json_string(const std::string& value) {
  std::cout << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '\\':
        std::cout << "\\\\";
        break;
      case '"':
        std::cout << "\\\"";
        break;
      case '\n':
        std::cout << "\\n";
        break;
      case '\r':
        std::cout << "\\r";
        break;
      case '\t':
        std::cout << "\\t";
        break;
      default:
        if (character < 0x20) {
          std::cout << "\\u00" << std::hex << std::setw(2)
                    << std::setfill('0') << static_cast<int>(character)
                    << std::dec << std::setfill(' ');
        } else {
          std::cout << character;
        }
    }
  }
  std::cout << '"';
}

void print_timing(const TimingSummary& timing) {
  std::cout << "{\"median_us\": " << std::setprecision(9)
            << timing.median_us << ", \"p90_us\": " << timing.p90_us
            << ", \"mean_us\": " << timing.mean_us
            << ", \"min_us\": " << timing.min_us
            << ", \"max_us\": " << timing.max_us << '}';
}

void print_resources(const KernelResources& resources) {
  std::cout << "{\"registers_per_thread\": "
            << resources.registers_per_thread
            << ", \"static_shared_bytes\": "
            << resources.static_shared_bytes
            << ", \"dynamic_shared_bytes\": "
            << resources.dynamic_shared_bytes
            << ", \"local_bytes_per_thread\": "
            << resources.local_bytes_per_thread
            << ", \"active_ctas_per_sm\": "
            << resources.active_ctas_per_sm << ", \"threads_per_cta\": "
            << resources.threads_per_cta << ", \"warps_per_cta\": "
            << resources.warps_per_cta << ", \"resident_warps\": "
            << resources.resident_warps << '}';
}

void print_comparison(const Comparison& comparison) {
  std::cout << "{\"full_output\": true, \"all_finite\": "
            << (comparison.all_finite ? "true" : "false")
            << ", \"bitwise_equal\": "
            << (comparison.bitwise_equal ? "true" : "false")
            << ", \"bitwise_mismatch_elements\": "
            << comparison.bitwise_mismatch_elements
            << ", \"abs_tolerance\": " << kComparisonAbsTolerance
            << ", \"mismatch_elements_at_tolerance\": "
            << comparison.mismatch_elements_at_tolerance
            << ", \"max_abs_error\": ";
  if (std::isfinite(comparison.max_abs_error)) {
    std::cout << comparison.max_abs_error;
  } else {
    std::cout << "null";
  }
  std::cout << ", \"first_difference\": ";
  if (comparison.first_mismatch_element < 0) {
    std::cout << "null";
  } else {
    const int64_t element = comparison.first_mismatch_element;
    const int64_t group = element / kM64OutputElements;
    const int64_t in_group = element % kM64OutputElements;
    std::cout << "{\"element\": " << element << ", \"group\": "
              << group << ", \"row\": " << in_group / kD
              << ", \"column\": " << in_group % kD
              << ", \"baseline_bits\": "
              << comparison.first_baseline_bits << ", \"split_bits\": "
              << comparison.first_split_bits << ", \"baseline_value\": "
              << comparison.first_baseline_value << ", \"split_value\": "
              << comparison.first_split_value << '}';
  }
  std::cout << '}';
}

void print_split_ranges(int nblocks) {
  std::cout << '[';
  for (int split = 0; split < kSplitParts; ++split) {
    if (split != 0) {
      std::cout << ", ";
    }
    std::cout << "{\"split\": " << split << ", \"begin_block\": "
              << split * nblocks / kSplitParts << ", \"end_block\": "
              << (split + 1) * nblocks / kSplitParts << '}';
  }
  std::cout << ']';
}

void print_json(const Args& args, const cudaDeviceProp& properties,
                int runtime_version, int sm_count,
                const KernelResources& baseline_resources,
                const KernelResources& partial_resources,
                const KernelResources& merge_resources, bool executed,
                bool comparison_available, const Comparison& comparison,
                bool timing_available, const TimingSummary& baseline_timing,
                const TimingSummary& partial_timing,
                const TimingSummary& merge_timing,
                const TimingSummary& combined_timing,
                const PairSummary& pairs) {
  const bool resource_pass = bm32_resource_gate(baseline_resources) &&
                             bm32_resource_gate(partial_resources);
  const bool comparison_pass =
      comparison.all_finite && comparison.max_abs_error <= kComparisonAbsTolerance;
  const int baseline_ctas = args.groups * 2;
  const int partial_ctas = baseline_ctas * kSplitParts;
  const int runtime_wave_ctas = sm_count * 2;
  const double combined_speedup_pct =
      timing_available
          ? 100.0 * (baseline_timing.median_us - combined_timing.median_us) /
                baseline_timing.median_us
          : 0.0;
  std::cout << "{\n";
  std::cout << "  \"device\": {\"logical_index\": " << args.device
            << ", \"name\": ";
  print_json_string(properties.name);
  std::cout << ", \"capability\": [" << properties.major << ", "
            << properties.minor << "], \"cuda_runtime\": "
            << runtime_version << ", \"sm_count\": " << sm_count << "},\n";
  std::cout << "  \"target\": \"sm_70\",\n";
  std::cout << "  \"shape\": {\"m64_query_groups\": " << args.groups
            << ", \"m32_ctas_per_unsplit_grid\": " << baseline_ctas
            << ", \"m32_ctas_per_split_partial_grid\": " << partial_ctas
            << ", \"split_parts\": " << kSplitParts
            << ", \"M\": 64, \"BM\": 32, \"D\": " << kD
            << ", \"BN\": " << kBlockN << ", \"nblocks\": "
            << args.nblocks << ", \"N\": " << args.nblocks * kBlockN
            << ", \"hkv\": 1, \"kv_shared_across_query_groups\": true, "
               "\"split_block_ranges\": ";
  print_split_ranges(args.nblocks);
  std::cout << "},\n";
  std::cout << "  \"wave_model\": {\"runtime_sm_count\": " << sm_count
            << ", \"ctas_per_sm\": 2, \"runtime_full_wave_ctas\": "
            << runtime_wave_ctas << ", \"baseline_full_waves\": "
            << baseline_ctas / runtime_wave_ctas
            << ", \"baseline_tail_ctas\": "
            << baseline_ctas % runtime_wave_ctas
            << ", \"partial_full_waves\": "
            << partial_ctas / runtime_wave_ctas
            << ", \"partial_tail_ctas\": "
            << partial_ctas % runtime_wave_ctas
            << ", \"v100_72sm_reference\": {\"full_wave_ctas\": 144, "
               "\"baseline_full_waves\": "
            << baseline_ctas / 144 << ", \"baseline_tail_ctas\": "
            << baseline_ctas % 144 << ", \"partial_full_waves\": "
            << partial_ctas / 144 << ", \"partial_tail_ctas\": "
            << partial_ctas % 144 << "}},\n";
  std::cout << "  \"input_pattern\": ";
  print_json_string(args.pattern);
  std::cout << ",\n";
  std::cout << "  \"layouts\": {\"q\": \"[group,M64,D256] fp16\", "
               "\"k_v\": \"[N,D256] fp16 shared for all query groups\", "
               "\"partial_accumulator\": \"[m32_cta,split,M32,D256] fp32\", "
               "\"partial_row_state\": \"[m32_cta,split,M32] fp32 max/sum\", "
               "\"output\": \"[group,M64,D256] fp16\"},\n";
  std::cout << "  \"paths\": {\"baseline\": \"unsplit: 2 CTA/M64 group; "
               "BM32, 512 threads; accepted ALL_P=true PAIR_SCRATCH=true\", "
               "\"partial\": \"3-way split-KV: 6 CTA/M64 group; each BM32 "
               "partial is 512 threads and preserves ALL_P+PAIR_SCRATCH\", "
               "\"merge\": \"one 256-thread CTA/BM32; stable max/sum weighted "
               "merge of three unnormalized FP32 accumulators\"},\n";
  std::cout << "  \"execution\": {\"profile_only\": "
            << (args.profile_only ? "true" : "false")
            << ", \"smoke_only\": "
            << (args.smoke_only ? "true" : "false")
            << ", \"kernels_executed\": " << (executed ? "true" : "false")
            << ", \"resource_gate_pass\": "
            << (resource_pass ? "true" : "false")
            << ", \"comparison_gate_pass\": "
            << (comparison_available && comparison_pass ? "true" : "false")
            << "},\n";
  std::cout << "  \"resources\": {\"baseline\": ";
  print_resources(baseline_resources);
  std::cout << ", \"partial\": ";
  print_resources(partial_resources);
  std::cout << ", \"merge\": ";
  print_resources(merge_resources);
  std::cout << "},\n";
  std::cout << "  \"resource_gate\": {\"partial_required_threads_per_cta\": "
            << kBm32Threads
            << ", \"partial_required_registers_per_thread\": 64, "
               "\"partial_required_static_shared_bytes\": "
            << sizeof(BM32PairScratchShared)
            << ", \"partial_required_dynamic_shared_bytes\": 0, "
               "\"partial_required_local_bytes_per_thread\": 0, "
               "\"partial_required_active_ctas_per_sm\": 2, "
               "\"baseline_required_same_shape\": true, "
               "\"runtime_pass\": "
            << (resource_pass ? "true" : "false")
            << ", \"ptxas_validation\": \"Python harness requires 0 stack "
               "frame, spill stores, and spill loads for baseline and partial\", "
               "\"sass_validation\": \"Python harness rejects LDL/STL in "
               "baseline and partial\"},\n";
  std::cout << "  \"comparison\": ";
  if (comparison_available) {
    print_comparison(comparison);
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"timing\": ";
  if (timing_available) {
    std::cout << "{\"unit\": \"us per grid launch\", \"baseline\": ";
    print_timing(baseline_timing);
    std::cout << ", \"partial\": ";
    print_timing(partial_timing);
    std::cout << ", \"merge\": ";
    print_timing(merge_timing);
    std::cout << ", \"combined\": ";
    print_timing(combined_timing);
    std::cout << ", \"combined_speedup_vs_baseline_pct\": "
              << combined_speedup_pct << '}';
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"pairs\": ";
  if (timing_available) {
    std::cout << "{\"count\": " << pairs.count
              << ", \"split_faster\": " << pairs.split_faster
              << ", \"baseline_faster\": " << pairs.baseline_faster
              << ", \"ties\": " << pairs.ties
              << ", \"combined_minus_baseline_median_us\": "
              << pairs.split_minus_baseline_median_us
              << ", \"combined_minus_baseline_mean_us\": "
              << pairs.split_minus_baseline_mean_us << '}';
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"measurement\": {\"warmup_rounds\": " << args.warmup
            << ", \"rounds\": " << args.rounds
            << ", \"launches_per_sample\": " << args.launches_per_sample
            << ", \"combined_definition\": \"partial grid followed by merge "
               "grid in one CUDA event interval\"}\n";
  std::cout << "}\n";
}

bool profile_kernel_selected(const Args& args, const char* kernel) {
  return args.profile_kernel == "all" || args.profile_kernel == kernel;
}

int run(const Args& args) {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (args.device < 0 || args.device >= device_count) {
    std::cerr << "Requested logical CUDA device " << args.device
              << " is unavailable; visible device count is " << device_count
              << '\n';
    return EXIT_FAILURE;
  }
  if (args.groups < 1 || args.nblocks < kSplitParts || args.warmup < 0 ||
      args.rounds < 1 || args.launches_per_sample < 1) {
    std::cerr << "groups must be positive; nblocks must be at least "
              << kSplitParts
              << "; rounds and launches must be positive; warmup cannot be "
                 "negative\n";
    return EXIT_FAILURE;
  }
  if (args.pattern != "random" && args.pattern != "alternating") {
    std::cerr << "--pattern must be random or alternating\n";
    return EXIT_FAILURE;
  }

  CUDA_CHECK(cudaSetDevice(args.device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, args.device));
  if (properties.major != 7 || properties.minor != 0) {
    std::cerr << "This microbenchmark requires SM70, got " << properties.major
              << '.' << properties.minor << '\n';
    return EXIT_FAILURE;
  }
  int runtime_version = 0;
  int sm_count = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    args.device));
  const KernelResources baseline_resources = query_resources(
      sm70_native_bm32_splitkv_unsplit_baseline, kBm32Threads);
  const KernelResources partial_resources =
      query_resources(sm70_native_bm32_splitkv_partial, kBm32Threads);
  const KernelResources merge_resources =
      query_resources(sm70_native_bm32_splitkv_merge, kMergeThreads);
  const Comparison no_comparison;
  const TimingSummary no_timing;
  const PairSummary no_pairs;
  if (!bm32_resource_gate(baseline_resources) ||
      !bm32_resource_gate(partial_resources)) {
    print_json(args, properties, runtime_version, sm_count, baseline_resources,
               partial_resources, merge_resources, false, false, no_comparison,
               false, no_timing, no_timing, no_timing, no_timing, no_pairs);
    return EXIT_FAILURE;
  }

  const size_t query_elements =
      static_cast<size_t>(args.groups) * kM64QElements;
  const size_t kv_elements =
      static_cast<size_t>(args.nblocks) * kBlockN * kD;
  const size_t output_elements =
      static_cast<size_t>(args.groups) * kM64OutputElements;
  const size_t partial_slots = static_cast<size_t>(args.groups) * 2 * kSplitParts;
  const size_t partial_accumulator_elements = partial_slots * kOutputElements;
  const size_t partial_row_elements = partial_slots * kM;
  std::vector<__half> host_query(query_elements);
  std::vector<__half> host_key(kv_elements);
  std::vector<__half> host_value(kv_elements);
  uint32_t random_state = 0x6d2b79f5u;
  fill_input(&host_query, args.pattern, &random_state, 1);
  fill_input(&host_key, args.pattern, &random_state, 2);
  fill_input(&host_value, args.pattern, &random_state, 3);

  __half* device_query = nullptr;
  __half* device_key = nullptr;
  __half* device_value = nullptr;
  __half* device_baseline = nullptr;
  __half* device_split = nullptr;
  float* device_partial_accumulator = nullptr;
  float* device_partial_row_max = nullptr;
  float* device_partial_row_sum = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_query),
                        query_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_key),
                        kv_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_value),
                        kv_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_baseline),
                        output_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_split),
                        output_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_partial_accumulator),
                        partial_accumulator_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_partial_row_max),
                        partial_row_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_partial_row_sum),
                        partial_row_elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_query, host_query.data(),
                        query_elements * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_key, host_key.data(),
                        kv_elements * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_value, host_value.data(),
                        kv_elements * sizeof(__half), cudaMemcpyHostToDevice));

  const dim3 baseline_grid(args.groups * 2);
  const dim3 partial_grid(args.groups * 2, kSplitParts);
  const dim3 merge_grid(args.groups * 2);
  auto launch_baseline = [&] {
    sm70_native_bm32_splitkv_unsplit_baseline<<<baseline_grid, kBm32Threads>>>(
        device_query, device_key, device_value, device_baseline, args.groups,
        args.nblocks);
    CUDA_CHECK(cudaGetLastError());
  };
  auto launch_partial = [&] {
    sm70_native_bm32_splitkv_partial<<<partial_grid, kBm32Threads>>>(
        device_query, device_key, device_value, device_partial_accumulator,
        device_partial_row_max, device_partial_row_sum, args.groups,
        args.nblocks);
    CUDA_CHECK(cudaGetLastError());
  };
  auto launch_merge = [&] {
    sm70_native_bm32_splitkv_merge<<<merge_grid, kMergeThreads>>>(
        device_partial_accumulator, device_partial_row_max,
        device_partial_row_sum, device_split, args.groups);
    CUDA_CHECK(cudaGetLastError());
  };
  auto launch_combined = [&] {
    launch_partial();
    launch_merge();
  };
  auto free_device_buffers = [&] {
    CUDA_CHECK(cudaFree(device_partial_row_sum));
    CUDA_CHECK(cudaFree(device_partial_row_max));
    CUDA_CHECK(cudaFree(device_partial_accumulator));
    CUDA_CHECK(cudaFree(device_split));
    CUDA_CHECK(cudaFree(device_baseline));
    CUDA_CHECK(cudaFree(device_value));
    CUDA_CHECK(cudaFree(device_key));
    CUDA_CHECK(cudaFree(device_query));
  };

  if (args.profile_only) {
    if (args.profile_kernel == "merge") {
      launch_partial();
    }
    if (profile_kernel_selected(args, "baseline")) {
      launch_baseline();
    }
    if (profile_kernel_selected(args, "partial")) {
      launch_partial();
    }
    if (profile_kernel_selected(args, "merge")) {
      launch_merge();
    }
    if (profile_kernel_selected(args, "combined")) {
      launch_combined();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    free_device_buffers();
    print_json(args, properties, runtime_version, sm_count, baseline_resources,
               partial_resources, merge_resources, true, false, no_comparison,
               false, no_timing, no_timing, no_timing, no_timing, no_pairs);
    return EXIT_SUCCESS;
  }

  for (int warmup = 0; warmup < args.warmup; ++warmup) {
    if ((warmup & 1) == 0) {
      launch_baseline();
      launch_combined();
    } else {
      launch_combined();
      launch_baseline();
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  launch_baseline();
  launch_combined();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__half> host_baseline(output_elements);
  std::vector<__half> host_split(output_elements);
  CUDA_CHECK(cudaMemcpy(host_baseline.data(), device_baseline,
                        output_elements * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_split.data(), device_split,
                        output_elements * sizeof(__half), cudaMemcpyDeviceToHost));
  const Comparison comparison = compare_outputs(host_baseline, host_split);
  const bool comparison_pass = comparison.all_finite &&
                               comparison.max_abs_error <= kComparisonAbsTolerance;
  if (!comparison_pass) {
    free_device_buffers();
    print_json(args, properties, runtime_version, sm_count, baseline_resources,
               partial_resources, merge_resources, true, true, comparison, false,
               no_timing, no_timing, no_timing, no_timing, no_pairs);
    return EXIT_FAILURE;
  }
  if (args.smoke_only) {
    free_device_buffers();
    print_json(args, properties, runtime_version, sm_count, baseline_resources,
               partial_resources, merge_resources, true, true, comparison, false,
               no_timing, no_timing, no_timing, no_timing, no_pairs);
    return EXIT_SUCCESS;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  auto time_launches = [&](const auto& launch) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int launch_index = 0; launch_index < args.launches_per_sample;
         ++launch_index) {
      launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    return static_cast<double>(elapsed_ms) * 1000.0 /
           args.launches_per_sample;
  };

  std::vector<double> baseline_samples;
  std::vector<double> partial_samples;
  std::vector<double> merge_samples;
  std::vector<double> combined_samples;
  baseline_samples.reserve(args.rounds);
  partial_samples.reserve(args.rounds);
  merge_samples.reserve(args.rounds);
  combined_samples.reserve(args.rounds);
  for (int round = 0; round < args.rounds; ++round) {
    if ((round & 1) == 0) {
      baseline_samples.push_back(time_launches(launch_baseline));
      partial_samples.push_back(time_launches(launch_partial));
      merge_samples.push_back(time_launches(launch_merge));
      combined_samples.push_back(time_launches(launch_combined));
    } else {
      combined_samples.push_back(time_launches(launch_combined));
      merge_samples.push_back(time_launches(launch_merge));
      partial_samples.push_back(time_launches(launch_partial));
      baseline_samples.push_back(time_launches(launch_baseline));
    }
  }
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));

  const TimingSummary baseline_timing = summarize(baseline_samples);
  const TimingSummary partial_timing = summarize(partial_samples);
  const TimingSummary merge_timing = summarize(merge_samples);
  const TimingSummary combined_timing = summarize(combined_samples);
  const PairSummary pairs = summarize_pairs(baseline_samples, combined_samples);
  free_device_buffers();
  print_json(args, properties, runtime_version, sm_count, baseline_resources,
             partial_resources, merge_resources, true, true, comparison, true,
             baseline_timing, partial_timing, merge_timing, combined_timing,
             pairs);
  return EXIT_SUCCESS;
}

void print_usage(const char* program) {
  std::cerr << "Usage: " << program
            << " [--device N] [--groups N] [--nblocks N] [--warmup N]"
               " [--rounds N] [--launches N|--launches-per-sample N]"
               " [--pattern random|alternating] [--profile-only] [--smoke-only]"
               " [--profile-kernel baseline|partial|merge|combined|all]\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    auto parse_int = [&](int* destination) {
      if (index + 1 >= argc) {
        std::cerr << "Missing value for " << argument << '\n';
        std::exit(EXIT_FAILURE);
      }
      *destination = std::stoi(argv[++index]);
    };
    if (argument == "--device") {
      parse_int(&args.device);
    } else if (argument == "--groups") {
      parse_int(&args.groups);
    } else if (argument == "--nblocks") {
      parse_int(&args.nblocks);
    } else if (argument == "--warmup") {
      parse_int(&args.warmup);
    } else if (argument == "--rounds") {
      parse_int(&args.rounds);
    } else if (argument == "--launches" ||
               argument == "--launches-per-sample") {
      parse_int(&args.launches_per_sample);
    } else if (argument == "--profile-only") {
      args.profile_only = true;
    } else if (argument == "--smoke-only") {
      args.smoke_only = true;
    } else if (argument == "--profile-kernel" && index + 1 < argc) {
      args.profile_kernel = argv[++index];
      if (args.profile_kernel != "baseline" &&
          args.profile_kernel != "partial" && args.profile_kernel != "merge" &&
          args.profile_kernel != "combined" && args.profile_kernel != "all") {
        std::cerr
            << "--profile-kernel must be baseline, partial, merge, combined, or all\n";
        std::exit(EXIT_FAILURE);
      }
    } else if (argument == "--pattern" && index + 1 < argc) {
      args.pattern = argv[++index];
    } else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      print_usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  return run(parse_args(argc, argv));
}
