// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
constexpr int kBaselineThreads = 512;
constexpr int kCandidateThreads = 384;
constexpr int kQPanelElements = kPanelM * kD;
constexpr int kQElements = kM * kD;
constexpr int kM64QElements = kM64 * kD;
constexpr int kPPanelElements = kPanelM * kSoftmaxPanelN;
constexpr int kOutputElements = kM * kD;
constexpr int kM64OutputElements = kM64 * kD;
constexpr float kNegativeInfinity = -1e30f;

static_assert(kM == 2 * kPanelM);
static_assert(kM64 == 4 * kPanelM);
static_assert(kBlockN == kQKWarps * kPanelN);
static_assert(kSoftmaxPanelN == 2 * kPanelK);
static_assert(kBaselineThreads / 32 == 2 * kQKWarps);
static_assert(kCandidateThreads / 32 == 3 * (kM64 / kPanelM));

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

// Four query-owner warps retain Q plus per-warp QK/P scratch in shared memory
// and their complete D=256 output in registers.
struct alignas(16) M64Shared {
  __half query[kM64QElements];
  __half probability[kM64 / kPanelM][kPPanelElements];
  float row_max[kM64];
  float row_sum[kM64];
  float row_exp_diff[kM64];
};

static_assert(sizeof(M64Shared) == 37632,
              "query-owner M64 storage changed unexpectedly");
static_assert(sizeof(M64Shared) <= 48 * 1024,
              "query-owner M64 storage exceeds the two-CTA gate");

struct Args {
  int device = 0;
  int groups = 72;
  int nblocks = 4;
  int warmup = 20;
  int rounds = 100;
  int launches_per_sample = 8;
  bool profile_only = false;
  bool smoke_only = false;
  std::string profile_kernel = "both";
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
  int candidate_faster = 0;
  int baseline_faster = 0;
  int ties = 0;
  double candidate_minus_baseline_median_us = 0.0;
  double candidate_minus_baseline_mean_us = 0.0;
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

struct Exactness {
  bool bitwise_equal = false;
  int mismatch_words = 0;
  uint32_t xor_reduction = 0;
  uint32_t max_word_xor = 0;
  float max_abs_error = 0.0f;
  int64_t first_mismatch_element = -1;
  uint16_t first_baseline_bits = 0;
  uint16_t first_candidate_bits = 0;
  float first_baseline_value = 0.0f;
  float first_candidate_value = 0.0f;
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
        (warp_in_pair * 16 + element_pair * 2 + lane_row) * kBlockN
        + pair_column;
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
        (warp_in_pair * 16 + element_pair * 2 + lane_row) * kBlockN
        + pair_column;
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

extern "C" __global__ __launch_bounds__(kBaselineThreads, 2)
void sm70_native_bm64_allp_bm32_pair_scratch_baseline(
    const __half* __restrict__ query, const __half* __restrict__ key,
    const __half* __restrict__ value, __half* __restrict__ output, int groups,
    int nblocks) {
  __shared__ __align__(16) BM32PairScratchShared shared;

  const int block = blockIdx.x;
  const int group = block >> 1;
  if (group >= groups) {
    return;
  }
  const int m32_panel = block & 1;
  {
    const __half* query_group =
        query + static_cast<int64_t>(group) * kM64QElements +
        m32_panel * kQElements;
    stage_swizzled_q_panel(query_group, shared.query, threadIdx.x,
                           kBaselineThreads);
    stage_swizzled_q_panel(query_group + kQPanelElements,
                            shared.query + kQPanelElements, threadIdx.x,
                            kBaselineThreads);
  }
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
      const __half* key_group = key + static_cast<int64_t>(group) * nblocks *
                                        kBlockN * kD;
      spill_qk_warp_pv_accumulators(shared.score, accumulator_top,
                                    accumulator_bottom);
      asm volatile("" ::: "memory");
      AccumulatorFragment qk_top;
      AccumulatorFragment qk_bottom;
      qk_pair_accumulate(shared.query, shared.query + kQPanelElements,
                         key_group +
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
      // Make lane 0's online-softmax state visible to this warp's next panel.
      __syncwarp();
    }
    __syncthreads();

    const int d_offset = (threadIdx.x >> 5) * kPanelN;
    const __half* value_group =
        value + static_cast<int64_t>(group) * nblocks * kBlockN * kD;
#pragma unroll
    for (int panel = 0; panel < kSoftmaxPanelsPerBlock; ++panel) {
      if (shared.block_index != 0 || panel != 0) {
        scale_phase_reuse_accumulators(accumulator_top, accumulator_bottom,
                                       shared.row_exp_diff[panel]);
      }
      update_phase_reuse_pv_panel(
          shared.probability_top[panel], shared.probability_bottom[panel],
          value_group +
              (shared.block_index * kBlockN + panel * kSoftmaxPanelN) * kD,
          d_offset, accumulator_top, accumulator_bottom);
    }
    // All warps must finish reading the current block index and V panels
    // before thread 0 advances the shared loop state.
    __syncthreads();
    if (threadIdx.x == 0) {
      ++shared.block_index;
    }
    __syncthreads();
  }

  __half* output_group =
      output + static_cast<int64_t>(group) * kM64OutputElements +
      m32_panel * kOutputElements;
  const int d_offset = (threadIdx.x >> 5) * kPanelN;
  store_accumulator_output(accumulator_top, output_group, shared.row_sum, 0,
                           d_offset);
  store_accumulator_output(accumulator_bottom, output_group, shared.row_sum,
                           kPanelM, d_offset);
}

__device__ __forceinline__ void q_owner_qk_pair_accumulate(
    const __half* __restrict__ shared_query,
    const __half* __restrict__ key_panel,
    AccumulatorFragment& qk_0,
    AccumulatorFragment& qk_1) {
  MatrixAFragment a_fragment;
  QKMatrixBFragment b_fragment;
  nvcuda::wmma::fill_fragment(qk_0, 0.0f);
  nvcuda::wmma::fill_fragment(qk_1, 0.0f);

#pragma unroll
  for (int k_offset = 0; k_offset < kD; k_offset += kPanelK) {
    load_swizzled_matrix_a_fragment(a_fragment, shared_query, k_offset);
    nvcuda::wmma::load_matrix_sync(b_fragment, key_panel + k_offset, kD);
    nvcuda::wmma::mma_sync(qk_0, a_fragment, b_fragment, qk_0);
    nvcuda::wmma::load_matrix_sync(
        b_fragment, key_panel + kPanelN * kD + k_offset, kD);
    nvcuda::wmma::mma_sync(qk_1, a_fragment, b_fragment, qk_1);
  }
}

template <bool MAX_REDUCTION>
__device__ __forceinline__ float q_owner_reduce_pair(
    float left, float right) {
  if constexpr (MAX_REDUCTION) {
    return fmaxf(left, right);
  }
  return left + right;
}

template <bool MAX_REDUCTION>
__device__ __forceinline__ float q_owner_reduce_panel_row(
    const AccumulatorFragment& fragment_0,
    const AccumulatorFragment& fragment_1,
    int element_base) {
  constexpr unsigned kMask = 0xffffffffU;
  const int lane = threadIdx.x & 31;
  const int canonical_lane = lane & ~0xa;

  const float fragment_0_low = q_owner_reduce_pair<MAX_REDUCTION>(
      fragment_0.x[element_base], fragment_0.x[element_base + 1]);
  const float fragment_0_high = q_owner_reduce_pair<MAX_REDUCTION>(
      fragment_0.x[element_base + 4], fragment_0.x[element_base + 5]);
  const float fragment_1_low = q_owner_reduce_pair<MAX_REDUCTION>(
      fragment_1.x[element_base], fragment_1.x[element_base + 1]);
  const float fragment_1_high = q_owner_reduce_pair<MAX_REDUCTION>(
      fragment_1.x[element_base + 4], fragment_1.x[element_base + 5]);

  const float quartet_0_low = q_owner_reduce_pair<MAX_REDUCTION>(
      fragment_0_low, __shfl_xor_sync(kMask, fragment_0_low, 2));
  const float quartet_0_high = q_owner_reduce_pair<MAX_REDUCTION>(
      fragment_0_high, __shfl_xor_sync(kMask, fragment_0_high, 2));
  const float quartet_1_low = q_owner_reduce_pair<MAX_REDUCTION>(
      fragment_1_low, __shfl_xor_sync(kMask, fragment_1_low, 2));
  const float quartet_1_high = q_owner_reduce_pair<MAX_REDUCTION>(
      fragment_1_high, __shfl_xor_sync(kMask, fragment_1_high, 2));

  const float even_group = q_owner_reduce_pair<MAX_REDUCTION>(
      quartet_0_low, quartet_1_low);
  const float odd_group = q_owner_reduce_pair<MAX_REDUCTION>(
      quartet_0_high, quartet_1_high);
  const float even_0 = __shfl_sync(kMask, even_group, canonical_lane);
  const float even_2 =
      __shfl_sync(kMask, even_group, canonical_lane | 8);
  const float odd_0 = __shfl_sync(kMask, odd_group, canonical_lane);
  const float odd_2 = __shfl_sync(kMask, odd_group, canonical_lane | 8);
  return q_owner_reduce_pair<MAX_REDUCTION>(
      q_owner_reduce_pair<MAX_REDUCTION>(even_0, even_2),
      q_owner_reduce_pair<MAX_REDUCTION>(odd_0, odd_2));
}

__device__ __forceinline__ void q_owner_write_probability(
    AccumulatorFragment& fragment,
    __half* __restrict__ probability,
    int column_offset,
    float first_row_max,
    float second_row_max) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int element = 0; element < fragment.num_elements; ++element) {
    const int row = accumulator_fragment_row(lane, element);
    const int column =
        column_offset + accumulator_fragment_column(lane, element);
    const float row_max = (element & 2) == 0 ? first_row_max : second_row_max;
    const float value =
        __expf(fmaxf(fragment.x[element] - row_max, -80.0f));
    fragment.x[element] = value;
    probability[swizzled_matrix_a_offset(row, column)] =
        __float2half_rn(value);
  }
}

__device__ __forceinline__ void q_owner_softmax(
    AccumulatorFragment& qk_0,
    AccumulatorFragment& qk_1,
    __half* __restrict__ probability,
    float& first_row_max,
    float& first_row_sum,
    float& second_row_max,
    float& second_row_sum,
    float& first_row_exp_diff,
    float& second_row_exp_diff) {
  const float panel_max_first =
      q_owner_reduce_panel_row<true>(qk_0, qk_1, 0);
  const float panel_max_second =
      q_owner_reduce_panel_row<true>(qk_0, qk_1, 2);
  const float new_max_first = fmaxf(first_row_max, panel_max_first);
  const float new_max_second = fmaxf(second_row_max, panel_max_second);
  first_row_exp_diff = __expf(first_row_max - new_max_first);
  second_row_exp_diff = __expf(second_row_max - new_max_second);

  q_owner_write_probability(
      qk_0, probability, 0, new_max_first, new_max_second);
  q_owner_write_probability(
      qk_1, probability, kPanelN, new_max_first, new_max_second);
  const float panel_sum_first =
      q_owner_reduce_panel_row<false>(qk_0, qk_1, 0);
  const float panel_sum_second =
      q_owner_reduce_panel_row<false>(qk_0, qk_1, 2);
  first_row_sum = first_row_exp_diff * first_row_sum + panel_sum_first;
  second_row_sum = second_row_exp_diff * second_row_sum + panel_sum_second;
  first_row_max = new_max_first;
  second_row_max = new_max_second;
}

__device__ __forceinline__ void q_owner_store_output_tile(
    const AccumulatorFragment& accumulator,
    __half* __restrict__ output,
    int d_offset,
    float first_row_sum,
    float second_row_sum) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int element = 0; element < accumulator.num_elements; ++element) {
    const int row = accumulator_fragment_row(lane, element);
    const int column =
        d_offset + accumulator_fragment_column(lane, element);
    const float row_sum =
        (element & 2) == 0 ? first_row_sum : second_row_sum;
    const float inverse_sum = 1.0f / fmaxf(row_sum, 1e-24f);
    output[row * kD + column] =
        __float2half_rn(accumulator.x[element] * inverse_sum);
  }
}

__device__ __forceinline__ void q_owner_sync_group(int query_owner) {
  const int barrier_id = query_owner + 1;
  asm volatile("bar.sync %0, 96;" : : "r"(barrier_id) : "memory");
}

#define Q_OWNER_D_TILES(OP) \
  OP(0)                     \
  OP(1)                     \
  OP(2)                     \
  OP(3)                     \
  OP(4)                     \
  OP(5)                     \
  OP(6)                     \
  OP(7)

extern "C" __global__ __launch_bounds__(kCandidateThreads, 2)
void sm70_native_bm64_allp_m64_candidate(
    const __half* __restrict__ query, const __half* __restrict__ key,
    const __half* __restrict__ value, __half* __restrict__ output, int groups,
    int nblocks) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  M64Shared& shared = *reinterpret_cast<M64Shared*>(shared_bytes);

  const int group = blockIdx.x;
  if (group >= groups) {
    return;
  }
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int query_owner = warp / 3;
  const int owner_role = warp - query_owner * 3;
  const int owner_thread = owner_role * 32 + lane;
  const __half* query_panel =
      query + static_cast<int64_t>(group) * kM64QElements
      + query_owner * kQPanelElements;
  __half* shared_query = shared.query + query_owner * kQPanelElements;
  __half* probability = shared.probability[query_owner];
  stage_swizzled_q_panel(query_panel, shared_query, owner_thread, 96);
  if (owner_thread < kPanelM) {
    const int state_row = query_owner * kPanelM + owner_thread;
    shared.row_max[state_row] = kNegativeInfinity;
    shared.row_sum[state_row] = 0.0f;
  }
  q_owner_sync_group(query_owner);

  if (owner_role == 0) {
    float first_row_max = kNegativeInfinity;
    float second_row_max = kNegativeInfinity;
    float first_row_sum = 0.0f;
    float second_row_sum = 0.0f;
    const __half* key_group =
        key + static_cast<int64_t>(group) * nblocks * kBlockN * kD;
    for (int block = 0; block < nblocks; ++block) {
#pragma unroll
      for (int panel = 0; panel < kSoftmaxPanelsPerBlock; ++panel) {
        const int panel_token = block * kBlockN + panel * kSoftmaxPanelN;
        AccumulatorFragment qk_0;
        AccumulatorFragment qk_1;
        q_owner_qk_pair_accumulate(
            shared_query, key_group + panel_token * kD, qk_0, qk_1);
        float first_row_exp_diff;
        float second_row_exp_diff;
        q_owner_softmax(
            qk_0, qk_1, probability,
            first_row_max, first_row_sum,
            second_row_max, second_row_sum,
            first_row_exp_diff, second_row_exp_diff);
        if ((lane & 0xa) == 0) {
          const int first_row = accumulator_fragment_row(lane, 0);
          const int first_state_row = query_owner * kPanelM + first_row;
          const int second_state_row = first_state_row + 2;
          shared.row_exp_diff[first_state_row] = first_row_exp_diff;
          shared.row_exp_diff[second_state_row] = second_row_exp_diff;
          shared.row_max[first_state_row] = first_row_max;
          shared.row_max[second_state_row] = second_row_max;
          shared.row_sum[first_state_row] = first_row_sum;
          shared.row_sum[second_state_row] = second_row_sum;
        }
        q_owner_sync_group(query_owner);
        q_owner_sync_group(query_owner);
      }
    }
  } else {
    const int d_half = owner_role - 1;
#define Q_OWNER_DECLARE_ACCUMULATOR(INDEX) \
  AccumulatorFragment accumulator_##INDEX;
    Q_OWNER_D_TILES(Q_OWNER_DECLARE_ACCUMULATOR)
#undef Q_OWNER_DECLARE_ACCUMULATOR
#define Q_OWNER_FILL_ACCUMULATOR(INDEX) \
  nvcuda::wmma::fill_fragment(accumulator_##INDEX, 0.0f);
    Q_OWNER_D_TILES(Q_OWNER_FILL_ACCUMULATOR)
#undef Q_OWNER_FILL_ACCUMULATOR

    const __half* value_group =
        value + static_cast<int64_t>(group) * nblocks * kBlockN * kD;
    for (int block = 0; block < nblocks; ++block) {
#pragma unroll
      for (int panel = 0; panel < kSoftmaxPanelsPerBlock; ++panel) {
        const int panel_token = block * kBlockN + panel * kSoftmaxPanelN;
        q_owner_sync_group(query_owner);
        if (block != 0 || panel != 0) {
          const int first_row = accumulator_fragment_row(lane, 0);
          const float first_row_exp_diff =
              shared.row_exp_diff[query_owner * kPanelM + first_row];
          const float second_row_exp_diff =
              shared.row_exp_diff[query_owner * kPanelM + first_row + 2];
#define Q_OWNER_SCALE_ACCUMULATOR(INDEX)                         \
  scale_accumulator_two_rows(accumulator_##INDEX,                \
                             first_row_exp_diff, second_row_exp_diff);
          Q_OWNER_D_TILES(Q_OWNER_SCALE_ACCUMULATOR)
#undef Q_OWNER_SCALE_ACCUMULATOR
        }
        const __half* value_panel = value_group + panel_token * kD;
        MatrixAFragment probability_fragment;
        PVMatrixBFragment value_fragment;
        load_swizzled_matrix_a_fragment(
            probability_fragment, probability, 0);
#define Q_OWNER_UPDATE_PV_0(INDEX)                                      \
  nvcuda::wmma::load_matrix_sync(                                      \
      value_fragment,                                                  \
      value_panel + (d_half * 8 + INDEX) * kPanelN, kD);                \
  nvcuda::wmma::mma_sync(accumulator_##INDEX, probability_fragment,     \
                         value_fragment, accumulator_##INDEX);
        Q_OWNER_D_TILES(Q_OWNER_UPDATE_PV_0)
#undef Q_OWNER_UPDATE_PV_0
        load_swizzled_matrix_a_fragment(
            probability_fragment, probability, kPanelK);
#define Q_OWNER_UPDATE_PV_1(INDEX)                                      \
  nvcuda::wmma::load_matrix_sync(                                      \
      value_fragment,                                                  \
      value_panel + kPanelK * kD                                       \
          + (d_half * 8 + INDEX) * kPanelN, kD);                       \
  nvcuda::wmma::mma_sync(accumulator_##INDEX, probability_fragment,     \
                         value_fragment, accumulator_##INDEX);
        Q_OWNER_D_TILES(Q_OWNER_UPDATE_PV_1)
#undef Q_OWNER_UPDATE_PV_1
        q_owner_sync_group(query_owner);
      }
    }

    __half* output_panel =
        output + static_cast<int64_t>(group) * kM64OutputElements
        + query_owner * kQPanelElements;
    const int first_row = accumulator_fragment_row(lane, 0);
    const float output_first_row_sum =
        shared.row_sum[query_owner * kPanelM + first_row];
    const float output_second_row_sum =
        shared.row_sum[query_owner * kPanelM + first_row + 2];
#define Q_OWNER_STORE_ACCUMULATOR(INDEX)                              \
  q_owner_store_output_tile(accumulator_##INDEX, output_panel,        \
                            (d_half * 8 + INDEX) * kPanelN,            \
                            output_first_row_sum,                      \
                            output_second_row_sum);
    Q_OWNER_D_TILES(Q_OWNER_STORE_ACCUMULATOR)
#undef Q_OWNER_STORE_ACCUMULATOR
  }
}

#undef Q_OWNER_D_TILES

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

uint32_t half_word_bits(const __half* values, size_t word_index) {
  uint32_t bits = 0;
  std::memcpy(&bits, values + 2 * word_index, sizeof(bits));
  return bits;
}

uint16_t half_bits(const __half* values, size_t index) {
  uint16_t bits = 0;
  std::memcpy(&bits, values + index, sizeof(bits));
  return bits;
}

Exactness compare_outputs(const std::vector<__half>& baseline,
                          const std::vector<__half>& candidate) {
  Exactness result;
  result.bitwise_equal = true;
  const size_t word_count = baseline.size() / 2;
  for (size_t word = 0; word < word_count; ++word) {
    const uint32_t word_xor =
        half_word_bits(baseline.data(), word) ^
        half_word_bits(candidate.data(), word);
    result.xor_reduction ^= word_xor;
    result.max_word_xor = std::max(result.max_word_xor, word_xor);
    result.bitwise_equal &= word_xor == 0;
    result.mismatch_words += word_xor != 0;
    if (word_xor != 0 && result.first_mismatch_element < 0) {
      const size_t first = 2 * word;
      const size_t index = half_bits(baseline.data(), first) !=
                                   half_bits(candidate.data(), first)
                               ? first
                               : first + 1;
      result.first_mismatch_element = static_cast<int64_t>(index);
      result.first_baseline_bits = half_bits(baseline.data(), index);
      result.first_candidate_bits = half_bits(candidate.data(), index);
      result.first_baseline_value = __half2float(baseline[index]);
      result.first_candidate_value = __half2float(candidate[index]);
    }
  }
  for (size_t index = 0; index < baseline.size(); ++index) {
    result.max_abs_error =
        std::max(result.max_abs_error,
                 std::fabs(__half2float(baseline[index]) -
                           __half2float(candidate[index])));
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
                            const std::vector<double>& candidate) {
  std::vector<double> deltas;
  deltas.reserve(baseline.size());
  PairSummary result;
  result.count = static_cast<int>(baseline.size());
  for (size_t index = 0; index < baseline.size(); ++index) {
    const double delta = candidate[index] - baseline[index];
    deltas.push_back(delta);
    if (candidate[index] < baseline[index]) {
      ++result.candidate_faster;
    } else if (baseline[index] < candidate[index]) {
      ++result.baseline_faster;
    } else {
      ++result.ties;
    }
  }
  const TimingSummary summary = summarize(deltas);
  result.candidate_minus_baseline_median_us = summary.median_us;
  result.candidate_minus_baseline_mean_us = summary.mean_us;
  return result;
}

template <typename Kernel>
KernelResources query_resources(Kernel kernel, int threads_per_cta,
                                size_t dynamic_shared_bytes = 0) {
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int active_ctas = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_ctas, kernel, threads_per_cta, dynamic_shared_bytes));
  KernelResources result;
  result.registers_per_thread = attributes.numRegs;
  result.static_shared_bytes = attributes.sharedSizeBytes;
  result.dynamic_shared_bytes = dynamic_shared_bytes;
  result.local_bytes_per_thread = attributes.localSizeBytes;
  result.active_ctas_per_sm = active_ctas;
  result.threads_per_cta = threads_per_cta;
  result.warps_per_cta = threads_per_cta / 32;
  result.resident_warps = result.active_ctas_per_sm * result.warps_per_cta;
  return result;
}

bool bm32_resource_gate(const KernelResources& resources) {
  return resources.registers_per_thread == 64 &&
         resources.local_bytes_per_thread == 0 &&
         resources.static_shared_bytes == sizeof(BM32PairScratchShared) &&
         resources.active_ctas_per_sm == 2;
}

bool m64_resource_gate(const KernelResources& resources) {
  return resources.registers_per_thread <= 85 &&
         resources.local_bytes_per_thread == 0 &&
         resources.dynamic_shared_bytes == sizeof(M64Shared) &&
         resources.static_shared_bytes <= 48 * 1024 &&
         resources.active_ctas_per_sm == 2;
}

void configure_m64_shared_memory() {
  CUDA_CHECK(cudaFuncSetAttribute(sm70_native_bm64_allp_m64_candidate,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  sizeof(M64Shared)));
  CUDA_CHECK(cudaFuncSetAttribute(
      sm70_native_bm64_allp_m64_candidate,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      static_cast<int>(cudaSharedmemCarveoutMaxShared)));
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

void print_exactness(const Exactness& exactness, int64_t word_count) {
  std::cout << "{\"word_dtype\": \"uint32 packed fp16\", \"word_count\": "
            << word_count << ", \"full_output\": true, \"bitwise_equal\": "
            << (exactness.bitwise_equal ? "true" : "false")
            << ", \"mismatch_words\": " << exactness.mismatch_words
            << ", \"xor\": {\"max_word\": " << exactness.max_word_xor
            << ", \"reduction\": " << exactness.xor_reduction
            << "}, \"max_abs_error\": " << exactness.max_abs_error
            << ", \"first_difference\": ";
  if (exactness.first_mismatch_element < 0) {
    std::cout << "null";
  } else {
    const int64_t element = exactness.first_mismatch_element;
    const int64_t group = element / kM64OutputElements;
    const int64_t in_group = element % kM64OutputElements;
    std::cout << "{\"element\": " << element << ", \"group\": "
              << group << ", \"row\": " << in_group / kD
              << ", \"column\": " << in_group % kD
              << ", \"baseline_bits\": " << exactness.first_baseline_bits
              << ", \"candidate_bits\": "
              << exactness.first_candidate_bits
              << ", \"baseline_value\": "
              << exactness.first_baseline_value
              << ", \"candidate_value\": "
              << exactness.first_candidate_value
              << ", \"reason\": \"per-row FP32 QK, online-softmax, or PV "
                 "accumulation order diverged before fp16 rounding\"}";
  }
  std::cout << '}';
}

void print_json(const Args& args, const cudaDeviceProp& properties,
                int runtime_version, int sm_count,
                const KernelResources& baseline_resources,
                const KernelResources& candidate_resources, bool executed,
                bool exactness_available, const Exactness& exactness,
                bool timing_available, const TimingSummary& baseline_timing,
                const TimingSummary& candidate_timing,
                const PairSummary& pairs) {
  const bool resource_pass = bm32_resource_gate(baseline_resources) &&
                             m64_resource_gate(candidate_resources);
  const double candidate_speedup_pct =
      timing_available
          ? 100.0 * (baseline_timing.median_us - candidate_timing.median_us) /
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
  std::cout << "  \"shape\": {\"groups\": " << args.groups
            << ", \"nblocks\": " << args.nblocks
            << ", \"M\": 64, \"D\": 256, \"BN\": 128, \"N\": "
            << args.nblocks * kBlockN << "},\n";
  std::cout << "  \"input_pattern\": ";
  print_json_string(args.pattern);
  std::cout << ",\n";
  std::cout << "  \"layouts\": {\"q\": \"[group,M64,D256] fp16\", "
               "\"k_v\": \"[group,N,D256] contiguous fp16\", "
               "\"output\": \"[group,M64,D256] fp16\"},\n";
  std::cout << "  \"paths\": {\n";
  std::cout << "    \"baseline\": \"2 CTA/M64 group; BM32, 512 threads; "
               "accepted ALL_P=true PAIR_SCRATCH=true path\",\n";
  std::cout << "    \"candidate\": \"1 CTA/M64 group; BM64, 384 threads; "
               "four M16 producer/two-consumer warp groups\",\n";
  std::cout << "    \"candidate_qk\": \"one producer warp per owner "
               "computes both N16 fragments for each N32 panel\",\n";
  std::cout << "    \"candidate_softmax\": \"each owner uses private "
               "score/state shared memory; synchronization is warp-local\",\n";
  std::cout << "    \"candidate_pv\": \"two warps per owner each retain "
               "eight D16 FP32 output fragments\",\n";
  std::cout << "    \"cross_block_overlap\": \"disabled\",\n";
  std::cout << "    \"score_buffers\": 1,\n";
  std::cout << "    \"probability_panels\": 4,\n";
  std::cout << "    \"baseline_shared_layout_bytes\": "
            << sizeof(BM32PairScratchShared) << ",\n";
  std::cout << "    \"candidate_shared_layout_bytes\": " << sizeof(M64Shared)
            << "\n";
  std::cout << "  },\n";
  std::cout << "  \"baseline_pair_scratch\": {\"storage\": \"shared.score\", "
               "\"ownership\": \"each QK warp pair owns one disjoint 32-column score slab\", "
               "\"row\": \"16*warp_in_pair+2*element_pair+lane/16 (+8 for bottom)\", "
               "\"column\": \"32*warp_pair+2*(lane%16)\", "
               "\"minimum_bank_replay\": 2, "
               "\"handoff\": \"one 64-thread named barrier per QK warp pair\", "
               "\"qk_warps\": 8, \"fragments_per_warp\": 2, "
               "\"shared_v2_spills_per_fragment\": 4, "
               "\"shared_v2_reloads_per_fragment\": 4, "
               "\"shared_v2_spills_per_qk_warp\": 8, "
               "\"shared_v2_reloads_per_qk_warp\": 8, "
               "\"shared_v2_spills_per_cta\": 64, "
               "\"shared_v2_reloads_per_cta\": 64},\n";
  std::cout << "  \"candidate_storage\": {\"q_persistent_across_kv_blocks\": true, "
               "\"qk_scores_in_registers\": true, "
               "\"cta_barriers\": 0, "
               "\"q_bytes\": "
            << kM64QElements * sizeof(__half)
            << ", \"score_bytes\": 0"
            << ", \"probability_bytes\": "
            << (kM64 / kPanelM) * kPPanelElements * sizeof(__half)
            << "},\n";
  std::cout << "  \"execution\": {\"profile_only\": "
            << (args.profile_only ? "true" : "false")
            << ", \"smoke_only\": "
            << (args.smoke_only ? "true" : "false")
            << ", \"resource_gate_pass\": "
            << (resource_pass ? "true" : "false")
            << ", \"kernels_executed\": "
            << (executed ? "true" : "false") << "},\n";
  std::cout << "  \"resources\": {\"baseline\": ";
  print_resources(baseline_resources);
  std::cout << ", \"candidate\": ";
  print_resources(candidate_resources);
  std::cout << "},\n";
  std::cout << "  \"resource_gate\": {\"baseline_required_registers_per_thread\": 64, "
               "\"candidate_max_registers_per_thread\": 85, "
               "\"requires_zero_stack_local_spills\": true, "
               "\"candidate_max_static_shared_bytes\": "
            << 48 * 1024
            << ", \"candidate_max_dynamic_shared_bytes\": " << 48 * 1024
            << ", \"baseline_required_active_ctas_per_sm\": 2, "
               "\"candidate_required_active_ctas_per_sm\": 2, "
               "\"runtime_pass\": "
            << (resource_pass ? "true" : "false")
            << ", \"ptxas_validation\": \"compile with --ptxas-options=-v; "
               "require 0 stack frame, 0 spill stores, 0 spill loads\"},\n";
  std::cout << "  \"exactness\": ";
  if (exactness_available) {
    print_exactness(exactness,
                    static_cast<int64_t>(args.groups) * kM64OutputElements /
                        2);
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"timing\": ";
  if (timing_available) {
    std::cout << "{\"unit\": \"us per grid launch\", \"baseline\": ";
    print_timing(baseline_timing);
    std::cout << ", \"candidate\": ";
    print_timing(candidate_timing);
    std::cout << ", \"candidate_speedup_vs_baseline_pct\": "
              << candidate_speedup_pct << '}';
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"pairs\": ";
  if (timing_available) {
    std::cout << "{\"count\": " << pairs.count
              << ", \"candidate_faster\": " << pairs.candidate_faster
              << ", \"baseline_faster\": " << pairs.baseline_faster
              << ", \"ties\": " << pairs.ties
              << ", \"candidate_minus_baseline_median_us\": "
              << pairs.candidate_minus_baseline_median_us
              << ", \"candidate_minus_baseline_mean_us\": "
              << pairs.candidate_minus_baseline_mean_us << '}';
  } else {
    std::cout << "null";
  }
  std::cout << ",\n";
  std::cout << "  \"tradeoff\": {\"baseline_ctas_per_logical_group\": 2, "
               "\"candidate_ctas_per_logical_group\": 1, "
               "\"baseline_qk_k_fragment_reuse\": 2, "
               "\"candidate_qk_k_fragment_reuse\": 2, "
               "\"baseline_pv_v_fragment_reuse\": 2, "
               "\"candidate_pv_v_fragment_reuse\": 4, "
               "\"q_persistent_across_blocks\": true, "
               "\"cross_block_overlap\": \"disabled\", "
               "\"candidate_minus_baseline_median_us\": ";
  if (timing_available) {
    std::cout << pairs.candidate_minus_baseline_median_us
              << ", \"candidate_speedup_pct\": "
              << candidate_speedup_pct
              << ", \"reuse_offsets_lost_occupancy_at_wall_time\": "
              << (candidate_timing.median_us < baseline_timing.median_us
                      ? "true"
                      : "false");
  } else {
    std::cout << "null, \"candidate_speedup_pct\": null, "
                 "\"reuse_offsets_lost_occupancy_at_wall_time\": null";
  }
  std::cout << "},\n";
  std::cout << "  \"measurement\": {\"warmup_pairs\": " << args.warmup
            << ", \"rounds\": " << args.rounds
            << ", \"launches_per_sample\": " << args.launches_per_sample
            << ", \"interleaving\": \"baseline/candidate order alternates each round\"}\n";
  std::cout << "}\n";
}

bool profile_kernel_selected(const Args& args, const char* kernel) {
  return args.profile_kernel == "both" || args.profile_kernel == kernel;
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
  if (args.groups < 1 || args.nblocks < 1 || args.warmup < 0 ||
      args.rounds < 1 || args.launches_per_sample < 1) {
    std::cerr << "groups, nblocks, rounds, and launches must be positive; "
                 "warmup cannot be negative\n";
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
  configure_m64_shared_memory();
  const KernelResources baseline_resources = query_resources(
      sm70_native_bm64_allp_bm32_pair_scratch_baseline, kBaselineThreads);
  const KernelResources candidate_resources = query_resources(
      sm70_native_bm64_allp_m64_candidate, kCandidateThreads,
      sizeof(M64Shared));
  const Exactness no_exactness;
  const TimingSummary no_timing;
  const PairSummary no_pairs;
  if (!bm32_resource_gate(baseline_resources) ||
      !m64_resource_gate(candidate_resources)) {
    print_json(args, properties, runtime_version, sm_count, baseline_resources,
               candidate_resources, false, false, no_exactness, false,
               no_timing, no_timing, no_pairs);
    return EXIT_FAILURE;
  }

  const size_t query_elements =
      static_cast<size_t>(args.groups) * kM64QElements;
  const size_t kv_elements = static_cast<size_t>(args.groups) * args.nblocks *
                             kBlockN * kD;
  const size_t output_elements =
      static_cast<size_t>(args.groups) * kM64OutputElements;
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
  __half* device_candidate = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_query),
                        query_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_key),
                        kv_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_value),
                        kv_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_baseline),
                        output_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_candidate),
                        output_elements * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(device_query, host_query.data(),
                        query_elements * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_key, host_key.data(), kv_elements * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_value, host_value.data(),
                        kv_elements * sizeof(__half), cudaMemcpyHostToDevice));

  const dim3 baseline_grid(args.groups * 2);
  const dim3 candidate_grid(args.groups);
  auto launch_baseline = [&] {
    sm70_native_bm64_allp_bm32_pair_scratch_baseline<<<baseline_grid,
                                                         kBaselineThreads>>>(
        device_query, device_key, device_value, device_baseline, args.groups,
        args.nblocks);
    CUDA_CHECK(cudaGetLastError());
  };
  auto launch_candidate = [&] {
    sm70_native_bm64_allp_m64_candidate<<<candidate_grid,
                                           kCandidateThreads,
                                           sizeof(M64Shared)>>>(
        device_query, device_key, device_value, device_candidate, args.groups,
        args.nblocks);
    CUDA_CHECK(cudaGetLastError());
  };
  auto free_device_buffers = [&] {
    CUDA_CHECK(cudaFree(device_candidate));
    CUDA_CHECK(cudaFree(device_baseline));
    CUDA_CHECK(cudaFree(device_value));
    CUDA_CHECK(cudaFree(device_key));
    CUDA_CHECK(cudaFree(device_query));
  };

  if (args.profile_only) {
    if (profile_kernel_selected(args, "baseline")) {
      launch_baseline();
    }
    if (profile_kernel_selected(args, "candidate")) {
      launch_candidate();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    free_device_buffers();
    print_json(args, properties, runtime_version, sm_count, baseline_resources,
               candidate_resources, true, false, no_exactness, false,
               no_timing, no_timing, no_pairs);
    return EXIT_SUCCESS;
  }

  for (int warmup = 0; warmup < args.warmup; ++warmup) {
    if ((warmup & 1) == 0) {
      launch_baseline();
      launch_candidate();
    } else {
      launch_candidate();
      launch_baseline();
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  launch_baseline();
  launch_candidate();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__half> host_baseline(output_elements);
  std::vector<__half> host_candidate(output_elements);
  CUDA_CHECK(cudaMemcpy(host_baseline.data(), device_baseline,
                        output_elements * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_candidate.data(), device_candidate,
                        output_elements * sizeof(__half), cudaMemcpyDeviceToHost));
  const Exactness exactness = compare_outputs(host_baseline, host_candidate);
  if (!exactness.bitwise_equal) {
    free_device_buffers();
    print_json(args, properties, runtime_version, sm_count, baseline_resources,
               candidate_resources, true, true, exactness, false, no_timing,
               no_timing, no_pairs);
    return EXIT_FAILURE;
  }
  if (args.smoke_only) {
    free_device_buffers();
    print_json(args, properties, runtime_version, sm_count, baseline_resources,
               candidate_resources, true, true, exactness, false, no_timing,
               no_timing, no_pairs);
    return EXIT_SUCCESS;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  auto time_launches = [&](bool candidate) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int launch = 0; launch < args.launches_per_sample; ++launch) {
      if (candidate) {
        launch_candidate();
      } else {
        launch_baseline();
      }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    return static_cast<double>(elapsed_ms) * 1000.0 /
           args.launches_per_sample;
  };

  std::vector<double> baseline_samples;
  std::vector<double> candidate_samples;
  baseline_samples.reserve(args.rounds);
  candidate_samples.reserve(args.rounds);
  for (int round = 0; round < args.rounds; ++round) {
    double baseline_us = 0.0;
    double candidate_us = 0.0;
    if ((round & 1) == 0) {
      baseline_us = time_launches(false);
      candidate_us = time_launches(true);
    } else {
      candidate_us = time_launches(true);
      baseline_us = time_launches(false);
    }
    baseline_samples.push_back(baseline_us);
    candidate_samples.push_back(candidate_us);
  }
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));

  const TimingSummary baseline_timing = summarize(baseline_samples);
  const TimingSummary candidate_timing = summarize(candidate_samples);
  const PairSummary pairs =
      summarize_pairs(baseline_samples, candidate_samples);
  free_device_buffers();
  print_json(args, properties, runtime_version, sm_count, baseline_resources,
             candidate_resources, true, true, exactness, true, baseline_timing,
             candidate_timing, pairs);
  return EXIT_SUCCESS;
}

void print_usage(const char* program) {
  std::cerr << "Usage: " << program
            << " [--device N] [--groups N] [--nblocks N] [--warmup N]"
               " [--rounds N] [--launches N|--launches-per-sample N]"
               " [--pattern random|alternating] [--profile-only] [--smoke-only]"
               " [--profile-kernel baseline|candidate|both]\n";
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
          args.profile_kernel != "candidate" &&
          args.profile_kernel != "both") {
        std::cerr << "--profile-kernel must be baseline, candidate, or both\n";
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
