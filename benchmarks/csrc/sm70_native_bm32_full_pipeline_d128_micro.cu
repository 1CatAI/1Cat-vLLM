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
constexpr int kD = 256;
constexpr int kCandidateD = kD / 2;
constexpr int kBlockN = 128;
constexpr int kPanelM = 16;
constexpr int kPanelN = 16;
constexpr int kPanelK = 16;
constexpr int kSoftmaxPanelN = 32;
constexpr int kSoftmaxPanelsPerBlock = kBlockN / kSoftmaxPanelN;
constexpr int kQKWarps = kBlockN / kPanelN;
constexpr int kBaselinePVWarps = kD / kPanelN;
constexpr int kCandidatePVWarps = kCandidateD / kPanelN;
constexpr int kBaselineThreads = 512;
constexpr int kCandidateThreads = 512;
constexpr int kCandidateCtasPerGroup = 2;
constexpr int kQPanelElements = kPanelM * kD;
constexpr int kQElements = kM * kD;
constexpr int kScoreElements = kPanelM * kBlockN;
constexpr int kPPanelElements = kPanelM * kSoftmaxPanelN;
constexpr int kOutputElements = kM * kD;
constexpr float kNegativeInfinity = -1e30f;
constexpr unsigned int kFullWarpMask = 0xffffffffu;
constexpr int kCounterSpinLimit = 1 << 20;
constexpr uint32_t kProtocolTimeoutScore = 0x10000000u;
constexpr uint32_t kProtocolTimeoutPConsumed = 0x20000000u;
constexpr uint32_t kProtocolTimeoutPReady = 0x30000000u;

static_assert(kM == 2 * kPanelM);
static_assert(kBlockN == kQKWarps * kPanelN);
static_assert(kD == kBaselinePVWarps * kPanelN);
static_assert(kCandidateD == kCandidatePVWarps * kPanelN);
static_assert(kSoftmaxPanelN == 2 * kPanelK);
static_assert(kQKWarps + kCandidatePVWarps == kCandidateThreads / 32);

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

struct alignas(16) BaselineShared {
  __half query[kQPanelElements];
  float score[kScoreElements];
  __half probability[kPPanelElements];
  float output[kPanelM * kD];
  float row_max[kPanelM];
  float row_sum[kPanelM];
};

// The two probability slots are sufficient for the producer/consumer handoff.
// Scores do not need a second buffer because the QK warps finish the current
// block's softmax before any warp overwrites its score columns.
struct alignas(16) CandidateShared {
  __half query[kQElements];
  float score_top[kScoreElements];
  float score_bottom[kScoreElements];
  __half probability[2][2][kPPanelElements];
  float row_max[kM];
  float row_sum[kM];
  float row_exp_diff[2][kM];
  alignas(16) int score_ready;
  int p_ready[2];
  int p_consumed[2];
};

static_assert(sizeof(CandidateShared) <= 48 * 1024,
              "candidate shared storage exceeds the 48 KiB gate");

struct Args {
  int device = 0;
  int groups = 64;
  int nblocks = 4;
  int warmup = 10;
  int rounds = 20;
  int launches_per_sample = 4;
  bool profile_only = false;
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
};

struct ProtocolDiagnostics {
  bool timed_out = false;
  int timeout_ctas = 0;
  int first_cta = -1;
  uint32_t first_code = 0;
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

__device__ __forceinline__ void qk_single_accumulate(
    const __half* __restrict__ shared_query,
    const __half* __restrict__ key_tile, AccumulatorFragment& accumulator) {
  MatrixAFragment a_fragment;
  QKMatrixBFragment b_fragment;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);

#pragma unroll
  for (int k_offset = 0; k_offset < kD; k_offset += kPanelK) {
    load_swizzled_matrix_a_fragment(a_fragment, shared_query, k_offset);
    nvcuda::wmma::load_matrix_sync(b_fragment, key_tile + k_offset, kD);
    nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
  }
}

__device__ __forceinline__ void qk_pair_accumulate(
    const __half* __restrict__ shared_query_top,
    const __half* __restrict__ shared_query_bottom,
    const __half* __restrict__ key_tile, AccumulatorFragment& top_accumulator,
    AccumulatorFragment& bottom_accumulator) {
  MatrixAFragment top_a_fragment;
  MatrixAFragment bottom_a_fragment;
  QKMatrixBFragment b_fragment;
  nvcuda::wmma::fill_fragment(top_accumulator, 0.0f);
  nvcuda::wmma::fill_fragment(bottom_accumulator, 0.0f);

#pragma unroll
  for (int k_offset = 0; k_offset < kD; k_offset += kPanelK) {
    load_swizzled_matrix_a_fragment(top_a_fragment, shared_query_top,
                                    k_offset);
    load_swizzled_matrix_a_fragment(bottom_a_fragment, shared_query_bottom,
                                    k_offset);
    nvcuda::wmma::load_matrix_sync(b_fragment, key_tile + k_offset, kD);
    nvcuda::wmma::mma_sync(top_accumulator, top_a_fragment, b_fragment,
                           top_accumulator);
    nvcuda::wmma::mma_sync(bottom_accumulator, bottom_a_fragment, b_fragment,
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
                        __shfl_down_sync(kFullWarpMask, thread_max, offset));
  }
  const float panel_max = __shfl_sync(kFullWarpMask, thread_max, 0);
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
    thread_sum += __shfl_down_sync(kFullWarpMask, thread_sum, offset);
  }
  const float panel_sum = __shfl_sync(kFullWarpMask, thread_sum, 0);
  if (lane == 0) {
    row_sum[state_row] = exp_diff * row_sum[state_row] + panel_sum;
    row_max[state_row] = new_max;
  }
  return exp_diff;
}

__device__ __forceinline__ void scale_shared_output_row(
    float* __restrict__ output_row, float scale) {
  const int lane = threadIdx.x & 31;
  float4* output_vectors = reinterpret_cast<float4*>(output_row);
#pragma unroll
  for (int vector = lane; vector < kD / 4; vector += 32) {
    float4 value = output_vectors[vector];
    value.x *= scale;
    value.y *= scale;
    value.z *= scale;
    value.w *= scale;
    output_vectors[vector] = value;
  }
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

__device__ __forceinline__ void scale_candidate_accumulators(
    AccumulatorFragment& top_accumulator, AccumulatorFragment& bottom_accumulator,
    const float* __restrict__ row_exp_diff) {
  const int lane = threadIdx.x & 31;
  const int row = accumulator_fragment_row(lane, 0);
  scale_accumulator_two_rows(top_accumulator, row_exp_diff[row],
                             row_exp_diff[row + 2]);
  scale_accumulator_two_rows(bottom_accumulator,
                             row_exp_diff[kPanelM + row],
                             row_exp_diff[kPanelM + row + 2]);
}

__device__ __forceinline__ void update_baseline_pv_panel(
    const __half* __restrict__ probability,
    const __half* __restrict__ value_panel, float* __restrict__ output,
    int d_offset) {
  MatrixAFragment a_fragment;
  PVMatrixBFragment b_fragment;
  AccumulatorFragment accumulator;
  nvcuda::wmma::load_matrix_sync(accumulator, output + d_offset, kD,
                                 nvcuda::wmma::mem_row_major);

#pragma unroll
  for (int k_offset = 0; k_offset < kSoftmaxPanelN; k_offset += kPanelK) {
    load_swizzled_matrix_a_fragment(a_fragment, probability, k_offset);
    nvcuda::wmma::load_matrix_sync(b_fragment,
                                   value_panel + k_offset * kD + d_offset,
                                   kD);
    nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
  }
  nvcuda::wmma::store_matrix_sync(output + d_offset, accumulator, kD,
                                  nvcuda::wmma::mem_row_major);
}

__device__ __forceinline__ void update_candidate_pv_panel(
    const __half* __restrict__ probability_top,
    const __half* __restrict__ probability_bottom,
    const __half* __restrict__ value_panel, int d_offset,
    AccumulatorFragment& top_accumulator,
    AccumulatorFragment& bottom_accumulator) {
  MatrixAFragment a_fragment;
  PVMatrixBFragment b_fragment;

  nvcuda::wmma::load_matrix_sync(b_fragment, value_panel + d_offset, kD);
  load_swizzled_matrix_a_fragment(a_fragment, probability_top, 0);
  nvcuda::wmma::mma_sync(top_accumulator, a_fragment, b_fragment,
                         top_accumulator);
  load_swizzled_matrix_a_fragment(a_fragment, probability_bottom, 0);
  nvcuda::wmma::mma_sync(bottom_accumulator, a_fragment, b_fragment,
                         bottom_accumulator);

  nvcuda::wmma::load_matrix_sync(
      b_fragment, value_panel + kPanelK * kD + d_offset, kD);
  load_swizzled_matrix_a_fragment(a_fragment, probability_top, kPanelK);
  nvcuda::wmma::mma_sync(top_accumulator, a_fragment, b_fragment,
                         top_accumulator);
  load_swizzled_matrix_a_fragment(a_fragment, probability_bottom, kPanelK);
  nvcuda::wmma::mma_sync(bottom_accumulator, a_fragment, b_fragment,
                         bottom_accumulator);
}

__device__ __forceinline__ void produce_candidate_probability_panel(
    CandidateShared* shared, int qk_warp, int panel, int probability_slot) {
#pragma unroll 1
  for (int row_step = 0; row_step < 4; ++row_step) {
    const int row = qk_warp + row_step * kCandidatePVWarps;
    const int local_row = row & (kPanelM - 1);
    const bool top = row < kPanelM;
    const float exp_diff = make_probability_row(
        (top ? shared->score_top : shared->score_bottom) +
            local_row * kBlockN,
        shared->probability[probability_slot][top ? 0 : 1], local_row,
        shared->row_max, shared->row_sum, row, panel);
    if ((threadIdx.x & 31) == 0) {
      shared->row_exp_diff[probability_slot][row] = exp_diff;
    }
  }
}

__device__ __forceinline__ void publish_warp_counter(int* counter) {
  asm volatile("" ::: "memory");
  __syncwarp(kFullWarpMask);
  __threadfence_block();
  __syncwarp(kFullWarpMask);
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(counter, 1);
  }
}

__device__ __forceinline__ uint32_t protocol_error_code(uint32_t phase,
                                                         int generation,
                                                         int warp) {
  return phase |
         ((static_cast<uint32_t>(generation) & 0xffffu) << 8) |
         (static_cast<uint32_t>(warp) & 0xffu);
}

__device__ __forceinline__ bool wait_for_counter(
    int* counter, int expected, uint32_t* protocol_error, uint32_t phase,
    int generation, int warp) {
  const int lane = threadIdx.x & 31;
  int observed = 0;
#pragma unroll 1
  for (int spin = 0; spin < kCounterSpinLimit; ++spin) {
    if (lane == 0) {
      observed = atomicAdd(counter, 0);
    }
    observed = __shfl_sync(kFullWarpMask, observed, 0);
    if (observed >= expected) {
      asm volatile("" ::: "memory");
      __syncwarp(kFullWarpMask);
      __threadfence_block();
      __syncwarp(kFullWarpMask);
      return true;
    }
  }
  if (lane == 0) {
    atomicCAS(protocol_error + blockIdx.x, 0u,
              protocol_error_code(phase, generation, warp));
  }
  __syncwarp(kFullWarpMask);
  return false;
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

extern "C" __global__ __launch_bounds__(kBaselineThreads, 2)
void sm70_native_bm32_d128_baseline(
    const __half* __restrict__ query, const __half* __restrict__ key,
    const __half* __restrict__ value, __half* __restrict__ output, int groups,
    int nblocks) {
  __shared__ __align__(16) BaselineShared shared;

  const int block = blockIdx.x;
  const int group = block >> 1;
  if (group >= groups) {
    return;
  }
  const int m_panel = block & 1;
  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const __half* query_group =
      query + static_cast<int64_t>(group) * kQElements +
      m_panel * kQPanelElements;
  const __half* key_group =
      key + static_cast<int64_t>(group) * nblocks * kBlockN * kD;
  const __half* value_group =
      value + static_cast<int64_t>(group) * nblocks * kBlockN * kD;
  __half* output_group =
      output + static_cast<int64_t>(group) * kOutputElements +
      m_panel * kPanelM * kD;

  stage_swizzled_q_panel(query_group, shared.query, thread, kBaselineThreads);
  if (thread < kPanelM) {
    shared.row_max[thread] = kNegativeInfinity;
    shared.row_sum[thread] = 0.0f;
  }
  for (int index = thread; index < kPanelM * kD;
       index += kBaselineThreads) {
    shared.output[index] = 0.0f;
  }
  __syncthreads();

  for (int block_n = 0; block_n < nblocks; ++block_n) {
    if (warp < kQKWarps) {
      const int n_offset = warp * kPanelN;
      AccumulatorFragment accumulator;
      qk_single_accumulate(shared.query,
                           key_group + block_n * kBlockN * kD +
                               n_offset * kD,
                           accumulator);
      nvcuda::wmma::store_matrix_sync(shared.score + n_offset, accumulator,
                                      kBlockN,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

#pragma unroll
    for (int panel = 0; panel < kSoftmaxPanelsPerBlock; ++panel) {
      const float exp_diff = make_probability_row(
          shared.score + warp * kBlockN, shared.probability, warp,
          shared.row_max, shared.row_sum, warp, panel);
      if (block_n != 0 || panel != 0) {
        scale_shared_output_row(shared.output + warp * kD, exp_diff);
      }
      __syncthreads();

      update_baseline_pv_panel(
          shared.probability,
          value_group + (block_n * kBlockN + panel * kSoftmaxPanelN) * kD,
          shared.output, warp * kPanelN);
      __syncthreads();
    }
  }

  for (int index = thread; index < kPanelM * kD;
       index += kBaselineThreads) {
    const int row = index / kD;
    const int column = index - row * kD;
    const float inverse_sum = 1.0f / fmaxf(shared.row_sum[row], 1e-24f);
    output_group[index] =
        __float2half_rn(shared.output[row * kD + column] * inverse_sum);
  }
}

extern "C" __global__ __launch_bounds__(kCandidateThreads, 2)
void sm70_native_bm32_d128_candidate(
    const __half* __restrict__ query, const __half* __restrict__ key,
    const __half* __restrict__ value, __half* __restrict__ output, int groups,
    int nblocks, uint32_t* __restrict__ protocol_error) {
  __shared__ __align__(16) CandidateShared shared;

  const int candidate_cta = blockIdx.x;
  const int group = candidate_cta >> 1;
  if (group >= groups) {
    return;
  }
  const int d_half = candidate_cta & 1;
  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const __half* query_group =
      query + static_cast<int64_t>(group) * kQElements;
  stage_swizzled_q_panel(query_group, shared.query, thread,
                         kCandidateThreads);
  stage_swizzled_q_panel(query_group + kQPanelElements,
                         shared.query + kQPanelElements, thread,
                         kCandidateThreads);
  if (thread < kM) {
    shared.row_max[thread] = kNegativeInfinity;
    shared.row_sum[thread] = 0.0f;
  }
  if (thread == 0) {
    shared.score_ready = 0;
    shared.p_ready[0] = 0;
    shared.p_ready[1] = 0;
    shared.p_consumed[0] = 0;
    shared.p_consumed[1] = 0;
  }

  // No accumulator fragment exists before this CTA barrier. All later
  // handoffs are warp-local counter epochs so the two PV accumulators stay
  // register resident.
  __syncthreads();

  if (warp < kQKWarps) {
    const __half* key_group =
        key + static_cast<int64_t>(group) * nblocks * kBlockN * kD;
    const int n_offset = warp * kPanelN;

    for (int block_n = 0; block_n < nblocks; ++block_n) {
      {
        AccumulatorFragment top_accumulator;
        AccumulatorFragment bottom_accumulator;
        qk_pair_accumulate(shared.query, shared.query + kQPanelElements,
                           key_group + (block_n * kBlockN + n_offset) * kD,
                           top_accumulator, bottom_accumulator);
        nvcuda::wmma::store_matrix_sync(shared.score_top + n_offset,
                                        top_accumulator, kBlockN,
                                        nvcuda::wmma::mem_row_major);
        nvcuda::wmma::store_matrix_sync(shared.score_bottom + n_offset,
                                        bottom_accumulator, kBlockN,
                                        nvcuda::wmma::mem_row_major);
      }
      publish_warp_counter(&shared.score_ready);
      if (!wait_for_counter(&shared.score_ready,
                            kQKWarps * (block_n + 1), protocol_error,
                            kProtocolTimeoutScore, block_n, warp)) {
        return;
      }

#pragma unroll 1
      for (int panel = 0; panel < kSoftmaxPanelsPerBlock; ++panel) {
        const int generation = block_n * kSoftmaxPanelsPerBlock + panel;
        const int probability_slot = panel & 1;
        if (generation >= 2 &&
            !wait_for_counter(&shared.p_consumed[probability_slot],
                              kCandidatePVWarps * (generation >> 1),
                              protocol_error, kProtocolTimeoutPConsumed,
                              generation, warp)) {
          return;
        }

        produce_candidate_probability_panel(&shared, warp, panel,
                                             probability_slot);
        publish_warp_counter(&shared.p_ready[probability_slot]);
        const int ready_epoch =
            kQKWarps * ((generation >> 1) + 1);
        if (!wait_for_counter(&shared.p_ready[probability_slot], ready_epoch,
                              protocol_error, kProtocolTimeoutPReady,
                              generation, warp)) {
          return;
        }
      }
    }
  } else {
    const __half* value_group =
        value + static_cast<int64_t>(group) * nblocks * kBlockN * kD;
    const int pv_warp = warp - kQKWarps;
    const int d_offset = d_half * kCandidateD + pv_warp * kPanelN;
    AccumulatorFragment top_accumulator;
    AccumulatorFragment bottom_accumulator;
    nvcuda::wmma::fill_fragment(top_accumulator, 0.0f);
    nvcuda::wmma::fill_fragment(bottom_accumulator, 0.0f);

    for (int block_n = 0; block_n < nblocks; ++block_n) {
#pragma unroll 1
      for (int panel = 0; panel < kSoftmaxPanelsPerBlock; ++panel) {
        const int generation = block_n * kSoftmaxPanelsPerBlock + panel;
        const int probability_slot = panel & 1;
        const int ready_epoch =
            kQKWarps * ((generation >> 1) + 1);
        if (!wait_for_counter(&shared.p_ready[probability_slot], ready_epoch,
                              protocol_error, kProtocolTimeoutPReady,
                              generation, warp)) {
          return;
        }
        if (generation != 0) {
          scale_candidate_accumulators(
              top_accumulator, bottom_accumulator,
              shared.row_exp_diff[probability_slot]);
        }
        update_candidate_pv_panel(
            shared.probability[probability_slot][0],
            shared.probability[probability_slot][1],
            value_group + (block_n * kBlockN + panel * kSoftmaxPanelN) * kD,
            d_offset, top_accumulator, bottom_accumulator);
        publish_warp_counter(&shared.p_consumed[probability_slot]);
      }
    }

    __half* output_group =
        output + static_cast<int64_t>(group) * kOutputElements;
    store_accumulator_output(top_accumulator, output_group, shared.row_sum,
                             0, d_offset);
    store_accumulator_output(bottom_accumulator, output_group, shared.row_sum,
                             kPanelM, d_offset);
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

uint32_t half_word_bits(const __half* values, size_t word_index) {
  uint32_t bits;
  std::memcpy(&bits, values + 2 * word_index, sizeof(bits));
  return bits;
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
  }
  for (size_t index = 0; index < baseline.size(); ++index) {
    result.max_abs_error =
        std::max(result.max_abs_error,
                 std::fabs(__half2float(baseline[index]) -
                           __half2float(candidate[index])));
  }
  return result;
}

ProtocolDiagnostics read_protocol_diagnostics(const uint32_t* device_errors,
                                              int candidate_ctas) {
  std::vector<uint32_t> host_errors(candidate_ctas);
  CUDA_CHECK(cudaMemcpy(host_errors.data(), device_errors,
                        host_errors.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  ProtocolDiagnostics result;
  for (int cta = 0; cta < candidate_ctas; ++cta) {
    if (host_errors[cta] == 0) {
      continue;
    }
    if (!result.timed_out) {
      result.first_cta = cta;
      result.first_code = host_errors[cta];
    }
    result.timed_out = true;
    ++result.timeout_ctas;
  }
  return result;
}

void report_protocol_timeout(const ProtocolDiagnostics& diagnostics) {
  if (!diagnostics.timed_out) {
    return;
  }
  std::cerr << "Candidate counter protocol timeout: ctas="
            << diagnostics.timeout_ctas << ", first_cta="
            << diagnostics.first_cta << ", code=0x" << std::hex
            << diagnostics.first_code << std::dec << '\n';
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
  result.local_bytes_per_thread = attributes.localSizeBytes;
  result.active_ctas_per_sm = active_ctas;
  result.threads_per_cta = threads_per_cta;
  result.warps_per_cta = threads_per_cta / 32;
  result.resident_warps = result.active_ctas_per_sm * result.warps_per_cta;
  return result;
}

bool candidate_resource_gate_passes(const KernelResources& resources) {
  return resources.registers_per_thread <= 64 &&
         resources.static_shared_bytes <= 48 * 1024 &&
         resources.local_bytes_per_thread == 0 &&
         resources.active_ctas_per_sm >= 2;
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
            << ", \"local_bytes_per_thread\": "
            << resources.local_bytes_per_thread
            << ", \"active_ctas_per_sm\": "
            << resources.active_ctas_per_sm << ", \"threads_per_cta\": "
            << resources.threads_per_cta << ", \"warps_per_cta\": "
            << resources.warps_per_cta << ", \"resident_warps\": "
            << resources.resident_warps << '}';
}

void print_resource_gate(const KernelResources& candidate_resources) {
  std::cout << "{\"candidate_runtime_pass\": "
            << (candidate_resource_gate_passes(candidate_resources) ? "true"
                                                                    : "false")
            << ", \"max_registers_per_thread\": 64"
               ", \"max_static_shared_bytes\": 49152"
               ", \"required_local_bytes_per_thread\": 0"
               ", \"min_active_ctas_per_sm\": 2"
               ", \"requires_ptxas_stack_spill_check\": true}";
}

void print_header_json(const Args& args, const cudaDeviceProp& properties,
                       int runtime_version, int sm_count) {
  std::cout << "  \"device\": {\"logical_index\": " << args.device
            << ", \"name\": ";
  print_json_string(properties.name);
  std::cout << ", \"capability\": [" << properties.major << ", "
            << properties.minor << "], \"cuda_runtime\": "
            << runtime_version << ", \"sm_count\": " << sm_count << "},\n";
  std::cout << "  \"target\": \"sm_70\",\n";
  std::cout << "  \"shape\": {\"groups\": " << args.groups
            << ", \"nblocks\": " << args.nblocks
            << ", \"M\": 32, \"D\": 256, \"candidate_D\": 128"
               ", \"BN\": 128, \"N\": "
            << args.nblocks * kBlockN << "},\n";
  std::cout << "  \"input_pattern\": ";
  print_json_string(args.pattern);
  std::cout << ",\n";
}

void print_profile_json(const Args& args, const cudaDeviceProp& properties,
                        int runtime_version, int sm_count,
                        const KernelResources& baseline_resources,
                        const KernelResources& candidate_resources) {
  std::cout << "{\n";
  print_header_json(args, properties, runtime_version, sm_count);
  std::cout << "  \"mode\": \"profile_only\",\n";
  std::cout << "  \"profile_kernel\": ";
  print_json_string(args.profile_kernel);
  std::cout << ",\n";
  std::cout << "  \"resources\": {\"baseline\": ";
  print_resources(baseline_resources);
  std::cout << ", \"candidate\": ";
  print_resources(candidate_resources);
  std::cout << "},\n";
  std::cout << "  \"resource_gate\": ";
  print_resource_gate(candidate_resources);
  std::cout << "\n}\n";
}

void print_resource_failure_json(const Args& args,
                                 const cudaDeviceProp& properties,
                                 int runtime_version, int sm_count,
                                 const KernelResources& baseline_resources,
                                 const KernelResources& candidate_resources) {
  std::cout << "{\n";
  print_header_json(args, properties, runtime_version, sm_count);
  std::cout << "  \"mode\": \"resource_gate_failed\",\n";
  std::cout << "  \"resources\": {\"baseline\": ";
  print_resources(baseline_resources);
  std::cout << ", \"candidate\": ";
  print_resources(candidate_resources);
  std::cout << "},\n";
  std::cout << "  \"resource_gate\": ";
  print_resource_gate(candidate_resources);
  std::cout << "\n}\n";
}

void print_json(const Args& args, const cudaDeviceProp& properties,
                int runtime_version, int sm_count, const Exactness& exactness,
                const TimingSummary& baseline_timing,
                const TimingSummary& candidate_timing,
                const PairSummary& pairs,
                const KernelResources& baseline_resources,
                const KernelResources& candidate_resources,
                const ProtocolDiagnostics& protocol_diagnostics) {
  const double candidate_speedup_pct =
      100.0 * (baseline_timing.median_us - candidate_timing.median_us) /
      baseline_timing.median_us;
  const bool candidate_faster = candidate_timing.median_us < baseline_timing.median_us;
  std::cout << "{\n";
  print_header_json(args, properties, runtime_version, sm_count);
  std::cout << "  \"layouts\": {\"q\": \"[group,M32,D256] fp16\", "
               "\"k_v\": \"[group,N,D256] contiguous fp16\", "
               "\"output\": \"[group,M32,D256] fp16\"},\n";
  std::cout << "  \"paths\": {\n";
  std::cout << "    \"baseline\": \"2 CTA/group; BM16, D256, 512 threads; serial QK-softmax-PV\",\n";
  std::cout << "    \"candidate\": \"2 CTA/group; each BM32, D128, 512 threads; 8 QK producer and 8 PV consumer warps\",\n";
  std::cout << "    \"candidate_output_halves\": \"CTA 0 writes D[0:128], CTA 1 writes D[128:256]\",\n";
  std::cout << "    \"candidate_qk_order\": \"one K fragment, top then bottom M16\",\n";
  std::cout << "    \"candidate_pv_order\": \"one V fragment, top then bottom M16 for one D16\",\n";
  std::cout << "    \"candidate_pv_accumulators_per_warp\": 2,\n";
  std::cout << "    \"candidate_sync\": \"one initialization CTA barrier; warp-local score/P epochs afterwards\"\n";
  std::cout << "  },\n";
  std::cout << "  \"exactness\": {\"word_dtype\": \"uint32 packed fp16\", \"word_count\": "
            << static_cast<int64_t>(args.groups) * kOutputElements / 2
            << ", \"full_output\": true, \"bitwise_equal\": "
            << (exactness.bitwise_equal ? "true" : "false")
            << ", \"mismatch_words\": " << exactness.mismatch_words
            << ", \"xor\": {\"max_word\": " << exactness.max_word_xor
            << ", \"reduction\": " << exactness.xor_reduction
            << "}, \"max_abs_error\": " << exactness.max_abs_error << "},\n";
  std::cout << "  \"timing\": {\"unit\": \"us per grid launch\", \"baseline\": ";
  print_timing(baseline_timing);
  std::cout << ", \"candidate\": ";
  print_timing(candidate_timing);
  std::cout << ", \"candidate_speedup_vs_baseline_pct\": "
            << candidate_speedup_pct << "},\n";
  std::cout << "  \"pairs\": {\"count\": " << pairs.count
            << ", \"candidate_faster\": " << pairs.candidate_faster
            << ", \"baseline_faster\": " << pairs.baseline_faster
            << ", \"ties\": " << pairs.ties
            << ", \"candidate_minus_baseline_median_us\": "
            << pairs.candidate_minus_baseline_median_us
            << ", \"candidate_minus_baseline_mean_us\": "
            << pairs.candidate_minus_baseline_mean_us << "},\n";
  std::cout << "  \"measurement\": {\"warmup_pairs\": " << args.warmup
            << ", \"rounds\": " << args.rounds
            << ", \"launches_per_sample\": " << args.launches_per_sample
            << ", \"interleaving\": \"baseline/candidate alternates each round\"},\n";
  std::cout << "  \"resources\": {\"baseline\": ";
  print_resources(baseline_resources);
  std::cout << ", \"candidate\": ";
  print_resources(candidate_resources);
  std::cout << "},\n";
  std::cout << "  \"protocol_diagnostics\": {\"timed_out\": "
            << (protocol_diagnostics.timed_out ? "true" : "false")
            << ", \"timeout_ctas\": " << protocol_diagnostics.timeout_ctas
            << ", \"first_cta\": " << protocol_diagnostics.first_cta
            << ", \"first_code\": " << protocol_diagnostics.first_code
            << "},\n";
  std::cout << "  \"full_pipeline_result\": {\"candidate_faster\": "
            << (candidate_faster ? "true" : "false")
            << ", \"qk_duplication_offsets_pv_reuse\": "
            << (candidate_faster ? "false" : "true") << "},\n";
  std::cout << "  \"resource_gate\": ";
  print_resource_gate(candidate_resources);
  std::cout << "\n}\n";
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

  const KernelResources baseline_resources =
      query_resources(sm70_native_bm32_d128_baseline, kBaselineThreads);
  const KernelResources candidate_resources =
      query_resources(sm70_native_bm32_d128_candidate, kCandidateThreads);
  if (!candidate_resource_gate_passes(candidate_resources)) {
    print_resource_failure_json(args, properties, runtime_version, sm_count,
                                baseline_resources, candidate_resources);
    return EXIT_FAILURE;
  }

  const size_t query_elements = static_cast<size_t>(args.groups) * kQElements;
  const size_t kv_elements = static_cast<size_t>(args.groups) * args.nblocks *
                             kBlockN * kD;
  const size_t output_elements =
      static_cast<size_t>(args.groups) * kOutputElements;
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
  uint32_t* device_protocol_error = nullptr;
  const int candidate_ctas = args.groups * kCandidateCtasPerGroup;
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
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_protocol_error),
                        static_cast<size_t>(candidate_ctas) *
                            sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(device_query, host_query.data(),
                        query_elements * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_key, host_key.data(),
                        kv_elements * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_value, host_value.data(),
                        kv_elements * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(device_protocol_error, 0,
                        static_cast<size_t>(candidate_ctas) *
                            sizeof(uint32_t)));

  const dim3 baseline_grid(args.groups * 2);
  const dim3 candidate_grid(candidate_ctas);
  auto launch_baseline = [&] {
    sm70_native_bm32_d128_baseline<<<baseline_grid, kBaselineThreads>>>(
        device_query, device_key, device_value, device_baseline, args.groups,
        args.nblocks);
    CUDA_CHECK(cudaGetLastError());
  };
  auto launch_candidate = [&] {
    sm70_native_bm32_d128_candidate<<<candidate_grid, kCandidateThreads>>>(
        device_query, device_key, device_value, device_candidate, args.groups,
        args.nblocks, device_protocol_error);
    CUDA_CHECK(cudaGetLastError());
  };
  auto free_device_buffers = [&] {
    CUDA_CHECK(cudaFree(device_protocol_error));
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
    const ProtocolDiagnostics diagnostics =
        read_protocol_diagnostics(device_protocol_error, candidate_ctas);
    free_device_buffers();
    print_profile_json(args, properties, runtime_version, sm_count,
                       baseline_resources, candidate_resources);
    report_protocol_timeout(diagnostics);
    return diagnostics.timed_out ? EXIT_FAILURE : EXIT_SUCCESS;
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
  ProtocolDiagnostics protocol_diagnostics =
      read_protocol_diagnostics(device_protocol_error, candidate_ctas);
  if (protocol_diagnostics.timed_out) {
    free_device_buffers();
    report_protocol_timeout(protocol_diagnostics);
    return EXIT_FAILURE;
  }

  launch_baseline();
  launch_candidate();
  CUDA_CHECK(cudaDeviceSynchronize());
  protocol_diagnostics =
      read_protocol_diagnostics(device_protocol_error, candidate_ctas);
  if (protocol_diagnostics.timed_out) {
    free_device_buffers();
    report_protocol_timeout(protocol_diagnostics);
    return EXIT_FAILURE;
  }
  std::vector<__half> host_baseline(output_elements);
  std::vector<__half> host_candidate(output_elements);
  CUDA_CHECK(cudaMemcpy(host_baseline.data(), device_baseline,
                        output_elements * sizeof(__half),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_candidate.data(), device_candidate,
                        output_elements * sizeof(__half),
                        cudaMemcpyDeviceToHost));
  const Exactness exactness = compare_outputs(host_baseline, host_candidate);

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
  protocol_diagnostics =
      read_protocol_diagnostics(device_protocol_error, candidate_ctas);
  if (protocol_diagnostics.timed_out) {
    free_device_buffers();
    report_protocol_timeout(protocol_diagnostics);
    return EXIT_FAILURE;
  }

  const TimingSummary baseline_timing = summarize(baseline_samples);
  const TimingSummary candidate_timing = summarize(candidate_samples);
  const PairSummary pairs =
      summarize_pairs(baseline_samples, candidate_samples);
  free_device_buffers();
  print_json(args, properties, runtime_version, sm_count, exactness,
             baseline_timing, candidate_timing, pairs, baseline_resources,
             candidate_resources, protocol_diagnostics);
  return exactness.bitwise_equal ? EXIT_SUCCESS : EXIT_FAILURE;
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
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--device N] [--groups N] [--nblocks N] [--warmup N]"
                   " [--rounds N] [--launches N]"
                   " [--pattern random|alternating] [--profile-only]"
                   " [--profile-kernel baseline|candidate|both]\n";
      std::exit(EXIT_FAILURE);
    }
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  return run(parse_args(argc, argv));
}
