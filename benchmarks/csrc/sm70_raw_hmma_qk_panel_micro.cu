// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
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

constexpr int kM = 16;
constexpr int kN = 256;
constexpr int kK = 256;
constexpr int kBaselineN = 128;
constexpr int kNativeN = 16;
constexpr int kNativeK = 16;
constexpr int kRawM = 8;
constexpr int kRawN = 32;
constexpr int kRawK = 4;
constexpr int kRawRegisters = 8;
constexpr int kBaselineQKWarps = kBaselineN / kNativeN;
constexpr int kRawQKWarps = kN / kRawN;
constexpr int kBaselineThreads = 512;
constexpr int kRawThreads = 256;
constexpr int kQElements = kM * kK;
constexpr int kOutputElementsPerGroup = kM * kN;
constexpr int kKeyElementsPerGroup = kN * kK;

using AFragment = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kM, kNativeN,
                                         kNativeK, __half,
                                         nvcuda::wmma::row_major>;
using BFragment = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kM, kNativeN,
                                         kNativeK, __half,
                                         nvcuda::wmma::col_major>;
using CFragment = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kM,
                                         kNativeN, kNativeK, float>;

struct Args {
  int device = 0;
  int groups = 1024;
  int warmup = 20;
  int rounds = 100;
  int launches_per_sample = 8;
  bool profile_only = false;
  std::string profile_kernel = "both";
  std::string raw_variant = "raw_k4_vector";
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
  int raw_faster = 0;
  int baseline_faster = 0;
  int ties = 0;
  double raw_minus_baseline_median_us = 0.0;
  double raw_minus_baseline_mean_us = 0.0;
};

struct KernelResources {
  int registers_per_thread = 0;
  size_t static_shared_bytes = 0;
  size_t local_bytes_per_thread = 0;
  int active_ctas_per_sm = 0;
  int resident_total_warps = 0;
  int resident_qk_warps = 0;
  int threads_per_cta = 0;
  int qk_warps_per_cta = 0;
};

struct Exactness {
  bool bitwise_equal = false;
  int mismatch_words = 0;
  uint32_t xor_reduction = 0;
  uint32_t max_word_xor = 0;
  float max_abs_error = 0.0f;
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

#define CUDA_CHECK(expression) check_cuda((expression), #expression, __FILE__, __LINE__)

__device__ __forceinline__ int swizzled_q_row_slot(int row) {
  return (row & 3) | ((row & 8) >> 1) | ((row & 4) << 1);
}

__device__ __forceinline__ int swizzled_q_offset(int row, int column) {
  const int k_tile = column / kNativeK;
  const int within_tile = column - k_tile * kNativeK;
  const int plane = within_tile >> 3;
  const int inner = within_tile & 7;
  return k_tile * kM * kNativeK + plane * kM * 8 +
         swizzled_q_row_slot(row) * 8 + inner;
}

__device__ __forceinline__ void stage_swizzled_query(
    const __half* __restrict__ query, __half* __restrict__ shared_query,
    int thread, int thread_count) {
  constexpr int kHalfPerUint4 = sizeof(uint4) / sizeof(__half);
  constexpr int kQueryVectors = kQElements / kHalfPerUint4;
  constexpr int kVectorsPerRow = kK / kHalfPerUint4;
  const uint4* query_vectors = reinterpret_cast<const uint4*>(query);
  uint4* shared_vectors = reinterpret_cast<uint4*>(shared_query);

  for (int index = thread; index < kQueryVectors; index += thread_count) {
    const int row = index / kVectorsPerRow;
    const int vector_column = index % kVectorsPerRow;
    const int k_tile = vector_column >> 1;
    const int plane = vector_column & 1;
    const int slot = swizzled_q_row_slot(row);
    shared_vectors[k_tile * (2 * kM) + plane * kM + slot] =
        __ldg(query_vectors + index);
  }
}

__device__ __forceinline__ void load_swizzled_matrix_a_fragment(
    AFragment& fragment, const __half* __restrict__ shared_query,
    int k_offset) {
  const int lane = threadIdx.x & 31;
  const int row = (lane & 3) + ((lane >> 4) & 1) * 4 +
                  ((lane >> 2) & 1) * 8;
  const int slot = swizzled_q_row_slot(row);
  const int tile_offset = (k_offset / kNativeK) * kM * kNativeK;
  uint32_t address = static_cast<uint32_t>(
      __cvta_generic_to_shared(shared_query + tile_offset + slot * 8));
  uint32_t* words = reinterpret_cast<uint32_t*>(&fragment);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]),
                 "=r"(words[3])
               : "r"(address)
               : "memory");
  address += kM * 8 * sizeof(__half);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(words[4]), "=r"(words[5]), "=r"(words[6]),
                 "=r"(words[7])
               : "r"(address)
               : "memory");
}

// The m8n8k4 coordinates are derived from SM70_MMA_884::thread_offset_C(),
// static_offset_C(), ReshapeC(), and SmemCopy_MMA_884_B in the TurboMind SM70
// headers. The preceding raw-HMMA probe validates the canonical K4 order.
__device__ __forceinline__ int raw_a_row(int lane) {
  return (lane & 3) + ((lane >> 4) * 4);
}

__device__ __forceinline__ int raw_n_tile(int lane) {
  return (lane & 12) * 2;
}

__device__ __forceinline__ int raw_b_col_n(int lane) {
  return raw_n_tile(lane) + (lane & 3) + ((lane >> 4) * 4);
}

__device__ __forceinline__ int raw_c_row(int lane, int reg) {
  return (lane & 1) + (reg & 2) + ((lane >> 4) * 4);
}

__device__ __forceinline__ int raw_c_col(int lane, int reg) {
  return raw_n_tile(lane) + (lane & 2) + (reg & 4) + (reg & 1);
}

__device__ __forceinline__ uint32_t pack_half2(__half low, __half high) {
  return static_cast<uint32_t>(__half_as_ushort(low)) |
         (static_cast<uint32_t>(__half_as_ushort(high)) << 16);
}

// cudaMalloc provides the base alignment; the group, token, K4, and K16 strides
// preserve the natural 8-byte and 16-byte alignment of the vector loads below.
__device__ __forceinline__ void load_key_k4_v2_u32(uint32_t& b0,
                                                     uint32_t& b1,
                                                     const __half* key_k4) {
  asm volatile("ld.global.v2.u32 {%0, %1}, [%2];"
               : "=r"(b0), "=r"(b1)
               : "l"(key_k4)
               : "memory");
}

__device__ __forceinline__ void load_key_k8_v4_u32(
    uint32_t& b0, uint32_t& b1, uint32_t& b2, uint32_t& b3,
    const __half* key_k8) {
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
               : "l"(key_k8)
               : "memory");
}

__device__ __forceinline__ void load_key_k16_v4_u32(
    uint32_t& b0, uint32_t& b1, uint32_t& b2, uint32_t& b3, uint32_t& b4,
    uint32_t& b5, uint32_t& b6, uint32_t& b7, const __half* key_k16) {
  load_key_k8_v4_u32(b0, b1, b2, b3, key_k16);
  load_key_k8_v4_u32(b4, b5, b6, b7, key_k16 + 8);
}

__device__ __forceinline__ void load_swizzled_raw_a_k8_v4_u32(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    const __half* __restrict__ shared_query, int row, int k16, int plane) {
  const int slot = swizzled_q_row_slot(row);
  const int tile_offset = (k16 / kNativeK) * kM * kNativeK;
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
      shared_query + tile_offset + plane * kM * 8 + slot * 8));
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
               : "r"(address)
               : "memory");
}

__device__ __forceinline__ void load_swizzled_raw_a_k16_v4_u32(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint32_t& a4,
    uint32_t& a5, uint32_t& a6, uint32_t& a7,
    const __half* __restrict__ shared_query, int row, int k16) {
  load_swizzled_raw_a_k8_v4_u32(a0, a1, a2, a3, shared_query, row, k16, 0);
  load_swizzled_raw_a_k8_v4_u32(a4, a5, a6, a7, shared_query, row, k16, 1);
}

__device__ __forceinline__ void load_swizzled_raw_a_k4_v2_u32(
    uint32_t& a0, uint32_t& a1, const __half* __restrict__ shared_query,
    int row, int k16, int k4) {
  const int slot = swizzled_q_row_slot(row);
  const int tile_offset = (k16 / kNativeK) * kM * kNativeK;
  const int plane = k4 >> 1;
  const int k4_in_plane = k4 & 1;
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
      shared_query + tile_offset + plane * kM * 8 + slot * 8 +
      k4_in_plane * kRawK));
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
               : "=r"(a0), "=r"(a1)
               : "r"(address)
               : "memory");
}

__device__ __forceinline__ void mma_m8n8k4_row_col(
    float (&d)[kRawRegisters], uint32_t a0, uint32_t a1, uint32_t b0,
    uint32_t b1, const float (&c)[kRawRegisters]) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, "
      "{%8, %9}, {%10, %11}, {%12, %13, %14, %15, %16, %17, %18, %19};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),
        "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]),
        "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]),
        "f"(c[7]));
}

extern "C" __global__ __launch_bounds__(kBaselineThreads, 2)
void qk_panel_baseline_kernel(const __half* __restrict__ query,
                              const __half* __restrict__ key,
                              float* __restrict__ output, int groups) {
  __shared__ __align__(16) __half shared_query[kQElements];

  const int block = blockIdx.x;
  const int group = block >> 1;
  if (group >= groups) {
    return;
  }
  const int thread = threadIdx.x;
  const __half* query_group =
      query + static_cast<int64_t>(group) * kQElements;
  stage_swizzled_query(query_group, shared_query, thread, kBaselineThreads);
  __syncthreads();

  const int warp = thread >> 5;
  if (warp >= kBaselineQKWarps) {
    return;
  }
  const int n_offset = (block & 1) * kBaselineN + warp * kNativeN;
  const __half* key_group = key + static_cast<int64_t>(group) *
                                       kKeyElementsPerGroup;
  AFragment a_fragment;
  BFragment b_fragment;
  CFragment accumulator;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);

#pragma unroll
  for (int k_offset = 0; k_offset < kK; k_offset += kNativeK) {
    load_swizzled_matrix_a_fragment(a_fragment, shared_query, k_offset);
    nvcuda::wmma::load_matrix_sync(
        b_fragment, key_group + n_offset * kK + k_offset, kK);
    nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
  }

  float* output_tile = output + static_cast<int64_t>(group) *
                                     kOutputElementsPerGroup + n_offset;
  nvcuda::wmma::store_matrix_sync(output_tile, accumulator, kN,
                                  nvcuda::wmma::mem_row_major);
}

extern "C" __global__ __launch_bounds__(kRawThreads, 4)
void qk_panel_raw_k4_vector_kernel(const __half* __restrict__ query,
                                   const __half* __restrict__ key,
                                   float* __restrict__ output, int groups) {
  __shared__ __align__(16) __half shared_query[kQElements];

  const int block = blockIdx.x;
  const int group = block >> 1;
  if (group >= groups) {
    return;
  }
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int m_offset = (block & 1) * kRawM;
  const int n_offset = warp * kRawN;
  const __half* query_group =
      query + static_cast<int64_t>(group) * kQElements;
  const __half* key_group = key + static_cast<int64_t>(group) *
                                       kKeyElementsPerGroup;
  stage_swizzled_query(query_group, shared_query, thread, kRawThreads);
  __syncthreads();

  float accumulator[kRawRegisters] = {0.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f};
  const int q_row = m_offset + raw_a_row(lane);
  const int key_token = n_offset + raw_b_col_n(lane);

#pragma unroll
  for (int k16 = 0; k16 < kK; k16 += kNativeK) {
#pragma unroll
    for (int k4 = 0; k4 < kNativeK / kRawK; ++k4) {
      const int k_offset = k16 + k4 * kRawK;
      const uint32_t a0 = pack_half2(
          shared_query[swizzled_q_offset(q_row, k_offset)],
          shared_query[swizzled_q_offset(q_row, k_offset + 1)]);
      const uint32_t a1 = pack_half2(
          shared_query[swizzled_q_offset(q_row, k_offset + 2)],
          shared_query[swizzled_q_offset(q_row, k_offset + 3)]);
      uint32_t b0;
      uint32_t b1;
      load_key_k4_v2_u32(b0, b1,
                          key_group + key_token * kK + k_offset);
      mma_m8n8k4_row_col(accumulator, a0, a1, b0, b1, accumulator);
    }
  }

  float* output_group = output + static_cast<int64_t>(group) *
                                      kOutputElementsPerGroup;
#pragma unroll
  for (int reg = 0; reg < kRawRegisters; ++reg) {
    const int row = m_offset + raw_c_row(lane, reg);
    const int column = n_offset + raw_c_col(lane, reg);
    output_group[row * kN + column] = accumulator[reg];
  }
}

extern "C" __global__ __launch_bounds__(kRawThreads, 4)
void qk_panel_raw_k16_stage_kernel(const __half* __restrict__ query,
                                   const __half* __restrict__ key,
                                   float* __restrict__ output, int groups) {
  __shared__ __align__(16) __half shared_query[kQElements];

  const int block = blockIdx.x;
  const int group = block >> 1;
  if (group >= groups) {
    return;
  }
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int m_offset = (block & 1) * kRawM;
  const int n_offset = warp * kRawN;
  const __half* query_group =
      query + static_cast<int64_t>(group) * kQElements;
  const __half* key_group = key + static_cast<int64_t>(group) *
                                       kKeyElementsPerGroup;
  stage_swizzled_query(query_group, shared_query, thread, kRawThreads);
  __syncthreads();

  float accumulator[kRawRegisters] = {0.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f};
  const int q_row = m_offset + raw_a_row(lane);
  const int key_token = n_offset + raw_b_col_n(lane);

#pragma unroll
  for (int k16 = 0; k16 < kK; k16 += kNativeK) {
    uint32_t a0;
    uint32_t a1;
    uint32_t a2;
    uint32_t a3;
    uint32_t a4;
    uint32_t a5;
    uint32_t a6;
    uint32_t a7;
    uint32_t b0;
    uint32_t b1;
    uint32_t b2;
    uint32_t b3;
    uint32_t b4;
    uint32_t b5;
    uint32_t b6;
    uint32_t b7;
    const __half* key_k16 = key_group + key_token * kK + k16;
    load_key_k16_v4_u32(b0, b1, b2, b3, b4, b5, b6, b7, key_k16);
    load_swizzled_raw_a_k16_v4_u32(a0, a1, a2, a3, a4, a5, a6, a7,
                                    shared_query, q_row, k16);
    mma_m8n8k4_row_col(accumulator, a0, a1, b0, b1, accumulator);
    mma_m8n8k4_row_col(accumulator, a2, a3, b2, b3, accumulator);
    mma_m8n8k4_row_col(accumulator, a4, a5, b4, b5, accumulator);
    mma_m8n8k4_row_col(accumulator, a6, a7, b6, b7, accumulator);
  }

  float* output_group = output + static_cast<int64_t>(group) *
                                      kOutputElementsPerGroup;
#pragma unroll
  for (int reg = 0; reg < kRawRegisters; ++reg) {
    const int row = m_offset + raw_c_row(lane, reg);
    const int column = n_offset + raw_c_col(lane, reg);
    output_group[row * kN + column] = accumulator[reg];
  }
}

extern "C" __global__ __launch_bounds__(kRawThreads, 4)
void qk_panel_raw_k16_double_kernel(const __half* __restrict__ query,
                                    const __half* __restrict__ key,
                                    float* __restrict__ output, int groups) {
  __shared__ __align__(16) __half shared_query[kQElements];

  const int block = blockIdx.x;
  const int group = block >> 1;
  if (group >= groups) {
    return;
  }
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int m_offset = (block & 1) * kRawM;
  const int n_offset = warp * kRawN;
  const __half* query_group =
      query + static_cast<int64_t>(group) * kQElements;
  const __half* key_group = key + static_cast<int64_t>(group) *
                                       kKeyElementsPerGroup;
  stage_swizzled_query(query_group, shared_query, thread, kRawThreads);
  __syncthreads();

  float accumulator[kRawRegisters] = {0.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f};
  const int q_row = m_offset + raw_a_row(lane);
  const int key_token = n_offset + raw_b_col_n(lane);
  const __half* first_key_k16 = key_group + key_token * kK;
  uint32_t b0;
  uint32_t b1;
  uint32_t b2;
  uint32_t b3;
  uint32_t b4;
  uint32_t b5;
  uint32_t b6;
  uint32_t b7;
  load_key_k16_v4_u32(b0, b1, b2, b3, b4, b5, b6, b7, first_key_k16);

#pragma unroll
  for (int k16 = 0; k16 < kK; k16 += kNativeK) {
    uint32_t a0;
    uint32_t a1;
    uint32_t a2;
    uint32_t a3;
    uint32_t a4;
    uint32_t a5;
    uint32_t a6;
    uint32_t a7;
    uint32_t next_b0;
    uint32_t next_b1;
    uint32_t next_b2;
    uint32_t next_b3;
    uint32_t next_b4;
    uint32_t next_b5;
    uint32_t next_b6;
    uint32_t next_b7;
    const bool has_next = k16 + kNativeK < kK;
    const __half* next_key_k16 = first_key_k16 + k16 + kNativeK;

    // Keep the accumulator-dependent K4 order while preloading the next K16.
    load_swizzled_raw_a_k8_v4_u32(a0, a1, a2, a3, shared_query, q_row,
                                   k16, 0);
    mma_m8n8k4_row_col(accumulator, a0, a1, b0, b1, accumulator);
    if (has_next) {
      load_key_k8_v4_u32(next_b0, next_b1, next_b2, next_b3,
                          next_key_k16);
    }
    mma_m8n8k4_row_col(accumulator, a2, a3, b2, b3, accumulator);
    if (has_next) {
      load_key_k8_v4_u32(next_b4, next_b5, next_b6, next_b7,
                          next_key_k16 + 8);
    }
    load_swizzled_raw_a_k8_v4_u32(a4, a5, a6, a7, shared_query, q_row,
                                   k16, 1);
    mma_m8n8k4_row_col(accumulator, a4, a5, b4, b5, accumulator);
    mma_m8n8k4_row_col(accumulator, a6, a7, b6, b7, accumulator);
    if (has_next) {
      b0 = next_b0;
      b1 = next_b1;
      b2 = next_b2;
      b3 = next_b3;
      b4 = next_b4;
      b5 = next_b5;
      b6 = next_b6;
      b7 = next_b7;
    }
  }

  float* output_group = output + static_cast<int64_t>(group) *
                                      kOutputElementsPerGroup;
#pragma unroll
  for (int reg = 0; reg < kRawRegisters; ++reg) {
    const int row = m_offset + raw_c_row(lane, reg);
    const int column = n_offset + raw_c_col(lane, reg);
    output_group[row * kN + column] = accumulator[reg];
  }
}

extern "C" __global__ __launch_bounds__(kRawThreads, 4)
void qk_panel_raw_m16n256_reuse_b_kernel(
    const __half* __restrict__ query, const __half* __restrict__ key,
    float* __restrict__ output, int groups) {
  __shared__ __align__(16) __half shared_query[kQElements];

  const int group = blockIdx.x;
  if (group >= groups) {
    return;
  }
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int n_offset = warp * kRawN;
  const __half* query_group =
      query + static_cast<int64_t>(group) * kQElements;
  const __half* key_group = key + static_cast<int64_t>(group) *
                                       kKeyElementsPerGroup;
  stage_swizzled_query(query_group, shared_query, thread, kRawThreads);
  __syncthreads();

  float top_accumulator[kRawRegisters] = {0.0f, 0.0f, 0.0f, 0.0f,
                                            0.0f, 0.0f, 0.0f, 0.0f};
  float bottom_accumulator[kRawRegisters] = {0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f};
  const int top_q_row = raw_a_row(lane);
  const int bottom_q_row = top_q_row + kRawM;
  const int key_token = n_offset + raw_b_col_n(lane);

#pragma unroll
  for (int k16 = 0; k16 < kK; k16 += kNativeK) {
    uint32_t a0;
    uint32_t a1;
    uint32_t b0;
    uint32_t b1;
    uint32_t b2;
    uint32_t b3;
    uint32_t b4;
    uint32_t b5;
    uint32_t b6;
    uint32_t b7;
    const __half* key_k16 = key_group + key_token * kK + k16;
    load_key_k16_v4_u32(b0, b1, b2, b3, b4, b5, b6, b7, key_k16);

    load_swizzled_raw_a_k4_v2_u32(a0, a1, shared_query, top_q_row, k16, 0);
    mma_m8n8k4_row_col(top_accumulator, a0, a1, b0, b1, top_accumulator);
    load_swizzled_raw_a_k4_v2_u32(a0, a1, shared_query, bottom_q_row, k16,
                                   0);
    mma_m8n8k4_row_col(bottom_accumulator, a0, a1, b0, b1,
                        bottom_accumulator);

    load_swizzled_raw_a_k4_v2_u32(a0, a1, shared_query, top_q_row, k16, 1);
    mma_m8n8k4_row_col(top_accumulator, a0, a1, b2, b3, top_accumulator);
    load_swizzled_raw_a_k4_v2_u32(a0, a1, shared_query, bottom_q_row, k16,
                                   1);
    mma_m8n8k4_row_col(bottom_accumulator, a0, a1, b2, b3,
                        bottom_accumulator);

    load_swizzled_raw_a_k4_v2_u32(a0, a1, shared_query, top_q_row, k16, 2);
    mma_m8n8k4_row_col(top_accumulator, a0, a1, b4, b5, top_accumulator);
    load_swizzled_raw_a_k4_v2_u32(a0, a1, shared_query, bottom_q_row, k16,
                                   2);
    mma_m8n8k4_row_col(bottom_accumulator, a0, a1, b4, b5,
                        bottom_accumulator);

    load_swizzled_raw_a_k4_v2_u32(a0, a1, shared_query, top_q_row, k16, 3);
    mma_m8n8k4_row_col(top_accumulator, a0, a1, b6, b7, top_accumulator);
    load_swizzled_raw_a_k4_v2_u32(a0, a1, shared_query, bottom_q_row, k16,
                                   3);
    mma_m8n8k4_row_col(bottom_accumulator, a0, a1, b6, b7,
                        bottom_accumulator);
  }

  float* output_group = output + static_cast<int64_t>(group) *
                                      kOutputElementsPerGroup;
#pragma unroll
  for (int reg = 0; reg < kRawRegisters; ++reg) {
    const int row = raw_c_row(lane, reg);
    const int column = n_offset + raw_c_col(lane, reg);
    output_group[row * kN + column] = top_accumulator[reg];
    output_group[(row + kRawM) * kN + column] = bottom_accumulator[reg];
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
  return static_cast<float>(value) / 512.0f;
}

uint32_t float_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
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
                            const std::vector<double>& raw) {
  std::vector<double> deltas;
  deltas.reserve(baseline.size());
  PairSummary result;
  result.count = static_cast<int>(baseline.size());
  for (size_t index = 0; index < baseline.size(); ++index) {
    const double delta = raw[index] - baseline[index];
    deltas.push_back(delta);
    if (raw[index] < baseline[index]) {
      ++result.raw_faster;
    } else if (baseline[index] < raw[index]) {
      ++result.baseline_faster;
    } else {
      ++result.ties;
    }
  }
  const TimingSummary summary = summarize(deltas);
  result.raw_minus_baseline_median_us = summary.median_us;
  result.raw_minus_baseline_mean_us = summary.mean_us;
  return result;
}

template <typename Kernel>
KernelResources query_resources(Kernel kernel, int threads_per_cta,
                                int qk_warps_per_cta, int device) {
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
  result.qk_warps_per_cta = qk_warps_per_cta;
  result.resident_total_warps = active_ctas * (threads_per_cta / 32);
  result.resident_qk_warps = active_ctas * qk_warps_per_cta;
  return result;
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
          std::cout << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<int>(character) << std::dec << std::setfill(' ');
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
            << resources.active_ctas_per_sm
            << ", \"resident_total_warps\": "
            << resources.resident_total_warps
            << ", \"resident_qk_warps\": "
            << resources.resident_qk_warps
            << ", \"threads_per_cta\": " << resources.threads_per_cta
            << ", \"qk_warps_per_cta\": " << resources.qk_warps_per_cta
            << '}';
}

void print_json(const Args& args, const cudaDeviceProp& properties,
                int runtime_version, int sm_count, int max_threads_per_sm,
                int max_shared_per_sm, const Exactness& exactness,
                const TimingSummary& baseline_timing,
                const TimingSummary& raw_timing, const PairSummary& pairs,
                const KernelResources& baseline_resources,
                const KernelResources& raw_resources) {
  const double raw_speedup_pct =
      100.0 * (baseline_timing.median_us - raw_timing.median_us) /
      baseline_timing.median_us;
  std::cout << "{\n";
  std::cout << "  \"device\": {\n";
  std::cout << "    \"logical_index\": " << args.device << ",\n";
  std::cout << "    \"name\": ";
  print_json_string(properties.name);
  std::cout << ",\n";
  std::cout << "    \"capability\": [" << properties.major << ", "
            << properties.minor << "],\n";
  std::cout << "    \"cuda_runtime\": " << runtime_version << ",\n";
  std::cout << "    \"sm_count\": " << sm_count << ",\n";
  std::cout << "    \"max_threads_per_sm\": " << max_threads_per_sm
            << ",\n";
  std::cout << "    \"max_shared_per_sm\": " << max_shared_per_sm << "\n";
  std::cout << "  },\n";
  std::cout << "  \"target\": \"sm_70\",\n";
  std::cout << "  \"shape\": {\n";
  std::cout << "    \"groups\": " << args.groups << ",\n";
  std::cout << "    \"output\": \"[groups, M16, N256]\",\n";
  std::cout << "    \"K\": 256,\n";
  std::cout << "    \"query_layout\": \"[group, M16, K256] token-major\",\n";
  std::cout << "    \"key_layout\": \"[group, N256, K256] token-major stride 256\"\n";
  std::cout << "  },\n";
  std::cout << "  \"paths\": {\n";
  std::cout << "    \"baseline\": \"2 CTA/group, BM16xBN128, 512 threads; first 8 warps run native WMMA\",\n";
  std::cout << "    \"raw\": ";
  if (args.raw_variant == "raw_m16n256_reuse_b") {
    print_json_string(
        "1 CTA/group, BM16xBN256, 256 threads; 8 warps reuse B across M8 halves");
  } else {
    print_json_string(
        "2 CTA/group, BM8xBN256, 256 threads; 8 warps run raw m8n8k4 row.col");
  }
  std::cout << ",\n";
  std::cout << "    \"raw_variant\": ";
  print_json_string(args.raw_variant);
  std::cout << ",\n";
  std::cout << "    \"baseline_ctas_per_group\": 2,\n";
  std::cout << "    \"raw_ctas_per_group\": "
            << (args.raw_variant == "raw_m16n256_reuse_b" ? 1 : 2) << ",\n";
  std::cout << "    \"shared_q\": \"accepted-path swizzled uint4 staging and canonical K order\",\n";
  std::cout << "    \"raw_k4_order\": [0, 1, 2, 3],\n";
  std::cout << "    \"raw_b_load\": ";
  if (args.raw_variant == "raw_k4_vector") {
    print_json_string(
        "aligned inline PTX ld.global.v2.u32; one 64-bit load per token/K4");
  } else if (args.raw_variant == "raw_k16_stage") {
    print_json_string(
        "two aligned inline PTX ld.global.v4.u32 loads per token/K16");
  } else if (args.raw_variant == "raw_k16_double") {
    print_json_string(
        "K16 ping-pong: next two ld.global.v4.u32 loads interleaved with current HMMA");
  } else {
    print_json_string(
        "two aligned inline PTX ld.global.v4.u32 loads per token/K16 reused by top and bottom M8 HMMA");
  }
  std::cout << "\n";
  std::cout << "  },\n";
  std::cout << "  \"exactness\": {\n";
  std::cout << "    \"word_dtype\": \"uint32\",\n";
  std::cout << "    \"word_count\": "
            << static_cast<int64_t>(args.groups) * kOutputElementsPerGroup
            << ",\n";
  std::cout << "    \"bitwise_equal\": "
            << (exactness.bitwise_equal ? "true" : "false") << ",\n";
  std::cout << "    \"mismatch_words\": " << exactness.mismatch_words
            << ",\n";
  std::cout << "    \"xor\": {\"max_word\": "
            << exactness.max_word_xor << ", \"reduction\": "
            << exactness.xor_reduction << "},\n";
  std::cout << "    \"max_abs_error\": " << exactness.max_abs_error << "\n";
  std::cout << "  },\n";
  std::cout << "  \"timing\": {\n";
  std::cout << "    \"unit\": \"us per grid launch\",\n";
  std::cout << "    \"baseline\": ";
  print_timing(baseline_timing);
  std::cout << ",\n    \"raw\": ";
  print_timing(raw_timing);
  std::cout << ",\n    \"raw_speedup_vs_baseline_pct\": "
            << raw_speedup_pct << "\n";
  std::cout << "  },\n";
  std::cout << "  \"pairs\": {\n";
  std::cout << "    \"count\": " << pairs.count << ",\n";
  std::cout << "    \"raw_faster\": " << pairs.raw_faster << ",\n";
  std::cout << "    \"baseline_faster\": " << pairs.baseline_faster
            << ",\n";
  std::cout << "    \"ties\": " << pairs.ties << ",\n";
  std::cout << "    \"raw_minus_baseline_median_us\": "
            << pairs.raw_minus_baseline_median_us << ",\n";
  std::cout << "    \"raw_minus_baseline_mean_us\": "
            << pairs.raw_minus_baseline_mean_us << "\n";
  std::cout << "  },\n";
  std::cout << "  \"measurement\": {\n";
  std::cout << "    \"warmup_pairs\": " << args.warmup << ",\n";
  std::cout << "    \"rounds\": " << args.rounds << ",\n";
  std::cout << "    \"launches_per_sample\": " << args.launches_per_sample
            << ",\n";
  std::cout << "    \"interleaving\": \"baseline/raw order alternates every round\"\n";
  std::cout << "  },\n";
  std::cout << "  \"resources\": {\n";
  std::cout << "    \"baseline\": ";
  print_resources(baseline_resources);
  std::cout << ",\n    \"raw\": ";
  print_resources(raw_resources);
  std::cout << "\n  }\n";
  std::cout << "}\n";
}

bool profile_kernel_selected(const Args& args, const char* kernel) {
  return args.profile_kernel == "both" || args.profile_kernel == kernel;
}

bool is_raw_variant(const std::string& value) {
  return value == "raw_k4_vector" || value == "raw_k16_stage" ||
         value == "raw_k16_double" || value == "raw_m16n256_reuse_b";
}

int run(const Args& args) {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (args.device < 0 || args.device >= device_count) {
    std::cerr << "Requested logical CUDA device " << args.device
              << " is unavailable; visible device count is " << device_count << '\n';
    return EXIT_FAILURE;
  }
  if (args.groups < 1 || args.warmup < 0 || args.rounds < 1 ||
      args.launches_per_sample < 1) {
    std::cerr << "groups, rounds, and launches-per-sample must be positive; "
                 "warmup cannot be negative\n";
    return EXIT_FAILURE;
  }
  if (!is_raw_variant(args.raw_variant)) {
    std::cerr << "Unsupported --raw-variant " << args.raw_variant << '\n';
    return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaSetDevice(args.device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, args.device));
  if (properties.major != 7 || properties.minor != 0) {
    std::cerr << "This probe requires SM70, got " << properties.major << '.'
              << properties.minor << '\n';
    return EXIT_FAILURE;
  }

  int runtime_version = 0;
  int sm_count = 0;
  int max_threads_per_sm = 0;
  int max_shared_per_sm = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    args.device));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_sm,
                                    cudaDevAttrMaxThreadsPerMultiProcessor,
                                    args.device));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_per_sm,
                                    cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                                    args.device));

  const size_t query_elements = static_cast<size_t>(args.groups) * kQElements;
  const size_t key_elements =
      static_cast<size_t>(args.groups) * kKeyElementsPerGroup;
  const size_t output_elements =
      static_cast<size_t>(args.groups) * kOutputElementsPerGroup;
  std::vector<__half> host_query(query_elements);
  std::vector<__half> host_key(key_elements);
  uint32_t random_state = 0x6d2b79f5u;
  for (__half& value : host_query) {
    value = __float2half_rn(random_half_value(&random_state));
  }
  for (__half& value : host_key) {
    value = __float2half_rn(random_half_value(&random_state));
  }

  __half* device_query = nullptr;
  __half* device_key = nullptr;
  float* device_baseline = nullptr;
  float* device_raw = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_query),
                        query_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_key),
                        key_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_baseline),
                        output_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_raw),
                        output_elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_query, host_query.data(),
                        query_elements * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_key, host_key.data(),
                        key_elements * sizeof(__half), cudaMemcpyHostToDevice));

  const dim3 baseline_grid(args.groups * 2);
  const dim3 raw_grid(args.raw_variant == "raw_m16n256_reuse_b" ? args.groups
                                                                   : args.groups * 2);
  auto launch_baseline = [&] {
    qk_panel_baseline_kernel<<<baseline_grid, kBaselineThreads>>>(
        device_query, device_key, device_baseline, args.groups);
    CUDA_CHECK(cudaGetLastError());
  };
  auto launch_raw = [&] {
    if (args.raw_variant == "raw_k4_vector") {
      qk_panel_raw_k4_vector_kernel<<<raw_grid, kRawThreads>>>(
          device_query, device_key, device_raw, args.groups);
    } else if (args.raw_variant == "raw_k16_stage") {
      qk_panel_raw_k16_stage_kernel<<<raw_grid, kRawThreads>>>(
          device_query, device_key, device_raw, args.groups);
    } else if (args.raw_variant == "raw_k16_double") {
      qk_panel_raw_k16_double_kernel<<<raw_grid, kRawThreads>>>(
          device_query, device_key, device_raw, args.groups);
    } else {
      qk_panel_raw_m16n256_reuse_b_kernel<<<raw_grid, kRawThreads>>>(
          device_query, device_key, device_raw, args.groups);
    }
    CUDA_CHECK(cudaGetLastError());
  };

  if (args.profile_only) {
    if (profile_kernel_selected(args, "baseline")) {
      launch_baseline();
    }
    if (profile_kernel_selected(args, "raw")) {
      launch_raw();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(device_raw));
    CUDA_CHECK(cudaFree(device_baseline));
    CUDA_CHECK(cudaFree(device_key));
    CUDA_CHECK(cudaFree(device_query));
    return EXIT_SUCCESS;
  }

  for (int warmup = 0; warmup < args.warmup; ++warmup) {
    launch_baseline();
    launch_raw();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  launch_baseline();
  launch_raw();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> host_baseline(output_elements);
  std::vector<float> host_raw(output_elements);
  CUDA_CHECK(cudaMemcpy(host_baseline.data(), device_baseline,
                        output_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_raw.data(), device_raw,
                        output_elements * sizeof(float), cudaMemcpyDeviceToHost));

  Exactness exactness;
  exactness.bitwise_equal = true;
  for (size_t index = 0; index < output_elements; ++index) {
    const uint32_t word_xor =
        float_bits(host_baseline[index]) ^ float_bits(host_raw[index]);
    exactness.xor_reduction ^= word_xor;
    exactness.max_word_xor = std::max(exactness.max_word_xor, word_xor);
    exactness.bitwise_equal &= word_xor == 0;
    exactness.mismatch_words += word_xor != 0;
    const float abs_error = std::fabs(host_baseline[index] - host_raw[index]);
    exactness.max_abs_error = std::max(exactness.max_abs_error, abs_error);
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  auto time_launches = [&](bool raw) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int launch = 0; launch < args.launches_per_sample; ++launch) {
      if (raw) {
        launch_raw();
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
  std::vector<double> raw_samples;
  baseline_samples.reserve(args.rounds);
  raw_samples.reserve(args.rounds);
  for (int round = 0; round < args.rounds; ++round) {
    double baseline_us = 0.0;
    double raw_us = 0.0;
    if ((round & 1) == 0) {
      baseline_us = time_launches(false);
      raw_us = time_launches(true);
    } else {
      raw_us = time_launches(true);
      baseline_us = time_launches(false);
    }
    baseline_samples.push_back(baseline_us);
    raw_samples.push_back(raw_us);
  }
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));

  const KernelResources baseline_resources = query_resources(
      qk_panel_baseline_kernel, kBaselineThreads, kBaselineQKWarps,
      args.device);
  KernelResources raw_resources;
  if (args.raw_variant == "raw_k4_vector") {
    raw_resources = query_resources(qk_panel_raw_k4_vector_kernel,
                                    kRawThreads, kRawQKWarps, args.device);
  } else if (args.raw_variant == "raw_k16_stage") {
    raw_resources = query_resources(qk_panel_raw_k16_stage_kernel,
                                    kRawThreads, kRawQKWarps, args.device);
  } else if (args.raw_variant == "raw_k16_double") {
    raw_resources = query_resources(qk_panel_raw_k16_double_kernel,
                                    kRawThreads, kRawQKWarps, args.device);
  } else {
    raw_resources = query_resources(qk_panel_raw_m16n256_reuse_b_kernel,
                                    kRawThreads, kRawQKWarps, args.device);
  }
  const TimingSummary baseline_timing = summarize(baseline_samples);
  const TimingSummary raw_timing = summarize(raw_samples);
  const PairSummary pairs = summarize_pairs(baseline_samples, raw_samples);

  CUDA_CHECK(cudaFree(device_raw));
  CUDA_CHECK(cudaFree(device_baseline));
  CUDA_CHECK(cudaFree(device_key));
  CUDA_CHECK(cudaFree(device_query));
  print_json(args, properties, runtime_version, sm_count, max_threads_per_sm,
             max_shared_per_sm, exactness, baseline_timing, raw_timing, pairs,
             baseline_resources, raw_resources);
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
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
    } else if (argument == "--warmup") {
      parse_int(&args.warmup);
    } else if (argument == "--rounds") {
      parse_int(&args.rounds);
    } else if (argument == "--launches-per-sample") {
      parse_int(&args.launches_per_sample);
    } else if (argument == "--profile-only") {
      args.profile_only = true;
    } else if (argument == "--raw-variant" && index + 1 < argc) {
      args.raw_variant = argv[++index];
      if (!is_raw_variant(args.raw_variant)) {
        std::cerr << "--raw-variant must be raw_k4_vector, raw_k16_stage, or "
                     "raw_k16_double, or raw_m16n256_reuse_b\n";
        return EXIT_FAILURE;
      }
    } else if (argument == "--profile-kernel" && index + 1 < argc) {
      args.profile_kernel = argv[++index];
      if (args.profile_kernel != "baseline" && args.profile_kernel != "raw" &&
          args.profile_kernel != "both") {
        std::cerr << "--profile-kernel must be baseline, raw, or both\n";
        return EXIT_FAILURE;
      }
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--device N] [--groups N] [--warmup N] [--rounds N]"
                   " [--launches-per-sample N] [--profile-only]"
                   " [--raw-variant raw_k4_vector|raw_k16_stage|raw_k16_double|"
                   "raw_m16n256_reuse_b]"
                   " [--profile-kernel baseline|raw|both]\n";
      return EXIT_FAILURE;
    }
  }
  return run(args);
}
