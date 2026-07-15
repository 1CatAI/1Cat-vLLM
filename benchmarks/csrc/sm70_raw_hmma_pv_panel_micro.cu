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
constexpr int kD = 256;
constexpr int kK = 32;
constexpr int kNativeN = 16;
constexpr int kNativeK = 16;
constexpr int kRawM = 8;
constexpr int kRawD = 32;
constexpr int kRawK = 4;
constexpr int kRawRegisters = 8;
constexpr int kBaselineThreads = 512;
constexpr int kRawThreads = 256;
constexpr int kPElementsPerGroup = kM * kK;
constexpr int kVElementsPerGroup = kK * kD;
constexpr int kOutputElementsPerGroup = kM * kD;

static_assert(kK / kRawK == 8);
static_assert(kD / kNativeN == 16);
static_assert(kD / kRawD == 8);

using NativeAFragment =
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kM, kNativeN, kNativeK,
                          __half, nvcuda::wmma::row_major>;
using NativeBFragment =
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kM, kNativeN, kNativeK,
                          __half, nvcuda::wmma::row_major>;
using NativeCFragment = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kM,
                                                kNativeN, kNativeK, float>;

struct Args {
  int device = 0;
  int groups = 1024;
  int warmup = 20;
  int rounds = 100;
  int launches_per_sample = 8;
  bool profile_only = false;
  std::string profile_kernel = "all";
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
  int resident_active_warps = 0;
  int threads_per_cta = 0;
  int warps_per_cta = 0;
};

struct Exactness {
  bool bitwise_equal = false;
  int mismatch_words = 0;
  uint32_t xor_reduction = 0;
  uint32_t max_word_xor = 0;
  float max_abs_error = 0.0f;
};

enum class Variant : int {
  kBaseline = 0,
  kRawSingle = 1,
  kRawPipelined = 2,
  kRawM16ReuseV = 3,
  kRawM16ReuseVK16Double = 4,
};

constexpr int kVariantCount = 5;

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

__device__ __forceinline__ void stage_p_to_shared(
    const __half* __restrict__ p, __half* __restrict__ shared_p, int thread,
    int thread_count) {
  constexpr int kHalfPerUint4 = sizeof(uint4) / sizeof(__half);
  constexpr int kPVectorCount = kPElementsPerGroup / kHalfPerUint4;
  const uint4* p_vectors = reinterpret_cast<const uint4*>(p);
  uint4* shared_vectors = reinterpret_cast<uint4*>(shared_p);

  for (int index = thread; index < kPVectorCount; index += thread_count) {
    shared_vectors[index] = __ldg(p_vectors + index);
  }
}

// These mappings are the full-warp SM70 m8n8k4 fragment mapping proven by the
// raw 16x32 order probe. The row.row B operand assigns one K row and four
// adjacent D values to each lane.
__device__ __forceinline__ int raw_a_row(int lane) {
  return (lane & 3) + ((lane >> 4) * 4);
}

__device__ __forceinline__ int raw_n_tile(int lane) {
  return (lane & 12) * 2;
}

__device__ __forceinline__ int raw_b_row_k(int lane) {
  return lane & 3;
}

__device__ __forceinline__ int raw_b_row_d(int lane) {
  return raw_n_tile(lane) + ((lane >> 4) * 4);
}

__device__ __forceinline__ int raw_c_row(int lane, int reg) {
  return (lane & 1) + (reg & 2) + ((lane >> 4) * 4);
}

__device__ __forceinline__ int raw_c_col(int lane, int reg) {
  return raw_n_tile(lane) + (lane & 2) + (reg & 4) + (reg & 1);
}

__device__ __forceinline__ uint64_t load_shared_half4(
    const __half* __restrict__ pointer) {
  const uint32_t address = static_cast<uint32_t>(
      __cvta_generic_to_shared(pointer));
  uint64_t packed;
  asm volatile("ld.volatile.shared.u64 %0, [%1];" : "=l"(packed)
               : "r"(address)
               : "memory");
  return packed;
}

__device__ __forceinline__ uint64_t load_global_half4(
    const __half* __restrict__ pointer) {
  uint64_t packed;
  // ld.global.nc.u64 lowers to the one aligned LDG.E.64 required for each
  // four-D row.row operand; scalar half loads are intentionally absent.
  asm volatile("ld.global.nc.u64 %0, [%1];" : "=l"(packed) : "l"(pointer)
               : "memory");
  return packed;
}

__device__ __forceinline__ void load_raw_operands(
    const __half* __restrict__ shared_p, const __half* __restrict__ v_group,
    int p_row, int lane, int d_offset, int k_offset, uint64_t& packed_p,
    uint64_t& packed_v) {
  packed_p = load_shared_half4(shared_p + p_row * kK + k_offset);
  const int v_row = k_offset + raw_b_row_k(lane);
  const int v_col = d_offset + raw_b_row_d(lane);
  packed_v = load_global_half4(v_group + v_row * kD + v_col);
}

__device__ __forceinline__ void mma_m8n8k4_row_row(
    float (&d)[kRawRegisters], uint32_t a0, uint32_t a1, uint32_t b0,
    uint32_t b1, const float (&c)[kRawRegisters]) {
  // The memory clobber pins the explicit next-K4 operand loads before the
  // current HMMA in the pipelined source schedule.
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, "
      "{%8, %9}, {%10, %11}, {%12, %13, %14, %15, %16, %17, %18, %19};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),
        "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]),
        "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]),
        "f"(c[7])
      : "memory");
}

__device__ __forceinline__ void raw_single_step(
    const __half* __restrict__ shared_p, const __half* __restrict__ v_group,
    int p_row, int lane, int d_offset, int k_offset,
    float (&accumulator)[kRawRegisters]) {
  uint64_t packed_p;
  uint64_t packed_v;
  load_raw_operands(shared_p, v_group, p_row, lane, d_offset, k_offset,
                    packed_p, packed_v);
  const uint32_t a0 = static_cast<uint32_t>(packed_p);
  const uint32_t a1 = static_cast<uint32_t>(packed_p >> 32);
  const uint32_t b0 = static_cast<uint32_t>(packed_v);
  const uint32_t b1 = static_cast<uint32_t>(packed_v >> 32);
  mma_m8n8k4_row_row(accumulator, a0, a1, b0, b1, accumulator);
}

__device__ __forceinline__ void mma_packed_raw_operands(
    float (&accumulator)[kRawRegisters], uint64_t packed_p,
    uint64_t packed_v) {
  const uint32_t a0 = static_cast<uint32_t>(packed_p);
  const uint32_t a1 = static_cast<uint32_t>(packed_p >> 32);
  const uint32_t b0 = static_cast<uint32_t>(packed_v);
  const uint32_t b1 = static_cast<uint32_t>(packed_v >> 32);
  mma_m8n8k4_row_row(accumulator, a0, a1, b0, b1, accumulator);
}

__device__ __forceinline__ void load_reuse_v_operands(
    const __half* __restrict__ shared_p, const __half* __restrict__ v_group,
    int lane, int d_offset, int k_offset, uint64_t& packed_p_top,
    uint64_t& packed_p_bottom, uint64_t& packed_v) {
  const int p_row = raw_a_row(lane);
  packed_p_top = load_shared_half4(shared_p + p_row * kK + k_offset);
  packed_p_bottom =
      load_shared_half4(shared_p + (p_row + kRawM) * kK + k_offset);
  const int v_row = k_offset + raw_b_row_k(lane);
  const int v_col = d_offset + raw_b_row_d(lane);
  packed_v = load_global_half4(v_group + v_row * kD + v_col);
}

__device__ __forceinline__ void mma_reuse_v_pair(
    float (&accumulator_top)[kRawRegisters],
    float (&accumulator_bottom)[kRawRegisters], uint64_t packed_p_top,
    uint64_t packed_p_bottom, uint64_t packed_v) {
  mma_packed_raw_operands(accumulator_top, packed_p_top, packed_v);
  mma_packed_raw_operands(accumulator_bottom, packed_p_bottom, packed_v);
}

__device__ __forceinline__ void raw_m16_reuse_v_step(
    const __half* __restrict__ shared_p, const __half* __restrict__ v_group,
    int lane, int d_offset, int k_offset,
    float (&accumulator_top)[kRawRegisters],
    float (&accumulator_bottom)[kRawRegisters]) {
  uint64_t packed_p_top;
  uint64_t packed_p_bottom;
  uint64_t packed_v;
  load_reuse_v_operands(shared_p, v_group, lane, d_offset, k_offset,
                        packed_p_top, packed_p_bottom, packed_v);
  mma_reuse_v_pair(accumulator_top, accumulator_bottom, packed_p_top,
                   packed_p_bottom, packed_v);
}

__device__ __forceinline__ void store_raw_output(
    float* __restrict__ output_group, int m_offset, int d_offset, int lane,
    const float (&accumulator)[kRawRegisters]) {
#pragma unroll
  for (int reg = 0; reg < kRawRegisters; ++reg) {
    const int row = m_offset + raw_c_row(lane, reg);
    const int column = d_offset + raw_c_col(lane, reg);
    output_group[row * kD + column] = accumulator[reg];
  }
}

extern "C" __global__ __launch_bounds__(kBaselineThreads, 2)
void pv_panel_baseline_kernel(const __half* __restrict__ p,
                              const __half* __restrict__ v,
                              float* __restrict__ output, int groups) {
  __shared__ __align__(16) __half shared_p[kPElementsPerGroup];

  const int group = blockIdx.x;
  if (group >= groups) {
    return;
  }
  const int thread = threadIdx.x;
  const __half* p_group =
      p + static_cast<int64_t>(group) * kPElementsPerGroup;
  stage_p_to_shared(p_group, shared_p, thread, kBaselineThreads);
  __syncthreads();

  const int warp = thread >> 5;
  const int d_offset = warp * kNativeN;
  const __half* v_group =
      v + static_cast<int64_t>(group) * kVElementsPerGroup;
  NativeAFragment a_fragment;
  NativeBFragment b_fragment;
  NativeCFragment accumulator;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);

  nvcuda::wmma::load_matrix_sync(a_fragment, shared_p, kK);
  nvcuda::wmma::load_matrix_sync(b_fragment, v_group + d_offset, kD);
  nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
  nvcuda::wmma::load_matrix_sync(a_fragment, shared_p + kNativeK, kK);
  nvcuda::wmma::load_matrix_sync(b_fragment,
                                 v_group + kNativeK * kD + d_offset, kD);
  nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);

  float* output_tile = output + static_cast<int64_t>(group) *
                                      kOutputElementsPerGroup + d_offset;
  nvcuda::wmma::store_matrix_sync(output_tile, accumulator, kD,
                                  nvcuda::wmma::mem_row_major);
}

extern "C" __global__ __launch_bounds__(kRawThreads, 4)
void pv_panel_raw_single_kernel(const __half* __restrict__ p,
                                const __half* __restrict__ v,
                                float* __restrict__ output, int groups) {
  __shared__ __align__(16) __half shared_p[kPElementsPerGroup];

  const int block = blockIdx.x;
  const int group = block >> 1;
  if (group >= groups) {
    return;
  }
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int m_offset = (block & 1) * kRawM;
  const int d_offset = warp * kRawD;
  const int p_row = m_offset + raw_a_row(lane);
  const __half* p_group =
      p + static_cast<int64_t>(group) * kPElementsPerGroup;
  const __half* v_group =
      v + static_cast<int64_t>(group) * kVElementsPerGroup;
  stage_p_to_shared(p_group, shared_p, thread, kRawThreads);
  __syncthreads();

  float accumulator[kRawRegisters] = {0.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f};
  // Each K4 call uses exactly one LDS.64 P load and one LDG.E.64 V load.
  // The eight calls deliberately preserve canonical FP32 accumulation order.
  raw_single_step(shared_p, v_group, p_row, lane, d_offset, 0, accumulator);
  raw_single_step(shared_p, v_group, p_row, lane, d_offset, 4, accumulator);
  raw_single_step(shared_p, v_group, p_row, lane, d_offset, 8, accumulator);
  raw_single_step(shared_p, v_group, p_row, lane, d_offset, 12, accumulator);
  raw_single_step(shared_p, v_group, p_row, lane, d_offset, 16, accumulator);
  raw_single_step(shared_p, v_group, p_row, lane, d_offset, 20, accumulator);
  raw_single_step(shared_p, v_group, p_row, lane, d_offset, 24, accumulator);
  raw_single_step(shared_p, v_group, p_row, lane, d_offset, 28, accumulator);

  float* output_group = output + static_cast<int64_t>(group) *
                                      kOutputElementsPerGroup;
  store_raw_output(output_group, m_offset, d_offset, lane, accumulator);
}

extern "C" __global__ __launch_bounds__(kRawThreads, 4)
void pv_panel_raw_pipelined_kernel(const __half* __restrict__ p,
                                   const __half* __restrict__ v,
                                   float* __restrict__ output, int groups) {
  __shared__ __align__(16) __half shared_p[kPElementsPerGroup];

  const int block = blockIdx.x;
  const int group = block >> 1;
  if (group >= groups) {
    return;
  }
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int m_offset = (block & 1) * kRawM;
  const int d_offset = warp * kRawD;
  const int p_row = m_offset + raw_a_row(lane);
  const __half* p_group =
      p + static_cast<int64_t>(group) * kPElementsPerGroup;
  const __half* v_group =
      v + static_cast<int64_t>(group) * kVElementsPerGroup;
  stage_p_to_shared(p_group, shared_p, thread, kRawThreads);
  __syncthreads();

  float accumulator[kRawRegisters] = {0.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f};
  uint64_t p0_k0;
  uint64_t v0_k0;
  uint64_t p0_k4;
  uint64_t v0_k4;
  uint64_t p0_k8;
  uint64_t v0_k8;
  uint64_t p0_k12;
  uint64_t v0_k12;
  uint64_t p1_k16;
  uint64_t v1_k16;
  uint64_t p1_k20;
  uint64_t v1_k20;
  uint64_t p1_k24;
  uint64_t v1_k24;
  uint64_t p1_k28;
  uint64_t v1_k28;

  // Bank 0 is a complete K16 stage: four LDS.64 P loads plus four LDG.E.64
  // V loads produce eight P and eight V 32-bit operand words before its four
  // canonical HMMAs. Bank 1 is filled during those HMMAs and becomes the
  // next complete K16 stage without changing FP32 accumulation order.
  load_raw_operands(shared_p, v_group, p_row, lane, d_offset, 0, p0_k0,
                    v0_k0);
  load_raw_operands(shared_p, v_group, p_row, lane, d_offset, 4, p0_k4,
                    v0_k4);
  load_raw_operands(shared_p, v_group, p_row, lane, d_offset, 8, p0_k8,
                    v0_k8);
  load_raw_operands(shared_p, v_group, p_row, lane, d_offset, 12, p0_k12,
                    v0_k12);

  load_raw_operands(shared_p, v_group, p_row, lane, d_offset, 16, p1_k16,
                    v1_k16);
  mma_packed_raw_operands(accumulator, p0_k0, v0_k0);
  load_raw_operands(shared_p, v_group, p_row, lane, d_offset, 20, p1_k20,
                    v1_k20);
  mma_packed_raw_operands(accumulator, p0_k4, v0_k4);
  load_raw_operands(shared_p, v_group, p_row, lane, d_offset, 24, p1_k24,
                    v1_k24);
  mma_packed_raw_operands(accumulator, p0_k8, v0_k8);
  load_raw_operands(shared_p, v_group, p_row, lane, d_offset, 28, p1_k28,
                    v1_k28);
  mma_packed_raw_operands(accumulator, p0_k12, v0_k12);

  mma_packed_raw_operands(accumulator, p1_k16, v1_k16);
  mma_packed_raw_operands(accumulator, p1_k20, v1_k20);
  mma_packed_raw_operands(accumulator, p1_k24, v1_k24);
  mma_packed_raw_operands(accumulator, p1_k28, v1_k28);

  float* output_group = output + static_cast<int64_t>(group) *
                                      kOutputElementsPerGroup;
  store_raw_output(output_group, m_offset, d_offset, lane, accumulator);
}

extern "C" __global__ __launch_bounds__(kRawThreads, 4)
void pv_panel_raw_m16d256_reuse_v_kernel(const __half* __restrict__ p,
                                         const __half* __restrict__ v,
                                         float* __restrict__ output,
                                         int groups) {
  __shared__ __align__(16) __half shared_p[kPElementsPerGroup];

  const int group = blockIdx.x;
  if (group >= groups) {
    return;
  }
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int d_offset = warp * kRawD;
  const __half* p_group =
      p + static_cast<int64_t>(group) * kPElementsPerGroup;
  const __half* v_group =
      v + static_cast<int64_t>(group) * kVElementsPerGroup;
  stage_p_to_shared(p_group, shared_p, thread, kRawThreads);
  __syncthreads();

  float accumulator_top[kRawRegisters] = {0.0f, 0.0f, 0.0f, 0.0f,
                                            0.0f, 0.0f, 0.0f, 0.0f};
  float accumulator_bottom[kRawRegisters] = {0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f};
  // Every K4 issues top then bottom with one shared V operand register pair.
  raw_m16_reuse_v_step(shared_p, v_group, lane, d_offset, 0,
                       accumulator_top, accumulator_bottom);
  raw_m16_reuse_v_step(shared_p, v_group, lane, d_offset, 4,
                       accumulator_top, accumulator_bottom);
  raw_m16_reuse_v_step(shared_p, v_group, lane, d_offset, 8,
                       accumulator_top, accumulator_bottom);
  raw_m16_reuse_v_step(shared_p, v_group, lane, d_offset, 12,
                       accumulator_top, accumulator_bottom);
  raw_m16_reuse_v_step(shared_p, v_group, lane, d_offset, 16,
                       accumulator_top, accumulator_bottom);
  raw_m16_reuse_v_step(shared_p, v_group, lane, d_offset, 20,
                       accumulator_top, accumulator_bottom);
  raw_m16_reuse_v_step(shared_p, v_group, lane, d_offset, 24,
                       accumulator_top, accumulator_bottom);
  raw_m16_reuse_v_step(shared_p, v_group, lane, d_offset, 28,
                       accumulator_top, accumulator_bottom);

  float* output_group = output + static_cast<int64_t>(group) *
                                      kOutputElementsPerGroup;
  store_raw_output(output_group, 0, d_offset, lane, accumulator_top);
  store_raw_output(output_group, kRawM, d_offset, lane, accumulator_bottom);
}

extern "C" __global__ __launch_bounds__(kRawThreads, 4)
void pv_panel_raw_m16d256_reuse_v_k16_double_kernel(
    const __half* __restrict__ p, const __half* __restrict__ v,
    float* __restrict__ output, int groups) {
  __shared__ __align__(16) __half shared_p[kPElementsPerGroup];

  const int group = blockIdx.x;
  if (group >= groups) {
    return;
  }
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int d_offset = warp * kRawD;
  const __half* p_group =
      p + static_cast<int64_t>(group) * kPElementsPerGroup;
  const __half* v_group =
      v + static_cast<int64_t>(group) * kVElementsPerGroup;
  stage_p_to_shared(p_group, shared_p, thread, kRawThreads);
  __syncthreads();

  float accumulator_top[kRawRegisters] = {0.0f, 0.0f, 0.0f, 0.0f,
                                            0.0f, 0.0f, 0.0f, 0.0f};
  float accumulator_bottom[kRawRegisters] = {0.0f, 0.0f, 0.0f, 0.0f,
                                               0.0f, 0.0f, 0.0f, 0.0f};
  uint64_t p0_top_k0;
  uint64_t p0_bottom_k0;
  uint64_t v0_k0;
  uint64_t p0_top_k4;
  uint64_t p0_bottom_k4;
  uint64_t v0_k4;
  uint64_t p0_top_k8;
  uint64_t p0_bottom_k8;
  uint64_t v0_k8;
  uint64_t p0_top_k12;
  uint64_t p0_bottom_k12;
  uint64_t v0_k12;
  uint64_t p1_top_k16;
  uint64_t p1_bottom_k16;
  uint64_t v1_k16;
  uint64_t p1_top_k20;
  uint64_t p1_bottom_k20;
  uint64_t v1_k20;
  uint64_t p1_top_k24;
  uint64_t p1_bottom_k24;
  uint64_t v1_k24;
  uint64_t p1_top_k28;
  uint64_t p1_bottom_k28;
  uint64_t v1_k28;

  load_reuse_v_operands(shared_p, v_group, lane, d_offset, 0, p0_top_k0,
                        p0_bottom_k0, v0_k0);
  load_reuse_v_operands(shared_p, v_group, lane, d_offset, 4, p0_top_k4,
                        p0_bottom_k4, v0_k4);
  load_reuse_v_operands(shared_p, v_group, lane, d_offset, 8, p0_top_k8,
                        p0_bottom_k8, v0_k8);
  load_reuse_v_operands(shared_p, v_group, lane, d_offset, 12, p0_top_k12,
                        p0_bottom_k12, v0_k12);

  load_reuse_v_operands(shared_p, v_group, lane, d_offset, 16, p1_top_k16,
                        p1_bottom_k16, v1_k16);
  mma_reuse_v_pair(accumulator_top, accumulator_bottom, p0_top_k0,
                   p0_bottom_k0, v0_k0);
  load_reuse_v_operands(shared_p, v_group, lane, d_offset, 20, p1_top_k20,
                        p1_bottom_k20, v1_k20);
  mma_reuse_v_pair(accumulator_top, accumulator_bottom, p0_top_k4,
                   p0_bottom_k4, v0_k4);
  load_reuse_v_operands(shared_p, v_group, lane, d_offset, 24, p1_top_k24,
                        p1_bottom_k24, v1_k24);
  mma_reuse_v_pair(accumulator_top, accumulator_bottom, p0_top_k8,
                   p0_bottom_k8, v0_k8);
  load_reuse_v_operands(shared_p, v_group, lane, d_offset, 28, p1_top_k28,
                        p1_bottom_k28, v1_k28);
  mma_reuse_v_pair(accumulator_top, accumulator_bottom, p0_top_k12,
                   p0_bottom_k12, v0_k12);

  mma_reuse_v_pair(accumulator_top, accumulator_bottom, p1_top_k16,
                   p1_bottom_k16, v1_k16);
  mma_reuse_v_pair(accumulator_top, accumulator_bottom, p1_top_k20,
                   p1_bottom_k20, v1_k20);
  mma_reuse_v_pair(accumulator_top, accumulator_bottom, p1_top_k24,
                   p1_bottom_k24, v1_k24);
  mma_reuse_v_pair(accumulator_top, accumulator_bottom, p1_top_k28,
                   p1_bottom_k28, v1_k28);

  float* output_group = output + static_cast<int64_t>(group) *
                                      kOutputElementsPerGroup;
  store_raw_output(output_group, 0, d_offset, lane, accumulator_top);
  store_raw_output(output_group, kRawM, d_offset, lane, accumulator_bottom);
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

Exactness compare_outputs(const std::vector<float>& reference,
                          const std::vector<float>& candidate) {
  Exactness result;
  result.bitwise_equal = true;
  for (size_t index = 0; index < reference.size(); ++index) {
    const uint32_t word_xor =
        float_bits(reference[index]) ^ float_bits(candidate[index]);
    result.xor_reduction ^= word_xor;
    result.max_word_xor = std::max(result.max_word_xor, word_xor);
    result.bitwise_equal &= word_xor == 0;
    result.mismatch_words += word_xor != 0;
    const float abs_error = std::fabs(reference[index] - candidate[index]);
    if (std::isnan(abs_error)) {
      result.max_abs_error = std::numeric_limits<float>::infinity();
    } else {
      result.max_abs_error = std::max(result.max_abs_error, abs_error);
    }
  }
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
  result.local_bytes_per_thread = attributes.localSizeBytes;
  result.active_ctas_per_sm = active_ctas;
  result.threads_per_cta = threads_per_cta;
  result.warps_per_cta = threads_per_cta / 32;
  result.resident_active_warps = active_ctas * result.warps_per_cta;
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
                    << static_cast<int>(character) << std::dec
                    << std::setfill(' ');
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

void print_pairs(const PairSummary& pairs) {
  std::cout << "{\"count\": " << pairs.count
            << ", \"candidate_faster\": " << pairs.candidate_faster
            << ", \"baseline_faster\": " << pairs.baseline_faster
            << ", \"ties\": " << pairs.ties
            << ", \"candidate_minus_baseline_median_us\": "
            << pairs.candidate_minus_baseline_median_us
            << ", \"candidate_minus_baseline_mean_us\": "
            << pairs.candidate_minus_baseline_mean_us << '}';
}

void print_exactness(const Exactness& exactness) {
  std::cout << "{\"bitwise_equal\": "
            << (exactness.bitwise_equal ? "true" : "false")
            << ", \"mismatch_words\": " << exactness.mismatch_words
            << ", \"xor\": {\"max_word\": " << exactness.max_word_xor
            << ", \"reduction\": " << exactness.xor_reduction << '}'
            << ", \"max_abs_error\": " << exactness.max_abs_error << '}';
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
            << ", \"resident_active_warps\": "
            << resources.resident_active_warps
            << ", \"threads_per_cta\": " << resources.threads_per_cta
            << ", \"warps_per_cta\": " << resources.warps_per_cta << '}';
}

double speedup_pct(const TimingSummary& baseline,
                   const TimingSummary& candidate) {
  return 100.0 * (baseline.median_us - candidate.median_us) /
         baseline.median_us;
}

void print_json(const Args& args, const cudaDeviceProp& properties,
                int runtime_version, int sm_count, int max_threads_per_sm,
                int max_shared_per_sm, const Exactness& baseline_raw_single,
                const Exactness& baseline_raw_pipelined,
                const Exactness& raw_single_raw_pipelined,
                const Exactness& baseline_raw_m16_reuse_v,
                const Exactness& baseline_raw_m16_reuse_v_k16_double,
                const Exactness& raw_m16_reuse_v_pair,
                const TimingSummary& baseline_timing,
                const TimingSummary& raw_single_timing,
                const TimingSummary& raw_pipelined_timing,
                const TimingSummary& raw_m16_reuse_v_timing,
                const TimingSummary& raw_m16_reuse_v_k16_double_timing,
                const PairSummary& raw_single_pairs,
                const PairSummary& raw_pipelined_pairs,
                const PairSummary& raw_m16_reuse_v_pairs,
                const PairSummary& raw_m16_reuse_v_k16_double_pairs,
                const KernelResources& baseline_resources,
                const KernelResources& raw_single_resources,
                const KernelResources& raw_pipelined_resources,
                const KernelResources& raw_m16_reuse_v_resources,
                const KernelResources& raw_m16_reuse_v_k16_double_resources) {
  const bool all_bitwise_equal = baseline_raw_single.bitwise_equal &&
                                 baseline_raw_pipelined.bitwise_equal &&
                                 raw_single_raw_pipelined.bitwise_equal &&
                                 baseline_raw_m16_reuse_v.bitwise_equal &&
                                 baseline_raw_m16_reuse_v_k16_double.bitwise_equal &&
                                 raw_m16_reuse_v_pair.bitwise_equal;
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
  std::cout << "    \"output\": \"[groups, M16, D256]\",\n";
  std::cout << "    \"K\": 32,\n";
  std::cout << "    \"p_layout\": \"[group, M16, K32] staged in shared\",\n";
  std::cout << "    \"v_layout\": \"[group, K32, D256] row-major token-major stride 256\"\n";
  std::cout << "  },\n";
  std::cout << "  \"paths\": {\n";
  std::cout << "    \"baseline\": \"A: 1 CTA/group, BM16xD256, 512 threads, 16 warps x D16; two native m16n16k16 row.row\",\n";
  std::cout << "    \"raw_single\": \"legacy BM8 split/raw-vector: 2 CTA/group, 256 threads each; V is read by both M halves\",\n";
  std::cout << "    \"raw_pipelined\": \"legacy BM8 split/K16-stage-double: 2 CTA/group; V is read by both M halves\",\n";
  std::cout << "    \"raw_m16d256_reuse_v\": \"1 CTA/group, 256 threads, 8 warps x D32; one V LDG.E.64 feeds top then bottom raw row.row HMMA per K4\",\n";
  std::cout << "    \"raw_m16d256_reuse_v_k16_double\": \"same M16 V reuse with a complete K16 operand bank and ping-pong next K16\",\n";
  std::cout << "    \"raw_k4_offsets\": [0, 4, 8, 12, 16, 20, 24, 28]\n";
  std::cout << "  },\n";
  std::cout << "  \"loader_classes\": {\n";
  std::cout << "    \"raw_scalar\": \"not_built; never used for a structural gate\",\n";
  std::cout << "    \"raw_vector\": \"raw_single: 8 K4 vector P/V operand loads\",\n";
  std::cout << "    \"k16_stage_double\": \"raw_pipelined: two named K16 operand banks\",\n";
  std::cout << "    \"raw_m16_reuse_v\": \"8 vector V operands reused by top and bottom\",\n";
  std::cout << "    \"raw_m16_reuse_v_k16_double\": \"M16 reuse-V with two named K16 banks\"\n";
  std::cout << "  },\n";
  std::cout << "  \"launch_topology\": {\n";
  std::cout << "    \"baseline\": {\"ctas_per_group\": 1, \"threads_per_cta\": 512},\n";
  std::cout << "    \"raw_single\": {\"ctas_per_group\": 2, \"threads_per_cta\": 256},\n";
  std::cout << "    \"raw_pipelined\": {\"ctas_per_group\": 2, \"threads_per_cta\": 256},\n";
  std::cout << "    \"raw_m16d256_reuse_v\": {\"ctas_per_group\": 1, \"threads_per_cta\": 256},\n";
  std::cout << "    \"raw_m16d256_reuse_v_k16_double\": {\"ctas_per_group\": 1, \"threads_per_cta\": 256}\n";
  std::cout << "  },\n";
  std::cout << "  \"exactness\": {\n";
  std::cout << "    \"word_dtype\": \"uint32\",\n";
  std::cout << "    \"word_count_per_comparison\": "
            << static_cast<int64_t>(args.groups) * kOutputElementsPerGroup
            << ",\n";
  std::cout << "    \"baseline_vs_raw_single\": ";
  print_exactness(baseline_raw_single);
  std::cout << ",\n    \"baseline_vs_raw_pipelined\": ";
  print_exactness(baseline_raw_pipelined);
  std::cout << ",\n    \"raw_single_vs_raw_pipelined\": ";
  print_exactness(raw_single_raw_pipelined);
  std::cout << ",\n    \"baseline_vs_raw_m16d256_reuse_v\": ";
  print_exactness(baseline_raw_m16_reuse_v);
  std::cout << ",\n    \"baseline_vs_raw_m16d256_reuse_v_k16_double\": ";
  print_exactness(baseline_raw_m16_reuse_v_k16_double);
  std::cout << ",\n    \"raw_m16d256_reuse_v_pair\": ";
  print_exactness(raw_m16_reuse_v_pair);
  std::cout << ",\n    \"all_bitwise_equal\": "
            << (all_bitwise_equal ? "true" : "false") << "\n";
  std::cout << "  },\n";
  std::cout << "  \"timing\": {\n";
  std::cout << "    \"unit\": \"us per grid launch\",\n";
  std::cout << "    \"baseline\": ";
  print_timing(baseline_timing);
  std::cout << ",\n    \"raw_single\": ";
  print_timing(raw_single_timing);
  std::cout << ",\n    \"raw_pipelined\": ";
  print_timing(raw_pipelined_timing);
  std::cout << ",\n    \"raw_m16d256_reuse_v\": ";
  print_timing(raw_m16_reuse_v_timing);
  std::cout << ",\n    \"raw_m16d256_reuse_v_k16_double\": ";
  print_timing(raw_m16_reuse_v_k16_double_timing);
  std::cout << ",\n    \"raw_single_speedup_vs_baseline_pct\": "
            << speedup_pct(baseline_timing, raw_single_timing);
  std::cout << ",\n    \"raw_pipelined_speedup_vs_baseline_pct\": "
            << speedup_pct(baseline_timing, raw_pipelined_timing);
  std::cout << ",\n    \"raw_m16d256_reuse_v_speedup_vs_baseline_pct\": "
            << speedup_pct(baseline_timing, raw_m16_reuse_v_timing);
  std::cout << ",\n    \"raw_m16d256_reuse_v_k16_double_speedup_vs_baseline_pct\": "
            << speedup_pct(baseline_timing,
                           raw_m16_reuse_v_k16_double_timing)
            << "\n";
  std::cout << "  },\n";
  std::cout << "  \"pairs\": {\n";
  std::cout << "    \"raw_single_vs_baseline\": ";
  print_pairs(raw_single_pairs);
  std::cout << ",\n    \"raw_pipelined_vs_baseline\": ";
  print_pairs(raw_pipelined_pairs);
  std::cout << ",\n    \"raw_m16d256_reuse_v_vs_baseline\": ";
  print_pairs(raw_m16_reuse_v_pairs);
  std::cout << ",\n    \"raw_m16d256_reuse_v_k16_double_vs_baseline\": ";
  print_pairs(raw_m16_reuse_v_k16_double_pairs);
  std::cout << "\n  },\n";
  std::cout << "  \"measurement\": {\n";
  std::cout << "    \"warmup_rounds\": " << args.warmup << ",\n";
  std::cout << "    \"warmup_launches_per_variant\": " << args.warmup
            << ",\n";
  std::cout << "    \"rounds\": " << args.rounds << ",\n";
  std::cout << "    \"launches_per_sample\": " << args.launches_per_sample
            << ",\n";
  std::cout << "    \"interleaving\": \"five-variant start order rotates every round\"\n";
  std::cout << "  },\n";
  std::cout << "  \"resources\": {\n";
  std::cout << "    \"baseline\": ";
  print_resources(baseline_resources);
  std::cout << ",\n    \"raw_single\": ";
  print_resources(raw_single_resources);
  std::cout << ",\n    \"raw_pipelined\": ";
  print_resources(raw_pipelined_resources);
  std::cout << ",\n    \"raw_m16d256_reuse_v\": ";
  print_resources(raw_m16_reuse_v_resources);
  std::cout << ",\n    \"raw_m16d256_reuse_v_k16_double\": ";
  print_resources(raw_m16_reuse_v_k16_double_resources);
  std::cout << "\n  }\n";
  std::cout << "}\n";
}

bool profile_kernel_selected(const Args& args, const char* kernel) {
  return args.profile_kernel == "all" || args.profile_kernel == kernel;
}

Variant variant_from_index(int index) {
  return static_cast<Variant>(index % kVariantCount);
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
  if (args.groups < 1 || args.warmup < 0 || args.rounds < 1 ||
      args.launches_per_sample < 1) {
    std::cerr << "groups, rounds, and launches-per-sample must be positive; "
                 "warmup cannot be negative\n";
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

  const size_t p_elements =
      static_cast<size_t>(args.groups) * kPElementsPerGroup;
  const size_t v_elements =
      static_cast<size_t>(args.groups) * kVElementsPerGroup;
  const size_t output_elements =
      static_cast<size_t>(args.groups) * kOutputElementsPerGroup;
  std::vector<__half> host_p(p_elements);
  std::vector<__half> host_v(v_elements);
  uint32_t random_state = 0x6d2b79f5u;
  for (__half& value : host_p) {
    value = __float2half_rn(random_half_value(&random_state));
  }
  for (__half& value : host_v) {
    value = __float2half_rn(random_half_value(&random_state));
  }

  __half* device_p = nullptr;
  __half* device_v = nullptr;
  float* device_baseline = nullptr;
  float* device_raw_single = nullptr;
  float* device_raw_pipelined = nullptr;
  float* device_raw_m16_reuse_v = nullptr;
  float* device_raw_m16_reuse_v_k16_double = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_p),
                        p_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_v),
                        v_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_baseline),
                        output_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_raw_single),
                        output_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_raw_pipelined),
                        output_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_raw_m16_reuse_v),
                        output_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_raw_m16_reuse_v_k16_double),
      output_elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_p, host_p.data(), p_elements * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_v, host_v.data(), v_elements * sizeof(__half),
                        cudaMemcpyHostToDevice));

  const dim3 baseline_grid(args.groups);
  const dim3 raw_split_grid(args.groups * 2);
  const dim3 raw_m16_grid(args.groups);
  auto launch = [&](Variant variant) {
    switch (variant) {
      case Variant::kBaseline:
        pv_panel_baseline_kernel<<<baseline_grid, kBaselineThreads>>>(
            device_p, device_v, device_baseline, args.groups);
        break;
      case Variant::kRawSingle:
        pv_panel_raw_single_kernel<<<raw_split_grid, kRawThreads>>>(
            device_p, device_v, device_raw_single, args.groups);
        break;
      case Variant::kRawPipelined:
        pv_panel_raw_pipelined_kernel<<<raw_split_grid, kRawThreads>>>(
            device_p, device_v, device_raw_pipelined, args.groups);
        break;
      case Variant::kRawM16ReuseV:
        pv_panel_raw_m16d256_reuse_v_kernel<<<raw_m16_grid, kRawThreads>>>(
            device_p, device_v, device_raw_m16_reuse_v, args.groups);
        break;
      case Variant::kRawM16ReuseVK16Double:
        pv_panel_raw_m16d256_reuse_v_k16_double_kernel<<<raw_m16_grid,
                                                         kRawThreads>>>(
            device_p, device_v, device_raw_m16_reuse_v_k16_double,
            args.groups);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
  };

  if (args.profile_only) {
    if (profile_kernel_selected(args, "baseline")) {
      launch(Variant::kBaseline);
    }
    if (profile_kernel_selected(args, "raw_single")) {
      launch(Variant::kRawSingle);
    }
    if (profile_kernel_selected(args, "raw_pipelined")) {
      launch(Variant::kRawPipelined);
    }
    if (profile_kernel_selected(args, "raw_m16d256_reuse_v")) {
      launch(Variant::kRawM16ReuseV);
    }
    if (profile_kernel_selected(args,
                                "raw_m16d256_reuse_v_k16_double")) {
      launch(Variant::kRawM16ReuseVK16Double);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(device_raw_m16_reuse_v_k16_double));
    CUDA_CHECK(cudaFree(device_raw_m16_reuse_v));
    CUDA_CHECK(cudaFree(device_raw_pipelined));
    CUDA_CHECK(cudaFree(device_raw_single));
    CUDA_CHECK(cudaFree(device_baseline));
    CUDA_CHECK(cudaFree(device_v));
    CUDA_CHECK(cudaFree(device_p));
    return EXIT_SUCCESS;
  }

  for (int warmup = 0; warmup < args.warmup; ++warmup) {
    for (int position = 0; position < kVariantCount; ++position) {
      launch(variant_from_index(warmup + position));
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  launch(Variant::kBaseline);
  launch(Variant::kRawSingle);
  launch(Variant::kRawPipelined);
  launch(Variant::kRawM16ReuseV);
  launch(Variant::kRawM16ReuseVK16Double);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> host_baseline(output_elements);
  std::vector<float> host_raw_single(output_elements);
  std::vector<float> host_raw_pipelined(output_elements);
  std::vector<float> host_raw_m16_reuse_v(output_elements);
  std::vector<float> host_raw_m16_reuse_v_k16_double(output_elements);
  CUDA_CHECK(cudaMemcpy(host_baseline.data(), device_baseline,
                        output_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_raw_single.data(), device_raw_single,
                        output_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_raw_pipelined.data(), device_raw_pipelined,
                        output_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_raw_m16_reuse_v.data(), device_raw_m16_reuse_v,
                        output_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_raw_m16_reuse_v_k16_double.data(),
                        device_raw_m16_reuse_v_k16_double,
                        output_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));

  const Exactness baseline_raw_single =
      compare_outputs(host_baseline, host_raw_single);
  const Exactness baseline_raw_pipelined =
      compare_outputs(host_baseline, host_raw_pipelined);
  const Exactness raw_single_raw_pipelined =
      compare_outputs(host_raw_single, host_raw_pipelined);
  const Exactness baseline_raw_m16_reuse_v =
      compare_outputs(host_baseline, host_raw_m16_reuse_v);
  const Exactness baseline_raw_m16_reuse_v_k16_double =
      compare_outputs(host_baseline, host_raw_m16_reuse_v_k16_double);
  const Exactness raw_m16_reuse_v_pair = compare_outputs(
      host_raw_m16_reuse_v, host_raw_m16_reuse_v_k16_double);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  auto time_launches = [&](Variant variant) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int launch_index = 0; launch_index < args.launches_per_sample;
         ++launch_index) {
      launch(variant);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    return static_cast<double>(elapsed_ms) * 1000.0 /
           args.launches_per_sample;
  };

  std::array<std::vector<double>, kVariantCount> samples;
  for (std::vector<double>& variant_samples : samples) {
    variant_samples.reserve(args.rounds);
  }
  for (int round = 0; round < args.rounds; ++round) {
    for (int position = 0; position < kVariantCount; ++position) {
      const Variant variant = variant_from_index(round + position);
      samples[static_cast<int>(variant)].push_back(time_launches(variant));
    }
  }
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));

  const KernelResources baseline_resources =
      query_resources(pv_panel_baseline_kernel, kBaselineThreads);
  const KernelResources raw_single_resources =
      query_resources(pv_panel_raw_single_kernel, kRawThreads);
  const KernelResources raw_pipelined_resources =
      query_resources(pv_panel_raw_pipelined_kernel, kRawThreads);
  const KernelResources raw_m16_reuse_v_resources =
      query_resources(pv_panel_raw_m16d256_reuse_v_kernel, kRawThreads);
  const KernelResources raw_m16_reuse_v_k16_double_resources = query_resources(
      pv_panel_raw_m16d256_reuse_v_k16_double_kernel, kRawThreads);
  const TimingSummary baseline_timing = summarize(samples[0]);
  const TimingSummary raw_single_timing = summarize(samples[1]);
  const TimingSummary raw_pipelined_timing = summarize(samples[2]);
  const TimingSummary raw_m16_reuse_v_timing = summarize(samples[3]);
  const TimingSummary raw_m16_reuse_v_k16_double_timing =
      summarize(samples[4]);
  const PairSummary raw_single_pairs = summarize_pairs(samples[0], samples[1]);
  const PairSummary raw_pipelined_pairs =
      summarize_pairs(samples[0], samples[2]);
  const PairSummary raw_m16_reuse_v_pairs =
      summarize_pairs(samples[0], samples[3]);
  const PairSummary raw_m16_reuse_v_k16_double_pairs =
      summarize_pairs(samples[0], samples[4]);

  CUDA_CHECK(cudaFree(device_raw_m16_reuse_v_k16_double));
  CUDA_CHECK(cudaFree(device_raw_m16_reuse_v));
  CUDA_CHECK(cudaFree(device_raw_pipelined));
  CUDA_CHECK(cudaFree(device_raw_single));
  CUDA_CHECK(cudaFree(device_baseline));
  CUDA_CHECK(cudaFree(device_v));
  CUDA_CHECK(cudaFree(device_p));
  print_json(args, properties, runtime_version, sm_count, max_threads_per_sm,
             max_shared_per_sm, baseline_raw_single, baseline_raw_pipelined,
             raw_single_raw_pipelined, baseline_raw_m16_reuse_v,
             baseline_raw_m16_reuse_v_k16_double, raw_m16_reuse_v_pair,
             baseline_timing, raw_single_timing, raw_pipelined_timing,
             raw_m16_reuse_v_timing, raw_m16_reuse_v_k16_double_timing,
             raw_single_pairs, raw_pipelined_pairs, raw_m16_reuse_v_pairs,
             raw_m16_reuse_v_k16_double_pairs, baseline_resources,
             raw_single_resources, raw_pipelined_resources,
             raw_m16_reuse_v_resources,
             raw_m16_reuse_v_k16_double_resources);
  return EXIT_SUCCESS;
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--device" && index + 1 < argc) {
      args.device = std::stoi(argv[++index]);
    } else if (argument == "--groups" && index + 1 < argc) {
      args.groups = std::stoi(argv[++index]);
    } else if (argument == "--warmup" && index + 1 < argc) {
      args.warmup = std::stoi(argv[++index]);
    } else if (argument == "--rounds" && index + 1 < argc) {
      args.rounds = std::stoi(argv[++index]);
    } else if (argument == "--launches-per-sample" && index + 1 < argc) {
      args.launches_per_sample = std::stoi(argv[++index]);
    } else if (argument == "--profile-only") {
      args.profile_only = true;
    } else if (argument == "--profile-kernel" && index + 1 < argc) {
      args.profile_kernel = argv[++index];
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--device N] [--groups N] [--warmup N] [--rounds N]"
                << " [--launches-per-sample N] [--profile-only]"
                << " [--profile-kernel baseline|raw_single|raw_pipelined|"
                << "raw_m16d256_reuse_v|"
                << "raw_m16d256_reuse_v_k16_double|all]\n";
      std::exit(EXIT_FAILURE);
    }
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  return run(parse_args(argc, argv));
}
