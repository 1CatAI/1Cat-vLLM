// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cuda_fp16.h>
#include <mma.h>

namespace flashinfer {
namespace attention {
namespace sm70 {

// Matches FlashInfer's init/update convention for the pointer control below.
enum class MMAMode : unsigned int {
  kInit = 0U,
  kInplaceUpdate = 1U,
};

constexpr int kVoltaMmaM = 16;
constexpr int kVoltaMmaN = 16;
constexpr int kVoltaMmaK = 16;

using AFragment = nvcuda::wmma::fragment<
    nvcuda::wmma::matrix_a, kVoltaMmaM, kVoltaMmaN, kVoltaMmaK, __half,
    nvcuda::wmma::row_major>;
using QKBFragment = nvcuda::wmma::fragment<
    nvcuda::wmma::matrix_b, kVoltaMmaM, kVoltaMmaN, kVoltaMmaK, __half,
    nvcuda::wmma::col_major>;
using PVBFragment = nvcuda::wmma::fragment<
    nvcuda::wmma::matrix_b, kVoltaMmaM, kVoltaMmaN, kVoltaMmaK, __half,
    nvcuda::wmma::row_major>;
using AccumulatorFragment =
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kVoltaMmaM,
                           kVoltaMmaN, kVoltaMmaK, float>;

__device__ __forceinline__ void load_a_fragment(AFragment& fragment,
                                                const __half* source,
                                                int leading_dimension) {
  nvcuda::wmma::load_matrix_sync(fragment, source, leading_dimension);
}

// source is the physical row-major K tile [N, K]. The col-major WMMA view
// presents it as logical B[K, N] for Q * K^T.
__device__ __forceinline__ void load_qk_b_fragment(QKBFragment& fragment,
                                                   const __half* source,
                                                   int leading_dimension) {
  nvcuda::wmma::load_matrix_sync(fragment, source, leading_dimension);
}

// source is a row-major V tile [K, N].
__device__ __forceinline__ void load_pv_b_fragment(PVBFragment& fragment,
                                                   const __half* source,
                                                   int leading_dimension) {
  nvcuda::wmma::load_matrix_sync(fragment, source, leading_dimension);
}

__device__ __forceinline__ void init_accumulator_fragment(
    AccumulatorFragment& accumulator, float value = 0.0f) {
  nvcuda::wmma::fill_fragment(accumulator, value);
}

__device__ __forceinline__ void load_accumulator_fragment(
    AccumulatorFragment& accumulator, const float* source,
    int leading_dimension = kVoltaMmaN) {
  nvcuda::wmma::load_matrix_sync(accumulator, source, leading_dimension,
                                 nvcuda::wmma::mem_row_major);
}

// Register-resident QK update. The caller owns accumulator lifetime and may
// invoke this repeatedly without any C-memory round trip.
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(
    AccumulatorFragment& accumulator, const AFragment& a_fragment,
    const QKBFragment& b_fragment) {
  nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
}

// Register-resident PV update.
__device__ __forceinline__ void mma_sync_m16n16k16_row_row_f16f16f32(
    AccumulatorFragment& accumulator, const AFragment& a_fragment,
    const PVBFragment& b_fragment) {
  nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
}

__device__ __forceinline__ void store_accumulator_fragment(
    float* destination, const AccumulatorFragment& accumulator,
    int leading_dimension = kVoltaMmaN) {
  nvcuda::wmma::store_matrix_sync(destination, accumulator,
                                  leading_dimension,
                                  nvcuda::wmma::mem_row_major);
}

// Functional pointer control only. Each call loads and stores C, so callers
// must use the fragment API above for a multi-K attention hot loop.
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(
    float* c, const __half* a, const __half* b, int c_ld = kVoltaMmaN,
    int a_ld = kVoltaMmaK, int b_ld = kVoltaMmaK) {
  AFragment a_fragment;
  QKBFragment b_fragment;
  AccumulatorFragment accumulator;
  load_a_fragment(a_fragment, a, a_ld);
  load_qk_b_fragment(b_fragment, b, b_ld);
  if constexpr (mma_mode == MMAMode::kInit) {
    init_accumulator_fragment(accumulator);
  } else {
    load_accumulator_fragment(accumulator, c, c_ld);
  }
  mma_sync_m16n16k16_row_col_f16f16f32(accumulator, a_fragment, b_fragment);
  store_accumulator_fragment(c, accumulator, c_ld);
}

// Functional pointer control only; not suitable for repeated K16 updates.
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_row_f16f16f32(
    float* c, const __half* a, const __half* b, int c_ld = kVoltaMmaN,
    int a_ld = kVoltaMmaK, int b_ld = kVoltaMmaN) {
  AFragment a_fragment;
  PVBFragment b_fragment;
  AccumulatorFragment accumulator;
  load_a_fragment(a_fragment, a, a_ld);
  load_pv_b_fragment(b_fragment, b, b_ld);
  if constexpr (mma_mode == MMAMode::kInit) {
    init_accumulator_fragment(accumulator);
  } else {
    load_accumulator_fragment(accumulator, c, c_ld);
  }
  mma_sync_m16n16k16_row_row_f16f16f32(accumulator, a_fragment, b_fragment);
  store_accumulator_fragment(c, accumulator, c_ld);
}

}  // namespace sm70
}  // namespace attention

namespace mma {

// Keep the original pointer-control names available in flashinfer::mma.
using SM70MMAMode = attention::sm70::MMAMode;

template <SM70MMAMode mma_mode = SM70MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(
    float* c, const __half* a, const __half* b,
    int c_ld = attention::sm70::kVoltaMmaN,
    int a_ld = attention::sm70::kVoltaMmaK,
    int b_ld = attention::sm70::kVoltaMmaK) {
  attention::sm70::mma_sync_m16n16k16_row_col_f16f16f32<mma_mode>(
      c, a, b, c_ld, a_ld, b_ld);
}

template <SM70MMAMode mma_mode = SM70MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_row_f16f16f32(
    float* c, const __half* a, const __half* b,
    int c_ld = attention::sm70::kVoltaMmaN,
    int a_ld = attention::sm70::kVoltaMmaK,
    int b_ld = attention::sm70::kVoltaMmaN) {
  attention::sm70::mma_sync_m16n16k16_row_row_f16f16f32<mma_mode>(
      c, a, b, c_ld, a_ld, b_ld);
}

}  // namespace mma
}  // namespace flashinfer
