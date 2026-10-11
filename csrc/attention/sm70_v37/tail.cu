#undef FLASH_NAMESPACE
#define FLASH_NAMESPACE onecat_v37_tail
/******************************************************************************
 * Copyright (c) 2026, 1CatAI.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/nn/functional.h>

#include <limits>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "../../../flash-attention-v100/kernel/kv_codec_traits.cuh"

namespace FLASH_NAMESPACE {

using namespace cute;

#include "../sm70_splitd/splitd_core.cuh"

template <bool RowScales>
cudaError_t dense_state_float_raw_impl(const void* q, const void* k,
                                       const void* v, float* state_max,
                                       float* state_sum, void* out,
                                       int query_len, int kv_len, int heads_q,
                                       int heads_kv, float softmax_scale,
                                       int unnormalized, cudaStream_t stream,
                                       flash_v100::KVScaleView scales) {
  if (q == nullptr || k == nullptr || v == nullptr || state_max == nullptr ||
      state_sum == nullptr || out == nullptr || query_len <= 0 ||
      kv_len < query_len || heads_q <= 0 || heads_kv <= 0 ||
      heads_q % heads_kv != 0 ||
      query_len % Sm70D256SplitDTraits::kBlockM != 0 ||
      kv_len % Sm70D256SplitDTraits::kBlockN != 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(Sm70D256SplitDTraits::kNThreads);
  const dim3 grid(query_len / Sm70D256SplitDTraits::kBlockM, 1, heads_q);
  auto kernel =
      unnormalized
          ? sm70_d256_splitd_dense_kernel<cutlass::half_t, false, false, true,
                                          true, true, false, RowScales>
          : sm70_d256_splitd_dense_kernel<cutlass::half_t, false, false, true,
                                          true, false, false, RowScales>;
  constexpr int kSmem =
      Sm70D256SplitDTraits::kSmemBytes +
      (RowScales ? Sm70D256SplitDTraits::kPElements * sizeof(cutlass::half_t)
                 : 0);
  cudaError_t result = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (result != cudaSuccess) {
    return result;
  }
  kernel<<<grid, block, kSmem, stream>>>(
      static_cast<const cutlass::half_t*>(q),
      static_cast<const cutlass::half_t*>(k),
      static_cast<const cutlass::half_t*>(v), static_cast<float*>(out),
      query_len * heads_q * Sm70D256SplitDTraits::kHeadDim,
      heads_q * Sm70D256SplitDTraits::kHeadDim, Sm70D256SplitDTraits::kHeadDim,
      kv_len * heads_kv * Sm70D256SplitDTraits::kHeadDim,
      heads_kv * Sm70D256SplitDTraits::kHeadDim, Sm70D256SplitDTraits::kHeadDim,
      kv_len * heads_kv * Sm70D256SplitDTraits::kHeadDim,
      heads_kv * Sm70D256SplitDTraits::kHeadDim, Sm70D256SplitDTraits::kHeadDim,
      query_len, kv_len, heads_q, heads_kv, softmax_scale * float(M_LOG2E),
      nullptr, 0, 0, nullptr, state_max, state_sum, nullptr, scales);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t onecat_v37_dense_state_float_raw(
    const void* q, const void* k, const void* v, float* state_max,
    float* state_sum, void* out, int query_len, int kv_len, int heads_q,
    int heads_kv, float softmax_scale, int unnormalized, cudaStream_t stream) {
  return dense_state_float_raw_impl<false>(
      q, k, v, state_max, state_sum, out, query_len, kv_len, heads_q, heads_kv,
      softmax_scale, unnormalized, stream, {});
}

extern "C" cudaError_t onecat_v37_dense_state_float_raw_with_scales(
    const void* q, const void* k, const void* v, float* state_max,
    float* state_sum, void* out, int query_len, int kv_len, int heads_q,
    int heads_kv, float softmax_scale, int unnormalized, cudaStream_t stream,
    flash_v100::KVScaleView scales) {
  if (!scales.key.data || !scales.value.data) return cudaErrorInvalidValue;
  return dense_state_float_raw_impl<true>(
      q, k, v, state_max, state_sum, out, query_len, kv_len, heads_q, heads_kv,
      softmax_scale, unnormalized, stream, scales);
}

// The compact-score operator calls this after joining its prefix and tail.
// Flags cover 64 query tokens (all six heads); only flagged tiles overwrite
// normalized FP32 residuals. Logits, statistics and PV accumulation stay FP32;
// the caller supplies centered/scaled V and restores its output afterwards.
extern "C" cudaError_t onecat_v37_recover_score_tiles_raw(
    const void* q, const void* k, const void* v, void* out,
    const int* recovery_tiles, int query_len, int kv_len, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      recovery_tiles == nullptr || query_len <= 0 || query_len % 64 != 0 ||
      kv_len < query_len || kv_len % 32 != 0) {
    return cudaErrorInvalidValue;
  }
  using Traits = Sm70D256SplitDTraits;
  auto kernel = sm70_d256_splitd_dense_kernel<cutlass::half_t, false, false,
                                              false, true, false, true>;
  cudaError_t result = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemBytes);
  if (result != cudaSuccess) return result;
  kernel<<<dim3(query_len / 64, 1, 6), Traits::kNThreads, Traits::kSmemBytes,
           stream>>>(static_cast<const cutlass::half_t*>(q),
                     static_cast<const cutlass::half_t*>(k),
                     static_cast<const cutlass::half_t*>(v),
                     static_cast<float*>(out), query_len * 6 * 256, 6 * 256,
                     256, kv_len * 256, 256, 256, kv_len * 256, 256, 256,
                     query_len, kv_len, 6, 1, 0.0625f * float(M_LOG2E), nullptr,
                     0, 0, nullptr, nullptr, nullptr, recovery_tiles, {});
  return cudaPeekAtLastError();
}
}  // namespace FLASH_NAMESPACE
