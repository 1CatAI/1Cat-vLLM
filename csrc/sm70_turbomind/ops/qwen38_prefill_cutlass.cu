// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "qwen38_prefill_cutlass.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>

#include <atomic>
#include <cstring>
#include <cstdlib>
#include <iostream>

namespace vllm::awq_sm70 {
namespace {

// Matches CUTLASS's generated
// cutlass_tensorop_f16_s884gemm_f16_128x256_32x2_nn_align8 operation. The
// profiler reports its logical instruction tile as 16x16x4, while the SM70
// DefaultGemmUniversal specialization is built from native m8n8k4 MMA ops.
using Sm70Fp8PrefillCutlassKernel =
    typename cutlass::gemm::kernel::DefaultGemmUniversal<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone, 8, cutlass::half_t,
        cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
        cutlass::half_t, cutlass::layout::RowMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
        cutlass::gemm::GemmShape<128, 256, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<8, 8, 4>,
        cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float,
                                                     float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 2,
        cutlass::arch::OpMultiplyAdd>::GemmKernel;
using Sm70Fp8PrefillCutlassGemm =
    cutlass::gemm::device::GemmUniversalAdapter<Sm70Fp8PrefillCutlassKernel>;

void maybe_log_sm70_fp8_prefill_cutlass_route(int64_t m, int64_t n, int64_t k) {
  const char* raw = std::getenv("VLLM_SM70_PROFILE_TRACE");
  if (raw == nullptr || std::strcmp(raw, "1") != 0) {
    return;
  }
  const unsigned bit = n == 8704   ? 1u
                       : k == 4352 ? 2u
                       : k == 1536 ? 4u
                       : n == 4096 ? 8u
                                   : 16u;
  static std::atomic<unsigned> logged_shapes{0};
  const unsigned previous =
      logged_shapes.fetch_or(bit, std::memory_order_relaxed);
  if ((previous & bit) == 0u) {
    std::cerr << "SM70 block-FP8 exact-8K CUTLASS projection route M=" << m
              << " N=" << n << " K=" << k << std::endl;
  }
}

}  // namespace

bool sm70_fp8_prefill_cutlass_out(torch::Tensor out, torch::Tensor in_feats,
                                  torch::Tensor dense_weight, bool gated_silu) {
  const char* raw = std::getenv("VLLM_SM70_FP8_PREFILL_CUTLASS");
  if ((raw != nullptr && std::atoi(raw) == 0) || gated_silu ||
      in_feats.dim() != 2 || dense_weight.dim() != 2) {
    return false;
  }
  const int64_t k = in_feats.size(1);
  const int64_t n = dense_weight.size(1);
  if (in_feats.size(0) < 8000 || in_feats.size(0) > 8192 || k <= 0 ||
      k % 128 != 0 || n <= 0 || n % 128 != 0) {
    return false;
  }

  TORCH_CHECK(out.is_cuda() && in_feats.is_cuda() && dense_weight.is_cuda(),
              "SM70 block-FP8 prefill CUTLASS GEMM expects CUDA tensors.");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Half &&
                  in_feats.scalar_type() == at::ScalarType::Half &&
                  dense_weight.scalar_type() == at::ScalarType::Half,
              "SM70 block-FP8 prefill CUTLASS GEMM expects float16 tensors.");
  TORCH_CHECK(
      out.is_contiguous() && in_feats.is_contiguous() &&
          dense_weight.is_contiguous(),
      "SM70 block-FP8 prefill CUTLASS GEMM expects contiguous tensors.");
  TORCH_CHECK(out.dim() == 2 && dense_weight.size(0) == k &&
                  out.size(0) == in_feats.size(0) && out.size(1) == n,
              "SM70 block-FP8 prefill CUTLASS GEMM shape mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int m = static_cast<int>(in_feats.size(0));
  const int problem_n = static_cast<int>(n);
  const int problem_k = static_cast<int>(k);
  auto* dense_ptr =
      reinterpret_cast<cutlass::half_t*>(dense_weight.data_ptr<at::Half>());
  auto* input_ptr =
      reinterpret_cast<cutlass::half_t*>(in_feats.data_ptr<at::Half>());
  auto* output_ptr =
      reinterpret_cast<cutlass::half_t*>(out.data_ptr<at::Half>());
  maybe_log_sm70_fp8_prefill_cutlass_route(m, problem_n, problem_k);
  typename Sm70Fp8PrefillCutlassGemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {problem_n, m, problem_k},
      1,
      {1.0f, 0.0f},
      dense_ptr,
      input_ptr,
      output_ptr,
      output_ptr,
      0,
      0,
      0,
      0,
      problem_n,
      problem_k,
      problem_n,
      problem_n};
  Sm70Fp8PrefillCutlassGemm gemm;
  const cutlass::Status status =
      gemm(arguments, nullptr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "SM70 block-FP8 prefill CUTLASS GEMM failed: ",
              cutlassGetStatusString(status), ".");
  return true;
}

}  // namespace vllm::awq_sm70
