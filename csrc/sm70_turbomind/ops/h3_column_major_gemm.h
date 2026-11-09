// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cublasLt.h>

// Host descriptors only: no persistent weights and no device workspace.
// The validated Volta algorithm uses FP16 operands and FP32 accumulation,
// without split-K or intermediate reduced-precision reductions.
class H3ColumnMajorGemmPlan {
 public:
  H3ColumnMajorGemmPlan(int device, int64_t m, int64_t n, int64_t k,
                        bool output_fp32)
      : device_(device), m_(m), n_(n), k_(k), output_fp32_(output_fp32) {
    TORCH_CHECK(
        m > 0 && n > 0 && k > 0 && m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
        "H3 column-major GEMM dimension overflow");
    const c10::cuda::CUDAGuard guard(device_);
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 7 && properties->minor == 0,
                "H3 column-major GEMM requires SM70");
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(
        cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
                "Warm up the H3 GEMM shape before CUDA graph capture");
    try {
      check(cublasLtMatmulDescCreate(&desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      // W is physically column-major [N,K], X.T column-major [K,M].
      check(cublasLtMatrixLayoutCreate(&weight_, CUDA_R_16F, n, k, n));
      check(cublasLtMatrixLayoutCreate(&input_, CUDA_R_16F, k, m, k));
      check(cublasLtMatrixLayoutCreate(
          &output_, output_fp32 ? CUDA_R_32F : CUDA_R_16F, n, m, n));
      check(cublasLtMatmulPreferenceCreate(&preference_));
      size_t workspace_bytes = 0;
      uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
      // Match run()'s 16-byte operand contract. The library otherwise assumes
      // 256-byte alignment while selecting a heuristic for offset views.
      uint32_t alignment = 16;
      check(cublasLtMatmulPreferenceSetAttribute(
          preference_, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &alignment,
          sizeof(alignment)));
      check(cublasLtMatmulPreferenceSetAttribute(
          preference_, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &alignment,
          sizeof(alignment)));
      check(cublasLtMatmulPreferenceSetAttribute(
          preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_bytes, sizeof(workspace_bytes)));
      check(cublasLtMatmulPreferenceSetAttribute(
          preference_, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
          sizeof(reduction)));
      cublasLtMatmulHeuristicResult_t choices[32];
      int count = 0;
      auto status = cublasLtMatmulAlgoGetHeuristic(
          at::cuda::getCurrentCUDABlasLtHandle(), desc_, weight_, input_,
          output_, output_, preference_, 32, choices, &count);
      if (status == CUBLAS_STATUS_NOT_SUPPORTED) return;
      check(status);
      for (int i = 0; i < count; ++i) {
        const auto& choice = choices[i];
        if (choice.state != CUBLAS_STATUS_SUCCESS || choice.workspaceSize)
          continue;
        int id = 0, split = 0;
        uint32_t tile = 0;
        size_t written = 0;
        check(cublasLtMatmulAlgoConfigGetAttribute(
            &choice.algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &written));
        check(cublasLtMatmulAlgoConfigGetAttribute(
            &choice.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile),
            &written));
        check(cublasLtMatmulAlgoConfigGetAttribute(
            &choice.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split,
            sizeof(split), &written));
        check(cublasLtMatmulAlgoConfigGetAttribute(
            &choice.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction,
            sizeof(reduction), &written));
        if (id == 21 && tile == 24 && split == 1 && reduction == 0) {
          algorithm_ = choice.algo;
          supported_ = true;
          break;
        }
      }
    } catch (...) {
      release();
      throw;
    }
  }

  ~H3ColumnMajorGemmPlan() { release(); }
  H3ColumnMajorGemmPlan(const H3ColumnMajorGemmPlan&) = delete;
  H3ColumnMajorGemmPlan& operator=(const H3ColumnMajorGemmPlan&) = delete;
  bool supported() const { return supported_; }

  torch::Tensor run(torch::Tensor input, torch::Tensor weight) const {
    TORCH_CHECK(supported_,
                "No validated zero-workspace H3 cuBLASLt algorithm");
    TORCH_CHECK(
        input.is_cuda() && weight.device() == input.device() &&
            input.get_device() == device_ && input.dim() == 2 &&
            weight.dim() == 2 && input.is_contiguous() &&
            input.scalar_type() == torch::kFloat16 &&
            weight.scalar_type() == torch::kFloat16 && input.size(0) == m_ &&
            input.size(1) == k_ && weight.size(0) == n_ &&
            weight.size(1) == k_ && weight.stride(0) == 1 &&
            weight.stride(1) == n_ &&
            reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
        "H3 column-major GEMM requires matching aligned FP16 CUDA matrices");
    const c10::cuda::CUDAGuard guard(device_);
    auto out = torch::empty(
        {m_, n_}, input.options().dtype(output_fp32_ ? torch::kFloat32
                                                     : torch::kFloat16));
    float alpha = 1, beta = 0;
    check(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), desc_, &alpha,
                         weight.data_ptr(), weight_, input.data_ptr(), input_,
                         &beta, out.data_ptr(), output_, out.data_ptr(),
                         output_, &algorithm_, nullptr, 0,
                         at::cuda::getCurrentCUDAStream()));
    return out;
  }

 private:
  static void check(cublasStatus_t status) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "H3 cuBLASLt failed: ", int(status));
  }
  void release() noexcept {
    if (preference_) cublasLtMatmulPreferenceDestroy(preference_);
    if (output_) cublasLtMatrixLayoutDestroy(output_);
    if (input_) cublasLtMatrixLayoutDestroy(input_);
    if (weight_) cublasLtMatrixLayoutDestroy(weight_);
    if (desc_) cublasLtMatmulDescDestroy(desc_);
  }
  int device_;
  int64_t m_, n_, k_;
  bool output_fp32_, supported_ = false;
  cublasLtMatmulDesc_t desc_{};
  cublasLtMatrixLayout_t weight_{}, input_{}, output_{};
  cublasLtMatmulPreference_t preference_{};
  cublasLtMatmulAlgo_t algorithm_{};
};
