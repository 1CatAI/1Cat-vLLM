// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cstring>
#include <vector>
#include <cublasLt.h>

namespace {
void check_lattice_lt(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation,
              " failed with cuBLASLt status ", static_cast<int>(status));
}

struct LatticeLtDescriptors {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  ~LatticeLtDescriptors() {
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (operation) cublasLtMatmulDescDestroy(operation);
  }
};

void create_lattice_lt(LatticeLtDescriptors& descriptors, int m, int n, int k) {
  check_lattice_lt(cublasLtMatmulDescCreate(&descriptors.operation,
                                            CUBLAS_COMPUTE_32F, CUDA_R_32F),
                   "create FP32 matmul");
  // Column-major views match the existing [K,N] dequantization workspace
  // and row-major activation/output storage without an additional transpose.
  check_lattice_lt(
      cublasLtMatrixLayoutCreate(&descriptors.a, CUDA_R_16F, n, k, n),
      "create weight layout");
  check_lattice_lt(
      cublasLtMatrixLayoutCreate(&descriptors.b, CUDA_R_16F, k, m, k),
      "create input layout");
  check_lattice_lt(
      cublasLtMatrixLayoutCreate(&descriptors.c, CUDA_R_16F, n, m, n),
      "create output layout");
}

bool lattice_lt_fp32_reduction(const cublasLtMatmulAlgo_t& algorithm) {
  uint32_t reduction = 0;
  int32_t splits = 1;
  size_t written = 0;
  check_lattice_lt(cublasLtMatmulAlgoConfigGetAttribute(
                       &algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                       &reduction, sizeof(reduction), &written),
                   "read reduction scheme");
  check_lattice_lt(cublasLtMatmulAlgoConfigGetAttribute(
                       &algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splits,
                       sizeof(splits), &written),
                   "read split-K count");
  return reduction == CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE ||
         (reduction == CUBLASLT_REDUCTION_SCHEME_NONE && splits <= 1);
}

void validate_lattice_lt_matrices(torch::Tensor out, torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && out.device() == input.device() &&
                  input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16 && input.dim() == 2 &&
                  out.dim() == 2 && input.is_contiguous() &&
                  out.is_contiguous() && input.size(0) >= 512 &&
                  out.size(0) == input.size(0) && input.size(0) <= INT_MAX &&
                  out.size(1) > 0 && out.size(1) <= INT_MAX &&
                  out.size(1) % 32 == 0 && input.size(1) > 0 &&
                  input.size(1) <= INT_MAX && input.size(1) % 256 == 0,
              "GGUF cuBLASLt requires aligned FP16 prefill matrices");
}

}  // namespace

std::vector<torch::Tensor> gguf_lattice_compact_lt_sm70_prepare(
    torch::Tensor input, torch::Tensor out, int64_t max_candidates) {
  validate_lattice_lt_matrices(out, input);
  TORCH_CHECK(max_candidates > 0 && max_candidates <= 64,
              "GGUF cuBLASLt candidate count must be 1 to 64");
  const c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
              "Prepare GGUF cuBLASLt plans before graph capture");
  LatticeLtDescriptors descriptors;
  create_lattice_lt(descriptors, input.size(0), out.size(1), input.size(1));
  check_lattice_lt(cublasLtMatmulPreferenceCreate(&descriptors.preference),
                   "create matmul preference");
  const uint64_t workspace_bytes = 32 * 1024 * 1024;
  const uint32_t reductions = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  check_lattice_lt(
      cublasLtMatmulPreferenceSetAttribute(
          descriptors.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_bytes, sizeof(workspace_bytes)),
      "set workspace budget");
  check_lattice_lt(
      cublasLtMatmulPreferenceSetAttribute(
          descriptors.preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
          &reductions, sizeof(reductions)),
      "require FP32 split-K reduction");
  std::vector<cublasLtMatmulHeuristicResult_t> candidates(max_candidates);
  int count = 0;
  check_lattice_lt(
      cublasLtMatmulAlgoGetHeuristic(
          at::cuda::getCurrentCUDABlasLtHandle(), descriptors.operation,
          descriptors.a, descriptors.b, descriptors.c, descriptors.c,
          descriptors.preference, max_candidates, candidates.data(), &count),
      "query FP32 matmul candidates");
  std::vector<cublasLtMatmulHeuristicResult_t> accepted;
  for (int i = 0; i < count; ++i)
    if (candidates[i].state == CUBLAS_STATUS_SUCCESS &&
        candidates[i].workspaceSize <= workspace_bytes &&
        lattice_lt_fp32_reduction(candidates[i].algo))
      accepted.push_back(candidates[i]);
  auto algorithms = torch::empty(
      {static_cast<int64_t>(accepted.size()),
       static_cast<int64_t>(sizeof(cublasLtMatmulAlgo_t))},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  auto sizes = torch::empty(
      {static_cast<int64_t>(accepted.size())},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  for (size_t i = 0; i < accepted.size(); ++i) {
    std::memcpy(
        algorithms.data_ptr<uint8_t>() + i * sizeof(cublasLtMatmulAlgo_t),
        &accepted[i].algo, sizeof(cublasLtMatmulAlgo_t));
    sizes.data_ptr<int64_t>()[i] = accepted[i].workspaceSize;
  }
  return {algorithms, sizes};
}

void gguf_lattice_lt_matmul_sm70_out(torch::Tensor out, torch::Tensor input,
                                     torch::Tensor scratch,
                                     torch::Tensor workspace,
                                     torch::Tensor algorithm) {
  validate_lattice_lt_matrices(out, input);
  const c10::cuda::CUDAGuard guard(input.device());
  const int m = input.size(0), n = out.size(1), k = input.size(1);
  TORCH_CHECK(scratch.device() == input.device() &&
                  scratch.scalar_type() == torch::kFloat16 &&
                  scratch.dim() == 2 && scratch.size(0) == k &&
                  scratch.size(1) == n && scratch.is_contiguous() &&
                  workspace.device() == input.device() &&
                  workspace.scalar_type() == torch::kUInt8 &&
                  workspace.is_contiguous() && algorithm.device().is_cpu() &&
                  algorithm.scalar_type() == torch::kUInt8 &&
                  algorithm.is_contiguous() &&
                  algorithm.numel() == sizeof(cublasLtMatmulAlgo_t),
              "GGUF cuBLASLt requires prepared CPU plans and CUDA workspaces");
  cublasLtMatmulAlgo_t plan;
  std::memcpy(&plan, algorithm.data_ptr<uint8_t>(), sizeof(plan));
  TORCH_CHECK(
      lattice_lt_fp32_reduction(plan),
      "GGUF cuBLASLt rejects output-type or in-place split-K reduction");
  LatticeLtDescriptors descriptors;
  create_lattice_lt(descriptors, m, n, k);
  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  cublasLtMatmulHeuristicResult_t checked{};
  check_lattice_lt(
      cublasLtMatmulAlgoCheck(handle, descriptors.operation, descriptors.a,
                              descriptors.b, descriptors.c, descriptors.c,
                              &plan, &checked),
      "check prepared FP32 algorithm");
  TORCH_CHECK(checked.state == CUBLAS_STATUS_SUCCESS &&
                  checked.workspaceSize <= workspace.numel(),
              "GGUF cuBLASLt plan workspace is unavailable");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const float alpha = 1.f, beta = 0.f;
  check_lattice_lt(
      cublasLtMatmul(handle, descriptors.operation, &alpha, scratch.data_ptr(),
                     descriptors.a, input.data_ptr(), descriptors.b, &beta,
                     out.data_ptr(), descriptors.c, out.data_ptr(),
                     descriptors.c, &plan, workspace.data_ptr(),
                     workspace.numel(), stream),
      "run FP32 cuBLASLt matmul");
}
