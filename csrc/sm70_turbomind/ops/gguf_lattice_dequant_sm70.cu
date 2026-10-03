// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// The shared codebooks retain llama.cpp/gguf-py's MIT license.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <type_traits>
#include "gguf_lattice_canonical.cuh"

namespace {
template <int Type>
constexpr int lattice_group = Type == 17 || Type == 22 || Type == 29 ? 16 : 32;

template <int Type>
__global__ void lattice_dequant_kernel(half* output, const void* weight,
                                       const void* stats, int k, int n) {
  using Decoder = vllm::sm70_gguf::LatticeCanonicalDecoder<Type>;
  using Transform = typename Decoder::Transform;
  __shared__ __align__(16) uint8_t grid[Transform::kCodebookBytes];
  Transform::initialize(grid);
  const int64_t fragment =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (fragment >= static_cast<int64_t>(k) * n / 8) return;
  const int lane = fragment % 32;
  const int64_t tile = fragment / 32;
  const int base = (tile % (k / 8)) * 8;
  const int col = (tile / (k / 8)) * 32 + lane;
  const auto decoded = Decoder::fragment(weight, stats, k, n, col, base, grid);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    output[static_cast<int64_t>(base + i) * n + col] = decoded[i];
}

int expected_group(int type) {
  TORCH_CHECK(type == 16 || type == 17 || type == 18 || type == 19 ||
                  type == 21 || type == 22 || type == 29,
              "GGUF lattice source type is unsupported");
  return type == 17 || type == 22 || type == 29 ? 16 : 32;
}
auto expected_stats(int type) {
  return type == 18 || type == 21   ? torch::kInt64
         : type == 19 || type == 29 ? torch::kInt16
                                    : torch::kInt32;
}
void validate_lattice_dequant(const torch::Tensor& out, const torch::Tensor& w,
                              const torch::Tensor& s, int type, int group) {
  TORCH_CHECK(group == expected_group(type),
              "GGUF lattice canonical group mismatch");
  TORCH_CHECK(
      out.is_cuda() && out.device() == w.device() && out.device() == s.device(),
      "GGUF lattice dequant tensors must share a CUDA device");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  w.scalar_type() == torch::kInt32 &&
                  s.scalar_type() == expected_stats(type),
              "GGUF lattice dequant storage dtype mismatch");
  TORCH_CHECK(out.dim() == 2 && w.dim() == 2 && s.dim() == 2 &&
                  out.is_contiguous() && w.is_contiguous() && s.is_contiguous(),
              "GGUF lattice dequant requires contiguous matrices");
  const int64_t k = out.size(0), n = out.size(1);
  TORCH_CHECK(k > 0 && n > 0 && k <= INT_MAX && n <= INT_MAX &&
                  k % group == 0 && n % 32 == 0 && w.size(0) == k &&
                  w.size(1) == n / 16 && s.size(0) == k / group &&
                  s.size(1) == n,
              "GGUF lattice dequant descriptor shape mismatch");
  TORCH_CHECK(
      out.data_ptr() != w.data_ptr() && out.data_ptr() != s.data_ptr(),
      "GGUF lattice dequant output must be separate from packed storage");
}
}  // namespace

void gguf_lattice_dequantize_sm70_out(torch::Tensor out, torch::Tensor weight,
                                      torch::Tensor stats, int64_t source_type,
                                      int64_t group_size) {
  validate_lattice_dequant(out, weight, stats, source_type, group_size);
  const c10::cuda::CUDAGuard guard(out.device());
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 0,
              "GGUF lattice dequant requires SM70");
  const int k = out.size(0), n = out.size(1);
  const int blocks = (static_cast<int64_t>(k) * n / 8 + 255) / 256;
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_LATTICE(TYPE)                                        \
  case TYPE:                                                        \
    lattice_dequant_kernel<TYPE><<<blocks, 256, 0, stream>>>(       \
        reinterpret_cast<half*>(out.data_ptr()), weight.data_ptr(), \
        stats.data_ptr(), k, n);                                    \
    break
  switch (source_type) {
    LAUNCH_LATTICE(16);
    LAUNCH_LATTICE(17);
    LAUNCH_LATTICE(18);
    LAUNCH_LATTICE(19);
    LAUNCH_LATTICE(21);
    LAUNCH_LATTICE(22);
    LAUNCH_LATTICE(29);
  }
#undef LAUNCH_LATTICE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_lattice_blas_sm70_out(torch::Tensor out, torch::Tensor input,
                                torch::Tensor weight, torch::Tensor stats,
                                int64_t source_type, torch::Tensor scratch,
                                int64_t group_size) {
  TORCH_CHECK(
      input.is_cuda() && input.device() == out.device() &&
          input.device() == scratch.device() &&
          input.scalar_type() == torch::kFloat16 &&
          out.scalar_type() == torch::kFloat16 && input.dim() == 2 &&
          out.dim() == 2 && input.is_contiguous() && out.is_contiguous(),
      "GGUF lattice BLAS requires contiguous FP16 matrices on one device");
  const int64_t m = input.size(0), k = input.size(1), n = out.size(1);
  TORCH_CHECK(m <= INT_MAX && out.size(0) == m && scratch.dim() == 2 &&
                  scratch.size(0) == k && scratch.size(1) == n,
              "GGUF lattice BLAS scratch or output shape mismatch");
  TORCH_CHECK(scratch.data_ptr() != out.data_ptr() &&
                  scratch.data_ptr() != input.data_ptr(),
              "GGUF lattice BLAS scratch must be separate from activations");
  if (m == 0) return;
  const c10::cuda::CUDAGuard guard(input.device());
  gguf_lattice_dequantize_sm70_out(scratch, weight, stats, source_type,
                                   group_size);
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  cublasMath_t saved_math;
  TORCH_CUDABLAS_CHECK(cublasGetMathMode(handle, &saved_math));
  TORCH_CUDABLAS_CHECK(cublasSetMathMode(
      handle, static_cast<cublasMath_t>(
                  CUBLAS_TENSOR_OP_MATH |
                  CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
  const float alpha = 1.f, beta = 0.f;
  // Row-major X[M,K] * W[K,N] is column-major W[N,K] * X[K,M].
  const auto status = cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, scratch.data_ptr(),
      CUDA_R_16F, n, input.data_ptr(), CUDA_R_16F, k, &beta, out.data_ptr(),
      CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  const auto restored = cublasSetMathMode(handle, saved_math);
  TORCH_CUDABLAS_CHECK(status);
  TORCH_CUDABLAS_CHECK(restored);
}
