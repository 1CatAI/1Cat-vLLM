// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Packed layout and register decoders reuse LMDeploy TurboMind (Apache-2.0).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <type_traits>

#include "src/turbomind/kernels/gemm/transform.h"

namespace {

template <int Bits>
__global__ void gguf_affine_dequant_kernel(half* output, const void* weight,
                                           const void* stats, int k, int n,
                                           int group_size) {
  const int64_t fragment =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (fragment >= static_cast<int64_t>(k) * n / 8) return;
  // Native SM70 packing: N tiles of 32, then K8 fragments, then N lanes.
  const int lane = fragment % 32;
  const int64_t tile = fragment / 32;
  const int k_base = (tile % (k / 8)) * 8;
  const int col = (tile / (k / 8)) * 32 + lane;
  const int64_t stat_index =
      static_cast<int64_t>(k_base / group_size) * n + col;
  constexpr int LowBits = Bits == 3 ? 2 : ((Bits == 5 || Bits == 6) ? 4 : Bits);
  using D = std::conditional_t<
      LowBits == 2, turbomind::uint2_t,
      std::conditional_t<LowBits == 4, turbomind::uint4_t, uint8_t>>;
  using S = std::conditional_t<Bits == 5 || Bits == 6, uint64_t, uint32_t>;
  using Transform = std::conditional_t<
      Bits == 3, turbomind::gemm::Transform_HMMA_SM70_CenteredBitPlane3,
      std::conditional_t<Bits == 5 || Bits == 6,
                         turbomind::gemm::Transform_HMMA_SM70_BitPlane<
                             4, Bits - 4, Bits == 5 ? 32 : 16>,
                         turbomind::gemm::Transform_HMMA_SIMT_B>>;
  turbomind::Array<D, 8> data[1][1];
  data[0][0] =
      reinterpret_cast<const turbomind::Array<D, 8>*>(weight)[fragment];
  turbomind::Array<S, 1> coefficients[1][1];
  S metadata = static_cast<const S*>(stats)[stat_index];
  if constexpr (Bits == 3) {
    const uint32_t high = (metadata >> 16) >> (k_base % group_size);
    metadata = (metadata & 65535U) | (high << 16);
  } else if constexpr (Bits == 5 || Bits == 6) {
    const uint32_t high =
        (metadata >> 32) >> ((k_base % group_size) * (Bits - 4));
    metadata =
        static_cast<uint32_t>(metadata) | (static_cast<uint64_t>(high) << 32);
  }
  coefficients[0][0][0] = metadata;
  turbomind::Array<half, 8> decoded[1][1];
  Transform::apply(decoded, 0, data, coefficients, 1);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    output[static_cast<int64_t>(k_base + i) * n + col] = decoded[0][0][i];
  }
}

void validate_dequant(const torch::Tensor& out, const torch::Tensor& weight,
                      const torch::Tensor& stats, int bits, int group_size) {
  TORCH_CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
                  bits == 8,
              "GGUF affine dequant canonical width is unsupported");
  TORCH_CHECK(group_size == ((bits == 3 || bits == 6) ? 16 : 32) ||
                  (bits == 2 && group_size == 16),
              "GGUF affine dequant group is unsupported");
  TORCH_CHECK(out.is_cuda() && out.device() == weight.device() &&
                  out.device() == stats.device(),
              "GGUF affine dequant tensors must share a CUDA device");
  TORCH_CHECK(
      out.scalar_type() == torch::kFloat16 &&
          weight.scalar_type() == torch::kInt32 &&
          stats.scalar_type() ==
              ((bits == 5 || bits == 6) ? torch::kInt64 : torch::kInt32),
      "GGUF affine dequant storage dtype mismatch");
  TORCH_CHECK(out.dim() == 2 && weight.dim() == 2 && stats.dim() == 2 &&
                  out.is_contiguous() && weight.is_contiguous() &&
                  stats.is_contiguous(),
              "GGUF affine dequant requires contiguous matrices");
  const int64_t k = out.size(0), n = out.size(1);
  const int low_bits = bits == 3 ? 2 : ((bits == 5 || bits == 6) ? 4 : bits);
  TORCH_CHECK(k > 0 && n > 0 && k <= INT_MAX && n <= INT_MAX &&
                  k % group_size == 0 && n % 32 == 0 && weight.size(0) == k &&
                  weight.size(1) == n * low_bits / 32 &&
                  stats.size(0) == k / group_size && stats.size(1) == n,
              "GGUF affine dequant descriptor shape mismatch");
}

}  // namespace

void gguf_affine_dequantize_sm70_out(torch::Tensor out, torch::Tensor weight,
                                     torch::Tensor stats, int64_t bits,
                                     int64_t group_size) {
  validate_dequant(out, weight, stats, bits, group_size);
  const c10::cuda::CUDAGuard guard(out.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "GGUF affine dequant requires SM70");
  const int k = out.size(0), n = out.size(1);
  const int64_t fragments = static_cast<int64_t>(k) * n / 8;
  const int blocks = (fragments + 255) / 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_GGUF_DQ(B)                                                   \
  gguf_affine_dequant_kernel<B><<<blocks, 256, 0, stream>>>(                \
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), weight.data_ptr(), \
      stats.data_ptr(), k, n, group_size)
  switch (bits) {
    case 2:
      LAUNCH_GGUF_DQ(2);
      break;
    case 3:
      LAUNCH_GGUF_DQ(3);
      break;
    case 4:
      LAUNCH_GGUF_DQ(4);
      break;
    case 5:
      LAUNCH_GGUF_DQ(5);
      break;
    case 6:
      LAUNCH_GGUF_DQ(6);
      break;
    case 8:
      LAUNCH_GGUF_DQ(8);
      break;
  }
#undef LAUNCH_GGUF_DQ
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_affine_blas_sm70_out(torch::Tensor out, torch::Tensor input,
                               torch::Tensor weight, torch::Tensor stats,
                               int64_t bits, torch::Tensor scratch,
                               int64_t group_size) {
  TORCH_CHECK(
      input.is_cuda() && input.device() == out.device() &&
          input.device() == scratch.device() &&
          input.scalar_type() == torch::kFloat16 &&
          out.scalar_type() == torch::kFloat16 && input.dim() == 2 &&
          out.dim() == 2 && input.is_contiguous() && out.is_contiguous(),
      "GGUF affine BLAS requires contiguous FP16 matrices on one device");
  const int64_t m = input.size(0), k = input.size(1), n = out.size(1);
  TORCH_CHECK(m <= INT_MAX && out.size(0) == m && scratch.dim() == 2 &&
                  scratch.size(0) == k && scratch.size(1) == n,
              "GGUF affine BLAS scratch or output shape mismatch");
  TORCH_CHECK(scratch.data_ptr() != out.data_ptr() &&
                  scratch.data_ptr() != input.data_ptr(),
              "GGUF affine BLAS scratch must be separate from activations");
  if (m == 0) return;
  const c10::cuda::CUDAGuard guard(input.device());
  gguf_affine_dequantize_sm70_out(scratch, weight, stats, bits, group_size);
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
