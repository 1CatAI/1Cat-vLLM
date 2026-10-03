// SPDX-License-Identifier: Apache-2.0
#include "ggml_dypes.cuh"
#include "torch_context.cuh"
#include "convert.cuh"
#include "ternary_dequant.cuh"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <cublas_v2.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/c/shim.h>

using torch::headeronly::ScalarType;

namespace {
template <typename scalar_t>
void run_upstream_dequantize(const Tensor& W, Tensor& output, int64_t type,
                             int64_t total, cudaStream_t stream) {
  if (type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) {
    const int64_t blocks = total / 256;
    if (type == GGML_TYPE_TQ1_0) {
      dequantize_ternary<scalar_t, GGML_TYPE_TQ1_0><<<blocks, 256, 0, stream>>>(
          static_cast<const uint8_t*>(W.data_ptr()),
          static_cast<scalar_t*>(output.data_ptr()), blocks);
    } else {
      dequantize_ternary<scalar_t, GGML_TYPE_TQ2_0><<<blocks, 256, 0, stream>>>(
          static_cast<const uint8_t*>(W.data_ptr()),
          static_cast<scalar_t*>(output.data_ptr()), blocks);
    }
    check_launch("ternary GGUF dequantization");
    return;
  }
  auto to_cuda = [&]() {
    if constexpr (std::is_same_v<scalar_t, float>) {
      return ggml_get_to_fp32_cuda(static_cast<ggml_type>(type));
    } else if constexpr (std::is_same_v<scalar_t, half>) {
      return ggml_get_to_fp16_cuda(static_cast<ggml_type>(type));
    } else {
      return ggml_get_to_bf16_cuda(static_cast<ggml_type>(type));
    }
  }();
  STD_TORCH_CHECK(to_cuda != nullptr,
                  "ggml_dequantize_upstream: no upstream dequantize kernel "
                  "for quantization type ",
                  type);
  to_cuda(W.data_ptr(), static_cast<scalar_t*>(output.data_ptr()), total,
          stream);
}

ScalarType blas_compute_dtype(int cc) {
  // Backend policy is selected by KernelConfig on the Python side. Do not
  // introduce an independent GGML_CUDA_CUBLAS_COMPUTE_TYPE environment switch.
  return fast_fp16_hardware_available(cc) ? ScalarType::Half
                                          : ScalarType::Float;
}

}  // namespace

Tensor run_upstream_blas(const Tensor& W, const Tensor& X, int64_t type,
                         int64_t row, int64_t k) {
  const int32_t device_index = X.get_device_index();
  const DeviceGuard device_guard(device_index);
  const cudaStream_t stream = current_stream(device_index);
  const int cc = ggml_cuda_info().devices[device_index].cc;
  const ScalarType compute_dtype = X.scalar_type() == ScalarType::Float
                                       ? ScalarType::Float
                                       : blas_compute_dtype(cc);
  const int64_t batch = X.size(0);
  STD_TORCH_CHECK(type != GGML_TYPE_MXFP4 || k % 256 == 0,
                  "upstream MXFP4 cuBLAS requires K aligned to 256 values");
  STD_TORCH_CHECK(row <= INT_MAX && k <= INT_MAX && batch <= INT_MAX,
                  "upstream cuBLAS dimensions exceed int32 limits");

  Tensor weights = torch::stable::new_empty(W, {row, k}, compute_dtype);
  if (compute_dtype == ScalarType::Float) {
    run_upstream_dequantize<float>(W, weights, type, row * k, stream);
  } else if (compute_dtype == ScalarType::Half) {
    run_upstream_dequantize<half>(W, weights, type, row * k, stream);
  } else {
    run_upstream_dequantize<nv_bfloat16>(W, weights, type, row * k, stream);
  }

  Tensor converted;
  const void* activation = X.data_ptr();
  if (X.scalar_type() != compute_dtype) {
    converted = torch::stable::new_empty(X, {batch, k}, compute_dtype);
    cast_contiguous_async(converted.data_ptr(), compute_dtype, X.data_ptr(),
                          X.scalar_type(), batch * k, stream);
    activation = converted.data_ptr();
  }
  const bool half_result =
      compute_dtype == ScalarType::Half && cc != GGML_CUDA_CC_VOLTA;
  const ScalarType result_dtype =
      half_result ? ScalarType::Half : ScalarType::Float;
  Tensor result = torch::stable::new_empty(X, {batch, row}, result_dtype);

  void* raw_handle = nullptr;
  TORCH_ERROR_CODE_CHECK(torch_get_current_cuda_blas_handle(&raw_handle));
  auto handle = static_cast<cublasHandle_t>(raw_handle);
  STD_TORCH_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
                  "upstream cuBLAS could not bind the Torch current stream");
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const half alpha_half = __float2half(1.0f);
  const half beta_half = __float2half(0.0f);
  const cudaDataType_t input_type =
      compute_dtype == ScalarType::Float  ? CUDA_R_32F
      : compute_dtype == ScalarType::Half ? CUDA_R_16F
                                          : CUDA_R_16BF;
  const cublasStatus_t status =
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(row),
                   static_cast<int>(batch), static_cast<int>(k),
                   half_result ? static_cast<const void*>(&alpha_half)
                               : static_cast<const void*>(&alpha),
                   weights.data_ptr(), input_type, static_cast<int>(k),
                   activation, input_type, static_cast<int>(k),
                   half_result ? static_cast<const void*>(&beta_half)
                               : static_cast<const void*>(&beta),
                   result.data_ptr(), half_result ? CUDA_R_16F : CUDA_R_32F,
                   static_cast<int>(row),
                   half_result ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  STD_TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                  "upstream cuBLAS GEMM failed with status ",
                  static_cast<int>(status));

  if (X.scalar_type() == result_dtype) {
    return result;
  }
  Tensor output = torch::stable::new_empty(X, {batch, row}, X.scalar_type());
  cast_contiguous_async(output.data_ptr(), X.scalar_type(), result.data_ptr(),
                        result_dtype, batch * row, stream);
  check_launch("upstream cuBLAS output");
  return output;
}

Tensor ggml_dequantize_upstream(Tensor W, int64_t type, int64_t m, int64_t n,
                                std::optional<ScalarType> dtype) {
  STD_TORCH_CHECK(W.is_cuda(),
                  "ggml_dequantize_upstream: W must be a CUDA tensor");
  STD_TORCH_CHECK(W.dim() == 2 && W.is_contiguous(),
                  "ggml_dequantize_upstream: W must be a contiguous rank-2 "
                  "tensor");
  STD_TORCH_CHECK(W.element_size() == 1,
                  "ggml_dequantize_upstream: W must contain packed byte data");
  STD_TORCH_CHECK(m >= 0 && n >= 0,
                  "ggml_dequantize_upstream: output dimensions must be "
                  "non-negative");
  STD_TORCH_CHECK(m == 0 || n <= std::numeric_limits<int64_t>::max() / m,
                  "ggml_dequantize_upstream: output dimensions overflow");
  STD_TORCH_CHECK(is_upstream_weight_type(type),
                  "ggml_dequantize_upstream: unsupported quantization type ",
                  type);

  const int64_t total = m * n;
  const auto quant_type = static_cast<ggml_type>(type);
  const int64_t block_size = ggml_blck_size(quant_type);
  const size_t type_size = ggml_type_size(quant_type);
  STD_TORCH_CHECK(n == 0 || n % block_size == 0,
                  "ggml_dequantize_upstream: n must be aligned to the "
                  "quantization block size");
  STD_TORCH_CHECK(
      W.size(1) == (n / block_size) * static_cast<int64_t>(type_size),
      "ggml_dequantize_upstream: packed row size does not match n and "
      "quantization type");
  if (quant_type == GGML_TYPE_MXFP4) {
    // convert.cu's MXFP4 row kernel consumes QK_K (256) values per launch.
    STD_TORCH_CHECK(
        n == 0 || n % 256 == 0,
        "ggml_dequantize_upstream: MXFP4 n must be aligned to 256 values");
  }
  STD_TORCH_CHECK(total == 0 || total % block_size == 0,
                  "ggml_dequantize_upstream: output element count must be "
                  "aligned to the quantization block size");
  // The convert kernels read rows[0..m) from the packed weight, so reject a
  // row count the input cannot back before launching anything. Callers may
  // legally decode a prefix (m < W.size(0)); they may not ask for more rows
  // than the input provides.
  STD_TORCH_CHECK(m <= W.size(0),
                  "ggml_dequantize_upstream: requested row count ", m,
                  " exceeds the packed input capacity ", W.size(0));

  // Validate the dtype before allocating so an unsupported request fails
  // deterministically regardless of the output shape (an empty output used
  // to return before dtype dispatch ever ran).
  const auto dtype_ = dtype.value_or(ScalarType::Half);
  STD_TORCH_CHECK(dtype_ == ScalarType::Float || dtype_ == ScalarType::Half ||
                      dtype_ == ScalarType::BFloat16,
                  "ggml_dequantize_upstream: output dtype must be fp32, fp16, "
                  "or bf16");

  Tensor output = torch::stable::new_empty(W, {m, n}, dtype_);
  if (total == 0) {
    return output;
  }

  const int32_t device_idx = W.get_device_index();
  const DeviceGuard device_guard(device_idx);
  cudaStream_t stream = current_stream(device_idx);
  if (dtype_ == ScalarType::Float) {
    run_upstream_dequantize<float>(W, output, type, total, stream);
  } else if (dtype_ == ScalarType::Half) {
    run_upstream_dequantize<half>(W, output, type, total, stream);
  } else {
    run_upstream_dequantize<nv_bfloat16>(W, output, type, total, stream);
  }
  check_launch("upstream dequantize");
  return output;
}
