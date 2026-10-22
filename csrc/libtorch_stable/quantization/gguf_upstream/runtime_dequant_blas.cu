// SPDX-License-Identifier: Apache-2.0
#include "ggml_dypes.cuh"
#include "torch_context.cuh"
#include "convert.cuh"
#include "dequantize.cuh"
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
#include <mma.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/c/shim.h>

using torch::headeronly::ScalarType;

namespace {
template <typename scalar_t>
__global__ void dequantize_q5_k_blocks(const void* source, scalar_t* output,
                                       int64_t blocks) {
  // Reuse upstream arithmetic verbatim. Four independent 64-thread
  // groups share a bounded persistent grid instead of scheduling one CTA
  // per GGUF block. The groups have no cross-warp dependencies.
  for (int64_t block = int64_t(blockIdx.x) * 4 + threadIdx.x / 64;
       block < blocks; block += int64_t(gridDim.x) * 4)
    dequantize_q5_K(source, block, output + block * QK_K, threadIdx.x % 64);
}

template <typename scalar_t>
void run_upstream_dequantize(const Tensor& W, Tensor& output, int64_t type,
                             int64_t total, cudaStream_t stream,
                             bool batched_blocks = false) {
  if constexpr (std::is_same_v<scalar_t, half>) {
    if (batched_blocks && type == GGML_TYPE_Q5_K &&
        ggml_cuda_info().devices[W.get_device_index()].cc ==
            GGML_CUDA_CC_VOLTA) {
      const int64_t blocks = total / QK_K;
      const int64_t grid = std::min(
          (blocks + 3) / 4,
          int64_t(ggml_cuda_info().devices[W.get_device_index()].nsm) * 8);
      dequantize_q5_k_blocks<<<grid, 256, 0, stream>>>(
          W.data_ptr(), static_cast<scalar_t*>(output.data_ptr()), blocks);
      check_launch("batched Q5_K dequantization");
      return;
    }
  }
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

__device__ __forceinline__ void decode_q5_quarter_vector(
    const void* weights, int64_t block, half* output, int quarter, int lane) {
  const auto& w = static_cast<const block_q5_K*>(weights)[block];
  const uint32_t low =
      reinterpret_cast<const uint32_t*>(w.qs + quarter * 32)[lane];
  const uint32_t high = reinterpret_cast<const uint32_t*>(w.qh)[lane];
  uint8_t scale, minimum;
  get_scale_min_k4(quarter * 2, w.scales, scale, minimum);
  const float d1 = __half2float(__low2half(w.dm)) * scale;
  const float m1 = __half2float(__high2half(w.dm)) * minimum;
  get_scale_min_k4(quarter * 2 + 1, w.scales, scale, minimum);
  const float d2 = __half2float(__low2half(w.dm)) * scale;
  const float m2 = __half2float(__high2half(w.dm)) * minimum;
  half lower[4], upper[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q = (low >> (8 * i)) & 255;
    const int h = (high >> (8 * i)) & 255;
    lower[i] = ggml_cuda_cast<half>(
        d1 * ((q & 15) + ((h >> (2 * quarter)) & 1) * 16) - m1);
    upper[i] = ggml_cuda_cast<half>(
        d2 * ((q >> 4) + ((h >> (2 * quarter + 1)) & 1) * 16) - m2);
  }
  auto* first = reinterpret_cast<half2*>(output + lane * 4);
  first[0] = __halves2half2(lower[0], lower[1]);
  first[1] = __halves2half2(lower[2], lower[3]);
  first[16] = __halves2half2(upper[0], upper[1]);
  first[17] = __halves2half2(upper[2], upper[3]);
}

template <bool Grouped, bool Vector = false, int Stride = (Vector ? 72 : 64)>
__global__ void q5_k_hmma_partials(const void* weights, const half* input,
                                   float* partials, int m, int n, int k,
                                   int splits, int groups) {
#if __CUDA_ARCH__ >= 700
  namespace wmma = nvcuda::wmma;
  // V4.1's vector path pads shared rows to spread WMMA column accesses
  // across banks. K64 tiles, fragments and accumulation order stay unchanged.
  __shared__ __align__(32) half activation[16 * Stride];
  // A prefix makes all compensated upstream destination pointers stay
  // inside this allocation while decoding just one 64-value quarter.
  __shared__ __align__(32) half decoded[256 + 64 * Stride];
  const int warp = threadIdx.x / 32;
  const int input_groups = Grouped ? groups : 1;
  const int input_group = Grouped ? blockIdx.x / (n / groups / 64) : 0;
  const int first = (k / 64) * blockIdx.y / splits;
  const int last = (k / 64) * (blockIdx.y + 1) / splits;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum;
  wmma::fill_fragment(accum, 0.f);
  for (int tile = first; tile < last; ++tile) {
    for (int i = threadIdx.x; i < 16 * 64; i += blockDim.x) {
      const int row = i / 64, col = i % 64;
      activation[row * Stride + col] =
          row < m ? input[(int64_t(row) * input_groups + input_group) * k +
                          tile * 64 + col]
                  : __float2half_rn(0.f);
    }
    const int quarter = tile % 4;
    if constexpr (Vector) {
      for (int row = threadIdx.x / 8; row < 64; row += blockDim.x / 8) {
        const int64_t block =
            int64_t(blockIdx.x * 64 + row) * (k / QK_K) + tile / 4;
        decode_q5_quarter_vector(weights, block, decoded + 256 + row * Stride,
                                 quarter, threadIdx.x % 8);
      }
    } else {
      for (int row = threadIdx.x / 16; row < 64; row += blockDim.x / 16) {
        const int64_t block =
            int64_t(blockIdx.x * 64 + row) * (k / QK_K) + tile / 4;
        dequantize_q5_K(weights, block,
                        decoded + 256 + row * Stride - quarter * 64,
                        quarter * 16 + threadIdx.x % 16);
      }
    }
    __syncthreads();
  #pragma unroll
    for (int offset = 0; offset < 64; offset += 16) {
      wmma::load_matrix_sync(a, activation + offset, Stride);
      wmma::load_matrix_sync(b, decoded + 256 + warp * 16 * Stride + offset,
                             Stride);
      wmma::mma_sync(accum, a, b, accum);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(
      partials + int64_t(blockIdx.y) * 16 * n + blockIdx.x * 64 + warp * 16,
      accum, n, wmma::mem_row_major);
#endif
}

__global__ void q5_k_hmma_reduce(const float* partials, half* output, int m,
                                 int n, int splits) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m * n) return;
  const int row = index / n, col = index % n;
  float value = 0.f;
  for (int split = 0; split < splits; ++split)
    value = __fadd_rn(value, partials[(int64_t(split) * 16 + row) * n + col]);
  output[index] = __float2half_rn(value);
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
    run_upstream_dequantize<half>(W, weights, type, row * k, stream, true);
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
                                std::optional<ScalarType> dtype,
                                bool batched_blocks) {
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
    run_upstream_dequantize<half>(W, output, type, total, stream,
                                  batched_blocks);
  } else {
    run_upstream_dequantize<nv_bfloat16>(W, output, type, total, stream);
  }
  check_launch("upstream dequantize");
  return output;
}

Tensor ggml_dense_q5_k_hmma(Tensor W, Tensor X, int64_t split_k,
                            int64_t workspace_bytes, bool grouped,
                            bool vector_decode) {
  const int expected_dim = grouped ? 3 : 2;
  STD_TORCH_CHECK(
      W.is_cuda() && X.is_cuda() &&
          W.get_device_index() == X.get_device_index() &&
          W.scalar_type() == ScalarType::Byte && W.dim() == 2 &&
          W.is_contiguous() && X.scalar_type() == ScalarType::Half &&
          X.dim() == expected_dim && X.is_contiguous() && X.size(0) > 0 &&
          X.size(0) <= 6 && W.size(0) > 0 && W.size(0) <= 65536 &&
          W.size(0) % 64 == 0 && X.size(-1) > 0 && X.size(-1) <= 16384 &&
          X.size(-1) % QK_K == 0 &&
          W.size(1) == X.size(-1) / QK_K * sizeof(block_q5_K),
      "Q5_K HMMA requires original packed rows and FP16 M1..6");
  const int groups = grouped ? X.size(1) : 1;
  STD_TORCH_CHECK(groups > 0 && groups <= 8 && W.size(0) % (groups * 64) == 0,
                  "Q5_K grouped HMMA requires whole 64-row output groups");
  const auto device = X.get_device_index();
  const DeviceGuard guard(device);
  const auto& info = ggml_cuda_info().devices[device];
  STD_TORCH_CHECK(info.cc == GGML_CUDA_CC_VOLTA, "Q5_K HMMA requires SM70");
  const int m = X.size(0), n = W.size(0), k = X.size(-1);
  STD_TORCH_CHECK(workspace_bytes > 0 && workspace_bytes <= 32 * 1024 * 1024,
                  "Q5_K HMMA workspace must be in (0, 32 MiB]");
  const int capacity = workspace_bytes / (16 * n * sizeof(float));
  const int automatic = (info.nsm * 8 + n / 64 - 1) / (n / 64);
  const int splits =
      split_k ? split_k : std::min({64, k / 64, automatic, capacity});
  STD_TORCH_CHECK(
      splits > 0 && splits <= 64 && splits <= k / 64 &&
          int64_t(splits) * 16 * n * sizeof(float) <= workspace_bytes,
      "Q5_K HMMA partial workspace exceeds its 32 MiB bound");
  Tensor partials =
      torch::stable::new_empty(X, {splits, 16, n}, ScalarType::Float);
  Tensor output = torch::stable::new_empty(X, {m, n}, ScalarType::Half);
  const auto stream = current_stream(device);
  // Keep the original full-N split count and summation order. Each output
  // group reads only its own input instead of computing every group and
  // discarding the other rows afterward.
  if (grouped && vector_decode)
    q5_k_hmma_partials<true, true><<<dim3(n / 64, splits), 128, 0, stream>>>(
        W.data_ptr(), static_cast<const half*>(X.data_ptr()),
        static_cast<float*>(partials.data_ptr()), m, n, k, splits, groups);
  else if (grouped)
    q5_k_hmma_partials<true><<<dim3(n / 64, splits), 128, 0, stream>>>(
        W.data_ptr(), static_cast<const half*>(X.data_ptr()),
        static_cast<float*>(partials.data_ptr()), m, n, k, splits, groups);
  else if (vector_decode)
    q5_k_hmma_partials<false, true><<<dim3(n / 64, splits), 128, 0, stream>>>(
        W.data_ptr(), static_cast<const half*>(X.data_ptr()),
        static_cast<float*>(partials.data_ptr()), m, n, k, splits, 1);
  else
    q5_k_hmma_partials<false><<<dim3(n / 64, splits), 128, 0, stream>>>(
        W.data_ptr(), static_cast<const half*>(X.data_ptr()),
        static_cast<float*>(partials.data_ptr()), m, n, k, splits, 1);
  q5_k_hmma_reduce<<<(m * n + 255) / 256, 256, 0, stream>>>(
      static_cast<const float*>(partials.data_ptr()),
      static_cast<half*>(output.data_ptr()), m, n, splits);
  check_launch("original Q5_K FP16 HMMA");
  return output;
}
