// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "h3_column_major_gemm.h"
#include "diffusion_epilogue.h"

namespace {
__global__ void prepare_fp16_rows(const float* input, half* output,
                                  float* scales, int64_t rows, int width) {
  __shared__ float warp_maxima[8];
  __shared__ float row_scale;
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    float maximum = 0.f;
    for (int col = tid; col < width; col += blockDim.x) {
      float value = fabsf(input[row * width + col]);
      // Nonfinite inputs remain nonfinite after conversion. Like torch.frexp,
      // use scale 1 when a row's maximum is nonfinite.
      maximum = fmaxf(maximum, isnan(value) ? INFINITY : value);
    }
#pragma unroll
    for (int delta = 16; delta; delta /= 2)
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
    if (lane == 0) warp_maxima[warp] = maximum;
    __syncthreads();
    if (warp == 0) {
      maximum = lane < 8 ? warp_maxima[lane] : 0.f;
#pragma unroll
      for (int delta = 16; delta; delta /= 2)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
      if (lane == 0) {
        int exponent = 0;
        if (isfinite(maximum)) frexpf(maximum, &exponent);
        row_scale = ldexpf(1.f, max(exponent - 11, 0));
        scales[row] = row_scale;
      }
    }
    __syncthreads();
    for (int col = tid; col < width; col += blockDim.x)
      output[row * width + col] =
          __float2half_rn(input[row * width + col] / row_scale);
    __syncthreads();
  }
}

__global__ void dequantize_rows(const int8_t* weights, const float* scales,
                                half* output, int64_t count, int64_t width) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(gridDim.x) * blockDim.x) {
    output[i] = __float2half_rn(float(weights[i]) * scales[i / width]);
  }
}

// Physical [K,N] storage retains logical [N,K] and output-channel scales.
__global__ void dequantize_columns(const int8_t* weights, const float* scales,
                                   half* output, int64_t count, int64_t rows) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(gridDim.x) * blockDim.x) {
    output[i] = __float2half_rn(float(weights[i]) * scales[i % rows]);
  }
}

__device__ __forceinline__ float convrot_butterfly(float a, float b, float c,
                                                   float d, int digit) {
  return digit == 0   ? a + b + c - d
         : digit == 1 ? a + b - c + d
         : digit == 2 ? a - b + c + d
                      : -a + b + c + d;
}

__global__ void convrot256(const half* input, half* output, int64_t groups) {
  const int lane = threadIdx.x % 32;
  // Each warp owns a full rotation group, with eight channels per lane.
  // Keep the checkpoint's radix-four operation order and FP32 intermediates.
  for (int64_t group =
           int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
       group < groups; group += int64_t(gridDim.x) * (blockDim.x / 32)) {
    float values[8], next[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      values[i] = __half2float(input[group * 256 + lane + 32 * i]);
#pragma unroll
    for (int stride = 1; stride <= 4; stride *= 4) {
      const int digit = (lane / stride) % 4;
      const int base = lane - digit * stride;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float a = __shfl_sync(0xffffffff, values[i], base);
        float b = __shfl_sync(0xffffffff, values[i], base + stride);
        float c = __shfl_sync(0xffffffff, values[i], base + 2 * stride);
        float d = __shfl_sync(0xffffffff, values[i], base + 3 * stride);
        values[i] = convrot_butterfly(a, b, c, d, digit);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int even = i & ~1, base = lane % 16;
      const int digit = lane / 16 + 2 * (i % 2);
      float a = __shfl_sync(0xffffffff, values[even], base);
      float b = __shfl_sync(0xffffffff, values[even], base + 16);
      float c = __shfl_sync(0xffffffff, values[even + 1], base);
      float d = __shfl_sync(0xffffffff, values[even + 1], base + 16);
      next[i] = convrot_butterfly(a, b, c, d, digit);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int base = i % 2;
      float value = convrot_butterfly(next[base], next[base + 2],
                                      next[base + 4], next[base + 6], i / 2);
      output[group * 256 + lane + 32 * i] =
          __float2half_rn(value * (1.f / 16.f));
    }
  }
}

void validate_cuda(const torch::Tensor& value) {
  TORCH_CHECK(value.is_cuda() && value.is_contiguous(),
              "H3 requires contiguous CUDA tensors");
  const auto* p = at::cuda::getDeviceProperties(value.get_device());
  TORCH_CHECK(p->major == 7 && p->minor == 0,
              "H3 SM70 operator requires Volta");
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor> h3_prepare_fp16(torch::Tensor input) {
  validate_cuda(input);
  TORCH_CHECK(input.dim() == 2 && input.scalar_type() == torch::kFloat32 &&
                  input.size(1) > 0 && input.size(1) <= INT_MAX,
              "H3 FP16 preparation requires a FP32 matrix with nonempty rows");
  const c10::cuda::CUDAGuard guard(input.device());
  auto output =
      torch::empty(input.sizes(), input.options().dtype(torch::kFloat16));
  auto scales = torch::empty({input.size(0), 1}, input.options());
  if (input.size(0)) {
    prepare_fp16_rows<<<std::min<int64_t>(input.size(0), 65535), 256, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), reinterpret_cast<half*>(output.data_ptr()),
        scales.data_ptr<float>(), input.size(0), input.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, scales};
}

torch::Tensor h3_dequantize(torch::Tensor weight, torch::Tensor scale) {
  TORCH_CHECK(
      weight.is_cuda() && weight.dim() == 2 &&
          (weight.is_contiguous() ||
           (weight.stride(0) == 1 && weight.stride(1) == weight.size(0))),
      "H3 INT8 weight requires dense row-major or column-major CUDA storage");
  validate_cuda(scale);
  TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == torch::kInt8,
              "H3 weight must be a signed INT8 matrix");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32 &&
                  scale.numel() == weight.size(0) &&
                  scale.device() == weight.device(),
              "H3 requires same-device FP32 row scales");
  const c10::cuda::CUDAGuard guard(weight.device());
  const bool columns = !weight.is_contiguous();
  auto output = columns ? torch::empty({weight.size(1), weight.size(0)},
                                       weight.options().dtype(torch::kFloat16))
                              .t()
                        : torch::empty(weight.sizes(),
                                       weight.options().dtype(torch::kFloat16));
  if (weight.numel()) {
    const int grid = std::min<int64_t>((weight.numel() + 255) / 256, 65535);
    if (columns) {
      dequantize_columns<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          weight.data_ptr<int8_t>(), scale.data_ptr<float>(),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()), weight.numel(),
          weight.size(0));
    } else {
      dequantize_rows<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          weight.data_ptr<int8_t>(), scale.data_ptr<float>(),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()), weight.numel(),
          weight.size(1));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

torch::Tensor h3_rotate(torch::Tensor input) {
  validate_cuda(input);
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 && input.dim() >= 1 &&
                  input.size(-1) % 256 == 0,
              "H3 rotation requires FP16 groups of 256");
  const c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty_like(input);
  const int64_t groups = input.numel() / 256;
  if (groups) {
    convrot256<<<std::min<int64_t>((groups + 3) / 4, 65535), 128, 0,
                 at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()), groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

torch::Tensor h3_fp16_gemm(torch::Tensor input, torch::Tensor weight,
                           bool output_fp32) {
  validate_cuda(input);
  validate_cuda(weight);
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 &&
                  input.size(1) == weight.size(1) &&
                  input.device() == weight.device() &&
                  input.scalar_type() == torch::kFloat16 &&
                  weight.scalar_type() == torch::kFloat16,
              "H3 GEMM requires matching FP16 [M,K] and [N,K] matrices");
  const c10::cuda::CUDAGuard guard(input.device());
  const int64_t m = input.size(0), n = weight.size(0), k = input.size(1);
  TORCH_CHECK(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
              "GEMM dimension overflow");
  auto output = torch::empty(
      {m, n},
      input.options().dtype(output_fp32 ? torch::kFloat32 : torch::kFloat16));
  if (!m || !n) return output;
  if (!k) return output.zero_();
  // cuBLAS is an existing TurboMind SM70 dispatch option. Explicit 32F compute
  // prevents reduced-precision accumulation; W8 decode is outside the GEMM.
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  cublasMath_t saved_math;
  TORCH_CHECK(cublasGetMathMode(handle, &saved_math) == CUBLAS_STATUS_SUCCESS,
              "cannot read cuBLAS math mode");
  TORCH_CHECK(
      cublasSetMathMode(
          handle, static_cast<cublasMath_t>(
                      CUBLAS_TENSOR_OP_MATH |
                      CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)) ==
          CUBLAS_STATUS_SUCCESS,
      "cannot enable FP32 GEMM reductions");
  float alpha = 1.f, beta = 0.f;
  auto status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                             weight.data_ptr(), CUDA_R_16F, k, input.data_ptr(),
                             CUDA_R_16F, k, &beta, output.data_ptr(),
                             output_fp32 ? CUDA_R_32F : CUDA_R_16F, n,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  auto restored = cublasSetMathMode(handle, saved_math);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "H3 FP16 GEMM failed: ", int(status));
  TORCH_CHECK(restored == CUBLAS_STATUS_SUCCESS,
              "cannot restore cuBLAS math mode");
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<H3ColumnMajorGemmPlan>(m, "ColumnMajorGemmPlan")
      .def(pybind11::init<int, int64_t, int64_t, int64_t, bool>())
      .def_property_readonly("supported", &H3ColumnMajorGemmPlan::supported)
      .def("run", &H3ColumnMajorGemmPlan::run);
  m.def("prepare_fp16", &h3_prepare_fp16);
  m.def("dequantize", &h3_dequantize);
  m.def("rotate", &h3_rotate);
  m.def("scaled_add_", &sm70_diffusion::scaled_add, pybind11::arg("output"),
        pybind11::arg("delta"), pybind11::arg("scales"), pybind11::arg("alpha"),
        pybind11::arg("offset") = 0);
  m.def("gemm", &h3_fp16_gemm, pybind11::arg("input"), pybind11::arg("weight"),
        pybind11::arg("output_fp32") = false);
}
