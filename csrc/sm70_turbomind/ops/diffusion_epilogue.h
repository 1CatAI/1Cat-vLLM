// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>

namespace sm70_diffusion {
template <typename Output>
__global__ void scaled_add_rows(Output* output, const float* delta,
                                const float* scales, int64_t count,
                                int64_t width, int64_t output_width,
                                int64_t offset, float alpha) {
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = index / width;
    const int64_t col = index - row * width;
    const int64_t destination = row * output_width + offset + col;
    // Retain the explicit FP32 scale-restoration boundary before addition.
    const float restored =
        scales ? __fmul_rn(delta[index], scales[row]) : delta[index];
    float base;
    if constexpr (std::is_same_v<Output, half>) {
      base = __half2float(output[destination]);
    } else {
      base = output[destination];
    }
    const float result = __fmaf_rn(alpha, restored, base);
    if constexpr (std::is_same_v<Output, half>) {
      output[destination] = __float2half_rn(result);
    } else {
      output[destination] = result;
    }
  }
}

inline torch::Tensor scaled_add(torch::Tensor output, torch::Tensor delta,
                                std::optional<torch::Tensor> scales,
                                double alpha, int64_t offset) {
  TORCH_CHECK(output.is_cuda() && output.dim() == 2 && output.is_contiguous(),
              "SM70 scaled addition requires contiguous CUDA [M,N] output");
  TORCH_CHECK(output.scalar_type() == torch::kFloat16 ||
                  output.scalar_type() == torch::kFloat32,
              "SM70 scaled addition output must be FP16 or FP32");
  TORCH_CHECK(delta.device() == output.device() && delta.dim() == 2 &&
                  delta.is_contiguous() &&
                  delta.scalar_type() == torch::kFloat32 &&
                  delta.size(0) == output.size(0),
              "SM70 scaled addition requires matching FP32 [M,K] delta");
  TORCH_CHECK(offset >= 0 && offset <= output.size(1) &&
                  delta.size(1) <= output.size(1) - offset,
              "SM70 scaled addition slice is outside output");
  TORCH_CHECK(std::isfinite(alpha) &&
                  std::abs(alpha) <= std::numeric_limits<float>::max(),
              "SM70 scaled addition alpha must be finite FP32");
  TORCH_CHECK(!output.requires_grad() && !delta.requires_grad(),
              "SM70 scaled addition is inference-only");
  at::assert_no_overlap(output, delta);
  const c10::cuda::CUDAGuard guard(output.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "SM70 scaled addition requires SM70");
  const float* scale_data = nullptr;
  if (scales.has_value()) {
    TORCH_CHECK(
        scales->device() == output.device() && scales->is_contiguous() &&
            scales->scalar_type() == torch::kFloat32 &&
            scales->numel() == output.size(0) && !scales->requires_grad(),
        "SM70 scaled addition needs one FP32 scale per row");
    at::assert_no_overlap(output, *scales);
    scale_data = scales->data_ptr<float>();
  }
  if (!delta.numel()) return output;
  const int blocks = std::min<int64_t>((delta.numel() + 255) / 256, 65535);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (output.scalar_type() == torch::kFloat16) {
    scaled_add_rows<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        delta.data_ptr<float>(), scale_data, delta.numel(), delta.size(1),
        output.size(1), offset, float(alpha));
  } else {
    scaled_add_rows<<<blocks, 256, 0, stream>>>(
        output.data_ptr<float>(), delta.data_ptr<float>(), scale_data,
        delta.numel(), delta.size(1), output.size(1), offset, float(alpha));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
}  // namespace sm70_diffusion
