// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/all.h>

#include <climits>

namespace {
// BF16 weight bits are expanded directly to FP32, including values below the
// FP16 subnormal range. Activations and the final output retain their FP16
// boundary. Each warp owns one output column and reuses weights for all rows.
template <int M>
__global__ void bf16_weight_linear_kernel(half* output, const half* input,
                                          const uint16_t* weight, int n,
                                          int k) {
  const int column = blockIdx.x * 4 + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (column >= n) return;
  float sums[M] = {};
  for (int index = lane; index < k; index += 32) {
    const float w = __uint_as_float(
        uint32_t(weight[static_cast<size_t>(column) * k + index]) << 16);
#pragma unroll
    for (int row = 0; row < M; ++row) {
      const float x = __half2float(input[static_cast<size_t>(row) * k + index]);
      sums[row] = __fmaf_rn(x, w, sums[row]);
    }
  }
#pragma unroll
  for (int row = 0; row < M; ++row) {
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
      sums[row] =
          __fadd_rn(sums[row], __shfl_down_sync(0xffffffff, sums[row], delta));
    }
    if (lane == 0)
      output[static_cast<size_t>(row) * n + column] =
          __float2half_rn(sums[row]);
  }
}

torch::Tensor bf16_weight_linear(torch::Tensor input, torch::Tensor weight) {
  TORCH_CHECK(
      input.is_cuda() && weight.is_cuda() && input.device() == weight.device(),
      "BF16 weight linear requires tensors on the same CUDA device");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf && weight.scalar_type() == at::kBFloat16,
      "BF16 weight linear requires FP16 input and original BF16 weights");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && input.is_contiguous() &&
                  weight.is_contiguous() && input.size(1) == weight.size(1),
              "BF16 weight linear requires compatible contiguous matrices");
  const int64_t m = input.size(0), n = weight.size(0), k = weight.size(1);
  TORCH_CHECK(
      m >= 1 && m <= 5 && n > 0 && k > 0 && n <= INT_MAX - 3 &&
          k <= INT_MAX - 31,
      "BF16 weight linear supports M1..5 and nonempty int32 dimensions");
  const c10::cuda::CUDAGuard guard(input.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  auto output = torch::empty({m, n}, input.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(M)                                                \
  bf16_weight_linear_kernel<M><<<(n + 3) / 4, 128, 0, stream>>>( \
      reinterpret_cast<half*>(output.data_ptr()),                \
      reinterpret_cast<const half*>(input.data_ptr()),           \
      reinterpret_cast<const uint16_t*>(weight.data_ptr()), n, k)
  switch (m) {
    case 1:
      LAUNCH(1);
      break;
    case 2:
      LAUNCH(2);
      break;
    case 3:
      LAUNCH(3);
      break;
    case 4:
      LAUNCH(4);
      break;
    case 5:
      LAUNCH(5);
      break;
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def("sm70_bf16_weight_linear(Tensor input, Tensor weight) -> Tensor");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sm70_bf16_weight_linear", &bf16_weight_linear);
}
