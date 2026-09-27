// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
template <bool Silu>
__global__ void rmsnorm_gated_exact_kernel(const half* x, const half* z,
                                           const half* weight, half* out,
                                           int rows, float eps) {
  constexpr int kWarps = 4;
  const int row = blockIdx.x * kWarps + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= rows) return;
  float values[4], gates[4], weights[4], sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = lane * 4 + i;
    values[i] = __half2float(x[row * 128 + col]);
    gates[i] = __half2float(z[row * 128 + col]);
    weights[i] = __half2float(weight[col]);
    // Match ATen's separate square and vector4 mean: four contiguous
    // products, a left-to-right local sum, then descending warp shuffles.
    // A conventional N128 tree or a contracted square/FMA changes bits.
    sum = __fadd_rn(sum, __fmul_rn(values[i], values[i]));
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, offset));
  }
  sum = __shfl_sync(0xffffffffU, sum, 0);
  const float mean = __fmul_rn(sum, 1.0f / 128);
  const float inverse = rsqrtf(__fadd_rn(mean, eps));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float normalized =
        __fmul_rn(__fmul_rn(values[i], inverse), weights[i]);
    // Keep the native FP32 exp/division and each pointwise boundary.
    // In particular, do not substitute Triton's approximate sigmoid.
    const float denominator = __fadd_rn(1.0f, expf(-gates[i]));
    const float activated = Silu ? gates[i] / denominator : 1.0f / denominator;
    out[row * 128 + lane * 4 + i] =
        __float2half_rn(__fmul_rn(normalized, activated));
  }
}

void rmsnorm_gated_exact(torch::Tensor out, torch::Tensor x, torch::Tensor z,
                         torch::Tensor weight, double eps, bool silu) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(1) == 128 &&
                  x.size(0) >= 1 && x.size(0) <= 192,
              "SM70 exact gated RMSNorm requires CUDA [1..192, 128]");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "Exact gated RMSNorm is SM70 only");
  for (const auto& tensor : {out, x, z, weight}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == x.device() &&
                    tensor.scalar_type() == at::kHalf && tensor.is_contiguous(),
                "Exact gated RMSNorm requires same-device contiguous FP16");
  }
  TORCH_CHECK(out.sizes() == x.sizes() && z.sizes() == x.sizes() &&
                  weight.sizes() == at::IntArrayRef({128}),
              "Invalid exact gated RMSNorm output/gate/weight geometry");
  const int rows = x.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(S)                                                    \
  rmsnorm_gated_exact_kernel<S><<<(rows + 3) / 4, 128, 0, stream>>>( \
      reinterpret_cast<const half*>(x.data_ptr()),                   \
      reinterpret_cast<const half*>(z.data_ptr()),                   \
      reinterpret_cast<const half*>(weight.data_ptr()),              \
      reinterpret_cast<half*>(out.data_ptr()), rows, static_cast<float>(eps))
  if (silu) {
    LAUNCH(true);
  } else {
    LAUNCH(false);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "sm70_rmsnorm_gated_exact_out(Tensor(a!) out, Tensor x, Tensor z, "
      "Tensor weight, float eps, bool silu) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sm70_rmsnorm_gated_exact_out", &rmsnorm_gated_exact);
}
