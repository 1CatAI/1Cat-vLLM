// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Fuse the eager FP16 clamp/SiLU/multiply sequence without changing its
// intermediate rounding boundaries or requantizing any expert weights.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cmath>

namespace {
__global__ void silu_mul_clamp_kernel(half* out, const half* gate,
                                      const half* up, int64_t size,
                                      float limit) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size) return;
  float g = __half2float(gate[index]);
  float u = __half2float(up[index]);
  // Comparison-based clamps preserve NaNs and signed zero as ATen does.
  g = g > limit ? limit : g;
  u = u < -limit ? -limit : u;
  u = u > limit ? limit : u;
  const float denominator = __fadd_rn(1.0f, expf(-g));
  // F.silu returns FP16 before the separate FP16 multiplication in the
  // reference. Retain that rounding; a single FP32 product changes MTP logits.
  const half activated = __float2half_rn(g / denominator);
  out[index] = __float2half_rn(__fmul_rn(__half2float(activated), u));
}

void silu_mul_clamp(torch::Tensor out, torch::Tensor gate, torch::Tensor up,
                    double limit) {
  TORCH_CHECK(
      std::isfinite(limit) && limit >= 0.0 && limit <= 65504.0,
      "GGUF clamped activation requires a finite FP16 nonnegative limit");
  TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == at::kHalf,
              "GGUF clamped activation requires CUDA FP16 inputs");
  for (const auto& tensor : {out, gate, up}) {
    TORCH_CHECK(tensor.device() == gate.device() &&
                    tensor.scalar_type() == at::kHalf &&
                    tensor.is_contiguous() && tensor.sizes() == gate.sizes(),
                "GGUF clamped activation requires matching contiguous tensors");
  }
  const c10::cuda::CUDAGuard guard(gate.device());
  const int64_t size = gate.numel();
  if (size == 0) return;
  // torch.clamp converts a scalar bound to the input dtype before comparison.
  const float rounded_limit = static_cast<float>(static_cast<at::Half>(limit));
  const auto stream = at::cuda::getCurrentCUDAStream();
  silu_mul_clamp_kernel<<<(size + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      reinterpret_cast<const half*>(gate.data_ptr()),
      reinterpret_cast<const half*>(up.data_ptr()), size, rounded_limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "gguf_silu_mul_clamp_out(Tensor(a!) out, Tensor gate, Tensor up, "
      "float limit) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("gguf_silu_mul_clamp_out", &silu_mul_clamp);
}
