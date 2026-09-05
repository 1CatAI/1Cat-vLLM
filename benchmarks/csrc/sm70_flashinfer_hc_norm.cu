// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <torch/types.h>
#include <flashinfer/hc/sm70/hc_combine_norm.cuh>

namespace {
template <class T, class B, class W>
void launch(torch::Tensor residual, torch::Tensor block,
            torch::Tensor injection, torch::Tensor weight,
            torch::Tensor combined, torch::Tensor output, float eps,
            int warps) {
  const int groups = injection.size(1), d = block.size(1);
  const int rounds = (d + 8 * 32 * warps - 1) / (8 * 32 * warps);
  const int shared =
      ((warps + 3) / 4 * 4 + rounds * 8 * 32 * warps) * sizeof(float);
  flashinfer::sm70::hc::HCCombineNormKernel<8, T, B, W>
      <<<dim3(residual.size(0), groups), dim3(32, warps), shared,
         at::cuda::getCurrentCUDAStream()>>>(
          (const B*)block.data_ptr(), (const T*)residual.data_ptr(),
          (const W*)weight.data_ptr(), (const W*)injection.data_ptr(),
          (T*)combined.data_ptr(), (T*)output.data_ptr(), groups, d,
          block.stride(0), residual.stride(0), injection.stride(0),
          weight.numel() == d, eps);
}

template <class T, class B>
void weight_dispatch(torch::Tensor r, torch::Tensor b, torch::Tensor i,
                     torch::Tensor w, torch::Tensor c, torch::Tensor o,
                     float eps, int warps) {
  if (w.scalar_type() == at::kHalf)
    launch<T, B, half>(r, b, i, w, c, o, eps, warps);
  else
    launch<T, B, float>(r, b, i, w, c, o, eps, warps);
}

void run(torch::Tensor r, torch::Tensor b, torch::Tensor i, torch::Tensor w,
         torch::Tensor c, torch::Tensor o, double eps, int64_t warps) {
  const c10::cuda::CUDAGuard guard(r.device());
  for (const auto& t : {r, b, i, w, c, o}) {
    TORCH_CHECK(t.is_cuda() && t.device() == r.device() && t.is_contiguous());
    TORCH_CHECK(t.scalar_type() == at::kHalf || t.scalar_type() == at::kFloat);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0);
  }
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0);
  TORCH_CHECK(r.dim() == 2 && b.dim() == 2 && i.dim() == 2 && w.dim() == 1);
  TORCH_CHECK(r.size(0) == b.size(0) && r.size(0) == i.size(0));
  TORCH_CHECK(i.size(1) > 0 && b.size(1) > 0 && b.size(1) <= 4096 &&
              b.size(1) % 8 == 0);
  TORCH_CHECK(r.size(1) == i.size(1) * b.size(1));
  TORCH_CHECK(w.numel() == b.size(1) || w.numel() == r.size(1));
  TORCH_CHECK(i.scalar_type() == w.scalar_type());
  TORCH_CHECK(c.sizes() == r.sizes() && o.sizes() == r.sizes() &&
              c.scalar_type() == r.scalar_type() &&
              o.scalar_type() == r.scalar_type());
  TORCH_CHECK(warps == 1 || warps == 2 || warps == 4 || warps == 8);
  TORCH_CHECK(eps > 0 && r.size(0) > 0);
  if (r.scalar_type() == at::kHalf) {
    if (b.scalar_type() == at::kHalf)
      weight_dispatch<half, half>(r, b, i, w, c, o, eps, warps);
    else
      weight_dispatch<half, float>(r, b, i, w, c, o, eps, warps);
  } else {
    if (b.scalar_type() == at::kHalf)
      weight_dispatch<float, half>(r, b, i, w, c, o, eps, warps);
    else
      weight_dispatch<float, float>(r, b, i, w, c, o, eps, warps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(_C_flashinfer_hc_sm70, m) {
  m.def(
      "run(Tensor residual, Tensor block, Tensor injection, Tensor weight, "
      "Tensor(a!) combined, Tensor(b!) output, float eps, int warps) -> ()");
}
TORCH_LIBRARY_IMPL(_C_flashinfer_hc_sm70, CUDA, m) { m.impl("run", &run); }
