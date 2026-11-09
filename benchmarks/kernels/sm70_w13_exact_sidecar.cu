// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/library.h>

#include "../../csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu"

namespace {

__global__ void merge_w13_partials(const float* partials, half* output) {
  const int tile = blockIdx.x;
  const int route = blockIdx.y;
  const int lane = threadIdx.x;
  float value = 0.0f;
#pragma unroll
  for (int warp = 0; warp < 16; ++warp) {
    value += partials[((route * 10 + tile) * 16 + warp) * 32 + lane];
  }
  const half rounded = __float2half(value);
  const int source_lane = (lane & 15) * 2;
  const unsigned bits = __half_as_ushort(rounded);
  const half gate = __ushort_as_half(
      static_cast<unsigned short>(__shfl_sync(0xffffffffu, bits, source_lane)));
  const half up = __ushort_as_half(static_cast<unsigned short>(
      __shfl_sync(0xffffffffu, bits, source_lane + 1)));
  if (lane < 16) {
    const float gate_f = __half2float(gate);
    const half silu = __float2half(gate_f / (1.0f + expf(-gate_f)));
    output[route * 160 + tile * 16 + lane] = __hmul(silu, up);
  }
}

void screen_w13(torch::Tensor out, torch::Tensor input, torch::Tensor weights,
                torch::Tensor scales, torch::Tensor ids, torch::Tensor scratch,
                int64_t mode) {
  TORCH_CHECK(mode >= 0 && mode <= 4);
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && weights.is_cuda() &&
              scales.is_cuda() && ids.is_cuda());
  TORCH_CHECK(
      out.scalar_type() == at::kHalf && input.scalar_type() == at::kHalf &&
      weights.scalar_type() == at::kInt && scales.scalar_type() == at::kHalf &&
      ids.scalar_type() == at::kInt);
  const int routes = static_cast<int>(ids.numel());
  TORCH_CHECK((routes == 10 || routes == 50) && input.dim() == 2 &&
              input.size(0) == routes / 10 && input.size(1) == 2560 &&
              out.sizes() == torch::IntArrayRef({routes, 160}) &&
              weights.sizes() == torch::IntArrayRef({512, 2560, 40}) &&
              scales.sizes() == torch::IntArrayRef({512, 160, 320}));
  for (const auto& t : {out, input, weights, scales, ids}) {
    TORCH_CHECK(t.is_contiguous() && t.device() == input.device());
  }
  const c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  const auto* w =
      reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>());
  const auto* s = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  if (mode >= 3) {
    TORCH_CHECK(scratch.device() == input.device() && scratch.is_contiguous() &&
                scratch.scalar_type() == at::kFloat &&
                scratch.numel() == routes * 10 * 16 * 32);
    if (mode == 3) {
      nvfp4_qpn_m1_sm70_kernel<16, true, true, true, 2>
          <<<dim3(10, routes, 2), 256, 0, stream>>>(
              x, w, s, ids.data_ptr<int32_t>(), y, 320, 2560, true,
              scratch.data_ptr<float>());
    } else {
      nvfp4_qpn_m1_sm70_kernel<16, true, true, true, 4>
          <<<dim3(10, routes, 4), 128, 0, stream>>>(
              x, w, s, ids.data_ptr<int32_t>(), y, 320, 2560, true,
              scratch.data_ptr<float>());
    }
    merge_w13_partials<<<dim3(10, routes), 32, 0, stream>>>(
        scratch.data_ptr<float>(), y);
  } else if (mode == 2) {
    nvfp4_qpn_m1_sm70_kernel<16, true, true, true>
        <<<dim3(10, routes), 512, 0, stream>>>(x, w, s, ids.data_ptr<int32_t>(),
                                               y, 320, 2560, true);
  } else if (mode == 1) {
    nvfp4_qpn_m1_sm70_kernel<16, true, true>
        <<<dim3(10, routes), 512, 0, stream>>>(x, w, s, ids.data_ptr<int32_t>(),
                                               y, 320, 2560, true);
  } else {
    nvfp4_qpn_m1_sm70_kernel<16, true, false>
        <<<dim3(10, routes), 512, 0, stream>>>(x, w, s, ids.data_ptr<int32_t>(),
                                               y, 320, 2560, true);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_w13_exact, ops) {
  ops.def(
      "run(Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
      "Tensor ids, Tensor(b!) scratch, int mode) -> ()");
  ops.impl("run", torch::kCUDA, &screen_w13);
}
