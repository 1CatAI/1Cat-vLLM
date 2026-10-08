// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/extension.h>

void publish(torch::Tensor ready, bool value) {
  c10::cuda::CUDAGuard guard(ready.device());
  TORCH_CHECK(ready.is_contiguous() && ready.scalar_type() == at::kInt &&
                  ready.numel() == 1,
              "invalid readiness flag");
  C10_CUDA_CHECK(cudaMemsetAsync(ready.data_ptr(), value ? 1 : 0, 4,
                                 at::cuda::getCurrentCUDAStream()));
}
void norm_ready_pair(torch::Tensor x, std::vector<torch::Tensor> codes,
                     std::vector<torch::Tensor> scales, torch::Tensor table,
                     torch::Tensor out, torch::Tensor ready, int64_t mode) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kHalf &&
                  x.sizes() == at::IntArrayRef({8, 5120}),
              "invalid input");
  TORCH_CHECK(codes.size() == 2 && scales.size() == 2 &&
                  out.sizes() == at::IntArrayRef({8, 4352}),
              "invalid pair");
  Segs segs{};
  segs.nseg = 2;
  segs.pair = 1;
  segs.hout = reinterpret_cast<half*>(out.data_ptr());
  segs.hld = 4352;
  segs.tab = reinterpret_cast<const uint4*>(table.data_ptr());
  for (int i = 0; i < 2; ++i) {
    Seg& segment = segs.s[i];
    segment.codes = reinterpret_cast<const uint4*>(codes[i].data_ptr());
    segment.scale = reinterpret_cast<const uint4*>(scales[i].data_ptr());
    segment.out = segs.hout;
    segment.out_ld = 4352;
    segment.n = 4352;
    segment.fmt = IQ3S;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
  if (mode == 2) {
    constexpr size_t cache_shared = (8 * 641 + TAB_VECS) * 16;
    const auto kernel = cached_dense_mv<4, 4, IQ3S, IQ3S>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, cache_shared));
    kernel<<<68, 512, cache_shared, stream>>>(segs, xp, 5120, 8, 5120, 160, 40,
                                              1, nullptr, nullptr, nullptr);
  } else if (mode == 1) {
    // At most 68 consumer CTAs on 80 SMs leave at least twelve SMs.
    // The shipped 40-CTA norm uses 56 registers/thread and 28 shared bytes,
    // permitting nine norm blocks per free SM: all producers can be resident.
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, x.get_device()));
    TORCH_CHECK(properties.major == 7 && properties.minor == 0 &&
                    properties.multiProcessorCount >= 80,
                "unsupported overlap residency");
    constexpr size_t shared = 4 * 256 * 16 + TAB_VECS * 16;
    ready_dense_mv<4, 4, IQ3S, IQ3S><<<68, 512, shared, stream>>>(
        segs, xp, 5120, 8, 5120, 160, 40, 1, nullptr, nullptr, nullptr,
        ready.data_ptr<int>());
  } else {
    launch<4, 4, IQ3S, IQ3S>(segs, xp, 5120, 8, 5120, 160, 40, 1, nullptr,
                             nullptr, 68, stream, nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &norm_ready_pair);
  m.def("publish", &publish);
}
