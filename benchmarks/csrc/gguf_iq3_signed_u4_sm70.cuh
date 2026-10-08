// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/extension.h>
void signed_u4_screen(torch::Tensor x, std::vector<torch::Tensor> codes,
                      std::vector<torch::Tensor> scales, torch::Tensor table,
                      torch::Tensor out, bool expanded) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                  x.scalar_type() == at::kHalf &&
                  x.sizes() == at::IntArrayRef({8, 5120}),
              "invalid M8 activation");
  TORCH_CHECK(codes.size() == 2 && scales.size() == 2 && out.is_contiguous() &&
                  out.device() == x.device() &&
                  out.scalar_type() == at::kHalf &&
                  out.sizes() == at::IntArrayRef({8, 4352}),
              "invalid pair output");
  Segs segs{};
  segs.nseg = 2;
  segs.pair = 1;
  segs.hout = reinterpret_cast<half*>(out.data_ptr());
  segs.hld = 4352;
  segs.tab =
      expanded ? nullptr : reinterpret_cast<const uint4*>(table.data_ptr());
  for (int i = 0; i < 2; ++i) {
    for (const auto& tensor : {codes[i], scales[i]})
      TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous() &&
                      tensor.scalar_type() == at::kByte,
                  "invalid weight plane");
    TORCH_CHECK(codes[i].numel() == 136 * 40 * 512 * (expanded ? 4 : 3) &&
                    scales[i].numel() == 136 * 40 * 256,
                "invalid plane size");
    Seg& segment = segs.s[i];
    segment.codes = reinterpret_cast<const uint4*>(codes[i].data_ptr());
    segment.scale = reinterpret_cast<const uint4*>(scales[i].data_ptr());
    segment.out = segs.hout;
    segment.out_ld = 4352;
    segment.n = 4352;
    segment.fmt = expanded ? LUT4 : IQ3S;
  }
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr size_t shared = 4 * 256 * 16 + TAB_VECS * 16;
  if (expanded) {
    const auto kernel = signed_dense_mv<4, 4, LUT4, LUT4>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    kernel<<<68, 512, shared, stream>>>(segs, xp, 5120, 8, 5120, 160, 40, 1,
                                        nullptr, nullptr, nullptr);
  } else {
    launch<4, 4, IQ3S, IQ3S>(segs, xp, 5120, 8, 5120, 160, 40, 1, nullptr,
                             nullptr, 68, stream, nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void signed_u4_single(torch::Tensor x, torch::Tensor codes,
                      torch::Tensor scales, torch::Tensor table,
                      torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
                      bool expanded, bool gdn_heads, int64_t split) {
  c10::cuda::CUDAGuard guard(x.device());
  const int k = x.size(1), n = out.size(1), tiles = (n + 63) / 64;
  TORCH_CHECK(x.sizes()[0] == 8 && k % 128 == 0 && n % 32 == 0 &&
                  x.is_contiguous() && out.is_contiguous(),
              "invalid single geometry");
  Segs segs{};
  segs.nseg = 1;
  segs.gdn_heads = gdn_heads;
  segs.tab =
      expanded ? nullptr : reinterpret_cast<const uint4*>(table.data_ptr());
  segs.s[0].codes = reinterpret_cast<const uint4*>(codes.data_ptr());
  segs.s[0].scale = reinterpret_cast<const uint4*>(scales.data_ptr());
  segs.s[0].out = reinterpret_cast<half*>(out.data_ptr());
  segs.s[0].out_ld = n;
  segs.s[0].n = n;
  segs.s[0].fmt = expanded ? LUT4 : IQ3S;
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr size_t shared = 4 * 256 * 16 + TAB_VECS * 16;
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
  if (expanded) {
    const auto kernel = signed_dense_mv<4, 2, LUT4, LUT4>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    kernel<<<dim3(tiles, split), 256, shared, stream>>>(
        segs, xp, k, 8, k, k / 32, k / 128, split, ws.data_ptr<float>(),
        cnt.data_ptr<int>(), nullptr);
  } else {
    launch<4, 2, IQ3S, IQ3S>(segs, xp, k, 8, k, k / 32, k / 128, split,
                             ws.data_ptr<float>(), cnt.data_ptr<int>(), tiles,
                             stream, nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void compare_weight_packets(Seg native, Seg expanded,
                                       const uint4* tables, int* mismatch,
                                       int k) {
  extern __shared__ uint4 grid[];
  for (int i = threadIdx.x; i < TAB_VECS; i += 32) grid[i] = tables[i];
  __syncthreads();
  Ld<IQ3S> original;
  Ld<LUT4> candidate;
  load<IQ3S>(original, native, blockIdx.x, blockIdx.y, k / 32, k / 128,
             threadIdx.x);
  load<LUT4>(candidate, expanded, blockIdx.x, blockIdx.y, k / 32, k / 128,
             threadIdx.x);
  int differences = 0;
#pragma unroll
  for (int step = 0; step < 4; ++step) {
    uint32_t a[16], b[16];
    decode<IQ3S>(original, step, a, reinterpret_cast<const half2*>(grid));
    signed_decode(candidate, step, b);
#pragma unroll
    for (int i = 0; i < 16; ++i) differences += a[i] != b[i];
  }
  if (differences) atomicAdd(mismatch, differences);
}
void compare_weights(torch::Tensor original_code, torch::Tensor original_scale,
                     torch::Tensor expanded_code, torch::Tensor expanded_scale,
                     torch::Tensor table, torch::Tensor mismatch, int64_t n,
                     int64_t k) {
  c10::cuda::CUDAGuard guard(original_code.device());
  Seg original{}, expanded{};
  original.codes = reinterpret_cast<const uint4*>(original_code.data_ptr());
  original.scale = reinterpret_cast<const uint4*>(original_scale.data_ptr());
  expanded.codes = reinterpret_cast<const uint4*>(expanded_code.data_ptr());
  expanded.scale = reinterpret_cast<const uint4*>(expanded_scale.data_ptr());
  compare_weight_packets<<<dim3(n / 32, k / 128), 32, TAB_VECS * 16,
                           at::cuda::getCurrentCUDAStream()>>>(
      original, expanded, reinterpret_cast<const uint4*>(table.data_ptr()),
      mismatch.data_ptr<int>(), k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &signed_u4_screen);
  m.def("single", &signed_u4_single);
  m.def("compare_weights", &compare_weights);
}
