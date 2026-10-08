// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research-only whole-MLP residency screen. Reuses the shipped DMV readers.
#include <cooperative_groups.h>
#include <torch/extension.h>

namespace {
template <int Down>
__global__ __launch_bounds__(512) void persistent_mlp(Segs gate, Segs down,
                                                      const half* x, half* h,
                                                      int k, int intermediate,
                                                      int gate_tiles,
                                                      int down_tiles) {
  for (int t = blockIdx.x; t < gate_tiles; t += gridDim.x) {
    dense_tile<4, 4, IQ3S, IQ3S>(gate, x, k, 8, k, k / 32, k / 128, 1, nullptr,
                                 nullptr, nullptr, t, 0);
    __syncthreads();
  }
  // Cooperative residency is checked at launch; no oversubscribed spin barrier.
  cooperative_groups::this_grid().sync();
  for (int t = blockIdx.x; t < down_tiles; t += gridDim.x) {
    dense_tile<4, 2, Down, Down>(down, h, intermediate, 8, intermediate,
                                 intermediate / 32, intermediate / 128, 1,
                                 nullptr, nullptr, nullptr, t, 0);
    __syncthreads();
  }
}

template <int Down>
void run_chain(Segs gate, Segs down, const half* x, half* h, int k,
               int intermediate, int gate_tiles, int down_tiles,
               bool persistent) {
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr size_t shared = 4 * 256 * 16 + TAB_VECS * 16;
  if (!persistent) {
    launch<4, 4, IQ3S, IQ3S>(gate, x, k, 8, k, k / 32, k / 128, 1, nullptr,
                             nullptr, gate_tiles, stream, nullptr);
    launch<4, 2, Down, Down>(down, h, intermediate, 8, intermediate,
                             intermediate / 32, intermediate / 128, 1, nullptr,
                             nullptr, down_tiles, stream, nullptr);
    return;
  }
  const auto kernel = persistent_mlp<Down>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  int active = 0, device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel,
                                                               512, shared));
  TORCH_CHECK(properties.cooperativeLaunch && properties.major == 7 &&
                  properties.minor == 0 && active >= 1,
              "persistent MLP needs SM70 cooperative residency");
  const int blocks = properties.multiProcessorCount;
  void* arguments[] = {&gate, &down,         &x,          &h,
                       &k,    &intermediate, &gate_tiles, &down_tiles};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(kernel), dim3(blocks), dim3(512), arguments,
      shared, stream));
}
}  // namespace

void mlp_research(torch::Tensor x, std::vector<torch::Tensor> codes,
                  std::vector<torch::Tensor> scale, torch::Tensor table,
                  torch::Tensor h, torch::Tensor out, int64_t down_format,
                  bool persistent) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                  x.is_contiguous() && x.sizes() == at::IntArrayRef({8, 5120}),
              "research workload is M8/K5120");
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(codes.size() == 3 && scale.size() == 3 &&
                  h.sizes() == at::IntArrayRef({8, 4352}) &&
                  out.sizes() == at::IntArrayRef({8, 5120}),
              "invalid research MLP geometry");
  auto check = [&](const torch::Tensor& tensor, at::ScalarType dtype) {
    TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous() &&
                    tensor.scalar_type() == dtype,
                "invalid research storage");
  };
  check(h, at::kHalf);
  check(out, at::kHalf);
  check(table, at::kByte);
  TORCH_CHECK(table.numel() == TAB_VECS * 16, "invalid codebook");
  Segs gate{}, down{};
  gate.nseg = 2;
  gate.pair = 1;
  gate.hout = reinterpret_cast<half*>(h.data_ptr());
  gate.hld = 4352;
  gate.tab = reinterpret_cast<const uint4*>(table.data_ptr());
  down.nseg = 1;
  down.tab = gate.tab;
  for (int i = 0; i < 3; ++i) {
    check(codes[i], at::kByte);
    check(scale[i], at::kByte);
    const int n = i == 2 ? 5120 : 4352;
    const int k = i == 2 ? 4352 : 5120;
    const int fmt = i == 2 ? down_format : IQ3S;
    TORCH_CHECK(fmt == IQ3S || fmt == LUT4 || fmt == Q4K,
                "unqualified research reader");
    TORCH_CHECK(
        codes[i].numel() ==
                (n / 32) * (k / 128) * 512 * (fmt == IQ3S ? 3 : 4) &&
            scale[i].numel() == (n / 32) * (k / 128) * (fmt == Q4K ? 512 : 256),
        "invalid plane sizes");
    Seg& segment = i == 2 ? down.s[0] : gate.s[i];
    segment.codes = reinterpret_cast<const uint4*>(codes[i].data_ptr());
    segment.scale = reinterpret_cast<const uint4*>(scale[i].data_ptr());
    segment.out = reinterpret_cast<half*>((i == 2 ? out : h).data_ptr());
    segment.out_ld = n;
    segment.n = n;
    segment.fmt = fmt;
  }
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
  half* hp = reinterpret_cast<half*>(h.data_ptr());
  if (down_format == IQ3S)
    run_chain<IQ3S>(gate, down, xp, hp, 5120, 4352, 68, 80, persistent);
  else if (down_format == LUT4)
    run_chain<LUT4>(gate, down, xp, hp, 5120, 4352, 68, 80, persistent);
  else
    run_chain<Q4K>(gate, down, xp, hp, 5120, 4352, 68, 80, persistent);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &mlp_research); }
