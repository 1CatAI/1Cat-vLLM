// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/extension.h>
namespace {
template <int FA, int FB>
void pair_screen(Segs segs, const half* x, int k, int variant) {
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr size_t shared = 4 * 256 * 16 + TAB_VECS * 16;
  if (variant == 0) {
    launch<4, 4, FA, FB>(segs, x, k, 8, k, k / 32, k / 128, 1, nullptr, nullptr,
                         68, stream, nullptr);
  } else if (variant == 2 || variant == 3) {
    if constexpr ((FA == IQ3S || FA == IQ3X) && (FB == IQ3S || FB == IQ3X)) {
      const auto kernel = variant == 3 ? compact_family_dense_mv<4, 4, FA, FB>
                                       : family_dense_mv<4, 4, FA, FB>;
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
      kernel<<<68, 512, shared, stream>>>(segs, x, k, 8, k, k / 32, k / 128, 1,
                                          nullptr, nullptr, nullptr);
    } else {
      TORCH_CHECK(false, "family variant requires IQ3 readers");
    }
  } else {
    const auto kernel = common_dense_mv<4, 4, FA, FB>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    kernel<<<68, 512, shared, stream>>>(segs, x, k, 8, k, k / 32, k / 128, 1,
                                        nullptr, nullptr, nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace
void run_pair(torch::Tensor x, std::vector<torch::Tensor> codes,
              std::vector<torch::Tensor> scales, torch::Tensor table,
              torch::Tensor out, int64_t fa, int64_t fb, int64_t variant) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                  x.is_contiguous() && x.sizes() == at::IntArrayRef({8, 5120}),
              "screen requires M8 K5120");
  TORCH_CHECK(codes.size() == 2 && scales.size() == 2 &&
                  out.sizes() == at::IntArrayRef({8, 4352}),
              "invalid pair");
  auto check = [&](const torch::Tensor& t, at::ScalarType type) {
    TORCH_CHECK(t.is_contiguous() && t.device() == x.device() &&
                    t.scalar_type() == type,
                "invalid screen storage");
  };
  check(table, at::kByte);
  check(out, at::kHalf);
  Segs segs{};
  segs.nseg = 2;
  segs.pair = 1;
  segs.tab = reinterpret_cast<const uint4*>(table.data_ptr());
  segs.hout = reinterpret_cast<half*>(out.data_ptr());
  segs.hld = 4352;
  for (int i = 0; i < 2; ++i) {
    check(codes[i], at::kByte);
    check(scales[i], at::kByte);
    Seg& s = segs.s[i];
    s.codes = reinterpret_cast<const uint4*>(codes[i].data_ptr());
    s.scale = reinterpret_cast<const uint4*>(scales[i].data_ptr());
    s.out = segs.hout;
    s.out_ld = 4352;
    s.n = 4352;
    s.fmt = i == 0 ? fa : fb;
    TORCH_CHECK(s.fmt == Q4K || s.fmt == LUT4 || s.fmt == IQ3S || s.fmt == IQ3X,
                "unqualified common-reader storage");
  }
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
#define COMB(A, B)                              \
  if (fa == A && fb == B) {                     \
    pair_screen<A, B>(segs, xp, 5120, variant); \
    return;                                     \
  }
  COMB(IQ3X, IQ3S);
  COMB(IQ3S, IQ3X);
  COMB(LUT4, IQ3S);
  COMB(IQ3S, LUT4);
  COMB(IQ3S, Q4K);
  COMB(Q4K, IQ3S);
#undef COMB
  TORCH_CHECK(false, "unsupported research pair");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run_pair); }
