// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Paired execution is derived from nvfp4_qpn2_sm70.cu / v100-skinny.
// The retained MIT notice is csrc/sm70_turbomind/ops/LICENSE.v100-skinny.
struct SignedPairReader {
  const uint8_t* base;
  int lane;
  __device__ SignedPairReader(const uint8_t* codes, const void*, int tile,
                              int groups, int lane, float)
      : base(codes + static_cast<size_t>(tile) * groups * 320), lane(lane) {}
  __device__ __forceinline__ void load(int group, half2* weights) const {
    const uint8_t* packet = base + static_cast<size_t>(group) * 320;
    const uint2 code = __ldg(reinterpret_cast<const uint2*>(packet) + lane);
    uint32_t bits =
        __ldg(reinterpret_cast<const unsigned short*>(packet + 256) + lane);
    if ((bits & 32767u) == 0) bits = 0;
    const half2 scale = h2(bits | (bits << 16));
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const uint32_t q = c == 0 ? code.x : code.y;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t shifted = j == 0 ? q << 1 : q >> (4 * j - 1);
        const uint32_t value = lop_or(shifted, 0x001e001eu, 0x64006400u);
        weights[c * 4 + j] =
            __hmul2(__hsub2(h2(value), h2(0x640f640fu)), scale);
      }
    }
  }
};
// HOST_BINDINGS
#include <torch/extension.h>
__global__ void decode_bundle(const uint8_t* bundle, half* output, int n,
                              int k) {
  const int lane = threadIdx.x, group = blockIdx.y;
  const int col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  SignedPairReader reader(bundle, nullptr, blockIdx.x, k / 16, lane, 1.0f);
  half2 weights[8];
  reader.load(group, weights);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int index = (blockIdx.x * 32 + col) * k + group * 16 + j * 2;
    output[index] = __low2half(weights[j]);
    output[index + 1] = __high2half(weights[j]);
  }
}
void decode_matrix(torch::Tensor bundle, torch::Tensor out) {
  c10::cuda::CUDAGuard guard(out.device());
  decode_bundle<<<dim3(out.size(0) / 32, out.size(1) / 16), 32, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      bundle.data_ptr<uint8_t>(), reinterpret_cast<half*>(out.data_ptr()),
      out.size(0), out.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run(torch::Tensor x, std::vector<torch::Tensor> codes,
         std::vector<torch::Tensor> scales, torch::Tensor table,
         torch::Tensor bundle, torch::Tensor out, bool candidate) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.sizes() == at::IntArrayRef({8, 5120}) &&
                  out.sizes() == at::IntArrayRef({8, 4352}),
              "invalid pair shape");
  auto stream = at::cuda::getCurrentCUDAStream();
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
  if (candidate) {
    signed_qpn_pair<false><<<136, 256, 0, stream>>>(
        bundle.data_ptr<uint8_t>(), nullptr, xp,
        reinterpret_cast<half*>(out.data_ptr()), 4352, 5120, 1.0f);
  } else {
    Segs segs{};
    segs.nseg = 2;
    segs.pair = 1;
    segs.hout = reinterpret_cast<half*>(out.data_ptr());
    segs.hld = 4352;
    segs.tab = reinterpret_cast<const uint4*>(table.data_ptr());
    for (int i = 0; i < 2; ++i) {
      auto& s = segs.s[i];
      s.codes = reinterpret_cast<const uint4*>(codes[i].data_ptr());
      s.scale = reinterpret_cast<const uint4*>(scales[i].data_ptr());
      s.out = segs.hout;
      s.out_ld = 4352;
      s.n = 4352;
      s.fmt = IQ3S;
    }
    launch<4, 4, IQ3S, IQ3S>(segs, xp, 5120, 8, 5120, 160, 40, 1, nullptr,
                             nullptr, 68, stream, nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("decode", &decode_matrix);
}
