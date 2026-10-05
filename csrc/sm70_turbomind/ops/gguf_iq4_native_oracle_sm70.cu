// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research-only operand oracle. Not included in production registrations.
#include <climits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "gguf_iq4_xs_native.cuh"

namespace {
template <class Output>
__global__ void iq4_native_operand_oracle(Output* output,
                                          const uint8_t* records, int k) {
  using D = vllm::sm70_gguf::Iq4XsNativeDecoder;
  const int col = threadIdx.x & 31;
  const int group = threadIdx.x >> 5;
  const int block = blockIdx.y;
  const int nb = k / 256;
  const uint8_t* tile = records + int64_t{blockIdx.x} * nb * 4352;
  const uint4 data = *reinterpret_cast<const uint4*>(tile + block * 4096 +
                                                     group * 512 + col * 16);
  const uint32_t packets[4] = {data.x, data.y, data.z, data.w};
  const auto params = D::parameters(tile, nb, block, col);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const auto values = D::fragment<Output>(params, group, packets[i]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      output[(int64_t{blockIdx.x} * 32 + col) * k + block * 256 + group * 32 +
             i * 8 + j] = values[j];
    }
  }
}

void iq4_native_decode(torch::Tensor output, torch::Tensor records) {
  TORCH_CHECK(output.is_cuda() && records.is_cuda() &&
                  output.device() == records.device() &&
                  output.is_contiguous() && records.is_contiguous() &&
                  output.dim() == 2 && records.dim() == 1 &&
                  records.scalar_type() == torch::kUInt8,
              "IQ4 native oracle requires contiguous CUDA matrices/records");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32 ||
                  output.scalar_type() == torch::kFloat16,
              "IQ4 native oracle supports Float and Half output");
  const int64_t n = output.size(0), k = output.size(1);
  TORCH_CHECK(n > 0 && n % 32 == 0 && k > 0 && k % 256 == 0 && n <= INT_MAX &&
                  k / 256 <= 65535 && records.numel() == n * (k / 256) * 136,
              "IQ4 native oracle requires complete N32/K256 original records");
  const c10::cuda::CUDAGuard guard(output.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(n / 32, k / 256);
  if (output.scalar_type() == torch::kFloat32)
    iq4_native_operand_oracle<float><<<grid, 256, 0, stream>>>(
        output.data_ptr<float>(), records.data_ptr<uint8_t>(), k);
  else
    iq4_native_operand_oracle<half><<<grid, 256, 0, stream>>>(
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        records.data_ptr<uint8_t>(), k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

#ifdef GGUF_IQ4_ORACLE_RESEARCH
TORCH_LIBRARY(gguf_iq4_native_oracle_research, ops) {
  ops.def("decode(Tensor(a!) output, Tensor records) -> ()");
  ops.impl("decode", torch::kCUDA, &iq4_native_decode);
}
#endif
