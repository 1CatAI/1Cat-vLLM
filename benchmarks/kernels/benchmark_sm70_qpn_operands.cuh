// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Appended to source-extracted research kernels by the benchmark builder.

__global__ void operand_evict(const int* input, unsigned long long* sink,
                              int n) {
  unsigned long long value = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    value += static_cast<unsigned>(input[i]);
  }
  for (int d = 16; d; d >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, d);
  }
  __shared__ unsigned long long partials[8];
  if ((threadIdx.x & 31) == 0) partials[threadIdx.x >> 5] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    value = 0;
    for (int w = 0; w < 8; ++w) value += partials[w];
    sink[blockIdx.x] = value;
  }
}

void evict(torch::Tensor input, torch::Tensor sink) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kInt32 &&
              input.is_contiguous() && input.numel() == 33554432);
  TORCH_CHECK(sink.device() == input.device() &&
              sink.scalar_type() == torch::kInt64 && sink.is_contiguous() &&
              sink.numel() == 256);
  c10::cuda::CUDAGuard guard(input.device());
  operand_evict<<<256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<int>(),
      reinterpret_cast<unsigned long long*>(sink.data_ptr<int64_t>()),
      input.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run(torch::Tensor output, torch::Tensor input, torch::Tensor codes,
         torch::Tensor scales, double global_scale, bool gated,
         int64_t variant) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
              input.is_contiguous() && input.dim() == 2 && input.size(0) == 8);
  const int m = input.size(0), k = input.size(1), n = output.size(1);
  TORCH_CHECK(k == (gated ? 5120 : 4352) && n == (gated ? 4352 : 5120));
  TORCH_CHECK(output.scalar_type() == torch::kFloat16 &&
              output.is_contiguous() && output.size(0) == m);
  TORCH_CHECK(codes.scalar_type() == torch::kUInt8 && codes.is_contiguous() &&
              codes.numel() == static_cast<int64_t>(n) * k / (gated ? 1 : 2));
  TORCH_CHECK(scales.scalar_type() == torch::kUInt8 && scales.is_contiguous() &&
              scales.numel() == static_cast<int64_t>(n) * k / (gated ? 8 : 16));
  TORCH_CHECK(output.device() == input.device() &&
              codes.device() == input.device() &&
              scales.device() == input.device());
  c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* w = codes.data_ptr<uint8_t>();
  const auto* s = scales.data_ptr<uint8_t>();
  const auto* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(output.data_ptr<at::Half>());
  const float scale = static_cast<float>(global_scale);
  switch (variant) {
    // GENERATED_LAUNCHES
    default:
      TORCH_CHECK(false, "Unknown operand diagnostic variant");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY(_qpn_operands700, library) {
  library.def(
      "run(Tensor(a!) out, Tensor input, Tensor codes, Tensor scales, "
      "float global_scale, bool gated, int variant) -> ()");
  library.def("evict(Tensor input, Tensor(a!) sink) -> ()");
}
TORCH_LIBRARY_IMPL(_qpn_operands700, CUDA, library) {
  library.impl("run", &run);
  library.impl("evict", &evict);
}
