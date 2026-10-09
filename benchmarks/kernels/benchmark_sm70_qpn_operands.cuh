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

void pack(torch::Tensor input, torch::Tensor output) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
              input.is_contiguous() && input.dim() == 2 && input.size(0) == 8 &&
              (input.size(1) == 5120 || input.size(1) == 4352));
  TORCH_CHECK(
      output.device() == input.device() && output.sizes() == input.sizes() &&
      output.scalar_type() == torch::kFloat16 && output.is_contiguous());
  c10::cuda::CUDAGuard guard(input.device());
  vllm::sm70::pack_k16_input<<<(input.numel() / 2 + 255) / 256, 256, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()), 8, input.size(1));
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
    case 10: {
      if (gated) {
        static bool configured = false;
        if (!configured) {
          C10_CUDA_CHECK(cudaFuncSetAttribute(
              operand_warp_stage_gate,
              cudaFuncAttributePreferredSharedMemoryCarveout, 100));
          configured = true;
        }
        operand_warp_stage_gate<<<272, 384, 0, stream>>>(w, s, x, y, scale);
      } else {
        operand_packed_scaled_chain_down<16, 2>
            <<<160, 512, 0, stream>>>(w, s, x, y, n, k, m, scale);
      }
      break;
    }
    case 8: {
      auto kernel = gated ? operand_n16<true> : operand_n16<false>;
      static bool configured[2] = {false, false};
      if (!configured[gated]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 50));
        int resident = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, kernel, 384, 0));
        TORCH_CHECK(resident >= 4,
                    "N16 failed four-CTA resource gate: ", resident);
        configured[gated] = true;
      }
      kernel<<<gated ? 272 : 320, 384, 0, stream>>>(w, s, x, y, scale);
      break;
    }
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
  library.def("pack(Tensor input, Tensor(a!) output) -> ()");
}
TORCH_LIBRARY_IMPL(_qpn_operands700, CUDA, library) {
  library.impl("run", &run);
  library.impl("evict", &evict);
  library.impl("pack", &pack);
}
