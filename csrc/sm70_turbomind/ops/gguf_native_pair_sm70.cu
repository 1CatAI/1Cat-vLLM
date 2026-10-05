// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Two-reader gated-pair operator. Model admission requires measured shapes.
#include <climits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "gguf_pair_shared_a_sm70.cuh"

using R21 = vllm::sm70_gguf::NativePairReader<21>;
using R23 = vllm::sm70_gguf::NativePairReader<23>;
void gguf_native_pair_sm70_out(torch::Tensor output, torch::Tensor input,
                               torch::Tensor gate, torch::Tensor up,
                               int64_t gate_type, int64_t up_type) {
  TORCH_CHECK(
      (gate_type == 21 && up_type == 23) || (gate_type == 23 && up_type == 21),
      "GGUF native pair requires IQ3_S/IQ4_XS in either orientation");
  TORCH_CHECK(output.is_cuda() && input.is_cuda() && gate.is_cuda() &&
                  up.is_cuda() && output.device() == input.device() &&
                  gate.device() == input.device() &&
                  up.device() == input.device() && output.is_contiguous() &&
                  input.is_contiguous() && gate.is_contiguous() &&
                  up.is_contiguous() && output.dim() == 2 && input.dim() == 2 &&
                  gate.dim() == 1 && up.dim() == 1 &&
                  output.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  gate.scalar_type() == torch::kUInt8 &&
                  up.scalar_type() == torch::kUInt8,
              "GGUF native pair requires contiguous CUDA matrices and "
              "original records");
  const int64_t m = input.size(0), k = input.size(1), n = output.size(1);
  TORCH_CHECK(m == 8 && output.size(0) == m && n > 0 && n % 32 == 0 && k > 0 &&
                  k % 1024 == 0 && n <= INT_MAX && k <= INT_MAX,
              "GGUF native pair requires M8/N32/K1024");
  TORCH_CHECK(gate.numel() == n * (k / 256) * (gate_type == 21 ? 110 : 136) &&
                  up.numel() == n * (k / 256) * (up_type == 21 ? 110 : 136),
              "Mixed pair record byte count mismatch");
  const c10::cuda::CUDAGuard guard(input.device());
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "GGUF native pair requires SM70");
  const auto stream = at::cuda::getCurrentCUDAStream();
  using namespace vllm::sm70_gguf;
  if (gate_type == 21) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        native_pair_shared_a_kernel<R21, R23>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    native_pair_shared_a_kernel<R21, R23><<<n / 32, 512, 0, stream>>>(
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), n, k);
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        native_pair_shared_a_kernel<R23, R21>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    native_pair_shared_a_kernel<R23, R21><<<n / 32, 512, 0, stream>>>(
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), n, k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
