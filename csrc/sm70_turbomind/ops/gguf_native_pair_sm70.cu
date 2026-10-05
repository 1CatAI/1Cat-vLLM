// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Two-reader gated-pair operator. Model admission requires measured shapes.
#include <climits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "gguf_pair_shared_a_sm70.cuh"

using R12 = vllm::sm70_gguf::NativePairReader<12>;
using R17 = vllm::sm70_gguf::NativePairReader<17>;
using R18 = vllm::sm70_gguf::NativePairReader<18>;
using R21 = vllm::sm70_gguf::NativePairReader<21>;
using R22 = vllm::sm70_gguf::NativePairReader<22>;
using R23 = vllm::sm70_gguf::NativePairReader<23>;
namespace {
template <class Gate, class Up>
void launch_pair(torch::Tensor output, torch::Tensor input, torch::Tensor gate,
                 torch::Tensor up, int n, int k, cudaStream_t stream) {
  using namespace vllm::sm70_gguf;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      native_pair_shared_a_kernel<Gate, Up>,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  native_pair_shared_a_kernel<Gate, Up><<<n / 32, 512, 0, stream>>>(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), n, k);
}
}  // namespace

void gguf_native_pair_sm70_out(torch::Tensor output, torch::Tensor input,
                               torch::Tensor gate, torch::Tensor up,
                               int64_t gate_type, int64_t up_type) {
  TORCH_CHECK(
      (gate_type == 21 && (up_type == 23 || up_type == 18)) ||
          ((gate_type == 23 || gate_type == 18) && up_type == 21) ||
          (gate_type == 22 && (up_type == 18 || up_type == 21)) ||
          ((gate_type == 18 || gate_type == 21) && up_type == 22) ||
          (gate_type == 17 && up_type == 18) ||
          (gate_type == 22 && up_type == 17) ||
          (gate_type == 18 && up_type == 23) ||
          (gate_type == 12 && (up_type == 23 || up_type == 21)) ||
          (gate_type == 21 && up_type == 12) ||
          (gate_type == 23 && up_type == 12),
      "GGUF native pair requires a supported original-byte format pair");
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
  const auto block_bytes = [](int64_t type) {
    return type == 12   ? 144
           : type == 18 ? 98
           : type == 21 ? 110
           : type == 22 ? 82
                        : 136;
  };
  TORCH_CHECK(gate.numel() == n * (k / 256) * block_bytes(gate_type) &&
                  up.numel() == n * (k / 256) * block_bytes(up_type),
              "Mixed pair record byte count mismatch");
  const c10::cuda::CUDAGuard guard(input.device());
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "GGUF native pair requires SM70");
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (gate_type == 17)
    launch_pair<R17, R18>(output, input, gate, up, n, k, stream);
  else if (gate_type == 22 && up_type == 17)
    launch_pair<R22, R17>(output, input, gate, up, n, k, stream);
  else if (gate_type == 22 && up_type == 21)
    launch_pair<R22, R21>(output, input, gate, up, n, k, stream);
  else if (gate_type == 21 && up_type == 22)
    launch_pair<R21, R22>(output, input, gate, up, n, k, stream);
  else if (gate_type == 22)
    launch_pair<R22, R18>(output, input, gate, up, n, k, stream);
  else if (up_type == 22)
    launch_pair<R18, R22>(output, input, gate, up, n, k, stream);
  else if (gate_type == 12 && up_type == 21)
    launch_pair<R12, R21>(output, input, gate, up, n, k, stream);
  else if (gate_type == 21 && up_type == 12)
    launch_pair<R21, R12>(output, input, gate, up, n, k, stream);
  else if (gate_type == 12)
    launch_pair<R12, R23>(output, input, gate, up, n, k, stream);
  else if (up_type == 12)
    launch_pair<R23, R12>(output, input, gate, up, n, k, stream);
  else if (gate_type == 18 && up_type == 23)
    launch_pair<R18, R23>(output, input, gate, up, n, k, stream);
  else if (gate_type == 21 && up_type == 23)
    launch_pair<R21, R23>(output, input, gate, up, n, k, stream);
  else if (gate_type == 23)
    launch_pair<R23, R21>(output, input, gate, up, n, k, stream);
  else if (gate_type == 18)
    launch_pair<R18, R21>(output, input, gate, up, n, k, stream);
  else
    launch_pair<R21, R18>(output, input, gate, up, n, k, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
