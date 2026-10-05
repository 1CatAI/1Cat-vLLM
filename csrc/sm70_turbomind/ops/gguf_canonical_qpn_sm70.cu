// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Reuse measured QPN bodies with existing canonical U4/LUT4 streams.
#include <climits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "gguf_linear_shared_a_sm70.cuh"

namespace vllm::sm70_gguf {
__global__ __launch_bounds__(512, 2) void canonical_iq4_n64_kernel(
    half* output, const half* input, const uint8_t* codes,
    const uint32_t* stats, float* partials, int* counters, int n, int k,
    int stat_stride) {
  __shared__ alignas(16) uint8_t shared[33793];
  native_linear_n64_body<CanonicalIq4Reader, true>(
      output, input, codes, stats, partials, counters, n, k, blockIdx.x,
      blockIdx.x, n, 0, stat_stride, shared);
}
}  // namespace vllm::sm70_gguf

namespace {
void validate_operand(torch::Tensor codes, torch::Tensor stats,
                      torch::Tensor input, int n, int k, bool lut) {
  TORCH_CHECK(
      codes.is_cuda() && stats.is_cuda() && codes.device() == input.device() &&
          stats.device() == input.device() && codes.is_contiguous() &&
          codes.scalar_type() == torch::kInt32 &&
          codes.sizes() == c10::IntArrayRef({k, n / 8}) &&
          stats.scalar_type() == (lut ? torch::kInt16 : torch::kInt32) &&
          stats.sizes() == c10::IntArrayRef({k / 32, n}) &&
          stats.stride(1) == 1 && stats.stride(0) >= n &&
          stats.stride(0) <= INT_MAX,
      "QPN requires canonical U4 codes and group32 stats on the "
      "activation device; coalesced stats may retain their row stride");
}
void validate_matrices(torch::Tensor output, torch::Tensor input,
                       int alignment) {
  TORCH_CHECK(
      output.is_cuda() && input.is_cuda() &&
          output.device() == input.device() && output.is_contiguous() &&
          input.is_contiguous() && output.dim() == 2 && input.dim() == 2 &&
          output.scalar_type() == torch::kFloat16 &&
          input.scalar_type() == torch::kFloat16 && input.size(0) == 8 &&
          output.size(0) == 8 && output.size(1) > 0 &&
          output.size(1) <= INT_MAX && input.size(1) > 0 &&
          input.size(1) <= INT_MAX && output.size(1) % alignment == 0 &&
          input.size(1) % 256 == 0,
      "Canonical QPN requires contiguous FP16 M8/N-tiled/K256 matrices");
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "Canonical QPN requires SM70");
}
template <class Reader>
void launch_pair(torch::Tensor output, torch::Tensor input, torch::Tensor gate,
                 torch::Tensor gate_stats, torch::Tensor up,
                 torch::Tensor up_stats, int n, int k, cudaStream_t stream) {
  using namespace vllm::sm70_gguf;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      native_pair_shared_a_kernel<Reader, Reader, true>,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  native_pair_shared_a_kernel<Reader, Reader, true><<<n / 32, 512, 0, stream>>>(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(gate.data_ptr<int32_t>()),
      reinterpret_cast<const uint8_t*>(up.data_ptr<int32_t>()), n, k,
      reinterpret_cast<const uint32_t*>(gate_stats.data_ptr()),
      reinterpret_cast<const uint32_t*>(up_stats.data_ptr()),
      gate_stats.stride(0), up_stats.stride(0));
}
}  // namespace

void gguf_canonical_pair_sm70_out(torch::Tensor output, torch::Tensor input,
                                  torch::Tensor gate, torch::Tensor gate_stats,
                                  torch::Tensor up, torch::Tensor up_stats,
                                  int64_t family) {
  validate_matrices(output, input, 32);
  TORCH_CHECK(family == 0 || family == 1,
              "Canonical pair requires affine or IQ4");
  const int n = output.size(1), k = input.size(1);
  TORCH_CHECK(k % 1024 == 0,
              "Canonical pair requires complete K1024 partitions");
  validate_operand(gate, gate_stats, input, n, k, family == 1);
  validate_operand(up, up_stats, input, n, k, family == 1);
  const c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (family == 1)
    launch_pair<vllm::sm70_gguf::CanonicalIq4Reader>(
        output, input, gate, gate_stats, up, up_stats, n, k, stream);
  else
    launch_pair<vllm::sm70_gguf::CanonicalAffineReader<4, 32>>(
        output, input, gate, gate_stats, up, up_stats, n, k, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_canonical_iq4_linear_sm70_out(
    torch::Tensor output, torch::Tensor input, torch::Tensor codes,
    torch::Tensor stats, torch::Tensor partials, torch::Tensor counters) {
  validate_matrices(output, input, 64);
  const int n = output.size(1), k = input.size(1);
  validate_operand(codes, stats, input, n, k, true);
  TORCH_CHECK(partials.is_cuda() && counters.is_cuda() &&
                  partials.device() == input.device() &&
                  counters.device() == input.device() &&
                  partials.is_contiguous() && counters.is_contiguous() &&
                  partials.scalar_type() == torch::kFloat32 &&
                  counters.scalar_type() == torch::kInt32 &&
                  partials.sizes() == c10::IntArrayRef({n / 64, 2, 512}) &&
                  counters.sizes() == c10::IntArrayRef({n / 64}),
              "Canonical IQ4 QPN requires matching FP32 reduction workspace");
  const c10::cuda::CUDAGuard guard(input.device());
  using namespace vllm::sm70_gguf;
  const auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      canonical_iq4_n64_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
      100));
  canonical_iq4_n64_kernel<<<dim3(n / 64, 2), 512, 0, stream>>>(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(codes.data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(stats.data_ptr<int16_t>()),
      partials.data_ptr<float>(), counters.data_ptr<int>(), n, k,
      stats.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
