// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "gguf_linear_shared_a_sm70.cuh"

namespace vllm::sm70_gguf {
template <class Reader, int Splits, bool Gdn>
__global__ __launch_bounds__(512, 2) void small_output_shared_a_kernel(
    half* output, const half* input, const uint8_t* weight, float* partials,
    int* counters) {
  __shared__ alignas(16) uint8_t shared[33793];
  native_linear_n64_body<Reader, false, Splits, Gdn>(
      output, input, weight, nullptr, partials, counters, 5120, 1536,
      blockIdx.x, blockIdx.x, 5120, 0, 5120, shared);
}
}  // namespace vllm::sm70_gguf

namespace {
template <int Type, int Splits, bool Gdn>
void launch(torch::Tensor output, torch::Tensor input, torch::Tensor weight,
            torch::Tensor partials, torch::Tensor counters,
            cudaStream_t stream) {
  using namespace vllm::sm70_gguf;
  using Reader = NativePairReader<Type>;
  TORCH_CHECK(weight.numel() == int64_t{5120} * 6 * Reader::kBlockBytes,
              "Small projection original record geometry mismatch");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      small_output_shared_a_kernel<Reader, Splits, Gdn>,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  small_output_shared_a_kernel<Reader, Splits, Gdn>
      <<<dim3(80, Splits), 512, 0, stream>>>(
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          weight.data_ptr<uint8_t>(), partials.data_ptr<float>(),
          counters.data_ptr<int>());
}

template <int Splits, bool Gdn>
void dispatch(torch::Tensor output, torch::Tensor input, torch::Tensor weight,
              torch::Tensor partials, torch::Tensor counters, int type,
              cudaStream_t stream) {
  switch (type) {
    case 16:
      launch<16, Splits, Gdn>(output, input, weight, partials, counters,
                              stream);
      break;
    case 17:
      launch<17, Splits, Gdn>(output, input, weight, partials, counters,
                              stream);
      break;
    case 18:
      launch<18, Splits, Gdn>(output, input, weight, partials, counters,
                              stream);
      break;
    case 21:
      launch<21, Splits, Gdn>(output, input, weight, partials, counters,
                              stream);
      break;
    case 22:
      launch<22, Splits, Gdn>(output, input, weight, partials, counters,
                              stream);
      break;
    case 23:
      launch<23, Splits, Gdn>(output, input, weight, partials, counters,
                              stream);
      break;
    case 29:
      launch<29, Splits, Gdn>(output, input, weight, partials, counters,
                              stream);
      break;
    default:
      TORCH_CHECK(false, "Unscreened small projection native format");
  }
}
}  // namespace

void gguf_small_output_sm70_out(torch::Tensor output, torch::Tensor input,
                                torch::Tensor weight, torch::Tensor partials,
                                torch::Tensor counters, int64_t type,
                                int64_t splits, bool gdn_head_tiling) {
  const auto device = input.device();
  auto check = [&](const torch::Tensor& tensor, at::ScalarType dtype) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == device &&
            tensor.is_contiguous() && tensor.scalar_type() == dtype,
        "Small GGUF projection requires matching contiguous CUDA operands");
  };
  check(output, torch::kFloat16);
  check(input, torch::kFloat16);
  check(weight, torch::kUInt8);
  check(partials, torch::kFloat32);
  check(counters, torch::kInt32);
  TORCH_CHECK(input.sizes() == c10::IntArrayRef({8, 1536}) &&
                  output.sizes() == c10::IntArrayRef({8, 5120}) &&
                  weight.dim() == 1 &&
                  partials.sizes() == c10::IntArrayRef({80, 2, 512}) &&
                  counters.sizes() == c10::IntArrayRef({80}) &&
                  (splits == 1 || splits == 2),
              "Small GGUF projection requires M8/N5120/K1536 and one or two "
              "partitions");
  const c10::cuda::CUDAGuard guard(device);
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "Requires SM70");
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (gdn_head_tiling) {
    if (splits == 1)
      dispatch<1, true>(output, input, weight, partials, counters, type,
                        stream);
    else
      dispatch<2, true>(output, input, weight, partials, counters, type,
                        stream);
  } else {
    if (splits == 1)
      dispatch<1, false>(output, input, weight, partials, counters, type,
                         stream);
    else
      dispatch<2, false>(output, input, weight, partials, counters, type,
                         stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
