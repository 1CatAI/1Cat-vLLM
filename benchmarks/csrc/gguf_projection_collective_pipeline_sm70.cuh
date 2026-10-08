// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research-only tile publication across projection and collective boundaries.
// Packet loads, sentinel handling and rank-order sum reuse custom_all_reduce.
#include <torch/extension.h>
#include "custom_all_reduce.cuh"
#include "sm70_turbomind/ops/gguf_projection_collective_pipeline_sm70.cuh"

namespace {

__global__ void init_pipeline_buffer(char* ptr, size_t bytes, bool control) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t offset = i * 2;
  const bool metadata =
      (offset >= vllm::kSm70PushNormOffset &&
       offset < vllm::kSm70PushNormOffset + vllm::kSm70PushNormMetaBytes) ||
      (offset >= vllm::kSm70PushNormPacketOffset &&
       offset < vllm::kSm70PushNormPacketOffset + vllm::kSm70PushNormMetaBytes);
  if (i < bytes / 2)
    reinterpret_cast<uint16_t*>(ptr)[i] =
        metadata ? 0 : vllm::kSm70Tp4PushAllreduceSentinel;
}

std::vector<int64_t> pipeline_buffer_sizes() {
  return {static_cast<int64_t>(vllm::kSm70Tp4PushAllreduceBufferBytes),
          static_cast<int64_t>(vllm::kSm70Tp4PushAllreduceBufferBytes)};
}

void pipeline_init(int64_t pointer, int64_t bytes, bool control) {
  init_pipeline_buffer<<<(bytes / 2 + 255) / 256, 256, 0,
                         at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<char*>(pointer), bytes, control);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pipeline_research(torch::Tensor x, torch::Tensor codes,
                       torch::Tensor scales, torch::Tensor table,
                       torch::Tensor partial, std::vector<int64_t> pointers,
                       int64_t rank, torch::Tensor residual,
                       torch::Tensor weight, torch::Tensor normalized,
                       torch::Tensor residual_out, int64_t format, bool fused) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                  x.is_contiguous() &&
                  x.sizes() == at::IntArrayRef({8, 4352}) &&
                  pointers.size() == 4 && rank >= 0 && rank < 4 &&
                  (format == IQ3S || format == LUT4 || format == Q4K),
              "invalid pipeline workload");
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(residual.scalar_type() == at::kFloat &&
                  weight.scalar_type() == at::kFloat,
              "FP32 residual and norm weight required");
  Segs down{};
  down.nseg = 1;
  down.tab = format == IQ3S ? reinterpret_cast<const uint4*>(table.data_ptr())
                            : nullptr;
  Seg& s = down.s[0];
  s.codes = reinterpret_cast<const uint4*>(codes.data_ptr());
  s.scale = reinterpret_cast<const uint4*>(scales.data_ptr());
  s.out = reinterpret_cast<half*>(partial.data_ptr());
  s.out_ld = s.n = 5120;
  s.fmt = format;
  vllm::RankData buffers{};
  for (int i = 0; i < 4; ++i)
    buffers.ptrs[i] = reinterpret_cast<void*>(pointers[i]);
  auto xp = reinterpret_cast<const half*>(x.data_ptr());
  auto norm = reinterpret_cast<half*>(normalized.data_ptr());
  if (format == IQ3S)
    run_pipeline<IQ3S>(down, xp, buffers, rank, residual.data_ptr<float>(),
                       weight.data_ptr<float>(), norm,
                       residual_out.data_ptr<float>(), 1e-6f, fused);
  else if (format == LUT4)
    run_pipeline<LUT4>(down, xp, buffers, rank, residual.data_ptr<float>(),
                       weight.data_ptr<float>(), norm,
                       residual_out.data_ptr<float>(), 1e-6f, fused);
  else
    run_pipeline<Q4K>(down, xp, buffers, rank, residual.data_ptr<float>(),
                      weight.data_ptr<float>(), norm,
                      residual_out.data_ptr<float>(), 1e-6f, fused);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &pipeline_research);
  m.def("init", &pipeline_init);
  m.def("sizes", &pipeline_buffer_sizes);
}
