// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <torch/types.h>
#include <flashinfer/gdn/sm70/gdn_fused_decode.cuh>

namespace {
using namespace flashinfer::sm70::gdn;

void run(torch::Tensor hidden, torch::Tensor weights, torch::Tensor qkv,
         torch::Tensor conv_w, torch::Tensor conv_bias, torch::Tensor conv,
         torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state,
         torch::Tensor indices, torch::Tensor output, torch::Tensor conv_out,
         torch::Tensor partial) {
  const c10::cuda::CUDAGuard guard(hidden.device());
  for (const auto& t : {hidden, weights, qkv, conv_w, conv_bias, conv, A_log,
                        dt_bias, state, indices, output, conv_out, partial})
    TORCH_CHECK(t.is_cuda() && t.device() == hidden.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 prototype only");
  const int B = hidden.size(0);
  TORCH_CHECK(B > 0 && B <= 64 && hidden.dim() == 2 &&
              hidden.size(1) == HIDDEN);
  TORCH_CHECK(weights.sizes() == torch::IntArrayRef({HIDDEN, N_BA}));
  TORCH_CHECK(qkv.sizes() == torch::IntArrayRef({B, QKV_DIM}) &&
              qkv.stride(1) == 1);
  TORCH_CHECK(conv_w.sizes() == torch::IntArrayRef({QKV_DIM, CONV_WIDTH}));
  TORCH_CHECK(conv_bias.numel() == 0 || conv_bias.numel() == QKV_DIM);
  TORCH_CHECK(conv.dim() == 3 && conv.size(1) == QKV_DIM && conv.size(2) == 3);
  TORCH_CHECK(state.dim() == 4 && state.size(0) == conv.size(0) &&
              state.size(1) == HV && state.size(2) == D && state.size(3) == D &&
              state.stride(1) == D * D && state.stride(2) == D &&
              state.stride(3) == 1);
  TORCH_CHECK(A_log.numel() == HV && dt_bias.numel() == HV &&
              indices.numel() == B);
  TORCH_CHECK(output.numel() == B * HV * D && conv_out.numel() == B * QKV_DIM);
  TORCH_CHECK(partial.numel() == B * N_BA * GEMV_NSPLIT);
  for (const auto& t : {hidden, weights, conv_w, conv_bias, A_log, dt_bias,
                        indices, output, conv_out, partial})
    TORCH_CHECK(t.is_contiguous());
  for (const auto& t : {hidden, weights, qkv, conv_w, conv_bias, conv, dt_bias,
                        output, conv_out})
    TORCH_CHECK(t.scalar_type() == at::kHalf);
  for (const auto& t : {state, A_log, partial})
    TORCH_CHECK(t.scalar_type() == at::kFloat);
  TORCH_CHECK(indices.scalar_type() == at::kInt);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 &&
              state.stride(0) % 4 == 0);
  const int block = 256;
  int occupancy = 0;
  if (B == 1) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, gdn_fused_decode_kernel<true>, block, 0));
  } else {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, gdn_fused_decode_kernel<false>, block, 0));
  }
  TORCH_CHECK(occupancy > 0);
  const int needed = (B * HV * D + ROWS_PER_WARP * 8 - 1) / (ROWS_PER_WARP * 8);
  const int grid = std::min(needed, occupancy * props->multiProcessorCount);
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(props->cooperativeLaunch,
              "GDN grid sync needs cooperative launch");
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeCooperative;
  attribute.val.cooperative = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
#define LAUNCH(B1)                                                          \
  C10_CUDA_CHECK(cudaLaunchKernelEx(                                        \
      &config, gdn_fused_decode_kernel<B1>, (const half*)hidden.data_ptr(), \
      (const half*)weights.data_ptr(), (const half*)qkv.data_ptr(),         \
      (const half*)conv_w.data_ptr(),                                       \
      conv_bias.numel() ? (const half*)conv_bias.data_ptr() : nullptr,      \
      (const half*)conv.data_ptr(), A_log.data_ptr<float>(),                \
      (const half*)dt_bias.data_ptr(), state.data_ptr<float>(),             \
      indices.data_ptr<int>(), 1.f / sqrtf(float(D)), state.stride(0),      \
      qkv.stride(0), conv.stride(0), conv.stride(1), conv.stride(2),        \
      (half*)output.data_ptr(), (half*)conv.data_ptr(),                     \
      state.data_ptr<float>(), partial.data_ptr<float>(),                   \
      (half*)conv_out.data_ptr(), B))
  if (B == 1) {
    LAUNCH(true);
  } else {
    LAUNCH(false);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_flashinfer_gdn_sm70, m) {
  m.def(
      "run(Tensor hidden, Tensor weights, Tensor qkv, Tensor conv_w, "
      "Tensor conv_bias, Tensor(a!) conv, Tensor A_log, Tensor dt_bias, "
      "Tensor(b!) state, Tensor indices, Tensor(c!) output, Tensor(d!) "
      "conv_out, "
      "Tensor(e!) partial) -> ()");
}
TORCH_LIBRARY_IMPL(_C_flashinfer_gdn_sm70, CUDA, m) { m.impl("run", &run); }
