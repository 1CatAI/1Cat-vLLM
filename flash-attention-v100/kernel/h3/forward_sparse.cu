// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Copyright contributors to the 1Cat-vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>

#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/kernel/default_gemm.h>

#include "default_fmha.h"

namespace {
using Half = cutlass::half_t;
using Kernel =
    typename cutlass::gemm::kernel::H3FMHA<Half, cutlass::arch::Sm70, true, 64,
                                           64, 128>::FMHAKernel;

// One warp compacts one mask row in ascending key-block order. No sorting,
// padded key arithmetic, host mask transfer or atomic update is necessary.
__global__ void pack_block_map(const bool* mask, int* indices, int* counts,
                               int rows, int blocks) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  int count = 0;
  for (int start = 0; start < blocks; start += 32) {
    const int block = start + lane;
    const bool selected = block < blocks && mask[int64_t(row) * blocks + block];
    const unsigned ballot = __ballot_sync(0xffffffff, selected);
    const unsigned before = (1u << lane) - 1;
    if (selected)
      indices[int64_t(row) * blocks + count + __popc(ballot & before)] = block;
    count += __popc(ballot);
  }
  if (lane == 0) counts[row] = count;
}

__global__ __launch_bounds__(128, 1) void sparse_attention(
    Kernel::DirectParams params) {
  extern __shared__ __align__(16) unsigned char storage[];
  Kernel kernel;
  kernel.template operator()<true>(
      params, *reinterpret_cast<Kernel::SharedStorage*>(storage));
}
}  // namespace

torch::Tensor sparse_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                             torch::Tensor block_map, torch::Tensor block_sizes,
                             double scale) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.scalar_type() == at::kHalf &&
                  q.is_contiguous() && q.size(0) > 0 && q.size(1) > 0 &&
                  q.size(1) % 64 == 0 && q.size(2) > 0 && q.size(3) == 128,
              "SM70 sparse attention requires contiguous FP16 [B,64*N,H,128]");
  for (const auto& operand : {k, v}) {
    TORCH_CHECK(
        operand.device() == q.device() && operand.sizes() == q.sizes() &&
            operand.scalar_type() == q.scalar_type() && operand.is_contiguous(),
        "SM70 sparse Q/K/V must share shape, device, dtype and layout");
  }
  TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
              "SM70 sparse attention is inference-only");
  TORCH_CHECK(std::isfinite(scale) && scale > 0 &&
                  scale <= std::numeric_limits<float>::max(),
              "SM70 sparse attention scale must be finite and positive");
  const int64_t blocks = q.size(1) / 64;
  const int64_t groups = q.size(0) * q.size(2);
  TORCH_CHECK(
      q.size(1) <= INT_MAX && groups <= 65535 && groups * blocks <= INT_MAX,
      "SM70 sparse attention shape exceeds index limits");
  TORCH_CHECK(block_map.device() == q.device() && block_map.dim() == 4 &&
                  block_map.scalar_type() == at::kBool &&
                  block_map.is_contiguous() && block_map.size(0) == q.size(0) &&
                  block_map.size(1) == q.size(2) &&
                  block_map.size(2) == blocks && block_map.size(3) == blocks,
              "SM70 sparse attention block map must be bool [B,H,N,N]");
  TORCH_CHECK(block_sizes.device() == q.device() && block_sizes.dim() == 1 &&
                  block_sizes.size(0) == blocks &&
                  block_sizes.is_contiguous() &&
                  block_sizes.scalar_type() == at::kInt,
              "SM70 sparse attention block sizes must be int32 [N]");
  const c10::cuda::CUDAGuard guard(q.device());
  auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "SM70 sparse attention requires SM70");
  TORCH_CHECK(
      block_sizes.min().item<int>() > 0 && block_sizes.max().item<int>() <= 64,
      "SM70 sparse attention block sizes must be in [1,64]");
  TORCH_CHECK(
      block_map.any(-1).all().item<bool>(),
      "SM70 sparse attention requires a selected key for every query block");
  // Aligned strides alone do not guarantee an aligned contiguous storage view.
  if (reinterpret_cast<uintptr_t>(q.data_ptr()) % 16) q = q.clone();
  if (reinterpret_cast<uintptr_t>(k.data_ptr()) % 16) k = k.clone();
  if (reinterpret_cast<uintptr_t>(v.data_ptr()) % 16) v = v.clone();
  auto indices = torch::empty({groups * blocks, blocks}, block_sizes.options());
  auto counts = torch::empty({groups * blocks}, block_sizes.options());
  auto output = torch::zeros_like(q);
  auto stream = at::cuda::getCurrentCUDAStream();
  pack_block_map<<<groups * blocks, 32, 0, stream>>>(
      block_map.data_ptr<bool>(), indices.data_ptr<int>(),
      counts.data_ptr<int>(), int(groups * blocks), int(blocks));
  Kernel::DirectParams params{reinterpret_cast<Half*>(q.data_ptr()),
                              reinterpret_cast<Half*>(k.data_ptr()),
                              reinterpret_cast<Half*>(v.data_ptr()),
                              reinterpret_cast<Half*>(output.data_ptr()),
                              int(q.size(1)),
                              int(k.size(1)),
                              int(q.size(2)),
                              float(scale),
                              indices.data_ptr<int>(),
                              counts.data_ptr<int>(),
                              block_sizes.data_ptr<int>(),
                              int(blocks)};
  sparse_attention<<<dim3(blocks, groups), 128, sizeof(Kernel::SharedStorage),
                     stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sparse_forward, pybind11::arg("q"), pybind11::arg("k"),
        pybind11::arg("v"), pybind11::arg("block_map"),
        pybind11::arg("block_sizes"), pybind11::arg("scale"));
}
