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
constexpr int kHeadDim = 128;
template <int Queries, int Keys>
using KernelFor =
    typename cutlass::gemm::kernel::H3FMHA<Half, cutlass::arch::Sm70, true,
                                           Queries, Keys, kHeadDim>::FMHAKernel;

template <int Queries, int Keys, bool Fixed>
__global__ __launch_bounds__(Queries * 2, 1) void h3_flash_v100_d128(
    typename KernelFor<Queries, Keys>::DirectParams params) {
  extern __shared__ __align__(16) unsigned char storage[];
  if constexpr (Fixed) {
    params.heads = 14;
    params.queries = 12323;
    params.keys = 12323;
  }
  KernelFor<Queries, Keys> kernel;
  kernel(params,
         *reinterpret_cast<typename KernelFor<Queries, Keys>::SharedStorage*>(
             storage));
}

template <int Queries, int Keys, bool Fixed = false>
void launch_attention(at::Tensor const& q, at::Tensor const& k,
                      at::Tensor const& v, at::Tensor& output, float scale) {
  using Kernel = KernelFor<Queries, Keys>;
  typename Kernel::DirectParams params{
      reinterpret_cast<Half*>(q.data_ptr()),
      reinterpret_cast<Half*>(k.data_ptr()),
      reinterpret_cast<Half*>(v.data_ptr()),
      reinterpret_cast<Half*>(output.data_ptr()),
      int(q.size(1)),
      int(k.size(1)),
      int(q.size(2)),
      scale};
  if constexpr (Keys == 128) {
    // The 64-query tile uses 34,304 bytes/block. Prefer full shared-memory
    // capacity for both query geometries without extra global storage.
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        h3_flash_v100_d128<Queries, Keys, Fixed>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  }
  if constexpr (Queries == 128) {
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(h3_flash_v100_d128<Queries, Keys, Fixed>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sizeof(typename Kernel::SharedStorage)));
  }
  h3_flash_v100_d128<Queries, Keys, Fixed>
      <<<dim3((q.size(1) + Queries - 1) / Queries, q.size(0) * q.size(2)),
         Kernel::kThreadCount, sizeof(typename Kernel::SharedStorage),
         at::cuda::getCurrentCUDAStream()>>>(params);
}

template <int Queries>
void dispatch_attention(at::Tensor const& q, at::Tensor const& k,
                        at::Tensor const& v, at::Tensor& output, float scale,
                        int selected) {
  if (selected == 128) {
    if (q.size(1) == 12323 && k.size(1) == 12323 && q.size(2) == 14)
      launch_attention<Queries, 128, true>(q, k, v, output, scale);
    else
      launch_attention<Queries, 128>(q, k, v, output, scale);
  } else {
    launch_attention<Queries, 64>(q, k, v, output, scale);
  }
}

at::Tensor aligned_contiguous(const at::Tensor& tensor) {
  auto result = tensor.contiguous();
  // contiguous() may preserve a contiguous view with an unaligned offset.
  if (reinterpret_cast<uintptr_t>(result.data_ptr()) % 16 != 0)
    result = result.clone();
  return result;
}
}  // namespace

at::Tensor h3_flash_attention_forward(at::Tensor q, at::Tensor k, at::Tensor v,
                                      double scale, int key_tile,
                                      int query_tile) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.scalar_type() == at::kHalf,
              "H3 FlashAttention-V100 requires CUDA FP16 BSND tensors");
  TORCH_CHECK(k.device() == q.device() && v.device() == q.device() &&
                  k.scalar_type() == q.scalar_type() &&
                  v.scalar_type() == q.scalar_type(),
              "H3 Q/K/V must share device and FP16 dtype");
  TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 &&
                  q.size(3) == kHeadDim && k.dim() == 4 && k.size(1) > 0 &&
                  k.size(0) == q.size(0) && k.size(2) == q.size(2) &&
                  k.size(3) == kHeadDim && k.sizes() == v.sizes(),
              "H3 FlashAttention-V100 requires non-empty D128 MHA");
  TORCH_CHECK(std::isfinite(scale) && scale > 0 &&
                  scale <= std::numeric_limits<float>::max(),
              "H3 attention scale must be finite and positive");
  TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
              "H3 FlashAttention-V100 is an inference-only operator");
  TORCH_CHECK(key_tile == 0 || key_tile == 64 || key_tile == 128,
              "H3 attention key tile must be 0, 64 or 128");
  TORCH_CHECK(query_tile == 64 || query_tile == 128,
              "H3 attention query tile must be 64 or 128");
  const c10::cuda::CUDAGuard guard(q.device());
  auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "H3 FlashAttention-V100 requires SM70");
  int64_t groups64 = q.size(0) * q.size(2);
  int64_t blocks64 = ((q.size(1) + query_tile - 1) / query_tile) * groups64;
  TORCH_CHECK(q.size(1) <= INT_MAX && k.size(1) <= INT_MAX &&
                  groups64 <= 65535 && blocks64 <= INT_MAX,
              "H3 attention shape exceeds kernel index limits");
  q = aligned_contiguous(q);
  k = aligned_contiguous(k);
  v = aligned_contiguous(v);
  auto output = at::empty_like(q);
  // Keep small/refiner requests on the previous arithmetic path. Large
  // self-attention reuses each Q fragment across twice as many keys.
  int selected =
      key_tile ? key_tile : (q.size(1) >= 1024 && k.size(1) >= 1024 ? 128 : 64);
  if (query_tile == 128)
    dispatch_attention<128>(q, k, v, output, float(scale), selected);
  else
    dispatch_attention<64>(q, k, v, output, float(scale), selected);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &h3_flash_attention_forward, pybind11::arg("q"),
        pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("scale"),
        pybind11::arg("key_tile") = 0, pybind11::arg("query_tile") = 64);
}
