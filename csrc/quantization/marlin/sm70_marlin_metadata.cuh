#pragma once

// Metadata helpers shared by the SM70 dense (_C) and MoE (_moe_C) Marlin
// dispatchers. Marlin's loaders store scales, zero points and biases in
// Marlin's column order; the SM70 kernels read and write logical columns.
// Internal linkage: each extension compiles its own copy.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/all.h>

namespace {

__device__ __forceinline__ int sm70_inverse_zp_interleave(int idx,
                                                          int num_bits) {
  if (num_bits == 4) {
    constexpr int inv[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    return inv[idx];
  }
  constexpr int inv[4] = {0, 2, 1, 3};
  return inv[idx];
}

__device__ __forceinline__ int sm70_inverse_scale_perm(int idx, int perm_len) {
  if (perm_len == 64) {
    return (idx % 8) * 8 + idx / 8;
  }
  constexpr int inv[32] = {0,  1,  8,  9,  16, 17, 24, 25, 2,  3,  10,
                           11, 18, 19, 26, 27, 4,  5,  12, 13, 20, 21,
                           28, 29, 6,  7,  14, 15, 22, 23, 30, 31};
  return inv[idx];
}

__global__ void sm70_unpermute_half_metadata_kernel(
    const half* __restrict__ src, half* __restrict__ dst, int64_t total,
    int perm_len) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t block = idx / perm_len;
  const int pos = static_cast<int>(idx % perm_len);
  dst[idx] = src[block * perm_len + sm70_inverse_scale_perm(pos, perm_len)];
}

// Undo Marlin's column permutation of fp16 metadata: the group-scale order
// (64) or the single-scale order (32) that marlin_permute_scales and
// marlin_permute_bias apply, per row of N columns.
torch::Tensor sm70_unpermute_half(torch::Tensor const& src, int perm_len) {
  TORCH_CHECK(src.scalar_type() == at::ScalarType::Half,
              "SM70 Marlin metadata must be fp16.");
  TORCH_CHECK(src.numel() % perm_len == 0,
              "SM70 Marlin metadata size is not divisible by ", perm_len);
  auto out = torch::empty_like(src);
  const int64_t total = out.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  sm70_unpermute_half_metadata_kernel<<<blocks, threads, 0,
                                        at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(src.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), total, perm_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace
