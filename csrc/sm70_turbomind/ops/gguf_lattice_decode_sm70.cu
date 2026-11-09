// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include "gguf_lattice_canonical.cuh"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"

namespace {
using turbomind::gemm::StridedPtr;

template <int Type>
__global__ void lattice_grouped_vec_kernel(half* output, const half* input,
                                           const int* offsets,
                                           const StridedPtr* weights,
                                           const StridedPtr* stats, int m,
                                           int k, int n) {
  using Decoder =
      vllm::sm70_gguf::LatticeCanonicalDecoder<Type, Type == 18 ? 8 : 1>;
  using Transform = typename Decoder::Transform;
  const int expert = blockIdx.y;
  const int begin = offsets[expert], end = offsets[expert + 1];
  if (begin == end) return;
  const int col = blockIdx.x * 16 + threadIdx.x % 16;
  const int part = threadIdx.x / 16;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  __shared__ __align__(16) uint8_t grid[Transform::kCodebookBytes];
  __shared__ float partial[8][16];
  Transform::initialize(grid);
  const void* weight = weights[expert].ptr;
  const void* coefficients = stats[expert].ptr;
  // Each row uses the same FP16 reconstructed weights as mma884. Products
  // and every reduction stage accumulate in FP32; no activation quantization.
  for (int row = begin; row < end; ++row) {
    float sum = 0.f;
    for (int base = part * 8; base < k; base += 16 * 8) {
      const auto decoded =
          Decoder::fragment(weight, coefficients, k, n, col, base, grid);
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const half2 w = *reinterpret_cast<const half2*>(&decoded[i]);
        const half2 a = *reinterpret_cast<const half2*>(
            input + static_cast<int64_t>(row) * k + base + i);
        const float2 wf = __half22float2(w), af = __half22float2(a);
        sum = fmaf(wf.x, af.x, sum);
        sum = fmaf(wf.y, af.y, sum);
      }
    }
    sum += __shfl_down_sync(0xffffffffU, sum, 16);
    if (lane < 16) partial[warp][lane] = sum;
    __syncthreads();
    if (warp == 0 && lane < 16) {
      float value = 0.f;
#pragma unroll
      for (int p = 0; p < 8; ++p) value += partial[p][lane];
      output[static_cast<int64_t>(row) * n + col] = __float2half_rn(value);
    }
    __syncthreads();
  }
}

template <int Type>
void launch_vec(torch::Tensor out, torch::Tensor input, torch::Tensor offsets,
                torch::Tensor weights, torch::Tensor stats, int experts,
                cudaStream_t stream) {
  const int m = input.size(0), k = input.size(1), n = out.size(1);
  lattice_grouped_vec_kernel<Type><<<dim3(n / 16, experts), 256, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      reinterpret_cast<const half*>(input.data_ptr()), offsets.data_ptr<int>(),
      reinterpret_cast<const StridedPtr*>(weights.data_ptr()),
      reinterpret_cast<const StridedPtr*>(stats.data_ptr()), m, k, n);
}
}  // namespace

void gguf_lattice_grouped_vec_sm70_out(torch::Tensor out, torch::Tensor input,
                                       torch::Tensor offsets,
                                       torch::Tensor weight_ptrs,
                                       torch::Tensor stats_ptrs,
                                       int64_t source_type, int64_t num_experts,
                                       int64_t group_size) {
  TORCH_CHECK(source_type == 16 || source_type == 17 || source_type == 18 ||
                  source_type == 19 || source_type == 21 || source_type == 22 ||
                  source_type == 29,
              "GGUF lattice decode source type is unsupported");
  const int group =
      source_type == 17 || source_type == 22 || source_type == 29 ? 16 : 32;
  TORCH_CHECK(group_size == group, "GGUF lattice decode group mismatch");
  TORCH_CHECK(input.is_cuda() && out.device() == input.device() &&
                  offsets.device() == input.device() &&
                  weight_ptrs.device() == input.device() &&
                  stats_ptrs.device() == input.device(),
              "GGUF lattice decode tensors must share a CUDA device");
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat16 &&
          out.scalar_type() == torch::kFloat16 && input.dim() == 2 &&
          out.dim() == 2 && input.is_contiguous() && out.is_contiguous() &&
          offsets.scalar_type() == torch::kInt32 && offsets.is_contiguous() &&
          offsets.dim() == 1 && weight_ptrs.scalar_type() == torch::kUInt8 &&
          stats_ptrs.scalar_type() == torch::kUInt8 &&
          weight_ptrs.is_contiguous() && stats_ptrs.is_contiguous(),
      "GGUF lattice decode requires FP16 matrices and prepared pointers");
  const int64_t m = input.size(0), k = input.size(1), n = out.size(1);
  TORCH_CHECK(m <= INT_MAX && n > 0 && n <= INT_MAX && n % 32 == 0 && k > 0 &&
                  k <= INT_MAX && k % group == 0 && out.size(0) == m &&
                  num_experts > 0 && num_experts <= 65535 &&
                  offsets.numel() == num_experts + 1 &&
                  weight_ptrs.numel() == num_experts * sizeof(StridedPtr) &&
                  stats_ptrs.numel() == num_experts * sizeof(StridedPtr),
              "GGUF lattice decode descriptor shape mismatch");
  if (m == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(TYPE)                                               \
  case TYPE:                                                       \
    launch_vec<TYPE>(out, input, offsets, weight_ptrs, stats_ptrs, \
                     num_experts, stream);                         \
    break
  switch (source_type) {
    LAUNCH(16);
    LAUNCH(17);
    LAUNCH(18);
    LAUNCH(19);
    LAUNCH(21);
    LAUNCH(22);
    LAUNCH(29);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
