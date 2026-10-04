// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Prepared layouts and register decoders reuse TurboMind (Apache-2.0).
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <type_traits>
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/transform.h"

namespace {
using turbomind::gemm::StridedPtr;

template <int Type>
struct CanonicalDecoder {
  using Code =
      std::conditional_t<Type == 20, turbomind::uint4_t, turbomind::uint2_t>;
  using Stats = std::conditional_t<Type == 20, uint16_t, uint32_t>;
  using Transform =
      std::conditional_t<Type == 20,
                         turbomind::gemm::Transform_HMMA_SM70_Lut4<0>,
                         turbomind::gemm::Transform_HMMA_SIMT_B>;

  __device__ static turbomind::Array<half, 8> fragment(const void* weight,
                                                       const void* stats, int k,
                                                       int n, int col,
                                                       int base) {
    // The same N32/K8 storage consumed by the canonical dequant and mma884.
    const int64_t packet =
        (int64_t{col / 32} * (k / 8) + base / 8) * 32 + col % 32;
    turbomind::Array<Code, 8> data[1][1];
    data[0][0] =
        reinterpret_cast<const turbomind::Array<Code, 8>*>(weight)[packet];
    turbomind::Array<Stats, 1> coefficients[1][1];
    coefficients[0][0][0] =
        static_cast<const Stats*>(stats)[int64_t{base / 32} * n + col];
    turbomind::Array<half, 8> decoded[1][1];
    Transform::apply(decoded, 0, data, coefficients, 1);
    return decoded[0][0];
  }
};

template <int Type>
__global__ void small_grouped_vec_kernel(half* output, const half* input,
                                         const int* offsets,
                                         const StridedPtr* weights,
                                         const StridedPtr* stats, int experts,
                                         int k, int n) {
  const int slot = blockIdx.y;
  // Use route slots to cover active experts. Binary search uses only the
  // already prepared offsets; no sort, padded tile or host count is needed.
  int lo = 0, hi = experts;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (offsets[mid + 1] <= slot)
      lo = mid + 1;
    else
      hi = mid;
  }
  const int expert = lo;
  const int begin = offsets[expert], end = offsets[expert + 1];
  if (slot != begin) return;
  const int col = blockIdx.x * 16 + threadIdx.x % 16;
  const int part = threadIdx.x / 16;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  __shared__ float partial[4][16];
  const void* weight = weights[expert].ptr;
  const void* coefficients = stats[expert].ptr;
  for (int row = begin; row < end; ++row) {
    float sum = 0.f;
    for (int base = part * 8; base < k; base += 64) {
      const auto decoded = CanonicalDecoder<Type>::fragment(
          weight, coefficients, k, n, col, base);
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const float2 w =
            __half22float2(*reinterpret_cast<const half2*>(&decoded[i]));
        const float2 x = __half22float2(*reinterpret_cast<const half2*>(
            input + int64_t{row} * k + base + i));
        sum = fmaf(w.x, x.x, sum);
        sum = fmaf(w.y, x.y, sum);
      }
    }
    sum += __shfl_down_sync(0xffffffffU, sum, 16);
    if (lane < 16) partial[warp][lane] = sum;
    __syncthreads();
    if (warp == 0 && lane < 16) {
      float value = 0.f;
#pragma unroll
      for (int p = 0; p < 4; ++p) value += partial[p][lane];
      output[int64_t{row} * n + col] = __float2half_rn(value);
    }
    __syncthreads();
  }
}

template <int Type>
void launch(torch::Tensor out, torch::Tensor input, torch::Tensor offsets,
            torch::Tensor weights, torch::Tensor stats, int experts,
            cudaStream_t stream) {
  small_grouped_vec_kernel<Type>
      <<<dim3(out.size(1) / 16, input.size(0)), 128, 0, stream>>>(
          reinterpret_cast<half*>(out.data_ptr()),
          reinterpret_cast<const half*>(input.data_ptr()),
          offsets.data_ptr<int>(),
          reinterpret_cast<const StridedPtr*>(weights.data_ptr()),
          reinterpret_cast<const StridedPtr*>(stats.data_ptr()), experts,
          input.size(1), out.size(1));
}
}  // namespace

void gguf_small_grouped_vec_sm70_out(torch::Tensor out, torch::Tensor input,
                                     torch::Tensor offsets,
                                     torch::Tensor weight_ptrs,
                                     torch::Tensor stats_ptrs,
                                     int64_t source_type, int64_t num_experts,
                                     int64_t group_size) {
  TORCH_CHECK(source_type == 20 || source_type == 42,
              "Small grouped vector supports IQ4_NL and Q2_0");
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && out.dim() == 2 &&
                  input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16 &&
                  input.is_contiguous() && out.is_contiguous(),
              "Small grouped vector requires contiguous FP16 matrices");
  for (const auto& tensor : {out, offsets, weight_ptrs, stats_ptrs})
    TORCH_CHECK(tensor.device() == input.device() && tensor.is_contiguous(),
                "Small grouped vector tensors must share a CUDA device");
  const int64_t m = input.size(0), k = input.size(1), n = out.size(1);
  TORCH_CHECK(m <= 320 && k > 0 && k <= INT_MAX && k % 32 == 0 && n > 0 &&
                  n <= INT_MAX && n % 32 == 0 && out.size(0) == m &&
                  num_experts > 0 && num_experts <= 65535 && group_size == 32 &&
                  offsets.dim() == 1 &&
                  offsets.scalar_type() == torch::kInt32 &&
                  offsets.numel() == num_experts + 1 &&
                  weight_ptrs.scalar_type() == torch::kUInt8 &&
                  stats_ptrs.scalar_type() == torch::kUInt8 &&
                  weight_ptrs.numel() == num_experts * sizeof(StridedPtr) &&
                  stats_ptrs.numel() == num_experts * sizeof(StridedPtr),
              "Small grouped vector descriptor mismatch");
  if (!m) return;
  const c10::cuda::CUDAGuard guard(input.device());
  const auto* device = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(device->major == 7 && device->minor == 0, "Requires SM70");
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (source_type == 20)
    launch<20>(out, input, offsets, weight_ptrs, stats_ptrs, num_experts,
               stream);
  else
    launch<42>(out, input, offsets, weight_ptrs, stats_ptrs, num_experts,
               stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
