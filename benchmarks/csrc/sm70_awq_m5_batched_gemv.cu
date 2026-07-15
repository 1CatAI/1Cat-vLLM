// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace {

constexpr int kM = 5;
constexpr int kCtaN = 4;
constexpr int kInterleave = 4;
constexpr int kOutputColsPerCta = kCtaN * kInterleave;
constexpr int kThreads = 128;
constexpr int kThreadK = 32;
constexpr int kCtaK = 1024;
constexpr int kTileK = 64;

__device__ __forceinline__ int awq_nibble_shift(int col_in_word) {
  // AWQ stores each 8-column group in [0, 2, 4, 6, 1, 3, 5, 7] order.
  return 4 * ((col_in_word >> 1) + ((col_in_word & 1) << 2));
}

__device__ __forceinline__ uint32_t load_awq_u4(
    const int32_t* __restrict__ qweight, int k, int col, int qwords_per_row) {
  const uint32_t packed =
      static_cast<uint32_t>(qweight[k * qwords_per_row + col / 8]);
  return (packed >> awq_nibble_shift(col & 7)) & 0xfu;
}

__global__ void prepare_weight_kernel(int32_t* __restrict__ prepared,
                                      const int32_t* __restrict__ qweight,
                                      int k, int n) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t words = static_cast<int64_t>(k) * n / 8;
  if (index >= words) {
    return;
  }

  const int words_per_n_group = k / 2;
  const int n_group = static_cast<int>(index / words_per_n_group);
  const int word_in_group = static_cast<int>(index % words_per_n_group);
  const int interleaved_k = word_in_group * 8;
  const int lane = (interleaved_k / kTileK) & (kInterleave - 1);
  const int real_k = (interleaved_k / (kTileK * kInterleave)) * kTileK +
                     (interleaved_k % kTileK);
  const int col = n_group * kInterleave + lane;
  const int qwords_per_row = n / 8;

  uint32_t packed = 0;
#pragma unroll
  for (int nibble = 0; nibble < 8; ++nibble) {
    // This makes the fast int4-to-half2 converter return logical K order.
    const int delta_k = nibble < 4 ? nibble * 2 : (nibble - 4) * 2 + 1;
    packed |= load_awq_u4(qweight, real_k + delta_k, col, qwords_per_row)
              << (nibble * 4);
  }
  prepared[index] = static_cast<int32_t>(packed);
}

__global__ void prepare_zero_bias_kernel(__half* __restrict__ zero_bias,
                                         const int32_t* __restrict__ qzeros,
                                         const __half* __restrict__ scales,
                                         int groups, int n) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(groups) * n;
  if (index >= elements) {
    return;
  }
  const int col = static_cast<int>(index % n);
  const int group = static_cast<int>(index / n);
  const uint32_t packed =
      static_cast<uint32_t>(qzeros[group * (n / 8) + col / 8]);
  const int zero = (packed >> awq_nibble_shift(col & 7)) & 0xf;
  const __half zero_h = __int2half_rn(zero);
  zero_bias[index] = __hneg(__hmul(zero_h, scales[index]));
}

// Same biased uint4 conversion dataflow used by TensorRT-LLM's interleaved
// batched-GEMV. The prepared word order makes the returned half array logical.
__device__ __forceinline__ uint4 convert_u4x8_to_half(uint32_t source) {
  uint4 result;
  auto* h = reinterpret_cast<uint32_t*>(&result);
  constexpr uint32_t kBottomMask = 0x000f000f;
  constexpr uint32_t kTopMask = 0x00f000f0;
  constexpr uint32_t kMagic = 0x64006400;
  constexpr uint32_t kImmLut = (0xf0 & 0xcc) | 0xaa;
  const uint32_t top = source >> 8;

  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(source), "n"(kBottomMask), "n"(kMagic), "n"(kImmLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(source), "n"(kTopMask), "n"(kMagic), "n"(kImmLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top), "n"(kBottomMask), "n"(kMagic), "n"(kImmLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top), "n"(kTopMask), "n"(kMagic), "n"(kImmLut));

  constexpr uint32_t kOneSixteenth = 0x2c002c00;
  constexpr uint32_t kNeg64 = 0xd400d400;
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(kMagic));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(h[1])
               : "r"(h[1]), "r"(kOneSixteenth), "r"(kNeg64));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(kMagic));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(h[3])
               : "r"(h[3]), "r"(kOneSixteenth), "r"(kNeg64));
  return result;
}

template <int AccumMode>
__global__ __launch_bounds__(kThreads) void awq_m5_batched_gemv_kernel(
    __half* __restrict__ out, const __half* __restrict__ input,
    const int32_t* __restrict__ prepared_weight,
    const __half* __restrict__ scales, const __half* __restrict__ zero_bias,
    int n, int k, int group_size) {
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane_id = tid & 31;
  const int interleave_lane = (tid >> 1) & (kInterleave - 1);
  const int thread_k = (tid >> 3) * kTileK + (tid & 1) * kThreadK;
  const int output_base = static_cast<int>(blockIdx.x) * kOutputColsPerCta;
  const int words_per_n_group = k / 2;

  __half2 acc_h[kM][kCtaN / 2];
  float acc_f[kM][kCtaN];
#pragma unroll
  for (int row = 0; row < kM; ++row) {
#pragma unroll
    for (int col_pair = 0; col_pair < kCtaN / 2; ++col_pair) {
      acc_h[row][col_pair] = __float2half2_rn(0.0f);
    }
#pragma unroll
    for (int col = 0; col < kCtaN; ++col) {
      acc_f[row][col] = 0.0f;
    }
  }

  for (int cta_k = 0; cta_k + thread_k < k; cta_k += kCtaK) {
    if constexpr (AccumMode == 1) {
#pragma unroll
      for (int row = 0; row < kM; ++row) {
#pragma unroll
        for (int col_pair = 0; col_pair < kCtaN / 2; ++col_pair) {
          acc_h[row][col_pair] = __float2half2_rn(0.0f);
        }
      }
    }
    const int real_k = cta_k + thread_k;
    const int group = real_k / group_size;
    __half scale_values[kCtaN];
    __half bias_values[kCtaN];
    uint4 packed_values[kCtaN];

#pragma unroll
    for (int col = 0; col < kCtaN; ++col) {
      const int real_col = output_base + interleave_lane + col * kInterleave;
      scale_values[col] = scales[group * n + real_col];
      bias_values[col] = zero_bias[group * n + real_col];
      const int n_group = static_cast<int>(blockIdx.x) * kCtaN + col;
      const int word_offset = cta_k / 2 + tid * 4;
      packed_values[col] = *reinterpret_cast<const uint4*>(
          prepared_weight + n_group * words_per_n_group + word_offset);
    }

    const __half2 scale01 = __halves2half2(scale_values[0], scale_values[1]);
    const __half2 scale23 = __halves2half2(scale_values[2], scale_values[3]);
    const __half2 bias01 = __halves2half2(bias_values[0], bias_values[1]);
    const __half2 bias23 = __halves2half2(bias_values[2], bias_values[3]);

#pragma unroll
    for (int word = 0; word < 4; ++word) {
      uint4 converted[kCtaN];
#pragma unroll
      for (int col = 0; col < kCtaN; ++col) {
        const auto* packed_words =
            reinterpret_cast<const uint32_t*>(&packed_values[col]);
        converted[col] = convert_u4x8_to_half(packed_words[word]);
      }
      const auto* q0 = reinterpret_cast<const __half*>(&converted[0]);
      const auto* q1 = reinterpret_cast<const __half*>(&converted[1]);
      const auto* q2 = reinterpret_cast<const __half*>(&converted[2]);
      const auto* q3 = reinterpret_cast<const __half*>(&converted[3]);
      __half2 weights[8][kCtaN / 2];
#pragma unroll
      for (int elem = 0; elem < 8; ++elem) {
        weights[elem][0] =
            __hfma2(__halves2half2(q0[elem], q1[elem]), scale01, bias01);
        weights[elem][1] =
            __hfma2(__halves2half2(q2[elem], q3[elem]), scale23, bias23);
      }

#pragma unroll
      for (int row = 0; row < kM; ++row) {
        const auto activations = *reinterpret_cast<const uint4*>(
            input + row * k + real_k + word * 8);
        const auto* act = reinterpret_cast<const __half*>(&activations);
#pragma unroll
        for (int elem = 0; elem < 8; ++elem) {
          if constexpr (AccumMode == 2) {
            const float x = __half2float(act[elem]);
            acc_f[row][0] = fmaf(x, __half2float(__low2half(weights[elem][0])),
                                 acc_f[row][0]);
            acc_f[row][1] = fmaf(x, __half2float(__high2half(weights[elem][0])),
                                 acc_f[row][1]);
            acc_f[row][2] = fmaf(x, __half2float(__low2half(weights[elem][1])),
                                 acc_f[row][2]);
            acc_f[row][3] = fmaf(x, __half2float(__high2half(weights[elem][1])),
                                 acc_f[row][3]);
          } else {
            const __half2 x = __half2half2(act[elem]);
            acc_h[row][0] = __hfma2(weights[elem][0], x, acc_h[row][0]);
            acc_h[row][1] = __hfma2(weights[elem][1], x, acc_h[row][1]);
          }
        }
      }
    }
    if constexpr (AccumMode == 1) {
#pragma unroll
      for (int row = 0; row < kM; ++row) {
#pragma unroll
        for (int col_pair = 0; col_pair < kCtaN / 2; ++col_pair) {
          acc_f[row][col_pair * 2] +=
              __half2float(__low2half(acc_h[row][col_pair]));
          acc_f[row][col_pair * 2 + 1] +=
              __half2float(__high2half(acc_h[row][col_pair]));
        }
      }
    }
  }

  __shared__ float partials[4 * kM * kCtaN * kInterleave];
#pragma unroll
  for (int row = 0; row < kM; ++row) {
#pragma unroll
    for (int col = 0; col < kCtaN; ++col) {
      float value;
      if constexpr (AccumMode != 0) {
        value = acc_f[row][col];
      } else {
        const __half2 packed = acc_h[row][col / 2];
        value =
            __half2float((col & 1) ? __high2half(packed) : __low2half(packed));
      }
      value += __shfl_xor_sync(0xffffffffu, value, 16);
      value += __shfl_xor_sync(0xffffffffu, value, 8);
      value += __shfl_xor_sync(0xffffffffu, value, 1);
      if (lane_id < 8 && (lane_id & 1) == 0) {
        const int lane = lane_id / 2;
        partials[warp * (kM * kCtaN * kInterleave) +
                 row * (kCtaN * kInterleave) + col * kInterleave + lane] =
            value;
      }
    }
  }
  __syncthreads();

  for (int index = tid; index < kM * kOutputColsPerCta; index += kThreads) {
    const int row = index / kOutputColsPerCta;
    const int local_col = index % kOutputColsPerCta;
    const int col = local_col / kInterleave;
    const int lane = local_col % kInterleave;
    float value = 0.0f;
#pragma unroll
    for (int source_warp = 0; source_warp < 4; ++source_warp) {
      value += partials[source_warp * (kM * kCtaN * kInterleave) +
                        row * (kCtaN * kInterleave) + col * kInterleave + lane];
    }
    out[row * n + output_base + local_col] = __float2half_rn(value);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> prepare(
    torch::Tensor qweight, torch::Tensor scales, torch::Tensor qzeros,
    int64_t group_size) {
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda() && qzeros.is_cuda(),
              "SM70 M5 AWQ prepare expects CUDA tensors.");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32, "qweight must be int32.");
  TORCH_CHECK(qzeros.scalar_type() == torch::kInt32, "qzeros must be int32.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16,
              "scales must be float16.");
  TORCH_CHECK(qweight.dim() == 2 && scales.dim() == 2 && qzeros.dim() == 2,
              "SM70 M5 AWQ prepare expects rank-2 tensors.");
  TORCH_CHECK(group_size == 128,
              "The first SM70 M5 prototype supports group_size=128 only.");

  qweight = qweight.contiguous();
  scales = scales.contiguous();
  qzeros = qzeros.contiguous();
  const c10::cuda::CUDAGuard device_guard(qweight.device());
  const int64_t k = qweight.size(0);
  const int64_t n = qweight.size(1) * 8;
  TORCH_CHECK(k % kCtaK == 0 || k % kCtaK == kCtaK / 2,
              "K must be divisible by 512 for the SM70 M5 prototype.");
  TORCH_CHECK(k % group_size == 0 && n % kOutputColsPerCta == 0,
              "K/group-size or N/CTA alignment mismatch.");
  TORCH_CHECK(scales.size(0) == k / group_size && scales.size(1) == n,
              "scales shape mismatch.");
  TORCH_CHECK(qzeros.size(0) == scales.size(0) && qzeros.size(1) * 8 == n,
              "qzeros shape mismatch.");

  auto prepared_weight = torch::empty_like(qweight);
  auto zero_bias = torch::empty_like(scales);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  const int64_t weight_words = qweight.numel();
  prepare_weight_kernel<<<static_cast<int>((weight_words + threads - 1) /
                                           threads),
                          threads, 0, stream>>>(
      prepared_weight.data_ptr<int32_t>(), qweight.data_ptr<int32_t>(),
      static_cast<int>(k), static_cast<int>(n));
  const int64_t metadata_elements = scales.numel();
  prepare_zero_bias_kernel<<<static_cast<int>(
                                 (metadata_elements + threads - 1) / threads),
                             threads, 0, stream>>>(
      reinterpret_cast<__half*>(zero_bias.data_ptr<at::Half>()),
      qzeros.data_ptr<int32_t>(),
      reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
      static_cast<int>(scales.size(0)), static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(prepared_weight, scales, zero_bias);
}

void out(torch::Tensor output, torch::Tensor input,
         torch::Tensor prepared_weight, torch::Tensor scales,
         torch::Tensor zero_bias, int64_t group_size, int64_t accum_mode) {
  TORCH_CHECK(output.is_cuda() && input.is_cuda() &&
                  prepared_weight.is_cuda() && scales.is_cuda() &&
                  zero_bias.is_cuda(),
              "SM70 M5 AWQ GEMV expects CUDA tensors.");
  TORCH_CHECK(output.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  scales.scalar_type() == torch::kFloat16 &&
                  zero_bias.scalar_type() == torch::kFloat16 &&
                  prepared_weight.scalar_type() == torch::kInt32,
              "SM70 M5 AWQ GEMV dtype mismatch.");
  TORCH_CHECK(input.dim() == 2 && input.size(0) == kM,
              "SM70 batched GEMV requires exact M=5.");
  TORCH_CHECK(group_size == 128, "SM70 batched GEMV requires group_size=128.");
  TORCH_CHECK(accum_mode >= 0 && accum_mode <= 2,
              "SM70 batched GEMV accum_mode must be 0, 1, or 2.");
  const c10::cuda::CUDAGuard device_guard(input.device());
  const int64_t k = input.size(1);
  const int64_t n = scales.size(1);
  TORCH_CHECK(output.size(0) == kM && output.size(1) == n &&
                  output.is_contiguous() && input.is_contiguous(),
              "SM70 M5 AWQ GEMV input/output shape mismatch.");
  TORCH_CHECK(prepared_weight.numel() == k * n / 8 &&
                  scales.size(0) == k / group_size &&
                  zero_bias.sizes() == scales.sizes(),
              "SM70 M5 AWQ GEMV prepared tensor shape mismatch.");

  const dim3 grid(static_cast<unsigned>(n / kOutputColsPerCta));
  const dim3 block(kThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (accum_mode == 2) {
    awq_m5_batched_gemv_kernel<2><<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        prepared_weight.data_ptr<int32_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zero_bias.data_ptr<at::Half>()),
        static_cast<int>(n), static_cast<int>(k), static_cast<int>(group_size));
  } else if (accum_mode == 1) {
    awq_m5_batched_gemv_kernel<1><<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        prepared_weight.data_ptr<int32_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zero_bias.data_ptr<at::Half>()),
        static_cast<int>(n), static_cast<int>(k), static_cast<int>(group_size));
  } else {
    awq_m5_batched_gemv_kernel<0><<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        prepared_weight.data_ptr<int32_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zero_bias.data_ptr<at::Half>()),
        static_cast<int>(n), static_cast<int>(k), static_cast<int>(group_size));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY(sm70_awq_m5_micro, m) {
  m.def(
      "prepare(Tensor qweight, Tensor scales, Tensor qzeros, int group_size) "
      "-> (Tensor, Tensor, Tensor)");
  m.def(
      "out(Tensor(a!) output, Tensor input, Tensor prepared_weight, "
      "Tensor scales, Tensor zero_bias, int group_size, int accum_mode) "
      "-> ()");
}

TORCH_LIBRARY_IMPL(sm70_awq_m5_micro, CUDA, m) {
  m.impl("prepare", &prepare);
  m.impl("out", &out);
}
