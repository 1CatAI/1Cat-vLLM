// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ void dequant_e2m1x8(unsigned packed, half2 scale,
                                               half2 out[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x0e000e00u;
  unsigned values[4];
  values[0] = ((packed << 12) & kSign) | ((packed << 9) & kExponentMantissa);
  values[1] = ((packed << 8) & kSign) | ((packed << 5) & kExponentMantissa);
  values[2] = ((packed << 4) & kSign) | ((packed << 1) & kExponentMantissa);
  values[3] = (packed & kSign) | ((packed >> 3) & kExponentMantissa);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    out[index] =
        __hmul2(*reinterpret_cast<const half2*>(&values[index]), scale);
  }
}

#define VLLM_SM70_MMA_8N8K4(C, A0, A1, B0, B1)                      \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int kSplitK>
__global__ void mxfp4_qpn_m1_sm70_kernel(const half* __restrict__ input,
                                         const uint32_t* __restrict__ weights,
                                         const uint8_t* __restrict__ scales,
                                         const int32_t* __restrict__ expert_ids,
                                         half* __restrict__ output, int n,
                                         int k, bool broadcast_input) {
  __shared__ float partials[kSplitK][32];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int route = blockIdx.y;
  const int expert = __ldg(expert_ids + route);
  if (expert < 0 || expert >= 256) {
    if (threadIdx.x < 32) {
      output[static_cast<size_t>(route) * n + tile * 32 + threadIdx.x] =
          __float2half(0.0f);
    }
    return;
  }

  const int quadpair = (lane >> 2) & 3;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col =
      ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / kSplitK;
  const int group_begin = warp * groups_per_warp;
  const int groups_k8 = k >> 3;
  const int tiles_n32 = n >> 5;

  const size_t words_per_expert = static_cast<size_t>(k) * n / 8;
  const uint32_t* expert_weights =
      weights + static_cast<size_t>(expert) * words_per_expert;
  const size_t scales_per_expert = static_cast<size_t>(k >> 5) * n;
  const uint8_t* expert_scales =
      scales + static_cast<size_t>(expert) * scales_per_expert;
  const half* input_row =
      input + static_cast<size_t>(broadcast_input ? 0 : route) * k;

  float accum[8] = {};
  const half2 exponent_rebias = __float2half2_rn(16384.0f);
#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const size_t tile_group_base =
        (static_cast<size_t>(tile) * groups_k8 + group * 2) * 32 + packed_col;
    const unsigned packed0 = __ldcs(expert_weights + tile_group_base);
    const unsigned packed1 = __ldcs(expert_weights + tile_group_base + 32);
    const size_t scale_index =
        (static_cast<size_t>(group >> 1) * tiles_n32 + tile) * 32 + packed_col;
    const uint8_t adjusted_exponent = __ldg(expert_scales + scale_index);
    const half scalar =
        __ushort_as_half(static_cast<unsigned short>(adjusted_exponent) << 10);
    const half2 scale =
        __hmul2(__halves2half2(scalar, scalar), exponent_rebias);

    half2 decoded[8];
    dequant_e2m1x8(packed0, scale, decoded);
    dequant_e2m1x8(packed1, scale, decoded + 4);
    const unsigned* b = reinterpret_cast<const unsigned*>(decoded);

    uint4 input01 = make_uint4(0, 0, 0, 0);
    uint4 input23 = make_uint4(0, 0, 0, 0);
    if (a_row == 0) {
      input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
      input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
    }
    const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
    VLLM_SM70_MMA_8N8K4(accum, a0[0], a0[1], b[0], b[1]);
    VLLM_SM70_MMA_8N8K4(accum, a0[2], a0[3], b[2], b[3]);
    VLLM_SM70_MMA_8N8K4(accum, a1[0], a1[1], b[4], b[5]);
    VLLM_SM70_MMA_8N8K4(accum, a1[2], a1[3], b[6], b[7]);
  }

  // Only eight lane roles own M=1's 32 output columns.
  if ((lane & 17) == 0) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
      for (int offset = 0; offset < 2; ++offset) {
        const int index = pair * 4 + offset;
        const int local_col = offset | (((lane >> 1) & 1) << 1) | (pair << 2);
        partials[warp][quadpair * 8 + local_col] = accum[index];
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < kSplitK; ++k_warp) {
      value += partials[k_warp][lane];
    }
    output[static_cast<size_t>(route) * n + tile * 32 + lane] =
        __float2half(value);
  }
}

template <int kSplitK>
__global__ void nvfp4_qpn_m1_sm70_kernel(const half* __restrict__ input,
                                         const uint32_t* __restrict__ weights,
                                         const half* __restrict__ scales,
                                         const int32_t* __restrict__ expert_ids,
                                         half* __restrict__ output, int n,
                                         int k, bool broadcast_input) {
  __shared__ float partials[kSplitK][32];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int route = blockIdx.y;
  const int expert = __ldg(expert_ids + route);
  if (expert < 0 || expert >= 512) {
    if (threadIdx.x < 32) {
      output[static_cast<size_t>(route) * n + tile * 32 + threadIdx.x] =
          __float2half(0.0f);
    }
    return;
  }

  const int quadpair = (lane >> 2) & 3;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col =
      ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / kSplitK;
  const int group_begin = warp * groups_per_warp;
  const int groups_k8 = k >> 3;
  const int tiles_n32 = n >> 5;

  const size_t words_per_expert = static_cast<size_t>(k) * n / 8;
  const uint32_t* expert_weights =
      weights + static_cast<size_t>(expert) * words_per_expert;
  const size_t scales_per_expert = static_cast<size_t>(k >> 4) * n;
  const half* expert_scales =
      scales + static_cast<size_t>(expert) * scales_per_expert;
  const half* input_row =
      input + static_cast<size_t>(broadcast_input ? 0 : route) * k;

  float accum[8] = {};
#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const size_t tile_group_base =
        (static_cast<size_t>(tile) * groups_k8 + group * 2) * 32 + packed_col;
    const unsigned packed0 = __ldcs(expert_weights + tile_group_base);
    const unsigned packed1 = __ldcs(expert_weights + tile_group_base + 32);
    const size_t scale_index =
        (static_cast<size_t>(group) * tiles_n32 + tile) * 32 + packed_col;
    const half scalar = __ldg(expert_scales + scale_index);
    // dequant_e2m1x8 materializes the FP4 payload with a 2^-14 exponent
    // offset so that every E2M1 value can be formed with integer bit ops.
    // MXFP4 folds the matching 2^14 correction into its exponent scale;
    // NVFP4's prepared FP16 group scale needs the same correction here.
    const half2 scale =
        __hmul2(__halves2half2(scalar, scalar), __float2half2_rn(16384.0f));

    half2 decoded[8];
    dequant_e2m1x8(packed0, scale, decoded);
    dequant_e2m1x8(packed1, scale, decoded + 4);
    const unsigned* b = reinterpret_cast<const unsigned*>(decoded);

    uint4 input01 = make_uint4(0, 0, 0, 0);
    uint4 input23 = make_uint4(0, 0, 0, 0);
    if (a_row == 0) {
      input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
      input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
    }
    const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
    VLLM_SM70_MMA_8N8K4(accum, a0[0], a0[1], b[0], b[1]);
    VLLM_SM70_MMA_8N8K4(accum, a0[2], a0[3], b[2], b[3]);
    VLLM_SM70_MMA_8N8K4(accum, a1[0], a1[1], b[4], b[5]);
    VLLM_SM70_MMA_8N8K4(accum, a1[2], a1[3], b[6], b[7]);
  }

  if ((lane & 17) == 0) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
      for (int offset = 0; offset < 2; ++offset) {
        const int index = pair * 4 + offset;
        const int local_col = offset | (((lane >> 1) & 1) << 1) | (pair << 2);
        partials[warp][quadpair * 8 + local_col] = accum[index];
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < kSplitK; ++k_warp) {
      value += partials[k_warp][lane];
    }
    output[static_cast<size_t>(route) * n + tile * 32 + lane] =
        __float2half(value);
  }
}

template <int kSplitK>
void launch_mxfp4_qpn_m1(torch::Tensor out, torch::Tensor input,
                         torch::Tensor weights, torch::Tensor scales,
                         torch::Tensor expert_ids, bool broadcast_input) {
  const int n = static_cast<int>(out.size(1));
  const int k = static_cast<int>(input.size(1));
  mxfp4_qpn_m1_sm70_kernel<kSplitK>
      <<<dim3(n / 32, 6), 32 * kSplitK, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
          scales.data_ptr<uint8_t>(), expert_ids.data_ptr<int32_t>(),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()), n, k,
          broadcast_input);
}

template <int kSplitK>
void launch_nvfp4_qpn_m1(torch::Tensor out, torch::Tensor input,
                         torch::Tensor weights, torch::Tensor scales,
                         torch::Tensor expert_ids, bool broadcast_input) {
  const int n = static_cast<int>(out.size(1));
  const int k = static_cast<int>(input.size(1));
  nvfp4_qpn_m1_sm70_kernel<kSplitK>
      <<<dim3(n / 32, 10), 32 * kSplitK, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          expert_ids.data_ptr<int32_t>(),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()), n, k,
          broadcast_input);
}

void dispatch_nvfp4_qpn_m1(torch::Tensor out, torch::Tensor input,
                           torch::Tensor weights, torch::Tensor scales,
                           torch::Tensor expert_ids, bool broadcast_input,
                           int64_t split_k) {
#define VLLM_NVFP4_QPN_M1_CASE(SPLIT)                                   \
  case SPLIT:                                                           \
    launch_nvfp4_qpn_m1<SPLIT>(out, input, weights, scales, expert_ids, \
                               broadcast_input);                        \
    break
  switch (split_k) {
    VLLM_NVFP4_QPN_M1_CASE(1);
    VLLM_NVFP4_QPN_M1_CASE(2);
    VLLM_NVFP4_QPN_M1_CASE(4);
    VLLM_NVFP4_QPN_M1_CASE(5);
    VLLM_NVFP4_QPN_M1_CASE(8);
    VLLM_NVFP4_QPN_M1_CASE(10);
    VLLM_NVFP4_QPN_M1_CASE(16);
    VLLM_NVFP4_QPN_M1_CASE(20);
    VLLM_NVFP4_QPN_M1_CASE(32);
    default:
      TORCH_CHECK(false, "nvfp4_moe_qpn_m1_sm70_out: unsupported split_k ",
                  split_k);
  }
#undef VLLM_NVFP4_QPN_M1_CASE
}

}  // namespace

void mxfp4_moe_qpn_m1_sm70_out(torch::Tensor out, torch::Tensor input,
                               torch::Tensor weights, torch::Tensor scales,
                               torch::Tensor expert_ids, bool broadcast_input) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && weights.is_cuda() &&
                  scales.is_cuda() && expert_ids.is_cuda(),
              "mxfp4_moe_qpn_m1_sm70_out: tensors must be CUDA tensors");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weights.scalar_type() == torch::kInt32 &&
                  scales.scalar_type() == torch::kUInt8 &&
                  expert_ids.scalar_type() == torch::kInt32,
              "mxfp4_moe_qpn_m1_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  weights.is_contiguous() && scales.is_contiguous() &&
                  expert_ids.is_contiguous(),
              "mxfp4_moe_qpn_m1_sm70_out: tensors must be contiguous");
  TORCH_CHECK(out.dim() == 2 && input.dim() == 2 && weights.dim() == 3 &&
                  scales.dim() == 3 && expert_ids.dim() == 1,
              "mxfp4_moe_qpn_m1_sm70_out: rank mismatch");
  TORCH_CHECK(out.size(0) == 6 && expert_ids.numel() == 6 &&
                  weights.size(0) == 256 && scales.size(0) == 256,
              "mxfp4_moe_qpn_m1_sm70_out: expected six routes and 256 experts");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == weights.get_device() &&
                  input.get_device() == scales.get_device() &&
                  input.get_device() == expert_ids.get_device(),
              "mxfp4_moe_qpn_m1_sm70_out: device mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  if (broadcast_input) {
    TORCH_CHECK(input.sizes() == torch::IntArrayRef({1, 4096}) &&
                    out.sizes() == torch::IntArrayRef({6, 1024}) &&
                    weights.sizes() == torch::IntArrayRef({256, 4096, 128}) &&
                    scales.sizes() == torch::IntArrayRef({256, 128, 1024}),
                "mxfp4_moe_qpn_m1_sm70_out: W13 tensor contract mismatch");
    launch_mxfp4_qpn_m1<16>(out, input, weights, scales, expert_ids, true);
  } else {
    TORCH_CHECK(input.sizes() == torch::IntArrayRef({6, 512}) &&
                    out.sizes() == torch::IntArrayRef({6, 4096}) &&
                    weights.sizes() == torch::IntArrayRef({256, 512, 512}) &&
                    scales.sizes() == torch::IntArrayRef({256, 16, 4096}),
                "mxfp4_moe_qpn_m1_sm70_out: W2 tensor contract mismatch");
    launch_mxfp4_qpn_m1<8>(out, input, weights, scales, expert_ids, false);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_moe_qpn_m1_sm70_out(torch::Tensor out, torch::Tensor input,
                               torch::Tensor weights, torch::Tensor scales,
                               torch::Tensor expert_ids, bool broadcast_input,
                               int64_t split_k) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && weights.is_cuda() &&
                  scales.is_cuda() && expert_ids.is_cuda(),
              "nvfp4_moe_qpn_m1_sm70_out: tensors must be CUDA tensors");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weights.scalar_type() == torch::kInt32 &&
                  scales.scalar_type() == torch::kFloat16 &&
                  expert_ids.scalar_type() == torch::kInt32,
              "nvfp4_moe_qpn_m1_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  weights.is_contiguous() && scales.is_contiguous() &&
                  expert_ids.is_contiguous(),
              "nvfp4_moe_qpn_m1_sm70_out: tensors must be contiguous");
  TORCH_CHECK(out.dim() == 2 && input.dim() == 2 && weights.dim() == 3 &&
                  scales.dim() == 3 && expert_ids.dim() == 1,
              "nvfp4_moe_qpn_m1_sm70_out: rank mismatch");
  TORCH_CHECK(out.size(0) == 10 && expert_ids.numel() == 10 &&
                  weights.size(0) == 512 && scales.size(0) == 512,
              "nvfp4_moe_qpn_m1_sm70_out: expected ten routes and 512 experts");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == weights.get_device() &&
                  input.get_device() == scales.get_device() &&
                  input.get_device() == expert_ids.get_device(),
              "nvfp4_moe_qpn_m1_sm70_out: device mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const int64_t groups_k16 = input.size(1) / 16;
  TORCH_CHECK(split_k > 0 && split_k <= 32 && groups_k16 % split_k == 0,
              "nvfp4_moe_qpn_m1_sm70_out: split_k must divide K/16");
  if (broadcast_input) {
    TORCH_CHECK(input.sizes() == torch::IntArrayRef({1, 2560}) &&
                    out.sizes() == torch::IntArrayRef({10, 320}) &&
                    weights.sizes() == torch::IntArrayRef({512, 2560, 40}) &&
                    scales.sizes() == torch::IntArrayRef({512, 160, 320}),
                "nvfp4_moe_qpn_m1_sm70_out: W13 tensor contract mismatch");
  } else {
    TORCH_CHECK(input.sizes() == torch::IntArrayRef({10, 160}) &&
                    out.sizes() == torch::IntArrayRef({10, 2560}) &&
                    weights.sizes() == torch::IntArrayRef({512, 160, 320}) &&
                    scales.sizes() == torch::IntArrayRef({512, 10, 2560}),
                "nvfp4_moe_qpn_m1_sm70_out: W2 tensor contract mismatch");
  }
  dispatch_nvfp4_qpn_m1(out, input, weights, scales, expert_ids,
                        broadcast_input, split_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
