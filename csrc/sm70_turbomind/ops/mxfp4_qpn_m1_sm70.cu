// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project
//
// The MXFP4 M=1 quadpair-N m8n8k4 layout is derived from dnv2003/v100-skinny
// (MIT). See LICENSE.v100-skinny in this directory for the retained MIT notice.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQwen38SharedGateHidden = 2560;
constexpr int kQwen38SharedGateThreads = 256;

__device__ __forceinline__ float qwen38_shared_gate_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__global__ void qwen38_shared_gate_exact_kernel(
    half* __restrict__ output, const half* __restrict__ input,
    const half* __restrict__ weight) {
  constexpr int kValuesPerThread =
      kQwen38SharedGateHidden / kQwen38SharedGateThreads;
  const int tid = threadIdx.x;
  float value = 0.0f;
#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    const int index = tid + item * kQwen38SharedGateThreads;
    value = __fmaf_rn(__half2float(input[index]), __half2float(weight[index]),
                      value);
  }
  value = qwen38_shared_gate_warp_sum(value);

  __shared__ float warp_partials[kQwen38SharedGateThreads / 32];
  __shared__ half shared_gate;
  if ((tid & 31) == 0) {
    warp_partials[tid >> 5] = value;
  }
  __syncthreads();
  if (tid < 32) {
    value = tid < kQwen38SharedGateThreads / 32 ? warp_partials[tid] : 0.0f;
    value = qwen38_shared_gate_warp_sum(value);
    if (tid == 0) {
      // Preserve the eager FP16 linear and sigmoid materialization points.
      const half linear = __float2half_rn(value);
      const float rounded_linear = __half2float(linear);
      shared_gate = __float2half_rn(1.0f / (1.0f + __expf(-rounded_linear)));
    }
  }
  __syncthreads();

  const half2 gate = __half2half2(shared_gate);
  auto* output2 = reinterpret_cast<half2*>(output);
  for (int index = tid; index < kQwen38SharedGateHidden / 2;
       index += blockDim.x) {
    output2[index] = __hmul2(output2[index], gate);
  }
}

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

template <int kSplitK, bool kFusedSwiGLU = false>
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
    if constexpr (kFusedSwiGLU) {
      if (threadIdx.x < 16) {
        output[static_cast<size_t>(route) * (n / 2) + tile * 16 + threadIdx.x] =
            __float2half(0.0f);
      }
    } else if (threadIdx.x < 32) {
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
  // Qwen3.8 routes ten experts per token. The broadcast form accepts either
  // B1's single input row or MTP4's five verifier rows without materializing
  // the 10x routed-input expansion.
  const int input_route = broadcast_input ? route / 10 : route;
  const half* input_row = input + static_cast<size_t>(input_route) * k;

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
    const half rounded = __float2half(value);
    if constexpr (kFusedSwiGLU) {
      const int source_lane = (lane & 15) * 2;
      const unsigned rounded_bits = __half_as_ushort(rounded);
      const half gate = __ushort_as_half(static_cast<unsigned short>(
          __shfl_sync(0xffffffffu, rounded_bits, source_lane)));
      const half up = __ushort_as_half(static_cast<unsigned short>(
          __shfl_sync(0xffffffffu, rounded_bits, source_lane + 1)));
      if (lane < 16) {
        const float gate_f = __half2float(gate);
        const half silu = __float2half(gate_f / (1.0f + expf(-gate_f)));
        output[static_cast<size_t>(route) * (n / 2) + tile * 16 + lane] =
            __hmul(silu, up);
      }
    } else {
      output[static_cast<size_t>(route) * n + tile * 32 + lane] = rounded;
    }
  }
}

// Qwen3.8 TP4 has ten K160 -> N2560 W2 routes. Keeping one route per warp
// retains split-K=1 accumulation, while grouping all routes for one N32 tile
// lets the CTA reduce them directly. Each route is rounded through FP16 before
// weighting, matching the former W2-output plus Triton-reduce path bit for bit.
__global__ void nvfp4_qwen38_w2_direct_reduce_kernel(
    const half* __restrict__ input, const uint32_t* __restrict__ weights,
    const half* __restrict__ scales, const int32_t* __restrict__ expert_ids,
    const float* __restrict__ topk_weights, half* __restrict__ output) {
  constexpr int kRoutes = 10;
  constexpr int kK = 160;
  constexpr int kN = 2560;
  constexpr int kExperts = 512;
  __shared__ half route_outputs[kRoutes][32];

  const int lane = threadIdx.x & 31;
  const int route = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int expert = __ldg(expert_ids + route);
  float accum[8] = {};

  if (expert >= 0 && expert < kExperts) {
    const int quadpair = (lane >> 2) & 3;
    const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int packed_col =
        ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
    constexpr int kGroupsK16 = kK >> 4;
    constexpr int kGroupsK8 = kK >> 3;
    constexpr int kTilesN32 = kN >> 5;
    constexpr size_t kWordsPerExpert = static_cast<size_t>(kK) * kN / 8;
    constexpr size_t kScalesPerExpert = static_cast<size_t>(kK >> 4) * kN;
    const uint32_t* expert_weights =
        weights + static_cast<size_t>(expert) * kWordsPerExpert;
    const half* expert_scales =
        scales + static_cast<size_t>(expert) * kScalesPerExpert;
    const half* input_row = input + static_cast<size_t>(route) * kK;

#pragma unroll
    for (int group = 0; group < kGroupsK16; ++group) {
      const size_t tile_group_base =
          (static_cast<size_t>(tile) * kGroupsK8 + group * 2) * 32 + packed_col;
      const unsigned packed0 = __ldcs(expert_weights + tile_group_base);
      const unsigned packed1 = __ldcs(expert_weights + tile_group_base + 32);
      const size_t scale_index =
          (static_cast<size_t>(group) * kTilesN32 + tile) * 32 + packed_col;
      const half scalar = __ldg(expert_scales + scale_index);
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
          route_outputs[route][quadpair * 8 + local_col] =
              __float2half(accum[index]);
        }
      }
    }
  } else if (lane < 4) {
#pragma unroll
    for (int offset = 0; offset < 8; ++offset) {
      route_outputs[route][lane * 8 + offset] = __float2half(0.0f);
    }
  }
  __syncthreads();

  if (route == 0) {
    float weighted = 0.0f;
#pragma unroll
    for (int selected = 0; selected < kRoutes; ++selected) {
      weighted = fmaf(__ldg(topk_weights + selected),
                      __half2float(route_outputs[selected][lane]), weighted);
    }
    output[tile * 32 + lane] = __float2half(weighted);
  }
}

__device__ __forceinline__ half nvfp4_raw_scale_to_half(uint8_t scale_code,
                                                        half scale_hi,
                                                        half scale_lo) {
  // This fast reconstruction is selected only after a load-time exhaustive
  // comparison with the checkpoint's prepared FP16 scales. The high/low split
  // avoids an FP32 multiply in every QPN inner-loop iteration.
  const unsigned half_bits =
      (static_cast<unsigned>(scale_code) << 7U) + 0x2000U;
  const half raw_scale = *reinterpret_cast<const half*>(&half_bits);
  const half correction = __hmul(raw_scale, scale_lo);
  return __hfma(raw_scale, scale_hi, correction);
}

__global__ void nvfp4_expand_raw_scales_sm70_kernel(
    half* __restrict__ output, const uint8_t* __restrict__ scale_codes,
    const float* __restrict__ global_scales, size_t elements,
    size_t scales_per_expert, int n, int global_stride, bool interleaved_w13,
    bool fast_decode_rounding) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const int expert = static_cast<int>(index / scales_per_expert);
  const int col = static_cast<int>(index % n);
  int global_slot = 0;
  if (global_stride == 2) {
    global_slot = interleaved_w13 ? ((col >> 5) & 1) : (col >= n / 2);
  }
  const float global_scale =
      global_scales[static_cast<size_t>(expert) * global_stride + global_slot];
  if (fast_decode_rounding) {
    const float scaled_global = global_scale * 16384.0f;
    const half scale_hi = __float2half_rn(scaled_global);
    const half scale_lo =
        __float2half_rn(scaled_global - __half2float(scale_hi));
    // Validate the effective scale consumed by HMMA directly. Dividing back
    // to the stored scale could hide a mismatch through FP16 underflow.
    output[index] =
        nvfp4_raw_scale_to_half(scale_codes[index], scale_hi, scale_lo);
  } else {
    const unsigned code = scale_codes[index];
    const unsigned magnitude = code & 0x7fU;
    unsigned half_bits =
        magnitude == 0x7fU ? 0x7e00U : (magnitude << 7U) + 0x2000U;
    if (magnitude < 8U) {
      const half subnormal = __float2half_rn(magnitude * 0x1p-9f);
      half_bits = __half_as_ushort(subnormal);
    }
    half_bits |= (code & 0x80U) << 8U;
    const half raw_scale = *reinterpret_cast<const half*>(&half_bits);
    output[index] = __float2half_rn(__half2float(raw_scale) * global_scale);
  }
}

template <int kSplitK>
__global__ void nvfp4_qpn_raw_scale_sm70_kernel(
    const half* __restrict__ input, const uint32_t* __restrict__ weights,
    const uint8_t* __restrict__ scale_codes,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ expert_ids, half* __restrict__ output, int n,
    int k, bool broadcast_input, bool interleaved_w13) {
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
  const bool is_w13 = k == 2560 && n == 320;

  const size_t words_per_expert = static_cast<size_t>(k) * n / 8;
  const uint32_t* expert_weights =
      weights + static_cast<size_t>(expert) * words_per_expert;
  const size_t scales_per_expert = static_cast<size_t>(k >> 4) * n;
  const uint8_t* expert_scale_codes =
      scale_codes + static_cast<size_t>(expert) * scales_per_expert;
  const int input_route = broadcast_input ? route / 10 : route;
  const half* input_row = input + static_cast<size_t>(input_route) * k;

  int global_slot = 0;
  if (is_w13) {
    global_slot = interleaved_w13 ? (tile & 1) : (tile >= tiles_n32 / 2);
  }
  const float global_scale =
      __ldg(global_scales + static_cast<size_t>(expert) * (is_w13 ? 2 : 1) +
            global_slot) *
      16384.0f;
  const half scale_hi = __float2half_rn(global_scale);
  const half scale_lo = __float2half_rn(global_scale - __half2float(scale_hi));

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
    const uint8_t scale_code = __ldg(expert_scale_codes + scale_index);
    const half scalar = nvfp4_raw_scale_to_half(scale_code, scale_hi, scale_lo);
    const half2 scale = __halves2half2(scalar, scalar);
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

template <int kSplitK, bool kInterleaved, bool kRawScale = false>
__global__ void nvfp4_qpn_w13_swiglu_batch_sm70_kernel(
    const half* __restrict__ input, const uint32_t* __restrict__ weights,
    const void* __restrict__ scales, const float* __restrict__ global_scales,
    const int32_t* __restrict__ expert_ids, half* __restrict__ output) {
  constexpr int kInput = 2560;
  constexpr int kOutput = 320;
  constexpr int kIntermediate = 160;
  constexpr int kTilesN32 = kOutput / 32;
  constexpr int kGroupsK16 = kInput / 16;
  __shared__ float partials[2][kSplitK][32];
  __shared__ half projected[2][32];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int projection = warp / kSplitK;
  const int split = warp % kSplitK;
  const int route = blockIdx.y;
  const int tile = kInterleaved
                       ? blockIdx.x * 2 + projection
                       : blockIdx.x + projection * (kIntermediate / 32);
  const int expert = __ldg(expert_ids + route);
  const int quadpair = (lane >> 2) & 3;
  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col =
      ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  constexpr int kGroupsPerWarp = kGroupsK16 / kSplitK;
  const int group_begin = split * kGroupsPerWarp;

  float accum[8] = {};
  if (expert >= 0 && expert < 512) {
    const size_t words_per_expert = static_cast<size_t>(kInput) * kOutput / 8;
    const uint32_t* expert_weights =
        weights + static_cast<size_t>(expert) * words_per_expert;
    const size_t scales_per_expert = static_cast<size_t>(kInput / 16) * kOutput;
    const half* expert_scales = reinterpret_cast<const half*>(scales) +
                                static_cast<size_t>(expert) * scales_per_expert;
    const uint8_t* expert_scale_codes =
        reinterpret_cast<const uint8_t*>(scales) +
        static_cast<size_t>(expert) * scales_per_expert;
    const half* input_row = input + static_cast<size_t>(route / 10) * kInput;
    half scale_hi = __float2half(0.0f);
    half scale_lo = __float2half(0.0f);
    if constexpr (kRawScale) {
      const float global_scale =
          __ldg(global_scales + static_cast<size_t>(expert) * 2 + projection) *
          16384.0f;
      scale_hi = __float2half_rn(global_scale);
      scale_lo = __float2half_rn(global_scale - __half2float(scale_hi));
    }

#pragma unroll 4
    for (int group = group_begin; group < group_begin + kGroupsPerWarp;
         ++group) {
      const size_t tile_group_base =
          (static_cast<size_t>(tile) * (kInput / 8) + group * 2) * 32 +
          packed_col;
      const unsigned packed0 = __ldcs(expert_weights + tile_group_base);
      const unsigned packed1 = __ldcs(expert_weights + tile_group_base + 32);
      const size_t scale_index =
          (static_cast<size_t>(group) * kTilesN32 + tile) * 32 + packed_col;
      half scalar;
      if constexpr (kRawScale) {
        scalar = nvfp4_raw_scale_to_half(
            __ldg(expert_scale_codes + scale_index), scale_hi, scale_lo);
      } else {
        const half prepared_scale = __ldg(expert_scales + scale_index);
        scalar = __hmul(prepared_scale, __float2half_rn(16384.0f));
      }
      const half2 scale = __halves2half2(scalar, scalar);
      half2 decoded[8];
      dequant_e2m1x8(packed0, scale, decoded);
      dequant_e2m1x8(packed1, scale, decoded + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(decoded);

      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      if (mma_row == 0) {
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
  }

  if ((lane & 17) == 0) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
      for (int offset = 0; offset < 2; ++offset) {
        const int index = pair * 4 + offset;
        const int output_col = offset | (((lane >> 1) & 1) << 1) | (pair << 2);
        if constexpr (kSplitK == 1) {
          projected[projection][quadpair * 8 + output_col] =
              __float2half(accum[index]);
        } else {
          partials[projection][split][quadpair * 8 + output_col] = accum[index];
        }
      }
    }
  }
  __syncthreads();

  if constexpr (kSplitK > 1) {
    if (warp < 2) {
      float value = 0.0f;
#pragma unroll
      for (int k_warp = 0; k_warp < kSplitK; ++k_warp) {
        value += partials[warp][k_warp][lane];
      }
      projected[warp][lane] = __float2half(value);
    }
    __syncthreads();
  }

  if (warp == 0) {
    const int projection_index = kInterleaved ? lane / 16 : 0;
    const int projection_col = kInterleaved ? (lane % 16) * 2 : lane;
    const half gate = projected[projection_index][projection_col];
    const half up = kInterleaved
                        ? projected[projection_index][projection_col + 1]
                        : projected[1][lane];
    const float gate_f = __half2float(gate);
    const half activated = __float2half(gate_f / (1.0f + expf(-gate_f)));
    output[static_cast<size_t>(route) * kIntermediate + blockIdx.x * 32 +
           lane] = __hmul(activated, up);
  }
}

template <bool kRawScale = false>
__global__ void nvfp4_qpn_w2_reduce_sm70_kernel(
    const half* __restrict__ input, const uint32_t* __restrict__ weights,
    const void* __restrict__ scales, const float* __restrict__ global_scales,
    const int32_t* __restrict__ expert_ids,
    const float* __restrict__ topk_weights, half* __restrict__ output,
    int tokens) {
  constexpr int kTopK = 10;
  constexpr int kInput = 160;
  constexpr int kOutput = 2560;
  constexpr int kTilesN32 = kOutput / 32;
  constexpr int kGroupsK16 = kInput / 16;
  __shared__ half routed_output[kTopK][32];

  const int lane = threadIdx.x & 31;
  const int slot = threadIdx.x >> 5;
  const int token = blockIdx.y;
  const int tile = blockIdx.x;
  const int route = token * kTopK + slot;
  const int expert = __ldg(expert_ids + route);
  const int quadpair = (lane >> 2) & 3;
  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col =
      ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);

  float accum[8] = {};
  if (token < tokens && expert >= 0 && expert < 512) {
    const size_t words_per_expert = static_cast<size_t>(kInput) * kOutput / 8;
    const uint32_t* expert_weights =
        weights + static_cast<size_t>(expert) * words_per_expert;
    const size_t scales_per_expert = static_cast<size_t>(kInput / 16) * kOutput;
    const half* expert_scales = reinterpret_cast<const half*>(scales) +
                                static_cast<size_t>(expert) * scales_per_expert;
    const uint8_t* expert_scale_codes =
        reinterpret_cast<const uint8_t*>(scales) +
        static_cast<size_t>(expert) * scales_per_expert;
    const half* input_row = input + static_cast<size_t>(route) * kInput;
    half scale_hi = __float2half(0.0f);
    half scale_lo = __float2half(0.0f);
    if constexpr (kRawScale) {
      const float global_scale = __ldg(global_scales + expert) * 16384.0f;
      scale_hi = __float2half_rn(global_scale);
      scale_lo = __float2half_rn(global_scale - __half2float(scale_hi));
    }

#pragma unroll
    for (int group = 0; group < kGroupsK16; ++group) {
      const size_t tile_group_base =
          (static_cast<size_t>(tile) * (kInput / 8) + group * 2) * 32 +
          packed_col;
      const unsigned packed0 = __ldcs(expert_weights + tile_group_base);
      const unsigned packed1 = __ldcs(expert_weights + tile_group_base + 32);
      const size_t scale_index =
          (static_cast<size_t>(group) * kTilesN32 + tile) * 32 + packed_col;
      half scalar;
      if constexpr (kRawScale) {
        scalar = nvfp4_raw_scale_to_half(
            __ldg(expert_scale_codes + scale_index), scale_hi, scale_lo);
      } else {
        const half prepared_scale = __ldg(expert_scales + scale_index);
        scalar = __hmul(prepared_scale, __float2half_rn(16384.0f));
      }
      const half2 scale = __halves2half2(scalar, scalar);
      half2 decoded[8];
      dequant_e2m1x8(packed0, scale, decoded);
      dequant_e2m1x8(packed1, scale, decoded + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(decoded);

      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      if (mma_row == 0) {
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
  }

  if ((lane & 17) == 0) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
      for (int offset = 0; offset < 2; ++offset) {
        const int index = pair * 4 + offset;
        const int output_col = offset | (((lane >> 1) & 1) << 1) | (pair << 2);
        routed_output[slot][quadpair * 8 + output_col] =
            __float2half(accum[index]);
      }
    }
  }
  __syncthreads();

  if (slot == 0 && token < tokens) {
    float weighted = 0.0f;
#pragma unroll
    for (int routed_slot = 0; routed_slot < kTopK; ++routed_slot) {
      weighted =
          fmaf(__half2float(routed_output[routed_slot][lane]),
               __ldg(topk_weights + token * kTopK + routed_slot), weighted);
    }
    output[static_cast<size_t>(token) * kOutput + tile * 32 + lane] =
        __float2half(weighted);
  }
}

template <bool kW13>
__global__ void nvfp4_glm53_moe_q8_qpn_sm70_kernel(
    const half* __restrict__ input, const uint32_t* __restrict__ weights,
    const half* __restrict__ scales, const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ sorted_row_idx, half* __restrict__ output) {
  constexpr int kSplits = kW13 ? 3 : 1;
  constexpr int kN = kW13 ? 512 : 4096;
  constexpr int kK = kW13 ? 4096 : 256;
  constexpr int kGroupsK16 = kK / 16;
  constexpr int kGroupsK8 = kK / 8;
  constexpr int kTilesN32 = kN / 32;
  __shared__ float partials[kSplits][32];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int route = blockIdx.y;
  const int source_slot = __ldg(sorted_row_idx + route);
  const int expert = __ldg(expert_ids + source_slot);
  if (expert < 0 || expert >= 288) {
    if (threadIdx.x < 32) {
      output[static_cast<size_t>(route) * kN + tile * 32 + threadIdx.x] =
          __float2half(0.0f);
    }
    return;
  }

  const int quadpair = (lane >> 2) & 3;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col =
      ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  int group_begin = 0;
  int group_end = kGroupsK16;
  if constexpr (kW13) {
    // TurboMind's CTA-K32 split-3 scheduler assigns 42/43/43 chunks.
    group_begin = warp == 0 ? 0 : (warp == 1 ? 84 : 170);
    group_end = warp == 0 ? 84 : (warp == 1 ? 170 : 256);
  }

  const size_t words_per_expert = static_cast<size_t>(kK) * kN / 8;
  const uint32_t* expert_weights =
      weights + static_cast<size_t>(expert) * words_per_expert;
  const size_t scales_per_expert = static_cast<size_t>(kK >> 4) * kN;
  const half* expert_scales =
      scales + static_cast<size_t>(expert) * scales_per_expert;
  const half* input_row = input + static_cast<size_t>(route) * kK;

  float accum[8] = {};
#pragma unroll 4
  for (int group = group_begin; group < group_end; ++group) {
    const size_t tile_group_base =
        (static_cast<size_t>(tile) * kGroupsK8 + group * 2) * 32 + packed_col;
    const unsigned packed0 = __ldcs(expert_weights + tile_group_base);
    const unsigned packed1 = __ldcs(expert_weights + tile_group_base + 32);
    const size_t scale_index =
        (static_cast<size_t>(group) * kTilesN32 + tile) * 32 + packed_col;
    const half scalar = __ldg(expert_scales + scale_index);
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
    float value = partials[0][lane];
    if constexpr (kW13) {
      // TurboMind serial split-K executes split1+split0, then split2+prior.
      value = __fadd_rn(partials[2][lane],
                        __fadd_rn(partials[1][lane], partials[0][lane]));
    }
    output[static_cast<size_t>(route) * kN + tile * 32 + lane] =
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
  const int routes = static_cast<int>(expert_ids.numel());
  nvfp4_qpn_m1_sm70_kernel<kSplitK><<<dim3(n / 32, routes), 32 * kSplitK, 0,
                                      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      expert_ids.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), n, k, broadcast_input);
}

template <int kSplitK>
void launch_nvfp4_qpn_raw_scale(torch::Tensor out, torch::Tensor input,
                                torch::Tensor weights,
                                torch::Tensor scale_codes,
                                torch::Tensor global_scales,
                                torch::Tensor expert_ids, bool broadcast_input,
                                bool interleaved_w13) {
  const int n = static_cast<int>(out.size(1));
  const int k = static_cast<int>(input.size(1));
  const int routes = static_cast<int>(expert_ids.numel());
  nvfp4_qpn_raw_scale_sm70_kernel<kSplitK>
      <<<dim3(n / 32, routes), 32 * kSplitK, 0,
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
          scale_codes.data_ptr<uint8_t>(), global_scales.data_ptr<float>(),
          expert_ids.data_ptr<int32_t>(),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()), n, k,
          broadcast_input, interleaved_w13);
}

void launch_nvfp4_qwen38_w13_fused_swiglu(torch::Tensor out,
                                          torch::Tensor input,
                                          torch::Tensor weights,
                                          torch::Tensor scales,
                                          torch::Tensor expert_ids) {
  constexpr int kN = 320;
  constexpr int kK = 2560;
  constexpr int kSplitK = 16;
  const int routes = static_cast<int>(expert_ids.numel());
  nvfp4_qpn_m1_sm70_kernel<kSplitK, true>
      <<<dim3(kN / 32, routes), 32 * kSplitK, 0,
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          expert_ids.data_ptr<int32_t>(),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()), kN, kK, true);
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

void dispatch_nvfp4_qpn_raw_scale(torch::Tensor out, torch::Tensor input,
                                  torch::Tensor weights,
                                  torch::Tensor scale_codes,
                                  torch::Tensor global_scales,
                                  torch::Tensor expert_ids,
                                  bool broadcast_input, bool interleaved_w13,
                                  int64_t split_k) {
#define VLLM_NVFP4_QPN_RAW_CASE(SPLIT)                                   \
  case SPLIT:                                                            \
    launch_nvfp4_qpn_raw_scale<SPLIT>(out, input, weights, scale_codes,  \
                                      global_scales, expert_ids,         \
                                      broadcast_input, interleaved_w13); \
    break
  switch (split_k) {
    VLLM_NVFP4_QPN_RAW_CASE(1);
    VLLM_NVFP4_QPN_RAW_CASE(2);
    VLLM_NVFP4_QPN_RAW_CASE(4);
    VLLM_NVFP4_QPN_RAW_CASE(5);
    VLLM_NVFP4_QPN_RAW_CASE(8);
    VLLM_NVFP4_QPN_RAW_CASE(10);
    VLLM_NVFP4_QPN_RAW_CASE(16);
    VLLM_NVFP4_QPN_RAW_CASE(20);
    VLLM_NVFP4_QPN_RAW_CASE(32);
    default:
      TORCH_CHECK(false,
                  "nvfp4_moe_qpn_raw_scale_sm70_out: unsupported split_k ",
                  split_k);
  }
#undef VLLM_NVFP4_QPN_RAW_CASE
}

}  // namespace

void nvfp4_expand_raw_scales_sm70_out(torch::Tensor out,
                                      torch::Tensor scale_codes,
                                      torch::Tensor global_scales,
                                      bool interleaved_w13,
                                      bool fast_decode_rounding) {
  TORCH_CHECK(out.is_cuda() && scale_codes.is_cuda() && global_scales.is_cuda(),
              "nvfp4_expand_raw_scales_sm70_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  scale_codes.scalar_type() == torch::kUInt8 &&
                  global_scales.scalar_type() == torch::kFloat32,
              "nvfp4_expand_raw_scales_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && scale_codes.is_contiguous() &&
                  global_scales.is_contiguous(),
              "nvfp4_expand_raw_scales_sm70_out: tensors must be contiguous");
  TORCH_CHECK(out.sizes() == scale_codes.sizes() && out.dim() == 3 &&
                  out.size(0) == 512,
              "nvfp4_expand_raw_scales_sm70_out: scale shape mismatch");
  const bool is_w13 = out.sizes() == torch::IntArrayRef({512, 160, 320});
  const bool is_w2 = out.sizes() == torch::IntArrayRef({512, 10, 2560});
  TORCH_CHECK(is_w13 || is_w2,
              "nvfp4_expand_raw_scales_sm70_out: unsupported Qwen3.8 shape");
  TORCH_CHECK(global_scales.sizes() == (is_w13 ? torch::IntArrayRef({512, 2})
                                               : torch::IntArrayRef({512, 1})),
              "nvfp4_expand_raw_scales_sm70_out: global-scale shape mismatch");
  TORCH_CHECK(out.get_device() == scale_codes.get_device() &&
                  out.get_device() == global_scales.get_device(),
              "nvfp4_expand_raw_scales_sm70_out: device mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  constexpr int threads = 256;
  const size_t elements = static_cast<size_t>(out.numel());
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  nvfp4_expand_raw_scales_sm70_kernel<<<blocks, threads, 0,
                                        at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      scale_codes.data_ptr<uint8_t>(), global_scales.data_ptr<float>(),
      elements, static_cast<size_t>(out.numel() / out.size(0)),
      static_cast<int>(out.size(2)), is_w13 ? 2 : 1, interleaved_w13,
      fast_decode_rounding);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
  const int64_t routes = expert_ids.numel();
  TORCH_CHECK(routes >= 10 && routes <= 160 && routes % 10 == 0 &&
                  out.size(0) == routes && weights.size(0) == 512 &&
                  scales.size(0) == 512,
              "nvfp4_moe_qpn_m1_sm70_out: expected 10..160 routes in "
              "10-route token groups and 512 experts");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == weights.get_device() &&
                  input.get_device() == scales.get_device() &&
                  input.get_device() == expert_ids.get_device(),
              "nvfp4_moe_qpn_m1_sm70_out: device mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const int64_t groups_k16 = input.size(1) / 16;
  TORCH_CHECK(split_k > 0 && split_k <= 32 && groups_k16 % split_k == 0,
              "nvfp4_moe_qpn_m1_sm70_out: split_k must divide K/16");
  const bool is_w13 = input.size(1) == 2560 && out.size(1) == 320;
  if (is_w13) {
    const int64_t expected_input_rows = broadcast_input ? routes / 10 : routes;
    TORCH_CHECK(
        input.sizes() == torch::IntArrayRef({expected_input_rows, 2560}) &&
            out.sizes() == torch::IntArrayRef({routes, 320}) &&
            weights.sizes() == torch::IntArrayRef({512, 2560, 40}) &&
            scales.sizes() == torch::IntArrayRef({512, 160, 320}),
        "nvfp4_moe_qpn_m1_sm70_out: W13 tensor contract mismatch");
  } else {
    TORCH_CHECK(!broadcast_input &&
                    input.sizes() == torch::IntArrayRef({routes, 160}) &&
                    out.sizes() == torch::IntArrayRef({routes, 2560}) &&
                    weights.sizes() == torch::IntArrayRef({512, 160, 320}) &&
                    scales.sizes() == torch::IntArrayRef({512, 10, 2560}),
                "nvfp4_moe_qpn_m1_sm70_out: W2 tensor contract mismatch");
  }
  dispatch_nvfp4_qpn_m1(out, input, weights, scales, expert_ids,
                        broadcast_input, split_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_moe_qpn_raw_scale_sm70_out(torch::Tensor out, torch::Tensor input,
                                      torch::Tensor weights,
                                      torch::Tensor scale_codes,
                                      torch::Tensor global_scales,
                                      torch::Tensor expert_ids,
                                      bool broadcast_input,
                                      bool interleaved_w13, int64_t split_k) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && weights.is_cuda() &&
                  scale_codes.is_cuda() && global_scales.is_cuda() &&
                  expert_ids.is_cuda(),
              "nvfp4_moe_qpn_raw_scale_sm70_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weights.scalar_type() == torch::kInt32 &&
                  scale_codes.scalar_type() == torch::kUInt8 &&
                  global_scales.scalar_type() == torch::kFloat32 &&
                  expert_ids.scalar_type() == torch::kInt32,
              "nvfp4_moe_qpn_raw_scale_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  weights.is_contiguous() && scale_codes.is_contiguous() &&
                  global_scales.is_contiguous() && expert_ids.is_contiguous(),
              "nvfp4_moe_qpn_raw_scale_sm70_out: tensors must be contiguous");
  const int64_t routes = expert_ids.numel();
  TORCH_CHECK(routes >= 10 && routes <= 160 && routes % 10 == 0 &&
                  out.size(0) == routes && weights.size(0) == 512 &&
                  scale_codes.size(0) == 512,
              "nvfp4_moe_qpn_raw_scale_sm70_out: expected 10..160 routes ");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == weights.get_device() &&
                  input.get_device() == scale_codes.get_device() &&
                  input.get_device() == global_scales.get_device() &&
                  input.get_device() == expert_ids.get_device(),
              "nvfp4_moe_qpn_raw_scale_sm70_out: device mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const int64_t groups_k16 = input.size(1) / 16;
  TORCH_CHECK(split_k > 0 && split_k <= 32 && groups_k16 % split_k == 0,
              "nvfp4_moe_qpn_raw_scale_sm70_out: split_k must divide K/16");
  const bool is_w13 = input.size(1) == 2560 && out.size(1) == 320;
  if (is_w13) {
    const int64_t expected_input_rows = broadcast_input ? routes / 10 : routes;
    TORCH_CHECK(
        input.sizes() == torch::IntArrayRef({expected_input_rows, 2560}) &&
            out.sizes() == torch::IntArrayRef({routes, 320}) &&
            weights.sizes() == torch::IntArrayRef({512, 2560, 40}) &&
            scale_codes.sizes() == torch::IntArrayRef({512, 160, 320}) &&
            global_scales.sizes() == torch::IntArrayRef({512, 2}),
        "nvfp4_moe_qpn_raw_scale_sm70_out: W13 tensor contract mismatch");
  } else {
    TORCH_CHECK(
        !broadcast_input && !interleaved_w13 &&
            input.sizes() == torch::IntArrayRef({routes, 160}) &&
            out.sizes() == torch::IntArrayRef({routes, 2560}) &&
            weights.sizes() == torch::IntArrayRef({512, 160, 320}) &&
            scale_codes.sizes() == torch::IntArrayRef({512, 10, 2560}) &&
            global_scales.sizes() == torch::IntArrayRef({512, 1}),
        "nvfp4_moe_qpn_raw_scale_sm70_out: W2 tensor contract mismatch");
  }
  dispatch_nvfp4_qpn_raw_scale(out, input, weights, scale_codes, global_scales,
                               expert_ids, broadcast_input, interleaved_w13,
                               split_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_qwen38_w2_direct_reduce_out(torch::Tensor out, torch::Tensor input,
                                       torch::Tensor weights,
                                       torch::Tensor scales,
                                       torch::Tensor expert_ids,
                                       torch::Tensor topk_weights) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && weights.is_cuda() &&
                  scales.is_cuda() && expert_ids.is_cuda() &&
                  topk_weights.is_cuda(),
              "nvfp4_qwen38_w2_direct_reduce_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weights.scalar_type() == torch::kInt32 &&
                  scales.scalar_type() == torch::kFloat16 &&
                  expert_ids.scalar_type() == torch::kInt32 &&
                  topk_weights.scalar_type() == torch::kFloat32,
              "nvfp4_qwen38_w2_direct_reduce_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  weights.is_contiguous() && scales.is_contiguous() &&
                  expert_ids.is_contiguous() && topk_weights.is_contiguous(),
              "nvfp4_qwen38_w2_direct_reduce_out: tensors must be contiguous");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({1, 2560}) &&
                  input.sizes() == torch::IntArrayRef({10, 160}) &&
                  weights.sizes() == torch::IntArrayRef({512, 160, 320}) &&
                  scales.sizes() == torch::IntArrayRef({512, 10, 2560}) &&
                  expert_ids.numel() == 10 && topk_weights.numel() == 10,
              "nvfp4_qwen38_w2_direct_reduce_out: shape mismatch");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == weights.get_device() &&
                  input.get_device() == scales.get_device() &&
                  input.get_device() == expert_ids.get_device() &&
                  input.get_device() == topk_weights.get_device(),
              "nvfp4_qwen38_w2_direct_reduce_out: device mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  nvfp4_qwen38_w2_direct_reduce_kernel<<<80, 320, 0,
                                         at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      expert_ids.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_qwen38_w13_fused_swiglu_out(torch::Tensor out, torch::Tensor input,
                                       torch::Tensor weights,
                                       torch::Tensor scales,
                                       torch::Tensor expert_ids) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && weights.is_cuda() &&
                  scales.is_cuda() && expert_ids.is_cuda(),
              "nvfp4_qwen38_w13_fused_swiglu_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weights.scalar_type() == torch::kInt32 &&
                  scales.scalar_type() == torch::kFloat16 &&
                  expert_ids.scalar_type() == torch::kInt32,
              "nvfp4_qwen38_w13_fused_swiglu_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  weights.is_contiguous() && scales.is_contiguous() &&
                  expert_ids.is_contiguous(),
              "nvfp4_qwen38_w13_fused_swiglu_out: tensors must be contiguous");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({10, 160}) &&
                  input.sizes() == torch::IntArrayRef({1, 2560}) &&
                  weights.sizes() == torch::IntArrayRef({512, 2560, 40}) &&
                  scales.sizes() == torch::IntArrayRef({512, 160, 320}) &&
                  expert_ids.numel() == 10,
              "nvfp4_qwen38_w13_fused_swiglu_out: shape mismatch");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == weights.get_device() &&
                  input.get_device() == scales.get_device() &&
                  input.get_device() == expert_ids.get_device(),
              "nvfp4_qwen38_w13_fused_swiglu_out: device mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  launch_nvfp4_qwen38_w13_fused_swiglu(out, input, weights, scales, expert_ids);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qwen38_shared_gate_exact_out(torch::Tensor out, torch::Tensor input,
                                  torch::Tensor weight) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && weight.is_cuda(),
              "qwen38_shared_gate_exact_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weight.scalar_type() == torch::kFloat16,
              "qwen38_shared_gate_exact_out: tensors must be float16");
  TORCH_CHECK(
      out.is_contiguous() && input.is_contiguous() && weight.is_contiguous(),
      "qwen38_shared_gate_exact_out: tensors must be contiguous");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({1, 2560}) &&
                  input.sizes() == torch::IntArrayRef({1, 2560}) &&
                  weight.sizes() == torch::IntArrayRef({1, 2560}),
              "qwen38_shared_gate_exact_out: expected M1/N1/K2560 tensors");
  TORCH_CHECK(out.get_device() == input.get_device() &&
                  out.get_device() == weight.get_device(),
              "qwen38_shared_gate_exact_out: device mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  qwen38_shared_gate_exact_kernel<<<1, kQwen38SharedGateThreads, 0,
                                    at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_glm53_moe_q8_qpn_sm70_out(torch::Tensor out, torch::Tensor input,
                                     torch::Tensor weights,
                                     torch::Tensor scales,
                                     torch::Tensor expert_ids,
                                     torch::Tensor sorted_row_idx, bool w13) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && weights.is_cuda() &&
                  scales.is_cuda() && expert_ids.is_cuda() &&
                  sorted_row_idx.is_cuda(),
              "nvfp4_glm53_moe_q8_qpn_sm70_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weights.scalar_type() == torch::kInt32 &&
                  scales.scalar_type() == torch::kFloat16 &&
                  expert_ids.scalar_type() == torch::kInt32 &&
                  sorted_row_idx.scalar_type() == torch::kInt32,
              "nvfp4_glm53_moe_q8_qpn_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  weights.is_contiguous() && scales.is_contiguous() &&
                  expert_ids.is_contiguous() && sorted_row_idx.is_contiguous(),
              "nvfp4_glm53_moe_q8_qpn_sm70_out: tensors must be contiguous");
  TORCH_CHECK(out.dim() == 2 && input.dim() == 2 && weights.dim() == 3 &&
                  scales.dim() == 3 && expert_ids.numel() == 64 &&
                  sorted_row_idx.numel() == 64,
              "nvfp4_glm53_moe_q8_qpn_sm70_out: q8 rank/route mismatch");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == weights.get_device() &&
                  input.get_device() == scales.get_device() &&
                  input.get_device() == expert_ids.get_device() &&
                  input.get_device() == sorted_row_idx.get_device(),
              "nvfp4_glm53_moe_q8_qpn_sm70_out: device mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "nvfp4_glm53_moe_q8_qpn_sm70_out: requires SM70");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (w13) {
    TORCH_CHECK(input.sizes() == torch::IntArrayRef({64, 4096}) &&
                    out.sizes() == torch::IntArrayRef({64, 512}) &&
                    weights.sizes() == torch::IntArrayRef({288, 4096, 64}) &&
                    scales.sizes() == torch::IntArrayRef({288, 256, 512}),
                "nvfp4_glm53_moe_q8_qpn_sm70_out: W13 shape mismatch");
    nvfp4_glm53_moe_q8_qpn_sm70_kernel<true><<<dim3(16, 64), 96, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        expert_ids.data_ptr<int32_t>(), sorted_row_idx.data_ptr<int32_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  } else {
    TORCH_CHECK(input.sizes() == torch::IntArrayRef({64, 256}) &&
                    out.sizes() == torch::IntArrayRef({64, 4096}) &&
                    weights.sizes() == torch::IntArrayRef({288, 256, 512}) &&
                    scales.sizes() == torch::IntArrayRef({288, 16, 4096}),
                "nvfp4_glm53_moe_q8_qpn_sm70_out: W2 shape mismatch");
    nvfp4_glm53_moe_q8_qpn_sm70_kernel<false><<<dim3(128, 64), 32, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        expert_ids.data_ptr<int32_t>(), sorted_row_idx.data_ptr<int32_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_moe_qpn_mtp5_sm70_out(torch::Tensor out, torch::Tensor input,
                                 torch::Tensor weights, torch::Tensor scales,
                                 torch::Tensor expert_ids, bool broadcast_input,
                                 int64_t split_k) {
  TORCH_CHECK(expert_ids.numel() == 50 && out.size(0) == 50,
              "nvfp4_moe_qpn_mtp5_sm70_out: expected fifty routes");
  nvfp4_moe_qpn_m1_sm70_out(out, input, weights, scales, expert_ids,
                            broadcast_input, split_k);
}

template <int kSplitK, bool kInterleaved>
void launch_nvfp4_qpn_w13_swiglu_batch(torch::Tensor out, torch::Tensor input,
                                       torch::Tensor weights,
                                       torch::Tensor scales,
                                       torch::Tensor expert_ids) {
  const int routes = static_cast<int>(expert_ids.numel());
  nvfp4_qpn_w13_swiglu_batch_sm70_kernel<kSplitK, kInterleaved>
      <<<dim3(5, static_cast<unsigned>(routes)), 64 * kSplitK, 0,
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
          scales.data_ptr<at::Half>(), nullptr, expert_ids.data_ptr<int32_t>(),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
}

template <int kSplitK, bool kInterleaved>
void launch_nvfp4_qpn_raw_w13_swiglu_batch(torch::Tensor out,
                                           torch::Tensor input,
                                           torch::Tensor weights,
                                           torch::Tensor scale_codes,
                                           torch::Tensor global_scales,
                                           torch::Tensor expert_ids) {
  const int routes = static_cast<int>(expert_ids.numel());
  nvfp4_qpn_w13_swiglu_batch_sm70_kernel<kSplitK, kInterleaved, true>
      <<<dim3(5, static_cast<unsigned>(routes)), 64 * kSplitK, 0,
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
          scale_codes.data_ptr<uint8_t>(), global_scales.data_ptr<float>(),
          expert_ids.data_ptr<int32_t>(),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
}

void nvfp4_moe_qpn_w13_swiglu_batch_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor weights,
    torch::Tensor scales, torch::Tensor expert_ids, bool interleaved) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && weights.is_cuda() &&
                  scales.is_cuda() && expert_ids.is_cuda(),
              "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weights.scalar_type() == torch::kInt32 &&
                  scales.scalar_type() == torch::kFloat16 &&
                  expert_ids.scalar_type() == torch::kInt32,
              "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  weights.is_contiguous() && scales.is_contiguous() &&
                  expert_ids.is_contiguous(),
              "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out: tensors must be "
              "contiguous");
  const int64_t tokens = input.size(0);
  const int64_t routes = tokens * 10;
  TORCH_CHECK((tokens == 4 || tokens == 8 || tokens == 16) &&
                  out.sizes() == torch::IntArrayRef({routes, 160}) &&
                  input.sizes() == torch::IntArrayRef({tokens, 2560}) &&
                  weights.sizes() == torch::IntArrayRef({512, 2560, 40}) &&
                  scales.sizes() == torch::IntArrayRef({512, 160, 320}) &&
                  expert_ids.sizes() == torch::IntArrayRef({routes}),
              "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out: expected exact "
              "M4/M8/M16 "
              "Qwen3.8 W13 tensors");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == weights.get_device() &&
                  input.get_device() == scales.get_device() &&
                  input.get_device() == expert_ids.get_device(),
              "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out: device mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
#define VLLM_LAUNCH_QWEN38_W13_SWIGLU(SPLIT)                               \
  do {                                                                     \
    if (interleaved) {                                                     \
      launch_nvfp4_qpn_w13_swiglu_batch<SPLIT, true>(out, input, weights,  \
                                                     scales, expert_ids);  \
    } else {                                                               \
      launch_nvfp4_qpn_w13_swiglu_batch<SPLIT, false>(out, input, weights, \
                                                      scales, expert_ids); \
    }                                                                      \
  } while (false)
  if (tokens == 4) {
    VLLM_LAUNCH_QWEN38_W13_SWIGLU(5);
  } else if (tokens == 8) {
    VLLM_LAUNCH_QWEN38_W13_SWIGLU(4);
  } else {
    VLLM_LAUNCH_QWEN38_W13_SWIGLU(1);
  }
#undef VLLM_LAUNCH_QWEN38_W13_SWIGLU
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor weights,
    torch::Tensor scale_codes, torch::Tensor global_scales,
    torch::Tensor expert_ids, bool interleaved) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && weights.is_cuda() &&
                  scale_codes.is_cuda() && global_scales.is_cuda() &&
                  expert_ids.is_cuda(),
              "nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out: tensors must be "
              "CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weights.scalar_type() == torch::kInt32 &&
                  scale_codes.scalar_type() == torch::kUInt8 &&
                  global_scales.scalar_type() == torch::kFloat32 &&
                  expert_ids.scalar_type() == torch::kInt32,
              "nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  weights.is_contiguous() && scale_codes.is_contiguous() &&
                  global_scales.is_contiguous() && expert_ids.is_contiguous(),
              "nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out: tensors must be "
              "contiguous");
  const int64_t tokens = input.size(0);
  const int64_t routes = tokens * 10;
  TORCH_CHECK((tokens == 4 || tokens == 8 || tokens == 16) &&
                  out.sizes() == torch::IntArrayRef({routes, 160}) &&
                  input.sizes() == torch::IntArrayRef({tokens, 2560}) &&
                  weights.sizes() == torch::IntArrayRef({512, 2560, 40}) &&
                  scale_codes.sizes() == torch::IntArrayRef({512, 160, 320}) &&
                  global_scales.sizes() == torch::IntArrayRef({512, 2}) &&
                  expert_ids.sizes() == torch::IntArrayRef({routes}),
              "nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out: expected exact "
              "M4/M8/M16 Qwen3.8 tensors");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == weights.get_device() &&
                  input.get_device() == scale_codes.get_device() &&
                  input.get_device() == global_scales.get_device() &&
                  input.get_device() == expert_ids.get_device(),
              "nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out: device mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
#define VLLM_LAUNCH_QWEN38_RAW_W13_SWIGLU(SPLIT)                        \
  do {                                                                  \
    if (interleaved) {                                                  \
      launch_nvfp4_qpn_raw_w13_swiglu_batch<SPLIT, true>(               \
          out, input, weights, scale_codes, global_scales, expert_ids); \
    } else {                                                            \
      launch_nvfp4_qpn_raw_w13_swiglu_batch<SPLIT, false>(              \
          out, input, weights, scale_codes, global_scales, expert_ids); \
    }                                                                   \
  } while (false)
  if (tokens == 4) {
    VLLM_LAUNCH_QWEN38_RAW_W13_SWIGLU(5);
  } else if (tokens == 8) {
    VLLM_LAUNCH_QWEN38_RAW_W13_SWIGLU(4);
  } else {
    VLLM_LAUNCH_QWEN38_RAW_W13_SWIGLU(1);
  }
#undef VLLM_LAUNCH_QWEN38_RAW_W13_SWIGLU
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_moe_qpn_w2_reduce_sm70_out(torch::Tensor out, torch::Tensor input,
                                      torch::Tensor weights,
                                      torch::Tensor scales,
                                      torch::Tensor expert_ids,
                                      torch::Tensor topk_weights) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && weights.is_cuda() &&
                  scales.is_cuda() && expert_ids.is_cuda() &&
                  topk_weights.is_cuda(),
              "nvfp4_moe_qpn_w2_reduce_sm70_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weights.scalar_type() == torch::kInt32 &&
                  scales.scalar_type() == torch::kFloat16 &&
                  expert_ids.scalar_type() == torch::kInt32 &&
                  topk_weights.scalar_type() == torch::kFloat32,
              "nvfp4_moe_qpn_w2_reduce_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  weights.is_contiguous() && scales.is_contiguous() &&
                  expert_ids.is_contiguous() && topk_weights.is_contiguous(),
              "nvfp4_moe_qpn_w2_reduce_sm70_out: tensors must be contiguous");
  const int64_t tokens = out.size(0);
  TORCH_CHECK(tokens >= 1 && tokens <= 16 &&
                  out.sizes() == torch::IntArrayRef({tokens, 2560}) &&
                  input.sizes() == torch::IntArrayRef({tokens * 10, 160}) &&
                  weights.sizes() == torch::IntArrayRef({512, 160, 320}) &&
                  scales.sizes() == torch::IntArrayRef({512, 10, 2560}) &&
                  expert_ids.sizes() == torch::IntArrayRef({tokens * 10}) &&
                  topk_weights.sizes() == torch::IntArrayRef({tokens, 10}),
              "nvfp4_moe_qpn_w2_reduce_sm70_out: expected Qwen3.8 E512/K10 "
              "W2 tensors for 1..16 tokens");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == weights.get_device() &&
                  input.get_device() == scales.get_device() &&
                  input.get_device() == expert_ids.get_device() &&
                  input.get_device() == topk_weights.get_device(),
              "nvfp4_moe_qpn_w2_reduce_sm70_out: device mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  nvfp4_qpn_w2_reduce_sm70_kernel<false>
      <<<dim3(80, static_cast<unsigned>(tokens)), 320, 0,
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
          scales.data_ptr<at::Half>(), nullptr, expert_ids.data_ptr<int32_t>(),
          topk_weights.data_ptr<float>(),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()),
          static_cast<int>(tokens));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_moe_qpn_raw_w2_reduce_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor weights,
    torch::Tensor scale_codes, torch::Tensor global_scales,
    torch::Tensor expert_ids, torch::Tensor topk_weights) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && weights.is_cuda() &&
                  scale_codes.is_cuda() && global_scales.is_cuda() &&
                  expert_ids.is_cuda() && topk_weights.is_cuda(),
              "nvfp4_moe_qpn_raw_w2_reduce_sm70_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  weights.scalar_type() == torch::kInt32 &&
                  scale_codes.scalar_type() == torch::kUInt8 &&
                  global_scales.scalar_type() == torch::kFloat32 &&
                  expert_ids.scalar_type() == torch::kInt32 &&
                  topk_weights.scalar_type() == torch::kFloat32,
              "nvfp4_moe_qpn_raw_w2_reduce_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  weights.is_contiguous() && scale_codes.is_contiguous() &&
                  global_scales.is_contiguous() && expert_ids.is_contiguous() &&
                  topk_weights.is_contiguous(),
              "nvfp4_moe_qpn_raw_w2_reduce_sm70_out: tensors must be "
              "contiguous");
  const int64_t tokens = out.size(0);
  TORCH_CHECK(tokens >= 1 && tokens <= 16 &&
                  out.sizes() == torch::IntArrayRef({tokens, 2560}) &&
                  input.sizes() == torch::IntArrayRef({tokens * 10, 160}) &&
                  weights.sizes() == torch::IntArrayRef({512, 160, 320}) &&
                  scale_codes.sizes() == torch::IntArrayRef({512, 10, 2560}) &&
                  global_scales.sizes() == torch::IntArrayRef({512, 1}) &&
                  expert_ids.sizes() == torch::IntArrayRef({tokens * 10}) &&
                  topk_weights.sizes() == torch::IntArrayRef({tokens, 10}),
              "nvfp4_moe_qpn_raw_w2_reduce_sm70_out: expected Qwen3.8 "
              "E512/K10 tensors for 1..16 tokens");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == weights.get_device() &&
                  input.get_device() == scale_codes.get_device() &&
                  input.get_device() == global_scales.get_device() &&
                  input.get_device() == expert_ids.get_device() &&
                  input.get_device() == topk_weights.get_device(),
              "nvfp4_moe_qpn_raw_w2_reduce_sm70_out: device mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  nvfp4_qpn_w2_reduce_sm70_kernel<true>
      <<<dim3(80, static_cast<unsigned>(tokens)), 320, 0,
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
          scale_codes.data_ptr<uint8_t>(), global_scales.data_ptr<float>(),
          expert_ids.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()),
          static_cast<int>(tokens));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
