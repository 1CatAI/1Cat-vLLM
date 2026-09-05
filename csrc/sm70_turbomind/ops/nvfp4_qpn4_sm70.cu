// SPDX-License-Identifier: Apache-2.0
// Qwen3.8 NVFP4 M=1 decode kernels for NVIDIA Volta SM70.
//
// The QPN4 quadpair-N m8n8k4 layout is derived from dnv2003/v100-skinny (MIT).
// See LICENSE.v100-skinny in this directory for the retained MIT notice.

#include <torch/all.h>
#include <torch/library.h>

#ifdef VLLM_QPN4_STANDALONE
  #include <ATen/core/dispatch/Dispatcher.h>
  #include <ATen/core/stack.h>
#endif
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kPrepareThreads = 256;
constexpr float kFp4Bias = 16384.0f;

struct SplitHalfScale {
  half hi;
  half lo;
};

SplitHalfScale split_half_scale(float value) {
  const half hi = __float2half_rn(value);
  const half lo = __float2half_rn(value - __half2float(hi));
  return {hi, lo};
}

__device__ __forceinline__ int qpn_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ int logical_k_from_physical(int physical) {
  const int local = physical & 7;
  return (physical & 8) + ((local & 3) << 1) + (local >> 2);
}

__global__ void nvfp4_qpn4_prepack_codes_kernel(
    uint8_t* __restrict__ codes, const uint8_t* __restrict__ qweight, int k,
    int n) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t packed_numel = static_cast<size_t>(k) * n / 2;
  if (index >= packed_numel) {
    return;
  }

  const int physical_byte = static_cast<int>(index & 7);
  size_t outer = index >> 3;
  const int lane = static_cast<int>(outer & 31);
  outer >>= 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int col = tile * 32 + qpn_col_from_lane(lane);
  const int physical_k0 = physical_byte * 2;
  const int logical_k0 = logical_k_from_physical(physical_k0);
  const int logical_k1 = logical_k_from_physical(physical_k0 + 1);
  const uint8_t lo =
      qweight[static_cast<size_t>(group * 16 + logical_k0) * n + col] & 0x0fU;
  const uint8_t hi =
      qweight[static_cast<size_t>(group * 16 + logical_k1) * n + col] & 0x0fU;
  codes[index] = lo | (hi << 4);
}

__global__ void nvfp4_qpn4_prepack_scales_kernel(
    half* __restrict__ packed_scales, const half* __restrict__ scales, int k,
    int n) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(k / 16) * n;
  if (index >= numel) {
    return;
  }
  const int lane = static_cast<int>(index & 31);
  size_t outer = index >> 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int col = tile * 32 + qpn_col_from_lane(lane);
  packed_scales[index] = __hmul(scales[static_cast<size_t>(group) * n + col],
                                __float2half_rn(kFp4Bias));
}

__global__ void nvfp4_qpn4_prepack_scale_codes_kernel(
    uint8_t* __restrict__ packed_scale_codes,
    const uint8_t* __restrict__ scale_codes, int k, int n) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(k / 16) * n;
  if (index >= numel) {
    return;
  }
  const int lane = static_cast<int>(index & 31);
  size_t outer = index >> 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int col = tile * 32 + qpn_col_from_lane(lane);
  packed_scale_codes[index] = scale_codes[static_cast<size_t>(group) * n + col];
}

__device__ __forceinline__ void fp4x8_to_half2x4(unsigned x, half2 out[4]) {
  constexpr unsigned kSign = 0x80008000U;
  constexpr unsigned kExponentMantissa = 0x0E000E00U;
  unsigned values[4];
  values[0] = ((x << 12) & kSign) | ((x << 9) & kExponentMantissa);
  values[1] = ((x << 8) & kSign) | ((x << 5) & kExponentMantissa);
  values[2] = ((x << 4) & kSign) | ((x << 1) & kExponentMantissa);
  values[3] = ((x << 0) & kSign) | ((x >> 3) & kExponentMantissa);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = *reinterpret_cast<const half2*>(&values[i]);
  }
}

__device__ __forceinline__ void fp4x16_to_half2x8(uint2 packed, half2 out[8]) {
  fp4x8_to_half2x4(packed.x, out);
  fp4x8_to_half2x4(packed.y, out + 4);
}

__device__ __forceinline__ half nvfp4_scale_code_to_half(uint8_t scale_code,
                                                         half scale_hi,
                                                         half scale_lo) {
  // NVFP4 group scales are positive, normal E4M3FN values. Adding the bias
  // delta while shifting the E4M3 payload produces the exact FP16 value.
  const unsigned half_bits =
      (static_cast<unsigned>(scale_code) << 7U) + 0x2000U;
  const half raw_scale = *reinterpret_cast<const half*>(&half_bits);
  const half correction = __hmul(raw_scale, scale_lo);
  return __hfma(raw_scale, scale_hi, correction);
}

template <bool UseScaleCode>
__global__ void nvfp4_qpn4_dequantize_sm70_kernel(
    half* __restrict__ output, const uint8_t* __restrict__ codes,
    const void* __restrict__ packed_scales, half scale_hi, half scale_lo, int n,
    int k) {
  const size_t word_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t word_count = static_cast<size_t>(k) * n / 16;
  if (word_index >= word_count) {
    return;
  }

  const int lane = static_cast<int>(word_index & 31);
  size_t outer = word_index >> 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int col = tile * 32 + qpn_col_from_lane(lane);
  half scale;
  if constexpr (UseScaleCode) {
    scale = nvfp4_scale_code_to_half(
        reinterpret_cast<const uint8_t*>(packed_scales)[word_index], scale_hi,
        scale_lo);
  } else {
    scale = reinterpret_cast<const half*>(packed_scales)[word_index];
  }
  half2 weights[8];
  fp4x16_to_half2x8(reinterpret_cast<const uint2*>(codes)[word_index], weights);
  const half2 scale2 = __halves2half2(scale, scale);
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const half2 value = __hmul2(weights[pair], scale2);
    // fp4x8_to_half2x4 pairs nibbles (0,4), (1,5), (2,6), (3,7)
    // within each 8-value word rather than adjacent physical K values.
    const int physical_k0 = (pair & 3) + ((pair & 4) ? 8 : 0);
    const int logical_k0 = logical_k_from_physical(physical_k0);
    const int logical_k1 = logical_k_from_physical(physical_k0 + 4);
    output[static_cast<size_t>(group * 16 + logical_k0) * n + col] =
        __low2half(value);
    output[static_cast<size_t>(group * 16 + logical_k1) * n + col] =
        __high2half(value);
  }
}

__global__ void nvfp4_qpn4_silu_and_mul_sm70_kernel(
    half* __restrict__ output, const half* __restrict__ gate_up, int rows,
    int hidden) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const half* row_input = gate_up + static_cast<size_t>(row) * hidden * 2;
  half* row_output = output + static_cast<size_t>(row) * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float gate = __half2float(row_input[col]);
    const float silu = gate / (1.0f + __expf(-gate));
    row_output[col] = __hmul(__float2half(silu), row_input[hidden + col]);
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

template <int SplitK, int NAcc, bool PrefetchCodes, bool UseScaleCode>
__global__ void nvfp4_qpn4_sm70_kernel(const uint8_t* __restrict__ codes,
                                       const void* __restrict__ packed_scales,
                                       half scale_hi, half scale_lo,
                                       const half* __restrict__ input,
                                       half* __restrict__ output, int n, int k,
                                       int m) {
  __shared__ float partials[SplitK][32];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const int group_begin = warp * groups_per_warp;
  const uint2* code_ptr = reinterpret_cast<const uint2*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const size_t scale_offset =
      static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half* scale_ptr =
      reinterpret_cast<const half*>(packed_scales) + scale_offset;
  const uint8_t* scale_code_ptr =
      reinterpret_cast<const uint8_t*>(packed_scales) + scale_offset;

  float accum[NAcc][8];
#pragma unroll
  for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[chain][i] = 0.0f;
    }
  }

  uint2 prefetched = make_uint2(0, 0);
  if constexpr (PrefetchCodes) {
    prefetched = __ldcs(code_ptr + static_cast<size_t>(group_begin) * 32);
  }

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const uint2 packed =
        PrefetchCodes ? prefetched
                      : __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    uint2 next = make_uint2(0, 0);
    if constexpr (PrefetchCodes) {
      if (group + 1 < group_begin + groups_per_warp) {
        next = __ldcs(code_ptr + static_cast<size_t>(group + 1) * 32);
      }
    }
    half2 weights[8];
    fp4x16_to_half2x8(packed, weights);
    half scale;
    if constexpr (UseScaleCode) {
      const uint8_t scale_code =
          __ldg(scale_code_ptr + static_cast<size_t>(group) * 32);
      scale = nvfp4_scale_code_to_half(scale_code, scale_hi, scale_lo);
    } else {
      scale = __ldg(scale_ptr + static_cast<size_t>(group) * 32);
    }
    const half2 scale2 = __halves2half2(scale, scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      weights[i] = __hmul2(weights[i], scale2);
    }

    uint4 input01 = make_uint4(0, 0, 0, 0);
    uint4 input23 = make_uint4(0, 0, 0, 0);
    if (row < m) {
      const half* input_row = input + static_cast<size_t>(row) * k;
      input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
      input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
    }
    const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
    VLLM_SM70_MMA_8N8K4(accum[0], a0[0], a0[1], b[0], b[1]);
    VLLM_SM70_MMA_8N8K4(accum[1 % NAcc], a0[2], a0[3], b[2], b[3]);
    VLLM_SM70_MMA_8N8K4(accum[2 % NAcc], a1[0], a1[1], b[4], b[5]);
    VLLM_SM70_MMA_8N8K4(accum[3 % NAcc], a1[2], a1[3], b[6], b[7]);
    if constexpr (PrefetchCodes) {
      prefetched = next;
    }
  }

#pragma unroll
  for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[0][i] += accum[chain][i];
    }
  }
  if ((lane & 17) == 0) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
      for (int offset = 0; offset < 2; ++offset) {
        const int i = pair * 4 + offset;
        const int output_col = offset | (((lane >> 1) & 1) << 1) | (pair << 2);
        partials[warp][quadpair * 8 + output_col] = accum[0][i];
      }
    }
  }
  __syncthreads();

  for (int element = threadIdx.x; element < 32; element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      value += partials[k_warp][element];
    }
    output[tile * 32 + element] = __float2half(value);
  }
}

template <int SplitK, int NAcc, bool PrefetchCodes, bool UseScaleCode>
__global__ void nvfp4_qpn4_gated_sm70_kernel(
    const uint8_t* __restrict__ codes, const void* __restrict__ packed_scales,
    half scale_hi, half scale_lo, const half* __restrict__ input,
    half* __restrict__ output, int hidden, int k, int m) {
  __shared__ float partials[2][SplitK][32];

  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int projection = warp_in_block / SplitK;
  const int warp = warp_in_block - projection * SplitK;
  const int hidden_tiles = hidden >> 5;
  const int tile = blockIdx.x + projection * hidden_tiles;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const int group_begin = warp * groups_per_warp;
  const uint2* code_ptr = reinterpret_cast<const uint2*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const size_t scale_offset =
      static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half* scale_ptr =
      reinterpret_cast<const half*>(packed_scales) + scale_offset;
  const uint8_t* scale_code_ptr =
      reinterpret_cast<const uint8_t*>(packed_scales) + scale_offset;

  float accum[NAcc][8];
#pragma unroll
  for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[chain][i] = 0.0f;
    }
  }
  uint2 prefetched = make_uint2(0, 0);
  if constexpr (PrefetchCodes) {
    prefetched = __ldcs(code_ptr + static_cast<size_t>(group_begin) * 32);
  }

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const uint2 packed =
        PrefetchCodes ? prefetched
                      : __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    uint2 next = make_uint2(0, 0);
    if constexpr (PrefetchCodes) {
      if (group + 1 < group_begin + groups_per_warp) {
        next = __ldcs(code_ptr + static_cast<size_t>(group + 1) * 32);
      }
    }
    half2 weights[8];
    fp4x16_to_half2x8(packed, weights);
    half scale;
    if constexpr (UseScaleCode) {
      const uint8_t scale_code =
          __ldg(scale_code_ptr + static_cast<size_t>(group) * 32);
      scale = nvfp4_scale_code_to_half(scale_code, scale_hi, scale_lo);
    } else {
      scale = __ldg(scale_ptr + static_cast<size_t>(group) * 32);
    }
    const half2 scale2 = __halves2half2(scale, scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      weights[i] = __hmul2(weights[i], scale2);
    }

    uint4 input01 = make_uint4(0, 0, 0, 0);
    uint4 input23 = make_uint4(0, 0, 0, 0);
    if (row < m) {
      const half* input_row = input + static_cast<size_t>(row) * k;
      input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
      input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
    }
    const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
    VLLM_SM70_MMA_8N8K4(accum[0], a0[0], a0[1], b[0], b[1]);
    VLLM_SM70_MMA_8N8K4(accum[1 % NAcc], a0[2], a0[3], b[2], b[3]);
    VLLM_SM70_MMA_8N8K4(accum[2 % NAcc], a1[0], a1[1], b[4], b[5]);
    VLLM_SM70_MMA_8N8K4(accum[3 % NAcc], a1[2], a1[3], b[6], b[7]);
    if constexpr (PrefetchCodes) {
      prefetched = next;
    }
  }

#pragma unroll
  for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[0][i] += accum[chain][i];
    }
  }
  if ((lane & 17) == 0) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
      for (int offset = 0; offset < 2; ++offset) {
        const int i = pair * 4 + offset;
        const int output_col = offset | (((lane >> 1) & 1) << 1) | (pair << 2);
        partials[projection][warp][quadpair * 8 + output_col] = accum[0][i];
      }
    }
  }
  __syncthreads();

  for (int element = threadIdx.x; element < 32; element += blockDim.x) {
    float gate = 0.0f;
    float up = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      gate += partials[0][k_warp][element];
      up += partials[1][k_warp][element];
    }
    const float silu = gate / (1.0f + __expf(-gate));
    output[blockIdx.x * 32 + element] = __float2half(silu * up);
  }
}

template <int SplitK, int NAcc, bool PrefetchCodes, bool UseScaleCode>
void launch_qpn4(const uint8_t* codes, const void* scales, half scale_hi,
                 half scale_lo, const half* input, half* output, int n, int k,
                 int m, cudaStream_t stream) {
  nvfp4_qpn4_sm70_kernel<SplitK, NAcc, PrefetchCodes, UseScaleCode>
      <<<(n / 32), (32 * SplitK), 0, stream>>>(
          codes, scales, scale_hi, scale_lo, input, output, n, k, m);
}

template <int SplitK, int NAcc, bool PrefetchCodes, bool UseScaleCode>
void launch_qpn4_gated(const uint8_t* codes, const void* scales, half scale_hi,
                       half scale_lo, const half* input, half* output,
                       int hidden, int k, int m, cudaStream_t stream) {
  nvfp4_qpn4_gated_sm70_kernel<SplitK, NAcc, PrefetchCodes, UseScaleCode>
      <<<(hidden / 32), (64 * SplitK), 0, stream>>>(
          codes, scales, scale_hi, scale_lo, input, output, hidden, k, m);
}

}  // namespace

std::vector<torch::Tensor> nvfp4_qpn4_prepare_sm70(torch::Tensor qweight,
                                                   torch::Tensor scales) {
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda(),
              "nvfp4_qpn4_prepare_sm70: tensors must be CUDA");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8 &&
                  scales.scalar_type() == torch::kFloat16,
              "nvfp4_qpn4_prepare_sm70: expected uint8 codes and FP16 scales");
  TORCH_CHECK(qweight.dim() == 2 && scales.dim() == 2 &&
                  qweight.is_contiguous() && scales.is_contiguous(),
              "nvfp4_qpn4_prepare_sm70: tensors must be contiguous matrices");
  TORCH_CHECK(qweight.get_device() == scales.get_device(),
              "nvfp4_qpn4_prepare_sm70: tensors must share one device");
  const int64_t k = qweight.size(0);
  const int64_t n = qweight.size(1);
  TORCH_CHECK(k > 0 && k % 128 == 0 && n > 0 && n % 32 == 0,
              "nvfp4_qpn4_prepare_sm70: shape alignment mismatch");
  TORCH_CHECK(scales.size(0) == k / 16 && scales.size(1) == n,
              "nvfp4_qpn4_prepare_sm70: scale shape mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto packed_codes = torch::empty(
      {k, n / 2},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kUInt8));
  auto packed_scales = torch::empty_like(scales);
  const int64_t code_numel = packed_codes.numel();
  const int code_blocks =
      static_cast<int>((code_numel + kPrepareThreads - 1) / kPrepareThreads);
  nvfp4_qpn4_prepack_codes_kernel<<<code_blocks, kPrepareThreads, 0, stream>>>(
      packed_codes.data_ptr<uint8_t>(), qweight.data_ptr<uint8_t>(),
      static_cast<int>(k), static_cast<int>(n));
  const int64_t scale_numel = packed_scales.numel();
  const int scale_blocks =
      static_cast<int>((scale_numel + kPrepareThreads - 1) / kPrepareThreads);
  nvfp4_qpn4_prepack_scales_kernel<<<scale_blocks, kPrepareThreads, 0,
                                     stream>>>(
      reinterpret_cast<half*>(packed_scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      static_cast<int>(k), static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed_codes, packed_scales};
}

std::vector<torch::Tensor> nvfp4_qpn4_prepare_scale_code_sm70(
    torch::Tensor qweight, torch::Tensor scale_codes) {
  TORCH_CHECK(qweight.is_cuda() && scale_codes.is_cuda(),
              "nvfp4_qpn4_prepare_scale_code_sm70: tensors must be CUDA");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8 &&
                  scale_codes.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "nvfp4_qpn4_prepare_scale_code_sm70: expected uint8 weights and "
              "float8_e4m3fn scale codes");
  TORCH_CHECK(qweight.dim() == 2 && scale_codes.dim() == 2 &&
                  qweight.is_contiguous() && scale_codes.is_contiguous(),
              "nvfp4_qpn4_prepare_scale_code_sm70: tensors must be contiguous "
              "matrices");
  TORCH_CHECK(qweight.get_device() == scale_codes.get_device(),
              "nvfp4_qpn4_prepare_scale_code_sm70: tensors must share one "
              "device");
  const int64_t k = qweight.size(0);
  const int64_t n = qweight.size(1);
  TORCH_CHECK(k > 0 && k % 128 == 0 && n > 0 && n % 32 == 0,
              "nvfp4_qpn4_prepare_scale_code_sm70: shape alignment mismatch");
  TORCH_CHECK(scale_codes.size(0) == k / 16 && scale_codes.size(1) == n,
              "nvfp4_qpn4_prepare_scale_code_sm70: scale shape mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto packed_codes = torch::empty(
      {k, n / 2},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kUInt8));
  auto packed_scale_codes = torch::empty(
      {k / 16, n},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kUInt8));
  const int64_t code_numel = packed_codes.numel();
  const int code_blocks =
      static_cast<int>((code_numel + kPrepareThreads - 1) / kPrepareThreads);
  nvfp4_qpn4_prepack_codes_kernel<<<code_blocks, kPrepareThreads, 0, stream>>>(
      packed_codes.data_ptr<uint8_t>(), qweight.data_ptr<uint8_t>(),
      static_cast<int>(k), static_cast<int>(n));
  const int64_t scale_numel = packed_scale_codes.numel();
  const int scale_blocks =
      static_cast<int>((scale_numel + kPrepareThreads - 1) / kPrepareThreads);
  nvfp4_qpn4_prepack_scale_codes_kernel<<<scale_blocks, kPrepareThreads, 0,
                                          stream>>>(
      packed_scale_codes.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(scale_codes.data_ptr()),
      static_cast<int>(k), static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed_codes, packed_scale_codes};
}

void nvfp4_qpn4_dequantize_sm70_out(torch::Tensor out, torch::Tensor codes,
                                    torch::Tensor scales, double global_scale,
                                    bool use_scale_code) {
  TORCH_CHECK(out.is_cuda() && codes.is_cuda() && scales.is_cuda(),
              "nvfp4_qpn4_dequantize_sm70_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8,
              "nvfp4_qpn4_dequantize_sm70_out: output must be FP16 and codes "
              "must be uint8");
  TORCH_CHECK((use_scale_code && scales.scalar_type() == torch::kUInt8) ||
                  (!use_scale_code && scales.scalar_type() == torch::kFloat16),
              "nvfp4_qpn4_dequantize_sm70_out: scale dtype mismatch");
  TORCH_CHECK(out.dim() == 2 && codes.dim() == 2 && scales.dim() == 2 &&
                  out.is_contiguous() && codes.is_contiguous() &&
                  scales.is_contiguous(),
              "nvfp4_qpn4_dequantize_sm70_out: tensors must be contiguous "
              "matrices");
  TORCH_CHECK(out.get_device() == codes.get_device() &&
                  out.get_device() == scales.get_device(),
              "nvfp4_qpn4_dequantize_sm70_out: tensors must share one device");
  const int64_t k = out.size(0);
  const int64_t n = out.size(1);
  TORCH_CHECK(k > 0 && k % 128 == 0 && n > 0 && n % 32 == 0,
              "nvfp4_qpn4_dequantize_sm70_out: shape alignment mismatch");
  TORCH_CHECK(codes.numel() == k * n / 2 && scales.numel() == k * n / 16,
              "nvfp4_qpn4_dequantize_sm70_out: packed tensor size mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  const int64_t word_count = k * n / 16;
  const int blocks =
      static_cast<int>((word_count + kPrepareThreads - 1) / kPrepareThreads);
  const SplitHalfScale split_scale =
      split_half_scale(static_cast<float>(global_scale) * kFp4Bias);
  const half zero_scale = __float2half_rn(0.0f);
  if (use_scale_code) {
    nvfp4_qpn4_dequantize_sm70_kernel<true>
        <<<blocks, kPrepareThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
            split_scale.hi, split_scale.lo, static_cast<int>(n),
            static_cast<int>(k));
  } else {
    nvfp4_qpn4_dequantize_sm70_kernel<false>
        <<<blocks, kPrepareThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            codes.data_ptr<uint8_t>(), scales.data_ptr<at::Half>(), zero_scale,
            zero_scale, static_cast<int>(n), static_cast<int>(k));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_qpn4_prefill_sm70_out(torch::Tensor out, int64_t dense_weight_ptr,
                                 torch::Tensor input, torch::Tensor codes,
                                 torch::Tensor scales, double global_scale,
                                 bool use_scale_code, bool gated_silu) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda(),
              "nvfp4_qpn4_prefill_sm70_out: input and output must be CUDA");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16,
              "nvfp4_qpn4_prefill_sm70_out: input and output must be FP16");
  TORCH_CHECK(input.dim() == 2 && out.dim() == 2 && input.is_contiguous() &&
                  out.is_contiguous(),
              "nvfp4_qpn4_prefill_sm70_out: invalid input or output");
  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  TORCH_CHECK(codes.dim() == 2 && codes.size(0) == k,
              "nvfp4_qpn4_prefill_sm70_out: code shape mismatch");
  const int64_t n = codes.size(1) * 2;
  TORCH_CHECK(out.size(0) == m && out.size(1) == (gated_silu ? n / 2 : n),
              "nvfp4_qpn4_prefill_sm70_out: output shape mismatch");

  // A zero pointer requests an operator-local workspace. This keeps the
  // 85 MiB dense FP16 buffer out of model load, AOT profile, and decode CUDA
  // graph capture. The caching allocator reuses the allocation across layers
  // during real prefill, while the worker's pre-capture empty_cache releases
  // the profiling allocation before decode graphs are recorded.
  auto dense_weight =
      dense_weight_ptr == 0
          ? torch::empty({k, n}, input.options())
          : torch::from_blob(reinterpret_cast<void*>(dense_weight_ptr), {k, n},
                             input.options());
  nvfp4_qpn4_dequantize_sm70_out(dense_weight, codes, scales, global_scale,
                                 use_scale_code);
  if (!gated_silu) {
    at::mm_out(out, input, dense_weight);
    return;
  }

  auto gate_up = at::mm(input, dense_weight);
  constexpr int kThreads = 256;
  nvfp4_qpn4_silu_and_mul_sm70_kernel<<<static_cast<int>(m), kThreads, 0,
                                        at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate_up.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(n / 2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifndef VLLM_QPN4_STANDALONE
void nvfp4_qpn2_dispatch_sm70_out(torch::Tensor out, torch::Tensor input,
                                  torch::Tensor codes, torch::Tensor scales,
                                  double global_scale, int64_t split_k,
                                  int64_t accumulator_chains,
                                  torch::Tensor tm_weight,
                                  torch::Tensor tm_scales,
                                  int64_t tm_group_size, int64_t tm_k_ld,
                                  int64_t tm_q_ld, bool gated_silu);
#endif

void nvfp4_qpn2_prefill_dispatch_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor codes,
    torch::Tensor scales, double global_scale, int64_t split_k,
    int64_t accumulator_chains, torch::Tensor tm_weight,
    torch::Tensor tm_scales, int64_t tm_group_size, int64_t tm_k_ld,
    int64_t tm_q_ld, bool gated_silu, int64_t min_prefill_m) {
  TORCH_CHECK(min_prefill_m > 8,
              "QPN2-packed prefill threshold must exceed M=8");
  if (input.size(0) >= min_prefill_m) {
    const int64_t k = input.size(1);
    const int64_t n = gated_silu ? out.size(1) * 2 : out.size(1);
    auto prefill_codes = codes.view({k, n / 2});
    auto prefill_scales = scales.view({k / 16, n});
    nvfp4_qpn4_prefill_sm70_out(out, 0, input, prefill_codes, prefill_scales,
                                global_scale, true, gated_silu);
    return;
  }

#ifndef VLLM_QPN4_STANDALONE
  nvfp4_qpn2_dispatch_sm70_out(out, input, codes, scales, global_scale, split_k,
                               accumulator_chains, tm_weight, tm_scales,
                               tm_group_size, tm_k_ld, tm_q_ld, gated_silu);
#else
  // Source overlays retain the validated decode extension. Route small M
  // through its existing opaque dispatch without exposing an M-dependent
  // branch to Dynamo; only the large-M branch is supplied by this sidecar.
  static const auto qpn2_dispatch =
      c10::Dispatcher::singleton().findSchemaOrThrow(
          "_C::nvfp4_qpn2_dispatch_sm70_out", "");
  torch::jit::Stack stack;
  stack.reserve(13);
  stack.emplace_back(out);
  stack.emplace_back(input);
  stack.emplace_back(codes);
  stack.emplace_back(scales);
  stack.emplace_back(global_scale);
  stack.emplace_back(split_k);
  stack.emplace_back(accumulator_chains);
  stack.emplace_back(tm_weight);
  stack.emplace_back(tm_scales);
  stack.emplace_back(tm_group_size);
  stack.emplace_back(tm_k_ld);
  stack.emplace_back(tm_q_ld);
  stack.emplace_back(gated_silu);
  qpn2_dispatch.callBoxed(&stack);
#endif
}

void nvfp4_qpn4_gemm_sm70_out(torch::Tensor out, torch::Tensor input,
                              torch::Tensor codes, torch::Tensor scales,
                              int64_t split_k, int64_t accumulator_chains,
                              bool prefetch_codes) {
  TORCH_CHECK(
      out.is_cuda() && input.is_cuda() && codes.is_cuda() && scales.is_cuda(),
      "nvfp4_qpn4_gemm_sm70_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8 &&
                  scales.scalar_type() == torch::kFloat16,
              "nvfp4_qpn4_gemm_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  codes.is_contiguous() && scales.is_contiguous(),
              "nvfp4_qpn4_gemm_sm70_out: tensors must be contiguous");
  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t n = out.size(1);
  TORCH_CHECK(m == 1 && out.size(0) == 1,
              "nvfp4_qpn4_gemm_sm70_out: M must be 1");
  TORCH_CHECK(k > 0 && k % 128 == 0 && n > 0 && n % 32 == 0,
              "nvfp4_qpn4_gemm_sm70_out: shape alignment mismatch");
  TORCH_CHECK(codes.numel() == k * n / 2 && scales.numel() == k * n / 16,
              "nvfp4_qpn4_gemm_sm70_out: packed tensor size mismatch");
  TORCH_CHECK(split_k == 4 || split_k == 8 || split_k == 10 || split_k == 16 ||
                  split_k == 17,
              "nvfp4_qpn4_gemm_sm70_out: unsupported split_k");
  TORCH_CHECK((k / 16) % split_k == 0,
              "nvfp4_qpn4_gemm_sm70_out: invalid split_k for K");
  TORCH_CHECK(accumulator_chains == 1 || accumulator_chains == 2,
              "nvfp4_qpn4_gemm_sm70_out: accumulator chains must be 1 or 2");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* code_ptr = codes.data_ptr<uint8_t>();
  const auto* scale_ptr =
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const auto* input_ptr =
      reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* output_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const half zero_scale = __float2half_rn(0.0f);

#define LAUNCH_QPN4(SPLIT, NACC, PREFETCH)                                \
  launch_qpn4<SPLIT, NACC, PREFETCH, false>(                              \
      code_ptr, scale_ptr, zero_scale, zero_scale, input_ptr, output_ptr, \
      static_cast<int>(n), static_cast<int>(k), static_cast<int>(m), stream)
#define DISPATCH_NACC_PREFETCH(SPLIT)                        \
  do {                                                       \
    if (accumulator_chains == 1 && !prefetch_codes) {        \
      LAUNCH_QPN4(SPLIT, 1, false);                          \
    } else if (accumulator_chains == 2 && !prefetch_codes) { \
      LAUNCH_QPN4(SPLIT, 2, false);                          \
    } else if (accumulator_chains == 1) {                    \
      LAUNCH_QPN4(SPLIT, 1, true);                           \
    } else {                                                 \
      LAUNCH_QPN4(SPLIT, 2, true);                           \
    }                                                        \
  } while (0)
  if (split_k == 4) {
    DISPATCH_NACC_PREFETCH(4);
  } else if (split_k == 8) {
    DISPATCH_NACC_PREFETCH(8);
  } else if (split_k == 10) {
    DISPATCH_NACC_PREFETCH(10);
  } else if (split_k == 16) {
    DISPATCH_NACC_PREFETCH(16);
  } else {
    DISPATCH_NACC_PREFETCH(17);
  }
#undef DISPATCH_NACC_PREFETCH
#undef LAUNCH_QPN4
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_qpn4_gated_sm70_out(torch::Tensor out, torch::Tensor input,
                               torch::Tensor codes, torch::Tensor scales,
                               int64_t split_k, int64_t accumulator_chains,
                               bool prefetch_codes) {
  TORCH_CHECK(
      out.is_cuda() && input.is_cuda() && codes.is_cuda() && scales.is_cuda(),
      "nvfp4_qpn4_gated_sm70_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8 &&
                  scales.scalar_type() == torch::kFloat16,
              "nvfp4_qpn4_gated_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  codes.is_contiguous() && scales.is_contiguous(),
              "nvfp4_qpn4_gated_sm70_out: tensors must be contiguous");
  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t hidden = out.size(1);
  const int64_t n = hidden * 2;
  TORCH_CHECK(m == 1 && out.size(0) == 1,
              "nvfp4_qpn4_gated_sm70_out: M must be 1");
  TORCH_CHECK(k > 0 && k % 128 == 0 && hidden > 0 && hidden % 32 == 0,
              "nvfp4_qpn4_gated_sm70_out: shape alignment mismatch");
  TORCH_CHECK(codes.numel() == k * n / 2 && scales.numel() == k * n / 16,
              "nvfp4_qpn4_gated_sm70_out: packed tensor size mismatch");
  TORCH_CHECK(split_k == 4 || split_k == 8 || split_k == 10 || split_k == 16,
              "nvfp4_qpn4_gated_sm70_out: unsupported split_k");
  TORCH_CHECK((k / 16) % split_k == 0,
              "nvfp4_qpn4_gated_sm70_out: invalid split_k for K");
  TORCH_CHECK(accumulator_chains == 1 || accumulator_chains == 2,
              "nvfp4_qpn4_gated_sm70_out: accumulator chains must be 1 or 2");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* code_ptr = codes.data_ptr<uint8_t>();
  const auto* scale_ptr =
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const auto* input_ptr =
      reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* output_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const half zero_scale = __float2half_rn(0.0f);

#define LAUNCH_QPN4_GATED(SPLIT, NACC, PREFETCH)                          \
  launch_qpn4_gated<SPLIT, NACC, PREFETCH, false>(                        \
      code_ptr, scale_ptr, zero_scale, zero_scale, input_ptr, output_ptr, \
      static_cast<int>(hidden), static_cast<int>(k), static_cast<int>(m), \
      stream)
#define DISPATCH_GATED_NACC_PREFETCH(SPLIT)                  \
  do {                                                       \
    if (accumulator_chains == 1 && !prefetch_codes) {        \
      LAUNCH_QPN4_GATED(SPLIT, 1, false);                    \
    } else if (accumulator_chains == 2 && !prefetch_codes) { \
      LAUNCH_QPN4_GATED(SPLIT, 2, false);                    \
    } else if (accumulator_chains == 1) {                    \
      LAUNCH_QPN4_GATED(SPLIT, 1, true);                     \
    } else {                                                 \
      LAUNCH_QPN4_GATED(SPLIT, 2, true);                     \
    }                                                        \
  } while (0)
  if (split_k == 4) {
    DISPATCH_GATED_NACC_PREFETCH(4);
  } else if (split_k == 8) {
    DISPATCH_GATED_NACC_PREFETCH(8);
  } else if (split_k == 10) {
    DISPATCH_GATED_NACC_PREFETCH(10);
  } else {
    DISPATCH_GATED_NACC_PREFETCH(16);
  }
#undef DISPATCH_GATED_NACC_PREFETCH
#undef LAUNCH_QPN4_GATED
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_qpn4_gemm_scale_code_sm70_out(torch::Tensor out, torch::Tensor input,
                                         torch::Tensor codes,
                                         torch::Tensor scale_codes,
                                         double global_scale, int64_t split_k,
                                         int64_t accumulator_chains,
                                         bool prefetch_codes) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && codes.is_cuda() &&
                  scale_codes.is_cuda(),
              "nvfp4_qpn4_gemm_scale_code_sm70_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8 &&
                  scale_codes.scalar_type() == torch::kUInt8,
              "nvfp4_qpn4_gemm_scale_code_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  codes.is_contiguous() && scale_codes.is_contiguous(),
              "nvfp4_qpn4_gemm_scale_code_sm70_out: tensors must be "
              "contiguous");
  TORCH_CHECK(out.get_device() == input.get_device() &&
                  out.get_device() == codes.get_device() &&
                  out.get_device() == scale_codes.get_device(),
              "nvfp4_qpn4_gemm_scale_code_sm70_out: tensors must share one "
              "device");
  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t n = out.size(1);
  TORCH_CHECK(m == 1 && out.size(0) == 1,
              "nvfp4_qpn4_gemm_scale_code_sm70_out: M must be 1");
  TORCH_CHECK(k > 0 && k % 128 == 0 && n > 0 && n % 32 == 0,
              "nvfp4_qpn4_gemm_scale_code_sm70_out: shape alignment "
              "mismatch");
  TORCH_CHECK(codes.numel() == k * n / 2 && scale_codes.numel() == k * n / 16,
              "nvfp4_qpn4_gemm_scale_code_sm70_out: packed tensor size "
              "mismatch");
  TORCH_CHECK(split_k == 4 || split_k == 8 || split_k == 10 || split_k == 16 ||
                  split_k == 17,
              "nvfp4_qpn4_gemm_scale_code_sm70_out: unsupported split_k");
  TORCH_CHECK((k / 16) % split_k == 0,
              "nvfp4_qpn4_gemm_scale_code_sm70_out: invalid split_k for K");
  TORCH_CHECK(accumulator_chains == 1 || accumulator_chains == 2,
              "nvfp4_qpn4_gemm_scale_code_sm70_out: accumulator chains must "
              "be 1 or 2");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* code_ptr = codes.data_ptr<uint8_t>();
  const auto* scale_code_ptr = scale_codes.data_ptr<uint8_t>();
  const SplitHalfScale split_scale =
      split_half_scale(static_cast<float>(global_scale) * kFp4Bias);
  const auto* input_ptr =
      reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* output_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());

#define LAUNCH_QPN4_LUT(SPLIT, NACC, PREFETCH)                             \
  launch_qpn4<SPLIT, NACC, PREFETCH, true>(                                \
      code_ptr, scale_code_ptr, split_scale.hi, split_scale.lo, input_ptr, \
      output_ptr, static_cast<int>(n), static_cast<int>(k),                \
      static_cast<int>(m), stream)
#define DISPATCH_QPN4_LUT(SPLIT)                             \
  do {                                                       \
    if (accumulator_chains == 1 && !prefetch_codes) {        \
      LAUNCH_QPN4_LUT(SPLIT, 1, false);                      \
    } else if (accumulator_chains == 2 && !prefetch_codes) { \
      LAUNCH_QPN4_LUT(SPLIT, 2, false);                      \
    } else if (accumulator_chains == 1) {                    \
      LAUNCH_QPN4_LUT(SPLIT, 1, true);                       \
    } else {                                                 \
      LAUNCH_QPN4_LUT(SPLIT, 2, true);                       \
    }                                                        \
  } while (0)
  if (split_k == 4) {
    DISPATCH_QPN4_LUT(4);
  } else if (split_k == 8) {
    DISPATCH_QPN4_LUT(8);
  } else if (split_k == 10) {
    DISPATCH_QPN4_LUT(10);
  } else if (split_k == 16) {
    DISPATCH_QPN4_LUT(16);
  } else {
    DISPATCH_QPN4_LUT(17);
  }
#undef DISPATCH_QPN4_LUT
#undef LAUNCH_QPN4_LUT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_qpn4_gated_scale_code_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor codes,
    torch::Tensor scale_codes, double global_scale, int64_t split_k,
    int64_t accumulator_chains, bool prefetch_codes) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda() && codes.is_cuda() &&
                  scale_codes.is_cuda(),
              "nvfp4_qpn4_gated_scale_code_sm70_out: tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8 &&
                  scale_codes.scalar_type() == torch::kUInt8,
              "nvfp4_qpn4_gated_scale_code_sm70_out: dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  codes.is_contiguous() && scale_codes.is_contiguous(),
              "nvfp4_qpn4_gated_scale_code_sm70_out: tensors must be "
              "contiguous");
  TORCH_CHECK(out.get_device() == input.get_device() &&
                  out.get_device() == codes.get_device() &&
                  out.get_device() == scale_codes.get_device(),
              "nvfp4_qpn4_gated_scale_code_sm70_out: tensors must share one "
              "device");
  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t hidden = out.size(1);
  const int64_t n = hidden * 2;
  TORCH_CHECK(m == 1 && out.size(0) == 1,
              "nvfp4_qpn4_gated_scale_code_sm70_out: M must be 1");
  TORCH_CHECK(k > 0 && k % 128 == 0 && hidden > 0 && hidden % 32 == 0,
              "nvfp4_qpn4_gated_scale_code_sm70_out: shape alignment "
              "mismatch");
  TORCH_CHECK(codes.numel() == k * n / 2 && scale_codes.numel() == k * n / 16,
              "nvfp4_qpn4_gated_scale_code_sm70_out: packed tensor size "
              "mismatch");
  TORCH_CHECK(split_k == 4 || split_k == 8 || split_k == 10 || split_k == 16,
              "nvfp4_qpn4_gated_scale_code_sm70_out: unsupported split_k");
  TORCH_CHECK((k / 16) % split_k == 0,
              "nvfp4_qpn4_gated_scale_code_sm70_out: invalid split_k for K");
  TORCH_CHECK(accumulator_chains == 1 || accumulator_chains == 2,
              "nvfp4_qpn4_gated_scale_code_sm70_out: accumulator chains "
              "must be 1 or 2");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* code_ptr = codes.data_ptr<uint8_t>();
  const auto* scale_code_ptr = scale_codes.data_ptr<uint8_t>();
  const SplitHalfScale split_scale =
      split_half_scale(static_cast<float>(global_scale) * kFp4Bias);
  const auto* input_ptr =
      reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* output_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());

#define LAUNCH_QPN4_GATED_LUT(SPLIT, NACC, PREFETCH)                       \
  launch_qpn4_gated<SPLIT, NACC, PREFETCH, true>(                          \
      code_ptr, scale_code_ptr, split_scale.hi, split_scale.lo, input_ptr, \
      output_ptr, static_cast<int>(hidden), static_cast<int>(k),           \
      static_cast<int>(m), stream)
#define DISPATCH_QPN4_GATED_LUT(SPLIT)                       \
  do {                                                       \
    if (accumulator_chains == 1 && !prefetch_codes) {        \
      LAUNCH_QPN4_GATED_LUT(SPLIT, 1, false);                \
    } else if (accumulator_chains == 2 && !prefetch_codes) { \
      LAUNCH_QPN4_GATED_LUT(SPLIT, 2, false);                \
    } else if (accumulator_chains == 1) {                    \
      LAUNCH_QPN4_GATED_LUT(SPLIT, 1, true);                 \
    } else {                                                 \
      LAUNCH_QPN4_GATED_LUT(SPLIT, 2, true);                 \
    }                                                        \
  } while (0)
  if (split_k == 4) {
    DISPATCH_QPN4_GATED_LUT(4);
  } else if (split_k == 8) {
    DISPATCH_QPN4_GATED_LUT(8);
  } else if (split_k == 10) {
    DISPATCH_QPN4_GATED_LUT(10);
  } else {
    DISPATCH_QPN4_GATED_LUT(16);
  }
#undef DISPATCH_QPN4_GATED_LUT
#undef LAUNCH_QPN4_GATED_LUT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_qpn4_dispatch_sm70_out(torch::Tensor out, int64_t dense_weight_ptr,
                                  torch::Tensor input, torch::Tensor codes,
                                  torch::Tensor scales, double global_scale,
                                  bool use_scale_code, bool gated_silu) {
  // Keep the dynamic-M decision inside the opaque operator so one compiled
  // graph can safely serve both M=1 decode and large-M prefill.
  if (input.size(0) == 1) {
    if (gated_silu) {
      TORCH_CHECK(use_scale_code,
                  "QPN4 gated decode requires E4M3 scale codes");
      nvfp4_qpn4_gated_scale_code_sm70_out(out, input, codes, scales,
                                           global_scale, 8, 2, false);
    } else if (use_scale_code) {
      nvfp4_qpn4_gemm_scale_code_sm70_out(out, input, codes, scales,
                                          global_scale, 17, 1, false);
    } else {
      nvfp4_qpn4_gemm_sm70_out(out, input, codes, scales, 17, 1, false);
    }
    return;
  }
  nvfp4_qpn4_prefill_sm70_out(out, dense_weight_ptr, input, codes, scales,
                              global_scale, use_scale_code, gated_silu);
}

#ifdef VLLM_QPN4_STANDALONE
TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def("nvfp4_qpn4_prepare_sm70(Tensor qweight, Tensor scales) -> Tensor[]");
  ops.impl("nvfp4_qpn4_prepare_sm70", torch::kCUDA, &nvfp4_qpn4_prepare_sm70);
  ops.def(
      "nvfp4_qpn4_prepare_scale_code_sm70(Tensor qweight, Tensor scale_codes) "
      "-> Tensor[]");
  ops.impl("nvfp4_qpn4_prepare_scale_code_sm70", torch::kCUDA,
           &nvfp4_qpn4_prepare_scale_code_sm70);
  ops.def(
      "nvfp4_qpn4_gemm_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor scales, int split_k, int accumulator_chains, "
      "bool prefetch_codes) -> ()");
  ops.impl("nvfp4_qpn4_gemm_sm70_out", torch::kCUDA, &nvfp4_qpn4_gemm_sm70_out);
  ops.def(
      "nvfp4_qpn4_gated_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor scales, int split_k, int accumulator_chains, "
      "bool prefetch_codes) -> ()");
  ops.impl("nvfp4_qpn4_gated_sm70_out", torch::kCUDA,
           &nvfp4_qpn4_gated_sm70_out);
  ops.def(
      "nvfp4_qpn4_gemm_scale_code_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor codes, Tensor scale_codes, float global_scale, int split_k, "
      "int accumulator_chains, bool prefetch_codes) -> ()");
  ops.impl("nvfp4_qpn4_gemm_scale_code_sm70_out", torch::kCUDA,
           &nvfp4_qpn4_gemm_scale_code_sm70_out);
  ops.def(
      "nvfp4_qpn4_gated_scale_code_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor codes, Tensor scale_codes, float global_scale, int split_k, "
      "int accumulator_chains, bool prefetch_codes) -> ()");
  ops.impl("nvfp4_qpn4_gated_scale_code_sm70_out", torch::kCUDA,
           &nvfp4_qpn4_gated_scale_code_sm70_out);
  ops.def(
      "nvfp4_qpn4_dequantize_sm70_out(Tensor(a!) out, Tensor codes, Tensor "
      "scales, float global_scale, bool use_scale_code) -> ()");
  ops.impl("nvfp4_qpn4_dequantize_sm70_out", torch::kCUDA,
           &nvfp4_qpn4_dequantize_sm70_out);
  ops.def(
      "nvfp4_qpn4_prefill_sm70_out(Tensor(a!) out, int dense_weight_ptr, "
      "Tensor input, Tensor codes, Tensor scales, float global_scale, bool "
      "use_scale_code, bool gated_silu) -> ()");
  ops.impl("nvfp4_qpn4_prefill_sm70_out", torch::kCUDA,
           &nvfp4_qpn4_prefill_sm70_out);
  ops.def(
      "nvfp4_qpn2_prefill_dispatch_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor codes, Tensor scales, float global_scale, int split_k, "
      "int accumulator_chains, Tensor tm_weight, Tensor tm_scales, "
      "int tm_group_size, int tm_k_ld, int tm_q_ld, bool gated_silu, "
      "int min_prefill_m) -> ()");
  ops.impl("nvfp4_qpn2_prefill_dispatch_sm70_out", torch::kCUDA,
           &nvfp4_qpn2_prefill_dispatch_sm70_out);
  ops.def(
      "nvfp4_qpn4_dispatch_sm70_out(Tensor(a!) out, int dense_weight_ptr, "
      "Tensor input, Tensor codes, Tensor scales, float global_scale, bool "
      "use_scale_code, bool gated_silu) -> ()");
  ops.impl("nvfp4_qpn4_dispatch_sm70_out", torch::kCUDA,
           &nvfp4_qpn4_dispatch_sm70_out);
}
#endif
