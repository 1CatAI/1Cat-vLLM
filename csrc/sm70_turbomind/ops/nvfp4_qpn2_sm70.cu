// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The QPN2 execution layout is derived from dnv2003/v100-skinny (MIT).
// See LICENSE.v100-skinny in this directory for the retained MIT notice.

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

#ifndef VLLM_NVFP4_QPN2_STANDALONE
void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void nvfp4_gemm_sm70_out(torch::Tensor out, torch::Tensor input,
                         torch::Tensor weight, torch::Tensor scales,
                         int64_t group_size, int64_t k_ld, int64_t q_ld,
                         bool gated_silu);
#endif

namespace {

constexpr int kPrepareThreads = 256;
constexpr int kQpn2RowsPerCta = 8;
constexpr int kQpn2MaxRows = 64;
constexpr int kQpn2DispatchMaxRows = 32;

__device__ __forceinline__ int qpn2_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ int qpn2_logical_k(int physical_k) {
  const int local = physical_k & 7;
  return (physical_k & 8) + ((local & 3) << 1) + (local >> 2);
}

__global__ void nvfp4_qpn2_prepack_codes_kernel(
    uint8_t* __restrict__ output, const uint8_t* __restrict__ weight, int n,
    int k) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k / 2;
  if (index >= numel) {
    return;
  }

  const int physical_byte = static_cast<int>(index & 7);
  size_t outer = index >> 3;
  const int lane = static_cast<int>(outer & 31);
  outer >>= 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  const int logical_k0 = group * 16 + qpn2_logical_k(physical_byte * 2);
  const int logical_k1 = group * 16 + qpn2_logical_k(physical_byte * 2 + 1);
  const int k_bytes = k >> 1;
  const uint8_t packed0 =
      weight[static_cast<size_t>(column) * k_bytes + (logical_k0 >> 1)];
  const uint8_t packed1 =
      weight[static_cast<size_t>(column) * k_bytes + (logical_k1 >> 1)];
  const uint8_t code0 =
      static_cast<uint8_t>((packed0 >> ((logical_k0 & 1) * 4)) & 0x0f);
  const uint8_t code1 =
      static_cast<uint8_t>((packed1 >> ((logical_k1 & 1) * 4)) & 0x0f);
  output[index] = static_cast<uint8_t>(code0 | (code1 << 4));
}

__global__ void nvfp4_qpn2_prepack_scales_kernel(
    uint8_t* __restrict__ output, const uint8_t* __restrict__ scales, int n,
    int k) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k / 16;
  if (index >= numel) {
    return;
  }

  const int lane = static_cast<int>(index & 31);
  size_t outer = index >> 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  output[index] = scales[static_cast<size_t>(column) * groups_k16 + group];
}

__device__ __forceinline__ half2 fp8e4m3_to_half2(uint8_t value) {
  const unsigned short bits =
      ((static_cast<unsigned short>(value) & 0x80u) << 8) |
      ((static_cast<unsigned short>(value) & 0x7fu) << 7);
  const half converted =
      __hmul(__ushort_as_half(bits), __ushort_as_half(0x5c00));
  return __halves2half2(converted, converted);
}

__device__ __forceinline__ void dequant_e2m1x8(unsigned packed, half2 scale,
                                               half2 output[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x0e000e00u;
  unsigned values[4];
  values[0] = ((packed << 12) & kSign) | ((packed << 9) & kExponentMantissa);
  values[1] = ((packed << 8) & kSign) | ((packed << 5) & kExponentMantissa);
  values[2] = ((packed << 4) & kSign) | ((packed << 1) & kExponentMantissa);
  values[3] = (packed & kSign) | ((packed >> 3) & kExponentMantissa);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    output[index] = __hmul2(*reinterpret_cast<half2*>(&values[index]), scale);
  }
}

#define VLLM_SM70_QPN2_MMA(C, A0, A1, B0, B1)                       \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int SplitK, int NAcc, int RowTiles = 1>
__global__ void nvfp4_qpn2_sm70_kernel(const uint8_t* __restrict__ codes,
                                       const uint8_t* __restrict__ group_scales,
                                       const half* __restrict__ input,
                                       half* __restrict__ output, int n, int k,
                                       int m, float global_scale) {
  static_assert(RowTiles == 1 || RowTiles == 2,
                "NVFP4 QPN2 supports one or two 8-row tiles");
  __shared__ float partials[SplitK][RowTiles * 256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row_base = blockIdx.y * kQpn2RowsPerCta * RowTiles;
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const int group_begin = warp * groups_per_warp;
  const uint2* code_ptr = reinterpret_cast<const uint2*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const uint8_t* scale_ptr =
      group_scales + static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half2 global_scale2 = __float2half2_rn(global_scale * 16384.0f);

  float accum[RowTiles][NAcc][8];
#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][chain][index] = 0.0f;
      }
    }
  }

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const uint2 packed = __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    const half2 scale = __hmul2(
        fp8e4m3_to_half2(__ldg(scale_ptr + static_cast<size_t>(group) * 32)),
        global_scale2);
    half2 weights[8];
    dequant_e2m1x8(packed.x, scale, weights);
    dequant_e2m1x8(packed.y, scale, weights + 4);

    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      const int row = row_base + row_tile * kQpn2RowsPerCta + local_row;
      if (row < m) {
        const half* input_row = input + static_cast<size_t>(row) * k;
        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
      }
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
      VLLM_SM70_QPN2_MMA(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][0][index] += accum[row_tile][chain][index];
      }
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int output_row = row_tile * kQpn2RowsPerCta + (index & 2) +
                             ((lane & 16) ? 4 : 0) + (lane & 1);
      const int output_col =
          (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
      partials[warp][output_row * 32 + quadpair * 8 + output_col] =
          accum[row_tile][0][index];
    }
  }
  __syncthreads();

  for (int element = threadIdx.x; element < RowTiles * 256;
       element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      value += partials[k_warp][element];
    }
    const int output_row = row_base + (element >> 5);
    const int output_col = element & 31;
    if (output_row < m) {
      output[static_cast<size_t>(output_row) * n + tile * 32 + output_col] =
          __float2half(value);
    }
  }
}

template <int SplitK, int NAcc, int RowTiles = 1>
__global__ void nvfp4_qpn2_gated_sm70_kernel(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ group_scales,
    const half* __restrict__ input, half* __restrict__ output, int hidden,
    int k, int m, float global_scale) {
  static_assert(RowTiles == 1 || RowTiles == 2,
                "NVFP4 gated QPN2 supports one or two 8-row tiles");
  __shared__ float partials[2][SplitK][RowTiles * 256];

  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int projection = warp_in_block / SplitK;
  const int warp = warp_in_block - projection * SplitK;
  const int hidden_tiles = hidden >> 5;
  const int tile = blockIdx.x + projection * hidden_tiles;
  const int quadpair = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row_base = blockIdx.y * kQpn2RowsPerCta * RowTiles;
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const int group_begin = warp * groups_per_warp;
  const uint2* code_ptr = reinterpret_cast<const uint2*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const uint8_t* scale_ptr =
      group_scales + static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half2 global_scale2 = __float2half2_rn(global_scale * 16384.0f);

  float accum[RowTiles][NAcc][8];
#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][chain][index] = 0.0f;
      }
    }
  }

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const uint2 packed = __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    const half2 scale = __hmul2(
        fp8e4m3_to_half2(__ldg(scale_ptr + static_cast<size_t>(group) * 32)),
        global_scale2);
    half2 weights[8];
    dequant_e2m1x8(packed.x, scale, weights);
    dequant_e2m1x8(packed.y, scale, weights + 4);

    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      const int row = row_base + row_tile * kQpn2RowsPerCta + local_row;
      if (row < m) {
        const half* input_row = input + static_cast<size_t>(row) * k;
        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
      }
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
      VLLM_SM70_QPN2_MMA(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][0][index] += accum[row_tile][chain][index];
      }
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int output_row = row_tile * kQpn2RowsPerCta + (index & 2) +
                             ((lane & 16) ? 4 : 0) + (lane & 1);
      const int output_col =
          (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
      partials[projection][warp][output_row * 32 + quadpair * 8 + output_col] =
          accum[row_tile][0][index];
    }
  }
  __syncthreads();

  for (int element = threadIdx.x; element < RowTiles * 256;
       element += blockDim.x) {
    float gate = 0.0f;
    float up = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      gate += partials[0][k_warp][element];
      up += partials[1][k_warp][element];
    }
    const int output_row = row_base + (element >> 5);
    const int output_col = element & 31;
    if (output_row < m) {
      // Match the existing SM70 silu_and_mul contract: round both GEMM
      // outputs to FP16 before the activation, round SiLU to FP16, then use
      // FP16 multiplication.  This makes fusion numerically equivalent to
      // QPN2 GEMM followed by the current activation kernel.
      const half gate_half = __float2half(gate);
      const half up_half = __float2half(up);
      const float gate_float = __half2float(gate_half);
      const half silu = __float2half(gate_float / (1.0f + expf(-gate_float)));
      output[static_cast<size_t>(output_row) * hidden + blockIdx.x * 32 +
             output_col] = __hmul(silu, up_half);
    }
  }
}

template <int SplitK, int NAcc, int RowTiles = 1>
void launch_qpn2(const uint8_t* codes, const uint8_t* scales, const half* input,
                 half* output, int n, int k, int m, float global_scale,
                 cudaStream_t stream) {
  constexpr int kRowsPerCta = kQpn2RowsPerCta * RowTiles;
  const dim3 grid(n / 32, (m + kRowsPerCta - 1) / kRowsPerCta);
  nvfp4_qpn2_sm70_kernel<SplitK, NAcc, RowTiles>
      <<<grid, (32 * SplitK), 0, stream>>>(codes, scales, input, output, n, k,
                                           m, global_scale);
}

template <int SplitK, int NAcc, int RowTiles = 1>
void launch_qpn2_gated(const uint8_t* codes, const uint8_t* scales,
                       const half* input, half* output, int hidden, int k,
                       int m, float global_scale, cudaStream_t stream) {
  constexpr int kRowsPerCta = kQpn2RowsPerCta * RowTiles;
  const dim3 grid(hidden / 32, (m + kRowsPerCta - 1) / kRowsPerCta);
  nvfp4_qpn2_gated_sm70_kernel<SplitK, NAcc, RowTiles>
      <<<grid, (64 * SplitK), 0, stream>>>(codes, scales, input, output, hidden,
                                           k, m, global_scale);
}

bool qpn2_m16_native_enabled(int m) {
  const char* value = std::getenv("VLLM_SM70_NVFP4_QPN2_M16_NATIVE");
  const bool enabled = m > kQpn2RowsPerCta && m <= 16 && value != nullptr &&
                       value[0] == '1' && value[1] == '\0';
  if (enabled) {
    static std::once_flag m16_log_once;
    std::call_once(m16_log_once, []() {
      std::fprintf(stderr,
                   "INFO SM70 NVFP4 QPN2 native M=9..16 two-row-tile "
                   "candidate enabled.\n");
    });
  }
  return enabled;
}

void check_qpn2_tensors(const torch::Tensor& out, const torch::Tensor& input,
                        const torch::Tensor& codes, const torch::Tensor& scales,
                        bool gated_silu) {
  TORCH_CHECK(
      out.is_cuda() && input.is_cuda() && codes.is_cuda() && scales.is_cuda(),
      "NVFP4 QPN2 tensors must be CUDA tensors");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  input.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8 &&
                  scales.scalar_type() == torch::kUInt8,
              "NVFP4 QPN2 dtype mismatch");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous() &&
                  codes.is_contiguous() && scales.is_contiguous(),
              "NVFP4 QPN2 tensors must be contiguous");
  TORCH_CHECK(out.dim() == 2 && input.dim() == 2,
              "NVFP4 QPN2 input and output must be matrices");
  TORCH_CHECK(out.get_device() == input.get_device() &&
                  out.get_device() == codes.get_device() &&
                  out.get_device() == scales.get_device(),
              "NVFP4 QPN2 tensors must share one device");
  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t n = gated_silu ? out.size(1) * 2 : out.size(1);
  TORCH_CHECK(m >= 1 && m <= kQpn2MaxRows && out.size(0) == m,
              "NVFP4 QPN2 requires M in [1, ", kQpn2MaxRows, "]");
  TORCH_CHECK(k > 0 && k % 64 == 0 && n > 0 && n % 32 == 0,
              "NVFP4 QPN2 shape alignment mismatch");
  TORCH_CHECK(codes.numel() == n * k / 2 && scales.numel() == n * k / 16,
              "NVFP4 QPN2 packed tensor size mismatch");
}

}  // namespace

std::vector<torch::Tensor> nvfp4_qpn2_prepare_sm70(torch::Tensor weight_packed,
                                                   torch::Tensor weight_scale) {
  TORCH_CHECK(weight_packed.is_cuda() && weight_scale.is_cuda(),
              "nvfp4_qpn2_prepare_sm70 expects CUDA tensors");
  TORCH_CHECK(weight_packed.scalar_type() == torch::kUInt8 &&
                  weight_scale.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "nvfp4_qpn2_prepare_sm70 expects uint8 weights and E4M3 scales");
  TORCH_CHECK(weight_packed.dim() == 2 && weight_scale.dim() == 2 &&
                  weight_packed.is_contiguous() && weight_scale.is_contiguous(),
              "nvfp4_qpn2_prepare_sm70 expects contiguous matrices");
  TORCH_CHECK(weight_packed.get_device() == weight_scale.get_device(),
              "nvfp4_qpn2_prepare_sm70 tensors must share one device");

  const int64_t n = weight_packed.size(0);
  const int64_t k = weight_packed.size(1) * 2;
  TORCH_CHECK(n > 0 && n % 32 == 0 && k > 0 && k % 64 == 0,
              "nvfp4_qpn2_prepare_sm70 shape alignment mismatch");
  TORCH_CHECK(weight_scale.size(0) == n && weight_scale.size(1) == k / 16,
              "nvfp4_qpn2_prepare_sm70 scale shape mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight_packed));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto codes = torch::empty_like(weight_packed);
  auto scales = torch::empty({n, k / 16}, torch::TensorOptions()
                                              .device(weight_packed.device())
                                              .dtype(torch::kUInt8));

  const int code_blocks =
      static_cast<int>((codes.numel() + kPrepareThreads - 1) / kPrepareThreads);
  nvfp4_qpn2_prepack_codes_kernel<<<code_blocks, kPrepareThreads, 0, stream>>>(
      codes.data_ptr<uint8_t>(), weight_packed.data_ptr<uint8_t>(),
      static_cast<int>(n), static_cast<int>(k));
  const int scale_blocks = static_cast<int>(
      (scales.numel() + kPrepareThreads - 1) / kPrepareThreads);
  nvfp4_qpn2_prepack_scales_kernel<<<scale_blocks, kPrepareThreads, 0,
                                     stream>>>(
      scales.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(weight_scale.data_ptr()),
      static_cast<int>(n), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes, scales};
}

void nvfp4_qpn2_gemm_sm70_out(torch::Tensor out, torch::Tensor input,
                              torch::Tensor codes, torch::Tensor scales,
                              double global_scale, int64_t split_k,
                              int64_t accumulator_chains) {
  check_qpn2_tensors(out, input, codes, scales, false);
  TORCH_CHECK(split_k == 8 || split_k == 16 || split_k == 32,
              "NVFP4 QPN2 split_k must be 8, 16, or 32");
  TORCH_CHECK((input.size(1) / 16) % split_k == 0,
              "NVFP4 QPN2 K/16 must be divisible by split_k");
  TORCH_CHECK(accumulator_chains == 1 || accumulator_chains == 2,
              "NVFP4 QPN2 accumulator_chains must be 1 or 2");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* code_ptr = codes.data_ptr<uint8_t>();
  const auto* scale_ptr = scales.data_ptr<uint8_t>();
  const auto* input_ptr =
      reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* output_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const int n = static_cast<int>(out.size(1));
  const int k = static_cast<int>(input.size(1));
  const int m = static_cast<int>(input.size(0));

#define VLLM_LAUNCH_QPN2(ROWS, SPLIT, NACC)                                  \
  launch_qpn2<SPLIT, NACC, ROWS>(code_ptr, scale_ptr, input_ptr, output_ptr, \
                                 n, k, m, static_cast<float>(global_scale),  \
                                 stream)
  const bool native_two_tile = qpn2_m16_native_enabled(m);
  if (native_two_tile && split_k == 8 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN2(2, 8, 1);
  } else if (native_two_tile && split_k == 8) {
    VLLM_LAUNCH_QPN2(2, 8, 2);
  } else if (native_two_tile && split_k == 16 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN2(2, 16, 1);
  } else if (native_two_tile && split_k == 16) {
    VLLM_LAUNCH_QPN2(2, 16, 2);
  } else if (split_k == 8 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN2(1, 8, 1);
  } else if (split_k == 8) {
    VLLM_LAUNCH_QPN2(1, 8, 2);
  } else if (split_k == 16 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN2(1, 16, 1);
  } else if (split_k == 16) {
    VLLM_LAUNCH_QPN2(1, 16, 2);
  } else if (accumulator_chains == 1) {
    VLLM_LAUNCH_QPN2(1, 32, 1);
  } else {
    VLLM_LAUNCH_QPN2(1, 32, 2);
  }
#undef VLLM_LAUNCH_QPN2
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_qpn2_gated_sm70_out(torch::Tensor out, torch::Tensor input,
                               torch::Tensor codes, torch::Tensor scales,
                               double global_scale, int64_t split_k,
                               int64_t accumulator_chains) {
  check_qpn2_tensors(out, input, codes, scales, true);
  TORCH_CHECK(split_k == 8 || split_k == 16,
              "NVFP4 QPN2 gated split_k must be 8 or 16");
  TORCH_CHECK((input.size(1) / 16) % split_k == 0,
              "NVFP4 QPN2 gated K/16 must be divisible by split_k");
  TORCH_CHECK(accumulator_chains == 1 || accumulator_chains == 2,
              "NVFP4 QPN2 gated accumulator_chains must be 1 or 2");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* code_ptr = codes.data_ptr<uint8_t>();
  const auto* scale_ptr = scales.data_ptr<uint8_t>();
  const auto* input_ptr =
      reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* output_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const int hidden = static_cast<int>(out.size(1));
  const int k = static_cast<int>(input.size(1));
  const int m = static_cast<int>(input.size(0));

#define VLLM_LAUNCH_QPN2_GATED(ROWS, SPLIT, NACC)               \
  launch_qpn2_gated<SPLIT, NACC, ROWS>(                         \
      code_ptr, scale_ptr, input_ptr, output_ptr, hidden, k, m, \
      static_cast<float>(global_scale), stream)
  const bool native_two_tile = qpn2_m16_native_enabled(m);
  if (native_two_tile && split_k == 8 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN2_GATED(2, 8, 1);
  } else if (native_two_tile && split_k == 8) {
    VLLM_LAUNCH_QPN2_GATED(2, 8, 2);
  } else if (split_k == 8 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN2_GATED(1, 8, 1);
  } else if (split_k == 8) {
    VLLM_LAUNCH_QPN2_GATED(1, 8, 2);
  } else if (accumulator_chains == 1) {
    VLLM_LAUNCH_QPN2_GATED(1, 16, 1);
  } else {
    VLLM_LAUNCH_QPN2_GATED(1, 16, 2);
  }
#undef VLLM_LAUNCH_QPN2_GATED
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifndef VLLM_NVFP4_QPN2_STANDALONE
void nvfp4_qpn2_dispatch_sm70_out(torch::Tensor out, torch::Tensor input,
                                  torch::Tensor codes, torch::Tensor scales,
                                  double global_scale, int64_t split_k,
                                  int64_t accumulator_chains,
                                  torch::Tensor tm_weight,
                                  torch::Tensor tm_scales,
                                  int64_t tm_group_size, int64_t tm_k_ld,
                                  int64_t tm_q_ld, bool gated_silu) {
  if (input.size(0) <= kQpn2DispatchMaxRows) {
    if (gated_silu) {
      nvfp4_qpn2_gated_sm70_out(out, input, codes, scales, global_scale,
                                split_k, accumulator_chains);
    } else {
      nvfp4_qpn2_gemm_sm70_out(out, input, codes, scales, global_scale, split_k,
                               accumulator_chains);
    }
    return;
  }

  if (!gated_silu) {
    nvfp4_gemm_sm70_out(out, input, tm_weight, tm_scales, tm_group_size,
                        tm_k_ld, tm_q_ld, false);
    return;
  }
  auto gate_up =
      torch::empty({input.size(0), out.size(1) * 2}, input.options());
  nvfp4_gemm_sm70_out(gate_up, input, tm_weight, tm_scales, tm_group_size,
                      tm_k_ld, tm_q_ld, false);
  silu_and_mul(out, gate_up);
}
#endif

#if defined(VLLM_NVFP4_QPN2_STANDALONE) && \
    !defined(VLLM_NVFP4_QPN2_BENCHMARK_CANDIDATE)
// Compile the exact production kernels as a task-local operator-race library
// before paying for a complete vLLM rebuild. Production registers these
// operators centrally in torch_bindings.cpp.
TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "nvfp4_qpn2_prepare_sm70(Tensor weight_packed, Tensor weight_scale) -> "
      "Tensor[]");
  ops.impl("nvfp4_qpn2_prepare_sm70", torch::kCUDA, &nvfp4_qpn2_prepare_sm70);
  ops.def(
      "nvfp4_qpn2_gemm_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor scales, float global_scale, int split_k, "
      "int accumulator_chains) -> ()");
  ops.impl("nvfp4_qpn2_gemm_sm70_out", torch::kCUDA, &nvfp4_qpn2_gemm_sm70_out);
  ops.def(
      "nvfp4_qpn2_gated_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor scales, float global_scale, int split_k, "
      "int accumulator_chains) -> ()");
  ops.impl("nvfp4_qpn2_gated_sm70_out", torch::kCUDA,
           &nvfp4_qpn2_gated_sm70_out);
}
#endif

#ifdef VLLM_NVFP4_QPN2_BENCHMARK_CANDIDATE
// Register a private namespace so the extended-M candidate can race the
// installed production operators without replacing vllm._C.
TORCH_LIBRARY_FRAGMENT(_qpn2_candidate, ops) {
  ops.def("prepare(Tensor weight_packed, Tensor weight_scale) -> Tensor[]");
  ops.impl("prepare", torch::kCUDA, &nvfp4_qpn2_prepare_sm70);
  ops.def(
      "gemm(Tensor(a!) out, Tensor input, Tensor codes, Tensor scales, "
      "float global_scale, int split_k, int accumulator_chains) -> ()");
  ops.impl("gemm", torch::kCUDA, &nvfp4_qpn2_gemm_sm70_out);
  ops.def(
      "gated(Tensor(a!) out, Tensor input, Tensor codes, Tensor scales, "
      "float global_scale, int split_k, int accumulator_chains) -> ()");
  ops.impl("gated", torch::kCUDA, &nvfp4_qpn2_gated_sm70_out);
}
#endif
