// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research-only uint4 code loads; unchanged split, HMMA order and precision.
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "nvfp4_qpn2_layout.cuh"
namespace {
constexpr int kQpn2RowsPerCta = 8;
__device__ __forceinline__ half2 fp8e4m3_to_half2(uint8_t value) {
  const unsigned short bits =
      ((static_cast<unsigned short>(value) & 0x80u) << 8) |
      ((static_cast<unsigned short>(value) & 0x7fu) << 7);
  const half converted =
      __hmul(__ushort_as_half(bits), __ushort_as_half(0x5c00));
  return __halves2half2(converted, converted);
}

__device__ __forceinline__ half2 nvfp4_effective_scale(uint8_t value,
                                                       float global_scale) {
  // Match TurboMind's W4A16 weights: multiply the exact E4M3 group scale
  // by the FP32 global scale before rounding once to FP16. Rounding the
  // global factor first changes the model's weights on every decode step.
  const half raw = __low2half(fp8e4m3_to_half2(value));
  const half scaled =
      __float2half_rn(__fmul_rn(__half2float(raw), global_scale));
  return __halves2half2(scaled, scaled);
}

// QPN2 stores [N/32, K/16, lane], while TurboMind V/Pack1 stores
// [K/16, N] in logical column order. Preserve FP32 multiply then FP16 RNE.
__global__ void nvfp4_qpn2_restore_tm_scales_kernel(half* output,
                                                    const uint8_t* scales,
                                                    float global_scale, int n,
                                                    int groups) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n * groups) {
    return;
  }
  const int group = index / n;
  const int column = index % n;
  const int col = column % 32;
  const int lane = ((col & 24) >> 1) | (col & 3) | ((col & 4) << 2);
  const int source = ((column / 32) * groups + group) * 32 + lane;
  const half raw = __low2half(fp8e4m3_to_half2(scales[source]));
  output[index] = __float2half_rn(__fmul_rn(__half2float(raw), global_scale));
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
    // Undo the FP4 exponent bias on the code, not on its scale. Scaling
    // the scale by 2^14 first loses subnormals and can overflow even when
    // the final dequantized weights are finite.
    const half2 code = __hmul2(*reinterpret_cast<half2*>(&values[index]),
                               __float2half2_rn(16384.0f));
    output[index] = __hmul2(code, scale);
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

template <int SplitK, int NAcc, int RowTiles = 1, bool TurboMindLayout = true,
          bool VectorCodes = true>
__global__ void vector_gated_kernel(const uint8_t* __restrict__ codes,
                                    const uint8_t* __restrict__ group_scales,
                                    const half* __restrict__ input,
                                    half* __restrict__ output, int hidden,
                                    int k, int m, float global_scale) {
  static_assert(RowTiles == 1 || RowTiles == 2,
                "NVFP4 gated QPN2 supports one or two 8-row tiles");
  __shared__ float partials[2][SplitK][RowTiles * 256];

  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int projection = warp_in_block / SplitK;
  const int warp = warp_in_block - projection * SplitK;
  const int hidden_tiles = hidden >> 5;
  const int output_tile = TurboMindLayout ? blockIdx.y : blockIdx.x;
  const int tile = output_tile + projection * hidden_tiles;
  const int quadpair = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row_base =
      (TurboMindLayout ? blockIdx.x : blockIdx.y) * kQpn2RowsPerCta * RowTiles;
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const int group_begin = warp * groups_per_warp;
  const Nvfp4Qpn2CodeReader<TurboMindLayout> reader(codes, tile, groups_k16,
                                                    lane);
  const uint8_t* scale_ptr =
      group_scales + static_cast<size_t>(tile) * groups_k16 * 32 + lane;

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
  for (int pair = 0; pair < (groups_per_warp + 1) / 2; ++pair) {
    uint4 vector_codes = make_uint4(0, 0, 0, 0);
    if constexpr (VectorCodes) {
      const int pairs = (groups_per_warp + 1) / 2;
      const uint4* ptr = reinterpret_cast<const uint4*>(codes);
      vector_codes =
          __ldcs(ptr + ((tile * SplitK + warp) * pairs + pair) * 32 + lane);
    }
#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
      const int relative = pair * 2 + phase;
      if (relative >= groups_per_warp) continue;
      const int group = group_begin + relative;
      const uint2 packed =
          VectorCodes
              ? (phase == 0 ? make_uint2(vector_codes.x, vector_codes.y)
                            : make_uint2(vector_codes.z, vector_codes.w))
              : reader.load(group);
      const half2 scale = nvfp4_effective_scale(
          __ldg(scale_ptr + static_cast<size_t>(group) * 32), global_scale);
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
      output[static_cast<size_t>(output_row) * hidden + output_tile * 32 +
             output_col] = __hmul(silu, up_half);
    }
  }
}

__global__ void pack_codes(uint8_t* out, const uint8_t* input) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  constexpr size_t words = 272 * 8 * 20 * 32 * 4;
  if (index >= words) return;
  const int part = index % 4, lane = (index / 4) % 32;
  const int pair = (index / 128) % 20;
  const int warp = (index / (128 * 20)) % 8;
  const int tile = index / (128 * 20 * 8);
  const int group = warp * 40 + pair * 2 + part / 2;
  const int col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  reinterpret_cast<uint32_t*>(out)[index] = reinterpret_cast<const uint32_t*>(
      input)[tile * 320 * 64 + group * 64 + col + (part % 2) * 32];
}
}  // namespace
at::Tensor vector_gate_pack(at::Tensor codes) {
  TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == at::kByte &&
              codes.is_contiguous());
  TORCH_CHECK(codes.numel() == 8704LL * 5120 / 2);
  const c10::cuda::CUDAGuard guard(codes.device());
  auto out = at::empty_like(codes);
  pack_codes<<<(out.numel() / 4 + 255) / 256, 256, 0,
               at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<uint8_t>(),
                                                   codes.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
void vector_gate_out(at::Tensor output, at::Tensor input, at::Tensor codes,
                     at::Tensor scales, double global_scale, bool vector) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
              input.sizes() == at::IntArrayRef({8, 5120}) &&
              input.is_contiguous());
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kHalf &&
              output.sizes() == at::IntArrayRef({8, 4352}) &&
              output.is_contiguous());
  TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == at::kByte &&
              codes.is_contiguous() && codes.numel() == 8704LL * 5120 / 2);
  TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == at::kByte &&
              scales.is_contiguous() && scales.numel() == 8704LL * 5120 / 16);
  TORCH_CHECK(input.device() == output.device() &&
              input.device() == codes.device() &&
              input.device() == scales.device());
  const c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (vector)
    vector_gated_kernel<8, 2, 1, true, true><<<dim3(1, 136), 512, 0, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()), 4352, 5120, 8,
        static_cast<float>(global_scale));
  else
    vector_gated_kernel<8, 2, 1, true, false><<<dim3(1, 136), 512, 0, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()), 4352, 5120, 8,
        static_cast<float>(global_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY(_nvfp4_vector_gate_micro, m) {
  m.def("pack(Tensor codes) -> Tensor");
  m.def(
      "out(Tensor(a!) output, Tensor input, Tensor codes, Tensor scales, float "
      "global_scale, bool vector) -> ()");
}
TORCH_LIBRARY_IMPL(_nvfp4_vector_gate_micro, CUDA, m) {
  m.impl("pack", &vector_gate_pack);
  m.impl("out", &vector_gate_out);
}
