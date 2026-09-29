// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "activation_pack_sm70.cuh"

namespace vllm::sm70 {

// Two adjacent projections share each activation fragment. The reader consumes
// the existing compressed layout; no persistent weight copy is required.
template <class Reader, int Split, bool Gated>
__global__ void qpn_pair_m16_kernel(const uint8_t* __restrict__ codes,
                                    const void* __restrict__ scales,
                                    const half* __restrict__ input,
                                    half* __restrict__ output, int width, int k,
                                    int m, float global_scale) {
  constexpr int kRows = 2;
  constexpr int kElements = kRows * 256;
  extern __shared__ __align__(16) unsigned char storage[];
  auto& partial = *reinterpret_cast<float (*)[2][Split][kElements]>(storage);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int qp = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int tile0 = Gated ? blockIdx.x : 2 * blockIdx.x;
  const int tile1 = Gated ? blockIdx.x + width / 32 : 2 * blockIdx.x + 1;
  const int groups = k / 16;
  const int per_warp = groups / Split;
  Reader readers[2] = {
      Reader(codes, scales, tile0, groups, lane, global_scale),
      Reader(codes, scales, tile1, groups, lane, global_scale)};
  float accum[2][kRows][2][8] = {};
#pragma unroll 1
  for (int group = warp * per_warp; group < (warp + 1) * per_warp; ++group) {
    half2 weights[2][8];
#pragma unroll
    for (int p = 0; p < 2; ++p) readers[p].load(group, weights[p]);
#pragma unroll
    for (int row_tile = 0; row_tile < kRows; ++row_tile) {
      uint4 a01 = make_uint4(0, 0, 0, 0);
      uint4 a23 = make_uint4(0, 0, 0, 0);
      const int row = row_tile * 8 + local_row;
      if (row < m) {
        const half* a = input + (static_cast<size_t>(group) * m + row) * 16;
        a01 = *reinterpret_cast<const uint4*>(a);
        a23 = *reinterpret_cast<const uint4*>(a + 8);
      }
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&a01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&a23);
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const unsigned* b = reinterpret_cast<const unsigned*>(weights[p]);
#define VLLM_QPN_PAIR_MMA(C, A0, A1, B0, B1)                        \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))
        VLLM_QPN_PAIR_MMA(accum[p][row_tile][0], a0[0], a0[1], b[0], b[1]);
        VLLM_QPN_PAIR_MMA(accum[p][row_tile][1], a0[2], a0[3], b[2], b[3]);
        VLLM_QPN_PAIR_MMA(accum[p][row_tile][0], a1[0], a1[1], b[4], b[5]);
        VLLM_QPN_PAIR_MMA(accum[p][row_tile][1], a1[2], a1[3], b[6], b[7]);
#undef VLLM_QPN_PAIR_MMA
      }
    }
  }
#pragma unroll
  for (int p = 0; p < 2; ++p) {
#pragma unroll
    for (int row_tile = 0; row_tile < kRows; ++row_tile) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int row =
            row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        partial[p][warp][row * 32 + qp * 8 + col] =
            accum[p][row_tile][0][i] + accum[p][row_tile][1][i];
      }
    }
  }
  __syncthreads();
  for (int e = threadIdx.x; e < kElements; e += blockDim.x) {
    float sum = 0.0f;
    float up = 0.0f;
#pragma unroll
    for (int w = 0; w < Split; ++w) {
      sum += partial[0][w][e];
      up += partial[1][w][e];
    }
    const int row = e / 32;
    if (row < m) {
      half result = __float2half(sum);
      if constexpr (Gated && Reader::kFp4) {
        // Preserve the FP4 gate/up intermediate FP16 rounding.
        const float gate = __half2float(result);
        result =
            __hmul(__float2half(gate / (1.0f + expf(-gate))), __float2half(up));
      } else if constexpr (Gated) {
        result = __float2half((sum / (1.0f + __expf(-sum))) * up);
      }
      output[static_cast<size_t>(row) * width + tile0 * 32 + e % 32] = result;
      if constexpr (!Gated) {
        output[static_cast<size_t>(row) * width + tile1 * 32 + e % 32] =
            __float2half(up);
      }
    }
  }
}

template <class Reader, int Split, bool Gated>
void launch_qpn_pair_m16(torch::Tensor out, torch::Tensor input,
                         torch::Tensor codes, torch::Tensor scales,
                         float global_scale, cudaStream_t stream) {
  constexpr int kShared = 2 * Split * 512 * sizeof(float);
  if constexpr (kShared > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        qpn_pair_m16_kernel<Reader, Split, Gated>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kShared));
  }
  auto packed = torch::empty_like(input);
  auto* a = reinterpret_cast<half*>(packed.data_ptr<at::Half>());
  pack_k16_input<<<(input.numel() / 2 + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()), a,
      input.size(0), input.size(1));
  qpn_pair_m16_kernel<Reader, Split, Gated>
      <<<out.size(1) / (Gated ? 32 : 64), 32 * Split, kShared, stream>>>(
          reinterpret_cast<const uint8_t*>(codes.data_ptr()), scales.data_ptr(),
          a, reinterpret_cast<half*>(out.data_ptr<at::Half>()), out.size(1),
          input.size(1), input.size(0), global_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::sm70
