// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "activation_pack_sm70.cuh"

namespace vllm::sm70 {

// Where a QKVZ projection writes its z tail and b/a when both run in one launch.
struct Sm70FusedBa {
  half* z = nullptr;
  const half* weight = nullptr;
  half* b = nullptr;
  half* a = nullptr;
  int qkv_n = 0;
  int ba_n = 0;
};

#define VLLM_QPN_BA_MMA(C, A0, A1, B0, B1)                          \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// One appended CTA of a QKVZ launch: eight rows of the short dense FP16 b/a
// projection over the K16 activations, using the launch's shared scratch.
template <int Warps>
__device__ __forceinline__ void fused_ba_tile(const half* input,
                                              const Sm70FusedBa& ba,
                                              float* partial, int first_row,
                                              int k, int m) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int quadpair = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row = first_row + local_row;
  const int column = quadpair * 8 + local_row;
  const int per_warp = (k >> 4) / Warps;
  float accum[2][8] = {};
  for (int group = warp * per_warp; group < (warp + 1) * per_warp; ++group) {
    uint4 a01 = make_uint4(0, 0, 0, 0), a23 = a01, b01 = a01, b23 = a01;
    if (row < m) {
      const half* a = input + (static_cast<size_t>(group) * m + row) * 16;
      a01 = *reinterpret_cast<const uint4*>(a);
      a23 = *reinterpret_cast<const uint4*>(a + 8);
    }
    if (column < ba.ba_n) {
      const half* b = ba.weight + static_cast<size_t>(column) * k + group * 16;
      b01 = *reinterpret_cast<const uint4*>(b);
      b23 = *reinterpret_cast<const uint4*>(b + 8);
    }
    const unsigned* a0 = reinterpret_cast<const unsigned*>(&a01);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&a23);
    const unsigned* b0 = reinterpret_cast<const unsigned*>(&b01);
    const unsigned* b1 = reinterpret_cast<const unsigned*>(&b23);
    VLLM_QPN_BA_MMA(accum[0], a0[0], a0[1], b0[0], b0[1]);
    VLLM_QPN_BA_MMA(accum[1], a0[2], a0[3], b0[2], b0[3]);
    VLLM_QPN_BA_MMA(accum[0], a1[0], a1[1], b1[0], b1[1]);
    VLLM_QPN_BA_MMA(accum[1], a1[2], a1[3], b1[2], b1[3]);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    partial[warp * 256 + r * 32 + quadpair * 8 + c] = accum[0][i] + accum[1][i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 256; e += blockDim.x) {
    const int out_row = first_row + e / 32;
    const int out_col = e % 32;
    if (out_row >= m || out_col >= ba.ba_n) continue;
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < Warps; ++w) sum += partial[w * 256 + e];
    const int half_n = ba.ba_n / 2;
    if (out_col < half_n) {
      ba.b[static_cast<size_t>(out_row) * half_n + out_col] = __float2half(sum);
    } else {
      ba.a[static_cast<size_t>(out_row) * half_n + out_col - half_n] =
          __float2half(sum);
    }
  }
}

// Stores one QKVZ element into the contiguous qkv or z output.
__device__ __forceinline__ void fused_ba_store(half* qkv, const Sm70FusedBa& ba,
                                               int n, int row, int column,
                                               half value) {
  if (column < ba.qkv_n) {
    qkv[static_cast<size_t>(row) * ba.qkv_n + column] = value;
  } else {
    ba.z[static_cast<size_t>(row) * (n - ba.qkv_n) + column - ba.qkv_n] = value;
  }
}

// Two adjacent projections share each activation fragment. The reader consumes
// the existing compressed layout; no persistent weight copy is required.
template <class Reader, int Split, bool Gated, bool FusedBA = false>
__global__ void qpn_pair_m16_kernel(const uint8_t* __restrict__ codes,
                                    const void* __restrict__ scales,
                                    const half* __restrict__ input,
                                    half* __restrict__ output,
                                    half* __restrict__ k16_output, int width,
                                    int k, int m, float global_scale,
                                    Sm70FusedBa fused_ba = {}) {
  constexpr int kRows = 2;
  constexpr int kElements = kRows * 256;
  extern __shared__ __align__(16) unsigned char storage[];
  auto& partial = *reinterpret_cast<float (*)[2][Split][kElements]>(storage);
  if constexpr (FusedBA) {
    const int pairs = width / 64;
    if (blockIdx.x >= pairs) {
      fused_ba_tile<Split>(input, fused_ba, reinterpret_cast<float*>(storage),
                           (blockIdx.x - pairs) * 8, k, m);
      return;
    }
  }
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
      if constexpr (FusedBA) {
        fused_ba_store(output, fused_ba, width, row, tile0 * 32 + e % 32,
                       result);
        fused_ba_store(output, fused_ba, width, row, tile1 * 32 + e % 32,
                       __float2half(up));
      } else {
        output[static_cast<size_t>(row) * width + tile0 * 32 + e % 32] = result;
        if constexpr (Gated) {
          if (k16_output != nullptr) {
            k16_output[k16_half_index(row, tile0 * 32 + e % 32, m)] = result;
          }
        }
        if constexpr (!Gated) {
          output[static_cast<size_t>(row) * width + tile1 * 32 + e % 32] =
              __float2half(up);
        }
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
  torch::Tensor packed, twin;
  auto* a = k16_packed_input(input, packed, stream);
  half* k16_output = Gated ? k16_twin_for(out, twin) : nullptr;
  qpn_pair_m16_kernel<Reader, Split, Gated>
      <<<out.size(1) / (Gated ? 32 : 64), 32 * Split, kShared, stream>>>(
          reinterpret_cast<const uint8_t*>(codes.data_ptr()), scales.data_ptr(),
          a, reinterpret_cast<half*>(out.data_ptr<at::Half>()), k16_output,
          out.size(1), input.size(1), input.size(0), global_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (k16_output != nullptr) k16_publish(out, twin);
}

// QKVZ pairs plus b/a in one launch. `out` is qkv; the z tail goes to fused_ba.z.
template <class Reader, int Split>
void launch_qpn_pair_m16_fused_ba(torch::Tensor out, torch::Tensor input,
                                  torch::Tensor codes, torch::Tensor scales,
                                  int width, const Sm70FusedBa& fused_ba,
                                  cudaStream_t stream) {
  constexpr int kShared = 2 * Split * 512 * sizeof(float);
  auto kernel = qpn_pair_m16_kernel<Reader, Split, false, true>;
  if constexpr (kShared > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared));
  }
  torch::Tensor packed;
  auto* a = k16_packed_input(input, packed, stream);
  const int m = static_cast<int>(input.size(0));
  kernel<<<width / 64 + (m + 7) / 8, 32 * Split, kShared, stream>>>(
      reinterpret_cast<const uint8_t*>(codes.data_ptr()), scales.data_ptr(), a,
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), nullptr, width,
      static_cast<int>(input.size(1)), m, 1.0f, fused_ba);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::sm70
