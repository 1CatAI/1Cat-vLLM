// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Benchmark-only grouped draft QPN8: existing channel packs and decoder,
// fused up/SiLU and down/route sum. No persistent grid or float atomics.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>

#include "fp8_qpn8_decode_sm70.cuh"

namespace {
#define DRAFT_MMA(C, A0, A1, B0, B1)                                \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <bool Integer>
__device__ __forceinline__ void load_weights(const uint8_t* codes, half scale,
                                             half2* weights) {
  const uint4 q = *reinterpret_cast<const uint4*>(codes);
  if constexpr (Integer) {
    const auto* bytes = reinterpret_cast<const int8_t*>(&q);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int first = (i / 4) * 8 + i % 4;
      weights[i] = __halves2half2(__int2half_rn(int(bytes[first])),
                                  __int2half_rn(int(bytes[first + 4])));
    }
  } else {
    fp8x8_to_half2x4_fast(make_uint2(q.x, q.y), weights);
    fp8x8_to_half2x4_fast(make_uint2(q.z, q.w), weights + 4);
  }
  const half2 s = __halves2half2(scale, scale);
#pragma unroll
  for (int i = 0; i < 8; ++i) weights[i] = __hmul2(weights[i], s);
}

__device__ __forceinline__ void accumulate(float* c, uint4 a, uint4 b,
                                           const half2* weights) {
  const auto* w = reinterpret_cast<const unsigned*>(weights);
  DRAFT_MMA(c, a.x, a.y, w[0], w[1]);
  DRAFT_MMA(c, a.z, a.w, w[2], w[3]);
  DRAFT_MMA(c, b.x, b.y, w[4], w[5]);
  DRAFT_MMA(c, b.z, b.w, w[6], w[7]);
}

template <bool Integer>
__global__ __launch_bounds__(256, 1) void draft_qpn8_up(const half* x,
                                                        const uint8_t* codes,
                                                        const half* scales,
                                                        const int* ids,
                                                        half* activation) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int route = blockIdx.y, expert = ids[route];
  const int quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int col = quad * 8 + r;
  __shared__ float sums[2][8][32];
  float gate[8] = {}, up[8] = {};
  if (expert >= 0 && expert < 512) {
    const half gs = scales[expert * 320 + blockIdx.x * 32 + col];
    const half us = scales[expert * 320 + (blockIdx.x + 5) * 32 + col];
    const int64_t offset = int64_t(expert) * 320 * 2560;
#pragma unroll 1
    for (int g = warp * 20; g < (warp + 1) * 20; ++g) {
      const half* input = x + (route / 10) * 2560 + g * 16;
      const uint4 a = *reinterpret_cast<const uint4*>(input);
      const uint4 b = *reinterpret_cast<const uint4*>(input + 8);
      half2 wg[8], wu[8];
      load_weights<Integer>(
          codes + offset + (blockIdx.x * 160 + g) * 512 + lane * 16, gs, wg);
      load_weights<Integer>(
          codes + offset + ((blockIdx.x + 5) * 160 + g) * 512 + lane * 16, us,
          wu);
      accumulate(gate, a, b, wg);
      accumulate(up, a, b, wu);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int cc =
        quad * 8 + (i & 1) + (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
    if (rr == 0) {
      sums[0][warp][cc] = gate[i];
      sums[1][warp][cc] = up[i];
    }
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    float g = sums[0][0][lane], u = sums[1][0][lane];
#pragma unroll
    for (int s = 1; s < 8; ++s) {
      g += sums[0][s][lane];
      u += sums[1][s][lane];
    }
    g = __half2float(__float2half_rn(g));
    activation[route * 160 + blockIdx.x * 32 + lane] =
        __hmul(__float2half_rn(g / (1.0f + expf(-g))), __float2half_rn(u));
  }
}

template <bool Integer>
__global__ __launch_bounds__(320, 1) void draft_qpn8_down(
    const half* activation, const uint8_t* codes, const half* scales,
    const int* ids, const float* probabilities, half* output) {
  const int lane = threadIdx.x % 32, route = threadIdx.x / 32;
  const int row = blockIdx.y, expert = ids[row * 10 + route];
  const int quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  __shared__ half partial[10][32];
  float accum[8] = {};
  if (expert >= 0 && expert < 512) {
    const int64_t offset = int64_t(expert) * 2560 * 160;
    const half s = scales[expert * 2560 + blockIdx.x * 32 + quad * 8 + r];
#pragma unroll 1
    for (int g = 0; g < 10; ++g) {
      const half* input = activation + (row * 10 + route) * 160 + g * 16;
      const uint4 a = *reinterpret_cast<const uint4*>(input);
      const uint4 b = *reinterpret_cast<const uint4*>(input + 8);
      half2 w[8];
      load_weights<Integer>(
          codes + offset + (blockIdx.x * 10 + g) * 512 + lane * 16, s, w);
      accumulate(accum, a, b, w);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int cc =
        quad * 8 + (i & 1) + (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
    if (rr == 0)
      partial[route][cc] =
          __float2half_rn(accum[i] * probabilities[row * 10 + route]);
  }
  __syncthreads();
  if (route == 0) {
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 10; ++j) sum += __half2float(partial[j][lane]);
    output[row * 2560 + blockIdx.x * 32 + lane] = __float2half_rn(sum);
  }
}
#undef DRAFT_MMA

template <bool Integer>
void run(torch::Tensor output, torch::Tensor activation, torch::Tensor x,
         torch::Tensor w13, torch::Tensor s13, torch::Tensor w2,
         torch::Tensor s2, torch::Tensor ids, torch::Tensor probabilities) {
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0);
  for (const auto& t :
       {output, activation, x, w13, s13, w2, s2, ids, probabilities})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  for (const auto& t : {output, activation, x, s13, s2})
    TORCH_CHECK(t.scalar_type() == at::kHalf);
  TORCH_CHECK(w13.scalar_type() == at::kByte && w2.scalar_type() == at::kByte &&
              ids.scalar_type() == at::kInt &&
              probabilities.scalar_type() == at::kFloat);
  const int m = x.size(0);
  TORCH_CHECK(x.dim() == 2 && (m == 1 || m == 5) && x.size(1) == 2560 &&
              output.sizes() == x.sizes() &&
              activation.sizes() == at::IntArrayRef({m * 10, 160}) &&
              ids.sizes() == at::IntArrayRef({m, 10}) &&
              probabilities.sizes() == ids.sizes() &&
              w13.numel() == int64_t(512) * 320 * 2560 &&
              w2.numel() == int64_t(512) * 2560 * 160 &&
              s13.sizes() == at::IntArrayRef({512, 320}) &&
              s2.sizes() == at::IntArrayRef({512, 2560}));
  const auto stream = at::cuda::getCurrentCUDAStream();
  draft_qpn8_up<Integer><<<dim3(5, m * 10), 256, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()), w13.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(s13.data_ptr()), ids.data_ptr<int>(),
      reinterpret_cast<half*>(activation.data_ptr()));
  draft_qpn8_down<Integer><<<dim3(80, m), 320, 0, stream>>>(
      reinterpret_cast<const half*>(activation.data_ptr()),
      w2.data_ptr<uint8_t>(), reinterpret_cast<const half*>(s2.data_ptr()),
      ids.data_ptr<int>(), probabilities.data_ptr<float>(),
      reinterpret_cast<half*>(output.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "sm70_mtp_moe_qpn8_chain_out(Tensor(a!) out, Tensor(b!) activation, "
      "Tensor x, Tensor w13, Tensor s13, Tensor w2, Tensor s2, Tensor ids, "
      "Tensor probabilities) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sm70_mtp_moe_qpn8_chain_out", &run<false>);
}

// The INT8 candidate uses the same byte layout and FP16 MMA operands. It is
// isolated from serving; model distribution and acceptance admission is
// pending.
TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "sm70_mtp_moe_int8_chain_out(Tensor(a!) output, "
      "Tensor(b!) activation, Tensor x, Tensor codes13, Tensor scales13, "
      "Tensor codes2, Tensor scales2, Tensor ids, Tensor probabilities) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sm70_mtp_moe_int8_chain_out", &run<true>);
}
