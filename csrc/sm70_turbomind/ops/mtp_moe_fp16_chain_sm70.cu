// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// FP16 draft expert chains with reassociated FP32 sums, no persistent grid.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
#define MTP_MMA(C, A0, A1, B0, B1)                                  \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

__global__ __launch_bounds__(256, 1) void w13_silu(const half* x,
                                                   const half* weights,
                                                   const int* ids,
                                                   half* activation) {
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int quad = (lane >> 2) & 3, inner_split = quad >> 1;
  const int split = warp * 2 + inner_split;
  const int route = blockIdx.y, row = route / 10, expert = ids[route];
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int col = blockIdx.x * 16 + (quad & 1) * 8 + r;
  __shared__ float partials[2][16][16];
  float gate_acc[8] = {}, up_acc[8] = {};
  if (expert >= 0 && expert < 512) {
#pragma unroll 1
    for (int g = 0; g < 10; ++g) {
      const int k = split * 160 + g * 16;
      const half* input = x + row * 2560 + k;
      const uint4 a = *reinterpret_cast<const uint4*>(input);
      const uint4 b = *reinterpret_cast<const uint4*>(input + 8);
      const half* w = weights + (int64_t(expert) * 320 + col) * 2560 + k;
      const uint4 g0 = *reinterpret_cast<const uint4*>(w);
      const uint4 g1 = *reinterpret_cast<const uint4*>(w + 8);
      const uint4 u0 = *reinterpret_cast<const uint4*>(w + 160 * 2560);
      const uint4 u1 = *reinterpret_cast<const uint4*>(w + 160 * 2560 + 8);
      MTP_MMA(gate_acc, a.x, a.y, g0.x, g0.y);
      MTP_MMA(gate_acc, a.z, a.w, g0.z, g0.w);
      MTP_MMA(gate_acc, b.x, b.y, g1.x, g1.y);
      MTP_MMA(gate_acc, b.z, b.w, g1.z, g1.w);
      MTP_MMA(up_acc, a.x, a.y, u0.x, u0.y);
      MTP_MMA(up_acc, a.z, a.w, u0.z, u0.w);
      MTP_MMA(up_acc, b.x, b.y, u1.x, u1.y);
      MTP_MMA(up_acc, b.z, b.w, u1.z, u1.w);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int cc =
        (quad & 1) * 8 + (i & 1) + (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
    if (rr == 0) {
      partials[0][split][cc] = gate_acc[i];
      partials[1][split][cc] = up_acc[i];
    }
  }
  __syncthreads();
  if (t < 16) {
    float gate = partials[0][0][t], up = partials[1][0][t];
#pragma unroll
    for (int s = 1; s < 16; ++s) {
      gate += partials[0][s][t];
      up += partials[1][s][t];
    }
    const float g = __half2float(__float2half_rn(gate));
    activation[route * 160 + blockIdx.x * 16 + t] =
        __hmul(__float2half_rn(g / (1.0f + expf(-g))), __float2half_rn(up));
  }
}
#undef MTP_MMA

__global__ __launch_bounds__(320, 1) void w2_weighted_sum(
    const half* activation, const half* weights, const int* ids,
    const float* probabilities, half* output) {
  const int lane = threadIdx.x % 32, route = threadIdx.x / 32;
  const int row = blockIdx.y, c = blockIdx.x * 32 + lane;
  const int expert = ids[row * 10 + route];
  __shared__ half input[10][160], partial[10][32], tile[10][32][18];
  for (int k = lane; k < 160; k += 32)
    input[route][k] = activation[(row * 10 + route) * 160 + k];
  __syncthreads();
  float accum = 0.0f;
  if (expert >= 0 && expert < 512) {
    const half* w = weights + (int64_t(expert) * 2560 + c) * 160;
    for (int base = 0; base < 160; base += 16) {
      // Pair adjacent 128-bit vectors of each output row. Scalar strided
      // half loads issued one sector per lane for only two useful bytes.
      for (int i = lane; i < 64; i += 32) {
        const int n = i / 2, k = (i % 2) * 8;
        union {
          uint4 vector;
          half elements[8];
        } data;
        data.vector =
            *reinterpret_cast<const uint4*>(w + (n - lane) * 160 + base + k);
#pragma unroll
        for (int j = 0; j < 8; ++j) tile[route][n][k + j] = data.elements[j];
      }
      __syncwarp();
#pragma unroll
      for (int k = 0; k < 16; ++k)
        accum = fmaf(__half2float(input[route][base + k]),
                     __half2float(tile[route][lane][k]), accum);
      __syncwarp();
    }
  }
  // Preserve the old FP16 boundary after route weighting. The final route
  // reduction is deterministic, with one owner for each output element.
  partial[route][lane] =
      __float2half_rn(accum * probabilities[row * 10 + route]);
  __syncthreads();
  if (route == 0) {
    float sum = 0.0f;
#pragma unroll
    for (int r = 0; r < 10; ++r) sum += __half2float(partial[r][lane]);
    output[row * 2560 + c] = __float2half_rn(sum);
  }
}

void run(torch::Tensor output, torch::Tensor activation, torch::Tensor x,
         torch::Tensor w13, torch::Tensor w2, torch::Tensor ids,
         torch::Tensor probabilities, int64_t stage) {
  TORCH_CHECK(stage >= 0 && stage <= 2,
              "Stage must be chain(0), up(1), down(2)");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0);
  for (const auto& t : {output, activation, x, w13, w2, ids, probabilities})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  for (const auto& t : {output, activation, x, w13, w2})
    TORCH_CHECK(t.scalar_type() == at::kHalf);
  TORCH_CHECK(ids.scalar_type() == at::kInt &&
              probabilities.scalar_type() == at::kFloat);
  const int m = x.size(0);
  TORCH_CHECK(x.dim() == 2 && (m == 1 || m == 5) && x.size(1) == 2560 &&
              w13.sizes() == at::IntArrayRef({512, 320, 2560}) &&
              w2.sizes() == at::IntArrayRef({512, 2560, 160}) &&
              output.sizes() == x.sizes() &&
              activation.sizes() == at::IntArrayRef({m * 10, 160}) &&
              ids.sizes() == at::IntArrayRef({m, 10}) &&
              probabilities.sizes() == ids.sizes());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (stage != 2)
    w13_silu<<<dim3(10, m * 10), 256, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(w13.data_ptr()), ids.data_ptr<int>(),
        reinterpret_cast<half*>(activation.data_ptr()));
  if (stage != 1)
    w2_weighted_sum<<<dim3(80, m), 320, 0, stream>>>(
        reinterpret_cast<const half*>(activation.data_ptr()),
        reinterpret_cast<const half*>(w2.data_ptr()), ids.data_ptr<int>(),
        probabilities.data_ptr<float>(),
        reinterpret_cast<half*>(output.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "sm70_mtp_moe_fp16_chain_out(Tensor(a!) out, Tensor(b!) activation, "
      "Tensor x, Tensor w13, Tensor w2, Tensor ids, Tensor probabilities, int "
      "stage=0) -> "
      "()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) { m.impl("sm70_mtp_moe_fp16_chain_out", &run); }
