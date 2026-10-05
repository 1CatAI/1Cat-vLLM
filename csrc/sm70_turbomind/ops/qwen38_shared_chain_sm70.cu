// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Single cooperative launch with 80 resident CTAs; no spinning or float
// atomics.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
#define SHARED_MMA(C, A0, A1, B0, B1)                               \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

__global__ __launch_bounds__(256, 1) void shared_chain(
    const half* x, const half* up, const half* down, const half* gate,
    float* partial, float* gate_logits, half* out, int m) {
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int tile = blockIdx.x % 10, split = blockIdx.x / 10;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3, col = quad * 8 + r;
  __shared__ float reduction[8][8][32];
  // Pad rows to keep vector loads aligned and distribute MMA rows over banks.
  __shared__ __align__(16) half activated[8][168];
  float acc[8] = {};
  // Each K320 partition is further divided among eight warps; all weights
  // are read once across the grid and reused for every token in this batch.
  const int begin = split * 20 + warp * 20 / 8;
  const int end = split * 20 + (warp + 1) * 20 / 8;
  for (int g = begin; g < end; ++g) {
    const half* w = up + (tile * 160 + g) * 512 + col * 8;
    const uint4 lo = *reinterpret_cast<const uint4*>(w);
    const uint4 hi = *reinterpret_cast<const uint4*>(w + 256);
    uint4 a = {}, b = {};
    if (r < m) {
      a = *reinterpret_cast<const uint4*>(x + r * 2560 + g * 16);
      b = *reinterpret_cast<const uint4*>(x + r * 2560 + g * 16 + 8);
    }
    SHARED_MMA(acc, a.x, a.y, lo.x, lo.y);
    SHARED_MMA(acc, a.z, a.w, lo.z, lo.w);
    SHARED_MMA(acc, b.x, b.y, hi.x, hi.y);
    SHARED_MMA(acc, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int cc =
        quad * 8 + (i & 1) + (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
    reduction[warp][rr][cc] = acc[i];
  }
  __syncthreads();
  if (t < m * 32) {
    const int rr = t / 32, cc = t % 32;
    float sum = reduction[0][rr][cc];
#pragma unroll
    for (int w = 1; w < 8; ++w) sum += reduction[w][rr][cc];
    partial[(split * m + rr) * 320 + tile * 32 + cc] = sum;
  }
  if (blockIdx.x == 0 && warp < m) {
    float sum = 0.0f;
    for (int k = lane; k < 2560; k += 32)
      sum = fmaf(__half2float(x[warp * 2560 + k]), __half2float(gate[k]), sum);
#pragma unroll
    for (int offset = 16; offset; offset /= 2)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) gate_logits[warp] = sum;
  }
  // Cooperative launch proves simultaneous residency. CUDA's grid barrier
  // publishes every K partition and the shared gate before down reads them.
  cooperative_groups::this_grid().sync();
  for (int index = t; index < m * 160; index += 256) {
    const int row = index / 160, c = index % 160;
    float g = partial[row * 320 + c], u = partial[row * 320 + c + 160];
#pragma unroll
    for (int s = 1; s < 8; ++s) {
      g += partial[(s * m + row) * 320 + c];
      u += partial[(s * m + row) * 320 + c + 160];
    }
    const float half_gate = __half2float(__float2half_rn(g));
    activated[row][c] =
        __hmul(__float2half_rn(half_gate / (1.0f + expf(-half_gate))),
               __float2half_rn(u));
  }
  __syncthreads();
  float down_acc[8] = {};
  const int begin_down = warp * 10 / 8;
  const int end_down = (warp + 1) * 10 / 8;
  for (int g = begin_down; g < end_down; ++g) {
    const half* w = down + (blockIdx.x * 10 + g) * 512 + col * 8;
    const uint4 lo = *reinterpret_cast<const uint4*>(w);
    const uint4 hi = *reinterpret_cast<const uint4*>(w + 256);
    uint4 a = {}, b = {};
    if (r < m) {
      a = *reinterpret_cast<const uint4*>(activated[r] + g * 16);
      b = *reinterpret_cast<const uint4*>(activated[r] + g * 16 + 8);
    }
    SHARED_MMA(down_acc, a.x, a.y, lo.x, lo.y);
    SHARED_MMA(down_acc, a.z, a.w, lo.z, lo.w);
    SHARED_MMA(down_acc, b.x, b.y, hi.x, hi.y);
    SHARED_MMA(down_acc, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int cc =
        quad * 8 + (i & 1) + (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
    reduction[warp][rr][cc] = down_acc[i];
  }
  __syncthreads();
  const int output_col = blockIdx.x * 32 + lane;
  if (warp < m) {
    float sum = reduction[0][warp][lane];
#pragma unroll
    for (int w = 1; w < 8; ++w) sum += reduction[w][warp][lane];
    const float g = __half2float(__float2half_rn(gate_logits[warp]));
    const half factor = __float2half_rn(1.0f / (1.0f + expf(-g)));
    out[warp * 2560 + output_col] = __hmul(__float2half_rn(sum), factor);
  }
}
#undef SHARED_MMA

void run(torch::Tensor out, torch::Tensor partial, torch::Tensor gate_logits,
         torch::Tensor x, torch::Tensor up, torch::Tensor down,
         torch::Tensor gate) {
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(
      props->major == 7 && props->minor == 0 && props->cooperativeLaunch,
      "Cooperative SM70 device required");
  for (const auto& t : {out, partial, gate_logits, x, up, down, gate})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  for (const auto& t : {out, x, up, down, gate})
    TORCH_CHECK(t.scalar_type() == at::kHalf);
  TORCH_CHECK(partial.scalar_type() == at::kFloat &&
              gate_logits.scalar_type() == at::kFloat);
  int m = x.size(0);
  TORCH_CHECK(x.dim() == 2 && m >= 1 && m <= 8 && x.size(1) == 2560 &&
              out.sizes() == x.sizes() &&
              up.sizes() == at::IntArrayRef({10, 160, 2, 32, 8}) &&
              down.sizes() == at::IntArrayRef({80, 10, 2, 32, 8}) &&
              gate.numel() == 2560 &&
              partial.sizes() == at::IntArrayRef({8, m, 320}) &&
              gate_logits.numel() == m);
  int resident_blocks = 0;
  AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resident_blocks, shared_chain, 256, 0));
  TORCH_CHECK(resident_blocks * props->multiProcessorCount >= 80,
              "Shared chain requires residency for all 80 CTAs");
  const auto* xp = reinterpret_cast<const half*>(x.data_ptr());
  const auto* up_ptr = reinterpret_cast<const half*>(up.data_ptr());
  const auto* down_ptr = reinterpret_cast<const half*>(down.data_ptr());
  const auto* gate_ptr = reinterpret_cast<const half*>(gate.data_ptr());
  auto* p = partial.data_ptr<float>();
  auto* g = gate_logits.data_ptr<float>();
  auto* output = reinterpret_cast<half*>(out.data_ptr());
  void* args[] = {&xp, &up_ptr, &down_ptr, &gate_ptr, &p, &g, &output, &m};
  AT_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(shared_chain), dim3(80), dim3(256), args, 0,
      at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "qwen38_shared_chain_sm70_out(Tensor(a!) out, Tensor(b!) partial, "
      "Tensor(c!) gate_logits, Tensor x, Tensor up, Tensor down, Tensor gate) "
      "-> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen38_shared_chain_sm70_out", &run);
}
