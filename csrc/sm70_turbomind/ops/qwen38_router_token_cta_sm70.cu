// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// One 16-warp CTA per token: FP32 projection and deterministic top10.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
#define ROUTER_MMA(C, A0, A1, B0, B1)                               \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

__device__ void max_pair(float& value, int& id, int offset) {
  const float other = __shfl_down_sync(0xffffffff, value, offset);
  const int other_id = __shfl_down_sync(0xffffffff, id, offset);
  if (other > value || (other == value && other_id < id)) {
    value = other;
    id = other_id;
  }
}

__global__ __launch_bounds__(512, 1) void router_token_cta(
    const half* x, const half* packed, half* logits, float* probabilities,
    int* ids, int* source_rows, int m) {
  const int token = blockIdx.x, t = threadIdx.x;
  const int warp = t / 32, lane = t % 32;
  const int split = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  __shared__ half projected[512];
  __shared__ float maxima[16], selected[10], denominator;
  __shared__ int maxima_ids[16], winners[10], winner;
  // Four N8 tiles per warp. The four quad pairs retain the existing K640
  // partitions; all input rows in the MMA repeat this CTA's single token.
  float accum[4][8] = {};
#pragma unroll 4
  for (int g = 0; g < 40; ++g) {
    const half* input = x + token * 2560 + split * 640 + g * 16;
    const uint4 a = *reinterpret_cast<const uint4*>(input);
    const uint4 b = *reinterpret_cast<const uint4*>(input + 8);
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
      const int group = warp * 4 + tile;
      const half* w = packed + (group * 40 * 64 + g * 64 + split * 8 + r) * 8;
      const uint4 lo = *reinterpret_cast<const uint4*>(w);
      const uint4 hi = *reinterpret_cast<const uint4*>(w + 256);
      ROUTER_MMA(accum[tile], a.x, a.y, lo.x, lo.y);
      ROUTER_MMA(accum[tile], a.z, a.w, lo.z, lo.w);
      ROUTER_MMA(accum[tile], b.x, b.y, hi.x, hi.y);
      ROUTER_MMA(accum[tile], b.z, b.w, hi.z, hi.w);
    }
  }
#pragma unroll
  for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float value = __shfl_sync(0xffffffff, accum[tile][i], lane & ~12);
#pragma unroll
      for (int s = 1; s < 4; ++s)
        value +=
            __shfl_sync(0xffffffff, accum[tile][i], (lane & ~12) | (s << 2));
      const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (warp * 4 + tile) * 8 + (i & 1) +
                      (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
      if (split == 0 && rr == 0) projected[col] = __float2half_rn(value);
    }
  }
  __syncthreads();
  float candidate = __half2float(projected[t]);
  logits[token * 512 + t] = projected[t];
  if (t == 0) denominator = 0.0f;
  float best = 0.0f;
#pragma unroll
  for (int choice = 0; choice < 10; ++choice) {
    float value = candidate;
    int id = t;
#pragma unroll
    for (int offset = 16; offset; offset /= 2) max_pair(value, id, offset);
    if (lane == 0) {
      maxima[warp] = value;
      maxima_ids[warp] = id;
    }
    __syncthreads();
    if (warp == 0) {
      value = lane < 16 ? maxima[lane] : -CUDART_INF_F;
      id = lane < 16 ? maxima_ids[lane] : 512;
#pragma unroll
      for (int offset = 16; offset; offset /= 2) max_pair(value, id, offset);
      if (lane == 0) {
        if (choice == 0) best = value;
        winner = id;
        winners[choice] = id;
        selected[choice] = exp2f((value - best) * 1.4426950408889634f);
        denominator += selected[choice];
      }
    }
    __syncthreads();
    if (t == winner) candidate = -CUDART_INF_F;
  }
  if (t < 10) {
    ids[token * 10 + t] = winners[t];
    probabilities[token * 10 + t] = selected[t] / denominator;
    source_rows[token * 10 + t] = t * m + token;
  }
}
#undef ROUTER_MMA

void run(torch::Tensor logits, torch::Tensor probabilities, torch::Tensor ids,
         torch::Tensor source_rows, torch::Tensor x, torch::Tensor packed) {
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 16 && x.size(1) == 2560,
      "Token-CTA router requires M1..16, K2560");
  for (const auto& t : {logits, probabilities, ids, source_rows, x, packed})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  const int m = x.size(0);
  TORCH_CHECK(
      x.scalar_type() == at::kHalf && packed.scalar_type() == at::kHalf &&
      logits.scalar_type() == at::kHalf &&
      probabilities.scalar_type() == at::kFloat &&
      ids.scalar_type() == at::kInt && source_rows.scalar_type() == at::kInt);
  TORCH_CHECK(logits.sizes() == at::IntArrayRef({m, 512}) &&
              probabilities.sizes() == at::IntArrayRef({m, 10}) &&
              ids.sizes() == at::IntArrayRef({m, 10}) &&
              source_rows.sizes() == at::IntArrayRef({m, 10}) &&
              packed.sizes() == at::IntArrayRef({64, 40, 2, 4, 8, 8}));
  router_token_cta<<<m, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(packed.data_ptr()),
      reinterpret_cast<half*>(logits.data_ptr()),
      probabilities.data_ptr<float>(), ids.data_ptr<int>(),
      source_rows.data_ptr<int>(), m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "qwen38_router_token_cta_sm70_out(Tensor(a!) logits, Tensor(b!) "
      "probabilities, "
      "Tensor(c!) ids, Tensor(d!) source_rows, Tensor x, Tensor packed) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen38_router_token_cta_sm70_out", &run);
}
