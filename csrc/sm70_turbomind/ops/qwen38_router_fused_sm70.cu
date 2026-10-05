// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// M2..8 router prototype: 80 independent producers, last-CTA selection/plan.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
constexpr int Experts = 512, Routes = 80, Pack = 8, Producers = 80;

// Same integer-only expert grouping as nvfp4_grouped_decode_sm70.cu.
__device__ void make_plan(const int* ids, int* rows, int* experts, int* sizes,
                          int* total, int routes, int* counts, int* groups) {
  const int t = threadIdx.x;
  for (int e = t; e <= Experts; e += blockDim.x) counts[e] = 0;
  if (t == 0) *total = 0;
  __syncthreads();
  int expert = 0, ordinal = 0;
  if (t < routes) {
    expert = ids[t];
    if (expert < 0 || expert >= Experts) expert = Experts;
    ordinal = atomicAdd(counts + expert, 1);
  }
  __syncthreads();
  if (t < routes && ordinal % Pack == 0) {
    const int group = atomicAdd(total, 1);
    groups[expert * (Routes / Pack) + ordinal / Pack] = group;
    experts[group] = expert;
    sizes[group] = min(Pack, counts[expert] - ordinal);
  }
  __syncthreads();
  if (t < routes) {
    const int group = groups[expert * (Routes / Pack) + ordinal / Pack];
    rows[group * Pack + ordinal % Pack] = t;
  }
}

__global__ void plan_only(const int* ids, int* rows, int* experts, int* sizes,
                          int* total, int routes) {
  __shared__ int counts[Experts + 1], groups[(Experts + 1) * (Routes / Pack)];
  make_plan(ids, rows, experts, sizes, total, routes, counts, groups);
}

__global__ __launch_bounds__(256, 1) void fused_router(
    const half* x, const half* weight, half* logits, float* probabilities,
    int* ids, int* source_rows, int* rows, int* experts, int* sizes, int* total,
    unsigned* counter, int m) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int expert = blockIdx.x + warp * Producers;
  if (expert < Experts) {
    float accum[8] = {};
    for (int k = lane * 2; k < 2560; k += 64) {
      const float2 w = __half22float2(
          *reinterpret_cast<const half2*>(weight + expert * 2560 + k));
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        if (r < m) {
          const float2 a =
              __half22float2(*reinterpret_cast<const half2*>(x + r * 2560 + k));
          accum[r] = fmaf(a.x, w.x, accum[r]);
          accum[r] = fmaf(a.y, w.y, accum[r]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      for (int offset = 16; offset; offset /= 2)
        accum[r] += __shfl_down_sync(0xffffffff, accum[r], offset);
      if (lane == 0 && r < m)
        logits[r * Experts + expert] = __float2half_rn(accum[r]);
    }
  }
  // Publication by all writers precedes the block's completion ticket. The
  // last producer reads every half through a volatile load; no spinning CTA
  // or cooperative-launch residency requirement is introduced.
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = atomicAdd(counter, 1) == Producers - 1;
  __syncthreads();
  if (!last) return;
  if (warp < m) {
    float values[16];
    const auto* published = reinterpret_cast<volatile unsigned short*>(logits);
    bool invalid = false;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float v = __half2float(
          __ushort_as_half(published[warp * Experts + j * 32 + lane]));
      values[j] = v;
      invalid |= isnan(v) || v == CUDART_INF_F;
    }
    float best = -CUDART_INF_F;
#pragma unroll
    for (int j = 0; j < 16; ++j) best = fmaxf(best, values[j]);
    for (int offset = 16; offset; offset /= 2)
      best = fmaxf(best, __shfl_xor_sync(0xffffffff, best, offset));
    invalid = __any_sync(0xffffffff, invalid) || best == -CUDART_INF_F;
    float selected[10];
    int selected_ids[10];
    float denominator = 0.0f;
    unsigned used = 0;
#pragma unroll
    for (int choice = 0; choice < 10; ++choice) {
      float v = -CUDART_INF_F;
      int id = Experts;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int e = j * 32 + lane;
        if (!(used & (1u << j)) &&
            (values[j] > v || (values[j] == v && e < id))) {
          v = values[j];
          id = e;
        }
      }
      for (int offset = 16; offset; offset /= 2) {
        const float other = __shfl_xor_sync(0xffffffff, v, offset);
        const int other_id = __shfl_xor_sync(0xffffffff, id, offset);
        if (other > v || (other == v && other_id < id)) {
          v = other;
          id = other_id;
        }
      }
      if (invalid) id = choice;
      selected_ids[choice] = id;
      selected[choice] =
          invalid ? 0.0f : exp2f((v - best) * 1.4426950408889634f);
      denominator += selected[choice];
      if (lane == id % 32) used |= 1u << (id / 32);
    }
    if (lane < 10) {
      ids[warp * 10 + lane] = selected_ids[lane];
      probabilities[warp * 10 + lane] =
          denominator > 0.0f ? selected[lane] / denominator : 0.0f;
      source_rows[warp * 10 + lane] = lane * m + warp;
    }
  }
  __syncthreads();
  __shared__ int counts[Experts + 1], groups[(Experts + 1) * (Routes / Pack)];
  make_plan(ids, rows, experts, sizes, total, m * 10, counts, groups);
  __syncthreads();
  if (threadIdx.x == 0) atomicExch(counter, 0);
}

void run(torch::Tensor logits, torch::Tensor probabilities, torch::Tensor ids,
         torch::Tensor source_rows, torch::Tensor rows, torch::Tensor experts,
         torch::Tensor sizes, torch::Tensor total, torch::Tensor counter,
         torch::Tensor x, torch::Tensor weight, bool plan_only_mode) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int m = x.size(0);
  TORCH_CHECK(x.dim() == 2 && m >= 2 && m <= 8 && x.size(1) == 2560);
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0);
  for (const auto& t : {logits, probabilities, ids, source_rows, rows, experts,
                        sizes, total, counter, x, weight})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  TORCH_CHECK(x.scalar_type() == at::kHalf &&
              weight.scalar_type() == at::kHalf &&
              logits.scalar_type() == at::kHalf &&
              probabilities.scalar_type() == at::kFloat);
  for (const auto& t : {ids, source_rows, rows, experts, sizes, total, counter})
    TORCH_CHECK(t.scalar_type() == at::kInt);
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({512, 2560}) &&
              logits.sizes() == at::IntArrayRef({m, 512}) &&
              probabilities.numel() == m * 10 && ids.numel() == m * 10 &&
              source_rows.numel() == m * 10 && rows.numel() >= m * 10 * Pack &&
              experts.numel() >= m * 10 && sizes.numel() >= m * 10 &&
              total.numel() == 1 && counter.numel() == 1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (plan_only_mode) {
    plan_only<<<1, 256, 0, stream>>>(
        ids.data_ptr<int>(), rows.data_ptr<int>(), experts.data_ptr<int>(),
        sizes.data_ptr<int>(), total.data_ptr<int>(), m * 10);
  } else {
    fused_router<<<Producers, 256, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(weight.data_ptr()),
        reinterpret_cast<half*>(logits.data_ptr()),
        probabilities.data_ptr<float>(), ids.data_ptr<int>(),
        source_rows.data_ptr<int>(), rows.data_ptr<int>(),
        experts.data_ptr<int>(), sizes.data_ptr<int>(), total.data_ptr<int>(),
        reinterpret_cast<unsigned*>(counter.data_ptr()), m);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "qwen38_router_fused_sm70_out(Tensor(a!) logits, Tensor(b!) "
      "probabilities, "
      "Tensor(c!) ids, Tensor(d!) source_rows, Tensor(e!) rows, "
      "Tensor(f!) experts, Tensor(g!) sizes, Tensor(h!) total, "
      "Tensor(i!) counter, Tensor x, Tensor weight, bool plan_only=False) -> "
      "()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen38_router_fused_sm70_out", &run);
}
