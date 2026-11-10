// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Parallel draft local selection; existing compact IPC is unchanged.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <climits>
#include <cmath>
#include <torch/library.h>
#include <torch/types.h>

namespace {
__device__ bool better(float b, int bi, float a, int ai) {
  if (isnan(b)) return !isnan(a) || bi < ai;
  if (isnan(a)) return false;
  return b > a || (b == a && bi < ai);
}

__device__ void reduce(float& value, int& id) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  __shared__ float values[4];
  __shared__ int ids[4];
#pragma unroll
  for (int offset = 16; offset; offset /= 2) {
    const float other = __shfl_down_sync(0xffffffff, value, offset);
    const int oi = __shfl_down_sync(0xffffffff, id, offset);
    if (lane + offset < 32 && better(other, oi, value, id)) {
      value = other;
      id = oi;
    }
  }
  if (lane == 0) {
    values[warp] = value;
    ids[warp] = id;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    value = values[0];
    id = ids[0];
#pragma unroll
    for (int i = 1; i < 4; ++i) {
      if (better(values[i], ids[i], value, id)) {
        value = values[i];
        id = ids[i];
      }
    }
  }
}

__global__ void segments(const half* logits, float* partial, int n, int valid,
                         int blocks, int start) {
  float value = -INFINITY;
  int id = INT_MAX;
  for (int c = blockIdx.x * 512 + threadIdx.x;
       c < valid && c < (blockIdx.x + 1) * 512; c += 128) {
    const float v = __half2float(logits[blockIdx.y * n + c]);
    if (better(v, start + c, value, id)) {
      value = v;
      id = start + c;
    }
  }
  reduce(value, id);
  if (threadIdx.x == 0) {
    const int offset = (blockIdx.y * blocks + blockIdx.x) * 2;
    partial[offset] = value;
    partial[offset + 1] = float(id);
  }
}

__global__ void merge(const float* partial, float* pairs, int blocks) {
  float value = -INFINITY;
  int id = INT_MAX;
  for (int c = threadIdx.x; c < blocks; c += 128) {
    const int offset = (blockIdx.x * blocks + c) * 2;
    const float v = partial[offset];
    const int i = int(partial[offset + 1]);
    if (better(v, i, value, id)) {
      value = v;
      id = i;
    }
  }
  reduce(value, id);
  if (threadIdx.x == 0) {
    pairs[blockIdx.x * 2] = value;
    pairs[blockIdx.x * 2 + 1] = float(id);
  }
}

void run(torch::Tensor pairs, torch::Tensor partial, torch::Tensor logits,
         int64_t valid, int64_t start) {
  const c10::cuda::CUDAGuard guard(logits.device());
  const auto* p = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(p->major == 7 && p->minor == 0);
  for (const auto& t : {pairs, partial, logits})
    TORCH_CHECK(t.is_cuda() && t.device() == logits.device() &&
                t.is_contiguous());
  TORCH_CHECK(logits.dim() == 2 && logits.scalar_type() == at::kHalf &&
              pairs.scalar_type() == at::kFloat &&
              partial.scalar_type() == at::kFloat);
  const int m = logits.size(0), n = logits.size(1),
            blocks = (valid + 511) / 512;
  TORCH_CHECK(m >= 1 && m <= 128 && valid >= 1 && valid <= n &&
              valid <= 262144 && start >= 0 && start <= 16777216 - valid &&
              pairs.sizes() == at::IntArrayRef({m, 2}) &&
              partial.sizes() == at::IntArrayRef({m, blocks, 2}));
  const auto stream = at::cuda::getCurrentCUDAStream();
  segments<<<dim3(blocks, m), 128, 0, stream>>>(
      reinterpret_cast<const half*>(logits.data_ptr()),
      partial.data_ptr<float>(), n, valid, blocks, start);
  merge<<<m, 128, 0, stream>>>(partial.data_ptr<float>(),
                               pairs.data_ptr<float>(), blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "qwen38_mtp_local_top1_sm70_out(Tensor(a!) pairs, Tensor(b!) partial, "
      "Tensor logits, int valid, int start) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen38_mtp_local_top1_sm70_out", &run);
}
