// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// One resident CTA per intermediate tile. Each CTA produces its W2 partial,
// releases a flag, then consumes all tiles in a fixed order for owned outputs.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda/atomic>

__device__ float sum_warp(float v) {
  for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffff, v, d);
  return v;
}
__device__ float sig(float v) { return 1.f / (1.f + expf(-v)); }

__global__ __launch_bounds__(256, 2) void shared_expert_chain(
    const half* x, const half* w13, const half* w2, const half* gate,
    float* partials, uint32_t* flags, uint32_t* epochs, half* output, int m) {
  const int tile = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t / 32;
  __shared__ half activation[5][16];
  __shared__ float gate_sums[8];
  __shared__ float gate_sigmoid[5];
  const uint32_t generation = epochs[tile] + 1;
  for (int row = 0; row < m; ++row) {
    float g = 0;
    for (int k = t; k < 2560; k += 256)
      g = fmaf(__half2float(x[row * 2560 + k]), __half2float(gate[k]), g);
    g = sum_warp(g);
    if (lane == 0) gate_sums[warp] = g;
    __syncthreads();
    if (t == 0) {
      g = 0;
#pragma unroll
      for (int w = 0; w < 8; ++w) g += gate_sums[w];
      gate_sigmoid[row] =
          __half2float(__float2half_rn(sig(__half2float(__float2half_rn(g)))));
    }
    // Four independent projection rows per warp; FP16 materializations
    // retain the gate/up -> SiLU -> multiply boundaries of the model.
    for (int group = 0; group < 2; ++group) {
      const int i = group * 8 + warp, column = tile * 16 + i;
      float a = 0, b = 0;
      for (int k = lane; k < 2560; k += 32) {
        const float v = __half2float(x[row * 2560 + k]);
        a = fmaf(v, __half2float(w13[column * 2560 + k]), a);
        b = fmaf(v, __half2float(w13[(160 + column) * 2560 + k]), b);
      }
      a = sum_warp(a);
      b = sum_warp(b);
      if (lane == 0) {
        a = __half2float(__float2half_rn(a));
        b = __half2float(__float2half_rn(b));
        activation[row][i] =
            __hmul(__float2half_rn(a * sig(a)), __float2half_rn(b));
      }
    }
    __syncthreads();
    for (int n = t; n < 2560; n += 256) {
      float value = 0;
#pragma unroll
      for (int i = 0; i < 16; ++i)
        value = fmaf(__half2float(activation[row][i]),
                     __half2float(w2[(tile * 16 + i) * 2560 + n]), value);
      partials[(tile * m + row) * 2560 + n] = value;
    }
    __syncthreads();
  }
  // Every writer orders its own stores before the leader publishes readiness.
  __threadfence();
  __syncthreads();
  if (t == 0) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(flags[tile]);
    flag.store(generation, cuda::memory_order_release);
  }
  __syncthreads();
  if (t < 10) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(flags[t]);
    while (flag.load(cuda::memory_order_acquire) != generation) {
    }
  }
  __syncthreads();
  const int n = tile * 256 + t;
  for (int row = 0; row < m; ++row) {
    float value = 0;
#pragma unroll
    for (int split = 0; split < 10; ++split)
      value += partials[(split * m + row) * 2560 + n];
    value = __half2float(__float2half_rn(value));
    output[row * 2560 + n] = __float2half_rn(value * gate_sigmoid[row]);
  }
  if (t == 0) epochs[tile] = generation;
}

void run(torch::Tensor x, torch::Tensor w13, torch::Tensor w2,
         torch::Tensor gate, torch::Tensor partials, torch::Tensor flags,
         torch::Tensor epochs, torch::Tensor output) {
  const int m = x.size(0);
  TORCH_CHECK((m == 1 || m == 5) && x.size(1) == 2560);
  TORCH_CHECK(w13.sizes() == at::IntArrayRef({320, 2560}));
  TORCH_CHECK(w2.sizes() == at::IntArrayRef({160, 2560}));
  TORCH_CHECK(gate.numel() == 2560 && partials.numel() == 10 * m * 2560);
  for (const auto& tensor : {x, w13, w2, gate, output}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous());
    TORCH_CHECK(tensor.scalar_type() == at::kHalf &&
                tensor.device() == x.device());
  }
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  shared_expert_chain<<<10, 256, 0, stream>>>(
      (half*)x.data_ptr(), (half*)w13.data_ptr(), (half*)w2.data_ptr(),
      (half*)gate.data_ptr(), partials.data_ptr<float>(),
      (uint32_t*)flags.data_ptr(), (uint32_t*)epochs.data_ptr(),
      (half*)output.data_ptr(), m);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }
