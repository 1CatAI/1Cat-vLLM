// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Four K producers per intermediate tile preserve shared-expert parallelism.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda/atomic>
#include <cuda_fp16.h>

__device__ void mma(float (&acc)[8], uint32_t a0, uint32_t a1, uint32_t b0,
                    uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]),
        "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
__device__ float warp_sum(float x) {
  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}
__global__ __launch_bounds__(256, 2) void shared_expert_mma_chain(
    const half* x, const half* w13, const half* w2, const half* gate,
    float* projection, float* partial, uint32_t* flags, uint32_t* epochs,
    half* output, int m) {
  const int block = blockIdx.x, tile = block % 10, part = block / 10;
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0), quad = (lane >> 2) & 3,
            col = quad * 8 + r;
  const uint32_t generation = epochs[block] + 1;
  __shared__ float projected[8 * 5 * 32], gate_partials[8], gate_sigmoid[5];
  __shared__ half activation[5 * 16];
  float accum[8] = {};
#pragma unroll 4
  for (int g = part * 40 + warp * 5; g < part * 40 + (warp + 1) * 5; ++g) {
    const half* weight = w13 + (tile * 160 + g) * 512 + col * 8;
    const uint4 lo = *(const uint4*)weight, hi = *(const uint4*)(weight + 256);
    uint4 a = {}, b = {};
    if (r < m) {
      a = *(const uint4*)(x + r * 2560 + g * 16);
      b = *(const uint4*)(x + r * 2560 + g * 16 + 8);
    }
    mma(accum, a.x, a.y, lo.x, lo.y);
    mma(accum, a.z, a.w, lo.z, lo.w);
    mma(accum, b.x, b.y, hi.x, hi.y);
    mma(accum, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int n =
        quad * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
    if (row < m) projected[(warp * m + row) * 32 + n] = accum[i];
  }
  __syncthreads();
  if (t < m * 32) {
    const int row = t / 32, n = t % 32;
    float value = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) value += projected[(w * m + row) * 32 + n];
    projection[(block * m + row) * 32 + n] = value;
  }
  __threadfence();
  __syncthreads();
  if (t == 0) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(flags[block]);
    flag.store(generation, cuda::memory_order_release);
  }
  __syncthreads();
  // Each tile waits only for its four K producers, independently of other
  // tiles.
  if (t < 4) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(
        flags[t * 10 + tile]);
    while (flag.load(cuda::memory_order_acquire) != generation)
      __nanosleep(128);
  }
  __syncthreads();
  if (t < m * 16) {
    const int row = t / 16, n = t % 16;
    float a = 0, b = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int index = ((k * 10 + tile) * m + row) * 32 + n * 2;
      a += projection[index];
      b += projection[index + 1];
    }
    a = __half2float(__float2half_rn(a));
    b = __half2float(__float2half_rn(b));
    activation[t] = __float2half_rn((a / (1.f + expf(-a))) * b);
  }
  __syncthreads();
  uint4 a = {}, b = {};
  if (r < m) {
    a = *(const uint4*)(activation + r * 16);
    b = *(const uint4*)(activation + r * 16 + 8);
  }
  for (int nt = part * 20 + warp; nt < (part + 1) * 20; nt += 8) {
    const half* weight = w2 + (nt * 10 + tile) * 512 + col * 8;
    const uint4 lo = *(const uint4*)weight, hi = *(const uint4*)(weight + 256);
    float down[8] = {};
    mma(down, a.x, a.y, lo.x, lo.y);
    mma(down, a.z, a.w, lo.z, lo.w);
    mma(down, b.x, b.y, hi.x, hi.y);
    mma(down, b.z, b.w, hi.z, hi.w);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int n = nt * 32 + quad * 8 +
                    ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      if (row < m) partial[(tile * m + row) * 2560 + n] = down[i];
    }
  }
  __threadfence();
  __syncthreads();
  if (t == 0) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(
        flags[40 + block]);
    flag.store(generation, cuda::memory_order_release);
  }
  __syncthreads();
  // Scalar gate work overlaps completion of the other intermediate tiles.
  for (int row = 0; row < m; ++row) {
    float value = 0;
    for (int k = t; k < 2560; k += 256)
      value =
          fmaf(__half2float(x[row * 2560 + k]), __half2float(gate[k]), value);
    value = warp_sum(value);
    if (lane == 0) gate_partials[warp] = value;
    __syncthreads();
    if (t == 0) {
      value = 0;
      for (int w = 0; w < 8; ++w) value += gate_partials[w];
      value = __half2float(__float2half_rn(value));
      gate_sigmoid[row] =
          __half2float(__float2half_rn(1.f / (1.f + expf(-value))));
    }
    __syncthreads();
  }
  if (t < 10) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(
        flags[40 + part * 10 + t]);
    while (flag.load(cuda::memory_order_acquire) != generation)
      __nanosleep(128);
  }
  __syncthreads();
  const int n = part * 640 + tile * 64 + t;
  if (t < 64) {
    for (int row = 0; row < m; ++row) {
      float value = 0;
      for (int split = 0; split < 10; ++split)
        value += partial[(split * m + row) * 2560 + n];
      value = __half2float(__float2half_rn(value));
      output[row * 2560 + n] = __float2half_rn(value * gate_sigmoid[row]);
    }
  }
  if (t == 0) epochs[block] = generation;
}
void run(torch::Tensor x, torch::Tensor w13, torch::Tensor w2,
         torch::Tensor gate, torch::Tensor projection, torch::Tensor partial,
         torch::Tensor flags, torch::Tensor epochs, torch::Tensor out) {
  const int m = x.size(0);
  TORCH_CHECK((m == 1 || m == 5) && x.size(1) == 2560);
  TORCH_CHECK(w13.numel() == 320 * 2560 && w2.numel() == 2560 * 160);
  TORCH_CHECK(projection.numel() >= 40 * m * 32 &&
              partial.numel() >= 10 * m * 2560 && flags.numel() >= 80 &&
              epochs.numel() >= 40);
  const auto* props = at::cuda::getCurrentDeviceProperties();
  int active = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active, shared_expert_mma_chain, 256, 0) == cudaSuccess);
  TORCH_CHECK(props->major == 7 && props->minor == 0 &&
              props->cooperativeLaunch &&
              active * props->multiProcessorCount >= 40);
  auto* xp = (half*)x.data_ptr();
  auto* up = (half*)w13.data_ptr();
  auto* down = (half*)w2.data_ptr();
  auto* gp = (half*)gate.data_ptr();
  auto* proj = projection.data_ptr<float>();
  auto* part = partial.data_ptr<float>();
  auto* flag = (uint32_t*)flags.data_ptr();
  auto* epoch = (uint32_t*)epochs.data_ptr();
  auto* output = (half*)out.data_ptr();
  void* arguments[] = {&xp,   &up,   &down,  &gp,     &proj,
                       &part, &flag, &epoch, &output, (void*)&m};
  TORCH_CHECK(cudaLaunchCooperativeKernel(
                  (void*)shared_expert_mma_chain, dim3(40), dim3(256),
                  arguments, 0,
                  c10::cuda::getCurrentCUDAStream().stream()) == cudaSuccess);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }
