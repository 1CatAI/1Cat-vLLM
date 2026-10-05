// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Preserve parallel state tiles; fixed norm consumers use independent flags.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda/atomic>
#include <cuda_fp16.h>

namespace {
constexpr int Threads = 128, Producers = 192, Blocks = 204;
__device__ float warp_sum(float x) {
  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}
__device__ float column_sum(float x) {
  for (int d = 8; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}
__device__ void wait(uint32_t* p, uint32_t generation) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(*p);
  while (flag.load(cuda::memory_order_acquire) != generation) {
  }
}
__global__ __launch_bounds__(Threads, 3) void gdn_tile_chain(
    const half* qkv, const half* a, const half* b, const float* al,
    const half* bias, const half* z, const half* weight, const float* initial,
    float* states, half* core, float* norms, uint32_t* flags, uint32_t* epochs,
    half* output, int m, float eps) {
  const int t = threadIdx.x, block = blockIdx.x;
  const uint32_t generation = epochs[block] + 1;
  __shared__ float q[128], k[128], sums[8], normalizer_q, normalizer_k, decay,
      beta;
  if (block < Producers) {
    const int hv = block / 16, tile = block % 16, hq = hv / 3, lane = t % 16,
              v = tile * 8 + t / 16;
    float state[8];
#pragma unroll
    for (int r = 0; r < 8; ++r)
      state[r] = initial[(hv * 128 + v) * 128 + r * 16 + lane];
    for (int row = 0; row < m; ++row) {
      q[t] = __half2float(qkv[row * 2560 + hq * 128 + t]);
      k[t] = __half2float(qkv[row * 2560 + 512 + hq * 128 + t]);
      float sq = warp_sum(q[t] * q[t]), sk = warp_sum(k[t] * k[t]);
      if ((t & 31) == 0) {
        sums[t / 32] = sq;
        sums[4 + t / 32] = sk;
      }
      __syncthreads();
      if (t == 0) {
        sq = 0;
        sk = 0;
        for (int w = 0; w < 4; ++w) {
          sq += sums[w];
          sk += sums[4 + w];
        }
        normalizer_q = rsqrtf(sq + 1e-6f);
        normalizer_k = rsqrtf(sk + 1e-6f);
        const float av =
            __half2float(a[row * 12 + hv]) + __half2float(bias[hv]);
        const float softplus = av > 20.f ? av : log1pf(expf(av));
        decay = expf(-expf(al[hv]) * softplus);
        beta = 1.f / (1.f + expf(-__half2float(b[row * 12 + hv])));
      }
      __syncthreads();
      q[t] *= normalizer_q;
      k[t] *= normalizer_k;
      __syncthreads();
      float dot = 0;
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        state[r] *= decay;
        dot = fmaf(state[r], k[r * 16 + lane], dot);
      }
      dot = column_sum(dot);
      const float delta =
          (__half2float(qkv[row * 2560 + 1024 + hv * 128 + v]) - dot) * beta;
      dot = 0;
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        state[r] = fmaf(delta, k[r * 16 + lane], state[r]);
        states[((row * 12 + hv) * 128 + v) * 128 + r * 16 + lane] = state[r];
        dot = fmaf(state[r], q[r * 16 + lane], dot);
      }
      dot = column_sum(dot) * 0.08838834764831845f;
      if (lane == 0) {
        const half rounded = __float2half_rn(dot);
        core[(row * 12 + hv) * 128 + v] = rounded;
        const float fv = __half2float(rounded);
        sums[t / 16] = fv * fv;
      }
      __syncthreads();
      if (t == 0) {
        float sum = 0;
        for (int v = 0; v < 8; ++v) sum += sums[v];
        norms[row * Producers + block] = sum;
      }
      __threadfence();
      __syncthreads();
      if (t == 0) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(
            flags[row * Producers + block]);
        flag.store(generation, cuda::memory_order_release);
      }
      __syncthreads();
    }
  } else {
    const int hv = block - Producers;
    for (int row = 0; row < m; ++row) {
      float sum = 0;
      if (t < 16) {
        wait(flags + row * Producers + hv * 16 + t, generation);
        sum = norms[row * Producers + hv * 16 + t];
      }
      sum = warp_sum(sum);
      if (t == 0) normalizer_q = rsqrtf(sum / 128.f + eps);
      __syncthreads();
      const int index = (row * 12 + hv) * 128 + t;
      const float zv = __half2float(z[index]);
      const float value =
          __half2float(core[index]) * normalizer_q * __half2float(weight[t]);
      output[index] = __float2half_rn(value * zv / (1.f + expf(-zv)));
      __syncthreads();
    }
  }
  if (t == 0) epochs[block] = generation;
}
}  // namespace
void run(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor al,
         torch::Tensor bias, torch::Tensor z, torch::Tensor weight,
         torch::Tensor initial, torch::Tensor states, torch::Tensor core,
         torch::Tensor norms, torch::Tensor flags, torch::Tensor epochs,
         torch::Tensor output, double epsilon) {
  const int m = qkv.size(0);
  TORCH_CHECK((m == 1 || m == 5) && qkv.size(1) == 2560);
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, gdn_tile_chain, Threads, 0));
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->cooperativeLaunch &&
              active * properties->multiProcessorCount >= Blocks);
  const half* qp = (half*)qkv.data_ptr();
  const half* ap = (half*)a.data_ptr();
  const half* bp = (half*)b.data_ptr();
  const float* alp = al.data_ptr<float>();
  const half* dp = (half*)bias.data_ptr();
  const half* zp = (half*)z.data_ptr();
  const half* wp = (half*)weight.data_ptr();
  const float* ip = initial.data_ptr<float>();
  float* sp = states.data_ptr<float>();
  half* cp = (half*)core.data_ptr();
  float* np = norms.data_ptr<float>();
  auto* fp = (uint32_t*)flags.data_ptr();
  auto* ep = (uint32_t*)epochs.data_ptr();
  half* op = (half*)output.data_ptr();
  float eps = epsilon;
  void* args[] = {&qp, &ap, &bp, &alp, &dp, &zp, &wp,       &ip,
                  &sp, &cp, &np, &fp,  &ep, &op, (void*)&m, &eps};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      (const void*)gdn_tile_chain, dim3(Blocks), dim3(Threads), args, 0,
      c10::cuda::getCurrentCUDAStream()));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }
