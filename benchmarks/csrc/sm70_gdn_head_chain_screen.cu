// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research: one CTA per value head owns the complete recurrence and gated norm.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

__device__ float sub_sum(float v) {
  for (int d = 4; d; d >>= 1) v += __shfl_xor_sync(0xffffffff, v, d);
  return v;
}
__device__ float warp_reduce(float v) {
  for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffff, v, d);
  return v;
}
__global__ __launch_bounds__(256) void gdn_head_chain(
    const half* qkv, const half* a, const half* b, const float* al,
    const half* bias, const half* z, const half* weight, const float* initial,
    float* final_states, half* output, int m, float eps) {
  const int hv = blockIdx.x, hq = hv / 3, t = threadIdx.x, lane = t % 8,
            group = t / 8;
  __shared__ float q[128], k[128], norms[8], rounded[128], inv, decay, beta;
  float state[4][16];
#pragma unroll
  for (int c = 0; c < 4; ++c)
#pragma unroll
    for (int r = 0; r < 16; ++r)
      state[c][r] = initial[(hv * 128 + group * 4 + c) * 128 + r * 8 + lane];
  for (int row = 0; row < m; ++row) {
    float qn = 0, kn = 0;
    if (t < 128) {
      q[t] = __half2float(qkv[row * 2560 + hq * 128 + t]);
      k[t] = __half2float(qkv[row * 2560 + 512 + hq * 128 + t]);
      qn = q[t] * q[t];
      kn = k[t] * k[t];
    }
    qn = warp_reduce(qn);
    kn = warp_reduce(kn);
    if ((t & 31) == 0) norms[t / 32] = qn;
    __syncthreads();
    float sum = 0;
    if (t < 8) sum = norms[t];
    sum = warp_reduce(sum);
    if (t == 0) inv = rsqrtf(sum + 1e-6f);
    __syncthreads();
    if (t < 128) q[t] *= inv;
    __syncthreads();
    if ((t & 31) == 0) norms[t / 32] = kn;
    __syncthreads();
    sum = t < 8 ? norms[t] : 0;
    sum = warp_reduce(sum);
    if (t == 0) {
      inv = rsqrtf(sum + 1e-6f);
      const float x = __half2float(a[row * 12 + hv]) + __half2float(bias[hv]);
      decay = expf(-expf(al[hv]) * (x <= 20.f ? log1pf(expf(x)) : x));
      beta = 1.f / (1.f + expf(-__half2float(b[row * 12 + hv])));
    }
    __syncthreads();
    if (t < 128) k[t] *= inv;
    __syncthreads();
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      float kv = 0;
#pragma unroll
      for (int r = 0; r < 16; ++r) kv = fmaf(state[c][r], k[r * 8 + lane], kv);
      kv = sub_sum(kv);
      const int col = group * 4 + c;
      const float v = __half2float(qkv[row * 2560 + 1024 + hv * 128 + col]);
      const float delta = (v - decay * kv) * beta;
      float dot = 0;
#pragma unroll
      for (int r = 0; r < 16; ++r) {
        state[c][r] = fmaf(k[r * 8 + lane], delta, decay * state[c][r]);
        dot = fmaf(state[c][r], q[r * 8 + lane], dot);
        final_states[((row * 12 + hv) * 128 + col) * 128 + r * 8 + lane] =
            state[c][r];
      }
      dot = sub_sum(dot);
      if (lane == 0)
        rounded[col] = __half2float(__float2half_rn(dot * 0.0883883476483f));
    }
    __syncthreads();
    sum = t < 128 ? rounded[t] * rounded[t] : 0;
    sum = warp_reduce(sum);
    if ((t & 31) == 0) norms[t / 32] = sum;
    __syncthreads();
    sum = t < 8 ? norms[t] : 0;
    sum = warp_reduce(sum);
    if (t == 0) inv = rsqrtf(sum / 128.f + eps);
    __syncthreads();
    if (t < 128) {
      const float gate = __half2float(z[(row * 12 + hv) * 128 + t]);
      const float normalized = rounded[t] * inv * __half2float(weight[t]);
      output[(row * 12 + hv) * 128 + t] =
          __float2half_rn(normalized * gate / (1.f + expf(-gate)));
    }
    __syncthreads();
  }
}
void run(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor al,
         torch::Tensor bias, torch::Tensor z, torch::Tensor weight,
         torch::Tensor initial, torch::Tensor final_states,
         torch::Tensor output, double eps) {
  const int m = qkv.size(0);
  TORCH_CHECK((m == 1 || m == 5) && qkv.sizes() == at::IntArrayRef({m, 2560}));
  TORCH_CHECK(initial.numel() == 12 * 128 * 128 &&
              final_states.numel() == m * 12 * 128 * 128);
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  gdn_head_chain<<<12, 256, 0, stream>>>(
      (half*)qkv.data_ptr(), (half*)a.data_ptr(), (half*)b.data_ptr(),
      al.data_ptr<float>(), (half*)bias.data_ptr(), (half*)z.data_ptr(),
      (half*)weight.data_ptr(), initial.data_ptr<float>(),
      final_states.data_ptr<float>(), (half*)output.data_ptr(), m, eps);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }
