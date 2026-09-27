// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// MTP4 PLE convolution: rollback, four dilated taps, SiLU, state commit.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
__device__ half as_half(half x) { return x; }
__device__ half as_half(float x) { return __float2half_rn(x); }
__device__ void store_half(half* ptr, half x) { *ptr = x; }
__device__ void store_half(float* ptr, half x) { *ptr = __half2float(x); }
template <class State>
__global__ void ple_conv(const half* x, State* state, const half* weight,
                         const int* ids, const int* starts, const int* accepted,
                         half* out, int rows, int hidden, int64_t s0,
                         int64_t s1, int64_t s2) {
  int c = blockIdx.x * 128 + threadIdx.x;
  if (c >= hidden) return;
  int sid = ids[0], count = starts[1] - starts[0],
      rollback = sid ? max(0, min(4, accepted[0] - 1)) : 0;
  half h[14];
#pragma unroll
  for (int t = 0; t < 9; ++t)
    h[t] = sid ? as_half(state[sid * s0 + c * s1 + (rollback + t) * s2])
               : __float2half_rn(0.f);
#pragma unroll
  for (int t = 0; t < 5; ++t)
    h[t + 9] = t < count ? x[t * hidden + c] : __float2half_rn(0.f);
  half w[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) w[k] = weight[c * 4 + k];
#pragma unroll
  for (int r = 0; r < 5; ++r) {
    float v = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
      v = __fmaf_rn(__half2float(h[r + k * 3]), __half2float(w[k]), v);
    v = __half2float(__float2half_rn(v));
    out[r * hidden + c] = r < count
                              ? __float2half_rn(v / __fadd_rn(1.f, expf(-v)))
                              : __float2half_rn(0.f);
  }
  for (int r = 5; r < rows; ++r) out[r * hidden + c] = __float2half_rn(0.f);
  if (sid) {
#pragma unroll
    for (int t = 0; t < 13; ++t)
      if (t < 8 + count)
        store_half(state + sid * s0 + c * s1 + t * s2, h[t + 1]);
  }
}

void ple_spec_conv(torch::Tensor output, torch::Tensor state, torch::Tensor x,
                   torch::Tensor weight, torch::Tensor ids,
                   torch::Tensor starts, torch::Tensor accepted) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(1) == 10240 &&
                  (x.size(0) == 5 || x.size(0) == 10),
              "SM70 PLE MTP4 requires M5/M10 and H10240");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  for (const auto& t : {output, state, x, weight, ids, starts, accepted})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device(),
                "PLE MTP4 tensors must share a CUDA device");
  for (const auto& t : {output, x, weight})
    TORCH_CHECK(t.scalar_type() == at::kHalf && t.is_contiguous(),
                "PLE MTP4 input, weight and output must be contiguous FP16");
  for (const auto& t : {ids, starts, accepted})
    TORCH_CHECK(
        t.scalar_type() == at::kInt && t.dim() == 1 && t.is_contiguous(),
        "PLE MTP4 metadata must be contiguous int32 vectors");
  TORCH_CHECK(ids.numel() == 1 && starts.numel() >= 2 && accepted.numel() >= 1,
              "PLE MTP4 fast path requires one speculative request");
  TORCH_CHECK(state.dim() == 3 && state.size(0) > 0 && state.size(1) == 10240 &&
                  state.size(2) == 13 &&
                  (state.scalar_type() == at::kHalf ||
                   state.scalar_type() == at::kFloat),
              "PLE MTP4 requires FP16/FP32 [states,10240,13] cache");
  TORCH_CHECK((state.stride(2) == 1 && state.stride(1) >= 13) ||
                  (state.stride(1) == 1 && state.stride(2) >= 10240),
              "Unsupported PLE cache layout");
  TORCH_CHECK(
      state.stride(0) >= 10239 * state.stride(1) + 12 * state.stride(2) + 1,
      "PLE cache rows must not overlap");
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({10240, 4}) &&
                  output.sizes() == x.sizes(),
              "Invalid PLE MTP4 weight/output geometry");
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(T)                                                            \
  ple_conv<T><<<80, 128, 0, stream>>>(                                       \
      reinterpret_cast<const half*>(x.data_ptr()),                           \
      reinterpret_cast<T*>(state.data_ptr()),                                \
      reinterpret_cast<const half*>(weight.data_ptr()), ids.data_ptr<int>(), \
      starts.data_ptr<int>(), accepted.data_ptr<int>(),                      \
      reinterpret_cast<half*>(output.data_ptr()), x.size(0), 10240,          \
      state.stride(0), state.stride(1), state.stride(2))
  if (state.scalar_type() == at::kHalf) {
    LAUNCH(half);
  } else {
    LAUNCH(float);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "qwen38_ple_spec_sm70_out(Tensor(a!) output, Tensor(b!) state, Tensor x, "
      "Tensor weight, Tensor ids, Tensor starts, Tensor accepted) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen38_ple_spec_sm70_out", &ple_spec_conv);
}
