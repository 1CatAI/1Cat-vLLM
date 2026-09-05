// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Benchmark-only FP16 multi-row HC. No production registration or dispatch.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
#define HC_MMA(C, A0, A1, B0, B1)                                   \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// Four warps produce the four HC branch gates for the same Rows x 32 tile.
// Packed weights [4, H/32, K/16, 2, 32, 8] permit coalesced 128-bit reads.
template <int Rows>
__global__ void up_mix(const half* x, const half* weight, const half* residual,
                       half* out, int m) {
  __shared__ half gates[4][Rows][32];
  const int lane = threadIdx.x % 32, branch = threadIdx.x / 32;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3, col = quad * 8 + r;
  float acc[Rows / 8][8] = {};
#pragma unroll
  for (int g = 0; g < 20; ++g) {
    const int idx = ((branch * 80 + blockIdx.x) * 20 + g) * 512 + col * 8;
    const uint4 wlo = *reinterpret_cast<const uint4*>(weight + idx);
    const uint4 whi = *reinterpret_cast<const uint4*>(weight + idx + 256);
#pragma unroll
    for (int pack = 0; pack < Rows / 8; ++pack) {
      const int token = blockIdx.y * Rows + pack * 8 + r;
      uint4 lo = make_uint4(0, 0, 0, 0), hi = make_uint4(0, 0, 0, 0);
      if (token < m) {
        lo = *reinterpret_cast<const uint4*>(x + token * 320 + g * 16);
        hi = *reinterpret_cast<const uint4*>(x + token * 320 + g * 16 + 8);
      }
      HC_MMA(acc[pack], lo.x, lo.y, wlo.x, wlo.y);
      HC_MMA(acc[pack], lo.z, lo.w, wlo.z, wlo.w);
      HC_MMA(acc[pack], hi.x, hi.y, whi.x, whi.y);
      HC_MMA(acc[pack], hi.z, hi.w, whi.z, whi.w);
    }
  }
#pragma unroll
  for (int pack = 0; pack < Rows / 8; ++pack) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int cc = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      gates[branch][pack * 8 + rr][quad * 8 + cc] =
          __float2half_rn(acc[pack][i]);
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < Rows * 32; idx += blockDim.x) {
    const int rr = idx / 32, cc = idx % 32, t = blockIdx.y * Rows + rr;
    if (t < m) {
      float value = 0;
#pragma unroll
      for (int hc = 0; hc < 4; ++hc) {
        const float gate = __half2float(gates[hc][rr][cc]);
        const float scale = 1.0f / (1.0f + expf(-gate));
        value =
            fmaf(scale,
                 __half2float(
                     residual[t * 10240 + hc * 2560 + blockIdx.x * 32 + cc]),
                 value);
      }
      out[t * 2560 + blockIdx.x * 32 + cc] = __float2half_rn(value * 0.25f);
    }
  }
}

// Global Split-K keeps short output-N from starving the 80 SMs. Four warps
// also split K inside each CTA; the second kernel performs an ordered FP32
// reduction and retains the original FP16 rounding before SiLU/injection.
template <int Split>
__global__ void down_partial(const half* x, const half* w, float* partial,
                             int m) {
  __shared__ float sums[4][8][32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3, col = quad * 8 + r;
  const int token = blockIdx.y * 8 + r;
  float acc[8] = {};
  const int begin = (blockIdx.z * 4 + warp) * (640 / (Split * 4));
  const int end = begin + 640 / (Split * 4);
#pragma unroll 2
  for (int g = begin; g < end; ++g) {
    const int idx = (blockIdx.x * 640 + g) * 512 + col * 8;
    const uint4 wlo = *reinterpret_cast<const uint4*>(w + idx);
    const uint4 whi = *reinterpret_cast<const uint4*>(w + idx + 256);
    uint4 lo = make_uint4(0, 0, 0, 0), hi = make_uint4(0, 0, 0, 0);
    if (token < m) {
      lo = *reinterpret_cast<const uint4*>(x + token * 10240 + g * 16);
      hi = *reinterpret_cast<const uint4*>(x + token * 10240 + g * 16 + 8);
    }
    HC_MMA(acc, lo.x, lo.y, wlo.x, wlo.y);
    HC_MMA(acc, lo.z, lo.w, wlo.z, wlo.w);
    HC_MMA(acc, hi.x, hi.y, whi.x, whi.y);
    HC_MMA(acc, hi.z, hi.w, whi.z, whi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int cc = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    sums[warp][rr][quad * 8 + cc] = acc[i];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < 256; idx += blockDim.x) {
    const int rr = idx / 32, cc = idx % 32, t = blockIdx.y * 8 + rr;
    float value = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) value += sums[s][rr][cc];
    if (t < m)
      partial[(blockIdx.z * m + t) * 352 + blockIdx.x * 32 + cc] = value;
  }
}

template <int Split>
__global__ void down_tail(const float* partial, half* lora, half* inject,
                          int m) {
  const int t = blockIdx.x, col = threadIdx.x;
  if (col >= 324) return;
  float value = 0;
#pragma unroll
  for (int s = 0; s < Split; ++s) value += partial[(s * m + t) * 352 + col];
  const half rounded = __float2half_rn(value);
  if (col < 320) {
    const float v = __half2float(rounded) * 0.25f;
    lora[t * 320 + col] = __float2half_rn(v / (1.0f + expf(-v)));
  } else {
    inject[t * 4 + col - 320] = rounded;
  }
}

void check_half(const torch::Tensor& t, const torch::Tensor& ref) {
  TORCH_CHECK(t.is_cuda() && t.device() == ref.device() && t.is_contiguous() &&
              t.scalar_type() == at::kHalf);
}
void run_up(torch::Tensor out, torch::Tensor x, torch::Tensor w,
            torch::Tensor residual) {
  const c10::cuda::CUDAGuard guard(x.device());
  for (const auto& t : {out, x, w, residual}) check_half(t, x);
  TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 16 &&
              x.size(1) == 320);
  const int m = x.size(0);
  TORCH_CHECK(out.numel() == m * 2560 && residual.numel() == m * 10240 &&
              w.numel() == 10240 * 320);
#define UP(R)                                             \
  up_mix<R><<<dim3(80, (m + R - 1) / R), 128, 0,          \
              at::cuda::getCurrentCUDAStream()>>>(        \
      reinterpret_cast<const half*>(x.data_ptr()),        \
      reinterpret_cast<const half*>(w.data_ptr()),        \
      reinterpret_cast<const half*>(residual.data_ptr()), \
      reinterpret_cast<half*>(out.data_ptr()), m)
  if (m > 8) {
    UP(16);
  } else {
    UP(8);
  }
#undef UP
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run_down(torch::Tensor lora, torch::Tensor inject, torch::Tensor x,
              torch::Tensor w, torch::Tensor tmp, int64_t split) {
  const c10::cuda::CUDAGuard guard(x.device());
  for (const auto& t : {lora, inject, x, w}) check_half(t, x);
  TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 16 &&
              x.size(1) == 10240);
  const int m = x.size(0);
  TORCH_CHECK(lora.numel() == m * 320 && inject.numel() == m * 4 &&
              w.numel() == 352 * 10240);
  TORCH_CHECK(tmp.is_cuda() && tmp.device() == x.device() &&
              tmp.is_contiguous() && tmp.scalar_type() == at::kFloat &&
              tmp.numel() >= split * m * 352);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define CASE(S)                                                             \
  case S:                                                                   \
    down_partial<S><<<dim3(11, (m + 7) / 8, S), 128, 0, stream>>>(          \
        reinterpret_cast<const half*>(x.data_ptr()),                        \
        reinterpret_cast<const half*>(w.data_ptr()), tmp.data_ptr<float>(), \
        m);                                                                 \
    down_tail<S><<<m, 352, 0, stream>>>(                                    \
        tmp.data_ptr<float>(), reinterpret_cast<half*>(lora.data_ptr()),    \
        reinterpret_cast<half*>(inject.data_ptr()), m);                     \
    break
  switch (split) {
    CASE(4);
    CASE(8);
    CASE(16);
    CASE(20);
    default:
      TORCH_CHECK(false, "Unsupported split");
  }
#undef CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace
TORCH_LIBRARY(sm70_hc_screen, m) {
  m.def("up(Tensor(a!) out, Tensor x, Tensor w, Tensor residual) -> ()");
  m.def(
      "down(Tensor(a!) lora, Tensor(b!) inject, Tensor x, Tensor w, Tensor(c!) "
      "tmp, int split) -> ()");
}
TORCH_LIBRARY_IMPL(sm70_hc_screen, CUDA, m) {
  m.impl("up", &run_up);
  m.impl("down", &run_down);
}
