// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Checkpoint-FP16 Qwen3.8 TP4 GDN input. No quantization or K-tree change.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
#define GDN_MMA(C, A0, A1, B0, B1)                                  \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

__global__ __launch_bounds__(128, 4) void gdn_input_batch_kernel(
    const half* x, const half* qw, const half* bw, half* qkv, half* z,
    half* b_out, half* a_out, int m) {
  __shared__ float partial[4][8][32];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const bool is_ba = blockIdx.x == 128;
  const int row_group = blockIdx.y;
  // The small b/a projection retains its four contiguous 640-element K
  // partitions. QKVZ retains the original single ordered K reduction.
  if (!is_ba && warp != 0) return;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int col = quad * 8 + r, row = row_group * 8 + r;
  const half* input = x + static_cast<size_t>(row) * 2560;
  const half* w = is_ba ? bw : qw + static_cast<size_t>(blockIdx.x) * 2560 * 32;
  const int start = is_ba ? warp * 40 : 0;
  const int end = is_ba ? (warp + 1) * 40 : 160;
  float acc[8] = {};
#pragma unroll 32
  for (int g = start; g < end; ++g) {
    const uint4 lo = *reinterpret_cast<const uint4*>(w + (g * 64 + col) * 8);
    const uint4 hi =
        *reinterpret_cast<const uint4*>(w + (g * 64 + 32 + col) * 8);
    uint4 a = make_uint4(0, 0, 0, 0), b = a;
    if (row < m) {
      a = *reinterpret_cast<const uint4*>(input + g * 16);
      b = *reinterpret_cast<const uint4*>(input + g * 16 + 8);
    }
    GDN_MMA(acc, a.x, a.y, lo.x, lo.y);
    GDN_MMA(acc, a.z, a.w, lo.z, lo.w);
    GDN_MMA(acc, b.x, b.y, hi.x, hi.y);
    GDN_MMA(acc, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int cc =
        quad * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
    if (is_ba) {
      partial[warp][rr][cc] = acc[i];
    } else if (row_group * 8 + rr < m) {
      const int outcol = blockIdx.x * 32 + cc;
      const half value = __float2half_rn(acc[i]);
      if (outcol < 2560)
        qkv[(row_group * 8 + rr) * 2560 + outcol] = value;
      else
        z[(row_group * 8 + rr) * 1536 + outcol - 2560] = value;
    }
  }
  if (is_ba) {
    __syncthreads();
    for (int i = threadIdx.x; i < 8 * 24; i += 128) {
      const int rr = i / 24, cc = i % 24;
      // cuBLASLt FP32 workspace reduction: preserve left-to-right order.
      // Balanced alternatives produce measurable FP16 bit differences.
      const float value =
          ((partial[0][rr][cc] + partial[1][rr][cc]) + partial[2][rr][cc]) +
          partial[3][rr][cc];
      if (row_group * 8 + rr < m) {
        if (cc < 12)
          b_out[(row_group * 8 + rr) * 12 + cc] = __float2half_rn(value);
        else
          a_out[(row_group * 8 + rr) * 12 + cc - 12] = __float2half_rn(value);
      }
    }
  }
}
#undef GDN_MMA

void gdn_input_batch(torch::Tensor qkv, torch::Tensor z, torch::Tensor b,
                     torch::Tensor a, torch::Tensor x, torch::Tensor qw,
                     torch::Tensor bw) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(1) == 2560 &&
                  x.size(0) >= 2 && x.size(0) <= 16,
              "Qwen3.8 batched GDN requires CUDA [2..16, 2560] input");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "Qwen3.8 batched GDN is SM70 only");
  for (const auto& t : {qkv, z, b, a, x, qw, bw}) {
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous() &&
                    t.scalar_type() == at::kHalf,
                "Qwen3.8 batched GDN requires same-device contiguous FP16");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "Qwen3.8 batched GDN requires 16-byte aligned storage");
  }
  TORCH_CHECK(qw.sizes() == at::IntArrayRef({128, 160, 2, 32, 8}) &&
                  bw.sizes() == at::IntArrayRef({1, 160, 2, 32, 8}),
              "Invalid packed Qwen3.8 GDN weight geometry");
  const int m = x.size(0);
  TORCH_CHECK(qkv.sizes() == at::IntArrayRef({m, 2560}) &&
                  z.sizes() == at::IntArrayRef({m, 1536}) &&
                  b.sizes() == at::IntArrayRef({m, 12}) &&
                  a.sizes() == at::IntArrayRef({m, 12}),
              "Invalid Qwen3.8 GDN output geometry");
  gdn_input_batch_kernel<<<dim3(129, (m + 7) / 8), 128, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(qw.data_ptr()),
      reinterpret_cast<const half*>(bw.data_ptr()),
      reinterpret_cast<half*>(qkv.data_ptr()),
      reinterpret_cast<half*>(z.data_ptr()),
      reinterpret_cast<half*>(b.data_ptr()),
      reinterpret_cast<half*>(a.data_ptr()), m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "qwen38_gdn_input_batch_sm70_out(Tensor(a!) qkv, Tensor(b!) z, "
      "Tensor(c!) b, Tensor(d!) a, Tensor x, Tensor qw, Tensor bw) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen38_gdn_input_batch_sm70_out", &gdn_input_batch);
}
