// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Mixed-CTA projection follows fp8_qpn8_sm70.cu (v100-skinny, MIT).
// See LICENSE.v100-skinny. Decode formulas come from gguf_lattice_compact.cuh.
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "gguf_lattice_compact.cuh"

#define MMA(C, A0, A1, B0, B1)                                      \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))
namespace {
template <int Split, bool Fused>
__global__ void gguf_gdn_input_kernel(half* qkv, half* z, half* b, half* a,
                                      const half* input, const uint8_t* codes,
                                      const half* ba) {
  constexpr int K = 5120, N = 4096, Q = 2560, Z = 1536;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tile = blockIdx.x;
  __shared__ float partial[Split][256];
  if constexpr (Fused) {
    if (tile >= N / 32) {
      // The same two-row FP32 scalar reduction used by the qualified FP8 route.
      const int token = (tile - N / 32) / 12, group = threadIdx.x / 256;
      const int row = ((tile - N / 32) % 12) * 2 + group, t = threadIdx.x % 256;
      float value = 0;
      const half2* x = reinterpret_cast<const half2*>(input + token * K);
      const half2* w = reinterpret_cast<const half2*>(ba + row * K);
      for (int pair = t; pair < K / 2; pair += 256) {
        const float2 xv = __half22float2(__ldg(x + pair));
        const float2 wv = __half22float2(__ldg(w + pair));
        value = fmaf(xv.x, wv.x, value);
        value = fmaf(xv.y, wv.y, value);
      }
      for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
      if (lane == 0) partial[group][(t >> 5)] = value;
      __syncthreads();
      if ((t >> 5) == 0) {
        value = lane < 8 ? partial[group][lane] : 0;
        for (int offset = 16; offset; offset >>= 1)
          value += __shfl_down_sync(0xffffffff, value, offset);
        if (lane == 0) {
          if (row < 12)
            b[token * 12 + row] = __float2half_rn(value);
          else
            a[token * 12 + row - 12] = __float2half_rn(value);
        }
      }
      return;
    }
  }
  using D = vllm::sm70_gguf::LatticeCompactDecoder<21>;
  __shared__ __align__(16) uint8_t grid[D::kCodebookBytes];
  D::initialize(grid);
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0), quad = (lane >> 2) & 3,
            col = quad * 8 + r;
  const int count = (K / 16) / Split, begin = warp * count;
  const uint8_t* macro = codes + tile * (K / 256) * 32 * 110;
  const half* x = input + r * K;
  float accum[8] = {};
  typename D::Parameters params{};
  int loaded = -1;
  for (int g = begin; g < begin + count; ++g) {
    const int block = g / 16, octet = (g % 16) * 2;
    const uint8_t* tiledata = macro + block * 32 * 110;
    if (block != loaded) {
      params = D::parameters<true>(tiledata, 32, col);
      loaded = block;
    }
    const uint32_t* p = reinterpret_cast<const uint32_t*>(tiledata) +
                        octet * 26 + (col * 26 >> 5);
    const uint32_t p0 =
        __funnelshift_r(__ldcs(p), __ldcs(p + 1), (col * 26) & 31) & 0x3ffffff;
    const uint32_t p1 =
        __funnelshift_r(__ldcs(p + 26), __ldcs(p + 27), (col * 26) & 31) &
        0x3ffffff;
    const auto w0 = D::fragment<half>(params, p0, octet, grid);
    const auto w1 = D::fragment<half>(params, p1, octet, grid);
    const uint4 v0 = *reinterpret_cast<const uint4*>(&w0),
                v1 = *reinterpret_cast<const uint4*>(&w1);
    const uint4 x0 = *reinterpret_cast<const uint4*>(x + g * 16),
                x1 = *reinterpret_cast<const uint4*>(x + g * 16 + 8);
    MMA(accum, x0.x, x0.y, v0.x, v0.y);
    MMA(accum, x0.z, x0.w, v0.z, v0.w);
    MMA(accum, x1.x, x1.y, v1.x, v1.y);
    MMA(accum, x1.z, x1.w, v1.z, v1.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int cc =
        quad * 8 + (i & 1) + (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
    partial[warp][rr * 32 + cc] = accum[i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 256; e += blockDim.x) {
    float value = 0;
#pragma unroll
    for (int s = 0; s < Split; ++s) value += partial[s][e];
    const int row = e / 32, col = tile * 32 + e % 32;
    if (col < Q)
      qkv[row * Q + col] = __float2half_rn(value);
    else
      z[row * Z + col - Q] = __float2half_rn(value);
  }
}
void run(torch::Tensor qkv, torch::Tensor z, torch::Tensor b, torch::Tensor a,
         torch::Tensor x, torch::Tensor codes, torch::Tensor ba, bool fused) {
  TORCH_CHECK(x.is_cuda() && x.sizes() == at::IntArrayRef({8, 5120}) &&
              x.scalar_type() == at::kHalf);
  const c10::cuda::CUDAGuard guard(x.device());
  for (const auto& t : {qkv, z, b, a, x, ba})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous() &&
                t.scalar_type() == at::kHalf);
  TORCH_CHECK(qkv.sizes() == at::IntArrayRef({8, 2560}) &&
              z.sizes() == at::IntArrayRef({8, 1536}) &&
              b.sizes() == at::IntArrayRef({8, 12}) && a.sizes() == b.sizes());
  TORCH_CHECK(codes.device() == x.device() &&
              codes.scalar_type() == at::kByte && codes.is_contiguous() &&
              codes.numel() == 4096 * 20 * 110 &&
              ba.sizes() == at::IntArrayRef({24, 5120}));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (fused)
    gguf_gdn_input_kernel<16, true><<<224, 512, 0, stream>>>(
        (half*)qkv.data_ptr(), (half*)z.data_ptr(), (half*)b.data_ptr(),
        (half*)a.data_ptr(), (const half*)x.data_ptr(),
        codes.data_ptr<uint8_t>(), (const half*)ba.data_ptr());
  else
    gguf_gdn_input_kernel<16, false><<<128, 512, 0, stream>>>(
        (half*)qkv.data_ptr(), (half*)z.data_ptr(), (half*)b.data_ptr(),
        (half*)a.data_ptr(), (const half*)x.data_ptr(),
        codes.data_ptr<uint8_t>(), (const half*)ba.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace
#undef MMA
#ifdef GGUF_GDN_INPUT_RESEARCH
TORCH_LIBRARY(gguf_gdn_input_research, m) {
  m.def(
      "run(Tensor(a!) qkv, Tensor(b!) z, Tensor(c!) b, Tensor(d!) a, Tensor x, "
      "Tensor codes, Tensor ba, bool fused) -> ()");
}
TORCH_LIBRARY_IMPL(gguf_gdn_input_research, CUDA, m) { m.impl("run", &run); }
#endif
