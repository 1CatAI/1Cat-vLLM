// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research-only exact-shape HC up/mix screen. Not runtime dispatch.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {

__device__ __forceinline__ void mma(float (&d)[8], uint32_t a0, uint32_t a1,
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// Match the Triton HC post-op, not CUDA's --use_fast_math division.
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0,%1,%2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float sigmoid(float x) {
  float e, d;
  asm("mul.f32 %0,%1,0fBFB8AA3B;" : "=f"(e) : "f"(x));
  asm("ex2.approx.f32 %0,%1;" : "=f"(e) : "f"(e));
  asm("add.f32 %0,%1,0f3F800000;" : "=f"(d) : "f"(e));
  return div_full(1.0f, d);
}

// Each quad pair computes one branch's 8x8 output. All four branch values
// for a hidden column live at identical lane offsets in the four quad pairs.
// PairRows shares the same 16-byte weight load across two independent M8
// accumulators. The K sequence in each accumulator is unchanged.
template <bool PairRows, int Warps, int Unroll, bool FuseMix>
__global__ __launch_bounds__(32 * Warps, 4) void hc_up_batch(
    const half* __restrict__ lora, const half* __restrict__ packed,
    const half* __restrict__ branches, half* __restrict__ out, int rows,
    int hidden, int hidden_offset) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int tile = blockIdx.x * Warps + warp;
  if (tile >= hidden / 8) return;
  const int group = PairRows ? 0 : blockIdx.y;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int branch = (lane >> 2) & 3;
  const int col = branch * 8 + r;
  const half* w = packed + static_cast<size_t>(tile) * 320 * 32;
  float accum[PairRows ? 2 : 1][8] = {};
#pragma unroll Unroll
  for (int g = 0; g < 20; ++g) {
    const uint4 lo = *reinterpret_cast<const uint4*>(w + (g * 64 + col) * 8);
    const uint4 hi =
        *reinterpret_cast<const uint4*>(w + (g * 64 + 32 + col) * 8);
#pragma unroll
    for (int p = 0; p < (PairRows ? 2 : 1); ++p) {
      const int row = (group + p) * 8 + r;
      uint4 a = make_uint4(0, 0, 0, 0), b = a;
      if (row < rows) {
        const half* x = lora + row * 320 + g * 16;
        a = *reinterpret_cast<const uint4*>(x);
        b = *reinterpret_cast<const uint4*>(x + 8);
      }
      mma(accum[p], a.x, a.y, lo.x, lo.y);
      mma(accum[p], a.z, a.w, lo.z, lo.w);
      mma(accum[p], b.x, b.y, hi.x, hi.y);
      mma(accum[p], b.z, b.w, hi.z, hi.w);
    }
  }
#pragma unroll
  for (int p = 0; p < (PairRows ? 2 : 1); ++p) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row =
          (group + p) * 8 + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
      const int h =
          tile * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      const half gate = __float2half_rn(accum[p][i]);
      if constexpr (FuseMix) {
        const float gate_f = __half2float(gate);
        float v = 0.0f;
        if (row < rows) {
          v = __half2float(
              branches[row * 10240 + branch * 2560 + hidden_offset + h]);
        }
        const float s = sigmoid(gate_f);
        float mixed = 0.0f;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int src = (lane & ~12) | (b << 2);
          const float gs = __shfl_sync(0xffffffff, s, src);
          const float x = __shfl_sync(0xffffffff, v, src);
          mixed = fmaf(gs, x, mixed);
        }
        if (branch == 0 && row < rows) {
          out[row * hidden + h] = __float2half_rn(div_full(mixed, 4.0f));
        }
      } else if (row < rows) {
        out[row * 4 * hidden + branch * hidden + h] = gate;
      }
    }
  }
}

void run(torch::Tensor lora, torch::Tensor packed, torch::Tensor branches,
         torch::Tensor output, int64_t hidden_offset, bool paired,
         int64_t warps, int64_t unroll, bool fuse_mix) {
  TORCH_CHECK(lora.is_cuda() && lora.dim() == 2 && lora.size(1) == 320);
  const c10::cuda::CUDAGuard guard(lora.device());
  const int m = lora.size(0);
  TORCH_CHECK(m >= 2 && m <= 16);
  const int hidden = packed.numel() / (4 * 320);
  TORCH_CHECK((hidden == 640 || hidden == 2560) && hidden_offset >= 0 &&
              hidden_offset + hidden <= 2560);
  for (const auto& t : {lora, packed, branches, output}) {
    TORCH_CHECK(t.is_cuda() && t.device() == lora.device() &&
                t.scalar_type() == torch::kFloat16 && t.is_contiguous());
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0);
  }
  TORCH_CHECK(branches.dim() == 2 && branches.size(0) == m &&
              branches.size(1) == 10240);
  TORCH_CHECK(output.dim() == 2 && output.size(0) == m &&
              output.size(1) == hidden * (fuse_mix ? 1 : 4));
  TORCH_CHECK(warps == 1 || warps == 4);
  TORCH_CHECK(unroll == 4 || unroll == 8);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(P, W, U, F)                                                \
  hc_up_batch<P, W, U, F>                                                 \
      <<<dim3((hidden / 8 + W - 1) / W, P ? 1 : (m + 7) / 8), 32 * W, 0,  \
         stream>>>(reinterpret_cast<const half*>(lora.data_ptr()),        \
                   reinterpret_cast<const half*>(packed.data_ptr()),      \
                   reinterpret_cast<const half*>(branches.data_ptr()),    \
                   reinterpret_cast<half*>(output.data_ptr()), m, hidden, \
                   hidden_offset)
#define FUSION(P, W, U)     \
  if (fuse_mix) {           \
    LAUNCH(P, W, U, true);  \
  } else {                  \
    LAUNCH(P, W, U, false); \
  }
#define UNROLL(P, W) \
  if (unroll == 4) { \
    FUSION(P, W, 4); \
  } else {           \
    FUSION(P, W, 8); \
  }
#define WARPS(P)    \
  if (warps == 1) { \
    UNROLL(P, 1);   \
  } else {          \
    UNROLL(P, 4);   \
  }
  if (paired) {
    WARPS(true);
  } else {
    WARPS(false);
  }
#undef WARPS
#undef UNROLL
#undef FUSION
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
