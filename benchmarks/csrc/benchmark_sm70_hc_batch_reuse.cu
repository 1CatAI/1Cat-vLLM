// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research-only exact-shape HC projection screens. Not runtime dispatch.
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
// HC down's high-precision cuBLAS route uses twenty FP32 K=512
// partitions. Keep those partitions independent until the ordered finish.
// Unlike the up projection, each quad pair owns adjacent output columns.
template <bool PairRows, int Warps, bool WarpM16, int N = 352>
__global__ __launch_bounds__(32 * Warps, 4) void hc_down_partials(
    const half* __restrict__ x, const half* __restrict__ packed,
    float* __restrict__ partials, int rows) {
  constexpr int TileN = WarpM16 ? 16 : 32;
  const int lane = threadIdx.x % 32;
  const int tile = blockIdx.x * Warps + threadIdx.x / 32;
  if (tile >= N / TileN) return;
  const int group = (PairRows || WarpM16) ? 0 : blockIdx.y;
  const int split = blockIdx.z;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int output_quad = WarpM16 ? quad % 2 : quad;
  const int col = output_quad * 8 + r;
  const int row_base = WarpM16 ? (quad / 2) * 8 : group * 8;
  const half* w = packed + static_cast<size_t>(tile) * 10240 * TileN;
  float accum[PairRows ? 2 : 1][8] = {};
#pragma unroll 4
  for (int g = 0; g < 32; ++g) {
    const int kg = split * 32 + g;
    const uint4 lo =
        *reinterpret_cast<const uint4*>(w + (kg * 2 * TileN + col) * 8);
    const uint4 hi =
        *reinterpret_cast<const uint4*>(w + (kg * 2 * TileN + TileN + col) * 8);
#pragma unroll
    for (int p = 0; p < (PairRows ? 2 : 1); ++p) {
      const int row = row_base + p * 8 + r;
      uint4 a = make_uint4(0, 0, 0, 0), b = a;
      if (row < rows) {
        const half* input = x + row * 10240 + kg * 16;
        a = *reinterpret_cast<const uint4*>(input);
        b = *reinterpret_cast<const uint4*>(input + 8);
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
          row_base + p * 8 + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
      const int n = tile * TileN + output_quad * 8 +
                    ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      if (row < rows) {
        partials[(split * rows + row) * N + n] = accum[p][i];
      }
    }
  }
}

template <bool WriteProjection>
__global__ void hc_down_finish(const float* __restrict__ partials,
                               half* __restrict__ projection,
                               half* __restrict__ lora,
                               half* __restrict__ injection, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = i / 336;
  const int col = i % 336;
  if (row >= rows) return;
  float acc = 0.0f;
#pragma unroll
  for (int split = 0; split < 20; ++split) {
    acc += partials[(split * rows + row) * 352 + col];
  }
  // GEMM's FP16 output boundary precedes SiLU. Removing it is not allowed.
  const half rounded = __float2half_rn(acc);
  if constexpr (WriteProjection) projection[i] = rounded;
  if (col < 320) {
    const float v = div_full(__half2float(rounded), 4.0f);
    lora[row * 320 + col] = __float2half_rn(v * sigmoid(v));
  } else if (col < 324) {
    injection[row * 4 + col - 320] = rounded;
  }
}

void run_down(torch::Tensor input, torch::Tensor packed, torch::Tensor partials,
              torch::Tensor projection, torch::Tensor lora,
              torch::Tensor injection, bool paired, int64_t warps,
              bool write_projection, bool warp_m16) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.size(1) == 10240);
  const c10::cuda::CUDAGuard guard(input.device());
  const int rows = input.size(0);
  TORCH_CHECK(rows >= 2 && rows <= 16 && (warps == 1 || warps == 4));
  TORCH_CHECK(!warp_m16 || (!paired && warps == 1));
  for (const auto& t : {input, packed, projection, lora, injection}) {
    TORCH_CHECK(t.is_cuda() && t.device() == input.device() &&
                t.scalar_type() == torch::kFloat16 && t.is_contiguous());
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0);
  }
  TORCH_CHECK(packed.numel() == 352 * 10240);
  TORCH_CHECK(partials.is_cuda() && partials.device() == input.device() &&
              partials.scalar_type() == torch::kFloat32 &&
              partials.is_contiguous());
  TORCH_CHECK(partials.sizes() == torch::IntArrayRef({20, rows, 352}));
  TORCH_CHECK(projection.sizes() == torch::IntArrayRef({rows, 336}));
  TORCH_CHECK(lora.sizes() == torch::IntArrayRef({rows, 320}));
  TORCH_CHECK(injection.sizes() == torch::IntArrayRef({rows, 4}));
  const auto stream = at::cuda::getCurrentCUDAStream();
#define DOWN(P, W, M)                                                          \
  hc_down_partials<P, W, M>                                                    \
      <<<dim3(((M ? 22 : 11) + W - 1) / W, (P || M) ? 1 : (rows + 7) / 8, 20), \
         32 * W, 0, stream>>>(                                                 \
          reinterpret_cast<const half*>(input.data_ptr()),                     \
          reinterpret_cast<const half*>(packed.data_ptr()),                    \
          partials.data_ptr<float>(), rows)
#define DOWN_WARPS(P)  \
  if (warps == 1) {    \
    DOWN(P, 1, false); \
  } else {             \
    DOWN(P, 4, false); \
  }
  if (warp_m16) {
    DOWN(false, 1, true);
  } else if (paired) {
    DOWN_WARPS(true);
  } else {
    DOWN_WARPS(false);
  }
#undef DOWN_WARPS
#undef DOWN
#define FINISH(P)                                                  \
  hc_down_finish<P><<<(rows * 336 + 127) / 128, 128, 0, stream>>>( \
      partials.data_ptr<float>(),                                  \
      reinterpret_cast<half*>(projection.data_ptr()),              \
      reinterpret_cast<half*>(lora.data_ptr()),                    \
      reinterpret_cast<half*>(injection.data_ptr()), rows)
  if (write_projection) {
    FINISH(true);
  } else {
    FINISH(false);
  }
#undef FINISH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The TP4 screen reduces these partials inside its gather kernel. Keeping
// the shard's eighty LoRA columns plus the next eight original columns
// matches the already-audited rank3 injection ownership of the HC gather.
void run_down_shard(torch::Tensor input, torch::Tensor packed,
                    torch::Tensor partials) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.size(1) == 10240);
  const c10::cuda::CUDAGuard guard(input.device());
  const int rows = input.size(0);
  TORCH_CHECK(rows >= 2 && rows <= 16);
  for (const auto& t : {input, packed}) {
    TORCH_CHECK(t.is_cuda() && t.device() == input.device() &&
                t.scalar_type() == at::kHalf && t.is_contiguous());
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0);
  }
  TORCH_CHECK(packed.numel() == 96 * 10240);
  TORCH_CHECK(partials.is_cuda() && partials.device() == input.device() &&
              partials.scalar_type() == at::kFloat &&
              partials.is_contiguous() &&
              partials.sizes() == torch::IntArrayRef({20, rows, 96}));
  hc_down_partials<false, 1, false, 96><<<dim3(3, (rows + 7) / 8, 20), 32, 0,
                                          at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(input.data_ptr()),
      reinterpret_cast<const half*>(packed.data_ptr()),
      partials.data_ptr<float>(), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("run_down", &run_down);
  m.def("run_down_shard", &run_down_shard);
}
