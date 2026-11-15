// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Checkpoint-FP16 Qwen3.8 TP4 GDN input. No quantization or K-tree change.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>
#include <type_traits>

namespace {
#define GDN_MMA(C, A0, A1, B0, B1)                                  \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <bool Packed, int Groups>
__global__ __launch_bounds__(128, 4) void gdn_input_batch_kernel(
    const half* x, const half* qw, const half* bw, half* qkv, half* z,
    half* b_out, half* a_out, int m) {
  __shared__ float partial[4][Groups * 8][32];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const bool is_ba = blockIdx.x == 128;
  const int tile = blockIdx.x;
  // The small b/a projection retains its four contiguous 640-element K
  // partitions. QKVZ spreads 128 N32 tiles across all 80 SMs and retains
  // the original single ordered K reduction while reusing weights across M.
  if (!is_ba && warp != 0) return;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int col = quad * 8 + r;
  const half* w = is_ba
                      ? bw
                      : (Packed ? (qw + static_cast<size_t>(tile) * 2560 * 32)
                                : (qw + static_cast<size_t>(tile) * 32 * 2560));
  const int start = is_ba ? warp * 40 : 0;
  const int end = is_ba ? (warp + 1) * 40 : 160;
  float acc[Groups][8] = {};
#pragma unroll 16
  for (int g = start; g < end; ++g) {
    uint4 lo = make_uint4(0, 0, 0, 0), hi = lo;
    if constexpr (Packed) {
      lo = *reinterpret_cast<const uint4*>(w + (g * 64 + col) * 8);
      hi = *reinterpret_cast<const uint4*>(w + (g * 64 + 32 + col) * 8);
    } else if (!is_ba || col < 24) {
      // Same fragments and K order, but read the original checkpoint layout.
      lo = *reinterpret_cast<const uint4*>(w + col * 2560 + g * 16);
      hi = *reinterpret_cast<const uint4*>(w + col * 2560 + g * 16 + 8);
    }
#pragma unroll
    for (int p = 0; p < Groups; ++p) {
      const int row = p * 8 + r;
      uint4 a = make_uint4(0, 0, 0, 0), b = a;
      if (row < m) {
        const half* input = x + static_cast<size_t>(row) * 2560 + g * 16;
        a = *reinterpret_cast<const uint4*>(input);
        b = *reinterpret_cast<const uint4*>(input + 8);
      }
      GDN_MMA(acc[p], a.x, a.y, lo.x, lo.y);
      GDN_MMA(acc[p], a.z, a.w, lo.z, lo.w);
      GDN_MMA(acc[p], b.x, b.y, hi.x, hi.y);
      GDN_MMA(acc[p], b.z, b.w, hi.z, hi.w);
    }
  }
#pragma unroll
  for (int p = 0; p < Groups; ++p) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int rr = p * 8 + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
      const int cc =
          quad * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      if (is_ba) {
        partial[warp][rr][cc] = acc[p][i];
      } else if (rr < m) {
        const int outcol = tile * 32 + cc;
        const half value = __float2half_rn(acc[p][i]);
        if (outcol < 2560)
          qkv[rr * 2560 + outcol] = value;
        else
          z[rr * 1536 + outcol - 2560] = value;
      }
    }
  }
  if (is_ba) {
    __syncthreads();
    for (int i = threadIdx.x; i < Groups * 8 * 24; i += 128) {
      const int rr = i / 24, cc = i % 24;
      // cuBLASLt FP32 workspace reduction: preserve left-to-right order.
      // Balanced alternatives produce measurable FP16 bit differences.
      const float value =
          ((partial[0][rr][cc] + partial[1][rr][cc]) + partial[2][rr][cc]) +
          partial[3][rr][cc];
      if (rr < m) {
        if (cc < 12)
          b_out[rr * 12 + cc] = __float2half_rn(value);
        else
          a_out[rr * 12 + cc - 12] = __float2half_rn(value);
      }
    }
  }
}

// Both high-precision small-batch cuBLAS projections use four contiguous
// FP32 K partitions (router: K640, output: K384). Each independent m8n8k4
// quad pair owns one partition, with an ordered warp-shuffle reduction.
// N8 tiles distribute work over more SMs without adding weight copies or
// changing the dot-product tree. This direct entry remains screening-only.
template <int K>
__global__ __launch_bounds__(32, 8) void dense_batch_kernel(const half* x,
                                                            const half* weight,
                                                            half* output, int m,
                                                            int n) {
  const int lane = threadIdx.x;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int split = (lane >> 2) & 3;
  const int col = blockIdx.x * 8 + r;
  const int row = blockIdx.y * 8 + r;
  float acc[8] = {};
#pragma unroll 4
  for (int g = split * (K / 4 / 16); g < (split + 1) * (K / 4 / 16); ++g) {
    const half* w = weight + static_cast<size_t>(col) * K + g * 16;
    const uint4 lo = *reinterpret_cast<const uint4*>(w);
    const uint4 hi = *reinterpret_cast<const uint4*>(w + 8);
    uint4 a = make_uint4(0, 0, 0, 0), b = a;
    if (row < m) {
      const half* input = x + static_cast<size_t>(row) * K + g * 16;
      a = *reinterpret_cast<const uint4*>(input);
      b = *reinterpret_cast<const uint4*>(input + 8);
    }
    GDN_MMA(acc, a.x, a.y, lo.x, lo.y);
    GDN_MMA(acc, a.z, a.w, lo.z, lo.w);
    GDN_MMA(acc, b.x, b.y, hi.x, hi.y);
    GDN_MMA(acc, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int cc = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    const int base = lane & ~12;
    float value = __shfl_sync(0xffffffff, acc[i], base);
#pragma unroll
    for (int s = 1; s < 4; ++s)
      value =
          __fadd_rn(value, __shfl_sync(0xffffffff, acc[i], base | (s << 2)));
    if (split == 0 && blockIdx.y * 8 + rr < m) {
      output[(blockIdx.y * 8 + rr) * n + blockIdx.x * 8 + cc] =
          __float2half_rn(value);
    }
  }
}
template <bool RowMajor>
__global__ __launch_bounds__(32, 8) void router_split_quad_kernel(const half* x,
                                                                  const half* w,
                                                                  half* out,
                                                                  int m) {
  const int lane = threadIdx.x;
  const int col_tile = blockIdx.x;
  const int row_base = blockIdx.y * 8;
  const int split = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row = row_base + r;
  float acc[8] = {};
#pragma unroll 10
  for (int g = 0; g < 40; ++g) {
    const half* weights;
    if constexpr (RowMajor)
      weights = w + (col_tile * 8 + r) * 2560 + split * 640 + g * 16;
    else
      weights = w + (col_tile * 40 * 64 + g * 64 + split * 8 + r) * 8;
    const uint4 lo = *reinterpret_cast<const uint4*>(weights);
    const uint4 hi =
        *reinterpret_cast<const uint4*>(weights + (RowMajor ? 8 : 32 * 8));
    uint4 a = {}, b = {};
    if (row < m) {
      const half* input = x + row * 2560 + split * 640 + g * 16;
      a = *reinterpret_cast<const uint4*>(input);
      b = *reinterpret_cast<const uint4*>(input + 8);
    }
    GDN_MMA(acc, a.x, a.y, lo.x, lo.y);
    GDN_MMA(acc, a.z, a.w, lo.z, lo.w);
    GDN_MMA(acc, b.x, b.y, hi.x, hi.y);
    GDN_MMA(acc, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = row_base + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
    const int cc =
        col_tile * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
    float value = __shfl_sync(0xffffffff, acc[i], lane & ~12);
#pragma unroll
    for (int s = 1; s < 4; ++s)
      value = __fadd_rn(
          value, __shfl_sync(0xffffffff, acc[i], (lane & ~12) | (s << 2)));
    if (split == 0 && rr < m) out[rr * 512 + cc] = __float2half_rn(value);
  }
}

__device__ __forceinline__ float partial_float(half value) {
  return __half2float(value);
}
__device__ __forceinline__ float partial_float(float value) { return value; }

// The shared expert's original cuBLASLt projection uses eight K320
// partitions. Keep the exact FP16/FP32 partial rounding and left-to-right
// split sum inside shared memory with one 8-warp CTA per (N32 gate+up pair,
// 8-row group) so concurrent batches parallelize across SMs while hitting L2.
template <typename Partial>
__global__ __launch_bounds__(256, 4) void shared_up_fused_silu_kernel(
    const half* x, const half* w, half* out, int m) {
  __shared__ Partial smem_part[8][8][32];
  const int split = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int col = quad * 8 + r;
  const int row_base = blockIdx.y * 8;
  const int row = row_base + r;
  float acc_g[8] = {};
  float acc_u[8] = {};
#pragma unroll 10
  for (int g = split * 20; g < (split + 1) * 20; ++g) {
    const int off_g = (blockIdx.x * 160 + g) * 512 + col * 8;
    const int off_u = ((blockIdx.x + 5) * 160 + g) * 512 + col * 8;
    const uint4 lo_g = *reinterpret_cast<const uint4*>(w + off_g);
    const uint4 hi_g = *reinterpret_cast<const uint4*>(w + off_g + 256);
    const uint4 lo_u = *reinterpret_cast<const uint4*>(w + off_u);
    const uint4 hi_u = *reinterpret_cast<const uint4*>(w + off_u + 256);
    uint4 a = {}, b = {};
    if (row < m) {
      a = *reinterpret_cast<const uint4*>(x + row * 2560 + g * 16);
      b = *reinterpret_cast<const uint4*>(x + row * 2560 + g * 16 + 8);
    }
    GDN_MMA(acc_g, a.x, a.y, lo_g.x, lo_g.y);
    GDN_MMA(acc_g, a.z, a.w, lo_g.z, lo_g.w);
    GDN_MMA(acc_g, b.x, b.y, hi_g.x, hi_g.y);
    GDN_MMA(acc_g, b.z, b.w, hi_g.z, hi_g.w);
    GDN_MMA(acc_u, a.x, a.y, lo_u.x, lo_u.y);
    GDN_MMA(acc_u, a.z, a.w, lo_u.z, lo_u.w);
    GDN_MMA(acc_u, b.x, b.y, hi_u.x, hi_u.y);
    GDN_MMA(acc_u, b.z, b.w, hi_u.z, hi_u.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int cc =
        quad * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
    if constexpr (std::is_same_v<Partial, half>)
      smem_part[split][rr][cc] = __float2half_rn(acc_g[i]);
    else
      smem_part[split][rr][cc] = acc_g[i];
  }
  __syncthreads();
  half gate_saved = __float2half_rn(0.f);
  const int rr_t = threadIdx.x >> 5, cc_t = threadIdx.x & 31;
  if (row_base + rr_t < m) {
    float gate = partial_float(smem_part[0][rr_t][cc_t]);
#pragma unroll
    for (int s = 1; s < 8; ++s)
      gate = __fadd_rn(gate, partial_float(smem_part[s][rr_t][cc_t]));
    gate_saved = __float2half_rn(gate);
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int cc =
        quad * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
    if constexpr (std::is_same_v<Partial, half>)
      smem_part[split][rr][cc] = __float2half_rn(acc_u[i]);
    else
      smem_part[split][rr][cc] = acc_u[i];
  }
  __syncthreads();
  if (row_base + rr_t < m) {
    float up = partial_float(smem_part[0][rr_t][cc_t]);
#pragma unroll
    for (int s = 1; s < 8; ++s)
      up = __fadd_rn(up, partial_float(smem_part[s][rr_t][cc_t]));
    const half u = __float2half_rn(up);
    const float value = __half2float(gate_saved);
    const half activated =
        __float2half_rn(value / __fadd_rn(1.f, expf(-value)));
    out[(row_base + rr_t) * 160 + blockIdx.x * 32 + cc_t] =
        __hmul(activated, u);
  }
}

__global__ void shared_gate_mul_kernel(const half* logits, const half* source,
                                       half* out, int m) {
  const int index = blockIdx.x * 128 + threadIdx.x;
  if (index >= m * 2560) return;
  const float value = __half2float(logits[index / 2560]);
  const half factor = __float2half_rn(1.f / __fadd_rn(1.f, expf(-value)));
  out[index] = __hmul(source[index], factor);
}

__device__ __forceinline__ float warp_sum_down(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val = __fadd_rn(val, __shfl_down_sync(0xffffffffu, val, offset));
  return val;
}

template <int K_DOWN>
__global__ __launch_bounds__(128, 4) void shared_down_gated_m1_kernel(
    const half* __restrict__ hidden_x, const half* __restrict__ act_out,
    const half* __restrict__ down_w, const half* __restrict__ gate_w,
    half* __restrict__ out) {
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  float gsum = 0.0f;
#pragma unroll
  for (int item = 0; item < 20; ++item) {
    const int idx = tid + item * 128;
    gsum =
        __fmaf_rn(__half2float(hidden_x[idx]), __half2float(gate_w[idx]), gsum);
  }
  gsum = warp_sum_down(gsum);
  __shared__ float gate_partials[4];
  __shared__ half shared_gate;
  __shared__ float act_s[K_DOWN];
  if (lane == 0) gate_partials[warp] = gsum;
  for (int k = tid; k < K_DOWN; k += 128) act_s[k] = __half2float(act_out[k]);
  __syncthreads();
  if (tid == 0) {
    const float tot =
        __fadd_rn(__fadd_rn(__fadd_rn(gate_partials[0], gate_partials[1]),
                            gate_partials[2]),
                  gate_partials[3]);
    const float rounded = __half2float(__float2half_rn(tot));
    shared_gate = __float2half_rn(1.0f / (1.0f + __expf(-rounded)));
  }
  __syncthreads();
  const int sub = tid & 3;
  const int r_local = tid >> 2;
  const int col = blockIdx.x * 32 + r_local;
  const half* wrow = down_w + static_cast<size_t>(col) * K_DOWN;
  constexpr int kVecPerSub = K_DOWN / 32;
  float acc = 0.0f;
#pragma unroll
  for (int v = 0; v < kVecPerSub; ++v) {
    const int k = (v * 4 + sub) * 8;
    const uint4 wv = *reinterpret_cast<const uint4*>(wrow + k);
    const half* wh = reinterpret_cast<const half*>(&wv);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      acc = __fmaf_rn(act_s[k + j], __half2float(wh[j]), acc);
  }
  acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, 2));
  acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, 1));
  if (sub == 0) out[col] = __hmul(__float2half_rn(acc), shared_gate);
}

template <int K_DOWN>
__global__ __launch_bounds__(64, 8) void shared_down_gated_batch_kernel(
    const half* __restrict__ hidden_x, const half* __restrict__ act_out,
    const half* __restrict__ down_w, const half* __restrict__ gate_w,
    half* __restrict__ out, int m) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int col_base = blockIdx.x * 32;
  const int row_base = blockIdx.y * 8;
  const int gm = min(8, m - row_base);

  __shared__ half gate_s[8];
  __shared__ __align__(16) half act_s[8][K_DOWN + 8];
  __shared__ float red[8][32];

  uint4 gw[10];
#pragma unroll
  for (int s = 0; s < 10; ++s)
    gw[s] = *reinterpret_cast<const uint4*>(gate_w + (s * 32 + lane) * 8);

  for (int rr = warp * 4; rr < warp * 4 + 4; ++rr) {
    if (rr < gm) {
      const half* xrow = hidden_x + static_cast<size_t>(row_base + rr) * 2560;
      float gsum = 0.0f;
#pragma unroll
      for (int s = 0; s < 10; ++s) {
        const uint4 xv =
            *reinterpret_cast<const uint4*>(xrow + (s * 32 + lane) * 8);
        const half* xh = reinterpret_cast<const half*>(&xv);
        const half* wh = reinterpret_cast<const half*>(&gw[s]);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          gsum = __fmaf_rn(__half2float(xh[j]), __half2float(wh[j]), gsum);
      }
      gsum = warp_sum_down(gsum);
      if (lane == 0) {
        const float rounded = __half2float(__float2half_rn(gsum));
        gate_s[rr] = __float2half_rn(1.0f / (1.0f + __expf(-rounded)));
      }
    }
  }

  for (int idx = tid * 8; idx < gm * K_DOWN; idx += 64 * 8) {
    const int rr = idx / K_DOWN, kk = idx % K_DOWN;
    *reinterpret_cast<uint4*>(&act_s[rr][kk]) = *reinterpret_cast<const uint4*>(
        act_out + static_cast<size_t>(row_base + rr) * K_DOWN + kk);
  }
  __syncthreads();

  constexpr int kHalfSteps = K_DOWN / 32;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int col = col_base + quad * 8 + r;
  const half* wrow = down_w + static_cast<size_t>(col) * K_DOWN;
  float acc[8] = {};
#pragma unroll
  for (int s = 0; s < kHalfSteps; ++s) {
    const int g = warp * kHalfSteps + s;
    const uint4 lo = *reinterpret_cast<const uint4*>(wrow + g * 16);
    const uint4 hi = *reinterpret_cast<const uint4*>(wrow + g * 16 + 8);
    uint4 a = make_uint4(0, 0, 0, 0), b = a;
    if (r < gm) {
      a = *reinterpret_cast<const uint4*>(&act_s[r][g * 16]);
      b = *reinterpret_cast<const uint4*>(&act_s[r][g * 16 + 8]);
    }
    GDN_MMA(acc, a.x, a.y, lo.x, lo.y);
    GDN_MMA(acc, a.z, a.w, lo.z, lo.w);
    GDN_MMA(acc, b.x, b.y, hi.x, hi.y);
    GDN_MMA(acc, b.z, b.w, hi.z, hi.w);
  }
  if (warp == 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) red[i][lane] = acc[i];
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int rr = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int cc =
          quad * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      if (rr < gm) {
        const half dv = __float2half_rn(__fadd_rn(acc[i], red[i][lane]));
        out[static_cast<size_t>(row_base + rr) * 2560 + col_base + cc] =
            __hmul(dv, gate_s[rr]);
      }
    }
  }
}

#undef GDN_MMA

void dense_batch(torch::Tensor output, torch::Tensor x, torch::Tensor weight) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(0) >= 2 && x.size(0) <= 16,
              "Qwen3.8 dense batch requires CUDA M2..16");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  for (const auto& t : {output, x, weight}) {
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous() &&
                    t.scalar_type() == at::kHalf &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "Qwen3.8 dense batch requires aligned contiguous FP16");
  }
  TORCH_CHECK(weight.dim() == 2 && output.dim() == 2 &&
                  weight.size(1) == x.size(1) && output.size(0) == x.size(0) &&
                  output.size(1) == weight.size(0),
              "Qwen3.8 dense batch shape mismatch");
  const int m = x.size(0), n = weight.size(0), k = x.size(1);
  const bool router = n == 512 && k == 2560;
  const bool attention_output = n == 2560 && k == 1536;
  TORCH_CHECK(router || attention_output,
              "Qwen3.8 dense batch supports only router/output projections");
  const auto* input = reinterpret_cast<const half*>(x.data_ptr());
  const auto* w = reinterpret_cast<const half*>(weight.data_ptr());
  auto* out = reinterpret_cast<half*>(output.data_ptr());
  const dim3 grid(n / 8, (m + 7) / 8);
  if (router)
    dense_batch_kernel<2560><<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        input, w, out, m, n);
  else
    dense_batch_kernel<1536><<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        input, w, out, m, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
  const bool packed = qw.sizes() == at::IntArrayRef({128, 160, 2, 32, 8}) &&
                      bw.sizes() == at::IntArrayRef({1, 160, 2, 32, 8});
  const bool row_major = qw.sizes() == at::IntArrayRef({4096, 2560}) &&
                         bw.sizes() == at::IntArrayRef({24, 2560});
  TORCH_CHECK(packed || row_major, "Invalid Qwen3.8 GDN weight geometry");
  const int m = x.size(0);
  TORCH_CHECK(qkv.sizes() == at::IntArrayRef({m, 2560}) &&
                  z.sizes() == at::IntArrayRef({m, 1536}) &&
                  b.sizes() == at::IntArrayRef({m, 12}) &&
                  a.sizes() == at::IntArrayRef({m, 12}),
              "Invalid Qwen3.8 GDN output geometry");
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto packed_tag, auto groups_tag) {
    constexpr bool P = decltype(packed_tag)::value;
    constexpr int G = decltype(groups_tag)::value;
    gdn_input_batch_kernel<P, G><<<dim3(129, 1), 128, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(qw.data_ptr()),
        reinterpret_cast<const half*>(bw.data_ptr()),
        reinterpret_cast<half*>(qkv.data_ptr()),
        reinterpret_cast<half*>(z.data_ptr()),
        reinterpret_cast<half*>(b.data_ptr()),
        reinterpret_cast<half*>(a.data_ptr()), m);
  };
  if (packed) {
    if (m <= 8)
      launch(std::true_type{}, std::integral_constant<int, 1>{});
    else
      launch(std::true_type{}, std::integral_constant<int, 2>{});
  } else {
    if (m <= 8)
      launch(std::false_type{}, std::integral_constant<int, 1>{});
    else
      launch(std::false_type{}, std::integral_constant<int, 2>{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void router_batch(torch::Tensor output, torch::Tensor x, torch::Tensor packed) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(0) >= 2 &&
                  x.size(0) <= 40 && x.size(1) == 2560,
              "SM70 batch router requires M2..40, K2560");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  for (const auto& t : {output, x, packed})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous() &&
                    t.scalar_type() == at::kHalf &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "Batch router requires aligned contiguous FP16 storage");
  const bool row_major = packed.sizes() == at::IntArrayRef({512, 2560});
  TORCH_CHECK(
      (row_major || packed.sizes() == at::IntArrayRef({64, 40, 2, 4, 8, 8})) &&
          output.sizes() == at::IntArrayRef({x.size(0), 512}),
      "Invalid batch router geometry");
  const int m = x.size(0);
  const dim3 grid(64, (m + 7) / 8);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (row_major) {
    router_split_quad_kernel<true><<<grid, 32, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(packed.data_ptr()),
        reinterpret_cast<half*>(output.data_ptr()), m);
  } else {
    router_split_quad_kernel<false><<<grid, 32, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(packed.data_ptr()),
        reinterpret_cast<half*>(output.data_ptr()), m);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shared_up_batch(torch::Tensor output, torch::Tensor partial,
                     torch::Tensor x, torch::Tensor packed) {
  const bool fp32_partials = partial.scalar_type() == at::kFloat;
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(1) == 2560 &&
                  (fp32_partials ? (x.size(0) >= 2 && x.size(0) <= 40)
                                 : (x.size(0) % 5 == 0 && x.size(0) >= 5 &&
                                    x.size(0) <= 40)),
              "Shared expert batch projection requires K2560 and valid M");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  for (const auto& t : {output, x, packed})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous() &&
                    t.scalar_type() == at::kHalf &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "Shared expert batch requires aligned contiguous FP16 storage");
  TORCH_CHECK(partial.is_cuda() && partial.device() == x.device() &&
                  partial.is_contiguous() &&
                  (partial.scalar_type() == at::kHalf ||
                   partial.scalar_type() == at::kFloat) &&
                  (partial.numel() == 0 ||
                   reinterpret_cast<uintptr_t>(partial.data_ptr()) % 16 == 0),
              "Shared expert partials require aligned FP16 or FP32 storage");
  const int m = x.size(0);
  TORCH_CHECK(packed.sizes() == at::IntArrayRef({10, 160, 2, 32, 8}) &&
                  (partial.numel() == 0 ||
                   partial.sizes() == at::IntArrayRef({8, m, 320})) &&
                  output.sizes() == at::IntArrayRef({m, 160}),
              "Invalid shared expert batch geometry");
  const dim3 grid(5, (m + 7) / 8);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* input = reinterpret_cast<const half*>(x.data_ptr());
  const auto* weight = reinterpret_cast<const half*>(packed.data_ptr());
  auto* out = reinterpret_cast<half*>(output.data_ptr());
  if (partial.scalar_type() == at::kFloat) {
    shared_up_fused_silu_kernel<float>
        <<<grid, 256, 0, stream>>>(input, weight, out, m);
  } else {
    shared_up_fused_silu_kernel<half>
        <<<grid, 256, 0, stream>>>(input, weight, out, m);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shared_up_batch_fp32(torch::Tensor output, torch::Tensor partial,
                          torch::Tensor x, torch::Tensor packed) {
  TORCH_CHECK(partial.scalar_type() == at::kFloat, "FP32 partials required");
  shared_up_batch(output, partial, x, packed);
}

void shared_gate_mul(torch::Tensor output, torch::Tensor logits,
                     torch::Tensor source) {
  TORCH_CHECK(source.is_cuda() && source.dim() == 2 && source.size(0) >= 2 &&
                  source.size(0) <= 40 && source.size(1) == 2560,
              "Shared gate epilogue requires M2..40, N2560");
  const c10::cuda::CUDAGuard guard(source.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  for (const auto& t : {output, logits, source})
    TORCH_CHECK(t.is_cuda() && t.device() == source.device() &&
                    t.is_contiguous() && t.scalar_type() == at::kHalf,
                "Shared gate requires contiguous FP16 storage");
  const int m = source.size(0);
  TORCH_CHECK(output.sizes() == source.sizes() &&
                  logits.sizes() == at::IntArrayRef({m, 1}),
              "Invalid shared gate geometry");
  shared_gate_mul_kernel<<<(m * 2560 + 127) / 128, 128, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(logits.data_ptr()),
      reinterpret_cast<const half*>(source.data_ptr()),
      reinterpret_cast<half*>(output.data_ptr()), m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shared_down_gated(torch::Tensor out, torch::Tensor hidden_x,
                       torch::Tensor act_out, torch::Tensor down_w,
                       torch::Tensor gate_w) {
  TORCH_CHECK(hidden_x.is_cuda() && hidden_x.dim() == 2 &&
                  hidden_x.size(0) >= 1 && hidden_x.size(0) <= 64 &&
                  hidden_x.size(1) == 2560,
              "Shared down+gate requires CUDA [1..64, 2560] hidden_x");
  const c10::cuda::CUDAGuard guard(hidden_x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  for (const auto& t : {out, hidden_x, act_out, down_w, gate_w}) {
    TORCH_CHECK(t.is_cuda() && t.device() == hidden_x.device() &&
                    t.is_contiguous() && t.scalar_type() == at::kHalf &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "Shared down+gate requires 16B-aligned contiguous FP16");
  }
  const int m = hidden_x.size(0);
  const int k_down = act_out.size(1);
  TORCH_CHECK(act_out.dim() == 2 && act_out.size(0) == m &&
                  (k_down == 160 || k_down == 320 || k_down == 640) &&
                  down_w.sizes() == at::IntArrayRef({2560, k_down}) &&
                  gate_w.sizes() == at::IntArrayRef({1, 2560}) &&
                  out.sizes() == at::IntArrayRef({m, 2560}),
              "Invalid shared down+gate tensor shapes");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* hx = reinterpret_cast<const half*>(hidden_x.data_ptr());
  const auto* ao = reinterpret_cast<const half*>(act_out.data_ptr());
  const auto* dw = reinterpret_cast<const half*>(down_w.data_ptr());
  const auto* gw = reinterpret_cast<const half*>(gate_w.data_ptr());
  auto* op = reinterpret_cast<half*>(out.data_ptr());
  auto launch = [&](auto k_tag) {
    constexpr int KD = decltype(k_tag)::value;
    if (m == 1) {
      shared_down_gated_m1_kernel<KD>
          <<<80, 128, 0, stream>>>(hx, ao, dw, gw, op);
    } else {
      shared_down_gated_batch_kernel<KD>
          <<<dim3(80, (m + 7) / 8), 64, 0, stream>>>(hx, ao, dw, gw, op, m);
    }
  };
  if (k_down == 160)
    launch(std::integral_constant<int, 160>{});
  else if (k_down == 320)
    launch(std::integral_constant<int, 320>{});
  else
    launch(std::integral_constant<int, 640>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "qwen38_shared_up_batch_sm70_out(Tensor(a!) out, Tensor(b!) partial, "
      "Tensor x, Tensor packed) -> ()");
  m.def(
      "qwen38_shared_up_batch_fp32_sm70_out(Tensor(a!) out, Tensor(b!) "
      "partial, "
      "Tensor x, Tensor packed) -> ()");
  m.def(
      "qwen38_shared_gate_mul_sm70_out(Tensor(a!) out, Tensor logits, "
      "Tensor source) -> ()");
  m.def(
      "qwen38_shared_down_gated_sm70_out(Tensor(a!) out, Tensor hidden_x, "
      "Tensor act_out, Tensor down_w, Tensor gate_w) -> ()");
  m.def(
      "qwen38_router_batch_sm70_out(Tensor(a!) out, Tensor x, Tensor packed) "
      "-> ()");
  m.def(
      "qwen38_dense_batch_sm70_out(Tensor(a!) out, Tensor x, Tensor weight) -> "
      "()");
  m.def(
      "qwen38_gdn_input_batch_sm70_out(Tensor(a!) qkv, Tensor(b!) z, "
      "Tensor(c!) b, Tensor(d!) a, Tensor x, Tensor qw, Tensor bw) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen38_shared_up_batch_sm70_out", &shared_up_batch);
  m.impl("qwen38_shared_up_batch_fp32_sm70_out", &shared_up_batch_fp32);
  m.impl("qwen38_shared_gate_mul_sm70_out", &shared_gate_mul);
  m.impl("qwen38_shared_down_gated_sm70_out", &shared_down_gated);
  m.impl("qwen38_router_batch_sm70_out", &router_batch);
  m.impl("qwen38_dense_batch_sm70_out", &dense_batch);
  m.impl("qwen38_gdn_input_batch_sm70_out", &gdn_input_batch);
}
