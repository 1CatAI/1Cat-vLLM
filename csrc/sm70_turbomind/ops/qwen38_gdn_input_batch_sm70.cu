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

template <bool Packed, bool PairRows = false>
__global__ __launch_bounds__(128, 4) void gdn_input_batch_kernel(
    const half* x, const half* qw, const half* bw, half* qkv, half* z,
    half* b_out, half* a_out, int m) {
  constexpr int groups = PairRows ? 2 : 1;
  __shared__ float partial[4][groups * 8][32];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const bool is_ba = blockIdx.x == 128;
  const int row_group = blockIdx.y;
  // The small b/a projection retains its four contiguous 640-element K
  // partitions. QKVZ retains the original single ordered K reduction.
  if (!is_ba && warp != 0) return;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int col = quad * 8 + r;
  const half* w = is_ba ? bw : qw + static_cast<size_t>(blockIdx.x) * 2560 * 32;
  const int start = is_ba ? warp * 40 : 0;
  const int end = is_ba ? (warp + 1) * 40 : 160;
  float acc[groups][8] = {};
#pragma unroll 32
  for (int g = start; g < end; ++g) {
    uint4 lo = make_uint4(0, 0, 0, 0), hi = lo;
    if constexpr (Packed) {
      lo = *reinterpret_cast<const uint4*>(w + (g * 64 + col) * 8);
      hi = *reinterpret_cast<const uint4*>(w + (g * 64 + 32 + col) * 8);
    } else if (!is_ba || col < 24) {
      // Same fragments and K order, but read the original checkpoint layout.
      // This variant needs no second model-sized weight allocation.
      lo = *reinterpret_cast<const uint4*>(w + col * 2560 + g * 16);
      hi = *reinterpret_cast<const uint4*>(w + col * 2560 + g * 16 + 8);
    }
#pragma unroll
    for (int p = 0; p < groups; ++p) {
      const int row = (row_group + p) * 8 + r;
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
  for (int p = 0; p < groups; ++p) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int rr = p * 8 + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
      const int cc =
          quad * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      if (is_ba) {
        partial[warp][rr][cc] = acc[p][i];
      } else if (row_group * 8 + rr < m) {
        const int outcol = blockIdx.x * 32 + cc;
        const half value = __float2half_rn(acc[p][i]);
        if (outcol < 2560)
          qkv[(row_group * 8 + rr) * 2560 + outcol] = value;
        else
          z[(row_group * 8 + rr) * 1536 + outcol - 2560] = value;
      }
    }
  }
  if (is_ba) {
    __syncthreads();
    for (int i = threadIdx.x; i < groups * 8 * 24; i += 128) {
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
template <int Unroll>
__global__ __launch_bounds__(32, 8) void router_split_quad_kernel(const half* x,
                                                                  const half* w,
                                                                  half* out,
                                                                  int m) {
  const int lane = threadIdx.x, split = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0), row = blockIdx.y * 8 + r;
  float acc[8] = {};
#pragma unroll Unroll
  for (int g = 0; g < 40; ++g) {
    const half* weights =
        w + (blockIdx.x * 40 * 64 + g * 64 + split * 8 + r) * 8;
    const uint4 lo = *reinterpret_cast<const uint4*>(weights);
    const uint4 hi = *reinterpret_cast<const uint4*>(weights + 32 * 8);
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
    const int rr =
        blockIdx.y * 8 + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
    const int cc =
        blockIdx.x * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
    float value = __shfl_sync(0xffffffff, acc[i], lane & ~12);
#pragma unroll
    for (int s = 1; s < 4; ++s)
      value = __fadd_rn(
          value, __shfl_sync(0xffffffff, acc[i], (lane & ~12) | (s << 2)));
    if (split == 0 && rr < m) out[rr * 512 + cc] = __float2half_rn(value);
  }
}

// The shared expert's original cuBLASLt projection uses eight K320
// partitions. Keep the legacy FP16 partial schedule and support FP32
// partials when reduced-precision reduction is disabled.
template <typename Partial>
__global__ __launch_bounds__(32, 8) void shared_up_batch_kernel(
    const half* x, const half* w, Partial* partial, int m) {
  const int lane = threadIdx.x, quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row = blockIdx.y * 8 + r, col = quad * 8 + r;
  const int split = blockIdx.z;
  float acc[8] = {};
#pragma unroll 20
  for (int g = split * 20; g < (split + 1) * 20; ++g) {
    const int offset = (blockIdx.x * 160 + g) * 512 + col * 8;
    const uint4 lo = *reinterpret_cast<const uint4*>(w + offset);
    const uint4 hi = *reinterpret_cast<const uint4*>(w + offset + 256);
    uint4 a = {}, b = {};
    if (row < m) {
      a = *reinterpret_cast<const uint4*>(x + row * 2560 + g * 16);
      b = *reinterpret_cast<const uint4*>(x + row * 2560 + g * 16 + 8);
    }
    GDN_MMA(acc, a.x, a.y, lo.x, lo.y);
    GDN_MMA(acc, a.z, a.w, lo.z, lo.w);
    GDN_MMA(acc, b.x, b.y, hi.x, hi.y);
    GDN_MMA(acc, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr =
        blockIdx.y * 8 + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
    const int cc = blockIdx.x * 32 + quad * 8 +
                   ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
    if (rr < m) {
      if constexpr (std::is_same_v<Partial, half>)
        partial[(split * m + rr) * 320 + cc] = __float2half_rn(acc[i]);
      else
        partial[(split * m + rr) * 320 + cc] = acc[i];
    }
  }
}

__device__ __forceinline__ float partial_float(half value) {
  return __half2float(value);
}
__device__ __forceinline__ float partial_float(float value) { return value; }

template <typename Partial>
__global__ void shared_up_reduce_silu_kernel(const Partial* partial, half* out,
                                             int m) {
  const int k = blockIdx.x * 128 + threadIdx.x;
  if (k >= m * 160) return;
  const int r = k / 160, c = k % 160, index = r * 320 + c;
  float gate = partial_float(partial[index]);
  float up = partial_float(partial[index + 160]);
#pragma unroll
  for (int s = 1; s < 8; ++s) {
    gate = __fadd_rn(gate, partial_float(partial[s * m * 320 + index]));
    up = __fadd_rn(up, partial_float(partial[s * m * 320 + index + 160]));
  }
  const half g = __float2half_rn(gate), u = __float2half_rn(up);
  const float value = __half2float(g);
  // Match the native packed SiLU: full expf, FP16 activation, then FP16 mul.
  const half activated = __float2half_rn(value / __fadd_rn(1.f, expf(-value)));
  out[k] = __hmul(activated, u);
}

__global__ void shared_gate_mul_kernel(const half* logits, const half* source,
                                       half* out, int m) {
  const int index = blockIdx.x * 128 + threadIdx.x;
  if (index >= m * 2560) return;
  const float value = __half2float(logits[index / 2560]);
  const half factor = __float2half_rn(1.f / __fadd_rn(1.f, expf(-value)));
  out[index] = __hmul(source[index], factor);
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
  const bool paired = row_major && m > 8;
  const auto kernel = packed ? gdn_input_batch_kernel<true>
                             : (paired ? gdn_input_batch_kernel<false, true>
                                       : gdn_input_batch_kernel<false>);
  kernel<<<dim3(129, paired ? 1 : (m + 7) / 8), 128, 0,
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
void router_batch(torch::Tensor output, torch::Tensor x, torch::Tensor packed) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(0) >= 2 &&
                  x.size(0) <= 16 && x.size(1) == 2560,
              "SM70 batch router requires M2..16, K2560");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  for (const auto& t : {output, x, packed})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous() &&
                    t.scalar_type() == at::kHalf &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "Batch router requires aligned contiguous FP16 storage");
  TORCH_CHECK(packed.sizes() == at::IntArrayRef({64, 40, 2, 4, 8, 8}) &&
                  output.sizes() == at::IntArrayRef({x.size(0), 512}),
              "Invalid batch router geometry");
  router_split_quad_kernel<40><<<dim3(64, (x.size(0) + 7) / 8), 32, 0,
                                 at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(packed.data_ptr()),
      reinterpret_cast<half*>(output.data_ptr()), x.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shared_up_batch(torch::Tensor output, torch::Tensor partial,
                     torch::Tensor x, torch::Tensor packed) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 &&
                  (x.size(0) == 5 || x.size(0) == 10) && x.size(1) == 2560,
              "Shared expert batch projection requires M5/M10, K2560");
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
                  reinterpret_cast<uintptr_t>(partial.data_ptr()) % 16 == 0,
              "Shared expert partials require aligned FP16 or FP32 storage");
  const int m = x.size(0);
  TORCH_CHECK(packed.sizes() == at::IntArrayRef({10, 160, 2, 32, 8}) &&
                  partial.sizes() == at::IntArrayRef({8, m, 320}) &&
                  output.sizes() == at::IntArrayRef({m, 160}),
              "Invalid shared expert batch geometry");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* input = reinterpret_cast<const half*>(x.data_ptr());
  const auto* weight = reinterpret_cast<const half*>(packed.data_ptr());
  auto* out = reinterpret_cast<half*>(output.data_ptr());
  if (partial.scalar_type() == at::kFloat) {
    auto* p = reinterpret_cast<float*>(partial.data_ptr());
    shared_up_batch_kernel<float>
        <<<dim3(10, (m + 7) / 8, 8), 32, 0, stream>>>(input, weight, p, m);
    shared_up_reduce_silu_kernel<float>
        <<<(m * 160 + 127) / 128, 128, 0, stream>>>(p, out, m);
  } else {
    auto* p = reinterpret_cast<half*>(partial.data_ptr());
    shared_up_batch_kernel<half>
        <<<dim3(10, (m + 7) / 8, 8), 32, 0, stream>>>(input, weight, p, m);
    shared_up_reduce_silu_kernel<half>
        <<<(m * 160 + 127) / 128, 128, 0, stream>>>(p, out, m);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shared_gate_mul(torch::Tensor output, torch::Tensor logits,
                     torch::Tensor source) {
  TORCH_CHECK(source.is_cuda() && source.dim() == 2 &&
                  (source.size(0) == 5 || source.size(0) == 10) &&
                  source.size(1) == 2560,
              "Shared gate epilogue requires M5/M10, N2560");
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

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "qwen38_shared_up_batch_sm70_out(Tensor(a!) out, Tensor(b!) partial, "
      "Tensor x, Tensor packed) -> ()");
  m.def(
      "qwen38_shared_gate_mul_sm70_out(Tensor(a!) out, Tensor logits, "
      "Tensor source) -> ()");
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
  m.impl("qwen38_shared_gate_mul_sm70_out", &shared_gate_mul);
  m.impl("qwen38_router_batch_sm70_out", &router_batch);
  m.impl("qwen38_dense_batch_sm70_out", &dense_batch);
  m.impl("qwen38_gdn_input_batch_sm70_out", &gdn_input_batch);
}
