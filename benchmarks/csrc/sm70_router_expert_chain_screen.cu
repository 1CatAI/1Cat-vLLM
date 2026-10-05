// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research only. No runtime dispatch selects this operator.
// The m8n8k4 layout follows the existing SM70 skinny kernels; see
// csrc/sm70_turbomind/ops/LICENSE.v100-skinny for the retained MIT notice.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>

namespace {
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ unsigned acquire_gpu(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void release_gpu(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v)
               : "memory");
}

__device__ __forceinline__ void wait_gpu(const unsigned* p, unsigned expected) {
  unsigned value;
  do {
    asm volatile("ld.volatile.global.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(p)
                 : "memory");
    if (value != expected) __nanosleep(128);
  } while (value != expected);
  (void)acquire_gpu(p);
}

struct Nvfp4Reader {
  const unsigned* w13;
  const unsigned* w2;
  const half* s13;
  const half* s2;
  int hidden, intermediate;
  __device__ static void decode(unsigned packed, half scalar, half2* out) {
    constexpr unsigned sign = 0x80008000u, em = 0x0e000e00u;
    unsigned v[4] = {((packed << 12) & sign) | ((packed << 9) & em),
                     ((packed << 8) & sign) | ((packed << 5) & em),
                     ((packed << 4) & sign) | ((packed << 1) & em),
                     (packed & sign) | ((packed >> 3) & em)};
    half rebased = __hmul(scalar, __float2half_rn(16384.f));
#pragma unroll
    for (int i = 0; i < 4; ++i)
      out[i] = __hmul2(*(half2*)(v + i), __half2half2(rebased));
  }
};

#define MMA(C, A0, A1, B0, B1)                                      \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};"                                  \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

__device__ void gate_read(Nvfp4Reader r, int e, int slice, int g, int col,
                          half2* b) {
  size_t base = size_t(e) * r.hidden * r.intermediate / 4 +
                slice * r.hidden * 4 + g * 64 + col;
  half scale = r.s13[size_t(e) * r.hidden * r.intermediate / 8 +
                     g * r.intermediate * 2 + slice * 32 + col];
  r.decode(r.w13[base], scale, b);
  r.decode(r.w13[base + 32], scale, b + 4);
}
constexpr int kSlices = 10, kRoutes = 10, kHidden = 2560;
constexpr int kProducers = kSlices * kRoutes, kReducers = 20;
constexpr int kBlocks = kProducers + kReducers + 1;
struct Choice {
  float value;
  int index;
};
struct Maximum {
  __device__ Choice operator()(Choice a, Choice b) const {
    return a.value > b.value || (a.value == b.value && a.index < b.index) ? a
                                                                          : b;
  }
};
using Selection = cub::BlockReduce<Choice, kThreads>;

__global__ __launch_bounds__(kThreads, 2) void router_expert_chain(
    Nvfp4Reader reader, const half* input, const half* router, int* ids,
    float* weights, float* partial, half* output, unsigned* flags,
    unsigned* epochs, int m) {
  __shared__ float local[512];
  __shared__ half activation[16];
  __shared__ typename Selection::TempStorage selection;
  __shared__ int selected[10];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const bool owns_row0 = (lane & 17) == 0;
  const unsigned generation = epochs[blockIdx.x] + 1;
  for (int row = 0; row < m; ++row) {
    const half* x = input + row * kHidden;
    unsigned* ready = flags + row * (kProducers + 1);
    if (blockIdx.x == kProducers) {
      // A single multiwarp CTA produces all logits and exact top-10.
      // Warp MMA columns are independent experts, with FP32 accumulation.
      for (int bank = 0; bank < 2; ++bank) {
        float accum[8] = {};
        for (int g = 0; g < kHidden / 16; ++g) {
          half2 b[8];
          const half* w =
              router + (bank * 256 + warp * 32 + col) * kHidden + g * 16;
#pragma unroll
          for (int i = 0; i < 8; ++i)
            b[i] = __halves2half2(w[2 * i], w[2 * i + 1]);
          const unsigned* bp = (const unsigned*)b;
          uint4 a = make_uint4(0, 0, 0, 0), a1 = a;
          if ((lane & 19) == 0) {
            a = *(const uint4*)(x + g * 16);
            a1 = *(const uint4*)(x + g * 16 + 8);
          }
          MMA(accum, a.x, a.y, bp[0], bp[1]);
          MMA(accum, a.z, a.w, bp[2], bp[3]);
          MMA(accum, a1.x, a1.y, bp[4], bp[5]);
          MMA(accum, a1.z, a1.w, bp[6], bp[7]);
        }
        if (owns_row0) {
#pragma unroll
          for (int pair = 0; pair < 2; ++pair)
            for (int off = 0; off < 2; ++off) {
              const int n = bank * 256 + warp * 32 + ((lane >> 2) & 3) * 8 +
                            off + (((lane >> 1) & 1) << 1) + pair * 4;
              local[n] = __half2float(__float2half_rn(accum[pair * 4 + off]));
            }
        }
      }
      __syncthreads();
      for (int slot = 0; slot < kRoutes; ++slot) {
        Choice best{-INFINITY, 512};
        for (int expert = threadIdx.x; expert < 512; expert += kThreads) {
          bool included = false;
          for (int previous = 0; previous < slot; ++previous)
            included |= selected[previous] == expert;
          if (!included) best = Maximum{}(best, Choice{local[expert], expert});
        }
        Choice result = Selection(selection).Reduce(best, Maximum{});
        if (threadIdx.x == 0) selected[slot] = result.index;
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        float denominator = 0;
        for (int slot = 0; slot < kRoutes; ++slot) {
          ids[row * kRoutes + slot] = selected[slot];
          weights[row * kRoutes + slot] =
              expf(local[selected[slot]] - local[selected[0]]);
          denominator += weights[row * kRoutes + slot];
        }
        for (int slot = 0; slot < kRoutes; ++slot)
          weights[row * kRoutes + slot] /= denominator;
        release_gpu(ready, generation);
      }
      __syncthreads();
    } else if (blockIdx.x < kProducers) {
      wait_gpu(ready, generation);
      const int slot = blockIdx.x / kSlices, slice = blockIdx.x % kSlices;
      const int expert = ids[row * kRoutes + slot];
      float accum[8] = {};
      for (int g = warp; g < kHidden / 16; g += kWarps) {
        half2 decoded[8];
        gate_read(reader, expert, slice, g, col, decoded);
        const unsigned* b = (const unsigned*)decoded;
        uint4 a = make_uint4(0, 0, 0, 0), a1 = a;
        if ((lane & 19) == 0) {
          a = *(const uint4*)(x + g * 16);
          a1 = *(const uint4*)(x + g * 16 + 8);
        }
        MMA(accum, a.x, a.y, b[0], b[1]);
        MMA(accum, a.z, a.w, b[2], b[3]);
        MMA(accum, a1.x, a1.y, b[4], b[5]);
        MMA(accum, a1.z, a1.w, b[6], b[7]);
      }
      if (owns_row0) {
#pragma unroll
        for (int pair = 0; pair < 2; ++pair)
          for (int off = 0; off < 2; ++off)
            local[warp * 32 + ((lane >> 2) & 3) * 8 + off +
                  (((lane >> 1) & 1) << 1) + pair * 4] = accum[pair * 4 + off];
      }
      __syncthreads();
      if (threadIdx.x < 16) {
        float a = 0, b = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
          a += local[w * 32 + threadIdx.x * 2];
          b += local[w * 32 + threadIdx.x * 2 + 1];
        }
        a = __half2float(__float2half_rn(a));
        activation[threadIdx.x] =
            __hmul(__float2half_rn(a / (1.f + expf(-a))), __float2half_rn(b));
      }
      __syncthreads();
      // Each producer consumes its own intermediate tile immediately,
      // producing FP32 W2 partials for deterministic downstream sums.
      for (int tile = warp; tile < kHidden / 32; tile += kWarps) {
        float down[8] = {};
        const size_t code_base = size_t(expert) * kHidden * 160 / 8 +
                                 tile * 160 * 4 + slice * 64 + col;
        const half scale =
            reader.s2[size_t(expert) * kHidden * 160 / 16 +
                      (slice * (kHidden / 32) + tile) * 32 + col];
        half2 decoded[8];
        reader.decode(reader.w2[code_base], scale, decoded);
        reader.decode(reader.w2[code_base + 32], scale, decoded + 4);
        const unsigned* b = (const unsigned*)decoded;
        uint4 a = make_uint4(0, 0, 0, 0), a1 = a;
        if ((lane & 19) == 0) {
          a = *(const uint4*)activation;
          a1 = *(const uint4*)(activation + 8);
        }
        MMA(down, a.x, a.y, b[0], b[1]);
        MMA(down, a.z, a.w, b[2], b[3]);
        MMA(down, a1.x, a1.y, b[4], b[5]);
        MMA(down, a1.z, a1.w, b[6], b[7]);
        if (owns_row0) {
#pragma unroll
          for (int pair = 0; pair < 2; ++pair)
            for (int off = 0; off < 2; ++off) {
              const int n = tile * 32 + ((lane >> 2) & 3) * 8 + off +
                            (((lane >> 1) & 1) << 1) + pair * 4;
              partial[(size_t(row) * kProducers + blockIdx.x) * kHidden + n] =
                  down[pair * 4 + off];
            }
        }
      }
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) release_gpu(ready + 1 + blockIdx.x, generation);
      __syncthreads();
    } else {
      if (threadIdx.x < kProducers)
        wait_gpu(ready + 1 + threadIdx.x, generation);
      __syncthreads();
      const int n = (blockIdx.x - kProducers - 1) * 128 + threadIdx.x;
      if (threadIdx.x < 128) {
        float result = 0;
        for (int slot = 0; slot < kRoutes; ++slot) {
          float sum = 0;
#pragma unroll
          for (int slice = 0; slice < kSlices; ++slice)
            sum += partial[(size_t(row) * kProducers + slot * kSlices + slice) *
                               kHidden +
                           n];
          result += __half2float(__float2half_rn(sum)) *
                    weights[row * kRoutes + slot];
        }
        output[row * kHidden + n] = __float2half_rn(result);
      }
      __syncthreads();
    }
  }
  if (threadIdx.x == 0) epochs[blockIdx.x] = generation;
}
}  // namespace

void run_chain(torch::Tensor x, torch::Tensor router, torch::Tensor ids,
               torch::Tensor weights, torch::Tensor w13, torch::Tensor s13,
               torch::Tensor w2, torch::Tensor s2, torch::Tensor partial,
               torch::Tensor out, torch::Tensor flags, torch::Tensor epochs) {
  const int m = x.size(0);
  TORCH_CHECK((m == 1 || m == 5) && x.size(1) == kHidden);
  TORCH_CHECK(router.sizes() == at::IntArrayRef({512, kHidden}));
  TORCH_CHECK(partial.numel() >= size_t(m) * kProducers * kHidden);
  TORCH_CHECK(flags.numel() >= m * (kProducers + 1) &&
              epochs.numel() >= kBlocks);
  c10::cuda::CUDAGuard guard(x.device());
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, router_expert_chain, kThreads, 0));
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->cooperativeLaunch &&
              active * properties->multiProcessorCount >= kBlocks);
  Nvfp4Reader reader{(unsigned*)w13.data_ptr(),
                     (unsigned*)w2.data_ptr(),
                     (half*)s13.data_ptr(),
                     (half*)s2.data_ptr(),
                     kHidden,
                     160};
  auto* ip = (half*)x.data_ptr();
  auto* rp = (half*)router.data_ptr();
  auto* idp = ids.data_ptr<int>();
  auto* wp = weights.data_ptr<float>();
  auto* pp = partial.data_ptr<float>();
  auto* op = (half*)out.data_ptr();
  auto* fp = (unsigned*)flags.data_ptr();
  auto* ep = (unsigned*)epochs.data_ptr();
  void* arguments[] = {&reader, &ip, &rp, &idp, &wp,
                       &pp,     &op, &fp, &ep,  (void*)&m};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      (const void*)router_expert_chain, dim3(kBlocks), dim3(kThreads),
      arguments, 0, at::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run_chain); }
