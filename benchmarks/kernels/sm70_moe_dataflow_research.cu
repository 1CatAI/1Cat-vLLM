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
#include <type_traits>

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

// Decoder interfaces share one FP16-input / FP32-accumulation schedule.
struct Fp16Reader {
  const half* w13;
  const half* w2;
  int hidden, intermediate;
  __device__ void gate(int expert, int slice, int g, int col, half2* b) const {
    const half* row =
        w13 + (size_t(expert) * intermediate * 2 + slice * 32 + col) * hidden +
        g * 16;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      b[i] = __halves2half2(row[2 * i], row[2 * i + 1]);
  }
};
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

int down_bytes(const Fp16Reader& r, int topk) {
  return topk * 32 * std::min(r.intermediate, 48) * sizeof(half);
}
int down_bytes(const Nvfp4Reader& r, int topk) {
  return topk * r.intermediate * 20;
}
__device__ void gate_read(Fp16Reader r, int e, int slice, int g, int col,
                          half2* b) {
  r.gate(e, slice, g, col, b);
}
__device__ void gate_read(Nvfp4Reader r, int e, int slice, int g, int col,
                          half2* b) {
  size_t base = size_t(e) * r.hidden * r.intermediate / 4 +
                slice * r.hidden * 4 + g * 64 + col;
  half scale = r.s13[size_t(e) * r.hidden * r.intermediate / 8 +
                     g * r.intermediate * 2 + slice * 32 + col];
  r.decode(r.w13[base], scale, b);
  r.decode(r.w13[base + 32], scale, b + 4);
}
__device__ void prefetch_down(Fp16Reader r, int token, int tile, const int* ids,
                              int topk, int experts, unsigned char* storage) {
  auto* cache = (half*)storage;
  int cached = min(r.intermediate, 48);
  for (int idx = threadIdx.x; idx < topk * 32 * cached; idx += kThreads) {
    int slot = idx / (32 * cached), n = idx / cached % 32, k = idx % cached,
        e = ids[token * topk + slot];
    cache[idx] =
        e >= 0 && e < experts
            ? r.w2[(size_t(e) * r.hidden + tile * 32 + n) * r.intermediate + k]
            : __float2half_rn(0);
  }
}
__device__ void prefetch_down(Nvfp4Reader r, int token, int tile,
                              const int* ids, int topk, int experts,
                              unsigned char* storage) {
  auto* codes = (unsigned*)storage;
  auto* scales = (half*)(codes + topk * r.intermediate * 4);
  for (int idx = threadIdx.x; idx < topk * r.intermediate * 4;
       idx += kThreads) {
    int slot = idx / (r.intermediate * 4), local = idx % (r.intermediate * 4),
        e = ids[token * topk + slot];
    codes[idx] = e >= 0 && e < experts
                     ? r.w2[size_t(e) * r.hidden * r.intermediate / 8 +
                            tile * r.intermediate * 4 + local]
                     : 0;
  }
  for (int idx = threadIdx.x; idx < topk * r.intermediate * 2;
       idx += kThreads) {
    int slot = idx / (r.intermediate * 2), local = idx % (r.intermediate * 2),
        g = local / 32, col = local % 32, e = ids[token * topk + slot];
    scales[idx] = e >= 0 && e < experts
                      ? r.s2[size_t(e) * r.hidden * r.intermediate / 16 +
                             (g * (r.hidden / 32) + tile) * 32 + col]
                      : __float2half_rn(0);
  }
}
__device__ void down_read(Fp16Reader r, int e, int tile, int slot, int g,
                          int col, const unsigned char* storage, half2* b) {
  int cached = min(r.intermediate, 48);
  const half* row =
      g * 16 < cached
          ? (const half*)storage + (slot * 32 + col) * cached + g * 16
          : r.w2 + (size_t(e) * r.hidden + tile * 32 + col) * r.intermediate +
                g * 16;
#pragma unroll
  for (int i = 0; i < 8; ++i) b[i] = __halves2half2(row[2 * i], row[2 * i + 1]);
}
__device__ void down_read(Nvfp4Reader r, int, int, int slot, int g, int col,
                          const unsigned char* storage, half2* b, int topk) {
  const auto* codes = (const unsigned*)storage;
  const auto* scales = (const half*)(codes + topk * r.intermediate * 4);
  size_t offset = (slot * (r.intermediate / 8) + g * 2) * 32 + col;
  half scale = scales[(slot * (r.intermediate / 16) + g) * 32 + col];
  r.decode(codes[offset], scale, b);
  r.decode(codes[offset + 32], scale, b + 4);
}

template <class Reader, int M>
__global__ __launch_bounds__(kThreads, 3) void segment(
    __grid_constant__ const Reader reader, const half* input, const int* ids,
    const float* weights, float* scratch, half* output, unsigned* control,
    int runtime_m, int topk, int experts, int cached_bytes,
    int producer_workers) {
  extern __shared__ unsigned char storage[];
  __shared__ unsigned generation;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  bool owns_row0 = (lane & 17) == 0;
  int h = reader.hidden, intermediate = reader.intermediate,
      slices = intermediate / 16;
  int m = M ? M : runtime_m, nproducer = m * topk * slices,
      nreduce = m * (h / 32);
  auto* activated = (half*)scratch;
  if (threadIdx.x == 0) {
    generation = control[5 + blockIdx.x] + 1;
    if (blockIdx.x == 0) {
      control[4] = generation;
      release_gpu(control, generation);
    } else
      while (acquire_gpu(control) != generation) {
      }
  }
  __syncthreads();
  if (blockIdx.x < unsigned(producer_workers)) {
    auto* partial = (float*)storage;
    for (int task = blockIdx.x; task < nproducer; task += producer_workers) {
      int route = task / slices, slice = task % slices, expert = ids[route];
      float accum[8] = {};
      if (expert >= 0 && expert < experts) {
        const half* x = input + size_t(route / topk) * h;
        for (int g = warp; g < h / 16; g += kWarps) {
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
      }
      if (owns_row0) {
#pragma unroll
        for (int pair = 0; pair < 2; ++pair)
          for (int off = 0; off < 2; ++off)
            partial[warp * 32 + ((lane >> 2) & 3) * 8 + off +
                    (((lane >> 1) & 1) << 1) + pair * 4] =
                accum[pair * 4 + off];
      }
      __syncthreads();
      if (threadIdx.x < 16) {
        float gate = 0, up = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
          gate += partial[w * 32 + threadIdx.x * 2];
          up += partial[w * 32 + threadIdx.x * 2 + 1];
        }
        float gf = __half2float(__float2half_rn(gate));
        activated[size_t(route) * intermediate + slice * 16 + threadIdx.x] =
            __hmul(__float2half_rn(gf / (1.f + expf(-gf))),
                   __float2half_rn(up));
        __threadfence();
      }
      __syncthreads();
      if (threadIdx.x == 0)
        release_gpu(control + 5 + gridDim.x + task, generation);
      __syncthreads();
    }
  } else {
    int consumer_workers = gridDim.x - producer_workers;
    auto* partial = (half*)(storage + cached_bytes);
    for (int task = blockIdx.x - producer_workers; task < nreduce;
         task += consumer_workers) {
      int token = task / (h / 32), tile = task % (h / 32);
      // All W2 weights are available before waiting for the tiny activation.
      // FP16 uses a bounded prefix; NVFP4 caches the complete output tile.
      prefetch_down(reader, token, tile, ids, topk, experts, storage);
      __syncthreads();
      for (int slot = warp; slot < topk; slot += kWarps) {
        for (int g = lane; g < slices; g += 32)
          wait_gpu(control + 5 + gridDim.x + (token * topk + slot) * slices + g,
                   generation);
        __syncwarp();
        float accum[8] = {};
        int expert = ids[token * topk + slot];
        if (expert >= 0 && expert < experts) {
          const half* x =
              activated + size_t(token * topk + slot) * intermediate;
          for (int g = 0; g < slices; ++g) {
            half2 decoded[8];
            if constexpr (std::is_same_v<Reader, Nvfp4Reader>)
              down_read(reader, expert, tile, slot, g, col, storage, decoded,
                        topk);
            else
              down_read(reader, expert, tile, slot, g, col, storage, decoded);
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
        }
        if (owns_row0) {
#pragma unroll
          for (int pair = 0; pair < 2; ++pair)
            for (int off = 0; off < 2; ++off)
              partial[slot * 32 + ((lane >> 2) & 3) * 8 + off +
                      (((lane >> 1) & 1) << 1) + pair * 4] =
                  __float2half_rn(accum[pair * 4 + off]);
        }
      }
      __syncthreads();
      if (threadIdx.x < 32) {
        float result = 0;
        for (int slot = 0; slot < topk; ++slot)
          result += __half2float(partial[slot * 32 + threadIdx.x]) *
                    weights[token * topk + slot];
        output[size_t(token) * h + tile * 32 + threadIdx.x] =
            __float2half_rn(result);
      }
      __syncthreads();
    }
  }
  if (threadIdx.x == 0) control[5 + blockIdx.x] = generation;
}

template <class Reader>
void run(Reader reader, torch::Tensor input, torch::Tensor ids,
         torch::Tensor weights, torch::Tensor partial, torch::Tensor output,
         torch::Tensor control, int experts) {
  const int h = reader.hidden, i = reader.intermediate, m = input.size(0),
            k = ids.size(1);
  TORCH_CHECK(h > 0 && h % 32 == 0 && i > 0 && i % 16 == 0 && m > 0 && k > 0 &&
              experts > 0);
  TORCH_CHECK(
      input.scalar_type() == at::kHalf && output.scalar_type() == at::kHalf &&
      ids.scalar_type() == at::kInt && weights.scalar_type() == at::kFloat &&
      partial.scalar_type() == at::kFloat && control.scalar_type() == at::kInt);
  for (auto t : {input, ids, weights, partial, output, control})
    TORCH_CHECK(t.is_cuda() && t.device() == input.device() &&
                t.is_contiguous());
  TORCH_CHECK(ids.size(0) == m && weights.sizes() == ids.sizes() &&
              output.sizes() == input.sizes() && input.size(1) == h &&
              partial.numel() >= size_t(m) * k * (i / 16) * h &&
              control.numel() >= 5);
  c10::cuda::CUDAGuard guard(input.device());

  int cached = down_bytes(reader, k);
  int bytes = std::max(kWarps * 32 * int(sizeof(float)),
                       cached + k * 32 * int(sizeof(half)));
  cudaDeviceProp prop;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, input.get_device()));
  TORCH_CHECK(prop.cooperativeLaunch &&
              bytes <= int(prop.sharedMemPerBlockOptin));
  auto stream = at::cuda::getCurrentCUDAStream();
  int occupancy = 32;
#define GEOMETRY(M)                                                      \
  do {                                                                   \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                 \
        segment<Reader, M>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
        bytes));                                                         \
    int active = 0;                                                      \
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(        \
        &active, segment<Reader, M>, kThreads, bytes));                  \
    occupancy = std::min(occupancy, active);                             \
  } while (0)
  GEOMETRY(1);
  GEOMETRY(5);
  GEOMETRY(0);
#undef GEOMETRY
  TORCH_CHECK(occupancy > 0);
  int blocks = prop.multiProcessorCount * occupancy;
  TORCH_CHECK(blocks >= 2 &&
              control.numel() >= 5 + blocks + size_t(m) * k * (i / 16));
  int producers = std::min(m * k * (i / 16), std::max(1, blocks * 2 / 3));
#define LAUNCH(M)                                                            \
  do {                                                                       \
    auto* ip = (half*)input.data_ptr();                                      \
    auto* idp = ids.data_ptr<int>();                                         \
    auto* wp = weights.data_ptr<float>();                                    \
    auto* pp = partial.data_ptr<float>();                                    \
    auto* op = (half*)output.data_ptr();                                     \
    auto* cp = (unsigned*)control.data_ptr();                                \
    void* args[] = {&reader,   &ip,      &idp,    &wp,                       \
                    &pp,       &op,      &cp,     (void*)&m,                 \
                    (void*)&k, &experts, &cached, &producers};               \
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(                              \
        (const void*)segment<Reader, M>, dim3(blocks), dim3(kThreads), args, \
        bytes, stream));                                                     \
  } while (0)
  if (m == 1)
    LAUNCH(1);
  else if (m == 5)
    LAUNCH(5);
  else
    LAUNCH(0);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

void fp16(torch::Tensor x, torch::Tensor ids, torch::Tensor weights,
          torch::Tensor w13, torch::Tensor w2, torch::Tensor partial,
          torch::Tensor out, torch::Tensor control) {
  TORCH_CHECK(w13.dim() == 3 && w2.dim() == 3 &&
              w13.scalar_type() == at::kHalf && w2.scalar_type() == at::kHalf);
  int e = w13.size(0), h = x.size(1), i = w2.size(2);
  TORCH_CHECK(w13.size(1) == 2 * i && w13.size(2) == h && w2.size(0) == e &&
              w2.size(1) == h);
  for (auto t : {w13, w2})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  run(Fp16Reader{(half*)w13.data_ptr(), (half*)w2.data_ptr(), h, i}, x, ids,
      weights, partial, out, control, e);
}
void nvfp4(torch::Tensor x, torch::Tensor ids, torch::Tensor weights,
           torch::Tensor w13, torch::Tensor s13, torch::Tensor w2,
           torch::Tensor s2, torch::Tensor partial, torch::Tensor out,
           torch::Tensor control, int64_t intermediate, int64_t experts) {
  int h = x.size(1), i = intermediate, e = experts;
  for (auto t : {w13, s13, w2, s2})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous());
  TORCH_CHECK(w13.scalar_type() == at::kInt && w2.scalar_type() == at::kInt &&
              s13.scalar_type() == at::kHalf && s2.scalar_type() == at::kHalf);
  TORCH_CHECK(w13.numel() == size_t(e) * h * i / 4 &&
              w2.numel() == size_t(e) * h * i / 8 &&
              s13.numel() == size_t(e) * h * i / 8 &&
              s2.numel() == size_t(e) * h * i / 16);
  run(Nvfp4Reader{(unsigned*)w13.data_ptr(), (unsigned*)w2.data_ptr(),
                  (half*)s13.data_ptr(), (half*)s2.data_ptr(), h, i},
      x, ids, weights, partial, out, control, e);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp16", &fp16);
  m.def("nvfp4", &nvfp4);
}
