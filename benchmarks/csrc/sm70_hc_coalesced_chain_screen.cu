// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Fixed normalization producer, prefetched down rows, fixed gather consumer.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda/atomic>
#include <cuda_fp16.h>
#include <vector>

struct Peers {
  uint32_t* p[4];
};
constexpr int PacketStride = 336 + 2560 + 4 * 2560, Blocks = 27;
__device__ float warp_sum(float x) {
  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}
__device__ float sig(float x) { return 1.f / (1.f + expf(-x)); }
__device__ half visible_half(const half* p) {
  uint16_t value;
  asm volatile("ld.volatile.global.u16 %0, [%1];"
               : "=h"(value)
               : "l"(p)
               : "memory");
  return __ushort_as_half(value);
}
__device__ uint32_t packet(uint32_t* p, uint32_t generation) {
  uint32_t value;
  do {
    asm volatile("ld.volatile.global.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(p)
                 : "memory");
    if ((value >> 16) != (generation & 0xffff)) __nanosleep(128);
  } while ((value >> 16) != (generation & 0xffff));
  return value;
}
__device__ void push(Peers peers, int offset, uint32_t value) {
#pragma unroll
  for (int dst = 0; dst < 4; ++dst)
    asm volatile(
        "st.volatile.global.u32 [%0], %1;" ::"l"(peers.p[dst] + offset),
        "r"(value)
        : "memory");
}

__device__ void down_mma(float (&acc)[8], uint32_t a0, uint32_t a1, uint32_t b0,
                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]),
        "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
__device__ uint4 load_current(const half* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
template <bool ReduceCore>
__global__ __launch_bounds__(128, 4) void hc_coalesced(
    const half* residual, const half* core, const half* injection,
    const half* norm, const half* weight, half* combined, half* xn,
    half* gathered, half* lora, half* next_injection, Peers peers,
    uint32_t* epochs, uint32_t* norm_flags, float* partial, int rank, int m,
    float eps) {
  const int block = blockIdx.x, t = threadIdx.x;
  const int warp = t / 32, lane = t % 32;
  const uint32_t generation = epochs[block] + 1;
  __shared__ float sums[4], projected[16 * 5 * 8];
  __shared__ half reduced_core[ReduceCore ? 2560 : 1];
  if constexpr (ReduceCore) {
    if (block < 20) {
      for (int row = 0; row < m; ++row) {
        const int offset = ((generation & 1) * m + row) * PacketStride;
        const int h = block * 128 + t;
        push(peers, offset + 336 + 2560 + rank * 2560 + h,
             ((generation & 0xffff) << 16) |
                 __half_as_ushort(core[row * 2560 + h]));
      }
    }
  }
  if (block >= 22 && block < 26) {
    for (int row = 0; row < m; ++row) {
      const int offset = ((generation & 1) * m + row) * PacketStride;
      if constexpr (ReduceCore) {
        for (int h = t; h < 2560; h += 128) {
          float value = 0;
          for (int src = 0; src < 4; ++src) {
            const uint32_t word =
                packet(peers.p[rank] + offset + 336 + 2560 + src * 2560 + h,
                       generation);
            value += __half2float(__ushort_as_half(word & 0xffff));
          }
          reduced_core[h] = __float2half_rn(value);
        }
        __syncthreads();
      }
      const int branch = block - 22;
      const float scale =
          2.f * sig(__half2float(injection[row * 4 + branch]) / 4.f);
      float values[20], sum = 0;
#pragma unroll
      for (int i = 0; i < 20; ++i) {
        const int h = t + i * 128, k = branch * 2560 + h;
        const half c = ReduceCore ? reduced_core[h] : core[row * 2560 + h];
        const half value = __float2half_rn(fmaf(
            __half2float(c), scale, __half2float(residual[row * 10240 + k])));
        values[i] = __half2float(value);
        combined[row * 10240 + k] = value;
        sum = fmaf(values[i], values[i], sum);
      }
      sum = warp_sum(sum);
      if ((t & 31) == 0) sums[t / 32] = sum;
      __syncthreads();
      if (t == 0) {
        float total = 0;
        for (int warp = 0; warp < 4; ++warp) total += sums[warp];
        sums[0] = rsqrtf(total / 2560.f + eps);
      }
      __syncthreads();
      const float inv = sums[0];
#pragma unroll
      for (int i = 0; i < 20; ++i) {
        const int k = branch * 2560 + t + i * 128;
        const float value = values[i] * inv;
        xn[row * 10240 + k] =
            __float2half_rn(fmaf(value, __half2float(norm[k]), value));
      }
      __threadfence();
      __syncthreads();
      if (t == 0) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(
            norm_flags[row * 4 + branch]);
        flag.store(generation, cuda::memory_order_release);
      }
      __syncthreads();
    }
  } else if (block < 22) {
    const int group = block % 11, part = block / 11;
    // Independent normalization dependencies; no grid barrier.
    if (t < m * 4) {
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(norm_flags[t]);
      while (flag.load(cuda::memory_order_acquire) != generation)
        __nanosleep(128);
    }
    __syncthreads();
    const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int quad = (lane >> 2) & 3;
    float acc[8] = {};
    // Quad pairs own separate K partitions of the same eight output rows.
    for (int g = part * 320 + warp * 80 + quad * 20;
         g < part * 320 + warp * 80 + (quad + 1) * 20; ++g) {
      const half* w = weight + (group * 640 + g) * 128 + r * 8;
      const uint4 lo = *(const uint4*)w, hi = *(const uint4*)(w + 64);
      uint4 a = {}, b = {};
      if (r < m) {
        a = load_current(xn + r * 10240 + g * 16);
        b = load_current(xn + r * 10240 + g * 16 + 8);
      }
      down_mma(acc, a.x, a.y, lo.x, lo.y);
      down_mma(acc, a.z, a.w, lo.z, lo.w);
      down_mma(acc, b.x, b.y, hi.x, hi.y);
      down_mma(acc, b.z, b.w, hi.z, hi.w);
    }
    for (int i = 0; i < 8; ++i) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int n = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      if (row < m) projected[((warp * 4 + quad) * m + row) * 8 + n] = acc[i];
    }
    __syncthreads();
    if (t < m * 8) {
      float value = 0;
      for (int k = 0; k < 16; ++k) value += projected[k * m * 8 + t];
      partial[block * m * 8 + t] = value;
    }
    __threadfence();
    __syncthreads();
    if (t == 0) {
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(
          norm_flags[m * 4 + block]);
      flag.store(generation, cuda::memory_order_release);
    }
    __syncthreads();
    if (part == 0) {
      if (t == 0) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(
            norm_flags[m * 4 + block + 11]);
        while (flag.load(cuda::memory_order_acquire) != generation)
          __nanosleep(128);
      }
      __syncthreads();
      if (t < m * 8) {
        const int row = t / 8, local = group * 8 + t % 8;
        if (local < 81) {
          float value =
              partial[block * m * 8 + t] + partial[(block + 11) * m * 8 + t];
          value = __half2float(__float2half_rn(value));
          if (local < 80) {
            value /= 4.f;
            value *= sig(value);
          }
          const int column = local < 80 ? rank * 80 + local : 320 + rank;
          const int offset = ((generation & 1) * m + row) * PacketStride;
          // Eight adjacent lanes publish one contiguous peer packet.
          push(peers, offset + column,
               ((generation & 0xffff) << 16) |
                   __half_as_ushort(__float2half_rn(value)));
        }
      }
    }
  } else {
    for (int row = 0; row < m; ++row) {
      const int offset = ((generation & 1) * m + row) * PacketStride;
      for (int column = t; column < 324; column += 128) {
        const half value = __ushort_as_half(
            packet(peers.p[rank] + offset + column, generation) & 0xffff);
        gathered[row * 336 + column] = value;
        if (column < 320)
          lora[row * 320 + column] = value;
        else
          next_injection[row * 4 + column - 320] = value;
      }
    }
  }
  if (t == 0)
    for (int row = 0; row < m; ++row) epochs[row * Blocks + block] = generation;
}
void down(torch::Tensor residual, torch::Tensor core, torch::Tensor injection,
          torch::Tensor norm, torch::Tensor weight, torch::Tensor combined,
          torch::Tensor xn, torch::Tensor gathered, torch::Tensor lora,
          torch::Tensor next_injection, std::vector<int64_t> pointers,
          torch::Tensor epochs, torch::Tensor flags, torch::Tensor partial,
          int rank, double epsilon, bool reduce_core) {
  const int m = residual.size(0);
  TORCH_CHECK((m == 1 || m == 5) && residual.size(1) == 10240 &&
              pointers.size() == 4);
  TORCH_CHECK(epochs.numel() == m * Blocks && flags.numel() == m * 4 + 22 &&
              partial.numel() == 22 * m * 8 && weight.numel() == 88 * 10240);
  Peers peers;
  for (int r = 0; r < 4; ++r) peers.p[r] = (uint32_t*)pointers[r];
  const half* rp = (half*)residual.data_ptr();
  const half* cp = (half*)core.data_ptr();
  const half* ip = (half*)injection.data_ptr();
  const half* np = (half*)norm.data_ptr();
  const half* wp = (half*)weight.data_ptr();
  half* out = (half*)combined.data_ptr();
  half* xp = (half*)xn.data_ptr();
  half* gp = (half*)gathered.data_ptr();
  half* lp = (half*)lora.data_ptr();
  half* next = (half*)next_injection.data_ptr();
  auto* ep = (uint32_t*)epochs.data_ptr();
  auto* fp = (uint32_t*)flags.data_ptr();
  auto* pp = partial.data_ptr<float>();
  float eps = epsilon;
  void* args[] = {&rp,   &cp,    &ip, &np, &wp, &out,  &xp,       &gp, &lp,
                  &next, &peers, &ep, &fp, &pp, &rank, (void*)&m, &eps};
  const void* kernel = reduce_core ? (const void*)hc_coalesced<true>
                                   : (const void*)hc_coalesced<false>;
  int active = 0;
  C10_CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 128, 0));
  TORCH_CHECK(active *
                  at::cuda::getCurrentDeviceProperties()->multiProcessorCount >=
              Blocks);
  C10_CUDA_CHECK(
      cudaLaunchCooperativeKernel(kernel, dim3(Blocks), dim3(128), args, 0,
                                  c10::cuda::getCurrentCUDAStream()));
}
std::tuple<int64_t, pybind11::bytes> allocate(int m) {
  uint32_t* p;
  C10_CUDA_CHECK(cudaMalloc(&p, 2 * m * PacketStride * sizeof(uint32_t)));
  C10_CUDA_CHECK(cudaMemset(p, 0, 2 * m * PacketStride * sizeof(uint32_t)));
  cudaIpcMemHandle_t handle;
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, p));
  return {(int64_t)p, pybind11::bytes((char*)&handle, sizeof(handle))};
}
int64_t open_peer(pybind11::bytes bytes) {
  std::string data = bytes;
  TORCH_CHECK(data.size() == sizeof(cudaIpcMemHandle_t));
  cudaIpcMemHandle_t handle;
  memcpy(&handle, data.data(), sizeof(handle));
  void* p;
  C10_CUDA_CHECK(
      cudaIpcOpenMemHandle(&p, handle, cudaIpcMemLazyEnablePeerAccess));
  return (int64_t)p;
}

namespace mma_up {
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
template <bool PairRows, int Warps, int Unroll, bool FuseMix,
          bool FullOutput = false>
__device__ __forceinline__ void hc_up_batch_body(
    const half* __restrict__ lora, const half* __restrict__ packed,
    const half* __restrict__ branches, half* __restrict__ out, int rows,
    int hidden, int hidden_offset, int block_x, int block_y) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int tile = block_x * Warps + warp;
  if (tile >= hidden / 8) return;
  const int group = PairRows ? 0 : block_y;
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
          const int offset =
              FullOutput ? row * 2560 + hidden_offset + h : row * hidden + h;
          out[offset] = __float2half_rn(div_full(mixed, 4.0f));
        }
      } else if (row < rows) {
        out[row * 4 * hidden + branch * hidden + h] = gate;
      }
    }
  }
}

}  // namespace mma_up
__global__ __launch_bounds__(128, 4) void hc_up_mma(const half* lora,
                                                    const half* packed,
                                                    const half* xn, half* out,
                                                    int m) {
  mma_up::hc_up_batch_body<false, 4, 4, true, true>(lora, packed, xn, out, m,
                                                    2560, 0, blockIdx.x, 0);
}
void up(torch::Tensor lora, torch::Tensor packed, torch::Tensor xn,
        torch::Tensor output) {
  hc_up_mma<<<80, 128, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (half*)lora.data_ptr(), (half*)packed.data_ptr(), (half*)xn.data_ptr(),
      (half*)output.data_ptr(), lora.size(0));
  C10_CUDA_CHECK(cudaGetLastError());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("down", &down);
  module.def("up", &up);
  module.def("allocate", &allocate);
  module.def("open_peer", &open_peer);
  module.def("close_peer", [](int64_t p) {
    C10_CUDA_CHECK(cudaIpcCloseMemHandle((void*)p));
  });
  module.def("release", [](int64_t p) { C10_CUDA_CHECK(cudaFree((void*)p)); });
}
