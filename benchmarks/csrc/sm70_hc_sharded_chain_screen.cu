// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Down publishes peer packets; TP-sharded up/mix consumes and pushes output
// stripes.
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

__global__ __launch_bounds__(128, 3) void sharded_up_mix(
    const half* weight, const half* xn, half* output, half* injection,
    Peers peers, uint32_t* epochs, int rank, int m) {
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int block = blockIdx.x;
  const uint32_t generation = epochs[block] + 1;
  __shared__ half lora[320];
  for (int row = 0; row < m; ++row) {
    const int base = ((generation & 1) * m + row) * PacketStride;
    for (int k = t; k < 320; k += 128)
      lora[k] = __ushort_as_half(packet(peers.p[rank] + base + k, generation) &
                                 0xffff);
    if (block == 0 && t < 4)
      injection[row * 4 + t] = __ushort_as_half(
          packet(peers.p[rank] + base + 320 + t, generation) & 0xffff);
    __syncthreads();
    const int h = rank * 640 + block * 4 + warp;
    float mixed = 0;
#pragma unroll
    for (int branch = 0; branch < 4; ++branch) {
      float dot = 0;
      for (int k = lane; k < 320; k += 32)
        dot = fmaf(__half2float(lora[k]),
                   __half2float(weight[(branch * 2560 + h) * 320 + k]), dot);
      dot = warp_sum(dot);
      dot = __half2float(__float2half_rn(dot));
      const float gate = __half2float(__float2half_rn(sig(dot)));
      mixed =
          fmaf(gate, __half2float(xn[row * 10240 + branch * 2560 + h]), mixed);
    }
    if (lane == 0)
      push(peers, base + 336 + h,
           ((generation & 0xffff) << 16) |
               __half_as_ushort(__float2half_rn(mixed / 4.f)));
    __syncthreads();
    // Fixed per-CTA output stripes. Every producer has published before it
    // waits; consumers depend only on the packets for their own stripe.
    if (t < 16) {
      const int column = block * 16 + t;
      output[row * 2560 + column] = __ushort_as_half(
          packet(peers.p[rank] + base + 336 + column, generation) & 0xffff);
    }
    __syncthreads();
  }
  if (t == 0) epochs[block] = generation;
}
void up(torch::Tensor weight, torch::Tensor xn, torch::Tensor output,
        torch::Tensor injection, std::vector<int64_t> pointers,
        torch::Tensor epochs, int rank) {
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({10240, 320}) &&
              weight.is_contiguous());
  TORCH_CHECK(xn.dim() == 2 && xn.size(1) == 10240 &&
              (xn.size(0) == 1 || xn.size(0) == 5));
  TORCH_CHECK(pointers.size() == 4 && epochs.numel() == 160);
  Peers peers;
  for (int r = 0; r < 4; ++r) peers.p[r] = (uint32_t*)pointers[r];
  const half* w = (half*)weight.data_ptr();
  const half* x = (half*)xn.data_ptr();
  half* o = (half*)output.data_ptr();
  half* i = (half*)injection.data_ptr();
  auto* e = (uint32_t*)epochs.data_ptr();
  int m = xn.size(0);
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, sharded_up_mix, 128, 0));
  TORCH_CHECK(active *
                  at::cuda::getCurrentDeviceProperties()->multiProcessorCount >=
              160);
  void* args[] = {&w, &x, &o, &i, &peers, &e, &rank, &m};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      (const void*)sharded_up_mix, dim3(160), dim3(128), args, 0,
      c10::cuda::getCurrentCUDAStream()));
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
