// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research screen: redundant per-output normalization, deterministic dot,
// producer-pushed generation packets, and a separate up/mix consumer.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <vector>

struct Peers {
  uint32_t* p[4];
};
constexpr int kPacketStride = 336 + 2560 + 4 * 2560;
__device__ float warp_sum(float v) {
  for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffff, v, d);
  return v;
}
__device__ float sigmoid(float v) { return 1.f / (1.f + expf(-v)); }

template <bool ReduceCore>
__global__ __launch_bounds__(256, 2) void hc_norm_down_push(
    const half* residual, const half* core, const half* injection,
    const half* norm, const half* weight, half* combined, half* branches,
    Peers peers, uint32_t* counters, int rank, int m, float eps) {
  const int n = blockIdx.x, t = threadIdx.x;
  const int branch = t / 64, lane = t % 64;
  __shared__ half xn[10240];
  __shared__ half reduced_core[ReduceCore ? 2560 : 1];
  __shared__ float partial[8];
  __shared__ float dot[8];
  for (int row = ReduceCore ? 0 : blockIdx.y;
       row < (ReduceCore ? m : blockIdx.y + 1); ++row) {
    const uint32_t current_gen = counters[row * 81 + n] + 1;
    if constexpr (ReduceCore) {
      const int base =
          ((current_gen & 1) * m + row) * kPacketStride + 336 + 2560;
      if (n < 10) {
        const int h = n * 256 + t;
        const uint32_t word = ((current_gen & 0xffff) << 16) |
                              __half_as_ushort(core[row * 2560 + h]);
#pragma unroll
        for (int dst = 0; dst < 4; ++dst)
          asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(
                           peers.p[dst] + base + rank * 2560 + h),
                       "r"(word)
                       : "memory");
      }
      for (int h = t; h < 2560; h += 256) {
        float sum = 0;
#pragma unroll
        for (int src = 0; src < 4; ++src) {
          uint32_t word;
          do {
            asm volatile("ld.volatile.global.u32 %0, [%1];"
                         : "=r"(word)
                         : "l"(peers.p[rank] + base + src * 2560 + h)
                         : "memory");
          } while ((word >> 16) != (current_gen & 0xffff));
          sum += __half2float(__ushort_as_half(word & 0xffff));
        }
        reduced_core[h] = __float2half_rn(sum);
      }
      __syncthreads();
    }
    float values[40];
    float sum = 0;
    const float scale =
        2.f * sigmoid(__half2float(injection[row * 4 + branch]) / 4.f);
#pragma unroll
    for (int i = 0; i < 40; ++i) {
      const int h = lane + i * 64, k = branch * 2560 + h;
      const half cv = ReduceCore ? reduced_core[h] : core[row * 2560 + h];
      const half v = __float2half_rn(fmaf(
          __half2float(cv), scale, __half2float(residual[row * 10240 + k])));
      values[i] = __half2float(v);
      sum = fmaf(values[i], values[i], sum);
      if (n == 0) combined[row * 10240 + k] = v;
    }
    sum = warp_sum(sum);
    if ((t & 31) == 0) partial[t / 32] = sum;
    __syncthreads();
    const float inv =
        rsqrtf((partial[branch * 2] + partial[branch * 2 + 1]) / 2560.f + eps);
#pragma unroll
    for (int i = 0; i < 40; ++i) {
      const int k = branch * 2560 + lane + i * 64;
      const float v = values[i] * inv;
      xn[k] = __float2half_rn(fmaf(v, __half2float(norm[k]), v));
      if (n == 0) branches[row * 10240 + k] = xn[k];
    }
    __syncthreads();
    const int wrow = n < 80 ? rank * 80 + n : 320 + rank;
    float acc = 0;
    for (int k = t; k < 10240; k += 256)
      acc = fmaf(__half2float(xn[k]), __half2float(weight[wrow * 10240 + k]),
                 acc);
    acc = warp_sum(acc);
    if ((t & 31) == 0) dot[t / 32] = acc;
    __syncthreads();
    if (t == 0) {
      acc = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) acc += dot[i];
      float v = __half2float(__float2half_rn(acc));
      if (n < 80) {
        v /= 4.f;
        v *= sigmoid(v);
      }
      const uint32_t gen = ++counters[row * 81 + n];
      const int column = n < 80 ? rank * 80 + n : 320 + rank;
      const int index = ((gen & 1) * m + row) * kPacketStride + column;
      const uint32_t packet =
          ((gen & 0xffff) << 16) | __half_as_ushort(__float2half_rn(v));
#pragma unroll
      for (int dst = 0; dst < 4; ++dst)
        asm volatile(
            "st.volatile.global.u32 [%0], %1;" ::"l"(peers.p[dst] + index),
            "r"(packet)
            : "memory");
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(256, 2) void hc_up_mix_pull(
    const half* weight, const half* branches, half* output, half* injection,
    Peers peers, uint32_t* counters, int m, int rank) {
  const int t = threadIdx.x, tile = blockIdx.x;
  __shared__ half lora[320];
  __shared__ float partial[16][2];
  for (int row = 0; row < m; ++row) {
    const uint32_t gen = counters[row * 160 + tile] + 1;
    const int offset = ((gen & 1) * m + row) * kPacketStride;
    for (int k = t; k < 324; k += 256) {
      uint32_t word;
      do {
        asm volatile("ld.volatile.global.u32 %0, [%1];"
                     : "=r"(word)
                     : "l"(peers.p[rank] + offset + k)
                     : "memory");
      } while ((word >> 16) != (gen & 0xffff));
      if (k < 320)
        lora[k] = __ushort_as_half(word & 0xffff);
      else if (tile == 0)
        injection[row * 4 + k - 320] = __ushort_as_half(word & 0xffff);
    }
    __syncthreads();
    // Two warps per hidden position. No CTA waits for another local CTA.
    const int warp = t / 32, lane = t % 32,
              hidden = rank * 640 + tile * 4 + warp / 2;
    for (int b = 0; b < 4; ++b) {
      float acc = 0;
#pragma unroll
      for (int e = 0; e < 5; ++e) {
        const int k = e * 64 + (warp % 2) * 32 + lane;
        acc = fmaf(__half2float(lora[k]),
                   __half2float(weight[(b * 2560 + hidden) * 320 + k]), acc);
      }
      acc = warp_sum(acc);
      if (lane == 0) partial[(warp / 2) * 4 + b][warp % 2] = acc;
    }
    __syncthreads();
    if (t < 4) {
      const int h = tile * 4 + t;
      float mixed = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        float v = partial[t * 4 + b][0] + partial[t * 4 + b][1];
        mixed = fmaf(
            sigmoid(__half2float(__float2half_rn(v))),
            __half2float(branches[row * 10240 + b * 2560 + rank * 640 + h]),
            mixed);
      }
      const half value = __float2half_rn(mixed / 4.f);
      const uint32_t packet = ((gen & 0xffff) << 16) | __half_as_ushort(value);
#pragma unroll
      for (int dst = 0; dst < 4; ++dst)
        asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(
                         peers.p[dst] + offset + 336 + rank * 640 + h),
                     "r"(packet)
                     : "memory");
#pragma unroll
      for (int src = 0; src < 4; ++src) {
        uint32_t word;
        do {
          asm volatile("ld.volatile.global.u32 %0, [%1];"
                       : "=r"(word)
                       : "l"(peers.p[rank] + offset + 336 + src * 640 + h)
                       : "memory");
        } while ((word >> 16) != (gen & 0xffff));
        output[row * 2560 + src * 640 + h] = __ushort_as_half(word & 0xffff);
      }
    }
    __syncthreads();
    if (t == 0) counters[row * 160 + tile] = gen;
    __syncthreads();
  }
}

void run(torch::Tensor residual, torch::Tensor core, torch::Tensor injection,
         torch::Tensor norm, torch::Tensor down, torch::Tensor up,
         torch::Tensor combined, torch::Tensor branches, torch::Tensor output,
         torch::Tensor new_injection, std::vector<int64_t> pointers,
         torch::Tensor down_counters, torch::Tensor up_counters, int rank,
         double eps, bool reduce_core = false) {
  TORCH_CHECK(residual.size(0) == 1 || residual.size(0) == 5);
  TORCH_CHECK(pointers.size() == 4 && rank >= 0 && rank < 4);
  const int m = residual.size(0);
  Peers peers;
  for (int i = 0; i < 4; ++i)
    peers.p[i] = reinterpret_cast<uint32_t*>(pointers[i]);
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  if (!reduce_core) {
    hc_norm_down_push<false><<<dim3(81, m), 256, 0, stream>>>(
        (half*)residual.data_ptr(), (half*)core.data_ptr(),
        (half*)injection.data_ptr(), (half*)norm.data_ptr(),
        (half*)down.data_ptr(), (half*)combined.data_ptr(),
        (half*)branches.data_ptr(), peers, (uint32_t*)down_counters.data_ptr(),
        rank, m, eps);
  } else {
    const half* rp = (half*)residual.data_ptr();
    const half* cp = (half*)core.data_ptr();
    const half* ip = (half*)injection.data_ptr();
    const half* np = (half*)norm.data_ptr();
    const half* wp = (half*)down.data_ptr();
    half* combined_p = (half*)combined.data_ptr();
    half* branches_p = (half*)branches.data_ptr();
    uint32_t* counters_p = (uint32_t*)down_counters.data_ptr();
    float epsilon = eps;
    void* args[] = {&rp,         &cp,         &ip,         &np,
                    &wp,         &combined_p, &branches_p, &peers,
                    &counters_p, &rank,       (void*)&m,   &epsilon};
    const auto status = cudaLaunchCooperativeKernel(
        (void*)hc_norm_down_push<true>, dim3(81), dim3(256), args, 0, stream);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
  }
  const half* wp = (half*)up.data_ptr();
  const half* bp = (half*)branches.data_ptr();
  half* op = (half*)output.data_ptr();
  half* ip = (half*)new_injection.data_ptr();
  uint32_t* cp = (uint32_t*)up_counters.data_ptr();
  void* args[] = {&wp, &bp, &op, &ip, &peers, &cp, (void*)&m, &rank};
  const auto status = cudaLaunchCooperativeKernel(
      (void*)hc_up_mix_pull, dim3(160), dim3(256), args, 0, stream);
  TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
std::tuple<int64_t, pybind11::bytes> allocate(int m) {
  uint32_t* p;
  TORCH_CHECK(cudaMalloc(&p, 2 * m * kPacketStride * sizeof(uint32_t)) ==
              cudaSuccess);
  TORCH_CHECK(cudaMemset(p, 0, 2 * m * kPacketStride * sizeof(uint32_t)) ==
              cudaSuccess);
  cudaIpcMemHandle_t handle;
  TORCH_CHECK(cudaIpcGetMemHandle(&handle, p) == cudaSuccess);
  return {(int64_t)p, pybind11::bytes((char*)&handle, sizeof(handle))};
}
int64_t open_peer(pybind11::bytes bytes) {
  std::string data = bytes;
  TORCH_CHECK(data.size() == sizeof(cudaIpcMemHandle_t));
  cudaIpcMemHandle_t handle;
  memcpy(&handle, data.data(), sizeof(handle));
  void* p;
  TORCH_CHECK(cudaIpcOpenMemHandle(
                  &p, handle, cudaIpcMemLazyEnablePeerAccess) == cudaSuccess);
  return (int64_t)p;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run", &run, pybind11::arg("residual"), pybind11::arg("core"),
             pybind11::arg("injection"), pybind11::arg("norm"),
             pybind11::arg("down"), pybind11::arg("up"),
             pybind11::arg("combined"), pybind11::arg("branches"),
             pybind11::arg("output"), pybind11::arg("new_injection"),
             pybind11::arg("pointers"), pybind11::arg("down_counters"),
             pybind11::arg("up_counters"), pybind11::arg("rank"),
             pybind11::arg("eps"), pybind11::arg("reduce_core") = false);
  module.def("allocate", &allocate);
  module.def("open_peer", &open_peer);
  module.def("close_peer", [](int64_t p) { cudaIpcCloseMemHandle((void*)p); });
  module.def("release", [](int64_t p) { cudaFree((void*)p); });
}
