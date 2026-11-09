// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research-only TP4 HC screen. Reuses the source tree's push packet protocol,
// not an opaque communicator or a foreign/private shared library.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "../../csrc/custom_all_reduce.cuh"

namespace {
using namespace vllm;
using Pack = packed_t<half>::P;
constexpr size_t kBytes =
    kSm70Tp4PushAllreduceSignalBytes + 8 * kSm70Tp4PushAllreduceBytes;

__device__ __forceinline__ float divide(float a, float b) {
  float r;
  asm("div.full.f32 %0,%1,%2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float sigmoid(float x) {
  float e, d;
  asm("mul.f32 %0,%1,0fBFB8AA3B;" : "=f"(e) : "f"(x));
  asm("ex2.approx.f32 %0,%1;" : "=f"(e) : "f"(e));
  asm("add.f32 %0,%1,0f3F800000;" : "=f"(d) : "f"(e));
  return divide(1.0f, d);
}

template <bool Down>
__global__ void gather(RankData buffers, const void* input, half* output,
                       half* injection, int rank, int rows) {
  constexpr int cols = Down ? 88 : 640;
  constexpr int packs_per_row = cols / Pack::size;
  constexpr int stride = kSm70Tp4PushAllreduceBytes / sizeof(Pack);
  auto* local =
      const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[rank]));
  auto* epochs = reinterpret_cast<uint32_t*>(local);
  const unsigned epoch = epochs[blockIdx.x];
  const int base = epoch * 4 * stride;
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < rows * packs_per_row) {
    const int row = offset / packs_per_row;
    const int col = offset % packs_per_row * Pack::size;
    Pack value;
    if constexpr (Down) {
#pragma unroll
      for (int i = 0; i < Pack::size; ++i) {
        float acc = 0;
#pragma unroll
        for (int split = 0; split < 20; ++split) {
          acc += static_cast<const float*>(
              input)[(split * rows + row) * 96 + col + i];
        }
        const half projected = __float2half_rn(acc);
        if (col < 80) {
          const float x = divide(__half2float(projected), 4.0f);
          value.data[i] = __float2half_rn(__fmul_rn(x, sigmoid(x)));
        } else {
          value.data[i] = projected;
        }
      }
    } else {
      value = static_cast<const Pack*>(input)[offset];
    }
#pragma unroll
    for (int i = 0; i < Pack::size; ++i)
      sm70_push_escape_sentinel(value.data[i]);
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      auto* dest =
          const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[peer]));
      dest += kSm70Tp4PushAllreduceSignalBytes +
              (base + rank * stride) * sizeof(Pack);
      sm70_push_store_volatile_16b(value, dest, offset);
    }
    Pack values[4];
    while (true) {
      bool pending = false;
#pragma unroll
      for (int peer = 0; peer < 4; ++peer) {
        const void* source = local + kSm70Tp4PushAllreduceSignalBytes +
                             (base + peer * stride) * sizeof(Pack);
        sm70_push_load_volatile_16b(values[peer], source, offset);
#pragma unroll
        for (int i = 0; i < Pack::size; ++i)
          pending |= sm70_push_is_sentinel(values[peer].data[i]);
      }
      if (!pending) break;
    }
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      if constexpr (Down) {
        if (col < 80) {
          *reinterpret_cast<Pack*>(output + row * 320 + peer * 80 + col) =
              values[peer];
        } else if (peer == 3) {
          *reinterpret_cast<uint2*>(injection + row * 4) =
              *reinterpret_cast<uint2*>(&values[peer]);
        }
      } else {
        *reinterpret_cast<Pack*>(output + row * 2560 + peer * 640 + col) =
            values[peer];
      }
    }
    Pack empty;
#pragma unroll
    for (int i = 0; i < Pack::size; ++i)
      *reinterpret_cast<uint16_t*>(&empty.data[i]) =
          kSm70Tp4PushAllreduceSentinel;
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      void* source = local + kSm70Tp4PushAllreduceSignalBytes +
                     (base + peer * stride) * sizeof(Pack);
      sm70_push_store_volatile_16b(empty, source, offset);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) epochs[blockIdx.x] = (epoch + 1) % 2;
}

void run(const std::vector<int64_t>& pointers, int rank, torch::Tensor input,
         torch::Tensor output, torch::Tensor injection, bool down) {
  TORCH_CHECK(input.is_cuda());
  const c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(pointers.size() == 4 && rank >= 0 && rank < 4);
  RankData buffers{};
  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(pointers[i] != 0);
    buffers.ptrs[i] = reinterpret_cast<void*>(pointers[i]);
  }
  for (const auto& t : {input, output, injection})
    TORCH_CHECK(t.is_cuda() && t.device() == input.device() &&
                t.is_contiguous());
  TORCH_CHECK(output.dim() == 2 && output.scalar_type() == at::kHalf &&
              injection.scalar_type() == at::kHalf);
  const int rows = output.size(0);
  TORCH_CHECK(rows >= 2 && rows <= 16 &&
              injection.sizes() == torch::IntArrayRef({rows, 4}));
  if (down) {
    TORCH_CHECK(input.scalar_type() == at::kFloat &&
                input.sizes() == torch::IntArrayRef({20, rows, 96}) &&
                output.size(1) == 320);
  } else {
    TORCH_CHECK(input.scalar_type() == at::kHalf &&
                input.sizes() == torch::IntArrayRef({rows, 640}) &&
                output.size(1) == 2560);
  }
  const int blocks = (rows * (down ? 11 : 80) + 127) / 128;
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(D)                                                            \
  gather<D><<<blocks, 128, 0, stream>>>(                                     \
      buffers, input.data_ptr(), reinterpret_cast<half*>(output.data_ptr()), \
      reinterpret_cast<half*>(injection.data_ptr()), rank, rows)
  if (down) {
    LAUNCH(true);
  } else {
    LAUNCH(false);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<int64_t, pybind11::bytes> allocate() {
  void* p = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&p, kBytes));
  C10_CUDA_CHECK(cudaMemset(p, kSm70Tp4PushAllreduceSentinelByte, kBytes));
  C10_CUDA_CHECK(cudaMemset(p, 0, kSm70Tp4PushAllreduceSignalBytes));
  cudaIpcMemHandle_t handle;
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, p));
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  return {reinterpret_cast<int64_t>(p),
          pybind11::bytes(reinterpret_cast<char*>(&handle), sizeof(handle))};
}

int64_t ipc_open(pybind11::bytes data) {
  const std::string bytes = data;
  TORCH_CHECK(bytes.size() == sizeof(cudaIpcMemHandle_t));
  cudaIpcMemHandle_t handle;
  std::memcpy(&handle, bytes.data(), sizeof(handle));
  void* p = nullptr;
  C10_CUDA_CHECK(
      cudaIpcOpenMemHandle(&p, handle, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(p);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("allocate", &allocate);
  m.def("open", &ipc_open);
  m.def("close", [](int64_t p) {
    C10_CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(p)));
  });
  m.def("free", [](int64_t p) {
    C10_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(p)));
  });
}
