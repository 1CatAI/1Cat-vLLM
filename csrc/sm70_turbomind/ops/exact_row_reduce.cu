// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstring>
#include <array>

struct Peers {
  float* data[4];
  unsigned* flags[4];
};
constexpr int GRID = 80;
__device__ __forceinline__ void publish(unsigned* p, unsigned value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(value)
               : "memory");
}
__device__ __forceinline__ unsigned acquire(unsigned* p) {
  unsigned value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(p)
               : "memory");
  return value;
}
__device__ __forceinline__ void barrier(Peers peers, int rank, unsigned epoch,
                                        int phase) {
  __syncthreads();
  if (threadIdx.x < 4) {
    int peer = threadIdx.x;
    int location = phase * GRID * 4 + blockIdx.x * 4;
    publish(peers.flags[peer] + location + rank, epoch);
    while (acquire(peers.flags[rank] + location + peer) != epoch) {
    }
  }
  __syncthreads();
}
__device__ __forceinline__ float tree(unsigned code, float x0, float x1,
                                      float x2, float x3) {
  switch (code) {
    case 0:
      return __fadd_rn(x0, __fadd_rn(x1, __fadd_rn(x2, x3)));
    case 1:
      return __fadd_rn(x0, __fadd_rn(__fadd_rn(x1, x2), x3));
    case 2:
      return __fadd_rn(x0, __fadd_rn(__fadd_rn(x1, x3), x2));
    case 3:
      return __fadd_rn(__fadd_rn(x0, x1), __fadd_rn(x2, x3));
    case 4:
      return __fadd_rn(__fadd_rn(x0, x2), __fadd_rn(x1, x3));
    case 5:
      return __fadd_rn(__fadd_rn(x0, x3), __fadd_rn(x1, x2));
    case 6:
      return __fadd_rn(__fadd_rn(x0, __fadd_rn(x1, x2)), x3);
    case 7:
      return __fadd_rn(__fadd_rn(__fadd_rn(x0, x1), x2), x3);
    case 8:
      return __fadd_rn(__fadd_rn(__fadd_rn(x0, x2), x1), x3);
    case 9:
      return __fadd_rn(__fadd_rn(x0, __fadd_rn(x1, x3)), x2);
    case 10:
      return __fadd_rn(__fadd_rn(__fadd_rn(x0, x1), x3), x2);
    case 11:
      return __fadd_rn(__fadd_rn(__fadd_rn(x0, x3), x1), x2);
    case 12:
      return __fadd_rn(__fadd_rn(x0, __fadd_rn(x2, x3)), x1);
    case 13:
      return __fadd_rn(__fadd_rn(__fadd_rn(x0, x2), x3), x1);
    case 14:
      return __fadd_rn(__fadd_rn(__fadd_rn(x0, x3), x2), x1);
    default:
      return __int_as_float(0x7fffffff);
  }
}
__global__ void reduce_rows(Peers peers, const uint8_t* codes, float* output,
                            int64_t count, int64_t offset, int rank,
                            unsigned epoch) {
  barrier(peers, rank, epoch, 0);
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(gridDim.x) * blockDim.x) {
    int64_t at = offset + i;
    float x0 = __ldcg(peers.data[0] + at);
    float x1 = __ldcg(peers.data[1] + at);
    float x2 = __ldcg(peers.data[2] + at);
    float x3 = __ldcg(peers.data[3] + at);
    output[i] = tree(codes[i], x0, x1, x2, x3);
  }
  barrier(peers, rank, epoch, 1);
}
std::tuple<int64_t, pybind11::bytes> allocate(int64_t bytes) {
  TORCH_CHECK(bytes > 0);
  void* ptr = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&ptr, bytes));
  cudaIpcMemHandle_t handle;
  try {
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, ptr));
    C10_CUDA_CHECK(cudaMemset(ptr, 0, bytes));
  } catch (...) {
    cudaFree(ptr);
    throw;
  }
  return {reinterpret_cast<int64_t>(ptr),
          pybind11::bytes(reinterpret_cast<char*>(&handle), sizeof(handle))};
}
int64_t open_handle(pybind11::bytes bytes) {
  std::string value = bytes;
  TORCH_CHECK(value.size() == sizeof(cudaIpcMemHandle_t));
  cudaIpcMemHandle_t handle;
  std::memcpy(&handle, value.data(), sizeof(handle));
  void* ptr = nullptr;
  C10_CUDA_CHECK(
      cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(ptr);
}
void release(int64_t ptr, bool owner) {
  if (owner)
    C10_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ptr)));
  else
    C10_CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr)));
}
void run(torch::Tensor input, torch::Tensor codes, torch::Tensor output,
         std::vector<int64_t> pointers, std::vector<int64_t> flags, int rank,
         unsigned epoch) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat32 &&
              input.is_contiguous());
  TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == torch::kUInt8 &&
              codes.is_contiguous());
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kFloat32 &&
              output.is_contiguous());
  TORCH_CHECK(input.device() == codes.device() &&
              input.device() == output.device());
  TORCH_CHECK(rank >= 0 && rank < 4 && pointers.size() == 4 &&
              flags.size() == 4 && epoch > 0);
  TORCH_CHECK(input.numel() == output.numel() * 4 &&
              codes.numel() == output.numel());
  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(
      properties->major == 7 && properties->minor == 0 &&
          properties->multiProcessorCount >= GRID,
      "Exact peer reduction requires an SM70 device with at least 80 SMs");
  Peers peers;
  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(pointers[i] && flags[i]);
    peers.data[i] = reinterpret_cast<float*>(pointers[i]);
    peers.flags[i] = reinterpret_cast<unsigned*>(flags[i]);
  }
  C10_CUDA_CHECK(cudaMemcpyAsync(peers.data[rank], input.data_ptr(),
                                 input.nbytes(), cudaMemcpyDeviceToDevice,
                                 stream));
  reduce_rows<<<GRID, 256, 0, stream>>>(
      peers, codes.data_ptr<uint8_t>(), output.data_ptr<float>(),
      output.numel(), output.numel() * rank, rank, epoch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("allocate", &allocate);
  m.def("open_handle", &open_handle);
  m.def("release", &release);
  m.def("run", &run);
}
