// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__device__ unsigned long long timer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__global__ void boundary_kernel(int* value, int* dependency,
                                unsigned long long* times, int slot) {
  int x = *dependency;
  times[slot] = timer_ns();
  *value = x + 1;
}
__device__ unsigned load_flag(unsigned* p) {
  unsigned x;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(x)
               : "l"(p)
               : "memory");
  return x;
}
__device__ void store_flag(unsigned* p, unsigned x) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(x)
               : "memory");
}
__global__ void roundtrip_kernel(unsigned* local, unsigned* peer, int* status,
                                 unsigned begin, unsigned count,
                                 bool initiator) {
  for (unsigned g = begin; g < begin + count; ++g) {
    if (initiator) store_flag(peer, g);
    auto start = clock64();
    while (load_flag(local) < g) {
      if (clock64() - start > 3000000000ULL) {
        *status = 1;
        return;
      }
    }
    if (!initiator) store_flag(peer, g);
  }
}
void boundary(torch::Tensor value, torch::Tensor dependency,
              torch::Tensor times, int slot) {
  c10::cuda::CUDAGuard guard(value.device());
  boundary_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      value.data_ptr<int>(), dependency.data_ptr<int>(),
      reinterpret_cast<unsigned long long*>(times.data_ptr<int64_t>()), slot);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void enable_peer(int device, int peer) {
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  auto error = cudaDeviceEnablePeerAccess(peer, 0);
  if (error == cudaErrorPeerAccessAlreadyEnabled)
    cudaGetLastError();
  else
    TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
void roundtrip(torch::Tensor local, torch::Tensor peer, torch::Tensor status,
               unsigned begin, unsigned count, bool initiator) {
  c10::cuda::CUDAGuard guard(local.device());
  roundtrip_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<unsigned*>(local.data_ptr<int>()),
      reinterpret_cast<unsigned*>(peer.data_ptr<int>()), status.data_ptr<int>(),
      begin, count, initiator);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("boundary", &boundary);
  m.def("enable_peer", &enable_peer);
  m.def("roundtrip", &roundtrip);
}
