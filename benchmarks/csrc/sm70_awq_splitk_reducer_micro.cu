// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ int load_acquire(const int* ptr) {
  int value;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
               : "=r"(value)
               : "l"(ptr));
  return value;
}

__device__ __forceinline__ void store_release(int* ptr, int value) {
  asm volatile("st.global.release.gpu.b32 [%0], %1;\n"
               :
               : "l"(ptr), "r"(value));
}

__device__ void wait_for_split(int* lock, int split) {
  int state = 0;
  while (__syncthreads_and(state != split)) {
    state = threadIdx.x == 0 ? load_acquire(lock) : 0;
  }
  __syncthreads();
}

__global__ void serial_reduce_kernel(const float* __restrict__ partials,
                                     float* __restrict__ running,
                                     int* __restrict__ locks,
                                     float* __restrict__ output,
                                     int tiles,
                                     int elements,
                                     int splits) {
  const int tile = blockIdx.x;
  const int split = blockIdx.y;
  if (tile >= tiles || split >= splits) {
    return;
  }

  int* lock = locks + tile;
  wait_for_split(lock, split);

  const int64_t offset = (static_cast<int64_t>(split) * tiles + tile) * elements;
  float* running_tile = running + static_cast<int64_t>(tile) * elements;
  float* output_tile = output + static_cast<int64_t>(tile) * elements;
  for (int index = threadIdx.x; index < elements; index += blockDim.x) {
    const float prior = split == 0 ? 0.0f : running_tile[index];
    const float value = partials[offset + index] + prior;
    running_tile[index] = value;
    if (split == splits - 1) {
      output_tile[index] = value;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    store_release(lock, split == splits - 1 ? 0 : split + 1);
  }
}

__global__ void last_arrival_reduce_kernel(const float* __restrict__ partials,
                                           float* __restrict__ staged,
                                           int* __restrict__ counters,
                                           float* __restrict__ output,
                                           int tiles,
                                           int elements,
                                           int splits) {
  const int tile = blockIdx.x;
  const int split = blockIdx.y;
  if (tile >= tiles || split >= splits) {
    return;
  }

  const int64_t offset = (static_cast<int64_t>(split) * tiles + tile) * elements;
  for (int index = threadIdx.x; index < elements; index += blockDim.x) {
    staged[offset + index] = partials[offset + index];
  }

  // Each lane publishes its own staged stores before CTA thread 0 releases
  // the tile arrival counter. A thread-0 fence alone is not sufficient.
  __threadfence();
  __syncthreads();

  __shared__ int is_last;
  if (threadIdx.x == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> counter(counters[tile]);
    is_last = counter.fetch_add(1, cuda::memory_order_acq_rel) == splits - 1;
  }
  __syncthreads();
  if (!is_last) {
    return;
  }

  float* output_tile = output + static_cast<int64_t>(tile) * elements;
  for (int index = threadIdx.x; index < elements; index += blockDim.x) {
    float value = staged[static_cast<int64_t>(tile) * elements + index];
#pragma unroll 1
    for (int z = 1; z < splits; ++z) {
      value = staged[(static_cast<int64_t>(z) * tiles + tile) * elements + index] + value;
    }
    output_tile[index] = value;
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> counter(counters[tile]);
    counter.store(0, cuda::memory_order_release);
  }
}

void check_inputs(const torch::Tensor& partials,
                  const torch::Tensor& workspace,
                  const torch::Tensor& locks,
                  const torch::Tensor& output) {
  TORCH_CHECK(partials.is_cuda(), "partials must be CUDA.");
  TORCH_CHECK(partials.scalar_type() == torch::kFloat32,
              "partials must be float32.");
  TORCH_CHECK(partials.is_contiguous(), "partials must be contiguous.");
  TORCH_CHECK(partials.dim() == 3, "partials must have shape [splits, tiles, elements].");
  TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == torch::kFloat32,
              "workspace must be CUDA float32.");
  TORCH_CHECK(workspace.sizes() == partials.sizes(),
              "workspace must match partials shape.");
  TORCH_CHECK(locks.is_cuda() && locks.scalar_type() == torch::kInt32,
              "locks must be CUDA int32.");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kFloat32,
              "output must be CUDA float32.");
  TORCH_CHECK(partials.get_device() == workspace.get_device() &&
                  partials.get_device() == locks.get_device() &&
                  partials.get_device() == output.get_device(),
              "all tensors must use one CUDA device.");

  const int splits = static_cast<int>(partials.size(0));
  const int tiles = static_cast<int>(partials.size(1));
  const int elements = static_cast<int>(partials.size(2));
  TORCH_CHECK(splits > 1 && tiles > 0 && elements > 0,
              "partials dimensions must be positive and splits must exceed one.");
  TORCH_CHECK(locks.numel() == tiles, "locks must have one element per tile.");
  TORCH_CHECK(output.size(0) == tiles && output.size(1) == elements,
              "output must have shape [tiles, elements].");
}

void launch_serial(torch::Tensor partials,
                   torch::Tensor running,
                   torch::Tensor locks,
                   torch::Tensor output) {
  check_inputs(partials, running, locks, output);
  const c10::cuda::CUDAGuard guard(partials.device());
  const dim3 grid(partials.size(1), partials.size(0));
  const auto stream = at::cuda::getCurrentCUDAStream(partials.get_device());
  serial_reduce_kernel<<<grid, kThreads, 0, stream>>>(
      partials.data_ptr<float>(), running.data_ptr<float>(), locks.data_ptr<int>(),
      output.data_ptr<float>(), partials.size(1), partials.size(2), partials.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_last_arrival(torch::Tensor partials,
                         torch::Tensor staged,
                         torch::Tensor counters,
                         torch::Tensor output) {
  check_inputs(partials, staged, counters, output);
  const c10::cuda::CUDAGuard guard(partials.device());
  const dim3 grid(partials.size(1), partials.size(0));
  const auto stream = at::cuda::getCurrentCUDAStream(partials.get_device());
  last_arrival_reduce_kernel<<<grid, kThreads, 0, stream>>>(
      partials.data_ptr<float>(), staged.data_ptr<float>(), counters.data_ptr<int>(),
      output.data_ptr<float>(), partials.size(1), partials.size(2), partials.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("serial", &launch_serial, "SM70 serial split-K reducer");
  m.def("last_arrival", &launch_last_arrival,
        "SM70 last-arrival split-K reducer");
}
