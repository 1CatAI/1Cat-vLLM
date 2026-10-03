#include <torch/all.h>
#include <torch/cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <memory>

// Large resident UVA tables must not use the caching host allocator: it
// rounds a 9--12 GiB request to 16 GiB, defeating the model's host RAM budget.
torch::Tensor create_cuda_pinned_tensor(torch::Tensor& reference,
                                        at::IntArrayRef sizes) {
  TORCH_CHECK(reference.device().is_cpu(), "Reference tensor must be on CPU");
  size_t count = 1;
  for (const int64_t size : sizes) {
    TORCH_CHECK(size >= 0, "Pinned tensor dimensions must be nonnegative");
    TORCH_CHECK(size == 0 || count <= std::numeric_limits<size_t>::max() / size,
                "Pinned tensor size overflow");
    count *= size;
  }
  const size_t element_size = reference.element_size();
  TORCH_CHECK(count <= std::numeric_limits<size_t>::max() / element_size,
              "Pinned tensor byte size overflow");
  if (count == 0) return torch::empty(sizes, reference.options());

  void* host_ptr = nullptr;
  const cudaError_t error =
      cudaHostAlloc(&host_ptr, count * element_size, cudaHostAllocMapped);
  TORCH_CHECK(error == cudaSuccess,
              "cudaHostAlloc failed: ", cudaGetErrorString(error));
  auto owner =
      std::shared_ptr<void>(host_ptr, [](void* ptr) { cudaFreeHost(ptr); });
  return torch::from_blob(
      host_ptr, sizes, [owner = std::move(owner)](void*) {},
      reference.options());
}

// This function assumes that `cpu_tensor` is a CPU tensor,
// and that UVA (Unified Virtual Addressing) is enabled.
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  // handle empty tensor
  if (cpu_tensor.numel() == 0) {
    return torch::empty(cpu_tensor.sizes(),
                        cpu_tensor.options().device(torch::kCUDA));
  }

  if (cpu_tensor.is_pinned()) {
    // If CPU tensor is pinned, directly get the device pointer.
    void* host_ptr = const_cast<void*>(cpu_tensor.data_ptr());
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));

    return torch::from_blob(
        device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(),
        [base = cpu_tensor](void*) {},  // keep cpu tensor alive
        cpu_tensor.options().device(torch::kCUDA));
  }

  // If CPU tensor is not pinned, allocate a new pinned memory buffer.
  torch::Tensor contiguous_cpu = cpu_tensor.contiguous();
  size_t nbytes = contiguous_cpu.nbytes();

  void* host_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    AT_ERROR("cudaHostAlloc failed: ", cudaGetErrorString(err));
  }

  err = cudaMemcpy(host_ptr, contiguous_cpu.data_ptr(), nbytes,
                   cudaMemcpyDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    AT_ERROR("cudaMemcpy failed: ", cudaGetErrorString(err));
  }

  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    AT_ERROR("cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }

  auto deleter = [host_ptr](void*) { cudaFreeHost(host_ptr); };

  return torch::from_blob(device_ptr, contiguous_cpu.sizes(),
                          contiguous_cpu.strides(), deleter,
                          contiguous_cpu.options().device(torch::kCUDA));
}
