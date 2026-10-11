// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/tuner/cache_utils.h"
#include <stdexcept>
#include <string>

namespace turbomind::gemm {

CacheFlushing::CacheFlushing() {
  cudaDeviceProp props{};
  int device{};
  auto status = cudaGetDevice(&device);
  if (status == cudaSuccess) {
    status = cudaGetDeviceProperties(&props, device);
  }
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("TurboMind tuner device query failed: ") +
        cudaGetErrorString(status));
  }

  size_ = props.l2CacheSize;

  status = cudaMalloc(&buffer_, size_);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("TurboMind tuner L2 flush allocation failed: ") +
        cudaGetErrorString(status));
  }
}

void CacheFlushing::flush(cudaStream_t stream) {
  thread_local CacheFlushing inst{};
  inst(stream);
}

void CacheFlushing::operator()(cudaStream_t stream) const {
  cudaMemsetAsync(buffer_, 0, size_, stream);
}

}  // namespace turbomind::gemm
