// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#include "ggml_constants.cuh"

namespace {

[[noreturn]] void throw_formatted(const std::string& prefix, const char* fmt,
                                  va_list args) {
  char message[1024];
  std::vsnprintf(message, sizeof(message), fmt, args);
  throw std::runtime_error(prefix + message);
}

int device_attribute(cudaDeviceAttr attribute, int device, int fallback) {
  int value = fallback;
  const cudaError_t error = cudaDeviceGetAttribute(&value, attribute, device);
  if (error != cudaSuccess) {
    cudaGetLastError();
    return fallback;
  }
  return value;
}

}  // namespace

const ggml_cuda_device_info& ggml_cuda_info() {
  static const ggml_cuda_device_info info = [] {
    ggml_cuda_device_info result{};
    int device_count = 0;
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA device discovery failed: ") +
                               cudaGetErrorString(error));
    }
    if (device_count > GGML_CUDA_MAX_DEVICES) {
      throw std::runtime_error("CUDA device count exceeds GGML limit");
    }
    result.device_count = device_count;
    result.physical_device_count = device_count;
    for (int device = 0; device < device_count; ++device) {
      cudaDeviceProp properties{};
      if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        throw std::runtime_error("CUDA device property query failed");
      }
      auto& output = result.devices[device];
      output.cc = properties.major * 100 + properties.minor * 10;
      output.nsm = properties.multiProcessorCount;
      output.smpb = properties.sharedMemPerBlock;
      output.smpbo = static_cast<size_t>(
          device_attribute(cudaDevAttrMaxSharedMemoryPerBlockOptin, device,
                           static_cast<int>(properties.sharedMemPerBlock)));
      output.integrated = properties.integrated != 0;
      // This adapter does not create a GGML VMM pool.
      output.vmm = false;
      output.total_vram = properties.totalGlobalMem;
      output.warp_size = properties.warpSize;
      output.supports_cooperative_launch =
          device_attribute(cudaDevAttrCooperativeLaunch, device, 0) != 0;
      output.physical_device = device;
      output.physical_share_count = 1;
      output.virtual_index = 0;
    }
    return result;
  }();
  return info;
}

void ggml_cuda_set_device(int device) {
  const auto& info = ggml_cuda_info();
  if (device < 0 || device >= info.device_count) {
    throw std::runtime_error("invalid CUDA device index");
  }
  const cudaError_t error = cudaSetDevice(info.devices[device].physical_device);
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                             cudaGetErrorString(error));
  }
}

int ggml_cuda_get_device() {
  int device = -1;
  const cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("cudaGetDevice failed: ") +
                             cudaGetErrorString(error));
  }
  return device;
}

[[noreturn]] void ggml_cuda_error(const char* stmt, const char* func,
                                  const char* file, int line, const char* msg) {
  char buffer[1400];
  std::snprintf(buffer, sizeof(buffer),
                "upstream CUDA error: %s (%s at %s:%d): %s", msg, func, file,
                line, stmt);
  throw std::runtime_error(buffer);
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() = default;

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(
    int /*device*/, int /*stream_no*/) {
  GGML_ABORT("TorchScratchPool was not installed for the upstream context");
}

extern "C" {

int64_t ggml_nelements(const ggml_tensor* tensor) {
  int64_t result = 1;
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    result *= tensor->ne[i];
  }
  return result;
}

size_t ggml_element_size(const ggml_tensor* tensor) {
  return ggml_type_size(tensor->type);
}

size_t ggml_nbytes(const ggml_tensor* tensor) {
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (tensor->ne[i] <= 0) {
      return 0;
    }
  }
  const size_t block_size = ggml_blck_size(tensor->type);
  size_t result = 0;
  if (block_size == 1) {
    result = ggml_type_size(tensor->type);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
      result += (tensor->ne[i] - 1) * tensor->nb[i];
    }
  } else {
    result = tensor->ne[0] * tensor->nb[0] / block_size;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
      result += (tensor->ne[i] - 1) * tensor->nb[i];
    }
  }
  return result;
}

bool ggml_is_contiguous(const ggml_tensor* tensor) {
  const size_t block_size = ggml_blck_size(tensor->type);
  size_t expected = ggml_type_size(tensor->type);
  if (tensor->ne[0] != block_size && tensor->nb[0] != expected) {
    return false;
  }
  expected *= tensor->ne[0] / block_size;
  for (int i = 1; i < GGML_MAX_DIMS; ++i) {
    if (tensor->ne[i] != 1 && tensor->nb[i] != expected) {
      return false;
    }
    expected *= tensor->ne[i];
  }
  return true;
}

bool ggml_is_contiguously_allocated(const ggml_tensor* tensor) {
  return ggml_nbytes(tensor) == ggml_nelements(tensor) *
                                    ggml_type_size(tensor->type) /
                                    ggml_blck_size(tensor->type);
}

bool ggml_are_same_stride(const ggml_tensor* t0, const ggml_tensor* t1) {
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (t0->nb[i] != t1->nb[i]) {
      return false;
    }
  }
  return true;
}

[[noreturn]] void ggml_abort(const char* file, int line, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  throw_formatted(
      "GGML abort at " + std::string(file) + ":" + std::to_string(line) + ": ",
      fmt, args);
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t /*buffer*/,
                                          const ggml_tensor* tensor) {
  return tensor == nullptr ? 0 : ggml_nbytes(tensor);
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(
    ggml_backend_buffer_t /*buffer*/) {
  // The selected raw entry only uses this query to decide whether to clear
  // GGML-owned compute padding. Torch owns the storage, so there is no GGML
  // compute buffer to clear.
  return GGML_BACKEND_BUFFER_USAGE_ANY;
}

const char* ggml_type_name(enum ggml_type type) {
  switch (type) {
#define GGUF_NAME_CASE(value, block, cpp_type, name, quantized, mmq) \
  case value:                                                        \
    return name;
    GGUF_GGML_TYPE_TRAITS(GGUF_NAME_CASE)
#undef GGUF_NAME_CASE
    default:
      return "unsupported";
  }
}

int64_t ggml_blck_size(enum ggml_type type) {
  switch (type) {
#define GGUF_BLOCK_CASE(value, block, cpp_type, name, quantized, mmq) \
  case value:                                                         \
    return block;
    GGUF_GGML_TYPE_TRAITS(GGUF_BLOCK_CASE)
#undef GGUF_BLOCK_CASE
    default:
      throw std::runtime_error("unsupported GGML block size in CUDA adapter");
  }
}

size_t ggml_type_size(enum ggml_type type) {
  switch (type) {
#define GGUF_SIZE_CASE(value, block, cpp_type, name, quantized, mmq) \
  case value:                                                        \
    return sizeof(cpp_type);
    GGUF_GGML_TYPE_TRAITS(GGUF_SIZE_CASE)
#undef GGUF_SIZE_CASE
    default:
      throw std::runtime_error("unsupported GGML type size in CUDA adapter");
  }
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
  const int64_t block_size = ggml_blck_size(type);
  if (ne < 0 || ne % block_size != 0) {
    throw std::runtime_error("GGML row size is not block aligned");
  }
  return static_cast<size_t>(ne / block_size) * ggml_type_size(type);
}

bool ggml_is_quantized(enum ggml_type type) {
  switch (type) {
#define GGUF_QUANT_CASE(value, block, cpp_type, name, quantized, mmq) \
  case value:                                                         \
    return quantized != 0;
    GGUF_GGML_TYPE_TRAITS(GGUF_QUANT_CASE)
#undef GGUF_QUANT_CASE
    default:
      throw std::runtime_error(
          "unsupported GGML quantization type in CUDA adapter");
  }
}

}  // extern "C"
