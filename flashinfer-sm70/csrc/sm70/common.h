// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>

using tvm::ffi::TensorView;

inline void check_cuda(cudaError_t error) {
  TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

inline void check_tensor(TensorView t, DLDevice device, int code, int bits) {
  TVM_FFI_ICHECK(t.device().device_type == kDLCUDA && t.device().device_id == device.device_id &&
                 t.dtype().code == code && t.dtype().bits == bits && t.dtype().lanes == 1)
      << "Invalid SM70 tensor device or dtype";
}

inline cudaStream_t stream_for(TensorView t) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, t.device().device_id));
}

inline void check_sm70() {
  int device, major, minor;
  check_cuda(cudaGetDevice(&device));
  check_cuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  check_cuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  TVM_FFI_ICHECK(major == 7 && minor == 0) << "This kernel requires SM70";
}
