// SPDX-License-Identifier: Apache-2.0
#include <flashinfer/activation.cuh>
#include <flashinfer/norm.cuh>
#include <flashinfer/sm70/rope.cuh>

#include "common.h"

namespace {
void sm70_norm(TensorView out, TensorView input, TensorView weight, double eps, bool gemma,
               bool residual) {
  auto device = input.device();
  check_tensor(input, device, kDLFloat, 16);
  check_tensor(out, device, kDLFloat, 16);
  check_tensor(weight, device, kDLFloat, 16);
  TVM_FFI_ICHECK(input.ndim() == 2 && out.ndim() == 2 && weight.ndim() == 1);
  const auto rows = input.size(0), d = input.size(1);
  TVM_FFI_ICHECK(d > 0 && d <= 8192 && rows <= INT32_MAX && out.size(0) == rows &&
                 out.size(1) == d && weight.size(0) == d && input.stride(1) == 1 &&
                 out.stride(1) == 1 && weight.stride(0) == 1 && input.stride(0) >= d &&
                 out.stride(0) >= d && input.stride(0) <= UINT32_MAX &&
                 out.stride(0) <= UINT32_MAX && std::isfinite(eps) && eps > 0);
  const int vector_width = std::gcd(8, static_cast<int>(d));
  for (auto t : {input, out, weight})
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % (2 * vector_width) == 0)
        << "SM70 RMSNorm requires aligned vector storage";
  TVM_FFI_ICHECK(input.stride(0) % vector_width == 0 && out.stride(0) % vector_width == 0);
  tvm::ffi::CUDADeviceGuard guard(device.device_id);
  check_sm70();
  if (!rows) return;
  auto* x = static_cast<half*>(input.data_ptr());
  auto* y = static_cast<half*>(out.data_ptr());
  auto* w = static_cast<half*>(weight.data_ptr());
  auto stream = stream_for(input);
  // Reuse upstream vectorization and FP32 reductions. PDL is always disabled.
  if (residual) {
    check_cuda(gemma ? flashinfer::norm::GemmaFusedAddRMSNorm(x, y, w, rows, d, input.stride(0),
                                                              out.stride(0), eps, false, stream)
                     : flashinfer::norm::FusedAddRMSNorm(x, y, w, rows, d, input.stride(0),
                                                         out.stride(0), eps, false, stream));
  } else {
    check_cuda(gemma ? flashinfer::norm::GemmaRMSNorm(x, w, y, rows, d, input.stride(0),
                                                      out.stride(0), eps, false, stream)
                     : flashinfer::norm::RMSNorm(x, w, y, rows, d, input.stride(0), out.stride(0),
                                                 eps, false, stream));
  }
}

__device__ __forceinline__ float silu(const float& x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float gelu(const float& x) {
  return x * .5f * (1.f + erff(x * M_SQRT1_2));
}
__device__ __forceinline__ float gelu_tanh(const float& x) {
  return .5f * x * (1.f + tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
}

void activation(TensorView out, TensorView input, int64_t kind) {
  check_tensor(input, input.device(), kDLFloat, 16);
  check_tensor(out, input.device(), kDLFloat, 16);
  TVM_FFI_ICHECK(input.ndim() == 2 && out.ndim() == 2 && kind >= 0 && kind <= 2);
  int64_t rows = input.size(0), d = out.size(1);
  TVM_FFI_ICHECK(d > 0 && d <= INT32_MAX / 2 && rows <= INT32_MAX && input.size(1) == 2 * d &&
                 out.size(0) == rows && input.stride(1) == 1 && out.stride(1) == 1 &&
                 input.stride(0) == 2 * d && out.stride(0) == d);
  tvm::ffi::CUDADeviceGuard guard(input.device().device_id);
  check_sm70();
  if (!rows) return;
  const int vec = std::gcd(8, static_cast<int>(d));
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % (2 * vec) == 0 &&
                 reinterpret_cast<uintptr_t>(out.data_ptr()) % (2 * vec) == 0)
      << "SM70 activation requires aligned vector storage";
  const int threads = std::min<int>(256, d / vec);
#define LAUNCH_ACT(V, F)                                                                           \
  flashinfer::activation::act_and_mul_kernel<half, F, V><<<rows, threads, 0, stream_for(input)>>>( \
      static_cast<half*>(out.data_ptr()), static_cast<half*>(input.data_ptr()), d)
#define LAUNCH_VEC(V)         \
  if (kind == 0) {            \
    LAUNCH_ACT(V, silu);      \
  } else if (kind == 1) {     \
    LAUNCH_ACT(V, gelu);      \
  } else {                    \
    LAUNCH_ACT(V, gelu_tanh); \
  }
  if (vec == 8) {
    LAUNCH_VEC(8);
  } else if (vec == 4) {
    LAUNCH_VEC(4);
  } else if (vec == 2) {
    LAUNCH_VEC(2);
  } else {
    LAUNCH_VEC(1);
  }
#undef LAUNCH_VEC
#undef LAUNCH_ACT
  check_cuda(cudaGetLastError());
}

void rope(TensorView out, TensorView input, TensorView positions, int64_t rotary_dim,
          bool interleave, double scale, double theta) {
  check_tensor(input, input.device(), kDLFloat, 16);
  check_tensor(out, input.device(), kDLFloat, 16);
  TVM_FFI_ICHECK(input.ndim() == 3 && out.ndim() == 3 && positions.ndim() == 1);
  TVM_FFI_ICHECK(positions.dtype().bits == 32 || positions.dtype().bits == 64);
  check_tensor(positions, input.device(), kDLInt, positions.dtype().bits);
  for (int i = 0; i < 3; ++i) TVM_FFI_ICHECK(input.size(i) == out.size(i));
  TVM_FFI_ICHECK(input.stride(2) == 1 && out.stride(2) == 1 && input.size(0) == positions.size(0) &&
                 positions.stride(0) == 1 && input.size(2) > 0 && input.size(2) <= 8192 &&
                 rotary_dim > 0 && rotary_dim <= input.size(2) && rotary_dim % 2 == 0 &&
                 std::isfinite(scale) && scale > 0 && std::isfinite(theta) && theta > 0);
  tvm::ffi::CUDADeviceGuard guard(input.device().device_id);
  check_sm70();
  const int64_t count = input.size(0) * input.size(1) * input.size(2);
  if (!count) return;
#define LAUNCH_ROPE(P)                                                                             \
  flashinfer::sm70::rope_pos_ids<P><<<(count + 255) / 256, 256, 0, stream_for(input)>>>(           \
      static_cast<half*>(out.data_ptr()), static_cast<half*>(input.data_ptr()),                    \
      static_cast<P*>(positions.data_ptr()), count, input.size(1), input.size(2), input.stride(0), \
      input.stride(1), out.stride(0), out.stride(1), rotary_dim, interleave, scale, theta)
  if (positions.dtype().bits == 64) {
    LAUNCH_ROPE(int64_t);
  } else {
    LAUNCH_ROPE(int32_t);
  }
#undef LAUNCH_ROPE
  check_cuda(cudaGetLastError());
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(norm, sm70_norm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation, activation);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rope, rope);
