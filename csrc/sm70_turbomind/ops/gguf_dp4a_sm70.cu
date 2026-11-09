// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <type_traits>
#include "gguf_dp4a.cuh"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"

namespace {
using turbomind::gemm::StridedPtr;
using vllm::sm70_gguf::LatticeDot;
using vllm::sm70_gguf::Q8_1;

__global__ void quantize_q8(Q8_1* out, const half* input, int k) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= k) return;
  const float value = __half2float(input[int64_t(blockIdx.y) * k + column]);
  const int group = blockIdx.y * (k / 32) + column / 32;
  vllm::sm70_gguf::quantize_q8_1_warp(out + group, value);
}

__device__ float v41_swiglu(float gate, float up, float probability) {
  gate = gate > 10.f ? 10.f : gate;
  up = up > 10.f ? 10.f : up;
  up = up < -10.f ? -10.f : up;
  const float activated = __fdiv_rn(gate, __fadd_rn(1.f, expf(-gate)));
  return __fmul_rn(__fmul_rn(activated, up), probability);
}

__global__ void v41_swiglu_q8(Q8_1* out, const half* projection,
                              const float* probabilities) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= 576)
    return;  // Entire inactive warps; the 32-lane Q8 group stays full.
  const int route = blockIdx.y;
  const float gate = __half2float(projection[int64_t(route) * 1152 + col]);
  const float up = __half2float(projection[int64_t(route) * 1152 + 576 + col]);
  // Retain the FP16 down-input boundary before computing Q8_1 bytes.
  const float value =
      __half2float(__float2half_rn(v41_swiglu(gate, up, probabilities[route])));
  vllm::sm70_gguf::quantize_q8_1_warp(out + route * 18 + col / 32, value);
}

__global__ void v41_shared_swiglu(half* out, const half* gate, const half* up,
                                  int64_t count) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    out[index] = __float2half_rn(
        v41_swiglu(__half2float(gate[index]), __half2float(up[index]), 1.f));
}

template <int Type, bool Activated, class Index, int Lanes = 16,
          bool Quantized = false, bool Canonical = false,
          bool ScalarLut = false, bool BankAware = false, int FixedGroups = 0,
          bool RouteGuard = false>
__global__ void gate_up(void* output, const Q8_1* activation, const Index* ids,
                        const uint8_t* gate, const uint8_t* up, int n, int k,
                        int stride, int top_k,
                        const StridedPtr* gate_stats = nullptr,
                        const StridedPtr* up_stats = nullptr,
                        const int* reuse_mode = nullptr) {
  if constexpr (RouteGuard)
    if (*reuse_mode) return;
  using Dot = std::conditional_t<
      Canonical, vllm::sm70_gguf::CanonicalIntegerDot<Type>,
      std::conditional_t<ScalarLut, vllm::sm70_gguf::SignedLutDot<Type>,
                         LatticeDot<Type, BankAware>>>;
  __shared__ uint32_t book[Dot::kBookWords];
  __shared__ uint32_t masks[16];
  __shared__ half intermediate[32];
  extern __shared__ Q8_1 shared_x[];
  const int groups = k / 32, route = blockIdx.y;
  const Q8_1* x = activation + (route / top_k) * groups;
  for (int i = threadIdx.x; i < groups * 9; i += blockDim.x)
    reinterpret_cast<uint32_t*>(shared_x)[i] =
        reinterpret_cast<const uint32_t*>(x)[i];
  if constexpr (Canonical)
    __syncthreads();
  else
    Dot::initialize(book, masks);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  constexpr int Rows = Quantized ? 32 : 8;
  const int local_row = warp * (32 / Lanes) + lane / Lanes;
  const int row = blockIdx.x * Rows + local_row;
  if (row >= n) return;
  const int64_t expert_row = ids[route] * n + row;
  const uint8_t* g = gate + expert_row * stride;
  const uint8_t* u = up + expert_row * stride;
  float gs = 0.f, us = 0.f;
  if constexpr (FixedGroups > 0 && !Canonical) {
    // Compile-time K: each lane's groups are a fixed, fully unrolled list, so
    // every group's weight loads can be in flight together instead of one
    // dependent global round trip per loop iteration. The accumulation order
    // (lane's groups ascending, gate and up separately) is unchanged.
    static_assert(FixedGroups % Lanes == 0);
#pragma unroll
    for (int i = 0; i < FixedGroups / Lanes; ++i) {
      const int group = lane % Lanes + i * Lanes;
      gs += Dot::dot(g, group, shared_x[group], book, masks);
      us += Dot::dot(u, group, shared_x[group], book, masks);
    }
  } else {
    for (int group = lane % Lanes; group < groups; group += Lanes) {
      if constexpr (Canonical) {
        const int expert = ids[route];
        const auto* gate_ptrs = reinterpret_cast<const StridedPtr*>(gate);
        const auto* up_ptrs = reinterpret_cast<const StridedPtr*>(up);
        gs += Dot::dot(gate_ptrs[expert].ptr, gate_stats[expert].ptr, n, k, row,
                       group, shared_x[group]);
        us += Dot::dot(up_ptrs[expert].ptr, up_stats[expert].ptr, n, k, row,
                       group, shared_x[group]);
      } else {
        if constexpr (Type == 16 || Type == 17 || Type == 29) {
          gs = __fadd_rn(gs, Dot::dot(g, group, shared_x[group], book, masks));
          us = __fadd_rn(us, Dot::dot(u, group, shared_x[group], book, masks));
        } else {
          gs += Dot::dot(g, group, shared_x[group], book, masks);
          us += Dot::dot(u, group, shared_x[group], book, masks);
        }
      }
    }
  }
#pragma unroll
  for (int offset = Lanes / 2; offset; offset >>= 1) {
    gs += __shfl_down_sync(__activemask(), gs, offset, Lanes);
    us += __shfl_down_sync(__activemask(), us, offset, Lanes);
  }
  if (lane % Lanes == 0) {
    if constexpr (Activated) {
      // Keep the retained FP16 gate/up boundary before SiLU and multiply.
      const float g16 = __half2float(__float2half_rn(gs));
      const float u16 = __half2float(__float2half_rn(us));
      const half silu = __float2half_rn(g16 / (1.f + expf(-g16)));
      const half value = __hmul(silu, __float2half_rn(u16));
      if constexpr (Quantized)
        intermediate[local_row] = value;
      else
        static_cast<half*>(output)[int64_t(route) * n + row] = value;
    } else {
      static_cast<half*>(output)[int64_t(route) * 2 * n + row] =
          __float2half_rn(gs);
      static_cast<half*>(output)[int64_t(route) * 2 * n + n + row] =
          __float2half_rn(us);
    }
  }
  if constexpr (Quantized) {
    __syncthreads();
    if (threadIdx.x < 32)
      vllm::sm70_gguf::quantize_q8_1_warp(
          static_cast<Q8_1*>(output) + int64_t{route} * (n / 32) + blockIdx.x,
          __half2float(intermediate[threadIdx.x]));
  }
}

template <class Index, int Lanes, bool Quantized>
void launch_lut4_gate_up(torch::Tensor out, torch::Tensor activation,
                         torch::Tensor ids, torch::Tensor gate,
                         torch::Tensor gate_stats, torch::Tensor up,
                         torch::Tensor up_stats, int n) {
  constexpr int Rows = Quantized ? 32 : 8;
  gate_up<20, true, Index, Lanes, Quantized, true>
      <<<dim3(n / Rows, activation.size(0) * ids.size(1)),
         Quantized ? 32 * Lanes : 128, activation.size(1) * sizeof(Q8_1),
         at::cuda::getCurrentCUDAStream()>>>(
          out.data_ptr(), reinterpret_cast<const Q8_1*>(activation.data_ptr()),
          ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
          up.data_ptr<uint8_t>(), n, activation.size(1) * 32, 0, ids.size(1),
          reinterpret_cast<const StridedPtr*>(gate_stats.data_ptr()),
          reinterpret_cast<const StridedPtr*>(up_stats.data_ptr()));
}

template <class Index>
void dispatch_lut4_gate_up(torch::Tensor out, torch::Tensor activation,
                           torch::Tensor ids, torch::Tensor gate,
                           torch::Tensor gate_stats, torch::Tensor up,
                           torch::Tensor up_stats, int n, int lanes) {
  if (out.scalar_type() == torch::kFloat16)
    launch_lut4_gate_up<Index, 16, false>(out, activation, ids, gate,
                                          gate_stats, up, up_stats, n);
  else if (lanes == 4)
    launch_lut4_gate_up<Index, 4, true>(out, activation, ids, gate, gate_stats,
                                        up, up_stats, n);
  else if (lanes == 8)
    launch_lut4_gate_up<Index, 8, true>(out, activation, ids, gate, gate_stats,
                                        up, up_stats, n);
  else
    launch_lut4_gate_up<Index, 16, true>(out, activation, ids, gate, gate_stats,
                                         up, up_stats, n);
}

template <int Type, class Index, int Lanes, bool BankAware = false,
          int FixedGroups = 0>
void launch_quantized_gate_up(torch::Tensor out, torch::Tensor activation,
                              torch::Tensor ids, torch::Tensor gate,
                              torch::Tensor up) {
  const int n = gate.size(1), top_k = ids.size(1);
  gate_up<Type, true, Index, Lanes, true, false, false, BankAware, FixedGroups>
      <<<dim3(n / 32, activation.size(0) * top_k), 32 * Lanes,
         activation.size(1) * sizeof(Q8_1), at::cuda::getCurrentCUDAStream()>>>(
          out.data_ptr(), reinterpret_cast<const Q8_1*>(activation.data_ptr()),
          ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
          up.data_ptr<uint8_t>(), n, activation.size(1) * 32, gate.size(2),
          top_k);
}

template <int Type, class Index, bool BankAware = false>
void launch_gate_up(torch::Tensor out, torch::Tensor activation,
                    torch::Tensor ids, torch::Tensor gate, torch::Tensor up,
                    bool activated, int lanes) {
  if constexpr (Type == 16 || Type == 17 || Type == 29) {
    const int n = gate.size(1);
    gate_up<Type, false, Index, 16><<<dim3((n + 7) / 8, ids.numel()), 128,
                                      activation.size(1) * sizeof(Q8_1),
                                      at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr(), reinterpret_cast<const Q8_1*>(activation.data_ptr()),
        ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(),
        n, activation.size(1) * 32, gate.size(2), ids.size(1));
  } else {
    if (out.scalar_type() == torch::kUInt8) {
      if (lanes == 0)  // K2560, 16 lanes, unrolled bank-aware schedule
        launch_quantized_gate_up<Type, Index, 16, true, 80>(out, activation,
                                                            ids, gate, up);
      else if (lanes == 4)
        launch_quantized_gate_up<Type, Index, 4, BankAware>(out, activation,
                                                            ids, gate, up);
      else if (lanes == 8)
        launch_quantized_gate_up<Type, Index, 8, BankAware>(out, activation,
                                                            ids, gate, up);
      else
        launch_quantized_gate_up<Type, Index, 16, BankAware>(out, activation,
                                                             ids, gate, up);
      return;
    }
    const int n = gate.size(1), top_k = ids.size(1);
    const dim3 grid((n + 7) / 8, activation.size(0) * top_k);
    const auto stream = at::cuda::getCurrentCUDAStream();
    const size_t shared = activation.size(1) * sizeof(Q8_1);
    const auto output = reinterpret_cast<half*>(out.data_ptr());
    const auto x = reinterpret_cast<const Q8_1*>(activation.data_ptr());
    if (activated)
      gate_up<Type, true, Index, 16, false, false, false, BankAware>
          <<<grid, 128, shared, stream>>>(
              output, x, ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
              up.data_ptr<uint8_t>(), n, activation.size(1) * 32, gate.size(2),
              top_k);
    else
      gate_up<Type, false, Index, 16, false, false, false, BankAware>
          <<<grid, 128, shared, stream>>>(
              output, x, ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
              up.data_ptr<uint8_t>(), n, activation.size(1) * 32, gate.size(2),
              top_k);
  }
}

template <int Type, bool BankAware = false>
void dispatch_gate_up(torch::Tensor out, torch::Tensor activation,
                      torch::Tensor ids, torch::Tensor gate, torch::Tensor up,
                      bool activated, int lanes) {
  if (ids.scalar_type() == torch::kInt32)
    launch_gate_up<Type, int32_t, BankAware>(out, activation, ids, gate, up,
                                             activated, lanes);
  else
    launch_gate_up<Type, int64_t, BankAware>(out, activation, ids, gate, up,
                                             activated, lanes);
}

template <int Type, class Index>
void launch_scalar_lut_gate_up(torch::Tensor out, torch::Tensor activation,
                               torch::Tensor ids, torch::Tensor gate,
                               torch::Tensor up, bool activated) {
  const int n = gate.size(1), routes = activation.size(0) * ids.size(1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared = activation.size(1) * sizeof(Q8_1);
  const auto x = reinterpret_cast<const Q8_1*>(activation.data_ptr());
  if (out.scalar_type() == torch::kUInt8) {
    gate_up<Type, true, Index, 16, true, false, true>
        <<<dim3(n / 32, routes), 512, shared, stream>>>(
            out.data_ptr(), x, ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
            up.data_ptr<uint8_t>(), n, activation.size(1) * 32, gate.size(2),
            ids.size(1));
  } else if (activated) {
    gate_up<Type, true, Index, 16, false, false, true>
        <<<dim3((n + 7) / 8, routes), 128, shared, stream>>>(
            out.data_ptr(), x, ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
            up.data_ptr<uint8_t>(), n, activation.size(1) * 32, gate.size(2),
            ids.size(1));
  } else {
    gate_up<Type, false, Index, 16, false, false, true>
        <<<dim3((n + 7) / 8, routes), 128, shared, stream>>>(
            out.data_ptr(), x, ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
            up.data_ptr<uint8_t>(), n, activation.size(1) * 32, gate.size(2),
            ids.size(1));
  }
}

template <int Type>
void dispatch_scalar_lut_gate_up(torch::Tensor out, torch::Tensor activation,
                                 torch::Tensor ids, torch::Tensor gate,
                                 torch::Tensor up, bool activated) {
  if (ids.scalar_type() == torch::kInt32)
    launch_scalar_lut_gate_up<Type, int32_t>(out, activation, ids, gate, up,
                                             activated);
  else
    launch_scalar_lut_gate_up<Type, int64_t>(out, activation, ids, gate, up,
                                             activated);
}

// Quantize routed intermediate rows in shared memory, compute down directly
// from expert descriptors, and reduce route weights without sorting/gathering.
template <int Type, class Index, bool Quantized = false>
__global__ void down_unroute(half* output, const void* input, const Index* ids,
                             const float* route_weights,
                             const turbomind::gemm::StridedPtr* weights,
                             const turbomind::gemm::StridedPtr* stats, int n,
                             int k, int top_k) {
  extern __shared__ Q8_1 x[];
  __shared__ float partial[4][32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int groups = k / 32, token = blockIdx.y;
  if constexpr (Quantized) {
    const auto* source = static_cast<const uint32_t*>(input) +
                         int64_t{token} * top_k * groups * 9;
    for (int i = threadIdx.x; i < top_k * groups * 9; i += blockDim.x)
      reinterpret_cast<uint32_t*>(x)[i] = source[i];
  } else {
    for (int i = warp; i < top_k * groups; i += 4) {
      const half value = static_cast<const half*>(
          input)[int64_t{token} * top_k * k + i * 32 + lane];
      vllm::sm70_gguf::quantize_q8_1_warp(x + i, __half2float(value));
    }
  }
  __syncthreads();
  const int col = blockIdx.x * 32 + lane;
  float total = 0.f;
  for (int route = warp; route < top_k; route += 4) {
    const int slot = token * top_k + route;
    const int expert = ids[slot];
    float dot = 0.f;
    for (int group = 0; group < groups; ++group)
      dot += vllm::sm70_gguf::CanonicalIntegerDot<Type>::dot(
          weights[expert].ptr, stats[expert].ptr, n, k, col, group,
          x[route * groups + group]);
    // Preserve the down projection's FP16 boundary before FP32 route weighting.
    total += __half2float(__float2half_rn(dot)) * route_weights[slot];
  }
  partial[warp][lane] = total;
  __syncthreads();
  if (!warp) {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) sum += partial[i][lane];
    output[int64_t{token} * n + col] = __float2half_rn(sum);
  }
}

template <int Type, class Index, bool Quantized>
void launch_down_impl(torch::Tensor out, torch::Tensor input, torch::Tensor ids,
                      torch::Tensor route_weights, torch::Tensor weights,
                      torch::Tensor stats) {
  using turbomind::gemm::StridedPtr;
  const int top_k = ids.size(1), k = input.size(2) * (Quantized ? 32 : 1),
            n = out.size(1);
  down_unroute<Type, Index, Quantized>
      <<<dim3(n / 32, out.size(0)), 128, top_k*(k / 32) * sizeof(Q8_1),
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<half*>(out.data_ptr()), input.data_ptr(),
          ids.data_ptr<Index>(), route_weights.data_ptr<float>(),
          reinterpret_cast<const StridedPtr*>(weights.data_ptr()),
          reinterpret_cast<const StridedPtr*>(stats.data_ptr()), n, k, top_k);
}

template <int Type, class Index>
void launch_down(torch::Tensor out, torch::Tensor input, torch::Tensor ids,
                 torch::Tensor route_weights, torch::Tensor weights,
                 torch::Tensor stats) {
  if (input.scalar_type() == torch::kUInt8)
    launch_down_impl<Type, Index, true>(out, input, ids, route_weights, weights,
                                        stats);
  else
    launch_down_impl<Type, Index, false>(out, input, ids, route_weights,
                                         weights, stats);
}

template <int Type>
void dispatch_down(torch::Tensor out, torch::Tensor input, torch::Tensor ids,
                   torch::Tensor route_weights, torch::Tensor weights,
                   torch::Tensor stats) {
  if (ids.scalar_type() == torch::kInt32)
    launch_down<Type, int32_t>(out, input, ids, route_weights, weights, stats);
  else
    launch_down<Type, int64_t>(out, input, ids, route_weights, weights, stats);
}

void require_sm70() {
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0,
              "GGUF dp4a requires SM70");
}
}  // namespace

void gguf_quantize_q8_1_sm70_out(torch::Tensor out, torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.is_contiguous() &&
                  input.scalar_type() == torch::kFloat16 && input.size(1) > 0 &&
                  ((input.size(0) <= 20 && input.size(1) % 256 == 0) ||
                   ((input.size(0) > 0 && input.size(0) <= 36 &&
                     input.size(0) % 6 == 0) &&
                    input.size(1) == 576)) &&
                  out.device() == input.device() &&
                  out.scalar_type() == torch::kUInt8 && out.is_contiguous() &&
                  out.dim() == 3 && out.size(0) == input.size(0) &&
                  out.size(1) == input.size(1) / 32 &&
                  out.size(2) == sizeof(Q8_1),
              "Expected FP16 [M,K] and Q8_1 [M,K/32,36]");
  const c10::cuda::CUDAGuard guard(input.device());
  require_sm70();
  if (!input.size(0)) return;
  const dim3 grid((input.size(1) + 255) / 256, input.size(0));
  quantize_q8<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<Q8_1*>(out.data_ptr()),
      reinterpret_cast<const half*>(input.data_ptr()), input.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_v41_swiglu_q8_sm70_out(torch::Tensor out, torch::Tensor projection,
                                 torch::Tensor probabilities) {
  TORCH_CHECK(projection.is_cuda() && projection.is_contiguous() &&
                  projection.scalar_type() == torch::kFloat16 &&
                  projection.dim() == 4 && projection.size(0) > 0 &&
                  projection.size(0) <= 6 && projection.size(1) == 6 &&
                  projection.size(2) == 2 && projection.size(3) == 576 &&
                  out.device() == projection.device() && out.is_contiguous() &&
                  out.scalar_type() == torch::kUInt8 && out.dim() == 4 &&
                  out.size(0) == projection.size(0) && out.size(1) == 6 &&
                  out.size(2) == 18 && out.size(3) == sizeof(Q8_1) &&
                  probabilities.device() == projection.device() &&
                  probabilities.is_contiguous() &&
                  probabilities.scalar_type() == torch::kFloat32 &&
                  probabilities.dim() == 2 &&
                  probabilities.size(0) == projection.size(0) &&
                  probabilities.size(1) == 6,
              "V4.1 SwiGLU Q8 requires FP16 M1..6 projections and FP32 routes");
  const c10::cuda::CUDAGuard guard(projection.device());
  require_sm70();
  v41_swiglu_q8<<<dim3(3, projection.size(0) * 6), 256, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<Q8_1*>(out.data_ptr()),
      reinterpret_cast<const half*>(projection.data_ptr()),
      probabilities.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void deepseek_v41_swiglu_sm70_out(torch::Tensor out, torch::Tensor gate,
                                  torch::Tensor up) {
  TORCH_CHECK(gate.is_cuda() && gate.is_contiguous() && gate.dim() == 2 &&
                  gate.scalar_type() == torch::kFloat16 &&
                  up.device() == gate.device() && up.is_contiguous() &&
                  up.scalar_type() == torch::kFloat16 &&
                  up.sizes() == gate.sizes() && out.device() == gate.device() &&
                  out.is_contiguous() && out.scalar_type() == torch::kFloat16 &&
                  out.sizes() == gate.sizes(),
              "V4.1 shared SwiGLU requires matching contiguous FP16 matrices");
  const c10::cuda::CUDAGuard guard(gate.device());
  require_sm70();
  if (!gate.numel()) return;
  v41_shared_swiglu<<<(gate.numel() + 255) / 256, 256, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      reinterpret_cast<const half*>(gate.data_ptr()),
      reinterpret_cast<const half*>(up.data_ptr()), gate.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_dp4a_gate_up_sm70_out(torch::Tensor out, torch::Tensor activation,
                                torch::Tensor ids, torch::Tensor gate,
                                torch::Tensor up, int64_t source_type,
                                bool activated, int64_t lanes_per_row,
                                bool bank_aware) {
  TORCH_CHECK(source_type == 18 || source_type == 20 || source_type == 21 ||
                  source_type == 22 || source_type == 23 || source_type == 16 ||
                  source_type == 17 || source_type == 29,
              "Unsupported GGUF lattice dp4a reader");
  const int block_bytes = source_type == 16   ? 66
                          : source_type == 17 ? 74
                          : source_type == 29 ? 56
                          : source_type == 18 ? 98
                          : source_type == 21 ? 110
                          : source_type == 23 ? 136
                                              : 82;
  TORCH_CHECK(activation.is_cuda() &&
                  activation.scalar_type() == torch::kUInt8 &&
                  activation.dim() == 3 && activation.size(2) == sizeof(Q8_1) &&
                  activation.is_contiguous() && activation.size(0) > 0 &&
                  activation.size(0) <= 20,
              "Expected Q8_1 activation blocks");
  const int m = activation.size(0), k = activation.size(1) * 32;
  TORCH_CHECK(ids.device() == activation.device() &&
                  (ids.scalar_type() == torch::kInt32 ||
                   ids.scalar_type() == torch::kInt64) &&
                  ids.is_contiguous() && ids.dim() == 2 && ids.size(0) == m &&
                  ids.size(1) > 0 && ids.size(1) <= 16,
              "Invalid routing indices");
  TORCH_CHECK(
      gate.device() == activation.device() &&
          up.device() == activation.device() &&
          gate.scalar_type() == torch::kUInt8 &&
          up.scalar_type() == torch::kUInt8 && gate.is_contiguous() &&
          up.is_contiguous() && gate.dim() == 3 && gate.sizes() == up.sizes() &&
          gate.size(0) > 0 && gate.size(1) > 0 && k % 256 == 0 &&
          (source_type == 20
               ? gate.size(2) == k / 32 * 18
               : (gate.size(2) == k / 256 * block_bytes ||
                  gate.size(2) == ((k / 256 * block_bytes + 7) / 8 * 8))),
      "Expected aligned original GGUF lattice expert rows");
  const int n = gate.size(1), top_k = ids.size(1);
  const bool quantized = out.scalar_type() == torch::kUInt8;
  const bool v41 = source_type == 16 || source_type == 17 || source_type == 29;
  TORCH_CHECK(!v41 || (!activated && !quantized && !bank_aware && m > 0 &&
                       m <= 6 && k == 5120 && n == 576 && top_k == 6),
              "V4.1 gate/up requires unactivated FP16 M1..6, K5120 N576 top-6");
  // lanes_per_row 0: K2560 16-lane schedule with a fully unrolled group loop.
  TORCH_CHECK(lanes_per_row == 16 ||
                  (quantized && (lanes_per_row == 4 || lanes_per_row == 8 ||
                                 (lanes_per_row == 0 && k == 2560))),
              "Unsupported expert integer-dot row partition");
  TORCH_CHECK(
      out.device() == activation.device() && out.is_contiguous() &&
          (quantized ? activated && n % 32 == 0 && out.dim() == 4 &&
                           out.size(0) == m && out.size(1) == top_k &&
                           out.size(2) == n / 32 && out.size(3) == sizeof(Q8_1)
                     : out.scalar_type() == torch::kFloat16 &&
                           out.numel() ==
                               int64_t(m) * top_k * n * (activated ? 1 : 2)),
      "Invalid fused gate/up output");
  const c10::cuda::CUDAGuard guard(activation.device());
  require_sm70();
#define DISPATCH_LATTICE(TYPE)                                               \
  if (bank_aware)                                                            \
    dispatch_gate_up<TYPE, true>(out, activation, ids, gate, up, activated,  \
                                 lanes_per_row);                             \
  else                                                                       \
    dispatch_gate_up<TYPE, false>(out, activation, ids, gate, up, activated, \
                                  lanes_per_row)
  if (source_type == 16) {
    DISPATCH_LATTICE(16);
  } else if (source_type == 17) {
    DISPATCH_LATTICE(17);
  } else if (source_type == 29) {
    DISPATCH_LATTICE(29);
  } else if (source_type == 18) {
    DISPATCH_LATTICE(18);
  } else if (source_type == 20) {
    DISPATCH_LATTICE(20);
  } else if (source_type == 21) {
    DISPATCH_LATTICE(21);
  } else if (source_type == 23) {
    DISPATCH_LATTICE(23);
  } else {
    DISPATCH_LATTICE(22);
  }
#undef DISPATCH_LATTICE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
// Concurrent and speculative rows route many tokens to the same expert. The
// routes of each expert are taken `per_cta` at a time (in route order): the
// CTA of the first route of each batch decodes the expert's groups once and
// applies them to every route of its batch; the other CTAs exit. Fewer routes
// per CTA trade decode reuse for parallelism. Per-route arithmetic,
// reductions and the Q8_1 output are those of gate_up<Type, true, Index,
// Lanes, true> (IQ3_S: 16 lanes per row; IQ4_NL/IQ4_XS: 4).
constexpr int kGroupedChunk = 8;

template <int Type, class Index, bool Guarded = false>
__global__ void v41_gate_reuse(half* output, const Q8_1* activation,
                               const Index* ids, const uint8_t* gate,
                               const uint8_t* up, int stride,
                               const int* enabled) {
  if constexpr (Guarded)
    if (!*enabled) return;
  const Index expert = ids[blockIdx.y];
  // Every CTA of a repeated route exits before allocating/initializing its
  // codebook. The first occurrence computes all six tokens selecting it.
  for (int previous = 0; previous < blockIdx.y; ++previous)
    if (ids[previous] == expert) return;
  int members[6] = {};
#pragma unroll
  for (int token = 0; token < 6; ++token)
#pragma unroll
    for (int route = 0; route < 6; ++route)
      members[token] |= int(ids[token * 6 + route] == expert) << route;
  using Dot = LatticeDot<Type>;
  __shared__ uint32_t book[Dot::kBookWords], masks[16];
  Dot::initialize(book, masks);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int row = blockIdx.x * 8 + warp * 2 + lane / 16;
  const auto* g = gate + (int64_t(expert) * 576 + row) * stride;
  const auto* u = up + (int64_t(expert) * 576 + row) * stride;
  float gs[6] = {}, us[6] = {};
  for (int group = lane % 16; group < 160; group += 16) {
    const vllm::sm70_gguf::V41IntegerGroup<Type> dg(g, group, book, masks);
    const vllm::sm70_gguf::V41IntegerGroup<Type> du(u, group, book, masks);
#pragma unroll
    for (int token = 0; token < 6; ++token)
      if (members[token]) {
        const auto& x = activation[token * 160 + group];
        gs[token] = __fadd_rn(gs[token], dg.apply(x));
        us[token] = __fadd_rn(us[token], du.apply(x));
      }
  }
#pragma unroll
  for (int token = 0; token < 6; ++token) {
#pragma unroll
    for (int offset = 8; offset; offset >>= 1) {
      gs[token] += __shfl_down_sync(0xffffffffU, gs[token], offset, 16);
      us[token] += __shfl_down_sync(0xffffffffU, us[token], offset, 16);
    }
    if (lane % 16 == 0) {
#pragma unroll
      for (int route = 0; route < 6; ++route)
        if (members[token] & (1 << route)) {
          output[(int64_t(token) * 6 + route) * 1152 + row] =
              __float2half_rn(gs[token]);
          output[(int64_t(token) * 6 + route) * 1152 + 576 + row] =
              __float2half_rn(us[token]);
        }
    }
  }
}

template <int Type, class Index>
void launch_v41_gate_reuse(torch::Tensor out, torch::Tensor activation,
                           torch::Tensor ids, torch::Tensor gate,
                           torch::Tensor up, const int* enabled = nullptr) {
  if (enabled)
    v41_gate_reuse<Type, Index, true>
        <<<dim3(72, 36), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<half*>(out.data_ptr()),
            reinterpret_cast<const Q8_1*>(activation.data_ptr()),
            ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
            up.data_ptr<uint8_t>(), gate.size(2), enabled);
  else
    v41_gate_reuse<Type, Index>
        <<<dim3(72, 36), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<half*>(out.data_ptr()),
            reinterpret_cast<const Q8_1*>(activation.data_ptr()),
            ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
            up.data_ptr<uint8_t>(), gate.size(2), nullptr);
}

template <class Index>
__global__ void v41_reuse_plan(const Index* ids, int32_t* compact, int* enabled,
                               int threshold, int slots) {
  __shared__ int values[36], counts[2];
  const int i = threadIdx.x;
  if (i < 36) {
    const auto value = ids[i];
    if (value < 0 || value >= slots) asm volatile("trap;");
    compact[i] = values[i] = int(value);
  }
  __syncthreads();
  bool first = i < 36;
  if (first)
    for (int j = 0; j < i; ++j)
      if (values[j] == values[i]) first = false;
  const unsigned mask = __ballot_sync(0xffffffffU, first);
  if (i % 32 == 0) counts[i / 32] = __popc(mask);
  __syncthreads();
  if (!i) *enabled = counts[0] + counts[1] <= threshold;
}

template <int Type, class Index>
void launch_v41_gate_adaptive(torch::Tensor out, torch::Tensor activation,
                              torch::Tensor ids, torch::Tensor compact,
                              torch::Tensor mode, torch::Tensor gate,
                              torch::Tensor up) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  v41_reuse_plan<Index><<<1, 64, 0, stream>>>(
      ids.data_ptr<Index>(), compact.data_ptr<int32_t>(), mode.data_ptr<int>(),
      Type == 29 ? 32 : 24, gate.size(0));
  // Both branches are captured. The device decision executes exactly one;
  // the fallback retains the caller's original index width and dot order.
  gate_up<Type, false, Index, 16, false, false, false, false, 0, true>
      <<<dim3(72, 36), 128, 160 * sizeof(Q8_1), stream>>>(
          out.data_ptr(), reinterpret_cast<const Q8_1*>(activation.data_ptr()),
          ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
          up.data_ptr<uint8_t>(), 576, 5120, gate.size(2), 6, nullptr, nullptr,
          mode.data_ptr<int>());
  launch_v41_gate_reuse<Type, int32_t>(out, activation, compact, gate, up,
                                       mode.data_ptr<int>());
}

template <int Type, class Index, int Lanes, bool BankAware>
__global__ __launch_bounds__(32 * Lanes) void gate_up_grouped(
    Q8_1* output, const Q8_1* activation, const Index* ids, const uint8_t* gate,
    const uint8_t* up, int n, int k, int stride, int top_k, int routes,
    int per_cta) {
  using Dot = LatticeDot<Type, BankAware>;
  using Group =
      std::conditional_t<Type == 21, vllm::sm70_gguf::IQ3SGroup<BankAware>,
                         vllm::sm70_gguf::Lut4Group<Type>>;
  constexpr int Chunk = kGroupedChunk, Warps = Lanes;
  __shared__ uint32_t book[Dot::kBookWords];
  __shared__ uint32_t masks[16];
  __shared__ half intermediate[Chunk][32];
  __shared__ int members[512];
  __shared__ int member_count, rank;
  extern __shared__ Q8_1 shared_x[];
  const int leader = blockIdx.y, groups = k / 32;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const Index expert = ids[leader];
  // This route's rank among the expert's routes.
  int earlier = 0;
  for (int r = threadIdx.x; r < leader; r += blockDim.x)
    earlier += ids[r] == expert;
  if (threadIdx.x == 0) rank = 0;
  __syncthreads();
  if (earlier) atomicAdd(&rank, earlier);
  __syncthreads();
  if (rank % per_cta) return;
  if (warp == 0) {
    int count = 0;
    for (int base = leader; base < routes && count < per_cta; base += 32) {
      const int r = base + lane;
      const bool match = r < routes && ids[r] == expert;
      const unsigned ballot = __ballot_sync(0xffffffffU, match);
      const int index = count + __popc(ballot & ((1U << lane) - 1));
      if (match && index < per_cta) members[index] = r;
      count += __popc(ballot);
    }
    if (lane == 0) member_count = min(count, per_cta);
  }
  Dot::initialize(book, masks);
  const int count = member_count;
  const int local_row = warp * (32 / Lanes) + lane / Lanes;
  const int row = blockIdx.x * 32 + local_row;
  const int64_t expert_row = int64_t(expert) * n + row;
  const uint8_t* g = gate + expert_row * stride;
  const uint8_t* u = up + expert_row * stride;
  for (int first = 0; first < count; first += Chunk) {
    const int chunk = min(Chunk, count - first);
    for (int i = threadIdx.x; i < chunk * groups * 9; i += blockDim.x) {
      const int j = i / (groups * 9), w = i % (groups * 9);
      const Q8_1* x = activation + (members[first + j] / top_k) * groups;
      reinterpret_cast<uint32_t*>(shared_x)[i] =
          reinterpret_cast<const uint32_t*>(x)[w];
    }
    __syncthreads();
    float gs[Chunk], us[Chunk];
#pragma unroll
    for (int j = 0; j < Chunk; ++j) gs[j] = us[j] = 0.f;
    for (int group = lane % Lanes; group < groups; group += Lanes) {
      const Group dg(g, group, book, masks), du(u, group, book, masks);
#pragma unroll
      for (int j = 0; j < Chunk; ++j)
        if (j < chunk) {
          gs[j] += dg.apply(shared_x[j * groups + group]);
          us[j] += du.apply(shared_x[j * groups + group]);
        }
    }
#pragma unroll
    for (int j = 0; j < Chunk; ++j) {
#pragma unroll
      for (int offset = Lanes / 2; offset; offset >>= 1) {
        gs[j] += __shfl_down_sync(0xffffffffU, gs[j], offset, Lanes);
        us[j] += __shfl_down_sync(0xffffffffU, us[j], offset, Lanes);
      }
      if (j < chunk && lane % Lanes == 0) {
        const float g16 = __half2float(__float2half_rn(gs[j]));
        const float u16 = __half2float(__float2half_rn(us[j]));
        const half silu = __float2half_rn(g16 / (1.f + expf(-g16)));
        intermediate[j][local_row] = __hmul(silu, __float2half_rn(u16));
      }
    }
    __syncthreads();
    for (int j = warp; j < chunk; j += Warps)
      vllm::sm70_gguf::quantize_q8_1_warp(
          output + int64_t{members[first + j]} * (n / 32) + blockIdx.x,
          __half2float(intermediate[j][lane]));
    __syncthreads();
  }
}

template <int Type, class Index, int Lanes>
void launch_gate_up_grouped(torch::Tensor out, torch::Tensor activation,
                            torch::Tensor ids, torch::Tensor gate,
                            torch::Tensor up, bool bank_aware, int per_cta) {
  const int n = gate.size(1), k = activation.size(1) * 32;
  const int routes = ids.numel();
  const dim3 grid(n / 32, routes);
  const size_t shared = kGroupedChunk * activation.size(1) * sizeof(Q8_1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<Q8_1*>(out.data_ptr());
  const auto* x = reinterpret_cast<const Q8_1*>(activation.data_ptr());
  if (Type == 21 && bank_aware)
    gate_up_grouped<Type, Index, Lanes, true>
        <<<grid, 32 * Lanes, shared, stream>>>(
            o, x, ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
            up.data_ptr<uint8_t>(), n, k, gate.size(2), ids.size(1), routes,
            per_cta);
  else
    gate_up_grouped<Type, Index, Lanes, false>
        <<<grid, 32 * Lanes, shared, stream>>>(
            o, x, ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
            up.data_ptr<uint8_t>(), n, k, gate.size(2), ids.size(1), routes,
            per_cta);
}

template <class Index>
void dispatch_gate_up_grouped(torch::Tensor out, torch::Tensor activation,
                              torch::Tensor ids, torch::Tensor gate,
                              torch::Tensor up, int source_type,
                              bool bank_aware, int per_cta) {
  if (source_type == 21)
    launch_gate_up_grouped<21, Index, 16>(out, activation, ids, gate, up,
                                          bank_aware, per_cta);
  else if (source_type == 20)
    launch_gate_up_grouped<20, Index, 4>(out, activation, ids, gate, up,
                                         bank_aware, per_cta);
  else
    launch_gate_up_grouped<23, Index, 4>(out, activation, ids, gate, up,
                                         bank_aware, per_cta);
}
}  // namespace

static void v41_gate_reuse_out(torch::Tensor out, torch::Tensor activation,
                               torch::Tensor ids, torch::Tensor gate,
                               torch::Tensor up, int64_t source_type,
                               bool adaptive) {
  TORCH_CHECK(source_type == 16 || source_type == 17 || source_type == 29,
              "V4.1 integer reuse requires IQ1_M/IQ2 original blocks");
  const int bytes = source_type == 16 ? 66 : source_type == 17 ? 74 : 56;
  TORCH_CHECK(activation.is_cuda() && activation.is_contiguous() &&
                  activation.scalar_type() == torch::kUInt8 &&
                  activation.dim() == 3 && activation.size(0) == 6 &&
                  activation.size(1) == 160 &&
                  activation.size(2) == sizeof(Q8_1),
              "V4.1 integer reuse requires M6 K5120 Q8_1");
  TORCH_CHECK(ids.device() == activation.device() && ids.is_contiguous() &&
                  ids.dim() == 2 && ids.size(0) == 6 && ids.size(1) == 6 &&
                  (ids.scalar_type() == torch::kInt32 ||
                   ids.scalar_type() == torch::kInt64),
              "V4.1 integer reuse requires top-6 indices");
  TORCH_CHECK(gate.device() == activation.device() &&
                  gate.scalar_type() == torch::kUInt8 && gate.is_contiguous() &&
                  gate.dim() == 3 && gate.size(0) > 0 &&
                  gate.size(0) <= 65535 && gate.size(1) == 576 &&
                  (gate.size(2) == 20 * bytes ||
                   gate.size(2) == ((20 * bytes + 7) / 8 * 8)) &&
                  up.device() == gate.device() &&
                  up.scalar_type() == gate.scalar_type() &&
                  up.is_contiguous() && up.sizes() == gate.sizes(),
              "V4.1 integer reuse requires aligned raw expert carriers");
  TORCH_CHECK(out.device() == gate.device() && out.is_contiguous() &&
                  out.scalar_type() == torch::kFloat16 && out.dim() == 4 &&
                  out.size(0) == 6 && out.size(1) == 6 && out.size(2) == 2 &&
                  out.size(3) == 576,
              "V4.1 integer reuse requires FP16 M6 gate/up outputs");
  const c10::cuda::CUDAGuard guard(activation.device());
  require_sm70();
#define V41_REUSE(TYPE)                                                    \
  if (adaptive && ids.scalar_type() == torch::kInt32)                      \
    launch_v41_gate_adaptive<TYPE, int32_t>(out, activation, ids, compact, \
                                            mode, gate, up);               \
  else if (adaptive)                                                       \
    launch_v41_gate_adaptive<TYPE, int64_t>(out, activation, ids, compact, \
                                            mode, gate, up);               \
  else if (ids.scalar_type() == torch::kInt32)                             \
    launch_v41_gate_reuse<TYPE, int32_t>(out, activation, ids, gate, up);  \
  else                                                                     \
    launch_v41_gate_reuse<TYPE, int64_t>(out, activation, ids, gate, up)
  torch::Tensor compact, mode;
  if (adaptive) {
    compact = torch::empty({6, 6}, ids.options().dtype(torch::kInt32));
    mode = torch::empty({1}, ids.options().dtype(torch::kInt32));
  }
  if (source_type == 16) {
    V41_REUSE(16);
  } else if (source_type == 17) {
    V41_REUSE(17);
  } else {
    V41_REUSE(29);
  }
#undef V41_REUSE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_v41_dp4a_gate_reuse_sm70_out(torch::Tensor out,
                                       torch::Tensor activation,
                                       torch::Tensor ids, torch::Tensor gate,
                                       torch::Tensor up, int64_t source_type) {
  v41_gate_reuse_out(out, activation, ids, gate, up, source_type, false);
}

void gguf_v41_dp4a_gate_adaptive_sm70_out(torch::Tensor out,
                                          torch::Tensor activation,
                                          torch::Tensor ids, torch::Tensor gate,
                                          torch::Tensor up,
                                          int64_t source_type) {
  v41_gate_reuse_out(out, activation, ids, gate, up, source_type, true);
}

// Routed gate/up with SiLU and Q8_1 output, decoding each routed expert once
// per `per_cta` of its routes. Results equal gguf_dp4a_gate_up_sm70_out(...,
// activated=True, lanes_per_row=16 for IQ3_S or 4 for IQ4_NL/IQ4_XS) bit for
// bit.
void gguf_dp4a_gate_up_grouped_sm70_out(torch::Tensor out,
                                        torch::Tensor activation,
                                        torch::Tensor ids, torch::Tensor gate,
                                        torch::Tensor up, int64_t source_type,
                                        bool bank_aware, int64_t per_cta) {
  TORCH_CHECK(source_type == 20 || source_type == 21 || source_type == 23,
              "Grouped gate/up supports IQ4_NL, IQ3_S and IQ4_XS");
  TORCH_CHECK(per_cta >= 1 && per_cta <= 512, "Invalid routes per CTA");
  TORCH_CHECK(activation.is_cuda() &&
                  activation.scalar_type() == torch::kUInt8 &&
                  activation.dim() == 3 && activation.size(2) == sizeof(Q8_1) &&
                  activation.is_contiguous() && activation.size(0) > 0,
              "Expected Q8_1 activation blocks");
  const int m = activation.size(0), k = activation.size(1) * 32;
  TORCH_CHECK(ids.device() == activation.device() &&
                  (ids.scalar_type() == torch::kInt32 ||
                   ids.scalar_type() == torch::kInt64) &&
                  ids.is_contiguous() && ids.dim() == 2 && ids.size(0) == m &&
                  ids.size(1) > 0 && ids.numel() <= 512,
              "Grouped gate/up takes at most 512 routes");
  const int block_bytes = source_type == 21 ? 110 : 136;
  TORCH_CHECK(
      gate.device() == activation.device() &&
          up.device() == activation.device() &&
          gate.scalar_type() == torch::kUInt8 &&
          up.scalar_type() == torch::kUInt8 && gate.is_contiguous() &&
          up.is_contiguous() && gate.dim() == 3 && gate.sizes() == up.sizes() &&
          gate.size(1) % 32 == 0 && k % 256 == 0 &&
          (source_type == 20
               ? gate.size(2) == k / 32 * 18
               : (gate.size(2) == k / 256 * block_bytes ||
                  gate.size(2) == ((k / 256 * block_bytes + 7) / 8 * 8))),
      "Expected aligned original GGUF expert rows");
  const int n = gate.size(1);
  TORCH_CHECK(out.device() == activation.device() && out.is_contiguous() &&
                  out.scalar_type() == torch::kUInt8 && out.dim() == 4 &&
                  out.size(0) == m && out.size(1) == ids.size(1) &&
                  out.size(2) == n / 32 && out.size(3) == sizeof(Q8_1),
              "Invalid grouped gate/up output");
  const c10::cuda::CUDAGuard guard(activation.device());
  require_sm70();
  if (ids.scalar_type() == torch::kInt32)
    dispatch_gate_up_grouped<int32_t>(out, activation, ids, gate, up,
                                      source_type, bank_aware, per_cta);
  else
    dispatch_gate_up_grouped<int64_t>(out, activation, ids, gate, up,
                                      source_type, bank_aware, per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_dp4a_lut4_gate_up_sm70_out(
    torch::Tensor out, torch::Tensor activation, torch::Tensor ids,
    torch::Tensor gate, torch::Tensor gate_stats, torch::Tensor up,
    torch::Tensor up_stats, int64_t num_experts, int64_t lanes_per_row) {
  TORCH_CHECK(activation.is_cuda() && activation.is_contiguous() &&
                  activation.scalar_type() == torch::kUInt8 &&
                  activation.dim() == 3 && activation.size(0) > 0 &&
                  activation.size(0) <= 20 && activation.size(1) > 0 &&
                  activation.size(2) == sizeof(Q8_1),
              "Expected small Q8_1 activation blocks");
  const bool quantized = out.scalar_type() == torch::kUInt8;
  TORCH_CHECK(
      (quantized ? out.dim() == 4 && out.size(3) == sizeof(Q8_1)
                 : out.scalar_type() == torch::kFloat16 && out.dim() == 3) &&
          out.size(0) == activation.size(0) && out.size(2) > 0 &&
          ids.dim() == 2 && ids.size(0) == activation.size(0) &&
          ids.size(1) > 0 && ids.size(1) <= 16 &&
          (ids.scalar_type() == torch::kInt32 ||
           ids.scalar_type() == torch::kInt64) &&
          out.size(1) == ids.size(1),
      "Invalid canonical IQ4 gate/up output or routes");
  const int n = out.size(2) * (quantized ? 32 : 1);
  TORCH_CHECK(n <= 256 && n % 32 == 0 && num_experts > 0 &&
                  num_experts <= 65535 &&
                  (lanes_per_row == 16 ||
                   (quantized && (lanes_per_row == 4 || lanes_per_row == 8))),
              "Unsupported canonical IQ4 gate/up row partition");
  for (const auto& t : {out, ids, gate, gate_stats, up, up_stats})
    TORCH_CHECK(t.device() == activation.device() && t.is_contiguous(),
                "Canonical IQ4 descriptors must share the activation device");
  for (const auto& t : {gate, gate_stats, up, up_stats})
    TORCH_CHECK(t.scalar_type() == torch::kUInt8 &&
                    t.numel() == num_experts * sizeof(StridedPtr),
                "Invalid canonical IQ4 expert descriptor");
  const c10::cuda::CUDAGuard guard(activation.device());
  require_sm70();
  if (ids.scalar_type() == torch::kInt32)
    dispatch_lut4_gate_up<int32_t>(out, activation, ids, gate, gate_stats, up,
                                   up_stats, n, lanes_per_row);
  else
    dispatch_lut4_gate_up<int64_t>(out, activation, ids, gate, gate_stats, up,
                                   up_stats, n, lanes_per_row);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_dp4a_down_unroute_sm70_out(torch::Tensor out, torch::Tensor input,
                                     torch::Tensor ids,
                                     torch::Tensor route_weights,
                                     torch::Tensor weight_ptrs,
                                     torch::Tensor stats_ptrs,
                                     int64_t source_type, int64_t num_experts) {
  using turbomind::gemm::StridedPtr;
  TORCH_CHECK(source_type == 20 || source_type == 42,
              "dp4a down requires IQ4_NL or Q2_0 canonical integers");
  const bool quantized = input.scalar_type() == torch::kUInt8;
  TORCH_CHECK(
      input.is_cuda() && input.is_contiguous() &&
          (quantized
               ? input.dim() == 4 && input.size(3) == sizeof(Q8_1)
               : input.scalar_type() == torch::kFloat16 && input.dim() == 3) &&
          input.size(0) > 0 && input.size(0) <= 20 && input.size(1) > 0 &&
          input.size(1) <= 16 && input.size(2) > 0 &&
          (quantized ? input.size(2) <= 8
                     : input.size(2) <= 256 && input.size(2) % 32 == 0),
      "Expected FP16 or Q8_1 routed small intermediate");
  for (const auto& t : {out, ids, route_weights, weight_ptrs, stats_ptrs})
    TORCH_CHECK(t.device() == input.device() && t.is_contiguous(),
                "dp4a down tensors must share a CUDA device");
  TORCH_CHECK(
      out.scalar_type() == torch::kFloat16 && out.dim() == 2 &&
          out.size(0) == input.size(0) && out.size(1) > 0 &&
          out.size(1) % 32 == 0 && ids.dim() == 2 &&
          (ids.scalar_type() == torch::kInt32 ||
           ids.scalar_type() == torch::kInt64) &&
          ids.size(0) == input.size(0) && ids.size(1) == input.size(1) &&
          route_weights.scalar_type() == torch::kFloat32 &&
          route_weights.sizes() == ids.sizes() && num_experts > 0 &&
          num_experts <= 65535 && weight_ptrs.scalar_type() == torch::kUInt8 &&
          stats_ptrs.scalar_type() == torch::kUInt8 &&
          weight_ptrs.numel() == num_experts * sizeof(StridedPtr) &&
          stats_ptrs.numel() == num_experts * sizeof(StridedPtr),
      "Invalid dp4a down descriptor or routing shape");
  const c10::cuda::CUDAGuard guard(input.device());
  require_sm70();
  if (source_type == 20)
    dispatch_down<20>(out, input, ids, route_weights, weight_ptrs, stats_ptrs);
  else
    dispatch_down<42>(out, input, ids, route_weights, weight_ptrs, stats_ptrs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
template <int Type, class Index, bool Quantized, int Lanes = 1, int Tiles = 1>
__global__ void raw_down_unroute(half* output, const void* input,
                                 const Index* ids, const float* route_weights,
                                 const uint8_t* weights, int n, int k,
                                 int row_stride, int left_groups, int top_k) {
  extern __shared__ Q8_1 x[];
  constexpr bool lattice = Type == 16 || Type == 17 || Type == 29;
  constexpr int book_words = [] {
    if constexpr (lattice)
      return LatticeDot<Type>::kBookWords;
    else
      return 1;
  }();
  __shared__ uint32_t book[book_words], masks[lattice ? 16 : 1];
  __shared__ float partial[4][32 / Lanes];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int groups = lattice ? 18 : k / 32, token = blockIdx.y;
  if constexpr (Quantized) {
    const auto* source = static_cast<const uint32_t*>(input) +
                         int64_t{token} * top_k * groups * 9;
    for (int i = threadIdx.x; i < top_k * groups * 9; i += blockDim.x)
      reinterpret_cast<uint32_t*>(x)[i] = source[i];
  } else {
    for (int i = warp; i < top_k * groups; i += 4) {
      const half value = static_cast<const half*>(
          input)[int64_t{token} * top_k * k + i * 32 + lane];
      vllm::sm70_gguf::quantize_q8_1_warp(x + i, __half2float(value));
    }
  }
  if constexpr (lattice)
    LatticeDot<Type>::initialize(book, masks);
  else
    __syncthreads();
  // Reuse the activation and integer codebook across adjacent output tiles.
  // Each tile retains exactly the original K-group and route additions.
  for (int tile = 0; tile < Tiles; ++tile) {
    const int col = (blockIdx.x * Tiles + tile) * (32 / Lanes) + lane / Lanes;
    float total = 0.f;
    for (int route = warp; route < top_k; route += 4) {
      const int slot = token * top_k + route;
      const uint8_t* row =
          weights + (int64_t{ids[slot]} * n + col) * row_stride;
      float dot = 0.f;
      if constexpr (Lanes > 1) {
        static_assert(lattice && Quantized);
        // Neighboring lanes fetch neighboring K groups of the same packed
        // row. Shuffles then restore the original ascending FP32 additions;
        // a parallel reduction would change the numerical boundary.
        for (int first = 0; first < groups; first += Lanes) {
          const int group = first + lane % Lanes;
          const float value =
              group < groups ? LatticeDot<Type>::dot(row, group + left_groups,
                                                     x[route * groups + group],
                                                     book, masks)
                             : 0.f;
#pragma unroll
          for (int i = 0; i < Lanes; ++i) {
            const float ordered =
                __shfl_sync(0xffffffff, value, (lane / Lanes) * Lanes + i);
            if (first + i < groups) dot = __fadd_rn(dot, ordered);
          }
        }
      } else {
        for (int group = 0; group < groups; ++group) {
          if constexpr (lattice)
            dot = __fadd_rn(dot, LatticeDot<Type>::dot(
                                     row, group + left_groups,
                                     x[route * groups + group], book, masks));
          else
            dot += vllm::sm70_gguf::OriginalIntegerDot<Type>::dot(
                row, group + left_groups, x[route * groups + group]);
        }
      }
      total += __half2float(__float2half_rn(dot)) * route_weights[slot];
    }
    if (lane % Lanes == 0) partial[warp][lane / Lanes] = total;
    __syncthreads();
    if (!warp && lane % Lanes == 0) {
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) sum += partial[i][lane / Lanes];
      output[int64_t{token} * n + col] = __float2half_rn(sum);
    }
    if constexpr (Tiles > 1) __syncthreads();
  }
}

template <int Type, class Index, bool Quantized, int Tiles = 1>
void launch_raw_down(torch::Tensor out, torch::Tensor input, torch::Tensor ids,
                     torch::Tensor probabilities, torch::Tensor weights,
                     int left) {
  const int k = input.size(2) * (Quantized ? 32 : 1);
  constexpr int lanes =
      Quantized && (Type == 16 || Type == 17 || Type == 29) ? 4 : 1;
  raw_down_unroute<Type, Index, Quantized, lanes, Tiles>
      <<<dim3(out.size(1) / (32 / lanes) / Tiles, out.size(0)), 128,
         ids.size(1) * (k / 32) * sizeof(Q8_1),
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<half*>(out.data_ptr()), input.data_ptr(),
          ids.data_ptr<Index>(), probabilities.data_ptr<float>(),
          weights.data_ptr<uint8_t>(), out.size(1), k, weights.size(2),
          left / 32, ids.size(1));
}

template <int Type, class Index>
void dispatch_raw_down(torch::Tensor out, torch::Tensor input,
                       torch::Tensor ids, torch::Tensor probabilities,
                       torch::Tensor weights, int left, int tiles) {
  if (input.scalar_type() == torch::kUInt8) {
    if constexpr (Type == 16 || Type == 17 || Type == 29) {
      if (tiles == 2)
        launch_raw_down<Type, Index, true, 2>(out, input, ids, probabilities,
                                              weights, left);
      else if (tiles == 4)
        launch_raw_down<Type, Index, true, 4>(out, input, ids, probabilities,
                                              weights, left);
      else if (tiles == 8)
        launch_raw_down<Type, Index, true, 8>(out, input, ids, probabilities,
                                              weights, left);
      else
        launch_raw_down<Type, Index, true>(out, input, ids, probabilities,
                                           weights, left);
    } else {
      launch_raw_down<Type, Index, true>(out, input, ids, probabilities,
                                         weights, left);
    }
  } else
    launch_raw_down<Type, Index, false>(out, input, ids, probabilities, weights,
                                        left);
}
}  // namespace

static void raw_down_out(torch::Tensor out, torch::Tensor input,
                         torch::Tensor ids, torch::Tensor probabilities,
                         torch::Tensor weights, int64_t source_type,
                         int64_t left, int tiles) {
  const bool v41 = source_type == 16 || source_type == 17 || source_type == 29;
  TORCH_CHECK(source_type == 20 || source_type == 42 || v41,
              "Unsupported original integer down source");
  const bool quantized = input.scalar_type() == torch::kUInt8;
  TORCH_CHECK(tiles == 1 || (v41 && quantized &&
                             (tiles == 2 || tiles == 4 || tiles == 8)),
              "Tiled original down requires V4.1 Q8_1 input");
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() &&
                  (quantized ? input.dim() == 4 && input.size(3) == sizeof(Q8_1)
                             : input.dim() == 3 &&
                                   input.scalar_type() == torch::kFloat16),
              "Expected routed FP16 or Q8_1 input");
  const int k = input.size(2) * (quantized ? 32 : 1);
  const int block = v41 ? 256 : source_type == 20 ? 32 : 64;
  const int block_bytes = source_type == 16   ? 66
                          : source_type == 17 ? 74
                          : source_type == 29 ? 56
                                              : 18;
  TORCH_CHECK(input.size(0) > 0 && input.size(0) <= 20 && input.size(1) > 0 &&
                  input.size(1) <= 16 && k > 0 && (v41 ? k == 576 : k <= 256) &&
                  k % 32 == 0 && left >= 0 && left < block && left % 32 == 0,
              "Unsupported original down geometry");
  TORCH_CHECK(!v41 || ((input.size(0) > 0 && input.size(0) <= 6) &&
                       input.size(1) == 6 && out.dim() == 2 &&
                       out.size(1) == 5120 && left % 64 == 0),
              "V4.1 down requires FP16 or Q8_1 M1..6 top-6");
  for (const auto& tensor : {out, ids, probabilities, weights})
    TORCH_CHECK(tensor.device() == input.device() && tensor.is_contiguous(),
                "Original down tensors must share a CUDA device");
  TORCH_CHECK(
      out.scalar_type() == torch::kFloat16 && out.dim() == 2 &&
          out.size(0) == input.size(0) && out.size(1) > 0 &&
          out.size(1) % 32 == 0 && ids.dim() == 2 &&
          ids.size(0) == input.size(0) && ids.size(1) == input.size(1) &&
          (ids.scalar_type() == torch::kInt32 ||
           ids.scalar_type() == torch::kInt64) &&
          probabilities.scalar_type() == torch::kFloat32 &&
          probabilities.sizes() == ids.sizes() &&
          weights.scalar_type() == torch::kUInt8 && weights.dim() == 3 &&
          weights.size(0) > 0 && weights.size(0) <= 65535 &&
          weights.size(1) == out.size(1) &&
          weights.size(2) == ((left + k + block - 1) / block) * block_bytes,
      "Invalid original down storage or routing");
  const c10::cuda::CUDAGuard guard(input.device());
  require_sm70();
#define DISPATCH_RAW(TYPE)                                                    \
  if (ids.scalar_type() == torch::kInt32)                                     \
    dispatch_raw_down<TYPE, int32_t>(out, input, ids, probabilities, weights, \
                                     left, tiles);                            \
  else                                                                        \
    dispatch_raw_down<TYPE, int64_t>(out, input, ids, probabilities, weights, \
                                     left, tiles)
  if (source_type == 16) {
    DISPATCH_RAW(16);
  } else if (source_type == 17) {
    DISPATCH_RAW(17);
  } else if (source_type == 29) {
    DISPATCH_RAW(29);
  } else if (source_type == 20) {
    DISPATCH_RAW(20);
  } else {
    DISPATCH_RAW(42);
  }
#undef DISPATCH_RAW
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gguf_dp4a_raw_down_unroute_sm70_out(torch::Tensor out, torch::Tensor input,
                                         torch::Tensor ids,
                                         torch::Tensor probabilities,
                                         torch::Tensor weights,
                                         int64_t source_type, int64_t left) {
  raw_down_out(out, input, ids, probabilities, weights, source_type, left, 1);
}

void gguf_v41_dp4a_q8_down_tiled_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor ids,
    torch::Tensor probabilities, torch::Tensor weights, int64_t source_type,
    int64_t left, int64_t tiles) {
  TORCH_CHECK(
      (source_type == 16 || source_type == 17 || source_type == 29) &&
          input.scalar_type() == torch::kUInt8 &&
          (tiles == 0 || tiles == 1 || tiles == 2 || tiles == 4 || tiles == 8),
      "V4.1 tiled down requires original lattice rows and Q8_1 input");
  raw_down_out(
      out, input, ids, probabilities, weights, source_type, left,
      tiles ? tiles : (source_type == 29 ? (input.size(0) == 1 ? 2 : 4) : 1));
}

void gguf_dp4a_scalar_lut_gate_up_sm70_out(
    torch::Tensor out, torch::Tensor activation, torch::Tensor ids,
    torch::Tensor gate, torch::Tensor up, int64_t source_type, bool activated) {
  TORCH_CHECK(source_type == 18 || source_type == 21 || source_type == 22,
              "Unsupported signed-nibble source type");
  TORCH_CHECK(activation.is_cuda() && activation.is_contiguous() &&
                  activation.scalar_type() == torch::kUInt8 &&
                  activation.dim() == 3 && activation.size(0) > 0 &&
                  activation.size(0) <= 20 && activation.size(1) > 0 &&
                  activation.size(2) == sizeof(Q8_1),
              "Expected Q8_1 activation blocks");
  const int m = activation.size(0), k = activation.size(1) * 32;
  TORCH_CHECK(ids.device() == activation.device() && ids.is_contiguous() &&
                  ids.dim() == 2 && ids.size(0) == m && ids.size(1) > 0 &&
                  ids.size(1) <= 16 &&
                  (ids.scalar_type() == torch::kInt32 ||
                   ids.scalar_type() == torch::kInt64),
              "Invalid routing indices");
  TORCH_CHECK(gate.device() == activation.device() &&
                  up.device() == activation.device() && gate.is_contiguous() &&
                  up.is_contiguous() && gate.scalar_type() == torch::kUInt8 &&
                  up.scalar_type() == torch::kUInt8 && gate.dim() == 3 &&
                  gate.sizes() == up.sizes() && gate.size(0) > 0 &&
                  gate.size(1) > 0 && k % 256 == 0 &&
                  gate.size(2) == (k / 32) * 20,
              "Expected signed-nibble expert rows");
  const int n = gate.size(1), top_k = ids.size(1);
  const bool quantized = out.scalar_type() == torch::kUInt8;
  TORCH_CHECK(
      out.device() == activation.device() && out.is_contiguous() &&
          (quantized ? activated && n % 32 == 0 && out.dim() == 4 &&
                           out.size(0) == m && out.size(1) == top_k &&
                           out.size(2) == n / 32 && out.size(3) == sizeof(Q8_1)
                     : out.scalar_type() == torch::kFloat16 &&
                           out.numel() ==
                               int64_t(m) * top_k * n * (activated ? 1 : 2)),
      "Invalid signed-nibble gate/up output");
  const c10::cuda::CUDAGuard guard(activation.device());
  require_sm70();
  if (source_type == 18)
    dispatch_scalar_lut_gate_up<18>(out, activation, ids, gate, up, activated);
  else if (source_type == 21)
    dispatch_scalar_lut_gate_up<21>(out, activation, ids, gate, up, activated);
  else
    dispatch_scalar_lut_gate_up<22>(out, activation, ids, gate, up, activated);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
