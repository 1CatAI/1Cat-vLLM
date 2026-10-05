// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "gguf_dp4a.cuh"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"

namespace {
using vllm::sm70_gguf::LatticeDot;
using vllm::sm70_gguf::Q8_1;

__global__ void quantize_q8(Q8_1* out, const half* input, int k) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= k) return;
  const float value = __half2float(input[int64_t(blockIdx.y) * k + column]);
  const int group = blockIdx.y * (k / 32) + column / 32;
  vllm::sm70_gguf::quantize_q8_1_warp(out + group, value);
}

template <int Type, bool Activated, class Index>
__global__ void gate_up(half* output, const Q8_1* activation, const Index* ids,
                        const uint8_t* gate, const uint8_t* up, int n, int k,
                        int stride, int top_k) {
  using Dot = LatticeDot<Type>;
  __shared__ uint32_t book[Dot::kBookWords];
  __shared__ uint32_t masks[16];
  extern __shared__ Q8_1 shared_x[];
  const int groups = k / 32, route = blockIdx.y;
  const Q8_1* x = activation + (route / top_k) * groups;
  for (int i = threadIdx.x; i < groups * 9; i += blockDim.x)
    reinterpret_cast<uint32_t*>(shared_x)[i] =
        reinterpret_cast<const uint32_t*>(x)[i];
  Dot::initialize(book, masks);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  // Two independent rows per warp keep all lanes busy at K2560 (80 groups).
  const int row = blockIdx.x * 8 + warp * 2 + lane / 16;
  if (row >= n) return;
  const int64_t expert_row = ids[route] * n + row;
  const uint8_t* g = gate + expert_row * stride;
  const uint8_t* u = up + expert_row * stride;
  float gs = 0.f, us = 0.f;
  for (int group = lane % 16; group < groups; group += 16) {
    gs += Dot::dot(g, group, shared_x[group], book, masks);
    us += Dot::dot(u, group, shared_x[group], book, masks);
  }
#pragma unroll
  for (int offset = 8; offset; offset >>= 1) {
    gs += __shfl_down_sync(__activemask(), gs, offset, 16);
    us += __shfl_down_sync(__activemask(), us, offset, 16);
  }
  if (lane % 16 == 0) {
    if constexpr (Activated) {
      // Keep the retained FP16 gate/up boundary before SiLU and multiply.
      const float g16 = __half2float(__float2half_rn(gs));
      const float u16 = __half2float(__float2half_rn(us));
      const half silu = __float2half_rn(g16 / (1.f + expf(-g16)));
      output[int64_t(route) * n + row] = __hmul(silu, __float2half_rn(u16));
    } else {
      output[int64_t(route) * 2 * n + row] = __float2half_rn(gs);
      output[int64_t(route) * 2 * n + n + row] = __float2half_rn(us);
    }
  }
}

template <int Type, class Index>
void launch_gate_up(torch::Tensor out, torch::Tensor activation,
                    torch::Tensor ids, torch::Tensor gate, torch::Tensor up,
                    bool activated) {
  const int n = gate.size(1), top_k = ids.size(1);
  const dim3 grid((n + 7) / 8, activation.size(0) * top_k);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared = activation.size(1) * sizeof(Q8_1);
  const auto output = reinterpret_cast<half*>(out.data_ptr());
  const auto x = reinterpret_cast<const Q8_1*>(activation.data_ptr());
  if (activated)
    gate_up<Type, true, Index><<<grid, 128, shared, stream>>>(
        output, x, ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
        up.data_ptr<uint8_t>(), n, activation.size(1) * 32, gate.size(2),
        top_k);
  else
    gate_up<Type, false, Index><<<grid, 128, shared, stream>>>(
        output, x, ids.data_ptr<Index>(), gate.data_ptr<uint8_t>(),
        up.data_ptr<uint8_t>(), n, activation.size(1) * 32, gate.size(2),
        top_k);
}

template <int Type>
void dispatch_gate_up(torch::Tensor out, torch::Tensor activation,
                      torch::Tensor ids, torch::Tensor gate, torch::Tensor up,
                      bool activated) {
  if (ids.scalar_type() == torch::kInt32)
    launch_gate_up<Type, int32_t>(out, activation, ids, gate, up, activated);
  else
    launch_gate_up<Type, int64_t>(out, activation, ids, gate, up, activated);
}

// Quantize routed intermediate rows in shared memory, compute down directly
// from expert descriptors, and reduce route weights without sorting/gathering.
template <int Type, class Index>
__global__ void down_unroute(half* output, const half* input, const Index* ids,
                             const float* route_weights,
                             const turbomind::gemm::StridedPtr* weights,
                             const turbomind::gemm::StridedPtr* stats, int n,
                             int k, int top_k) {
  extern __shared__ Q8_1 x[];
  __shared__ float partial[4][32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int groups = k / 32, token = blockIdx.y;
  for (int i = warp; i < top_k * groups; i += 4) {
    const half value = input[int64_t{token} * top_k * k + i * 32 + lane];
    vllm::sm70_gguf::quantize_q8_1_warp(x + i, __half2float(value));
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

template <int Type, class Index>
void launch_down(torch::Tensor out, torch::Tensor input, torch::Tensor ids,
                 torch::Tensor route_weights, torch::Tensor weights,
                 torch::Tensor stats) {
  using turbomind::gemm::StridedPtr;
  const int top_k = ids.size(1), k = input.size(2), n = out.size(1);
  down_unroute<Type, Index>
      <<<dim3(n / 32, out.size(0)), 128, top_k*(k / 32) * sizeof(Q8_1),
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<half*>(out.data_ptr()),
          reinterpret_cast<const half*>(input.data_ptr()),
          ids.data_ptr<Index>(), route_weights.data_ptr<float>(),
          reinterpret_cast<const StridedPtr*>(weights.data_ptr()),
          reinterpret_cast<const StridedPtr*>(stats.data_ptr()), n, k, top_k);
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
                  input.scalar_type() == torch::kFloat16 &&
                  input.size(0) <= 20 && input.size(1) > 0 &&
                  input.size(1) % 256 == 0 && out.device() == input.device() &&
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

void gguf_dp4a_gate_up_sm70_out(torch::Tensor out, torch::Tensor activation,
                                torch::Tensor ids, torch::Tensor gate,
                                torch::Tensor up, int64_t source_type,
                                bool activated) {
  TORCH_CHECK(source_type == 18 || source_type == 21 || source_type == 22,
              "Unsupported GGUF lattice dp4a reader");
  const int block_bytes = source_type == 18 ? 98 : source_type == 21 ? 110 : 82;
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
  TORCH_CHECK(gate.device() == activation.device() &&
                  up.device() == activation.device() &&
                  gate.scalar_type() == torch::kUInt8 &&
                  up.scalar_type() == torch::kUInt8 && gate.is_contiguous() &&
                  up.is_contiguous() && gate.dim() == 3 &&
                  gate.sizes() == up.sizes() && gate.size(0) > 0 &&
                  gate.size(1) > 0 && k % 256 == 0 &&
                  gate.size(2) == ((k / 256 * block_bytes + 7) / 8 * 8),
              "Expected aligned original GGUF lattice expert rows");
  const int n = gate.size(1), top_k = ids.size(1);
  TORCH_CHECK(out.device() == activation.device() &&
                  out.scalar_type() == torch::kFloat16 && out.is_contiguous() &&
                  out.numel() == int64_t(m) * top_k * n * (activated ? 1 : 2),
              "Invalid fused gate/up output");
  const c10::cuda::CUDAGuard guard(activation.device());
  require_sm70();
  if (source_type == 18)
    dispatch_gate_up<18>(out, activation, ids, gate, up, activated);
  else if (source_type == 21)
    dispatch_gate_up<21>(out, activation, ids, gate, up, activated);
  else
    dispatch_gate_up<22>(out, activation, ids, gate, up, activated);
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
  TORCH_CHECK(
      input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
          input.is_contiguous() && input.dim() == 3 && input.size(0) > 0 &&
          input.size(0) <= 20 && input.size(1) > 0 && input.size(1) <= 16 &&
          input.size(2) > 0 && input.size(2) <= 256 && input.size(2) % 32 == 0,
      "Expected FP16 [M,top_k,K] with small aligned intermediate");
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
