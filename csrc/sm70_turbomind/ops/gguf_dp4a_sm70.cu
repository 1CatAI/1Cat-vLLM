// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "gguf_dp4a.cuh"

namespace {
using vllm::sm70_gguf::IQ3SDot;
using vllm::sm70_gguf::Q8_1;

__global__ void quantize_q8(Q8_1* out, const half* input, int k) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= k) return;
  const float value = __half2float(input[int64_t(blockIdx.y) * k + column]);
  float maximum = fabsf(value), sum = value;
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  const float d = maximum / 127.f;
  const int group = blockIdx.y * (k / 32) + column / 32;
  out[group].qs[column % 32] = maximum == 0.f ? 0 : int8_t(roundf(value / d));
  if (column % 32 == 0) out[group].ds = __floats2half2_rn(d, sum);
}

template <bool Activated>
__global__ void gate_up(half* output, const Q8_1* activation,
                        const int64_t* ids, const uint8_t* gate,
                        const uint8_t* up, int n, int k, int stride,
                        int top_k) {
  __shared__ uint32_t book[IQ3SDot::kBookWords];
  __shared__ uint32_t masks[16];
  extern __shared__ Q8_1 shared_x[];
  const int groups = k / 32, route = blockIdx.y;
  const Q8_1* x = activation + (route / top_k) * groups;
  for (int i = threadIdx.x; i < groups * 9; i += blockDim.x)
    reinterpret_cast<uint32_t*>(shared_x)[i] =
        reinterpret_cast<const uint32_t*>(x)[i];
  IQ3SDot::initialize(book, masks);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int row = blockIdx.x * 4 + warp;
  if (row >= n) return;
  const int64_t expert_row = ids[route] * n + row;
  const uint8_t* g = gate + expert_row * stride;
  const uint8_t* u = up + expert_row * stride;
  float gs = 0.f, us = 0.f;
  for (int group = lane; group < groups; group += 32) {
    gs += IQ3SDot::dot(g, group, shared_x[group], book, masks);
    us += IQ3SDot::dot(u, group, shared_x[group], book, masks);
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    gs += __shfl_down_sync(0xffffffff, gs, offset);
    us += __shfl_down_sync(0xffffffff, us, offset);
  }
  if (!lane) {
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
  TORCH_CHECK(source_type == 21, "IQ3_S is the first calibrated dp4a reader");
  TORCH_CHECK(activation.is_cuda() &&
                  activation.scalar_type() == torch::kUInt8 &&
                  activation.dim() == 3 && activation.size(2) == sizeof(Q8_1) &&
                  activation.is_contiguous() && activation.size(0) > 0 &&
                  activation.size(0) <= 20,
              "Expected Q8_1 activation blocks");
  const int m = activation.size(0), k = activation.size(1) * 32;
  TORCH_CHECK(ids.device() == activation.device() &&
                  ids.scalar_type() == torch::kInt64 && ids.is_contiguous() &&
                  ids.dim() == 2 && ids.size(0) == m && ids.size(1) > 0 &&
                  ids.size(1) <= 16,
              "Invalid routing indices");
  TORCH_CHECK(gate.device() == activation.device() &&
                  up.device() == activation.device() &&
                  gate.scalar_type() == torch::kUInt8 &&
                  up.scalar_type() == torch::kUInt8 && gate.is_contiguous() &&
                  up.is_contiguous() && gate.dim() == 3 &&
                  gate.sizes() == up.sizes() && gate.size(0) > 0 &&
                  gate.size(1) > 0 && k % 256 == 0 &&
                  gate.size(2) == ((k / 256 * 110 + 7) / 8 * 8),
              "Expected aligned original IQ3_S expert rows");
  const int n = gate.size(1), top_k = ids.size(1);
  TORCH_CHECK(out.device() == activation.device() &&
                  out.scalar_type() == torch::kFloat16 && out.is_contiguous() &&
                  out.numel() == int64_t(m) * top_k * n * (activated ? 1 : 2),
              "Invalid fused gate/up output");
  const c10::cuda::CUDAGuard guard(activation.device());
  require_sm70();
  const dim3 grid((n + 3) / 4, m * top_k);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared = activation.size(1) * sizeof(Q8_1);
  if (activated)
    gate_up<true><<<grid, 128, shared, stream>>>(
        reinterpret_cast<half*>(out.data_ptr()),
        reinterpret_cast<const Q8_1*>(activation.data_ptr()),
        ids.data_ptr<int64_t>(), gate.data_ptr<uint8_t>(),
        up.data_ptr<uint8_t>(), n, k, gate.size(2), top_k);
  else
    gate_up<false><<<grid, 128, shared, stream>>>(
        reinterpret_cast<half*>(out.data_ptr()),
        reinterpret_cast<const Q8_1*>(activation.data_ptr()),
        ids.data_ptr<int64_t>(), gate.data_ptr<uint8_t>(),
        up.data_ptr<uint8_t>(), n, k, gate.size(2), top_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
