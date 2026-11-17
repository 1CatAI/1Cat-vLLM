// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>

#include "gguf_q8_1.cuh"

namespace {

using vllm::sm70_gguf::Q8_1;

constexpr int kBlockSize = 32;
constexpr int kQ80Bytes = 34;
constexpr int kMaxTopK = 16;

__device__ __forceinline__ uint32_t load_u32_2(const uint8_t* p) {
  return uint32_t(*reinterpret_cast<const uint16_t*>(p)) |
         (uint32_t(*reinterpret_cast<const uint16_t*>(p + 2)) << 16);
}

__global__ __launch_bounds__(256) void mtp_quantize_q8_1_kernel(
    Q8_1* __restrict__ out, const half* __restrict__ x, int hidden_groups) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group = blockIdx.x * 8 + warp;
  if (group >= hidden_groups) return;
  const int token = blockIdx.y;
  const float val = __half2float(
      x[(static_cast<size_t>(token) * hidden_groups + group) * kBlockSize +
        lane]);
  vllm::sm70_gguf::quantize_q8_1_warp(
      out + static_cast<size_t>(token) * hidden_groups + group, val);
}

template <int Warps, int MinBlocks = 0>
__global__
__launch_bounds__(Warps * 32, MinBlocks) void mtp_q8_0_w13_swiglu_q8_1_kernel(
    Q8_1* __restrict__ inter_out, const Q8_1* __restrict__ x_q8,
    const uint8_t* __restrict__ w13, const int32_t* __restrict__ topk_ids,
    int num_experts, int inter, int hidden_groups, int top_k) {
  static_assert(Warps == 8 || Warps == 16);
  constexpr int kThreads = Warps * 32;
  constexpr int kRowsPerWarp = 32 / Warps;
  constexpr int kLanesPerRow = 32 / kRowsPerWarp;
  constexpr int kGroupStride = kLanesPerRow / 8;

  extern __shared__ Q8_1 smem_x[];
  __shared__ half smem_act[32];

  const int route = blockIdx.y;
  const int inter_groups = inter / kBlockSize;
  const int expert = topk_ids[route];
  if (expert < 0 || expert >= num_experts) {
    uint32_t* dst = reinterpret_cast<uint32_t*>(
        inter_out + static_cast<size_t>(route) * inter_groups + blockIdx.x);
    if (threadIdx.x < 9) dst[threadIdx.x] = 0;
    return;
  }

  const int token = route / top_k;
  const uint32_t* src_x = reinterpret_cast<const uint32_t*>(
      x_q8 + static_cast<size_t>(token) * hidden_groups);
  uint32_t* dst_x = reinterpret_cast<uint32_t*>(smem_x);
  for (int i = threadIdx.x; i < hidden_groups * 9; i += kThreads) {
    dst_x[i] = src_x[i];
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_in_warp = lane / kLanesPerRow;
  const int lane_in_row = lane % kLanesPerRow;
  const int grp_off = lane_in_row >> 3;
  const int sub = lane_in_row & 7;
  const int local_row = warp * kRowsPerWarp + row_in_warp;
  const int row = blockIdx.x * 32 + local_row;

  const size_t row_bytes = static_cast<size_t>(hidden_groups) * kQ80Bytes;
  const uint8_t* expert_base =
      w13 + static_cast<size_t>(expert) * (2 * inter) * row_bytes;
  const uint8_t* g_row = expert_base + static_cast<size_t>(row) * row_bytes;
  const uint8_t* u_row =
      expert_base + static_cast<size_t>(inter + row) * row_bytes;

  float acc_g = 0.0f;
  float acc_u = 0.0f;
#pragma unroll 4
  for (int group = grp_off; group < hidden_groups; group += kGroupStride) {
    const uint8_t* bg = g_row + group * kQ80Bytes;
    const uint8_t* bu = u_row + group * kQ80Bytes;
    const float dg = __half2float(*reinterpret_cast<const half*>(bg));
    const float du = __half2float(*reinterpret_cast<const half*>(bu));
    const int wg = static_cast<int>(load_u32_2(bg + 2 + sub * 4));
    const int wu = static_cast<int>(load_u32_2(bu + 2 + sub * 4));
    const Q8_1& xq = smem_x[group];
    const float dx = __low2float(xq.ds);
    const int xw = reinterpret_cast<const int*>(xq.qs)[sub];
    acc_g += (dg * dx) * static_cast<float>(__dp4a(wg, xw, 0));
    acc_u += (du * dx) * static_cast<float>(__dp4a(wu, xw, 0));
  }

#pragma unroll
  for (int offset = kLanesPerRow / 2; offset > 0; offset >>= 1) {
    acc_g += __shfl_down_sync(0xffffffff, acc_g, offset, kLanesPerRow);
    acc_u += __shfl_down_sync(0xffffffff, acc_u, offset, kLanesPerRow);
  }

  if (lane_in_row == 0) {
    const float g16 = __half2float(__float2half_rn(acc_g));
    const float u16 = __half2float(__float2half_rn(acc_u));
    const half silu = __float2half_rn(g16 / (1.0f + expf(-g16)));
    smem_act[local_row] = __hmul(silu, __float2half_rn(u16));
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    vllm::sm70_gguf::quantize_q8_1_warp(
        inter_out + static_cast<size_t>(route) * inter_groups + blockIdx.x,
        __half2float(smem_act[threadIdx.x]));
  }
}

template <int ColsPerCTA>
__global__ __launch_bounds__(256) void mtp_q8_0_w2_reduce_kernel(
    half* __restrict__ out, const Q8_1* __restrict__ inter_q8,
    const uint8_t* __restrict__ w2, const float* __restrict__ topk_weights,
    const int32_t* __restrict__ topk_ids, int num_experts, int hidden,
    int inter_groups, int top_k) {
  static_assert(ColsPerCTA == 8 || ColsPerCTA == 16);
  constexpr int kColWarps = ColsPerCTA / 4;
  constexpr int kRouteGroups = 8 / kColWarps;

  extern __shared__ Q8_1 smem_inter[];
  __shared__ float smem_routed[kMaxTopK][ColsPerCTA];

  const int token = blockIdx.y;
  const int total_blocks = top_k * inter_groups;
  const uint32_t* src_inter = reinterpret_cast<const uint32_t*>(
      inter_q8 + static_cast<size_t>(token) * total_blocks);
  uint32_t* dst_inter = reinterpret_cast<uint32_t*>(smem_inter);
  for (int i = threadIdx.x; i < total_blocks * 9; i += 256) {
    dst_inter[i] = src_inter[i];
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int col_warp = warp % kColWarps;
  const int route_grp = warp / kColWarps;
  const int local_col = col_warp * 4 + (lane >> 3);
  const int sub = lane & 7;
  const int col = blockIdx.x * ColsPerCTA + local_col;

  const size_t w2_row_bytes = static_cast<size_t>(inter_groups) * kQ80Bytes;
  for (int slot = route_grp; slot < top_k; slot += kRouteGroups) {
    const int route = token * top_k + slot;
    const int expert = topk_ids[route];
    float acc = 0.0f;
    if (col < hidden && expert >= 0 && expert < num_experts) {
      const uint8_t* row_ptr =
          w2 + (static_cast<size_t>(expert) * hidden + col) * w2_row_bytes;
      const Q8_1* x_slot = smem_inter + slot * inter_groups;
#pragma unroll 5
      for (int g = 0; g < inter_groups; ++g) {
        const uint8_t* b = row_ptr + g * kQ80Bytes;
        const float dw = __half2float(*reinterpret_cast<const half*>(b));
        const int ww = static_cast<int>(load_u32_2(b + 2 + sub * 4));
        const float dx = __low2float(x_slot[g].ds);
        const int xw = reinterpret_cast<const int*>(x_slot[g].qs)[sub];
        acc += (dw * dx) * static_cast<float>(__dp4a(ww, xw, 0));
      }
    }
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, offset, 8);
    }
    if (sub == 0) {
      const float prob = topk_weights[route];
      smem_routed[slot][local_col] = __half2float(__float2half_rn(acc)) * prob;
    }
  }
  __syncthreads();

  if (threadIdx.x < ColsPerCTA) {
    const int lc = threadIdx.x;
    const int out_col = blockIdx.x * ColsPerCTA + lc;
    if (out_col < hidden) {
      float sum = 0.0f;
      for (int slot = 0; slot < top_k; ++slot) {
        sum += smem_routed[slot][lc];
      }
      out[static_cast<size_t>(token) * hidden + out_col] = __float2half_rn(sum);
    }
  }
}

void sm70_mtp_q8_0_moe_out(torch::Tensor out, torch::Tensor workspace,
                           torch::Tensor x, torch::Tensor w13, torch::Tensor w2,
                           torch::Tensor topk_weights, torch::Tensor topk_ids) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() &&
                  x.scalar_type() == at::kHalf,
              "sm70_mtp_q8_0_moe_out requires contiguous CUDA FP16 x [M, H]");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0,
              "sm70_mtp_q8_0_moe_out requires SM70");

  const int tokens = x.size(0);
  const int hidden = x.size(1);
  TORCH_CHECK(tokens >= 1 && tokens <= 64 && hidden >= kBlockSize &&
                  hidden % kBlockSize == 0,
              "Invalid x shape for sm70_mtp_q8_0_moe_out");
  TORCH_CHECK(out.is_cuda() && out.device() == x.device() &&
                  out.is_contiguous() && out.scalar_type() == at::kHalf &&
                  out.sizes() == x.sizes(),
              "Invalid out tensor for sm70_mtp_q8_0_moe_out");
  TORCH_CHECK(w13.is_cuda() && w13.device() == x.device() &&
                  w13.is_contiguous() && w13.scalar_type() == at::kByte &&
                  w13.dim() == 3,
              "Invalid w13 tensor for sm70_mtp_q8_0_moe_out");
  TORCH_CHECK(w2.is_cuda() && w2.device() == x.device() && w2.is_contiguous() &&
                  w2.scalar_type() == at::kByte && w2.dim() == 3,
              "Invalid w2 tensor for sm70_mtp_q8_0_moe_out");

  const int num_experts = w13.size(0);
  const int two_inter = w13.size(1);
  TORCH_CHECK(num_experts > 0 && two_inter >= 2 * kBlockSize &&
                  two_inter % (2 * kBlockSize) == 0,
              "Invalid w13 expert/intermediate shape");
  const int inter = two_inter / 2;
  const int hidden_groups = hidden / kBlockSize;
  const int inter_groups = inter / kBlockSize;
  TORCH_CHECK(w13.size(2) == hidden_groups * kQ80Bytes &&
                  w2.sizes() == at::IntArrayRef({num_experts, hidden,
                                                 inter_groups * kQ80Bytes}),
              "Q8_0 packed weight byte geometry mismatch");

  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.device() == x.device() &&
                  topk_ids.is_contiguous() &&
                  topk_ids.scalar_type() == at::kInt && topk_ids.dim() == 2 &&
                  topk_ids.size(0) == tokens,
              "Invalid topk_ids tensor");
  const int top_k = topk_ids.size(1);
  TORCH_CHECK(top_k >= 1 && top_k <= kMaxTopK, "Unsupported top_k");
  TORCH_CHECK(topk_weights.is_cuda() && topk_weights.device() == x.device() &&
                  topk_weights.is_contiguous() &&
                  topk_weights.scalar_type() == at::kFloat &&
                  topk_weights.sizes() == topk_ids.sizes(),
              "Invalid topk_weights tensor");

  const size_t x_q8_elems = static_cast<size_t>(tokens) * hidden_groups;
  const size_t routes = static_cast<size_t>(tokens) * top_k;
  const size_t inter_q8_elems = routes * inter_groups;
  const size_t needed_bytes = (x_q8_elems + inter_q8_elems) * sizeof(Q8_1);
  TORCH_CHECK(workspace.is_cuda() && workspace.device() == x.device() &&
                  workspace.is_contiguous() &&
                  workspace.scalar_type() == at::kByte &&
                  static_cast<size_t>(workspace.numel()) >= needed_bytes &&
                  reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 4 == 0,
              "Insufficient or unaligned workspace for sm70_mtp_q8_0_moe_out");

  const size_t w13_smem = hidden_groups * sizeof(Q8_1);
  const size_t w2_smem =
      static_cast<size_t>(top_k) * inter_groups * sizeof(Q8_1);
  TORCH_CHECK(w13_smem <= 44 * 1024 && w2_smem <= 44 * 1024,
              "Shared memory requirement exceeds SM70 static limit");

  auto* x_q8 = reinterpret_cast<Q8_1*>(workspace.data_ptr());
  auto* inter_q8 = x_q8 + x_q8_elems;
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  mtp_quantize_q8_1_kernel<<<dim3((hidden_groups + 7) / 8, tokens), 256, 0,
                             stream>>>(
      x_q8, reinterpret_cast<const half*>(x.data_ptr()), hidden_groups);

  if (routes <= 40) {
    mtp_q8_0_w13_swiglu_q8_1_kernel<16, 0>
        <<<dim3(inter_groups, routes), 512, w13_smem, stream>>>(
            inter_q8, x_q8, w13.data_ptr<uint8_t>(),
            topk_ids.data_ptr<int32_t>(), num_experts, inter, hidden_groups,
            top_k);
  } else if (routes <= 160) {
    mtp_q8_0_w13_swiglu_q8_1_kernel<16, 2>
        <<<dim3(inter_groups, routes), 512, w13_smem, stream>>>(
            inter_q8, x_q8, w13.data_ptr<uint8_t>(),
            topk_ids.data_ptr<int32_t>(), num_experts, inter, hidden_groups,
            top_k);
  } else {
    mtp_q8_0_w13_swiglu_q8_1_kernel<8, 3>
        <<<dim3(inter_groups, routes), 256, w13_smem, stream>>>(
            inter_q8, x_q8, w13.data_ptr<uint8_t>(),
            topk_ids.data_ptr<int32_t>(), num_experts, inter, hidden_groups,
            top_k);
  }

  if (tokens < 4) {
    mtp_q8_0_w2_reduce_kernel<8>
        <<<dim3((hidden + 7) / 8, tokens), 256, w2_smem, stream>>>(
            reinterpret_cast<half*>(out.data_ptr()), inter_q8,
            w2.data_ptr<uint8_t>(), topk_weights.data_ptr<float>(),
            topk_ids.data_ptr<int32_t>(), num_experts, hidden, inter_groups,
            top_k);
  } else {
    mtp_q8_0_w2_reduce_kernel<16>
        <<<dim3((hidden + 15) / 16, tokens), 256, w2_smem, stream>>>(
            reinterpret_cast<half*>(out.data_ptr()), inter_q8,
            w2.data_ptr<uint8_t>(), topk_weights.data_ptr<float>(),
            topk_ids.data_ptr<int32_t>(), num_experts, hidden, inter_groups,
            top_k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "sm70_mtp_q8_0_moe_out(Tensor(a!) out, Tensor(b!) workspace, Tensor x, "
      "Tensor w13, Tensor w2, Tensor topk_weights, Tensor topk_ids) -> ()");
  m.impl("sm70_mtp_q8_0_moe_out", torch::kCUDA, &sm70_mtp_q8_0_moe_out);
}
