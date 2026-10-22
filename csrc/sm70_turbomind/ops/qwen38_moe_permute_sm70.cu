// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Qwen3.8 Flash-Next (E512, top-10, H2560) concurrent-decode MoE permute.
// One CTA places every routed slot stably by expert (counting sort), which
// equals the stable expert-only CUB sort of moe_permute, and writes the dense per-expert
// offsets the grouped GEMM consumes plus the inverse permutation the
// unpermute reduction reads. A second launch gathers the input rows. This
// replaces the generic route's copy, fill, radix sort, offset, expand and
// offset-narrowing launches for verification batches.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
constexpr int kExperts = 512;
constexpr int kHidden = 2560;
constexpr int kMaxSlots = 1024;  // one thread per routed slot

template <typename IdT>
__global__ __launch_bounds__(kMaxSlots, 1) void qwen38_moe_sort_kernel(
    const IdT* __restrict__ topk_ids, int slots, int* __restrict__ offsets,
    int* __restrict__ sorted_row_idx, int* __restrict__ inv_permuted_idx) {
  __shared__ int counts[kExperts + 1];
  __shared__ int placed[kExperts];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  for (int e = t; e <= kExperts; e += blockDim.x) counts[e] = 0;
  for (int e = t; e < kExperts; e += blockDim.x) placed[e] = 0;
  __syncthreads();
  // Out-of-range slots get distinct negative keys so they never match.
  const int expert = t < slots ? static_cast<int>(topk_ids[t]) : -1 - t;
  if (t < slots) atomicAdd(&counts[expert + 1], 1);
  __syncthreads();
  // Inclusive scan of counts[1..512] (Hillis-Steele) gives offsets[0..512].
  for (int stride = 1; stride <= kExperts; stride <<= 1) {
    int value = 0;
    if (t <= kExperts && t >= stride) value = counts[t - stride];
    __syncthreads();
    if (t <= kExperts && t >= stride) counts[t] += value;
    __syncthreads();
  }
  if (t <= kExperts) offsets[t] = counts[t];
  // Stable placement: warps in slot order, lanes in slot order within a warp,
  // equals the stable expert-only sort of moe_permute.
  const int warps = (slots + 31) >> 5;
  const unsigned peers = __match_any_sync(0xffffffffu, expert);
  const int rank = __popc(peers & ((1u << lane) - 1u));
  for (int w = 0; w < warps; ++w) {
    if (warp == w && t < slots) {
      const int position = counts[expert] + placed[expert] + rank;
      sorted_row_idx[position] = t;
      inv_permuted_idx[t] = position;
    }
    __syncthreads();
    if (warp == w && t < slots && rank == 0) placed[expert] += __popc(peers);
    __syncthreads();
  }
}

__global__ void qwen38_moe_gather_rows_kernel(
    const uint4* __restrict__ input, const int* __restrict__ sorted_row_idx,
    int top_k, uint4* __restrict__ permuted_input) {
  constexpr int kVectors = kHidden * sizeof(half) / sizeof(uint4);
  const int row = blockIdx.x;
  const int source = sorted_row_idx[row] / top_k;
  for (int v = threadIdx.x; v < kVectors; v += blockDim.x)
    permuted_input[row * kVectors + v] = input[source * kVectors + v];
}

void qwen38_moe_permute_dense(torch::Tensor input, torch::Tensor topk_ids,
                              torch::Tensor permuted_input,
                              torch::Tensor expert_offsets,
                              torch::Tensor sorted_row_idx,
                              torch::Tensor inv_permuted_idx) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
                  input.is_contiguous() && input.dim() == 2 &&
                  input.size(1) == kHidden,
              "Qwen3.8 MoE permute requires contiguous FP16 [T, 2560] input");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.is_contiguous() &&
                  topk_ids.dim() == 2 && topk_ids.size(0) == input.size(0) &&
                  (topk_ids.scalar_type() == at::kInt ||
                   topk_ids.scalar_type() == at::kLong),
              "Qwen3.8 MoE permute requires contiguous int32/int64 [T, K] ids");
  const int top_k = topk_ids.size(1);
  const int slots = topk_ids.numel();
  TORCH_CHECK(slots > 0 && slots <= kMaxSlots,
              "Qwen3.8 MoE permute supports 1..1024 routed slots");
  TORCH_CHECK(permuted_input.is_cuda() &&
                  permuted_input.scalar_type() == at::kHalf &&
                  permuted_input.is_contiguous() &&
                  permuted_input.size(0) >= slots &&
                  permuted_input.size(1) == kHidden,
              "Qwen3.8 MoE permute output rows are too small");
  for (const auto& t : {expert_offsets, sorted_row_idx, inv_permuted_idx})
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kInt && t.is_contiguous(),
                "Qwen3.8 MoE permute index buffers must be contiguous int32");
  TORCH_CHECK(expert_offsets.numel() >= kExperts + 1 &&
                  sorted_row_idx.numel() >= slots &&
                  inv_permuted_idx.numel() >= slots,
              "Qwen3.8 MoE permute index buffers are too small");
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (topk_ids.scalar_type() == at::kInt) {
    qwen38_moe_sort_kernel<int><<<1, kMaxSlots, 0, stream>>>(
        topk_ids.data_ptr<int>(), slots, expert_offsets.data_ptr<int>(),
        sorted_row_idx.data_ptr<int>(), inv_permuted_idx.data_ptr<int>());
  } else {
    qwen38_moe_sort_kernel<int64_t><<<1, kMaxSlots, 0, stream>>>(
        topk_ids.data_ptr<int64_t>(), slots, expert_offsets.data_ptr<int>(),
        sorted_row_idx.data_ptr<int>(), inv_permuted_idx.data_ptr<int>());
  }
  qwen38_moe_gather_rows_kernel<<<slots, 320, 0, stream>>>(
      reinterpret_cast<const uint4*>(input.data_ptr()),
      sorted_row_idx.data_ptr<int>(), top_k,
      reinterpret_cast<uint4*>(permuted_input.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "qwen38_moe_permute_dense_sm70_out(Tensor input, Tensor topk_ids, "
      "Tensor(a!) permuted_input, Tensor(b!) expert_offsets, "
      "Tensor(c!) sorted_row_idx, Tensor(d!) inv_permuted_idx) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen38_moe_permute_dense_sm70_out", &qwen38_moe_permute_dense);
}
