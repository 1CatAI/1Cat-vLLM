// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
// Private primitives for H3-owned, validated geometry. No activation is cached
// here.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace h3_vsa_layout {

__global__ void tile_three(const uint4* q, const uint4* k, const uint4* v,
                           const int* source_rows, uint4* out, int64_t vectors,
                           int64_t source_tokens, int64_t tiled_tokens,
                           int64_t row_vectors, int64_t batches) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= vectors) return;
  int64_t lane = i % row_vectors;
  int64_t row = (i / row_vectors) % tiled_tokens;
  int64_t batch = (i / (row_vectors * tiled_tokens)) % batches;
  int tensor = i / (row_vectors * tiled_tokens * batches);
  int original = source_rows[row];
  const uint4* input = tensor == 0 ? q : tensor == 1 ? k : v;
  uint4 value = make_uint4(0, 0, 0, 0);
  if (original >= 0 && original < source_tokens)
    value = input[(batch * source_tokens + original) * row_vectors + lane];
  out[i] = value;
}

__global__ void gate_untiling(const uint4* sparse, const uint4* compressed,
                              const uint4* gate, const int* tiled_rows,
                              uint4* output, int64_t vectors,
                              int64_t source_tokens, int64_t tiled_tokens,
                              int64_t row_vectors) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= vectors) return;
  int64_t lane = i % row_vectors;
  int64_t row = (i / row_vectors) % source_tokens;
  int64_t batch = i / (row_vectors * source_tokens);
  int tiled = tiled_rows[row];
  if (tiled < 0 || tiled >= tiled_tokens) {
    output[i] = make_uint4(0, 0, 0, 0);
    return;
  }
  uint4 sv = sparse[(batch * tiled_tokens + tiled) * row_vectors + lane];
  uint4 cv =
      compressed[(batch * (tiled_tokens / 64) + tiled / 64) * row_vectors +
                 lane];
  uint4 gv = gate[i];
  uint4 answer;
  auto* a = reinterpret_cast<__half2*>(&answer);
  const auto* s = reinterpret_cast<const __half2*>(&sv);
  const auto* c = reinterpret_cast<const __half2*>(&cv);
  const auto* g = reinterpret_cast<const __half2*>(&gv);
#pragma unroll
  for (int j = 0; j < 4; ++j) a[j] = __hadd2_rn(s[j], __hmul2_rn(c[j], g[j]));
  output[i] = answer;
}

void check_operand(const torch::Tensor& x, const torch::Tensor& q) {
  TORCH_CHECK(x.device() == q.device() && x.scalar_type() == at::kHalf &&
                  x.is_contiguous() && uintptr_t(x.data_ptr()) % 16 == 0,
              "H3 layout input must be aligned contiguous FP16 on one device");
}

torch::Tensor tile(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                   torch::Tensor rows, torch::Tensor out) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.size(3) == 128 &&
                  q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0,
              "H3 layout requires nonempty CUDA BSHD with D128");
  check_operand(q, q);
  check_operand(k, q);
  check_operand(v, q);
  check_operand(out, q);
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
              "QKV shapes differ");
  TORCH_CHECK(rows.device() == q.device() && rows.scalar_type() == at::kInt &&
                  rows.dim() == 1 && rows.is_contiguous() && rows.numel() > 0 &&
                  rows.numel() % 64 == 0,
              "H3 source map must be contiguous int32 tile rows");
  TORCH_CHECK(out.dim() == 5 && out.size(0) == 3 && out.size(1) == q.size(0) &&
                  out.size(2) == rows.numel() && out.size(3) == q.size(2) &&
                  out.size(4) == 128,
              "H3 tiled output shape mismatch");
  at::assert_no_overlap(out, rows);
  at::assert_no_overlap(out, q);
  at::assert_no_overlap(out, k);
  at::assert_no_overlap(out, v);
  c10::cuda::CUDAGuard guard(q.device());
  int64_t n = out.numel() / 8;
  tile_three<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint4*>(q.data_ptr()),
      reinterpret_cast<uint4*>(k.data_ptr()),
      reinterpret_cast<uint4*>(v.data_ptr()), rows.data_ptr<int>(),
      reinterpret_cast<uint4*>(out.data_ptr()), n, q.size(1), rows.numel(),
      q.size(2) * 16, q.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor finish(torch::Tensor sparse, torch::Tensor compressed,
                     torch::Tensor gate, torch::Tensor rows) {
  TORCH_CHECK(sparse.is_cuda() && sparse.dim() == 4 && sparse.size(3) == 128 &&
                  sparse.size(0) > 0 && sparse.size(1) > 0 &&
                  sparse.size(1) % 64 == 0 && sparse.size(2) > 0,
              "H3 sparse output requires nonempty CUDA tile geometry");
  check_operand(sparse, sparse);
  check_operand(compressed, sparse);
  check_operand(gate, sparse);
  TORCH_CHECK(gate.dim() == 4 && gate.size(1) > 0 &&
                  gate.size(0) == sparse.size(0) &&
                  gate.size(2) == sparse.size(2) && gate.size(3) == 128,
              "H3 gate geometry mismatch");
  TORCH_CHECK(compressed.dim() == 4 && compressed.size(0) == sparse.size(0) &&
                  compressed.size(1) == sparse.size(1) / 64 &&
                  compressed.size(2) == sparse.size(2) &&
                  compressed.size(3) == 128,
              "H3 compressed geometry mismatch");
  TORCH_CHECK(rows.device() == sparse.device() &&
                  rows.scalar_type() == at::kInt && rows.dim() == 1 &&
                  rows.is_contiguous() && rows.numel() == gate.size(1),
              "H3 untiling map mismatch");
  c10::cuda::CUDAGuard guard(sparse.device());
  auto out = torch::empty_like(gate);
  int64_t n = out.numel() / 8;
  gate_untiling<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint4*>(sparse.data_ptr()),
      reinterpret_cast<uint4*>(compressed.data_ptr()),
      reinterpret_cast<uint4*>(gate.data_ptr()), rows.data_ptr<int>(),
      reinterpret_cast<uint4*>(out.data_ptr()), n, gate.size(1), sparse.size(1),
      sparse.size(2) * 16);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace h3_vsa_layout
