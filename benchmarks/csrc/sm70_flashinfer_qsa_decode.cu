// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <torch/types.h>

#include <flashinfer/attention/sm70/qsa_decode.cuh>

namespace {
using namespace flashinfer::attention::sm70;

void run(torch::Tensor q, torch::Tensor k, torch::Tensor v,
         torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
         torch::Tensor offsets, torch::Tensor metadata, torch::Tensor zero,
         torch::Tensor partial, torch::Tensor lse, torch::Tensor output,
         int64_t splits) {
  const c10::cuda::CUDAGuard guard(q.device());
  for (const auto& t : {q, k, v, indices, table, requests, offsets, metadata,
                        zero, partial, lse, output}) {
    TORCH_CHECK(t.is_cuda() && t.device() == q.device());
  }
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 7 &&
                  at::cuda::getCurrentDeviceProperties()->minor == 0,
              "This benchmark entry is SM70 only; upstream gates unchanged");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && v.sizes() == k.sizes());
  TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
              v.scalar_type() == at::kHalf &&
              output.scalar_type() == at::kHalf &&
              zero.scalar_type() == at::kHalf);
  TORCH_CHECK(q.size(2) == 256 && k.size(3) == 256 && q.stride(2) == 1 &&
              k.stride(3) == 1 && v.strides() == k.strides());
  for (const auto& t : {q, k, v}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "Vector loads require 16-byte aligned inputs");
    for (int axis = 0; axis + 1 < t.dim(); ++axis)
      TORCH_CHECK(t.stride(axis) % 8 == 0,
                  "Vector loads require aligned row/head strides");
  }
  TORCH_CHECK(q.stride(0) <= UINT32_MAX && q.stride(1) <= UINT32_MAX);
  const int rows = q.size(0), heads = q.size(1), kv_heads = k.size(2);
  TORCH_CHECK(rows > 0 && heads > 0 && heads <= 32 && kv_heads > 0 &&
              heads % kv_heads == 0 && k.size(0) > 0 && k.size(1) > 0);
  const int group = heads / kv_heads;
  TORCH_CHECK(group == 1 || group == 2 || group == 4 || group == 6 ||
              group == 8);
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) == rows &&
              indices.size(1) > 0 && indices.stride(1) == 1);
  TORCH_CHECK(table.dim() == 2 && table.size(0) > 0 && table.size(1) > 0 &&
              table.stride(1) == 1);
  TORCH_CHECK(requests.dim() == 1 && requests.numel() == rows &&
              requests.is_contiguous());
  for (const auto& t : {indices, table, requests, metadata})
    TORCH_CHECK(t.scalar_type() == at::kInt);
  TORCH_CHECK(splits > 0 && splits <= 64);
  const int selected = indices.size(1);
  const int width = ((selected + splits - 1) / splits) * splits;
  TORCH_CHECK(int64_t(rows) * width < INT32_MAX);
  TORCH_CHECK(offsets.scalar_type() == at::kLong && offsets.is_contiguous() &&
              offsets.numel() == int64_t(rows) * width);
  TORCH_CHECK(metadata.is_contiguous() &&
              metadata.numel() == rows + 2 + 2 * rows * splits);
  TORCH_CHECK(zero.is_contiguous() && zero.numel() == 256);
  TORCH_CHECK(partial.scalar_type() == at::kFloat && partial.is_contiguous() &&
              partial.numel() == int64_t(rows) * splits * heads * 256);
  TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.is_contiguous() &&
              lse.numel() == int64_t(rows) * splits * heads);
  TORCH_CHECK(output.sizes() == q.sizes() && output.is_contiguous());
  const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  auto* meta = metadata.data_ptr<int32_t>();
  PrepareQSA<<<rows, 256, 0, stream>>>(
      indices.data_ptr<int32_t>(), table.data_ptr<int32_t>(),
      requests.data_ptr<int32_t>(), offsets.data_ptr<int64_t>(), meta, rows,
      selected, width, splits, k.size(1), table.size(1), table.size(0),
      k.size(0), indices.stride(0), table.stride(0), k.stride(0), k.stride(1));
  QSAParams p{};
  p.q = reinterpret_cast<const half*>(q.data_ptr());
  p.o = partial.data_ptr<float>();
  p.lse = lse.data_ptr<float>();
  p.paged_kv.batch_size = rows;
  p.paged_kv.num_heads = kv_heads;
  p.paged_kv.width = width;
  p.paged_kv.head_stride = k.stride(2);
  p.paged_kv.k_data = {reinterpret_cast<const half*>(k.data_ptr()),
                       reinterpret_cast<const half*>(zero.data_ptr())};
  p.paged_kv.v_data = {reinterpret_cast<const half*>(v.data_ptr()),
                       reinterpret_cast<const half*>(zero.data_ptr())};
  p.paged_kv.indptr = meta;
  p.paged_kv.offsets = offsets.data_ptr<int64_t>();
  p.padded_batch_size = rows * splits;
  p.num_qo_heads = heads;
  p.request_indices = meta + rows + 1;
  p.kv_tile_indices = p.request_indices + rows * splits;
  p.kv_chunk_size_ptr = p.kv_tile_indices + rows * splits;
  p.q_stride_n = q.stride(0);
  p.q_stride_h = q.stride(1);
#define DISPATCH(G)                                                         \
  case G:                                                                   \
    LaunchQSADecode<G>(p, reinterpret_cast<half*>(output.data_ptr()), rows, \
                       splits, stream);                                     \
    break
  switch (group) {
    DISPATCH(1);
    DISPATCH(2);
    DISPATCH(4);
    DISPATCH(6);
    DISPATCH(8);
  }
#undef DISPATCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_flashinfer_qsa_sm70, m) {
  m.def(
      "run(Tensor q, Tensor k, Tensor v, Tensor indices, Tensor table, "
      "Tensor requests, Tensor(a!) offsets, Tensor(b!) metadata, Tensor zero, "
      "Tensor(c!) partial, Tensor(d!) lse, Tensor(e!) output, int splits) -> "
      "()");
}
TORCH_LIBRARY_IMPL(_C_flashinfer_qsa_sm70, CUDA, m) { m.impl("run", &run); }
