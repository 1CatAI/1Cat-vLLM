// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "gguf_prefill_transcode.cuh"

namespace {
using namespace vllm::sm70_gguf::transcode;

template <int Type>
void launch(torch::Tensor& dst_w, torch::Tensor& dst_s, int64_t w_stride,
            int64_t s_stride, const torch::Tensor& src0,
            const std::optional<torch::Tensor>& src1,
            const torch::Tensor& experts, const torch::Tensor& count,
            int64_t ctas, int64_t n_out, int64_t k, int64_t k_offset,
            int64_t src_stride, torch::Tensor& error) {
  const int packs = static_cast<int>(n_out / kPack);
  const int64_t items = experts.numel() * packs;
  if (items == 0) return;
  const int grid = static_cast<int>(std::min<int64_t>(ctas, items));
  const size_t smem = size_t(kPack) * src_stride;
  auto kernel = transcode_kernel<Type>;
  if (smem > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
  kernel<<<grid, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
      dst_w.data_ptr<uint8_t>(), dst_s.data_ptr<uint8_t>(), w_stride, s_stride,
      src0.data_ptr<int64_t>(), src1 ? src1->data_ptr<int64_t>() : nullptr,
      experts.data_ptr<int32_t>(), count.data_ptr<int32_t>(), packs, int(n_out),
      int(k), int(k_offset), int(src_stride), error.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

// Converts the first `count` (device int32) experts listed in `experts`
// from their original rows into a stacked canonical bank, with at most
// `ctas` CTAs. src0/src1: int64 [num_experts] row-0 addresses (device or
// host-mapped) of each expert's source rows; src1 selects gate/up
// interleaving. Output
// columns are source columns [k_offset, k_offset + k) (a TP slice inside
// whole source blocks). dst_w/dst_s: uint8 banks, expert e at e * w_stride /
// e * s_stride bytes.
void gguf_prefill_transcode_sm70(torch::Tensor dst_w, torch::Tensor dst_s,
                                 int64_t w_stride, int64_t s_stride,
                                 torch::Tensor src0,
                                 std::optional<torch::Tensor> src1,
                                 torch::Tensor experts, torch::Tensor count,
                                 int64_t ctas, int64_t source_type,
                                 int64_t n_out, int64_t k, int64_t k_offset,
                                 int64_t src_stride, torch::Tensor error) {
  TORCH_CHECK(dst_w.is_cuda() && dst_s.is_cuda() && src0.is_cuda() &&
                  experts.is_cuda() && count.is_cuda() && error.is_cuda() &&
                  count.scalar_type() == at::kInt && count.numel() >= 1 &&
                  ctas > 0,
              "prefill transcode: tensors must be CUDA");
  TORCH_CHECK(
      dst_w.scalar_type() == at::kByte && dst_s.scalar_type() == at::kByte &&
          src0.scalar_type() == at::kLong &&
          experts.scalar_type() == at::kInt && error.scalar_type() == at::kInt,
      "prefill transcode: bad dtypes");
  TORCH_CHECK(!src1 || (src1->is_cuda() && src1->scalar_type() == at::kLong &&
                        src1->numel() == src0.numel()),
              "prefill transcode: bad second source");
  // Rows themselves need not be 8-byte aligned; a pack of 16 or 32 rows is
  // staged with 8-byte loads, so expert bases must be 8-byte aligned.
  TORCH_CHECK(n_out > 0 && n_out % kPack == 0 && k > 0 && src_stride > 0 &&
                  w_stride > 0 && s_stride > 0,
              "prefill transcode: bad geometry");
  const at::cuda::CUDAGuard guard(dst_w.device());
  // Source rows must hold whole blocks covering [k_offset, k_offset + k).
  auto rows_ok = [&](int block, int bytes) {
    return k % 32 == 0 && k_offset >= 0 && src_stride % bytes == 0 &&
           src_stride / bytes * block >= k_offset + k;
  };
#define CASE(TYPE, BLOCK, BYTES)                                               \
  case TYPE:                                                                   \
    TORCH_CHECK(rows_ok(BLOCK, BYTES), "prefill transcode: GGUF type ", TYPE,  \
                " rows do not cover the requested columns");                   \
    launch<TYPE>(dst_w, dst_s, w_stride, s_stride, src0, src1, experts, count, \
                 ctas, n_out, k, k_offset, src_stride, error);                 \
    break;
  switch (source_type) {
    CASE(18, 256, 98)
    CASE(21, 256, 110)
    CASE(22, 256, 82)
    CASE(20, 32, 18)
    CASE(23, 256, 136)
    CASE(42, 64, 18)
    default:
      TORCH_CHECK(false, "prefill transcode: unsupported GGUF type ",
                  source_type);
  }
#undef CASE
}
