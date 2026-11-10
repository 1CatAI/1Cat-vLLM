// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <torch/all.h>
#include <utility>
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace vllm::sm70 {

// A codec chooses converters, packed dtypes, group count and leading
// dimensions. Dense and grouped execution share packing geometry; the caller
// adds indices, offsets and group counts only after constructing these
// canonical layouts.
inline turbomind::gemm::MatrixLayout packed_weight_layout(
    const turbomind::gemm::LayoutConverter& converter, turbomind::DataType type,
    int64_t n, int64_t k, int64_t ld) {
  using namespace turbomind::gemm;
  const bool operand_a = get_operand_tag(converter.pack) == OPERAND_A;
  MatrixLayout desc{
      turbomind::kHalf, converter.order, static_cast<int>(n),
      static_cast<int>(k),
      converter.order == kRowMajor ? static_cast<int>(k) : static_cast<int>(n)};
  if (!operand_a) {
    std::swap(desc.rows, desc.cols);
    desc.order = ~desc.order;
  }
  desc.type = type;
  desc.pack = converter.pack;
  if (operand_a) desc = transpose(desc);
  desc.ld = static_cast<int>(ld);
  return desc;
}

inline turbomind::gemm::MatrixLayout packed_scale_layout(
    const turbomind::gemm::LayoutConverter& converter, turbomind::DataType type,
    int64_t n, int64_t groups, int64_t ld) {
  using namespace turbomind::gemm;
  const bool operand_u = get_operand_tag(converter.pack) == OPERAND_U;
  MatrixLayout desc{type, converter.order, static_cast<int>(n),
                    static_cast<int>(groups), static_cast<int>(n)};
  if (!operand_u) {
    std::swap(desc.rows, desc.cols);
    desc.order = ~desc.order;
  }
  desc.pack = converter.pack;
  if (operand_u) desc = transpose(desc);
  desc.ld = static_cast<int>(ld);
  return desc;
}

inline int run_dense_packed_gemm(
    turbomind::gemm::Gemm& gemm, const turbomind::gemm::Workspace& workspace,
    const turbomind::gemm::Operation& operation, cudaStream_t stream,
    const torch::Tensor& input, const torch::Tensor& out, const void* weight,
    const turbomind::gemm::MatrixLayout& weight_layout, const void* scales,
    const turbomind::gemm::MatrixLayout& scale_layout, int64_t n,
    int64_t input_ld) {
  using namespace turbomind::gemm;
  const MatrixLayout a{
      turbomind::kHalf, kRowMajor, static_cast<int>(input.size(0)),
      static_cast<int>(input.size(1)), static_cast<int>(input_ld)};
  const MatrixLayout u{};
  const MatrixLayout d{turbomind::kHalf, kRowMajor,
                       static_cast<int>(input.size(0)), static_cast<int>(n),
                       static_cast<int>(out.stride(0))};
  // In-place C/D, FP32 alpha/beta and the logical N before a gated epilogue
  // are intentional. AWQ keeps input_ld=K; other codecs pass the real stride.
  return gemm.Run(operation, 1.f, input.data_ptr(), a, nullptr, u, weight,
                  weight_layout, scales, scale_layout, 0.f, out.data_ptr(), d,
                  out.data_ptr(), d, workspace, stream);
}

}  // namespace vllm::sm70
