// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <torch/library.h>
#define ENABLE_SM70_TURBOMIND
#include "../../csrc/ops.h"

// Match the admitted single-request sidecar surface. This does not register
// experimental batch/QPN8 kernels or change the production implementation.
TORCH_LIBRARY_FRAGMENT(_C_qwen38, ops) {
  ops.def(
      "nvfp4_moe_qpn_m1_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor weights, Tensor scales, Tensor expert_ids, "
      "bool broadcast_input, int split_k) -> ()");
  ops.impl("nvfp4_moe_qpn_m1_sm70_out", torch::kCUDA,
           &nvfp4_moe_qpn_m1_sm70_out);
  ops.def(
      "nvfp4_moe_qpn_mtp5_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor weights, Tensor scales, Tensor expert_ids, "
      "bool broadcast_input, int split_k) -> ()");
  ops.impl("nvfp4_moe_qpn_mtp5_sm70_out", torch::kCUDA,
           &nvfp4_moe_qpn_mtp5_sm70_out);
  ops.def(
      "nvfp4_qwen38_w2_direct_reduce_out(Tensor(a!) out, Tensor input, "
      "Tensor weights, Tensor scales, Tensor expert_ids, "
      "Tensor topk_weights) -> ()");
  ops.impl("nvfp4_qwen38_w2_direct_reduce_out", torch::kCUDA,
           &nvfp4_qwen38_w2_direct_reduce_out);
  ops.def(
      "nvfp4_qwen38_w13_fused_swiglu_out(Tensor(a!) out, Tensor input, "
      "Tensor weights, Tensor scales, Tensor expert_ids) -> ()");
  ops.impl("nvfp4_qwen38_w13_fused_swiglu_out", torch::kCUDA,
           &nvfp4_qwen38_w13_fused_swiglu_out);
  ops.def(
      "qwen38_shared_gate_exact_out(Tensor(a!) out, Tensor input, "
      "Tensor weight) -> ()");
  ops.impl("qwen38_shared_gate_exact_out", torch::kCUDA,
           &qwen38_shared_gate_exact_out);
}
