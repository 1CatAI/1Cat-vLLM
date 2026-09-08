// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Compile the production QPN2/QPN4 sources without rebuilding all of vLLM.
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <torch/all.h>
#include <torch/library.h>

#include "../../csrc/ops.h"

void nvfp4_gemm_sm70_out(torch::Tensor out, torch::Tensor input,
                         torch::Tensor weight, torch::Tensor scales,
                         int64_t group_size, int64_t k_ld, int64_t q_ld,
                         bool gated_silu) {
  static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_C::nvfp4_gemm_sm70_out", "");
  torch::jit::Stack stack{out,        input, weight, scales,
                          group_size, k_ld,  q_ld,   gated_silu};
  op.callBoxed(&stack);
}

void silu_and_mul(torch::Tensor& out, torch::Tensor& input) {
  static const auto op =
      c10::Dispatcher::singleton().findSchemaOrThrow("_C::silu_and_mul", "");
  torch::jit::Stack stack{out, input};
  op.callBoxed(&stack);
}

TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def("nvfp4_qpn2_prepare_scales_sm70(Tensor weight_scale) -> Tensor");
  ops.def(
      "nvfp4_qpn2_restore_tm_scales_sm70_out(Tensor(a!) out, "
      "Tensor scales, float global_scale) -> ()");
  ops.impl("nvfp4_qpn2_restore_tm_scales_sm70_out", torch::kCUDA,
           &nvfp4_qpn2_restore_tm_scales_sm70_out);
  ops.def(
      "nvfp4_qpn2_compact_tm_gemm_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor weight, Tensor scales, float global_scale, int k_ld, "
      "int q_ld, bool gated_silu) -> ()");
  ops.impl("nvfp4_qpn2_compact_tm_gemm_sm70_out", torch::kCUDA,
           &nvfp4_qpn2_compact_tm_gemm_sm70_out);
  ops.impl("nvfp4_qpn2_prepare_scales_sm70", torch::kCUDA,
           &nvfp4_qpn2_prepare_scales_sm70);
  ops.def(
      "nvfp4_qpn2_tm_dispatch_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor tm_weight, Tensor scales, float global_scale, int split_k, "
      "int accumulator_chains, Tensor tm_scales, int tm_group_size, "
      "int tm_k_ld, int tm_q_ld, bool gated_silu, int min_prefill_m) -> ()");
  ops.impl("nvfp4_qpn2_tm_dispatch_sm70_out", torch::kCUDA,
           &nvfp4_qpn2_tm_dispatch_sm70_out);
}

// Control operators from the same compiler invocation as the shared reader.
TORCH_LIBRARY_FRAGMENT(_qpn2_shared_control, ops) {
  ops.def("prepare(Tensor weight_packed, Tensor weight_scale) -> Tensor[]");
  ops.impl("prepare", torch::kCUDA, &nvfp4_qpn2_prepare_sm70);
  ops.def(
      "dispatch(Tensor(a!) out, Tensor input, Tensor codes, Tensor scales, "
      "float global_scale, int split_k, int accumulator_chains, "
      "Tensor tm_weight, Tensor tm_scales, int tm_group_size, int tm_k_ld, "
      "int tm_q_ld, bool gated_silu, int min_prefill_m) -> ()");
  ops.impl("dispatch", torch::kCUDA, &nvfp4_qpn2_prefill_dispatch_sm70_out);
}

// Full-model overlays must update both layouts from the declared source.
// Older core DSOs can dispatch M9..32 to TurboMind even when current Python
// reports QPN2 M<=32. Load the core DSO before this optional implementation
// overlay, and use the same overlay in both model arms.
#ifdef VLLM_QPN2_SHARED_ALIGN_NATIVE_CONTROL
TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("nvfp4_qpn2_prepare_sm70", &nvfp4_qpn2_prepare_sm70);
  ops.impl("nvfp4_qpn2_gemm_sm70_out", &nvfp4_qpn2_gemm_sm70_out);
  ops.impl("nvfp4_qpn2_gated_sm70_out", &nvfp4_qpn2_gated_sm70_out);
  ops.impl("nvfp4_qpn2_dispatch_sm70_out", &nvfp4_qpn2_dispatch_sm70_out);
  ops.impl("nvfp4_qpn2_prefill_dispatch_sm70_out",
           &nvfp4_qpn2_prefill_dispatch_sm70_out);
}
#endif
