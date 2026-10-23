#include "flash_v100_policy.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <stdexcept>
#include "fused_mha.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "FlashAttention-2 implementation optimized for Volta";
  m.attr("paged_prefill_bm32_page_alignment") = 16;
  // D256/GQA6 E4M3 B1 supports standard p256/p1024 planning and FP32
  // partials in the same PARTIAL_T-parameterized launch/reducer as FP16.
  m.attr("xqa_shared_decode_strategy_revision") = 1;
  m.def("fwd", &flash_attention_forward,
        "FlashAttention-2 Forward Pass (Volta)");
  m.def("qk_scores_fwd", &flash_attention_qk_scores,
        "Debug FlashAttention QK score dump before softmax (Volta)");
  m.def("bwd", &flash_attention_backward,
        "FlashAttention-2 Backward Pass (Volta)");
  m.def("decode_paged_fwd", &flash_attention_decode_paged,
        "FlashAttention decode over paged KV cache (Volta)");
  m.def("decode_paged_xqa_fwd", &flash_attention_decode_paged_xqa,
        "FlashAttention XQA decode over paged KV cache (Volta)");
  m.def("decode_paged_xqa_staged_fwd", &flash_attention_decode_paged_xqa_staged,
        "Staged FlashAttention XQA decode over paged KV cache (Volta)");
  m.def("grouped_verify_paged_fwd", &flash_attention_grouped_verify_paged,
        "Exact grouped DFlash2 verification over paged KV cache (Volta)");
  m.def("grouped_e4m3_fp32_paged_fwd", &flash_attention_grouped_e4m3_fp32_paged,
        "Opt-in E4M3 small-query attention with FP32 partials and explicit row "
        "lengths");
  m.def("grouped_e4m3_fp32_precision_version",
        &flash_attention_grouped_e4m3_fp32_precision_version,
        "E4M3 grouped FP32 numerical implementation revision");
  m.def("tp2_e4m3_scalar_fast_version",
        &flash_attention_tp2_e4m3_scalar_fast_version,
        "Capability for the opt-in TP2 E4M3 scalar decoder");
  m.def("tp2_e4m3_scalar_fast_launch_count",
        &flash_attention_tp2_e4m3_scalar_fast_launch_count,
        "TP2 E4M3 scalar host dispatch count, including graph capture");
  m.attr("grouped_verify_e4m3") = true;
  m.def("grouped_verify_max_query_tokens",
        &flash_attention_grouped_verify_max_query_tokens,
        "Maximum query length supported by grouped DFlash2 verification");
  m.def("grouped_verify_request_major_abi_version",
        &flash_attention_grouped_verify_request_major_abi_version,
        "Request-major grouped DFlash2 forward ABI version");
  m.def("grouped_sparse_page4_fwd", &flash_attention_grouped_sparse_page4,
        "Grouped exact QSA page4 attention over paged KV cache (Volta)");
  m.def("grouped_sparse_page4_abi_version",
        &flash_attention_grouped_sparse_page4_abi_version,
        "Grouped sparse page4 forward ABI version");
  m.def("grouped_sparse_page4_plan_fwd",
        &flash_attention_grouped_sparse_page4_plan,
        "Build grouped exact QSA page4 tables over paged KV cache (Volta)");
  m.def("decode_paged_wmma_fwd", &flash_attention_decode_paged_wmma,
        "FlashAttention single-query decode through paged-prefill WMMA order "
        "(Volta)");
  m.def("decode_qk_scores_fwd", &flash_attention_decode_qk_scores,
        "Debug scalar paged decode QK score dump before softmax (Volta)");
  m.def("decode_turboquant_paged_fwd", &flash_attention_turboquant_decode_paged,
        "FlashAttention decode over TurboQuant paged KV cache (Volta)");
  m.def("dflash2_paged_bmhd_fwd", &flash_attention_dflash2_paged_bmhd,
        "Small noncausal DFlash2 paged attention in direct BMHD layout");
  m.def("prefill_paged_fwd", &flash_attention_prefill_paged,
        "FlashAttention prefill over paged KV cache (Volta)");
  m.def("prefill_paged_d256_bm32_allp_pair_scratch_fwd",
        &flash_attention_prefill_paged_d256_bm32_allp_pair_scratch,
        "Fixed causal D256 BM32 ALL_P pair-scratch paged prefill (SM70)");
  m.def("prefill_paged_d256_bm32_allp_pair_scratch_splitkv3_fwd",
        &flash_attention_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3,
        "Fixed causal D256 BM32 ALL_P pair-scratch three-way split-KV paged "
        "prefill (SM70)");
  m.def("prefill_paged_bfla_fwd", &flash_attention_prefill_paged_bfla,
        "BFLA sparse FlashAttention prefill over paged KV cache (Volta)");
  m.def("prefill_paged_splitkv_fwd", &flash_attention_prefill_paged_splitkv,
        "FlashAttention split-KV prefill over paged KV cache (Volta)");
  m.def("fp8_e5m2_paged_kv_to_fp16", &flash_attention_fp8_e5m2_paged_kv_to_fp16,
        "Expand paged FP8 E5M2 K/V into a preallocated FP16 paged workspace");
  m.def("fp8_e4m3_paged_kv_to_fp16", &flash_attention_fp8_e4m3_paged_kv_to_fp16,
        "Expand paged FP8 E4M3 K/V into a preallocated FP16 paged workspace");

  m.attr("policy_abi_version") = 1;
  m.attr("policy_fields") = flash_v100::policy::size;
  py::class_<flash_v100::policy::PreparedPolicy>(m, "PreparedPolicy")
      .def(py::init<const std::vector<std::optional<std::string>>&>())
      .def_property_readonly(
          "values",
          [](const flash_v100::policy::PreparedPolicy& p) { return p.values; })
      .def_property_readonly("observations",
                             [](const flash_v100::policy::PreparedPolicy& p) {
                               std::vector<unsigned long long> result;
                               for (const auto& value : p.observations)
                                 result.push_back(value.load());
                               return result;
                             });
  m.def("fwd_configured",
        flash_v100::policy::with_policy(&flash_attention_forward));
  m.def("qk_scores_fwd_configured",
        flash_v100::policy::with_policy(&flash_attention_qk_scores));
  m.def("bwd_configured",
        flash_v100::policy::with_policy(&flash_attention_backward));
  m.def("decode_paged_fwd_configured",
        flash_v100::policy::with_policy(&flash_attention_decode_paged));
  m.def("decode_paged_xqa_fwd_configured",
        flash_v100::policy::with_policy(&flash_attention_decode_paged_xqa));
  m.def("decode_paged_xqa_staged_fwd_configured",
        flash_v100::policy::with_policy(
            &flash_attention_decode_paged_xqa_staged));
  m.def("grouped_verify_paged_fwd_configured",
        flash_v100::policy::with_policy(&flash_attention_grouped_verify_paged));
  m.def("grouped_e4m3_fp32_paged_fwd_configured",
        flash_v100::policy::with_policy(
            &flash_attention_grouped_e4m3_fp32_paged));
  m.def("tp2_e4m3_scalar_fast_launch_count_configured",
        flash_v100::policy::with_policy(
            &flash_attention_tp2_e4m3_scalar_fast_launch_count));
  m.def("grouped_sparse_page4_fwd_configured",
        flash_v100::policy::with_policy(&flash_attention_grouped_sparse_page4));
  m.def("grouped_sparse_page4_plan_fwd_configured",
        flash_v100::policy::with_policy(
            &flash_attention_grouped_sparse_page4_plan));
  m.def("decode_paged_wmma_fwd_configured",
        flash_v100::policy::with_policy(&flash_attention_decode_paged_wmma));
  m.def("decode_qk_scores_fwd_configured",
        flash_v100::policy::with_policy(&flash_attention_decode_qk_scores));
  m.def("decode_turboquant_paged_fwd_configured",
        flash_v100::policy::with_policy(
            &flash_attention_turboquant_decode_paged));
  m.def("dflash2_paged_bmhd_fwd_configured",
        flash_v100::policy::with_policy(&flash_attention_dflash2_paged_bmhd));
  m.def("prefill_paged_fwd_configured",
        flash_v100::policy::with_policy(&flash_attention_prefill_paged));
  m.def("prefill_paged_d256_bm32_allp_pair_scratch_fwd_configured",
        flash_v100::policy::with_policy(
            &flash_attention_prefill_paged_d256_bm32_allp_pair_scratch));
  m.def(
      "prefill_paged_d256_bm32_allp_pair_scratch_splitkv3_fwd_configured",
      flash_v100::policy::with_policy(
          &flash_attention_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3));
  m.def("prefill_paged_bfla_fwd_configured",
        flash_v100::policy::with_policy(&flash_attention_prefill_paged_bfla));
  m.def(
      "prefill_paged_splitkv_fwd_configured",
      flash_v100::policy::with_policy(&flash_attention_prefill_paged_splitkv));
}
