#include <torch/extension.h>
#include <ATen/ATen.h>
#include <stdexcept>
#include "fused_mha.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// pi-lens-ignore: clang:missing_type_specifier
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "FlashAttention-2 implementation optimized for Volta";
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
  m.def("grouped_verify_max_query_tokens",
        &flash_attention_grouped_verify_max_query_tokens,
        "Maximum query length supported by grouped DFlash2 verification");
  m.def("grouped_sparse_page4_fwd", &flash_attention_grouped_sparse_page4,
        "Grouped exact QSA page4 attention over paged KV cache (Volta)");
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
  m.def("int8_block32_reshape_and_cache",
        &flash_attention_int8_block32_reshape_and_cache,
        "Quantize FP16 K/V into an INT8 cache with adaptive page block scales");
  m.def("int8_block32_decode_paged", &flash_attention_int8_block32_decode_paged,
        "Decode paged block-scaled INT8 K/V with in-register dequantization");
  m.def("int8_block32_prefill_paged",
        &flash_attention_int8_block32_prefill_paged,
        "Run causal prefix prefill over block-scaled INT8 K/V");
}
