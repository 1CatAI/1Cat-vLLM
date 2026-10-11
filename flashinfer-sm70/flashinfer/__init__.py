"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from importlib import import_module as _import_module

from .version import __version__, __git_commit__, __git_version__

_LAZY_EXPORTS = {
    "ActivationType": (".tllm_enums", "ActivationType"),
    "B12xMoEWrapper": (".fused_moe", "B12xMoEWrapper"),
    "BatchAttention": (".attention", "BatchAttention"),
    "BatchAttentionWithAttentionSinkWrapper": (
        ".attention",
        "BatchAttentionWithAttentionSinkWrapper",
    ),
    "BatchDecodeWithPagedKVCacheWrapper": (
        ".decode",
        "BatchDecodeWithPagedKVCacheWrapper",
    ),
    "BatchDecodeWithSharedPrefixPagedKVCacheWrapper": (
        ".cascade",
        "BatchDecodeWithSharedPrefixPagedKVCacheWrapper",
    ),
    "BatchMLAPagedAttentionWrapper": (".mla", "BatchMLAPagedAttentionWrapper"),
    "BatchPODWithPagedKVCacheWrapper": (".pod", "BatchPODWithPagedKVCacheWrapper"),
    "BatchPrefillWithCausalBidirectionalRangesWrapper": (
        ".attention",
        "BatchPrefillWithCausalBidirectionalRangesWrapper",
    ),
    "BatchPrefillWithPagedKVCacheWrapper": (
        ".prefill",
        "BatchPrefillWithPagedKVCacheWrapper",
    ),
    "BatchPrefillWithRaggedKVCacheWrapper": (
        ".prefill",
        "BatchPrefillWithRaggedKVCacheWrapper",
    ),
    "BatchPrefillWithSharedPrefixPagedKVCacheWrapper": (
        ".cascade",
        "BatchPrefillWithSharedPrefixPagedKVCacheWrapper",
    ),
    "BlockSparseAttentionWrapper": (".sparse", "BlockSparseAttentionWrapper"),
    "CUDAGraphBatchDecodeWithPagedKVCacheWrapper": (
        ".decode",
        "CUDAGraphBatchDecodeWithPagedKVCacheWrapper",
    ),
    "CakeFmhaRequestOrderedDecodePlan": (
        ".cake_fmha",
        "CakeFmhaRequestOrderedDecodePlan",
    ),
    "CakeRMSNormFunction": (".cake_rmsnorm_train", "CakeRMSNormFunction"),
    "CuteDslBf16MoEWrapper": (".fused_moe", "CuteDslBf16MoEWrapper"),
    "CuteDslMoEWrapper": (".fused_moe", "CuteDslMoEWrapper"),
    "CuteDslMxfp8Mxfp4MoEWrapper": (".fused_moe", "CuteDslMxfp8Mxfp4MoEWrapper"),
    "KDAPrefillPlanCache": (".kda_prefill", "KDAPrefillPlanCache"),
    "MeasurementPolicy": (".autotune_cache", "MeasurementPolicy"),
    "MiniMaxH3Mxfp8PreAttention": (".cake_minimax_h3", "MiniMaxH3Mxfp8PreAttention"),
    "MiniMaxH3Nvfp4PreAttention": (".cake_minimax_h3", "MiniMaxH3Nvfp4PreAttention"),
    "MiniMaxH3QkvQuantizePack": (".cake_minimax_h3", "MiniMaxH3QkvQuantizePack"),
    "MultiLevelCascadeAttentionWrapper": (
        ".cascade",
        "MultiLevelCascadeAttentionWrapper",
    ),
    "NVFP44Over6Config": (
        ".quantization.nvfp4_quantization_utils",
        "NVFP44Over6Config",
    ),
    "NVFP44Over6ErrMode": (
        ".quantization.nvfp4_quantization_utils",
        "NVFP44Over6ErrMode",
    ),
    "PODWithPagedKVCacheWrapper": (".pod", "PODWithPagedKVCacheWrapper"),
    "RecurrentKDAPrefillWorkspace": (".kda_prefill", "RecurrentKDAPrefillWorkspace"),
    "RecurrentKDAPrefillWrapper": (".kda", "RecurrentKDAPrefillWrapper"),
    "RoutingMethodType": (".tllm_enums", "RoutingMethodType"),
    "SegmentGEMMWrapper": (".gemm", "SegmentGEMMWrapper"),
    "SfLayout": (".tllm_enums", "SfLayout"),
    "TopKTieBreak": (".topk", "TopKTieBreak"),
    "VariableBlockSparseAttentionWrapper": (
        ".sparse",
        "VariableBlockSparseAttentionWrapper",
    ),
    "add_rmsnorm_fp4quant": (".norm", "add_rmsnorm_fp4quant"),
    "append_paged_kv_cache": (".page", "append_paged_kv_cache"),
    "append_paged_mla_kv_cache": (".page", "append_paged_mla_kv_cache"),
    "apply_llama31_rope": (".rope", "apply_llama31_rope"),
    "apply_llama31_rope_inplace": (".rope", "apply_llama31_rope_inplace"),
    "apply_llama31_rope_pos_ids": (".rope", "apply_llama31_rope_pos_ids"),
    "apply_llama31_rope_pos_ids_inplace": (
        ".rope",
        "apply_llama31_rope_pos_ids_inplace",
    ),
    "apply_rope": (".rope", "apply_rope"),
    "apply_rope_inplace": (".rope", "apply_rope_inplace"),
    "apply_rope_pos_ids": (".rope", "apply_rope_pos_ids"),
    "apply_rope_pos_ids_inplace": (".rope", "apply_rope_pos_ids_inplace"),
    "apply_rope_with_cos_sin_cache": (".rope", "apply_rope_with_cos_sin_cache"),
    "apply_rope_with_cos_sin_cache_inplace": (
        ".rope",
        "apply_rope_with_cos_sin_cache_inplace",
    ),
    "autotune": (".autotuner", "autotune"),
    "autotune_v2": (".autotune_cache", "autotune_v2"),
    "autotune_v2_reload": (".autotune_cache", "autotune_v2_reload"),
    "b12x_fused_moe": (".fused_moe", "b12x_fused_moe"),
    "block_scale_interleave": (
        ".quantization.fp4_quantization",
        "block_scale_interleave",
    ),
    "bmm_bf16": (".gemm", "bmm_bf16"),
    "bmm_fp8": (".gemm", "bmm_fp8"),
    "bmm_mxfp8": (".gemm", "bmm_mxfp8"),
    "cake_batch_context_with_kv_cache": (
        ".cake_fmha",
        "cake_batch_context_with_kv_cache",
    ),
    "cake_batch_decode_with_kv_cache": (
        ".cake_fmha",
        "cake_batch_decode_with_kv_cache",
    ),
    "cake_fmha_manifest": (".cake_fmha", "cake_fmha_manifest"),
    "cake_rmsnorm": (".cake_rmsnorm_train", "cake_rmsnorm"),
    "cake_rmsnorm_train_backward": (
        ".cake_rmsnorm_train",
        "cake_rmsnorm_train_backward",
    ),
    "cake_rmsnorm_train_backward_workspace_bytes": (
        ".cake_rmsnorm_train",
        "cake_rmsnorm_train_backward_workspace_bytes",
    ),
    "cake_rmsnorm_train_forward": (".cake_rmsnorm_train", "cake_rmsnorm_train_forward"),
    "cake_top_k_top_p_sampling_from_probs": (
        ".cake_sampling",
        "top_k_top_p_sampling_from_probs",
    ),
    "chain_speculative_sampling": (".sampling", "chain_speculative_sampling"),
    "chunk_gated_delta_product": (".gdp_prefill", "chunk_gated_delta_product"),
    "chunk_gated_delta_rule": (".gdn_prefill", "chunk_gated_delta_rule"),
    "chunk_gated_delta_rule2": (".gdn2_prefill", "chunk_gated_delta_rule2"),
    "compute_paged_mqa_logits_schedule": (
        ".attn_scores",
        "compute_paged_mqa_logits_schedule",
    ),
    "concat_mla_kv_quant_fp8": (".mla_kv_pack", "concat_mla_kv_quant_fp8"),
    "cudnn_batch_decode_with_kv_cache": (".decode", "cudnn_batch_decode_with_kv_cache"),
    "cudnn_frost_grouped_gemm1_swiglu": (
        ".fused_moe",
        "cudnn_frost_grouped_gemm1_swiglu",
    ),
    "cudnn_frost_grouped_gemm1_swiglu_workspace_size": (
        ".fused_moe",
        "cudnn_frost_grouped_gemm1_swiglu_workspace_size",
    ),
    "cute_dsl_fused_moe": (".fused_moe", "cute_dsl_fused_moe"),
    "cute_dsl_fused_moe_bf16": (".fused_moe", "cute_dsl_fused_moe_bf16"),
    "cute_dsl_fused_moe_mxfp8_mxfp4": (".fused_moe", "cute_dsl_fused_moe_mxfp8_mxfp4"),
    "cute_dsl_fused_moe_nvfp4": (".fused_moe", "cute_dsl_fused_moe_nvfp4"),
    "cutlass_fused_moe": (".fused_moe", "cutlass_fused_moe"),
    "e2m1_and_ufp8sf_scale_to_float": (
        ".quantization.fp4_quantization",
        "e2m1_and_ufp8sf_scale_to_float",
    ),
    "fast_decode_plan": (".decode", "fast_decode_plan"),
    "fi_trace": (".fi_trace", "fi_trace"),
    "fp4_paged_mqa_logits": (".attn_scores", "fp4_paged_mqa_logits"),
    "fp4_quantize": (".quantization.fp4_quantization", "fp4_quantize"),
    "fp8_paged_mqa_logits": (".attn_scores", "fp8_paged_mqa_logits"),
    "fused_add_rmsnorm": (".norm", "fused_add_rmsnorm"),
    "fused_add_rmsnorm_fp8_block_quant": (".norm", "fused_add_rmsnorm_fp8_block_quant"),
    "fused_add_rmsnorm_quant": (".norm", "fused_add_rmsnorm_quant"),
    "fused_dit_gate_residual_layernorm_gamma_beta": (
        ".norm",
        "fused_dit_gate_residual_layernorm_gamma_beta",
    ),
    "fused_dit_gate_residual_layernorm_scale_shift": (
        ".norm",
        "fused_dit_gate_residual_layernorm_scale_shift",
    ),
    "fused_dit_residual_layernorm_scale_shift": (
        ".norm",
        "fused_dit_residual_layernorm_scale_shift",
    ),
    "fused_kda_decode": (".kda_decode", "fused_kda_decode"),
    "fused_qk_rmsnorm_rope": (".norm", "fused_qk_rmsnorm_rope"),
    "fused_rmsnorm_silu": (".norm", "fused_rmsnorm_silu"),
    "gdn_fused_decode_step": (".gdn_kernels.experimental", "gdn_fused_decode_step"),
    "gdn_fused_decode_step_supported": (
        ".gdn_kernels.experimental",
        "gdn_fused_decode_step_supported",
    ),
    "gelu_and_mul": (".activation", "gelu_and_mul"),
    "gelu_tanh_and_mul": (".activation", "gelu_tanh_and_mul"),
    "gemma_fused_add_rmsnorm": (".norm", "gemma_fused_add_rmsnorm"),
    "gemma_rmsnorm": (".norm", "gemma_rmsnorm"),
    "get_batch_indices_positions": (".page", "get_batch_indices_positions"),
    "get_dcp_spec_counter_bytes": (".cake_dcp", "get_dcp_spec_counter_bytes"),
    "get_dcp_spec_workspace_size_bytes": (
        ".cake_dcp",
        "get_dcp_spec_workspace_size_bytes",
    ),
    "get_fp4_quantization_module": (
        ".quantization.fp4_quantization",
        "get_fp4_quantization_module",
    ),
    "get_seq_lens": (".page", "get_seq_lens"),
    "grouped_mm_bf16": (".grouped_mm", "grouped_mm_bf16"),
    "grouped_mm_fp4": (".grouped_mm", "grouped_mm_fp4"),
    "grouped_mm_fp8": (".grouped_mm", "grouped_mm_fp8"),
    "grouped_mm_mxfp8": (".grouped_mm", "grouped_mm_mxfp8"),
    "is_gated_activation": (".tllm_enums", "is_gated_activation"),
    "jit": (".jit", None),
    "kda_prefill_supports_fp32_checkpoints": (
        ".kda_prefill",
        "kda_prefill_supports_fp32_checkpoints",
    ),
    "launch_sm110_gqa_decode_prepared": (".decode", "launch_sm110_gqa_decode_prepared"),
    "layernorm": (".norm", "layernorm"),
    "layernorm_quant": (".norm", "layernorm_quant"),
    "make_nvfp4_global_scale": (
        ".quantization.nvfp4_quantization_utils",
        "make_nvfp4_global_scale",
    ),
    "mamba": (".mamba", None),
    "merge_state": (".cascade", "merge_state"),
    "merge_state_in_place": (".cascade", "merge_state_in_place"),
    "merge_states": (".cascade", "merge_states"),
    "mhc": (".mhc", None),
    "min_block_table_width": (".attn_scores", "min_block_table_width"),
    "min_p_sampling_from_probs": (".sampling", "min_p_sampling_from_probs"),
    "mm_bf16": (".gemm", "mm_bf16"),
    "mm_bf16_fp4": (".gemm", "mm_bf16_fp4"),
    "mm_fp4": (".gemm", "mm_fp4"),
    "mm_fp8": (".gemm", "mm_fp8"),
    "mm_mxfp8": (".gemm", "mm_mxfp8"),
    "mm_nvfp4_svdquant": (".gemm", "mm_nvfp4_svdquant"),
    "msa_ops": (".msa_ops", None),
    "mxfp4_dequantize": (".quantization.fp4_quantization", "mxfp4_dequantize"),
    "mxfp4_dequantize_host": (
        ".quantization.fp4_quantization",
        "mxfp4_dequantize_host",
    ),
    "mxfp4_quantize": (".quantization.fp4_quantization", "mxfp4_quantize"),
    "mxfp8_dequantize_host": (
        ".quantization.fp8_quantization",
        "mxfp8_dequantize_host",
    ),
    "mxfp8_grouped_quantize": (
        ".quantization.fp8_quantization",
        "mxfp8_grouped_quantize",
    ),
    "mxfp8_quantize": (".quantization.fp8_quantization", "mxfp8_quantize"),
    "next_positive_power_of_2": (".utils", "next_positive_power_of_2"),
    "nvfp4_attention_sm120": (".nvfp4_attention_sm120", None),
    "nvfp4_attention_sm120_fwd": (
        ".nvfp4_attention_sm120",
        "nvfp4_attention_sm120_fwd",
    ),
    "nvfp4_attention_sm120_quantize_qkv": (
        ".nvfp4_attention_sm120",
        "nvfp4_attention_sm120_quantize_qkv",
    ),
    "nvfp4_batched_quantize": (
        ".quantization.fp4_quantization",
        "nvfp4_batched_quantize",
    ),
    "nvfp4_block_scale_interleave": (
        ".quantization.fp4_quantization",
        "nvfp4_block_scale_interleave",
    ),
    "nvfp4_kv_dequantize": (".quantization.fp4_quantization", "nvfp4_kv_dequantize"),
    "nvfp4_kv_dequantize_paged": (
        ".quantization.fp4_quantization",
        "nvfp4_kv_dequantize_paged",
    ),
    "nvfp4_kv_quantize": (".quantization.fp4_quantization", "nvfp4_kv_quantize"),
    "nvfp4_quantize": (".quantization.fp4_quantization", "nvfp4_quantize"),
    "nvfp4_quantize_append_paged_kv_cache": (
        ".page",
        "nvfp4_quantize_append_paged_kv_cache",
    ),
    "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping": (
        ".page",
        "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping",
    ),
    "nvfp4_quantize_append_paged_mla_kv_cache": (
        ".page",
        "nvfp4_quantize_append_paged_mla_kv_cache",
    ),
    "nvfp4_quantize_paged_kv_cache": (
        ".quantization.fp4_quantization",
        "nvfp4_quantize_paged_kv_cache",
    ),
    "nvfp4_quantize_smooth": (".gemm", "nvfp4_quantize_smooth"),
    "packbits": (".quantization", "packbits"),
    "packed_fused_kda_decode": (".kda_decode", "packed_fused_kda_decode"),
    "packed_kda_decode": (".kda_decode", "packed_kda_decode"),
    "padded_seq_len": (".attn_scores", "padded_seq_len"),
    "plan_cake_fmha_request_ordered_paged_decode": (
        ".cake_fmha",
        "plan_cake_fmha_request_ordered_paged_decode",
    ),
    "precompile_paged_mqa_logits": (".attn_scores", "precompile_paged_mqa_logits"),
    "prepare_balanced_batch_decode_with_kv_cache": (
        ".decode",
        "prepare_balanced_batch_decode_with_kv_cache",
    ),
    "prepare_bf16_fp4_weights": (".gemm", "prepare_bf16_fp4_weights"),
    "prepare_bf16_kda_prefill": (".kda_prefill", "prepare_bf16_kda_prefill"),
    "prepare_low_latency_gemm_weights": (
        ".trtllm_low_latency_gemm",
        "prepare_low_latency_gemm_weights",
    ),
    "prepare_sm110_gqa_decode": (".decode", "prepare_sm110_gqa_decode"),
    "prepare_tf32_kda_prefill": (".kda_prefill", "prepare_tf32_kda_prefill"),
    "prims_ts_bf16_moe": (".fused_moe", "prims_ts_bf16_moe"),
    "prims_ts_bf16_routed_moe": (".fused_moe", "prims_ts_bf16_routed_moe"),
    "prims_ts_fp4_block_scale_moe": (".fused_moe", "prims_ts_fp4_block_scale_moe"),
    "prims_ts_fp4_block_scale_routed_moe": (
        ".fused_moe",
        "prims_ts_fp4_block_scale_routed_moe",
    ),
    "prims_ts_fp8_block_scale_moe": (".fused_moe", "prims_ts_fp8_block_scale_moe"),
    "prims_ts_fp8_block_scale_routed_moe": (
        ".fused_moe",
        "prims_ts_fp8_block_scale_routed_moe",
    ),
    "prims_ts_fp8_per_tensor_scale_moe": (
        ".fused_moe",
        "prims_ts_fp8_per_tensor_scale_moe",
    ),
    "recurrent_kda": (".kda", "recurrent_kda"),
    "reorder_rows_for_gated_act_gemm": (
        ".fused_moe",
        "reorder_rows_for_gated_act_gemm",
    ),
    "resolve_nvfp4_4over6": (
        ".quantization.nvfp4_quantization_utils",
        "resolve_nvfp4_4over6",
    ),
    "rmsnorm": (".norm", "rmsnorm"),
    "rmsnorm_fp4quant": (".norm", "rmsnorm_fp4quant"),
    "rmsnorm_quant": (".norm", "rmsnorm_quant"),
    "sampling_from_logits": (".sampling", "sampling_from_logits"),
    "sampling_from_probs": (".sampling", "sampling_from_probs"),
    "scaled_fp4_grouped_quantize": (
        ".quantization.fp4_quantization",
        "scaled_fp4_grouped_quantize",
    ),
    "segment_packbits": (".quantization", "segment_packbits"),
    "shuffle_matrix_a": (".quantization.fp4_quantization", "shuffle_matrix_a"),
    "shuffle_matrix_sf_a": (".quantization.fp4_quantization", "shuffle_matrix_sf_a"),
    "silu_and_mul": (".activation", "silu_and_mul"),
    "silu_and_mul_mxfp8_quantize": (".gated_act_mxfp8", "silu_and_mul_mxfp8_quantize"),
    "silu_and_mul_mxfp8_quantize_backward": (
        ".gated_act_mxfp8",
        "silu_and_mul_mxfp8_quantize_backward",
    ),
    "silu_and_mul_nvfp4_quantize": (
        ".quantization.fp4_quantization",
        "silu_and_mul_nvfp4_quantize",
    ),
    "silu_and_mul_scaled_nvfp4_experts_quantize": (
        ".activation",
        "silu_and_mul_scaled_nvfp4_experts_quantize",
    ),
    "single_decode_with_kv_cache": (".decode", "single_decode_with_kv_cache"),
    "single_prefill_with_kv_cache": (".prefill", "single_prefill_with_kv_cache"),
    "single_prefill_with_kv_cache_return_lse": (
        ".prefill",
        "single_prefill_with_kv_cache_return_lse",
    ),
    "sm110_gqa_decode": (".decode", "sm110_gqa_decode"),
    "sm70": (".sm70", None),
    "softmax": (".sampling", "softmax"),
    "svdquant_linear": (".gemm", "svdquant_linear"),
    "tgv_gemm_sm100": (".gemm", "tgv_gemm_sm100"),
    "top_k": (".topk", "top_k"),
    "top_k_mask_logits": (".sampling", "top_k_mask_logits"),
    "top_k_page_table_transform": (".topk", "top_k_page_table_transform"),
    "top_k_ragged_transform": (".topk", "top_k_ragged_transform"),
    "top_k_renorm_probs": (".sampling", "top_k_renorm_probs"),
    "top_k_sampling_from_probs": (".sampling", "top_k_sampling_from_probs"),
    "top_k_top_p_sampling_from_logits": (
        ".sampling",
        "top_k_top_p_sampling_from_logits",
    ),
    "top_k_top_p_sampling_from_probs": (".sampling", "top_k_top_p_sampling_from_probs"),
    "top_k_varlen": (".topk_varlen.topk_varlen", "top_k_varlen"),
    "top_p_renorm_probs": (".sampling", "top_p_renorm_probs"),
    "top_p_sampling_from_probs": (".sampling", "top_p_sampling_from_probs"),
    "topk": (".topk", None),
    "trace_apply": (".trace_apply", None),
    "trtllm_bf16_moe": (".fused_moe", "trtllm_bf16_moe"),
    "trtllm_bf16_routed_moe": (".fused_moe", "trtllm_bf16_routed_moe"),
    "trtllm_fmha_v2_prefill": (".prefill", "trtllm_fmha_v2_prefill"),
    "trtllm_fp4_block_scale_moe": (".fused_moe", "trtllm_fp4_block_scale_moe"),
    "trtllm_fp4_block_scale_routed_moe": (
        ".fused_moe",
        "trtllm_fp4_block_scale_routed_moe",
    ),
    "trtllm_fp8_block_scale_moe": (".fused_moe", "trtllm_fp8_block_scale_moe"),
    "trtllm_fp8_block_scale_routed_moe": (
        ".fused_moe",
        "trtllm_fp8_block_scale_routed_moe",
    ),
    "trtllm_fp8_per_tensor_scale_moe": (
        ".fused_moe",
        "trtllm_fp8_per_tensor_scale_moe",
    ),
    "trtllm_fp8_per_tensor_scale_routed_moe": (
        ".fused_moe",
        "trtllm_fp8_per_tensor_scale_routed_moe",
    ),
    "trtllm_sage_attention_quantize": (".prefill", "trtllm_sage_attention_quantize"),
    "xqa": (".xqa", "xqa"),
    "xqa_mla": (".xqa", "xqa_mla"),
}


def __getattr__(name):
    try:
        module, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    imported = _import_module(module, __name__)
    value = getattr(imported, attr) if attr else imported
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


# ---------------------------------------------------------------------------
# Trace Apply (opt-in): zero-code kernel substitution from the FlashInfer Trace.
# Activated only when FLASHINFER_TRACE_APPLY=1. This is the single edit to
# existing flashinfer code required by the Trace Apply design — the package
# itself lives in flashinfer/trace_apply/. Failures here must never break a
# normal import, so the install is best-effort.
# ---------------------------------------------------------------------------
import os as _os

if _os.environ.get("FLASHINFER_TRACE_APPLY", "0") not in ("0", "", "false", "False"):
    try:
        from . import trace_apply as trace_apply

        trace_apply._enable_apply_from_env()
    except Exception as _trace_apply_err:  # noqa: BLE001
        import logging as _logging

        _logging.getLogger("flashinfer.trace_apply").warning(
            "FLASHINFER_TRACE_APPLY is set but enabling Trace Apply failed: %s "
            "(continuing without Trace Apply).",
            _trace_apply_err,
        )


# ---------------------------------------------------------------------------
# Import-time version log: emit one line when FLASHINFER_LOGLEVEL >= 1 so
# that crash logs contain the exact commit without any manual archaeology.
# Respects FLASHINFER_LOGDEST (stdout / stderr / filepath) the same way
# api_logging.py does; defaults to stdout.
# ---------------------------------------------------------------------------
def _log_import_version() -> None:
    # Wrapped in a private function so no temp variables leak into the
    # flashinfer module namespace.  Two-level protection:
    #   inner try  – safely resolve the commit hash; falls back to "unknown"
    #                if __git_commit__ is missing or malformed so the log
    #                line is still emitted rather than silently suppressed.
    #   outer try  – absorbs every other failure (non-integer LOGLEVEL env
    #                var, closed/None stdout or stderr, unwritable log file)
    #                so a logging misconfiguration can never block the import.
    try:
        if int(_os.environ.get("FLASHINFER_LOGLEVEL", "0")) < 1:
            return
        try:
            _short = __git_commit__[:8] if __git_commit__ != "unknown" else "unknown"
        except Exception:
            _short = "unknown"
        _line = f"FlashInfer {__version__} (commit {_short})\n"
        _dest = _os.environ.get("FLASHINFER_LOGDEST", "stdout").replace(
            "%i", str(_os.getpid())
        )
        if _dest == "stderr":
            import sys as _sys

            _sys.stderr.write(_line)
            _sys.stderr.flush()
        elif _dest not in ("stdout", ""):
            with open(_dest, "a") as _f:
                _f.write(_line)
        else:
            print(_line, end="", flush=True)
    except Exception:
        pass  # never let import-time logging crash the import


_log_import_version()
del _log_import_version
