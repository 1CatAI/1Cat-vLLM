"""
Copyright (c) 2024 by FlashInfer team.

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

_LAZY_EXPORTS = {
    "BLACKWELL_MSA_VARIANTS": (".blackwell_msa", "BLACKWELL_MSA_VARIANTS"),
    "BLACKWELL_MSA_VARIANTS_BY_TARGET": (
        ".blackwell_msa",
        "BLACKWELL_MSA_VARIANTS_BY_TARGET",
    ),
    "CakeWarpDecodeTarget": (".cake_fused_moe_warp_decode", "CakeWarpDecodeTarget"),
    "JitSpec": (".core", "JitSpec"),
    "JitSpecCuteDsl": (".cute_dsl_core", "JitSpecCuteDsl"),
    "JitSpecNvcc": (".core", "JitSpecNvcc"),
    "JitSpecRegistry": (".core", "JitSpecRegistry"),
    "JitSpecStatus": (".core", "JitSpecStatus"),
    "MissingJITCacheError": (".core", "MissingJITCacheError"),
    "build_jit_specs": (".core", "build_jit_specs"),
    "cake_kda_affine_is_available": (".cake_kda", "cake_kda_affine_is_available"),
    "clear_cache_dir": (".core", "clear_cache_dir"),
    "cubin_loader": (".cubin_loader", None),
    "current_compilation_context": (".core", "current_compilation_context"),
    "env": (".env", None),
    "gen_act_and_mul_module": (".activation", "gen_act_and_mul_module"),
    "gen_alphamoe_fused_router_module": (
        ".fused_moe",
        "gen_alphamoe_fused_router_module",
    ),
    "gen_alphamoe_nvfp4_sm100_module": (
        ".fused_moe",
        "gen_alphamoe_nvfp4_sm100_module",
    ),
    "gen_alphamoe_sm100_module": (".fused_moe", "gen_alphamoe_sm100_module"),
    "gen_batch_attention_module": (".attention", "gen_batch_attention_module"),
    "gen_batch_decode_module": (".attention", "gen_batch_decode_module"),
    "gen_batch_mla_module": (".attention", "gen_batch_mla_module"),
    "gen_batch_pod_module": (".attention", "gen_batch_pod_module"),
    "gen_batch_prefill_module": (".attention", "gen_batch_prefill_module"),
    "gen_bgmv_moe_module": (".bgmv_moe", "gen_bgmv_moe_module"),
    "gen_blackwell_msa_module": (".blackwell_msa", "gen_blackwell_msa_module"),
    "gen_cake_bgmv_moe_module": (".cake_bgmv_moe", "gen_cake_bgmv_moe_module"),
    "gen_cake_fused_moe_warp_decode_module": (
        ".cake_fused_moe_warp_decode",
        "gen_cake_fused_moe_warp_decode_module",
    ),
    "gen_cake_kda_affine_module": (".cake_kda", "gen_cake_kda_affine_module"),
    "gen_cake_kda_m128_unbounded_softplus_module": (
        ".cake_kda",
        "gen_cake_kda_m128_unbounded_softplus_module",
    ),
    "gen_cake_megamoe_topk_reduce_module": (
        ".cake_megamoe_topk_reduce",
        "gen_cake_megamoe_topk_reduce_module",
    ),
    "gen_cake_sampling_module": (".cake_sampling", "gen_cake_sampling_module"),
    "gen_comm_alltoall_module": (".comm", "gen_comm_alltoall_module"),
    "gen_cudnn_fmha_module": (".attention", "gen_cudnn_fmha_module"),
    "gen_customize_batch_decode_module": (
        ".attention",
        "gen_customize_batch_decode_module",
    ),
    "gen_customize_batch_prefill_module": (
        ".attention",
        "gen_customize_batch_prefill_module",
    ),
    "gen_customize_single_decode_module": (
        ".attention",
        "gen_customize_single_decode_module",
    ),
    "gen_customize_single_prefill_module": (
        ".attention",
        "gen_customize_single_prefill_module",
    ),
    "gen_dcp_alltoall_module": (".comm", "gen_dcp_alltoall_module"),
    "gen_dsv3_fused_routing_module": (
        ".dsv3_optimizations",
        "gen_dsv3_fused_routing_module",
    ),
    "gen_dsv3_router_gemm_module": (
        ".dsv3_optimizations",
        "gen_dsv3_router_gemm_module",
    ),
    "gen_flash_kda_m128_module": (".flash_kda", "gen_flash_kda_m128_module"),
    "gen_flash_kda_m128_n16_module": (".flash_kda", "gen_flash_kda_m128_n16_module"),
    "gen_flash_kda_m64_module": (".flash_kda", "gen_flash_kda_m64_module"),
    "gen_fmha_cutlass_sm100a_module": (".attention", "gen_fmha_cutlass_sm100a_module"),
    "gen_fmha_v2_module": (".attention", "gen_fmha_v2_module"),
    "gen_fp4_kv_dequantization_module": (
        ".fp4_kv_dequantization",
        "gen_fp4_kv_dequantization_module",
    ),
    "gen_fp4_kv_quantization_module": (
        ".fp4_kv_quantization",
        "gen_fp4_kv_quantization_module",
    ),
    "gen_hash_topk_module": (".hash_topk", "gen_hash_topk_module"),
    "gen_jit_spec": (".core", "gen_jit_spec"),
    "gen_moe_alltoall_module": (".comm", "gen_moe_alltoall_module"),
    "gen_moe_utils_module": (".moe_utils", "gen_moe_utils_module"),
    "gen_monomoe_module": (".monomoe", "gen_monomoe_module"),
    "gen_nvfp4_attention_sm120_module": (
        ".nvfp4_attention_sm120",
        "gen_nvfp4_attention_sm120_module",
    ),
    "gen_pcie_ipc_comm_module": (".comm", "gen_pcie_ipc_comm_module"),
    "gen_pod_module": (".attention", "gen_pod_module"),
    "gen_prims_ts_q_token_kv_block_sparse_metadata_module": (
        ".q_token_kv_block_sparse_metadata",
        "gen_prims_ts_q_token_kv_block_sparse_metadata_module",
    ),
    "gen_single_decode_module": (".attention", "gen_single_decode_module"),
    "gen_single_prefill_module": (".attention", "gen_single_prefill_module"),
    "gen_tinygemm2_module": (".tinygemm2", "gen_tinygemm2_module"),
    "gen_tinygemm2_sm100_module": (".tinygemm2", "gen_tinygemm2_sm100_module"),
    "gen_trtllm_comm_module": (".comm", "gen_trtllm_comm_module"),
    "gen_trtllm_fmha_v2_sm120_module": (
        ".attention",
        "gen_trtllm_fmha_v2_sm120_module",
    ),
    "gen_trtllm_gen_fmha_module": (".attention", "gen_trtllm_gen_fmha_module"),
    "gen_trtllm_mnnvl_comm_module": (".comm", "gen_trtllm_mnnvl_comm_module"),
    "gen_vllm_comm_module": (".comm", "gen_vllm_comm_module"),
    "get_act_and_mul_cu_str": (".activation", "get_act_and_mul_cu_str"),
    "get_batch_attention_uri": (".attention", "get_batch_attention_uri"),
    "get_batch_decode_uri": (".attention", "get_batch_decode_uri"),
    "get_batch_mla_uri": (".attention", "get_batch_mla_uri"),
    "get_batch_prefill_uri": (".attention", "get_batch_prefill_uri"),
    "get_blackwell_msa_module": (".blackwell_msa", "get_blackwell_msa_module"),
    "get_blackwell_msa_uri": (".blackwell_msa", "get_blackwell_msa_uri"),
    "get_cake_fused_moe_warp_decode_module": (
        ".cake_fused_moe_warp_decode",
        "get_cake_fused_moe_warp_decode_module",
    ),
    "get_cake_fused_moe_warp_decode_uri": (
        ".cake_fused_moe_warp_decode",
        "get_cake_fused_moe_warp_decode_uri",
    ),
    "get_cake_kda_affine_module": (".cake_kda", "get_cake_kda_affine_module"),
    "get_cake_kda_prefill_module": (".cake_kda", "get_cake_kda_prefill_module"),
    "get_cake_megamoe_topk_reduce_module": (
        ".cake_megamoe_topk_reduce",
        "get_cake_megamoe_topk_reduce_module",
    ),
    "get_cake_megamoe_topk_reduce_uri": (
        ".cake_megamoe_topk_reduce",
        "get_cake_megamoe_topk_reduce_uri",
    ),
    "get_cake_sampling_uri": (".cake_sampling", "get_cake_sampling_uri"),
    "get_flash_kda_prefill_module": (".flash_kda", "get_flash_kda_prefill_module"),
    "get_pod_uri": (".attention", "get_pod_uri"),
    "get_single_decode_uri": (".attention", "get_single_decode_uri"),
    "get_single_prefill_uri": (".attention", "get_single_prefill_uri"),
    "is_cake_megamoe_topk_reduce_module_loaded": (
        ".cake_megamoe_topk_reduce",
        "is_cake_megamoe_topk_reduce_module_loaded",
    ),
    "jit_spec_registry": (".core", "jit_spec_registry"),
    "load_bgmv_moe_module": (".bgmv_moe", "load_bgmv_moe_module"),
    "load_blackwell_msa_module": (".blackwell_msa", "load_blackwell_msa_module"),
    "load_cake_bgmv_moe_module": (".cake_bgmv_moe", "load_cake_bgmv_moe_module"),
    "load_cake_fused_moe_warp_decode_module": (
        ".cake_fused_moe_warp_decode",
        "load_cake_fused_moe_warp_decode_module",
    ),
    "load_cake_kda_m128_unbounded_softplus_module": (
        ".cake_kda",
        "load_cake_kda_m128_unbounded_softplus_module",
    ),
    "load_cake_megamoe_topk_reduce_module": (
        ".cake_megamoe_topk_reduce",
        "load_cake_megamoe_topk_reduce_module",
    ),
    "load_cake_sampling_module": (".cake_sampling", "load_cake_sampling_module"),
    "load_flash_kda_m128_module": (".flash_kda", "load_flash_kda_m128_module"),
    "load_flash_kda_m128_n16_module": (".flash_kda", "load_flash_kda_m128_n16_module"),
    "load_flash_kda_m64_module": (".flash_kda", "load_flash_kda_m64_module"),
    "load_monomoe_module": (".monomoe", "load_monomoe_module"),
    "run_cake_megamoe_topk_reduce": (
        ".cake_megamoe_topk_reduce",
        "run_cake_megamoe_topk_reduce",
    ),
    "setup_cubin_loader": (".cubin_loader", "setup_cubin_loader"),
    "sm100a_nvcc_flags": (".core", "sm100a_nvcc_flags"),
    "sm100f_nvcc_flags": (".core", "sm100f_nvcc_flags"),
    "sm103a_nvcc_flags": (".core", "sm103a_nvcc_flags"),
    "sm107a_nvcc_flags": (".core", "sm107a_nvcc_flags"),
    "sm110a_nvcc_flags": (".core", "sm110a_nvcc_flags"),
    "sm120a_nvcc_flags": (".core", "sm120a_nvcc_flags"),
    "sm120f_nvcc_flags": (".core", "sm120f_nvcc_flags"),
    "sm121a_nvcc_flags": (".core", "sm121a_nvcc_flags"),
    "sm90a_nvcc_flags": (".core", "sm90a_nvcc_flags"),
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
