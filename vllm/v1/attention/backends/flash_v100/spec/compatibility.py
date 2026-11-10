# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Historical feature export names and process observation keys."""

LOG_KEYS = {
    "_logged_decode_dense_cache": "flash_v100._logged_decode_dense_cache",
    "_logged_decode_dense_reference": "flash_v100._logged_decode_dense_reference",
    "_logged_decode_flash": "flash_v100._logged_decode_flash",
    "_logged_decode_paged_prefill": "flash_v100._logged_decode_paged_prefill",
    "_logged_decode_paged_prefill_bhmd": "flash_v100._logged_decode_paged_prefill_bhmd",
    "_logged_decode_paged_prefill_bhmd_q_clone": (
        "flash_v100._logged_decode_paged_prefill_bhmd_q_clone"
    ),
    "_logged_decode_wmma_wrapper": "flash_v100._logged_decode_wmma_wrapper",
    "_warned_decode_fallback": "flash_v100._warned_decode_fallback",
    "_warned_decode_strict_fallback": "flash_v100._warned_decode_strict_fallback",
    "_warned_feature_fallback": "flash_v100._warned_feature_fallback",
    "_logged_prefill_flash": "flash_v100._logged_prefill_flash",
    "_logged_prefill_prefix_flash": "flash_v100._logged_prefill_prefix_flash",
    "_logged_prefill_prefix_contig_dense": (
        "flash_v100._logged_prefill_prefix_contig_dense"
    ),
    "_logged_prefill_prefix_bfla": "flash_v100._logged_prefill_prefix_bfla",
    "_logged_prefill_prefix_splitkv": "flash_v100._logged_prefill_prefix_splitkv",
    "_logged_prefill_paged_cache": "flash_v100._logged_prefill_paged_cache",
    "_logged_prefill_smallq_decode": "flash_v100._logged_prefill_smallq_decode",
    "_logged_prefill_prefix_decode_rows": (
        "flash_v100._logged_prefill_prefix_decode_rows"
    ),
    "_logged_prefill_prefix_decode_rows_grouped": (
        "flash_v100._logged_prefill_prefix_decode_rows_grouped"
    ),
    "_logged_prefill_smallq_decode_xqa": "flash_v100._logged_prefill_smallq_decode_xqa",
    "_logged_prefill_smallq_grouped_verify": (
        "flash_v100._logged_prefill_smallq_grouped_verify"
    ),
    "_logged_prefill_smallq_grouped_verify_gate": (
        "flash_v100._logged_prefill_smallq_grouped_verify_gate"
    ),
    "_logged_prefill_triton_safe": "flash_v100._logged_prefill_triton_safe",
    "_logged_fp8_prefill_bridge": "flash_v100._logged_fp8_prefill_bridge",
    "_logged_prefill_compare": "flash_v100._logged_prefill_compare",
    "_logged_dflash_prefix_dump": "flash_v100.prefix_dump",
    "_logged_prefill_ddtree_dense": "flash_v100.tree_dense",
    "_logged_prefill_ddtree_triton": "flash_v100.tree_paged",
    "_logged_prefill_ddtree_triton_fallback": "flash_v100.tree_fallback",
    "_warned_prefill_dense_splitkv3_oom": (
        "flash_v100._warned_prefill_dense_splitkv3_oom"
    ),
    "_warned_prefill_d256_gqa_architecture_oom": (
        "flash_v100._warned_prefill_d256_gqa_architecture_oom"
    ),
    "_logged_prefill_fa2_d256": "flash_v100._logged_prefill_fa2_d256",
    "_logged_prefill_dense_splitkv3": "flash_v100._logged_prefill_dense_splitkv3",
    "_logged_prefill_d256_gqa_architecture": (
        "flash_v100._logged_prefill_d256_gqa_architecture"
    ),
}

PUBLIC_EXPORTS = (
    "DFlash2SmallQGroupDescriptor",
    "DFlash2SmallQPreparedMetadata",
    "prepare_dflash2_smallq_group_metadata",
)

MASK_ALIASES = {"_ddtree_parent_ids_cpu": "parent_ids_cpu"}
