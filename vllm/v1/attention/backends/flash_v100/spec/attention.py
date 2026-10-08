# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature policies registered at the Flash-V100 attention boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100 import state as _state
from vllm.v1.attention.backends.flash_v100.spec import tree_masks
from vllm.v1.attention.backends.flash_v100.spec.attention_policy import (
    POLICY_FIELDS as POLICY_FIELDS,
)
from vllm.v1.attention.backends.flash_v100.spec.attention_policy import (
    SpecAttentionState as SpecAttentionState,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
)
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    KVCodec,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def reject_xqa(codec: KVCodec | None, attn_metadata: TritonAttentionMetadata) -> bool:
    return codec is FP8_E4M3 and getattr(
        attn_metadata, "is_dflash_selector_target", False
    )


def fallback_kind(layer_info: dict[str, Any]) -> bool:
    return bool(layer_info.get("is_dflash_draft_attn"))


def unsupported(
    policy: Any,
    layer_info: dict[str, Any],
    message: str,
    is_dflash_draft_attn: bool,
) -> None:
    if not (policy.allow_triton_fallback or is_dflash_draft_attn):
        raise RuntimeError(message)
    if policy.use_flash_v100 and not _state._warned_feature_fallback:
        if is_dflash_draft_attn:
            logger.warning(
                "FLASH_ATTN_V100 falling back to Triton for D-Flash "
                "draft attention layer %s because the SM70 Flash-V100 "
                "backend does not yet support this layer/config.",
                layer_info.get("layer_name"),
            )
        else:
            logger.warning("%s", message)
        _state._warned_feature_fallback = True
    _routing._record_route(
        "dflash_draft_triton_fallback"
        if is_dflash_draft_attn
        else "unsupported_triton_fallback"
    )


def validate_contract(
    validator: Callable,
    layer: torch.nn.Module,
    attn_metadata: TritonAttentionMetadata,
) -> None:
    validator(layer, attn_metadata)


def capture_prefix_kind(
    layer: torch.nn.Module, attn_metadata: TritonAttentionMetadata
) -> bool:
    return bool(getattr(layer, "is_dflash_draft_attn", False)) and (
        not bool(getattr(attn_metadata, "causal", True))
    )


def record_capture_prefix() -> None:
    # DFlash pre-inserts target context K/V before replay. Its
    # dummy capture has seq_len == query_len and would
    # otherwise freeze the no-prefix dense branch into the
    # graph. Bind directly to the non-causal paged-prefix
    # kernel; runtime updates its persistent sequence and
    # block-table buffers before every replay.
    _routing._record_route(
        _routing.ROUTE_SPECS["prefill_capture_dflash_noncausal_paged"].name
    )


def record_capture_layout(attn_metadata: TritonAttentionMetadata) -> None:
    if getattr(attn_metadata, "ddtree_parent_ids", None) is None:
        _routing._record_route(
            _routing.ROUTE_SPECS["prefill_capture_smallq_no_ddtree_metadata"].name
        )
    else:
        _routing._record_route(
            _routing.ROUTE_SPECS["prefill_capture_smallq_ddtree_metadata"].name
        )


# Compatibility names stay at the feature boundary, not in common assembly.
VERIFICATION_CONFIG_FIELDS = {
    "grouped_max_query": ("dflash2_grouped_verify_max_query_tokens", 0),
    "grouped_request_major_abi": (
        "dflash2_grouped_verify_request_major_abi_version",
        0,
    ),
    "grouped_min_model_len": ("dflash2_grouped_verify_min_model_len", 0),
    "grouped_enabled": ("use_dflash2_grouped_verify", False),
    "grouped_batch_enabled": ("use_dflash2_batched_grouped_verify", False),
}
VERIFICATION_OVERRIDES = {
    "admit_grouped_override": "_dflash2_grouped_verify_allowed",
    "run_grouped_override": "_call_dflash2_grouped_verify",
    "admit_xqa_override": "_smallq_decode_xqa_allowed",
    "run_smallq_override": "_call_flash_attn_smallq_decode_paged",
}


VERIFICATION_METHODS = {
    "_validate_dflash_attention_contract": "validate_contract",
    "_dflash2_grouped_verify_allowed": "grouped_verify_allowed",
    "_call_dflash2_grouped_verify": "call_grouped_verify",
    "_smallq_decode_xqa_allowed": "smallq_xqa_allowed",
    "_call_flash_attn_smallq_decode_paged": "call_smallq_decode_paged",
    "_small_query_decode_enabled": "small_query_enabled",
    "_flash_v100_ddtree_small_query_prefill_dense": "tree_prefill",
    "_flash_v100_small_query_prefill_as_decode": "small_query_prefill",
}
VALIDATION_METHOD = "_validate_dflash_attention_contract"


PREFILL_CALLBACK_FIELDS = {
    "tree_prefill": "_flash_v100_ddtree_small_query_prefill_dense",
    "small_query": "_flash_v100_small_query_prefill_as_decode",
}


def prefill_dependencies(state):
    return {
        "tree_requires_branch": tree_masks.parent_metadata_requires_branch,
        "supports_bmhd": getattr(
            state, "_flash_prefill_paged_supports_dflash2_bmhd", False
        ),
        "split_pages": getattr(state, "_flash_prefill_paged_dflash2_split_pages", ()),
    }


def verification_dependencies():
    return {
        "tree_seq_lens_match": tree_masks.triton_seq_lens_match,
        "tree_query_start_match": tree_masks.triton_query_start_loc_match,
        "tree_parent_ids": tree_masks.triton_parent_ids_for_query,
        "tree_visibility": tree_masks.build_visibility_mask,
    }
