# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature policies registered at the Flash-V100 attention boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import impl as _impl
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100 import state as _state
from vllm.v1.attention.backends.flash_v100 import verify as _verify
from vllm.v1.attention.backends.flash_v100.spec.attention_policy import (
    configure_prefill as configure_prefill,
)
from vllm.v1.attention.backends.flash_v100.spec.attention_policy import (
    configure_verifier as configure_verifier,
)
from vllm.v1.attention.backends.flash_v100.spec.attention_policy import (
    initialize_scalar_tail as initialize_scalar_tail,
)
from vllm.v1.attention.backends.flash_v100.spec.attention_policy import (
    initialize_verify_abi as initialize_verify_abi,
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
    self: _impl.FlashAttnV100Impl,
    layer_info: dict[str, Any],
    message: str,
    is_dflash_draft_attn: bool,
) -> None:
    if not (self.allow_triton_fallback or is_dflash_draft_attn):
        raise RuntimeError(message)
    if self.use_flash_v100 and not _state._warned_feature_fallback:
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
    self: _impl.FlashAttnV100Impl,
    layer: torch.nn.Module,
    attn_metadata: TritonAttentionMetadata,
) -> None:
    self._validate_dflash_attention_contract(layer, attn_metadata)


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


class SpecAttentionMethods:
    _sm70_scalar_tail_attention: Any
    dflash2_grouped_verify_max_query_tokens: int
    dflash2_grouped_verify_request_major_abi_version: int
    _flash_prefill_paged_supports_dflash2_bmhd: bool
    _flash_prefill_paged_dflash2_split_pages: tuple[int, ...]
    use_dflash2_grouped_verify: bool
    use_dflash2_batched_grouped_verify: bool
    dflash2_grouped_verify_min_model_len: int

    _validate_dflash_attention_contract = _verify._validate_dflash_attention_contract
    _dflash2_grouped_verify_allowed = _verify._dflash2_grouped_verify_allowed
    _call_dflash2_grouped_verify = _verify._call_dflash2_grouped_verify
    _flash_v100_ddtree_small_query_prefill_dense = (
        _verify._flash_v100_ddtree_small_query_prefill_dense
    )


@dataclass(frozen=True)
class AttentionHooks:
    initialize_scalar_tail: Callable[[_impl.FlashAttnV100Impl, bool], None]
    initialize_verify_abi: Callable[[_impl.FlashAttnV100Impl], None]
    configure_prefill: Callable[[_impl.FlashAttnV100Impl], None]
    configure_verifier: Callable[[_impl.FlashAttnV100Impl], None]
    reject_xqa: Callable[[KVCodec | None, TritonAttentionMetadata], bool]
    fallback_kind: Callable[[dict[str, Any]], bool]
    unsupported: Callable[[_impl.FlashAttnV100Impl, dict[str, Any], str, bool], None]
    validate_contract: Callable[
        [_impl.FlashAttnV100Impl, torch.nn.Module, TritonAttentionMetadata], None
    ]
    capture_prefix_kind: Callable[[torch.nn.Module, TritonAttentionMetadata], bool]
    record_capture_prefix: Callable[[], None]
    record_capture_layout: Callable[[TritonAttentionMetadata], None]


def register_attention_hooks() -> AttentionHooks:
    return AttentionHooks(
        initialize_scalar_tail=initialize_scalar_tail,
        initialize_verify_abi=initialize_verify_abi,
        configure_prefill=configure_prefill,
        configure_verifier=configure_verifier,
        reject_xqa=reject_xqa,
        fallback_kind=fallback_kind,
        unsupported=unsupported,
        validate_contract=validate_contract,
        capture_prefix_kind=capture_prefix_kind,
        record_capture_prefix=record_capture_prefix,
        record_capture_layout=record_capture_layout,
    )


ATTENTION_HOOKS = register_attention_hooks()


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
