# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PLE short-convolution metadata captured at model-state initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends import sm70_mtp_shortconv_meta as _sc_meta
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.sm70_mtp_shortconv_meta import (
    ShortConvGroupDescriptor as ShortConvGroupDescriptor,
)
from vllm.v1.worker.gpu.spec_decode import uses_dflash_selector_engine

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.attn_utils import AttentionGroup
    from vllm.v1.worker.gpu.input_batch import InputBatch


def shortconv_enabled(config, device, cache_mode) -> bool:
    spec = config.speculative_config
    mtp_on_device = (
        spec is not None
        and spec.method == "mtp"
        and device.type == "cuda"
        and current_platform.is_device_capability(70)
    )
    return bool(
        _sc_meta.enabled()
        and mtp_on_device
        and not uses_dflash_selector_engine(config)
        and cache_mode in ("none", "align")
    )


def prepare_shortconv_metadata(
    self: Any,
    input_batch: InputBatch,
    block_tables: tuple[torch.Tensor, ...],
    attn_groups: list[list[AttentionGroup]],
    query_start_loc_cpu: torch.Tensor,
    num_decode_draft_tokens_cpu: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_reqs: int,
    num_tokens: int,
    *,
    state_start_indices: torch.Tensor | None = None,
    req_index_mapping: torch.Tensor | None = None,
) -> dict[int, Any] | None:
    """ONECAT_MTP_SHORTCONV_GROUP_META for one FULL-graph batch: every PLE
    short-conv group's graph metadata in one launch, or None with a counted
    reason, which keeps the per-group path."""
    reason = _sc_meta.pure_spec_block_reason(
        num_decode_draft_tokens_cpu, query_start_loc_cpu, num_tokens
    )
    prepared_result = None
    if reason is None:
        num_spec_decodes = int((num_decode_draft_tokens_cpu >= 0).sum().item())
        builders = self._get_ple_shortconv_builders(attn_groups)
        if not builders:
            reason = "no_groups"
        else:
            prepared_result = _sc_meta.prepare_ple_shortconv_group_metadata(
                builders_by_group=builders,
                block_tables=block_tables,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=input_batch.query_start_loc,
                num_spec_decodes=num_spec_decodes,
                num_spec_decode_tokens=int(
                    query_start_loc_cpu[num_spec_decodes].item()
                ),
                num_reqs=num_reqs,
                descriptor=self._ple_shortconv_descriptor,
                state_start_indices=state_start_indices,
                req_index_mapping=req_index_mapping,
            )
            if prepared_result is None:
                reason = "prepare"
    if reason is not None or prepared_result is None:
        _sc_meta.note_route(f"fallback:{reason}")
        return None
    prepared, self._ple_shortconv_descriptor = prepared_result
    _sc_meta.note_route("fused")
    return prepared


class ShortConvMetadataProvider:
    """Per-engine builder cache and prepared descriptor for grouped metadata."""

    def __init__(self, config, device, cache_mode):
        self.enabled = shortconv_enabled(config, device, cache_mode)
        self._ple_shortconv_builders = None
        self._ple_shortconv_descriptor = None
        if self.enabled:
            init_logger(__name__).info_once(
                "ONECAT_MTP_SHORTCONV_GROUP_META: PLE short-conv group metadata in "
                "one launch for pure MTP verify batches."
            )

    def _get_ple_shortconv_builders(self, attn_groups):
        if self._ple_shortconv_builders is None:
            self._ple_shortconv_builders = [
                (group_id, builder)
                for group_id, groups in enumerate(attn_groups)
                for group in groups
                if isinstance(
                    builder := group.get_metadata_builder(0),
                    PleShortConvAttentionMetadataBuilder,
                )
            ]
        return self._ple_shortconv_builders

    def prepare(self, *args, **kwargs):
        return prepare_shortconv_metadata(self, *args, **kwargs)
