# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inputs, callbacks and compatibility fields for speculative metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from vllm.v1.attention.backends.flash_v100.workspace import MetadataWorkspace


@dataclass(frozen=True)
class MetadataInputs:
    builder_id: int
    vllm_config: Any
    device: torch.device
    block_size: int
    is_draft: bool


@dataclass(frozen=True)
class MetadataOps:
    base_build: Any
    attach_common: Any
    attach_prefix: Any
    attach_shape_hints: Any
    update_active_partitions: Any


class SmallQueryBuilder(Protocol):
    vllm_config: Any
    block_size: int
    metadata_workspace: MetadataWorkspace
    _use_sm70_dflash2_fused_smallq_metadata: bool


def metadata_view(attn_metadata: Any) -> Any:
    # Owned packets are canonical. Older external/synthetic metadata retains
    # its historical duck-typed field interface at this compatibility boundary.
    return getattr(attn_metadata, "spec_state", attn_metadata)


STATE_FIELDS = frozenset(
    {
        "_is_dflash_draft_model",
        "_is_dflash_selector_target",
        "_use_sm70_dflash2_fused_smallq_metadata",
        "metadata_workspace",
    }
)
INPUT_FIELDS = {
    "vllm_config": "vllm_config",
    "device": "device",
    "block_size": "block_size",
    "_is_speculative_draft_model": "is_draft",
}


class SpecMetadataPacket:
    ddtree_parent_ids: torch.Tensor | None
    ddtree_parent_ids_cpu: torch.Tensor | None
    ddtree_num_tree_tokens_cpu: torch.Tensor | None
    ddtree_seq_lens_restored_for_triton: bool
    ddtree_query_start_loc_restored_for_triton: bool
    is_dflash_selector_target: bool
    smallq_decode_block_table: torch.Tensor | None
    smallq_decode_seq_lens: torch.Tensor | None
    smallq_query_start_loc: torch.Tensor | None
    smallq_decode_max_seq_len_hint: int | None
    smallq_decode_workspace_seq_capacity_hint: int | None
    smallq_decode_partition_size_hint: int | None


METADATA_FIELDS = frozenset(SpecMetadataPacket.__annotations__)
# The old typing-only name is external compatibility, never an inherited mixin.
SpecMetadataFields = SpecMetadataPacket
