# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered speculative metadata hooks and compatibility method surface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.flash_v100 import workspace as _workspace
from vllm.v1.attention.backends.flash_v100.spec import (
    builder,
    draft,
    tree,
    verify_metadata,
)

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.flash_v100.metadata import (
        FlashAttnV100MetadataBuilder,
    )
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata


class SpecMetadataFields:
    ddtree_parent_ids: torch.Tensor | None
    ddtree_parent_ids_cpu: torch.Tensor | None
    ddtree_num_tree_tokens_cpu: torch.Tensor | None
    ddtree_seq_lens_restored_for_triton: bool
    ddtree_query_start_loc_restored_for_triton: bool
    is_dflash_selector_target: bool


class SpecMetadataMethods:
    _is_dflash_draft_model: bool
    _is_dflash_selector_target: bool
    _use_sm70_dflash2_fused_smallq_metadata: bool
    _draft_block_table: torch.Tensor | None
    _draft_seq_lens: torch.Tensor | None
    _draft_query_start_loc: torch.Tensor | None
    _flash_draft_buffer_shape: tuple[int, int] | None
    _smallq_decode_block_table: torch.Tensor | None
    _smallq_decode_seq_lens: torch.Tensor | None
    _smallq_query_start_loc: torch.Tensor | None
    _smallq_token_indices: torch.Tensor | None
    _smallq_buffer_shape: tuple[int, int, int] | None

    _attach_ddtree_metadata = tree._attach_ddtree_metadata
    _debug_draft_metadata = draft._debug_draft_metadata
    _ensure_flash_draft_graph_buffers = _workspace._ensure_flash_draft_graph_buffers
    _stabilize_draft_graph_metadata = draft._stabilize_draft_graph_metadata
    copy_dflash_graph_metadata = _workspace.copy_dflash_graph_metadata
    build_for_drafting = draft.build_for_drafting
    _configured_smallq_max_query_len = verify_metadata._configured_smallq_max_query_len
    _configured_smallq_max_model_len = verify_metadata._configured_smallq_max_model_len
    _smallq_buffer_token_capacity = verify_metadata._smallq_buffer_token_capacity
    _ensure_smallq_decode_buffers = _workspace._ensure_smallq_decode_buffers
    _clear_smallq_decode_metadata = verify_metadata._clear_smallq_decode_metadata
    _attach_prepared_dflash2_smallq_metadata = (
        verify_metadata._attach_prepared_dflash2_smallq_metadata
    )
    _update_smallq_decode_metadata = verify_metadata._update_smallq_decode_metadata
    build = builder.build


@dataclass(frozen=True)
class MetadataHooks:
    initialize: Callable[[FlashAttnV100MetadataBuilder, SpeculativeConfig | None], None]
    attach_common: Callable[
        [FlashAttnV100MetadataBuilder, TritonAttentionMetadata], None
    ]
    prepare_capture: Callable[
        [
            FlashAttnV100MetadataBuilder,
            TritonAttentionMetadata,
            CommonAttentionMetadata,
        ],
        None,
    ]


def register_metadata_hooks() -> MetadataHooks:
    """Register the built-in feature providers without a mutable global registry."""
    return MetadataHooks(
        initialize=builder.initialize_builder,
        attach_common=builder.attach_common,
        prepare_capture=builder.prepare_capture,
    )


METADATA_HOOKS = register_metadata_hooks()
