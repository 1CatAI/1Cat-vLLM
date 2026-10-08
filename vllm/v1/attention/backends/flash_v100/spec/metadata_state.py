# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative metadata operations before moving their receiver ownership."""

from __future__ import annotations

from vllm.v1.attention.backends.flash_v100 import workspace as _workspace
from vllm.v1.attention.backends.flash_v100.spec import (
    builder,
    draft,
    tree,
    verify_metadata,
)


class SpecMetadataMethods:
    _is_dflash_draft_model: bool
    _is_dflash_selector_target: bool
    _use_sm70_dflash2_fused_smallq_metadata: bool
    metadata_workspace: _workspace.MetadataWorkspace

    _attach_ddtree_metadata = tree._attach_ddtree_metadata
    _debug_draft_metadata = draft._debug_draft_metadata
    _ensure_flash_draft_graph_buffers = draft._ensure_flash_draft_graph_buffers
    _stabilize_draft_graph_metadata = draft._stabilize_draft_graph_metadata
    copy_dflash_graph_metadata = draft.copy_dflash_graph_metadata
    build_for_drafting = draft.build_for_drafting
    _configured_smallq_max_query_len = verify_metadata._configured_smallq_max_query_len
    _configured_smallq_max_model_len = verify_metadata._configured_smallq_max_model_len
    _smallq_buffer_token_capacity = verify_metadata._smallq_buffer_token_capacity
    _ensure_smallq_decode_buffers = verify_metadata._ensure_smallq_decode_buffers
    _clear_smallq_decode_metadata = verify_metadata._clear_smallq_decode_metadata
    _attach_prepared_dflash2_smallq_metadata = (
        verify_metadata._attach_prepared_dflash2_smallq_metadata
    )
    _update_smallq_decode_metadata = verify_metadata._update_smallq_decode_metadata
    build = builder.build
