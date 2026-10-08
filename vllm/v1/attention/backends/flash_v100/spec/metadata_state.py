# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Own speculative builder state with explicit inputs and common callbacks."""

from __future__ import annotations

from vllm.v1.attention.backends.flash_v100 import workspace as _workspace
from vllm.v1.attention.backends.flash_v100.spec import (
    builder,
    draft,
    features,
    tree,
    verify_metadata,
)
from vllm.v1.attention.backends.flash_v100.spec.metadata_contracts import (
    MetadataInputs,
    MetadataOps,
)


class SpecMetadataState:
    def __init__(self, inputs: MetadataInputs, ops: MetadataOps, spec_config):
        self.inputs = inputs
        self.ops = ops
        builder.initialize_builder(self, spec_config)
        self.feature = features.FEATURES.for_method(
            getattr(spec_config, "method", None)
        )

    @property
    def vllm_config(self):
        return self.inputs.vllm_config

    @property
    def device(self):
        return self.inputs.device

    @property
    def block_size(self):
        return self.inputs.block_size

    @property
    def _is_speculative_draft_model(self):
        return self.inputs.is_draft

    attach_common = builder.attach_common
    prepare_capture = builder.prepare_capture
    debug_metadata = draft._debug_draft_metadata

    _is_dflash_draft_model: bool
    _is_dflash_selector_target: bool
    _use_sm70_dflash2_fused_smallq_metadata: bool
    metadata_workspace: _workspace.MetadataWorkspace

    _attach_ddtree_metadata = tree.attach_metadata
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
    _attach_prepared_dflash2_smallq_metadata = verify_metadata.attach_prepared_metadata
    _update_smallq_decode_metadata = verify_metadata.update_decode_metadata
    build = builder.build
