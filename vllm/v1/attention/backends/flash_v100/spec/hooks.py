# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered speculative metadata hooks and compatibility method surface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata


class SpecMetadataFields:
    ddtree_parent_ids: torch.Tensor | None
    ddtree_parent_ids_cpu: torch.Tensor | None
    ddtree_num_tree_tokens_cpu: torch.Tensor | None
    ddtree_seq_lens_restored_for_triton: bool
    ddtree_query_start_loc_restored_for_triton: bool
    is_dflash_selector_target: bool


@dataclass(frozen=True)
class MetadataHooks:
    initialize: Callable[[Any, SpeculativeConfig | None], None]
    attach_common: Callable[[Any, TritonAttentionMetadata], None]
    prepare_capture: Callable[
        [
            Any,
            TritonAttentionMetadata,
            CommonAttentionMetadata,
        ],
        None,
    ]


def initialize_legacy(instance: Any, spec_config) -> None:
    instance.initialize_spec_state(spec_config)


def attach_common_legacy(instance: Any, attn_metadata) -> None:
    instance.spec_state.attach_common(attn_metadata)


def prepare_capture_legacy(instance: Any, attn_metadata, common_attn_metadata) -> None:
    instance.spec_state.prepare_capture(attn_metadata, common_attn_metadata)


def register_metadata_hooks() -> MetadataHooks:
    """Register the built-in feature providers without a mutable global registry."""
    return MetadataHooks(
        initialize=initialize_legacy,
        attach_common=attach_common_legacy,
        prepare_capture=prepare_capture_legacy,
    )


METADATA_HOOKS = register_metadata_hooks()
