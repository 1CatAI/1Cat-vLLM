# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered speculative metadata hooks and compatibility method surface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.flash_v100.spec import (
    builder,
)
from vllm.v1.attention.backends.flash_v100.spec.metadata_state import (
    SpecMetadataMethods as SpecMetadataMethods,
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
