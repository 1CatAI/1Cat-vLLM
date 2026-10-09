# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 speculative builder metadata owner."""

from __future__ import annotations

import torch

from vllm.config.sm70_dflash2 import (
    capture_sm70_dflash2_config,
    sm70_dflash2_enabled,
)
from vllm.config.speculative import get_dflash_model_draft_tokens
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_v100 import metadata as _metadata
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100.spec import (
    smallq_metadata as _smallq_metadata,
)
from vllm.v1.worker.gpu.spec_decode import uses_dflash_selector_engine

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def build(
    self: _metadata.FlashAttnV100MetadataBuilder,
    common_prefix_len,
    common_attn_metadata,
    fast_build: bool = False,
    ddtree_parent_ids: torch.Tensor | None = None,
    ddtree_num_tree_tokens_cpu: torch.Tensor | None = None,
    prepared_dflash2_smallq_metadata: (
        _smallq_metadata.DFlash2SmallQPreparedMetadata | None
    ) = None,
):
    attn_metadata = super(_metadata._spec_builder_super_owner, self).build(
        common_prefix_len, common_attn_metadata, fast_build
    )
    self._attach_common_flash_metadata(attn_metadata, common_attn_metadata)
    self._attach_prefix_anchored_metadata(attn_metadata, common_attn_metadata)
    self._attach_ddtree_metadata(
        attn_metadata,
        ddtree_parent_ids=ddtree_parent_ids,
        ddtree_num_tree_tokens_cpu=ddtree_num_tree_tokens_cpu,
    )
    num_reqs = max(0, int(attn_metadata.query_start_loc.numel()) - 1)
    ddtree_tree_verify = (
        ddtree_parent_ids is not None
        and ddtree_num_tree_tokens_cpu is not None
        and getattr(attn_metadata, "max_query_len", 1) > 1
        and bool(torch.any(ddtree_num_tree_tokens_cpu[:num_reqs] > 0).item())
    )
    if self._flash_draft_buffer_shape is not None and (
        getattr(attn_metadata, "max_query_len", 1) == 1 or self._is_dflash_draft_model
    ):
        # FULL graph capture binds q=1 decode to these persistent buffers.
        # DFlash binds its q=K+1 paged-prefill graph to the same buffers.
        # Refresh them on every runtime step so replay sees the current
        # request's block table and sequence metadata.
        self._stabilize_draft_graph_metadata(
            attn_metadata,
            common_attn_metadata,
        )
    if not ddtree_tree_verify:
        # EAGER build path: cap the small-query decode workspace/launch grid
        # to the runtime max_seq_len, mirroring build_for_cudagraph_capture
        # (:2431-2445). Without a cap, _update_smallq_decode_metadata stores
        # the full block-table capacity (== max_model_len worth of blocks) at
        # smallq_decode_workspace_seq_capacity_hint (:2350), and the interface
        # then launches ceil(max_model_len/partition_size) partitions where
        # only ceil(max_seq_len/partition_size) do work (1024 vs 17 at
        # max_model_len=262144 / S=4224 / ps=256). The cap keeps launch equal
        # to the runtime coverage; the interface floors it at effective
        # max_seq_len (_get_decode_plan:165-169), so it can never under-cover.
        if prepared_dflash2_smallq_metadata is not None:
            self._attach_prepared_dflash2_smallq_metadata(
                attn_metadata,
                prepared_dflash2_smallq_metadata,
            )
        else:
            self._update_smallq_decode_metadata(
                attn_metadata,
                common_attn_metadata,
                workspace_seq_capacity_cap=(
                    int(getattr(common_attn_metadata, "max_seq_len", 0) or 0) or None
                ),
            )
    self._attach_decode_shape_hints(attn_metadata, common_attn_metadata)
    self._update_decode_active_num_partitions(attn_metadata, stage="build")
    self._debug_draft_metadata("build", attn_metadata, common_attn_metadata)
    return attn_metadata


def initialize_builder(
    self: _metadata.FlashAttnV100MetadataBuilder, spec_config
) -> None:
    self._is_dflash_draft_model = self._is_speculative_draft_model and (
        getattr(spec_config, "method", None) == "dflash"
    )
    use_dflash = bool(
        spec_config is not None
        and callable(getattr(spec_config, "use_dflash", None))
        and spec_config.use_dflash()
    )
    selector_engine = uses_dflash_selector_engine(self.vllm_config)
    self._is_dflash_selector_target = bool(
        use_dflash
        and not self._is_speculative_draft_model
        and getattr(spec_config, "num_speculative_tokens", None) in (7, 15)
        and get_dflash_model_draft_tokens(spec_config) == 7
        and selector_engine
    )
    self._use_sm70_dflash2_fused_smallq_metadata = bool(
        sm70_dflash2_enabled(
            "fused_smallq_metadata", capture_sm70_dflash2_config(self.vllm_config)
        )
        and self.device.type == "cuda"
        and current_platform.is_device_capability(70)
        and use_dflash
        and selector_engine
    )
    if self._use_sm70_dflash2_fused_smallq_metadata:
        logger.info_once("SM70 DFlash2 fused Flash-V100 small-query metadata active.")
    self._draft_block_table = None
    self._draft_seq_lens = None
    self._draft_query_start_loc = None
    self._flash_draft_buffer_shape = None
    self._smallq_decode_block_table = None
    self._smallq_decode_seq_lens = None
    self._smallq_query_start_loc = None
    self._smallq_token_indices = None
    self._smallq_buffer_shape = None


def prepare_capture(
    self: _metadata.FlashAttnV100MetadataBuilder, attn_metadata, common_attn_metadata
) -> None:
    # The Triton builder shortens capture seq_lens to 1 so full graph
    # capture stays cheap. That is valid for single-token decode, but the
    # FA2 small-query MTP verifier replays a tiny causal prefill as paged
    # decode. Capturing that branch with seq_len < query_len creates
    # negative per-token decode lengths and can poison long-context graph
    # replay. Keep capture cheap while preserving a valid verifier shape.
    max_query_len = getattr(attn_metadata, "max_query_len", 1)
    if max_query_len > 1:
        attn_metadata.seq_lens.fill_(max_query_len)
        workspace_seq_capacity_cap = (
            int(getattr(common_attn_metadata, "max_seq_len", 0) or 0) or None
        )
        partition_size_hint = None
        if workspace_seq_capacity_cap is not None and workspace_seq_capacity_cap < int(
            self.vllm_config.model_config.max_model_len
        ):
            partition_size_hint = _routing._mtp_context_bucket_partition_size_hint()
        self._update_smallq_decode_metadata(
            attn_metadata,
            common_attn_metadata,
            force=True,
            workspace_seq_capacity_cap=workspace_seq_capacity_cap,
            partition_size_hint=partition_size_hint,
        )
    if max_query_len == 1 or self._is_dflash_draft_model:
        # PIECEWISE graph replay captures the q=1 decode kernel arguments
        # during metadata warmup. Runtime drafting updates the persistent
        # draft metadata buffers, so capture must bind the graph to the
        # same buffers instead of transient dummy capture tensors. DFlash
        # parallel drafting has q=K+1; its non-causal paged-prefill graph
        # consumes the same dynamic block table and sequence lengths.
        self._stabilize_draft_graph_metadata(
            attn_metadata,
            common_attn_metadata,
        )


def attach_common(self: _metadata.FlashAttnV100MetadataBuilder, attn_metadata) -> None:
    _metadata._as_flash_v100_metadata(
        attn_metadata
    ).is_dflash_selector_target = self._is_dflash_selector_target
