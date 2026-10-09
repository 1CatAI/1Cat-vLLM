# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 speculative draft metadata owner."""

from __future__ import annotations

import time
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import debug as _debug
from vllm.v1.attention.backends.flash_v100.spec import policy
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def _debug_draft_metadata(
    self: Any,
    stage: str,
    attn_metadata: TritonAttentionMetadata,
    common_attn_metadata,
) -> None:
    if not self._is_speculative_draft_model or not _debug._draft_graph_debug_enabled():
        return
    _debug._draft_graph_debug_log(
        f"builder:{stage}",
        "num_reqs=%s num_actual_tokens=%s max_query_len=%s max_seq_len=%s "
        "common_qsl_cpu=%s common_seq_cpu=%s %s %s %s %s %s %s %s",
        getattr(common_attn_metadata, "num_reqs", None),
        getattr(common_attn_metadata, "num_actual_tokens", None),
        getattr(common_attn_metadata, "max_query_len", None),
        getattr(common_attn_metadata, "max_seq_len", None),
        getattr(common_attn_metadata, "query_start_loc_cpu", None),
        getattr(common_attn_metadata, "seq_lens_cpu", None),
        _debug._format_tensor_debug(
            getattr(common_attn_metadata, "query_start_loc", None),
            "common_qsl",
        ),
        _debug._format_tensor_debug(
            getattr(common_attn_metadata, "seq_lens", None),
            "common_seq",
        ),
        _debug._format_tensor_debug(
            getattr(common_attn_metadata, "block_table_tensor", None),
            "common_bt",
        ),
        _debug._format_tensor_debug(
            getattr(attn_metadata, "query_start_loc", None),
            "attn_qsl",
        ),
        _debug._format_tensor_debug(
            getattr(attn_metadata, "seq_lens", None), "attn_seq"
        ),
        _debug._format_tensor_debug(
            getattr(attn_metadata, "block_table", None),
            "attn_bt",
        ),
        _debug._format_tensor_debug(
            getattr(attn_metadata, "smallq_decode_seq_lens", None),
            "smallq_seq",
        ),
    )


def _stabilize_draft_graph_metadata(
    self: Any,
    attn_metadata: TritonAttentionMetadata,
    common_attn_metadata,
) -> None:
    num_reqs = int(common_attn_metadata.num_reqs)
    if num_reqs <= 0:
        return

    block_table = attn_metadata.block_table[:num_reqs]
    if not self._ensure_flash_draft_graph_buffers(num_reqs, block_table):
        assert self.metadata_workspace.draft.shape is not None
        req_capacity, block_cols = self.metadata_workspace.draft.shape
        raise RuntimeError(
            "FLASH_ATTN_V100 draft CUDA graph metadata shape exceeds "
            "the captured persistent buffer capacity: "
            f"required_reqs={num_reqs}, "
            f"required_block_cols={int(block_table.shape[1])}, "
            f"capacity_reqs={req_capacity}, "
            f"capacity_block_cols={block_cols}. "
            "Replay would otherwise read stale draft metadata."
        )

    assert self.metadata_workspace.draft.block_table is not None
    assert self.metadata_workspace.draft.seq_lens is not None
    assert self.metadata_workspace.draft.query_start_loc is not None

    self.copy_dflash_graph_metadata(
        block_table,
        attn_metadata.seq_lens[:num_reqs],
        attn_metadata.query_start_loc[: num_reqs + 1],
    )

    attn_metadata.block_table = self.metadata_workspace.draft.block_table[:num_reqs]
    attn_metadata.seq_lens = self.metadata_workspace.draft.seq_lens[:num_reqs]
    attn_metadata.query_start_loc = self.metadata_workspace.draft.query_start_loc[
        : num_reqs + 1
    ]


def build_for_drafting(self: Any, common_attn_metadata, draft_index: int):
    profile_enabled = policy.worker_profile_enabled()
    profile_t0 = time.perf_counter() if profile_enabled else 0.0
    profile_stage_t0 = profile_t0
    attn_metadata = self.ops.base_build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
        fast_build=True,
    )
    super_build_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    self.ops.attach_common(attn_metadata, common_attn_metadata)
    attach_common_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    self._stabilize_draft_graph_metadata(attn_metadata, common_attn_metadata)
    stabilize_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    # EAGER drafting build path: same runtime max_seq_len cap as build()
    # above, so the drafter hot loop does not over-launch to the full
    # max_model_len envelope. See build() for the full rationale.
    self._update_smallq_decode_metadata(
        attn_metadata,
        common_attn_metadata,
        workspace_seq_capacity_cap=(
            int(getattr(common_attn_metadata, "max_seq_len", 0) or 0) or None
        ),
    )
    smallq_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    self.ops.attach_shape_hints(attn_metadata, common_attn_metadata)
    shape_hints_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    self.ops.update_active_partitions(
        attn_metadata,
        stage=f"draft{draft_index}",
    )
    active_partitions_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    self._debug_draft_metadata(
        f"draft{draft_index}",
        attn_metadata,
        common_attn_metadata,
    )
    debug_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    if profile_enabled:
        logger.info(
            "FLASH_ATTN_V100 DDTREE_WORKER_PROFILE build_for_drafting "
            "draft_index=%d total_ms=%.3f super_build_ms=%.3f "
            "attach_common_ms=%.3f stabilize_ms=%.3f smallq_ms=%.3f "
            "shape_hints_ms=%.3f active_partitions_ms=%.3f debug_ms=%.3f "
            "max_query_len=%s num_actual_tokens=%s num_reqs=%s",
            draft_index,
            (time.perf_counter() - profile_t0) * 1000.0,
            super_build_ms,
            attach_common_ms,
            stabilize_ms,
            smallq_ms,
            shape_hints_ms,
            active_partitions_ms,
            debug_ms,
            getattr(common_attn_metadata, "max_query_len", None),
            getattr(common_attn_metadata, "num_actual_tokens", None),
            getattr(common_attn_metadata, "num_reqs", None),
        )
    return attn_metadata


def _ensure_flash_draft_graph_buffers(
    self: Any,
    required_reqs: int,
    block_table: torch.Tensor,
) -> bool:
    req_capacity = max(
        int(self.vllm_config.scheduler_config.max_num_seqs),
        int(required_reqs),
        1,
    )
    block_cols = int(block_table.shape[1])
    return self.metadata_workspace.draft.ensure(
        req_capacity, block_cols, required_reqs, self.device
    )


def copy_dflash_graph_metadata(
    self: Any,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    """Refresh the three persistent inputs of a non-causal DFlash graph."""
    self.metadata_workspace.draft.copy_metadata(block_table, seq_lens, query_start_loc)
