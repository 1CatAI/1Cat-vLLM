# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 speculative verify metadata metadata owner."""

from __future__ import annotations

import time
from typing import Any, cast

import torch

from vllm.config.execution_policy import flash_v100_policy
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import config as _config
from vllm.v1.attention.backends.flash_v100.plan import diagnostics as _debug
from vllm.v1.attention.backends.flash_v100.spec import policy
from vllm.v1.attention.backends.flash_v100.spec import (
    smallq_metadata as _smallq_metadata,
)
from vllm.v1.attention.backends.flash_v100.spec.metadata_contracts import metadata_view
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
)

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def configured_smallq_max_query_len(
    self: Any,
) -> int:
    return int(cast(int, flash_v100_policy().smallq_max_q))


def configured_smallq_max_model_len(
    self: Any,
) -> int:
    return int(_config.raw("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN", "0"))


def smallq_buffer_token_capacity(self: Any, required_tokens: int) -> int:
    compilation_config = self.vllm_config.compilation_config
    graph_tokens = compilation_config.max_cudagraph_capture_size
    if graph_tokens is None and compilation_config.cudagraph_capture_sizes:
        graph_tokens = max(compilation_config.cudagraph_capture_sizes)
    if graph_tokens is None or graph_tokens <= 0:
        graph_tokens = required_tokens
    smallq_max_query_len = max(self._configured_smallq_max_query_len(), 0)
    max_num_seqs = max(int(self.vllm_config.scheduler_config.max_num_seqs), 1)
    # MTP verifier graph capture can bind a q=N branch before the runtime
    # request reaches the largest small-query shape. Keep the persistent
    # graph metadata buffers sized for the configured small-query envelope
    # instead of the first captured shape, otherwise replay would either
    # read stale metadata or trip the capacity guard at runtime.
    smallq_token_capacity = smallq_max_query_len * max_num_seqs
    return max(
        int(graph_tokens),
        int(required_tokens),
        int(smallq_token_capacity),
        1,
    )


def clear_smallq_decode_metadata(
    self: Any,
    attn_metadata: TritonAttentionMetadata,
) -> None:
    flash_metadata = metadata_view(attn_metadata)
    flash_metadata.smallq_decode_block_table = None
    flash_metadata.smallq_decode_seq_lens = None
    flash_metadata.smallq_query_start_loc = None
    flash_metadata.smallq_decode_max_seq_len_hint = None
    flash_metadata.smallq_decode_workspace_seq_capacity_hint = None
    flash_metadata.smallq_decode_partition_size_hint = None


def attach_prepared_metadata(
    self: Any,
    attn_metadata: TritonAttentionMetadata,
    prepared: _smallq_metadata.DFlash2SmallQPreparedMetadata,
) -> None:
    """Attach buffers refreshed by the cross-cache-group launch."""
    if prepared.builder_id != self.inputs.builder_id:
        raise ValueError("grouped small-query metadata belongs to another builder")
    if (
        self.metadata_workspace.smallq.block_table is None
        or self.metadata_workspace.smallq.seq_lens is None
        or self.metadata_workspace.smallq.query_start_loc is None
        or self.metadata_workspace.smallq.shape is None
    ):
        raise RuntimeError("grouped small-query metadata has no persistent buffers")
    token_capacity, req_capacity, _ = self.metadata_workspace.smallq.shape
    if prepared.num_query_tokens > token_capacity or prepared.num_reqs > req_capacity:
        raise RuntimeError("grouped small-query metadata exceeds captured capacity")

    self._clear_smallq_decode_metadata(attn_metadata)
    flash_metadata = metadata_view(attn_metadata)
    flash_metadata.smallq_decode_block_table = (
        self.metadata_workspace.smallq.block_table[: prepared.num_query_tokens]
    )
    flash_metadata.smallq_decode_seq_lens = self.metadata_workspace.smallq.seq_lens[
        : prepared.num_query_tokens
    ]
    flash_metadata.smallq_query_start_loc = (
        self.metadata_workspace.smallq.query_start_loc[: prepared.num_reqs + 1]
    )
    flash_metadata.smallq_decode_max_seq_len_hint = prepared.max_seq_len_hint
    flash_metadata.smallq_decode_workspace_seq_capacity_hint = (
        prepared.workspace_seq_capacity_hint
    )
    flash_metadata.smallq_decode_partition_size_hint = prepared.partition_size_hint


def update_decode_metadata(
    self: Any,
    attn_metadata: TritonAttentionMetadata,
    common_attn_metadata,
    *,
    force: bool = False,
    workspace_seq_capacity_cap: int | None = None,
    partition_size_hint: int | None = None,
) -> None:
    flash_metadata = metadata_view(attn_metadata)
    profile_enabled = policy.worker_profile_enabled()
    profile_t0 = time.perf_counter() if profile_enabled else 0.0
    profile_stage_t0 = profile_t0
    self._clear_smallq_decode_metadata(attn_metadata)
    clear_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0

    max_query_len = int(getattr(attn_metadata, "max_query_len", 1))
    smallq_max_query_len = self._configured_smallq_max_query_len()
    if (
        smallq_max_query_len <= 0
        or max_query_len <= 1
        or max_query_len > smallq_max_query_len
    ):
        return

    smallq_max_model_len = self._configured_smallq_max_model_len()
    max_model_len = int(self.vllm_config.model_config.max_model_len)
    if smallq_max_model_len > 0 and max_model_len > smallq_max_model_len:
        return

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
    if seq_lens_cpu is None:
        # This metadata path is on the drafter hot loop. Async speculative
        # decode may omit the exact CPU shadow, but Flash-V100 only needs
        # a CPU value here for small-query route and workspace hints. Use
        # the scheduler-maintained upper bound to avoid an implicit
        # seq_lens.to("cpu") synchronization.
        seq_lens_cpu = getattr(
            common_attn_metadata,
            "seq_lens_cpu_upper_bound",
            None,
        )
    if seq_lens_cpu is None:
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    has_prefix_context = bool(torch.any(query_lens_cpu != seq_lens_cpu).item())
    if (
        not force
        and not has_prefix_context
        and self.metadata_workspace.smallq.shape is None
    ):
        return

    num_query_tokens = int(attn_metadata.num_actual_tokens)
    num_reqs = int(common_attn_metadata.num_reqs)
    if num_query_tokens <= 0 or num_reqs <= 0:
        return

    block_table = attn_metadata.block_table[:num_reqs]
    guard_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    if not self._ensure_smallq_decode_buffers(
        num_query_tokens,
        num_reqs,
        block_table,
    ):
        assert self.metadata_workspace.smallq.shape is not None
        token_capacity, req_capacity, block_cols = self.metadata_workspace.smallq.shape
        raise RuntimeError(
            "FLASH_ATTN_V100 small-query CUDA graph metadata shape exceeds "
            "the captured persistent buffer capacity: "
            f"required_tokens={num_query_tokens}, "
            f"required_reqs={num_reqs}, "
            f"required_block_cols={int(block_table.shape[1])}, "
            f"capacity_tokens={token_capacity}, "
            f"capacity_reqs={req_capacity}, "
            f"capacity_block_cols={block_cols}. "
            "Replay would otherwise use stale captured metadata."
        )
    ensure_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    assert self.metadata_workspace.smallq.block_table is not None
    assert self.metadata_workspace.smallq.seq_lens is not None
    assert self.metadata_workspace.smallq.query_start_loc is not None
    assert self.metadata_workspace.smallq.token_indices is not None

    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    query_start_loc = attn_metadata.query_start_loc[: num_reqs + 1]
    real_num_query_tokens = int(query_start_loc_cpu[-1].item())
    if real_num_query_tokens > num_query_tokens:
        return
    padding_tokens = num_query_tokens - real_num_query_tokens
    seq_lens = attn_metadata.seq_lens[:num_reqs]
    prep_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    expand_ms, copy_ms = _expand_decode_rows(
        self,
        block_table,
        seq_lens,
        query_start_loc,
        num_reqs,
        num_query_tokens,
        real_num_query_tokens,
        padding_tokens,
        profile_enabled,
    )

    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    flash_metadata.smallq_decode_block_table = (
        self.metadata_workspace.smallq.block_table[:num_query_tokens]
    )
    flash_metadata.smallq_decode_seq_lens = self.metadata_workspace.smallq.seq_lens[
        :num_query_tokens
    ]
    flash_metadata.smallq_query_start_loc = (
        self.metadata_workspace.smallq.query_start_loc[: num_reqs + 1]
    )
    raw_seq_capacity = int(block_table.shape[1]) * int(self.block_size)
    max_seq_len_hint = int(seq_lens_cpu.max().item())
    if max_seq_len_hint > 0 and raw_seq_capacity > 0:
        # MTP verification reaches this backend as q>1 prefix prefill, but
        # the Flash-V100 long-context optimization still applies because
        # the actual compute is paged decode over each tiny query row.
        # Keep graph replay capacity fixed while letting kernels skip
        # inactive partitions for the current runtime sequence length.
        flash_metadata.smallq_decode_max_seq_len_hint = max_seq_len_hint
        if workspace_seq_capacity_cap is not None:
            # A distinct CUDA graph key guarantees replay only below this
            # bound. The block table remains full-width so runtime KV
            # addresses stay stable, while the captured workspace/grid is
            # reduced to the bounded context envelope.
            raw_seq_capacity = min(
                raw_seq_capacity,
                max(max_seq_len_hint, int(workspace_seq_capacity_cap)),
            )
        flash_metadata.smallq_decode_workspace_seq_capacity_hint = raw_seq_capacity
        flash_metadata.smallq_decode_partition_size_hint = partition_size_hint
    hint_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    if profile_enabled:
        logger.info(
            "FLASH_ATTN_V100 DDTREE_WORKER_PROFILE smallq_metadata "
            "total_ms=%.3f clear_ms=%.3f guard_ms=%.3f ensure_ms=%.3f "
            "prep_ms=%.3f expand_ms=%.3f copy_ms=%.3f hint_ms=%.3f "
            "num_reqs=%d num_query_tokens=%d real_query_tokens=%d "
            "padding_tokens=%d block_cols=%d fused=%s",
            (time.perf_counter() - profile_t0) * 1000.0,
            clear_ms,
            guard_ms,
            ensure_ms,
            prep_ms,
            expand_ms,
            copy_ms,
            hint_ms,
            num_reqs,
            num_query_tokens,
            real_num_query_tokens,
            padding_tokens,
            int(block_table.shape[1]),
            self._use_sm70_dflash2_fused_smallq_metadata,
        )
    _observe_smallq_update(
        self,
        force,
        num_reqs,
        num_query_tokens,
        real_num_query_tokens,
        padding_tokens,
        max_query_len,
        query_start_loc_cpu,
        seq_lens_cpu,
        attn_metadata,
        flash_metadata,
    )


def ensure_smallq_decode_buffers(
    self: Any,
    required_tokens: int,
    required_reqs: int,
    block_table: torch.Tensor,
) -> bool:
    token_capacity = self._smallq_buffer_token_capacity(required_tokens)
    req_capacity = max(
        min(
            int(self.vllm_config.scheduler_config.max_num_seqs),
            token_capacity,
        ),
        int(required_reqs),
        1,
    )
    block_cols = int(block_table.shape[1])
    return self.metadata_workspace.smallq.ensure(
        token_capacity,
        req_capacity,
        block_cols,
        required_tokens,
        required_reqs,
        self.device,
    )


# External compatibility; state owners bind the public calculations.
_attach_prepared_dflash2_smallq_metadata = attach_prepared_metadata
_update_smallq_decode_metadata = update_decode_metadata

# Public owner operations; legacy bindings are installed by package assembly.
LEGACY_ALIASES = {
    "_smallq_buffer_token_capacity": "smallq_buffer_token_capacity",
    "_ensure_smallq_decode_buffers": "ensure_smallq_decode_buffers",
    "_configured_smallq_max_query_len": "configured_smallq_max_query_len",
    "_clear_smallq_decode_metadata": "clear_smallq_decode_metadata",
    "_configured_smallq_max_model_len": "configured_smallq_max_model_len",
}


def _expand_decode_rows(
    self,
    block_table,
    seq_lens,
    query_start_loc,
    num_reqs,
    num_query_tokens,
    real_num_query_tokens,
    padding_tokens,
    profile_enabled,
):
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    if self._use_sm70_dflash2_fused_smallq_metadata:
        _smallq_metadata.sm70_prepare_smallq_decode_metadata(
            self.metadata_workspace.smallq.block_table,
            self.metadata_workspace.smallq.seq_lens,
            self.metadata_workspace.smallq.query_start_loc,
            block_table,
            seq_lens,
            query_start_loc,
            num_reqs=num_reqs,
            num_query_tokens=num_query_tokens,
            real_num_query_tokens=real_num_query_tokens,
        )
        expand_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        copy_ms = 0.0
    else:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        real_query_lens = query_lens
        repeat_query_lens = query_lens
        if padding_tokens > 0:
            repeat_query_lens = query_lens.clone()
            repeat_query_lens[-1] += padding_tokens

        effective_seq_lens = torch.maximum(
            seq_lens,
            real_query_lens.to(dtype=seq_lens.dtype),
        )
        clamped_block_table = block_table.clamp_min(0)
        decode_block_table = torch.repeat_interleave(
            clamped_block_table,
            repeat_query_lens,
            dim=0,
            output_size=num_query_tokens,
        ).contiguous()
        seq_lens_rep = torch.repeat_interleave(
            effective_seq_lens,
            repeat_query_lens,
            output_size=num_query_tokens,
        )
        query_lens_rep = torch.repeat_interleave(
            real_query_lens.to(dtype=seq_lens.dtype),
            repeat_query_lens,
            output_size=num_query_tokens,
        )
        start_locs_rep = torch.repeat_interleave(
            query_start_loc[:-1].to(dtype=seq_lens.dtype),
            repeat_query_lens,
            output_size=num_query_tokens,
        )
        token_indices = self.metadata_workspace.smallq.token_indices[
            :num_query_tokens
        ].to(dtype=seq_lens.dtype)
        offsets = token_indices - start_locs_rep + 1
        decode_seq_lens = (seq_lens_rep - query_lens_rep + offsets).contiguous()
        if padding_tokens > 0:
            padding_mask = token_indices >= real_num_query_tokens
            decode_seq_lens = torch.where(
                padding_mask,
                torch.zeros_like(decode_seq_lens),
                decode_seq_lens,
            ).contiguous()
            decode_block_table = torch.where(
                padding_mask[:, None],
                torch.zeros_like(decode_block_table),
                decode_block_table,
            ).contiguous()
        expand_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )

        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        self.metadata_workspace.smallq.block_table[:num_query_tokens].copy_(
            decode_block_table,
            non_blocking=True,
        )
        self.metadata_workspace.smallq.seq_lens[:num_query_tokens].copy_(
            decode_seq_lens,
            non_blocking=True,
        )
        self.metadata_workspace.smallq.query_start_loc[: num_reqs + 1].copy_(
            query_start_loc,
            non_blocking=True,
        )
        copy_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
    return expand_ms, copy_ms


def _observe_smallq_update(
    self,
    force,
    num_reqs,
    num_query_tokens,
    real_num_query_tokens,
    padding_tokens,
    max_query_len,
    query_start_loc_cpu,
    seq_lens_cpu,
    attn_metadata,
    flash_metadata,
):
    if _debug.draft_graph_debug_enabled():
        _debug.graph_metadata_debug_log(
            "smallq_update",
            "draft=%s force=%s num_reqs=%s num_query_tokens=%s "
            "real_num_query_tokens=%s padding_tokens=%s max_query_len=%s "
            "common_qsl_cpu=%s common_seq_cpu=%s %s %s %s %s %s %s",
            self._is_speculative_draft_model,
            force,
            num_reqs,
            num_query_tokens,
            real_num_query_tokens,
            padding_tokens,
            max_query_len,
            query_start_loc_cpu,
            seq_lens_cpu,
            _debug.format_tensor_debug(attn_metadata.query_start_loc, "attn_qsl"),
            _debug.format_tensor_debug(attn_metadata.seq_lens, "attn_seq"),
            _debug.format_tensor_debug(attn_metadata.block_table, "attn_bt"),
            _debug.format_tensor_debug(
                flash_metadata.smallq_decode_block_table,
                "smallq_bt",
            ),
            _debug.format_tensor_debug(
                flash_metadata.smallq_decode_seq_lens,
                "smallq_seq",
            ),
            _debug.format_tensor_debug(
                flash_metadata.smallq_query_start_loc,
                "smallq_qsl",
            ),
        )
