# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 attention metadata, its builder and mixed decode-row plans."""

from __future__ import annotations

import os
import time
from contextlib import suppress
from typing import cast

import torch

import vllm.envs as envs
from vllm.config.sm70_dflash2 import (
    capture_sm70_dflash2_config,
    sm70_dflash2_enabled,
)
from vllm.config.speculative import get_dflash_model_draft_tokens
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_v100 import debug as _debug
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100 import smallq_metadata as _smallq_metadata
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import PrefixAnchoredSWASpec
from vllm.v1.worker.gpu.spec_decode import uses_dflash_selector_engine

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


class FlashAttnV100Metadata(TritonAttentionMetadata):
    """Static view of Flash-V100 fields attached to Triton metadata."""

    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    causal: bool
    max_model_len: int
    flash_v100_cudagraph_capture: bool
    flash_v100_batch_context_routing: bool
    flash_v100_contig_dense_cache: dict[tuple[int, int, int, int, int], int]
    prefix_anchor_lens: torch.Tensor | None = None
    decode_sliding_window: int | None = None
    flash_v100_decode_max_seq_len_hint: int | None
    flash_v100_decode_workspace_seq_capacity_hint: int | None
    flash_v100_static_decode_seq_hint: int | None
    flash_v100_decode_active_num_partitions: torch.Tensor | None
    ddtree_parent_ids: torch.Tensor | None
    ddtree_parent_ids_cpu: torch.Tensor | None
    ddtree_num_tree_tokens_cpu: torch.Tensor | None
    ddtree_seq_lens_restored_for_triton: bool
    ddtree_query_start_loc_restored_for_triton: bool
    smallq_decode_block_table: torch.Tensor | None
    smallq_decode_seq_lens: torch.Tensor | None
    smallq_query_start_loc: torch.Tensor | None
    smallq_decode_max_seq_len_hint: int | None
    smallq_decode_workspace_seq_capacity_hint: int | None
    smallq_decode_partition_size_hint: int | None
    is_dflash_selector_target: bool


def _as_flash_v100_metadata(
    attn_metadata: TritonAttentionMetadata,
) -> FlashAttnV100Metadata:
    # The inherited Triton builder creates the object; this backend attaches
    # the fields above before any Flash-V100 path consumes them.
    return cast(FlashAttnV100Metadata, attn_metadata)


_MIXED_ROWS_PLAN_ATTR = "_flash_v100_mixed_decode_rows_plan"
_MIXED_ROWS_GROUP = 8


class _MixedDecodeRowsPlan:
    """Layout of the small-query rows of one mixed prefill/decode step.

    The host side depends only on ``query_start_loc`` and the sequence-length
    shadow, so it is built once per step and shared by every attention layer of
    the group instead of re-deriving it (lists, host-to-device copies, gathers)
    in each layer. Visible KV lengths are never taken from the host shadow: it
    can be an upper bound under async speculative decoding, so every row length
    is derived on the device from the authoritative ``seq_lens``.

    Tokens are ordered by request and then by position inside the request.
    A request with ``q`` query tokens is split into ``ceil(q / 8)`` groups of
    eight rows for the request-major grouped operator; the causal boundary of
    each row is carried by its own length, so slicing a longer span is exact.
    """

    __slots__ = (
        "rows",
        "max_query_len",
        "max_seq_len_hint",
        "num_groups",
        "src_idx",
        "token_req",
        "token_delta",
        "dst_idx",
        "group_req",
        "_token_lengths",
        "_group_lengths",
        "_group_table",
    )

    def __init__(
        self,
        rows: tuple[int, ...],
        qsl: list[int],
        seq_lens_host: list[int],
        device: torch.device,
    ) -> None:
        src: list[int] = []
        req: list[int] = []
        delta: list[int] = []
        dst: list[int] = []
        group_req: list[int] = []
        max_query_len = 0
        max_seq_len = 0
        for i in rows:
            q_len = qsl[i + 1] - qsl[i]
            max_query_len = max(max_query_len, q_len)
            max_seq_len = max(max_seq_len, int(seq_lens_host[i]))
            base = len(group_req) * _MIXED_ROWS_GROUP
            group_req.extend([i] * -(-q_len // _MIXED_ROWS_GROUP))
            for j in range(q_len):
                src.append(qsl[i] + j)
                req.append(i)
                delta.append(1 + j - q_len)
                dst.append(base + j)
        self.rows = rows
        self.max_query_len = max_query_len
        self.max_seq_len_hint = max_seq_len
        self.num_groups = len(group_req)
        packed = torch.tensor(
            src + req + delta + dst + group_req,
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type == "cuda",
        ).to(device, non_blocking=True)
        n = len(src)
        self.src_idx = packed[:n]
        self.token_req = packed[n : 2 * n]
        self.token_delta = packed[2 * n : 3 * n].to(torch.int32)
        self.dst_idx = packed[3 * n : 4 * n]
        self.group_req = packed[4 * n :]
        self._token_lengths: torch.Tensor | None = None
        self._group_lengths: torch.Tensor | None = None
        self._group_table: torch.Tensor | None = None

    def token_lengths(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """Visible KV length of every selected query token (int32, [T])."""
        if self._token_lengths is None:
            self._token_lengths = (
                seq_lens.index_select(0, self.token_req).to(torch.int32)
                + self.token_delta
            )
        return self._token_lengths

    def group_lengths(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """Row lengths of the padded eight-row groups (zero on padding rows)."""
        if self._group_lengths is None:
            lengths = torch.zeros(
                self.num_groups * _MIXED_ROWS_GROUP,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            lengths.index_copy_(0, self.dst_idx, self.token_lengths(seq_lens))
            self._group_lengths = lengths
        return self._group_lengths

    def group_table(self, block_table: torch.Tensor) -> torch.Tensor:
        """One block-table row per eight-row group ([G, columns])."""
        if self._group_table is None:
            self._group_table = block_table.index_select(0, self.group_req)
        return self._group_table


def _mixed_decode_rows_plan(
    attn_metadata: TritonAttentionMetadata,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    device: torch.device,
) -> _MixedDecodeRowsPlan | None:
    """Select the resident decode/verification rows of a mixed batch.

    Returns ``None`` when the batch has no such row, or only such rows (that
    shape is the uniform small-query batch and has its own route). The result is
    cached on the step's metadata object, which every layer of the group shares.
    """
    cached = getattr(attn_metadata, _MIXED_ROWS_PLAN_ATTR, False)
    if cached is not False:
        return cast(_MixedDecodeRowsPlan | None, cached)
    num_seqs = len(query_start_loc) - 1
    qsl = query_start_loc[: num_seqs + 1].tolist()
    seq_lens_host = seq_lens[:num_seqs].tolist()
    rows = tuple(
        i
        for i in range(num_seqs)
        if 1 <= qsl[i + 1] - qsl[i] <= max_query_len
        and int(seq_lens_host[i]) > qsl[i + 1] - qsl[i]
    )
    plan = (
        _MixedDecodeRowsPlan(rows, qsl, seq_lens_host, device)
        if rows and len(rows) != num_seqs
        else None
    )
    with suppress(AttributeError):
        setattr(attn_metadata, _MIXED_ROWS_PLAN_ATTR, plan)
    return plan


class FlashAttnV100MetadataBuilder(TritonAttentionMetadataBuilder):
    """Attach CPU metadata for the dense prefill path."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        cache_config = getattr(self.vllm_config, "cache_config", None)
        model_config = self.vllm_config.model_config
        hf_text_config = getattr(model_config, "hf_text_config", None)
        num_attention_heads = getattr(hf_text_config, "num_attention_heads", None)
        num_key_value_heads = getattr(hf_text_config, "num_key_value_heads", None)
        head_dim = getattr(hf_text_config, "head_dim", None)
        batch_context_shape_supported = (
            isinstance(num_attention_heads, int)
            and isinstance(num_key_value_heads, int)
            and num_key_value_heads > 0
            and num_attention_heads == 6 * num_key_value_heads
            and head_dim == 256
        )
        self._is_speculative_draft_model = (
            spec_config is not None
            and getattr(spec_config, "draft_model_config", None)
            is self.vllm_config.model_config
        )
        self._batch_context_routing_enabled = (
            envs.VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING
            and envs.VLLM_FLASH_V100_DECODE_PARTITION_SIZE is None
            and spec_config is None
            and _routing._batch_context_routing_cache_dtype_supported(
                getattr(cache_config, "cache_dtype", None)
            )
            and batch_context_shape_supported
        )
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
            logger.info_once(
                "SM70 DFlash2 fused Flash-V100 small-query metadata active."
            )
        self._draft_block_table: torch.Tensor | None = None
        self._draft_seq_lens: torch.Tensor | None = None
        # Prefix-anchored SWA: persistent per-request prompt-length buffer so
        # the device address stays stable across steps.
        kv_cache_spec = self.kv_cache_spec
        self.decode_sliding_window = (
            kv_cache_spec.decode_sliding_window
            if isinstance(kv_cache_spec, PrefixAnchoredSWASpec)
            else None
        )
        self.persistent_prefix_anchor_lens: torch.Tensor | None = None
        if self.decode_sliding_window is not None:
            self.persistent_prefix_anchor_lens = torch.empty(
                self.vllm_config.scheduler_config.max_num_seqs,
                dtype=torch.int32,
                device=self.device,
            )
        self._draft_query_start_loc: torch.Tensor | None = None
        self._flash_draft_buffer_shape: tuple[int, int] | None = None
        self._smallq_decode_block_table: torch.Tensor | None = None
        self._smallq_decode_seq_lens: torch.Tensor | None = None
        self._smallq_query_start_loc: torch.Tensor | None = None
        self._smallq_token_indices: torch.Tensor | None = None
        self._smallq_buffer_shape: tuple[int, int, int] | None = None
        self._decode_active_num_partitions: torch.Tensor | None = None

    def _attach_prefix_anchored_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
    ) -> None:
        window = self.decode_sliding_window
        if window is None:
            return

        prefix_anchor_lens = common_attn_metadata.prefix_anchor_lens
        if prefix_anchor_lens is None:
            raise RuntimeError(
                "prefix-anchored SWA requires per-request prefix lengths"
            )
        assert self.persistent_prefix_anchor_lens is not None
        anchor_reqs = common_attn_metadata.num_reqs
        if prefix_anchor_lens.ndim != 1 or prefix_anchor_lens.numel() < anchor_reqs:
            raise RuntimeError(
                "prefix-anchored SWA prefix lengths must have shape [num_reqs]"
            )
        if anchor_reqs > self.persistent_prefix_anchor_lens.numel():
            raise RuntimeError(
                "prefix-anchored SWA request count exceeds metadata capacity"
            )

        prefix_anchor_lens = prefix_anchor_lens.to(
            device=self.device, dtype=torch.int32, non_blocking=True
        )
        persistent_anchor_lens = self.persistent_prefix_anchor_lens[:anchor_reqs]
        persistent_anchor_lens.copy_(prefix_anchor_lens[:anchor_reqs])
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.prefix_anchor_lens = persistent_anchor_lens
        flash_metadata.decode_sliding_window = window

    def _attach_common_flash_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        if seq_lens_cpu is None:
            # Async speculative decode keeps device seq_lens authoritative and
            # may deliberately omit the exact CPU shadow. Flash-V100 only uses
            # this CPU view for route/partition hints; the kernel still consumes
            # attn_metadata.seq_lens, so an upper bound is preferable to the
            # deprecated lazy seq_lens.to("cpu") sync.
            seq_lens_cpu = getattr(
                common_attn_metadata,
                "seq_lens_cpu_upper_bound",
                None,
            )
        flash_metadata.seq_lens_cpu = (
            seq_lens_cpu
            if seq_lens_cpu is not None
            else common_attn_metadata.seq_lens_cpu
        )
        flash_metadata.causal = common_attn_metadata.causal
        flash_metadata.is_dflash_selector_target = self._is_dflash_selector_target
        flash_metadata.max_model_len = self.vllm_config.model_config.max_model_len
        flash_metadata.flash_v100_cudagraph_capture = False
        flash_metadata.flash_v100_batch_context_routing = (
            _routing._batch_context_routing_for_graph_variant(
                self._batch_context_routing_enabled,
                getattr(common_attn_metadata, "cudagraph_graph_variant", None),
            )
        )

    def _attach_ddtree_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        *,
        ddtree_parent_ids: torch.Tensor | None,
        ddtree_num_tree_tokens_cpu: torch.Tensor | None,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.ddtree_parent_ids = None
        flash_metadata.ddtree_parent_ids_cpu = None
        flash_metadata.ddtree_num_tree_tokens_cpu = None
        flash_metadata.ddtree_seq_lens_restored_for_triton = False
        flash_metadata.ddtree_query_start_loc_restored_for_triton = False
        if ddtree_parent_ids is None:
            return
        if ddtree_num_tree_tokens_cpu is None:
            raise ValueError(
                "ddtree_num_tree_tokens_cpu is required with ddtree_parent_ids"
            )
        if ddtree_parent_ids.ndim != 2:
            raise ValueError("ddtree_parent_ids must have shape [batch, slots]")
        if ddtree_num_tree_tokens_cpu.ndim != 1:
            raise ValueError("ddtree_num_tree_tokens_cpu must be a 1D tensor")
        num_reqs = int(attn_metadata.query_start_loc.numel() - 1)
        if ddtree_parent_ids.shape[0] < num_reqs:
            raise ValueError(
                "ddtree_parent_ids must cover active requests: "
                f"{ddtree_parent_ids.shape[0]} < {num_reqs}"
            )
        if ddtree_num_tree_tokens_cpu.numel() < num_reqs:
            raise ValueError(
                "ddtree_num_tree_tokens_cpu must cover active requests: "
                f"{ddtree_num_tree_tokens_cpu.numel()} < {num_reqs}"
            )
        flash_metadata.ddtree_parent_ids = ddtree_parent_ids
        flash_metadata.ddtree_num_tree_tokens_cpu = ddtree_num_tree_tokens_cpu
        if bool(torch.any(ddtree_num_tree_tokens_cpu[:num_reqs] > 0).item()):
            seq_lens = getattr(attn_metadata, "seq_lens", None)
            seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
            if seq_lens is not None and seq_lens_cpu is not None:
                if _routing._is_cuda_graph_capturing(seq_lens):
                    seq_lens[:num_reqs].copy_(
                        seq_lens_cpu[:num_reqs].to(
                            device=seq_lens.device,
                            dtype=seq_lens.dtype,
                        ),
                        non_blocking=True,
                    )
                flash_metadata.ddtree_seq_lens_restored_for_triton = True
            query_start_loc = getattr(attn_metadata, "query_start_loc", None)
            query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
            if query_start_loc is not None and query_start_loc_cpu is not None:
                num_boundaries = num_reqs + 1
                if _routing._is_cuda_graph_capturing(query_start_loc):
                    query_start_loc[:num_boundaries].copy_(
                        query_start_loc_cpu[:num_boundaries].to(
                            device=query_start_loc.device,
                            dtype=query_start_loc.dtype,
                        ),
                        non_blocking=True,
                    )
                flash_metadata.ddtree_query_start_loc_restored_for_triton = True

    def _attach_decode_shape_hints(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
        *,
        static_decode: bool = False,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.flash_v100_decode_max_seq_len_hint = None
        flash_metadata.flash_v100_decode_workspace_seq_capacity_hint = None
        flash_metadata.flash_v100_static_decode_seq_hint = None

        max_query_len = int(getattr(common_attn_metadata, "max_query_len", 0) or 0)
        if max_query_len != 1:
            return

        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0:
            max_seq_len_hint = int(seq_lens_cpu.max().item())
        else:
            max_seq_len_hint = int(getattr(common_attn_metadata, "max_seq_len", 0) or 0)
        if max_seq_len_hint <= 0:
            return

        flash_metadata.flash_v100_decode_max_seq_len_hint = max_seq_len_hint
        if not static_decode:
            return

        block_table = getattr(common_attn_metadata, "block_table_tensor", None)
        if block_table is None:
            block_table = getattr(attn_metadata, "block_table", None)
        raw_seq_capacity = (
            int(block_table.shape[1]) * int(self.block_size)
            if block_table is not None
            else max_seq_len_hint
        )
        static_seq_capacity = max(
            max_seq_len_hint,
            int(getattr(common_attn_metadata, "max_seq_len", 0) or 0),
        )
        workspace_seq_capacity = min(raw_seq_capacity, static_seq_capacity)
        if (
            raw_seq_capacity > max_seq_len_hint
            or workspace_seq_capacity > max_seq_len_hint
        ):
            flash_metadata.flash_v100_static_decode_seq_hint = workspace_seq_capacity
        flash_metadata.flash_v100_decode_workspace_seq_capacity_hint = (
            workspace_seq_capacity
        )

    def _ensure_decode_active_num_partitions(self) -> torch.Tensor:
        if self._decode_active_num_partitions is None:
            self._decode_active_num_partitions = torch.empty(
                (1,),
                dtype=torch.int32,
                device=self.device,
            )
        return self._decode_active_num_partitions

    def _update_decode_active_num_partitions(
        self,
        attn_metadata: TritonAttentionMetadata,
        *,
        stage: str,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.flash_v100_decode_active_num_partitions = None
        if not _routing._decode_dynamic_partitions_enabled():
            return

        max_seq_len_hint = getattr(
            attn_metadata,
            "flash_v100_decode_max_seq_len_hint",
            None,
        )
        if max_seq_len_hint is None:
            return

        if (
            getattr(
                attn_metadata,
                "flash_v100_decode_workspace_seq_capacity_hint",
                None,
            )
            is None
            and self._decode_active_num_partitions is None
        ):
            return

        partition_size = _routing._decode_partition_size_for_metadata(
            int(max_seq_len_hint)
        )
        active = max(1, (int(max_seq_len_hint) + partition_size - 1) // partition_size)
        active_num_partitions = self._ensure_decode_active_num_partitions()
        active_num_partitions.fill_(active)
        flash_metadata.flash_v100_decode_active_num_partitions = active_num_partitions
        _routing._trace_decode_active_metadata(
            stage=stage,
            max_seq_len_hint=int(max_seq_len_hint),
            workspace_seq_capacity_hint=getattr(
                attn_metadata,
                "flash_v100_decode_workspace_seq_capacity_hint",
                None,
            ),
            static_decode_seq_hint=getattr(
                attn_metadata,
                "flash_v100_static_decode_seq_hint",
                None,
            ),
            active=active,
            partition_size=partition_size,
        )

    def _debug_draft_metadata(
        self,
        stage: str,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
    ) -> None:
        if (
            not self._is_speculative_draft_model
            or not _debug._draft_graph_debug_enabled()
        ):
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

    def _ensure_flash_draft_graph_buffers(
        self,
        required_reqs: int,
        block_table: torch.Tensor,
    ) -> bool:
        req_capacity = max(
            int(self.vllm_config.scheduler_config.max_num_seqs),
            int(required_reqs),
            1,
        )
        block_cols = int(block_table.shape[1])
        shape = (req_capacity, block_cols)
        if self._flash_draft_buffer_shape == shape:
            return True

        if self._flash_draft_buffer_shape is not None:
            old_reqs, old_block_cols = self._flash_draft_buffer_shape
            return required_reqs <= old_reqs and block_cols == old_block_cols

        self._draft_block_table = torch.empty(
            (req_capacity, block_cols),
            dtype=torch.int32,
            device=self.device,
        )
        self._draft_seq_lens = torch.empty(
            (req_capacity,),
            dtype=torch.int32,
            device=self.device,
        )
        self._draft_query_start_loc = torch.empty(
            (req_capacity + 1,),
            dtype=torch.int32,
            device=self.device,
        )
        self._flash_draft_buffer_shape = shape
        return True

    def _stabilize_draft_graph_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
    ) -> None:
        num_reqs = int(common_attn_metadata.num_reqs)
        if num_reqs <= 0:
            return

        block_table = attn_metadata.block_table[:num_reqs]
        if not self._ensure_flash_draft_graph_buffers(num_reqs, block_table):
            assert self._flash_draft_buffer_shape is not None
            req_capacity, block_cols = self._flash_draft_buffer_shape
            raise RuntimeError(
                "FLASH_ATTN_V100 draft CUDA graph metadata shape exceeds "
                "the captured persistent buffer capacity: "
                f"required_reqs={num_reqs}, "
                f"required_block_cols={int(block_table.shape[1])}, "
                f"capacity_reqs={req_capacity}, "
                f"capacity_block_cols={block_cols}. "
                "Replay would otherwise read stale draft metadata."
            )

        assert self._draft_block_table is not None
        assert self._draft_seq_lens is not None
        assert self._draft_query_start_loc is not None

        self.copy_dflash_graph_metadata(
            block_table,
            attn_metadata.seq_lens[:num_reqs],
            attn_metadata.query_start_loc[: num_reqs + 1],
        )

        attn_metadata.block_table = self._draft_block_table[:num_reqs]
        attn_metadata.seq_lens = self._draft_seq_lens[:num_reqs]
        attn_metadata.query_start_loc = self._draft_query_start_loc[: num_reqs + 1]

    def copy_dflash_graph_metadata(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> None:
        """Refresh the three persistent inputs of a non-causal DFlash graph."""
        num_reqs = seq_lens.numel()
        assert self._draft_block_table is not None
        assert self._draft_seq_lens is not None
        assert self._draft_query_start_loc is not None
        self._draft_block_table[:num_reqs].copy_(block_table, non_blocking=True)
        self._draft_seq_lens[:num_reqs].copy_(
            seq_lens,
            non_blocking=True,
        )
        self._draft_query_start_loc[: num_reqs + 1].copy_(
            query_start_loc,
            non_blocking=True,
        )

    def _configured_smallq_max_query_len(self) -> int:
        return int(os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16"))

    def _configured_smallq_max_model_len(self) -> int:
        return int(os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN", "0"))

    def _smallq_buffer_token_capacity(self, required_tokens: int) -> int:
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

    def _ensure_smallq_decode_buffers(
        self,
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
        shape = (token_capacity, req_capacity, block_cols)
        if self._smallq_buffer_shape == shape:
            return True

        if self._smallq_buffer_shape is not None:
            old_tokens, old_reqs, old_block_cols = self._smallq_buffer_shape
            return (
                required_tokens <= old_tokens
                and required_reqs <= old_reqs
                and block_cols == old_block_cols
            )

        self._smallq_decode_block_table = torch.empty(
            (token_capacity, block_cols),
            dtype=torch.int32,
            device=self.device,
        )
        self._smallq_decode_seq_lens = torch.empty(
            (token_capacity,),
            dtype=torch.int32,
            device=self.device,
        )
        self._smallq_query_start_loc = torch.empty(
            (req_capacity + 1,),
            dtype=torch.int32,
            device=self.device,
        )
        self._smallq_token_indices = torch.arange(
            token_capacity,
            dtype=torch.int32,
            device=self.device,
        )
        self._smallq_buffer_shape = shape
        return True

    def _clear_smallq_decode_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.smallq_decode_block_table = None
        flash_metadata.smallq_decode_seq_lens = None
        flash_metadata.smallq_query_start_loc = None
        flash_metadata.smallq_decode_max_seq_len_hint = None
        flash_metadata.smallq_decode_workspace_seq_capacity_hint = None
        flash_metadata.smallq_decode_partition_size_hint = None

    def _attach_prepared_dflash2_smallq_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        prepared: _smallq_metadata.DFlash2SmallQPreparedMetadata,
    ) -> None:
        """Attach buffers refreshed by the cross-cache-group launch."""
        if prepared.builder_id != id(self):
            raise ValueError("grouped small-query metadata belongs to another builder")
        if (
            self._smallq_decode_block_table is None
            or self._smallq_decode_seq_lens is None
            or self._smallq_query_start_loc is None
            or self._smallq_buffer_shape is None
        ):
            raise RuntimeError("grouped small-query metadata has no persistent buffers")
        token_capacity, req_capacity, _ = self._smallq_buffer_shape
        if (
            prepared.num_query_tokens > token_capacity
            or prepared.num_reqs > req_capacity
        ):
            raise RuntimeError("grouped small-query metadata exceeds captured capacity")

        self._clear_smallq_decode_metadata(attn_metadata)
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        flash_metadata.smallq_decode_block_table = self._smallq_decode_block_table[
            : prepared.num_query_tokens
        ]
        flash_metadata.smallq_decode_seq_lens = self._smallq_decode_seq_lens[
            : prepared.num_query_tokens
        ]
        flash_metadata.smallq_query_start_loc = self._smallq_query_start_loc[
            : prepared.num_reqs + 1
        ]
        flash_metadata.smallq_decode_max_seq_len_hint = prepared.max_seq_len_hint
        flash_metadata.smallq_decode_workspace_seq_capacity_hint = (
            prepared.workspace_seq_capacity_hint
        )
        flash_metadata.smallq_decode_partition_size_hint = prepared.partition_size_hint

    def _update_smallq_decode_metadata(
        self,
        attn_metadata: TritonAttentionMetadata,
        common_attn_metadata,
        *,
        force: bool = False,
        workspace_seq_capacity_cap: int | None = None,
        partition_size_hint: int | None = None,
    ) -> None:
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        profile_enabled = _debug._dflash_ddtree_worker_profile_enabled()
        profile_t0 = time.perf_counter() if profile_enabled else 0.0
        profile_stage_t0 = profile_t0
        self._clear_smallq_decode_metadata(attn_metadata)
        clear_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
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
        if not force and not has_prefix_context and self._smallq_buffer_shape is None:
            return

        num_query_tokens = int(attn_metadata.num_actual_tokens)
        num_reqs = int(common_attn_metadata.num_reqs)
        if num_query_tokens <= 0 or num_reqs <= 0:
            return

        block_table = attn_metadata.block_table[:num_reqs]
        guard_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        if not self._ensure_smallq_decode_buffers(
            num_query_tokens,
            num_reqs,
            block_table,
        ):
            assert self._smallq_buffer_shape is not None
            token_capacity, req_capacity, block_cols = self._smallq_buffer_shape
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
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        assert self._smallq_decode_block_table is not None
        assert self._smallq_decode_seq_lens is not None
        assert self._smallq_query_start_loc is not None
        assert self._smallq_token_indices is not None

        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        query_start_loc = attn_metadata.query_start_loc[: num_reqs + 1]
        real_num_query_tokens = int(query_start_loc_cpu[-1].item())
        if real_num_query_tokens > num_query_tokens:
            return
        padding_tokens = num_query_tokens - real_num_query_tokens
        seq_lens = attn_metadata.seq_lens[:num_reqs]
        prep_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        if self._use_sm70_dflash2_fused_smallq_metadata:
            _smallq_metadata._sm70_prepare_smallq_decode_metadata(
                self._smallq_decode_block_table,
                self._smallq_decode_seq_lens,
                self._smallq_query_start_loc,
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
            token_indices = self._smallq_token_indices[:num_query_tokens].to(
                dtype=seq_lens.dtype
            )
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
            self._smallq_decode_block_table[:num_query_tokens].copy_(
                decode_block_table,
                non_blocking=True,
            )
            self._smallq_decode_seq_lens[:num_query_tokens].copy_(
                decode_seq_lens,
                non_blocking=True,
            )
            self._smallq_query_start_loc[: num_reqs + 1].copy_(
                query_start_loc,
                non_blocking=True,
            )
            copy_ms = (
                (time.perf_counter() - profile_stage_t0) * 1000.0
                if profile_enabled
                else 0.0
            )

        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        flash_metadata.smallq_decode_block_table = self._smallq_decode_block_table[
            :num_query_tokens
        ]
        flash_metadata.smallq_decode_seq_lens = self._smallq_decode_seq_lens[
            :num_query_tokens
        ]
        flash_metadata.smallq_query_start_loc = self._smallq_query_start_loc[
            : num_reqs + 1
        ]
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
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
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
        if _debug._draft_graph_debug_enabled():
            _debug._graph_metadata_debug_log(
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
                _debug._format_tensor_debug(attn_metadata.query_start_loc, "attn_qsl"),
                _debug._format_tensor_debug(attn_metadata.seq_lens, "attn_seq"),
                _debug._format_tensor_debug(attn_metadata.block_table, "attn_bt"),
                _debug._format_tensor_debug(
                    flash_metadata.smallq_decode_block_table,
                    "smallq_bt",
                ),
                _debug._format_tensor_debug(
                    flash_metadata.smallq_decode_seq_lens,
                    "smallq_seq",
                ),
                _debug._format_tensor_debug(
                    flash_metadata.smallq_query_start_loc,
                    "smallq_qsl",
                ),
            )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        capture_seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        capture_seq_lens_cpu = (
            capture_seq_lens_cpu.clone()
            if capture_seq_lens_cpu is not None
            else common_attn_metadata.seq_lens.detach().cpu().clone()
        )
        attn_metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        self._attach_common_flash_metadata(attn_metadata, common_attn_metadata)
        flash_metadata = _as_flash_v100_metadata(attn_metadata)
        self._attach_prefix_anchored_metadata(attn_metadata, common_attn_metadata)
        flash_metadata.seq_lens_cpu = capture_seq_lens_cpu

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
            if (
                workspace_seq_capacity_cap is not None
                and workspace_seq_capacity_cap
                < int(self.vllm_config.model_config.max_model_len)
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
        self._debug_draft_metadata(
            "capture",
            attn_metadata,
            common_attn_metadata,
        )
        self._attach_decode_shape_hints(
            attn_metadata,
            common_attn_metadata,
            static_decode=True,
        )
        flash_metadata.flash_v100_cudagraph_capture = True
        self._update_decode_active_num_partitions(attn_metadata, stage="capture")

        return attn_metadata

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build: bool = False,
        ddtree_parent_ids: torch.Tensor | None = None,
        ddtree_num_tree_tokens_cpu: torch.Tensor | None = None,
        prepared_dflash2_smallq_metadata: (
            _smallq_metadata.DFlash2SmallQPreparedMetadata | None
        ) = None,
    ):
        attn_metadata = super().build(
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
            getattr(attn_metadata, "max_query_len", 1) == 1
            or self._is_dflash_draft_model
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
                        int(getattr(common_attn_metadata, "max_seq_len", 0) or 0)
                        or None
                    ),
                )
        self._attach_decode_shape_hints(attn_metadata, common_attn_metadata)
        self._update_decode_active_num_partitions(attn_metadata, stage="build")
        self._debug_draft_metadata("build", attn_metadata, common_attn_metadata)
        return attn_metadata

    def build_for_drafting(self, common_attn_metadata, draft_index: int):
        profile_enabled = _debug._dflash_ddtree_worker_profile_enabled()
        profile_t0 = time.perf_counter() if profile_enabled else 0.0
        profile_stage_t0 = profile_t0
        attn_metadata = super().build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )
        super_build_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        self._attach_common_flash_metadata(attn_metadata, common_attn_metadata)
        attach_common_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        self._stabilize_draft_graph_metadata(attn_metadata, common_attn_metadata)
        stabilize_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
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
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        self._attach_decode_shape_hints(attn_metadata, common_attn_metadata)
        shape_hints_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        self._update_decode_active_num_partitions(
            attn_metadata,
            stage=f"draft{draft_index}",
        )
        active_partitions_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )
        profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
        self._debug_draft_metadata(
            f"draft{draft_index}",
            attn_metadata,
            common_attn_metadata,
        )
        debug_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
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
