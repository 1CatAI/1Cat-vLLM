# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 attention metadata, its builder and mixed decode-row plans."""

from __future__ import annotations

from contextlib import suppress
from typing import cast

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100.spec.hooks import (
    METADATA_HOOKS,
    SpecMetadataFields,
    SpecMetadataMethods,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import PrefixAnchoredSWASpec

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


class FlashAttnV100Metadata(SpecMetadataFields, TritonAttentionMetadata):
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
    smallq_decode_block_table: torch.Tensor | None
    smallq_decode_seq_lens: torch.Tensor | None
    smallq_query_start_loc: torch.Tensor | None
    smallq_decode_max_seq_len_hint: int | None
    smallq_decode_workspace_seq_capacity_hint: int | None
    smallq_decode_partition_size_hint: int | None


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


class FlashAttnV100MetadataBuilder(SpecMetadataMethods, TritonAttentionMetadataBuilder):
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
        METADATA_HOOKS.initialize(self, spec_config)
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
        METADATA_HOOKS.attach_common(self, attn_metadata)
        flash_metadata.max_model_len = self.vllm_config.model_config.max_model_len
        flash_metadata.flash_v100_cudagraph_capture = False
        flash_metadata.flash_v100_batch_context_routing = (
            _routing._batch_context_routing_for_graph_variant(
                self._batch_context_routing_enabled,
                getattr(common_attn_metadata, "cudagraph_graph_variant", None),
            )
        )

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

        METADATA_HOOKS.prepare_capture(self, attn_metadata, common_attn_metadata)
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


# Super calls extracted into the feature mixin continue after that mixin.
_spec_builder_super_owner = SpecMetadataMethods
