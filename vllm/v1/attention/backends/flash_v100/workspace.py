# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 mutable attention workspaces."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import cast

import torch

from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata


@dataclass
class DecodeCache:
    """Per-layer dense decode cache, invalidated at the original prefill sites."""

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    length: int = 0
    capacity: int = 0

    def invalidate(self) -> None:
        self.key = None
        self.value = None
        self.length = 0
        self.capacity = 0

    def ensure_capacity(
        self,
        required_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if (
            self.key is not None
            and self.value is not None
            and self.capacity >= required_len
            and self.key.shape[1] == num_kv_heads
            and self.key.shape[2] == head_dim
            and self.key.dtype == dtype
            and self.key.device == device
        ):
            return

        new_capacity = max(required_len, max(16, self.capacity * 2))
        new_k = torch.empty(
            (new_capacity, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        new_v = torch.empty(
            (new_capacity, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        if self.key is not None and self.value is not None and self.length > 0:
            new_k[: self.length].copy_(self.key[: self.length])
            new_v[: self.length].copy_(self.value[: self.length])

        self.key = new_k
        self.value = new_v
        self.capacity = new_capacity

    def get_kv_single_seq(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        seq_lens_cpu: torch.Tensor,
        block_size: int,
        head_dim: int,
        *,
        extract: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(seq_lens_cpu[0])
        q_len = int(attn_metadata.num_actual_tokens)
        num_kv_heads = key.shape[1]

        cache_hit = (
            self.key is not None
            and self.value is not None
            and seq_len > self.length
            and seq_len - q_len == self.length
        )

        if not cache_hit:
            k_cont, v_cont = extract(
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                total_tokens=seq_len,
            )
            self.ensure_capacity(
                seq_len,
                num_kv_heads,
                head_dim,
                k_cont.dtype,
                k_cont.device,
            )
            assert self.key is not None
            assert self.value is not None
            self.key[:seq_len].copy_(k_cont)
            self.value[:seq_len].copy_(v_cont)
            self.length = seq_len
            return (
                self.key[:seq_len],
                self.value[:seq_len],
            )

        self.ensure_capacity(
            seq_len,
            num_kv_heads,
            head_dim,
            key.dtype,
            key.device,
        )
        assert self.key is not None
        assert self.value is not None
        self.key[self.length : seq_len].copy_(key[:q_len])
        self.value[self.length : seq_len].copy_(value[:q_len])
        self.length = seq_len
        return (
            self.key[:seq_len],
            self.value[:seq_len],
        )


@dataclass
class V100Workspace:
    """Mutable state owned by one attention layer, separate from its policy."""

    decode_cache: DecodeCache = field(default_factory=DecodeCache)


_MIXED_ROWS_PLAN_ATTR = "_flash_v100_mixed_decode_rows_plan"
MIXED_ROWS_GROUP = 8


class MixedDecodeRowsPlan:
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
            base = len(group_req) * MIXED_ROWS_GROUP
            group_req.extend([i] * -(-q_len // MIXED_ROWS_GROUP))
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
                self.num_groups * MIXED_ROWS_GROUP,
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


def mixed_decode_rows_plan(
    attn_metadata: TritonAttentionMetadata,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    device: torch.device,
) -> MixedDecodeRowsPlan | None:
    """Select the resident decode/verification rows of a mixed batch.

    Returns ``None`` when the batch has no such row, or only such rows (that
    shape is the uniform small-query batch and has its own route). The result is
    cached on the step's metadata object, which every layer of the group shares.
    """
    cached = getattr(attn_metadata, _MIXED_ROWS_PLAN_ATTR, False)
    if cached is not False:
        return cast(MixedDecodeRowsPlan | None, cached)
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
        MixedDecodeRowsPlan(rows, qsl, seq_lens_host, device)
        if rows and len(rows) != num_seqs
        else None
    )
    with suppress(AttributeError):
        setattr(attn_metadata, _MIXED_ROWS_PLAN_ATTR, plan)
    return plan


# Compatibility exports for the legacy forwarding module.
_MixedDecodeRowsPlan = MixedDecodeRowsPlan
_MIXED_ROWS_GROUP = MIXED_ROWS_GROUP
_mixed_decode_rows_plan = mixed_decode_rows_plan


@dataclass
class DraftBuffers:
    """Persistent metadata allocation; a captured allocation never moves."""

    block_table: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    shape: tuple[int, int] | None = None

    def ensure(
        self,
        req_capacity: int,
        block_cols: int,
        required_reqs: int,
        device: torch.device,
    ) -> bool:
        shape = (req_capacity, block_cols)
        if self.shape == shape:
            return True

        if self.shape is not None:
            old_reqs, old_block_cols = self.shape
            return required_reqs <= old_reqs and block_cols == old_block_cols

        self.block_table = torch.empty(
            (req_capacity, block_cols),
            dtype=torch.int32,
            device=device,
        )
        self.seq_lens = torch.empty(
            (req_capacity,),
            dtype=torch.int32,
            device=device,
        )
        self.query_start_loc = torch.empty(
            (req_capacity + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.shape = shape
        return True

    def copy_metadata(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> None:
        """Refresh the three persistent inputs of a non-causal graph."""
        num_reqs = seq_lens.numel()
        assert self.block_table is not None
        assert self.seq_lens is not None
        assert self.query_start_loc is not None
        self.block_table[:num_reqs].copy_(block_table, non_blocking=True)
        self.seq_lens[:num_reqs].copy_(
            seq_lens,
            non_blocking=True,
        )
        self.query_start_loc[: num_reqs + 1].copy_(
            query_start_loc,
            non_blocking=True,
        )


@dataclass
class SmallQueryBuffers:
    """Persistent metadata allocation; a captured allocation never moves."""

    block_table: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    token_indices: torch.Tensor | None = None
    shape: tuple[int, int, int] | None = None

    def ensure(
        self,
        token_capacity: int,
        req_capacity: int,
        block_cols: int,
        required_tokens: int,
        required_reqs: int,
        device: torch.device,
    ) -> bool:
        shape = (token_capacity, req_capacity, block_cols)
        if self.shape == shape:
            return True

        if self.shape is not None:
            old_tokens, old_reqs, old_block_cols = self.shape
            return (
                required_tokens <= old_tokens
                and required_reqs <= old_reqs
                and block_cols == old_block_cols
            )

        self.block_table = torch.empty(
            (token_capacity, block_cols),
            dtype=torch.int32,
            device=device,
        )
        self.seq_lens = torch.empty(
            (token_capacity,),
            dtype=torch.int32,
            device=device,
        )
        self.query_start_loc = torch.empty(
            (req_capacity + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.token_indices = torch.arange(
            token_capacity,
            dtype=torch.int32,
            device=device,
        )
        self.shape = shape
        return True


@dataclass
class MetadataWorkspace:
    """Persistent inputs owned by one builder across capture and replay."""

    draft: DraftBuffers = field(default_factory=DraftBuffers)
    smallq: SmallQueryBuffers = field(default_factory=SmallQueryBuffers)


def allocate_growing_workspace(
    allocate: Callable[[], tuple[torch.Tensor, ...]],
    *,
    on_cuda: bool,
) -> tuple[torch.Tensor, ...] | None:
    """Allocate a grown workspace, retrying once after releasing cached blocks.

    The caller must drop its reference to the previous workspace *before*
    calling this, otherwise the old and the new buffer are resident at the same
    time and the growth can fail on memory its own predecessor is holding.
    Freed segments are smaller than the grown request, so a retry after
    ``empty_cache`` is what actually recovers the fragmented headroom.
    """
    try:
        return allocate()
    except torch.OutOfMemoryError:
        pass
    if on_cuda:
        torch.accelerator.empty_cache()
    try:
        return allocate()
    except torch.OutOfMemoryError:
        return None
