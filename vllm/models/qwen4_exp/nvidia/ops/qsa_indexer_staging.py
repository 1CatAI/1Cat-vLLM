# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, per-engine indexer staging; shared physical pages are copied once.

The host history remains authoritative. Every selection rebuilds the page
plan and copies live keys, so tentative overwrites and page reuse cannot leave
a stale device copy. Pages outside the bounded bank retain the paged reader.
Model execution orders borrowers on its stream; no staged views escape.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _plan_pages(
    Table,
    Requests,
    Positions,
    Lengths,
    Slots,
    Pages,
    Count,
    TableStride: tl.constexpr,
    TableColumnStride: tl.constexpr,
    Width: tl.constexpr,
    Rows: tl.constexpr,
    CachePages: tl.constexpr,
    Capacity: tl.constexpr,
    Page: tl.constexpr,
    Columns: tl.constexpr,
    Compress: tl.constexpr,
    BlockRows: tl.constexpr,
):
    request = tl.program_id(1)
    rows = tl.arange(0, BlockRows)
    req = tl.load(Requests + rows, rows < Rows, other=-1)
    position = tl.load(Positions + rows, rows < Rows, other=-1)
    length = tl.load(Lengths + request)
    end = tl.max(tl.where(req == request, tl.minimum(position + 1, length), 0), 0)
    end = tl.minimum(tl.maximum(end // Compress, 0), Columns)
    logical = tl.program_id(0) * 64 + tl.arange(0, 64)
    physical = tl.load(
        Table + request * TableStride + logical * TableColumnStride,
        (logical < Width) & (logical * Page < end),
        other=-1,
    )
    valid = (physical >= 0) & (physical < CachePages)
    old = tl.atomic_cas(
        Slots + tl.where(valid, physical, CachePages),
        tl.full((64,), -1, tl.int32),
        tl.full((64,), -2, tl.int32),
        sem="relaxed",
    )
    claimed = valid & (old == -1)
    slot = tl.atomic_add(
        Count + tl.zeros((64,), tl.int32), claimed.to(tl.int32), sem="relaxed"
    )
    tl.store(Pages + slot, physical, claimed & (slot < Capacity))
    tl.store(
        Slots + tl.maximum(physical, 0),
        tl.where(slot < Capacity, slot, -3),
        claimed,
    )


@triton.jit
def _copy_pages(
    Cache,
    Pages,
    Count,
    Staged,
    StridePage: tl.constexpr,
    StrideToken: tl.constexpr,
    StrideDim: tl.constexpr,
    Page: tl.constexpr,
    Capacity: tl.constexpr,
):
    slot = tl.program_id(0)
    count = tl.load(Count)
    if slot < tl.minimum(count, Capacity):
        physical = tl.load(Pages + slot)
        token = tl.program_id(1) * 16 + tl.arange(0, 16)
        dim = tl.arange(0, 128)
        key = tl.load(
            Cache
            + physical.to(tl.int64) * StridePage
            + token[:, None] * StrideToken
            + dim[None, :] * StrideDim,
            (token < Page)[:, None],
            other=0.0,
        )
        tl.store(
            Staged + (slot * Page + token[:, None]) * 128 + dim[None, :],
            key,
            (token < Page)[:, None],
        )


class QSAIndexerStagingWorkspace:
    """One bounded bank borrowed by successive indexer layers of an engine."""

    def __init__(self, tokens: int = 32768):
        if tokens <= 0:
            raise ValueError("Indexer staging capacity must be positive")
        self.tokens = tokens
        self.banks: dict[tuple, tuple[torch.Tensor, ...]] = {}

    def prepare(self, cache, table, requests, positions, lengths, compress, columns):
        page = cache.shape[1]
        capacity = min(cache.shape[0], triton.cdiv(self.tokens, page))
        key = (cache.device, cache.shape[0], page)
        if key not in self.banks:
            self.banks[key] = (
                torch.empty(
                    (capacity, page, 1, 128), dtype=torch.float16, device=cache.device
                ),
                torch.empty(cache.shape[0] + 1, dtype=torch.int32, device=cache.device),
                torch.empty(capacity, dtype=torch.int32, device=cache.device),
                torch.empty(1, dtype=torch.int32, device=cache.device),
            )
        staged, slots, pages, count = self.banks[key]
        slots.fill_(-1)
        slots[-1:].fill_(-3)
        count.zero_()
        _plan_pages[(triton.cdiv(table.shape[1], 64), table.shape[0])](
            table,
            requests,
            positions,
            lengths,
            slots,
            pages,
            count,
            table.stride(0),
            table.stride(1),
            table.shape[1],
            requests.numel(),
            cache.shape[0],
            capacity,
            page,
            columns,
            compress,
            triton.next_power_of_2(requests.numel()),
            num_warps=4,
        )
        _copy_pages[(capacity, triton.cdiv(page, 16))](
            cache,
            pages,
            count,
            staged,
            cache.stride(0),
            cache.stride(1),
            cache.stride(3),
            page,
            capacity,
            num_warps=4,
        )
        return staged, slots

    def close(self):
        self.banks.clear()
