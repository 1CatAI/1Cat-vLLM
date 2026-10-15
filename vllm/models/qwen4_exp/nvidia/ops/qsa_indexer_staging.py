# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, per-engine indexer staging; shared physical pages are copied once.

The host history remains authoritative. The default bank refreshes live pages
on every selection. The optional per-layer bank requires bound write tracking
and refreshes changed versions; collisions borrow the default refresh bank.
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
    Excluded,
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
    Exclude: tl.constexpr,
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
    if Exclude:
        retained = tl.load(Excluded + tl.maximum(physical, 0), valid, other=-3)
        valid = valid & (retained < 0)
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
        self.persistent_layers: dict[str, QSAIndexerPersistentLayer] = {}

    def prepare(
        self,
        cache,
        table,
        requests,
        positions,
        lengths,
        compress,
        columns,
        excluded=None,
    ):
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
            excluded if excluded is not None else slots,
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
            excluded is not None,
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

    def persistent_layer(self, name):
        if name not in self.persistent_layers:
            self.persistent_layers[name] = QSAIndexerPersistentLayer(self)
        return self.persistent_layers[name]

    def close(self):
        for layer in self.persistent_layers.values():
            layer.close()
        self.persistent_layers.clear()
        self.banks.clear()


# Version publication is ordered after the entire page-copy kernel. Every
# model writer bumps its page version, including speculative writes whose
# visibility may later roll back. Hash collisions use the shared refresh bank
# and its existing paged overflow, rather than treating a stale slot as valid.
@triton.jit
def _touch_pages(
    Slots,
    Versions,
    Rows: tl.constexpr,
    Page: tl.constexpr,
    CachePages: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    slot = tl.load(Slots + i, i < Rows, other=-1)
    page = slot // Page
    valid = (i < Rows) & (slot >= 0) & (page < CachePages)
    tl.atomic_add(Versions + tl.where(valid, page, 0), 1, valid)


@triton.jit
def _claim_pages(
    Table,
    Requests,
    Positions,
    Lengths,
    Slots,
    Lease,
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
    pos = tl.load(Positions + rows, rows < Rows, other=-1)
    length = tl.load(Lengths + request)
    end = tl.max(tl.where(req == request, tl.minimum(pos + 1, length), 0), 0)
    end = tl.minimum(tl.maximum(end // Compress, 0), Columns)
    logical = tl.program_id(0) * 64 + tl.arange(0, 64)
    physical = tl.load(
        Table + request * TableStride + logical * TableColumnStride,
        (logical < Width) & (logical * Page < end),
        other=-1,
    )
    valid = (physical >= 0) & (physical < CachePages)
    slot = tl.where(valid, physical % Capacity, Capacity)
    old = tl.atomic_cas(Lease + slot, tl.full((64,), -1, tl.int32), physical)
    admitted = valid & ((old == -1) | (old == physical))
    tl.store(Slots + tl.maximum(physical, 0), tl.where(admitted, slot, -3), valid)


@triton.jit
def _copy_changed(
    Cache,
    Lease,
    Versions,
    Tags,
    Stamps,
    Bank,
    StridePage: tl.constexpr,
    StrideToken: tl.constexpr,
    StrideDim: tl.constexpr,
    Page: tl.constexpr,
):
    slot = tl.program_id(0)
    physical = tl.load(Lease + slot)
    if physical >= 0:
        version = tl.load(Versions + physical)
        oldpage = tl.load(Tags + slot)
        oldversion = tl.load(Stamps + slot)
        if (oldpage != physical) | (oldversion != version):
            row = tl.program_id(1) * 16 + tl.arange(0, 16)
            dim = tl.arange(0, 128)
            key = tl.load(
                Cache
                + physical.to(tl.int64) * StridePage
                + row[:, None] * StrideToken
                + dim[None, :] * StrideDim,
                (row < Page)[:, None],
                other=0.0,
            )
            tl.store(
                Bank + (slot * Page + row[:, None]) * 128 + dim[None, :],
                key,
                (row < Page)[:, None],
            )


@triton.jit
def _publish_copies(
    Lease, Versions, Tags, Stamps, Copied, Capacity: tl.constexpr, B: tl.constexpr
):
    slot = tl.program_id(0) * B + tl.arange(0, B)
    physical = tl.load(Lease + slot, slot < Capacity, other=-1)
    active = (slot < Capacity) & (physical >= 0)
    version = tl.load(Versions + tl.maximum(physical, 0), active, other=0)
    oldpage = tl.load(Tags + slot, slot < Capacity, other=-1)
    oldversion = tl.load(Stamps + slot, slot < Capacity, other=0)
    tl.store(
        Copied + slot,
        active & ((oldpage != physical) | (oldversion != version)),
        slot < Capacity,
    )
    # Publication follows the completed copy kernel on the borrower's stream.
    tl.store(Tags + slot, physical, active)
    tl.store(Stamps + slot, version, active)


class QSAIndexerPersistentLayer:
    """A bounded layer bank; every compressed-key writer must call ``touch``.

    Versions never roll back with speculative visibility. Rebinding, including
    rebinding the same tensor, invalidates all tags before new graph capture.
    """

    def __init__(self, owner: QSAIndexerStagingWorkspace, tokens: int = 16384):
        if tokens <= 0:
            raise ValueError("Indexer staging capacity must be positive")
        self.tokens = tokens
        self.cache: torch.Tensor | None = None
        self.owner = owner
        self.closed = False

    def bind(self, cache: torch.Tensor) -> None:
        if self.closed:
            raise RuntimeError("Indexer staging owner is closed")
        self.cache = None
        if (
            cache.device.type != "cuda"
            or cache.dtype != torch.float16
            or cache.ndim != 4
            or cache.shape[2:] != (1, 128)
            or not cache.shape[0]
            or not cache.shape[1]
        ):
            return
        self.cache = cache
        self.page, self.pages = cache.shape[1], cache.shape[0]
        self.capacity = min(self.pages, triton.cdiv(self.tokens, self.page))
        device = cache.device
        self.bank = torch.empty(
            (self.capacity, self.page, 1, 128), device=device, dtype=torch.float16
        )
        self.versions = torch.zeros(self.pages, device=device, dtype=torch.int64)
        self.slots = torch.empty(self.pages + 1, device=device, dtype=torch.int32)
        self.lease = torch.empty(self.capacity + 1, device=device, dtype=torch.int32)
        self.tags = torch.full((self.capacity,), -1, device=device, dtype=torch.int32)
        self.stamps = torch.zeros(self.capacity, device=device, dtype=torch.int64)
        self.copied = torch.empty(self.capacity, device=device, dtype=torch.int32)

    def touch(self, slots: torch.Tensor) -> None:
        if self.cache is not None and slots.numel():
            _touch_pages[(triton.cdiv(slots.numel(), 256),)](
                slots, self.versions, slots.numel(), self.page, self.pages, 256
            )

    def prepare(self, cache, table, requests, positions, lengths, compress, columns):
        if self.closed:
            raise RuntimeError("Indexer staging owner is closed")
        if cache is not self.cache:
            return self.owner.prepare(
                cache, table, requests, positions, lengths, compress, columns
            )
        self.slots.fill_(-3)
        self.lease.fill_(-1)
        self.lease[-1:].fill_(-2)
        _claim_pages[(triton.cdiv(table.shape[1], 64), table.shape[0])](
            table,
            requests,
            positions,
            lengths,
            self.slots,
            self.lease,
            table.stride(0),
            table.stride(1),
            table.shape[1],
            requests.numel(),
            self.pages,
            self.capacity,
            self.page,
            columns,
            compress,
            triton.next_power_of_2(requests.numel()),
            num_warps=4,
        )
        _copy_changed[(self.capacity, triton.cdiv(self.page, 16))](
            cache,
            self.lease,
            self.versions,
            self.tags,
            self.stamps,
            self.bank,
            cache.stride(0),
            cache.stride(1),
            cache.stride(3),
            self.page,
            num_warps=4,
        )
        _publish_copies[(triton.cdiv(self.capacity, 256),)](
            self.lease,
            self.versions,
            self.tags,
            self.stamps,
            self.copied,
            self.capacity,
            256,
            num_warps=4,
        )
        spill = self.owner.prepare(
            cache,
            table,
            requests,
            positions,
            lengths,
            compress,
            columns,
            excluded=self.slots,
        )
        return self.bank, self.slots, *spill

    def close(self) -> None:
        self.cache = None
        self.closed = True
        for name in ("bank", "versions", "slots", "lease", "tags", "stamps", "copied"):
            if hasattr(self, name):
                delattr(self, name)
