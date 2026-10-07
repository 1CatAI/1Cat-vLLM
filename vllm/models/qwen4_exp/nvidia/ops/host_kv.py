# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned, per-vector E4M3 QSA history with a bounded device hot-page cache.

The selector's positions and masks are preserved. Cache contention falls back
to authoritative host bytes, never spins or drops selected pages. All storage
and workspaces are allocated before graph capture.
"""

import torch

from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32_bitcast,
    fp32_to_fp8_e4m3fn_bits,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

_WORKSPACES: dict[tuple, tuple[torch.Tensor, ...]] = {}


@triton.jit
def _write(
    K,
    V,
    Slots,
    History,
    Scales,
    Tags,
    Epoch,
    Ks: tl.constexpr,
    Vs: tl.constexpr,
    Blocks: tl.constexpr,
    Page: tl.constexpr,
    Dim: tl.constexpr,
    Sets: tl.constexpr,
):
    row = tl.program_id(0)
    if row == 0:
        tl.atomic_add(Epoch, 1)
    slot = tl.load(Slots + row)
    if slot >= 0 and slot < Blocks * Page:
        dims = tl.arange(0, Dim)
        kv = tl.arange(0, 2)
        values = tl.where(
            kv[:, None] == 0,
            tl.load(K + row * Ks + dims[None, :]),
            tl.load(V + row * Vs + dims[None, :]),
        ).to(tl.float32)
        scales = tl.maximum(tl.div_rn(tl.max(tl.abs(values), 1), 448.0), 2.0**-126)
        codes = fp32_to_fp8_e4m3fn_bits(tl.div_rn(values, scales[:, None]))
        block, token = slot // Page, slot % Page
        tl.store(
            History + ((block * 2 + kv[:, None]) * Page + token) * Dim + dims[None, :],
            codes,
        )
        tl.store(Scales + slot * 2 + kv, scales)
        # Writes and gathers are ordered on the execution stream. Invalidate
        # the whole four-token page, including tentative MTP overwrites.
        page = (slot // 4).to(tl.int32)
        ways = tl.arange(0, 4)
        tl.atomic_cas(
            Tags + (page % Sets) * 4 + ways,
            tl.broadcast_to(page, (4,)),
            tl.full((4,), -1, tl.int32),
        )


@triton.jit
def _protect(
    Indices,
    Table,
    Requests,
    Positions,
    Lengths,
    Tags,
    Stamps,
    Epoch,
    Width: tl.constexpr,
    TableWidth: tl.constexpr,
    TableStride: tl.constexpr,
    IndexStride: tl.constexpr,
    NumRequests: tl.constexpr,
    Blocks: tl.constexpr,
    Page: tl.constexpr,
    Sets: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * 256 + tl.arange(0, 256)
    ways = tl.arange(0, 4)
    logical = tl.load(Indices + row * IndexStride + columns, columns < Width, other=-1)
    req = tl.load(Requests + row)
    position = tl.load(Positions + row)
    length = tl.load(
        Lengths + tl.maximum(req, 0), (req >= 0) & (req < NumRequests), other=0
    )
    valid = (columns < Width) & (logical >= 0) & (logical <= position)
    valid &= (logical < length) & (req >= 0) & (req < NumRequests)
    valid &= logical // Page < TableWidth
    blocks = tl.load(
        Table + tl.maximum(req, 0) * TableStride + tl.maximum(logical, 0) // Page,
        valid,
        other=-1,
    )
    valid &= (blocks >= 0) & (blocks < Blocks)
    pages = (tl.maximum(blocks, 0) * Page + tl.maximum(logical, 0) % Page) // 4
    slots = (pages[:, None] % Sets) * 4 + ways[None, :]
    tags = tl.load(Tags + slots)
    epoch = tl.load(Epoch)
    tl.store(Stamps + slots, epoch, valid[:, None] & (tags == pages[:, None]))


@triton.jit
def _gather(
    History,
    Scales,
    Codes,
    HotScales,
    Tags,
    Stamps,
    Epoch,
    Locks,
    Hands,
    Stats,
    Indices,
    Table,
    Requests,
    Positions,
    Lengths,
    Out,
    Remapped,
    Width: tl.constexpr,
    Padded: tl.constexpr,
    TableWidth: tl.constexpr,
    TableStride: tl.constexpr,
    IndexStride: tl.constexpr,
    NumRequests: tl.constexpr,
    Blocks: tl.constexpr,
    Page: tl.constexpr,
    Dim: tl.constexpr,
    Sets: tl.constexpr,
):
    row, group = tl.program_id(0), tl.program_id(1)
    lanes = tl.arange(0, 4)
    dims = tl.arange(0, Dim)
    kv = tl.arange(0, 2)
    columns = group * 4 + lanes
    logical = tl.load(Indices + row * IndexStride + columns, columns < Width, other=-1)
    req = tl.load(Requests + row)
    position = tl.load(Positions + row)
    length = tl.load(
        Lengths + tl.maximum(req, 0), (req >= 0) & (req < NumRequests), other=0
    )
    valid = (columns < Width) & (logical >= 0) & (logical <= position)
    valid &= (logical < length) & (req >= 0) & (req < NumRequests)
    blocks = tl.load(
        Table + tl.maximum(req, 0) * TableStride + tl.maximum(logical, 0) // Page,
        valid & (logical // Page < TableWidth),
        other=-1,
    )
    valid &= (logical // Page < TableWidth) & (blocks >= 0) & (blocks < Blocks)
    tokens = tl.maximum(blocks, 0) * Page + tl.maximum(logical, 0) % Page
    pages = tokens // 4
    first = tl.sum(tl.where(lanes == 0, pages, 0), 0)
    aligned = (
        tl.sum((valid & (pages == first) & (tokens % 4 == lanes)).to(tl.int32), 0) == 4
    )
    bucket = first % Sets
    held = tl.full((), False, tl.int1)
    hit = tl.full((), False, tl.int1)
    selected = tl.full((), 0, tl.int32)
    if aligned:
        ways = tl.arange(0, 4)
        epoch = tl.load(Epoch)
        tags = tl.load(Tags + bucket * 4 + ways)
        hit = tl.sum((tags == first).to(tl.int32), 0) != 0
        if hit:
            selected = tl.min(tl.where(tags == first, ways, 4), 0)
            # Existing hits were protected before this launch. Acquire a newly
            # published tag as well, so its bytes are visible across CTAs.
            hit = (
                tl.atomic_cas(Tags + bucket * 4 + selected, first, first, sem="acquire")
                == first
            )
        if not hit:
            # A failed acquisition uses host memory. No CTA waits for a CTA.
            held = tl.atomic_cas(Locks + bucket, 0, 1) == 0
            if held:
                tags = tl.load(Tags + bucket * 4 + ways)
                hit = tl.sum((tags == first).to(tl.int32), 0) != 0
                if hit:
                    selected = tl.min(tl.where(tags == first, ways, 4), 0)
                else:
                    stamps = tl.load(Stamps + bucket * 4 + ways)
                    hand = tl.load(Hands + bucket) % 4
                    priority = (ways - hand + 4) % 4
                    victim = tl.min(tl.where(stamps != epoch, priority, 4), 0)
                    if victim < 4:
                        selected = (hand + victim) % 4
                        tl.store(Hands + bucket, selected + 1)
                    else:
                        tl.atomic_xchg(Locks + bucket, 0)
                        held = False
    hot = bucket * 4 + selected
    if hit:
        codes = tl.load(
            Codes
            + ((hot * 4 + lanes[:, None, None]) * 2 + kv[None, :, None]) * Dim
            + dims[None, None, :]
        )
        scales = tl.load(HotScales + (hot * 4 + lanes[:, None]) * 2 + kv[None, :])
    else:
        codes = tl.load(
            History
            + (
                (
                    tl.maximum(blocks, 0).to(tl.int64)[:, None, None] * 2
                    + kv[None, :, None]
                )
                * Page
                + (tl.maximum(logical, 0) % Page)[:, None, None]
            )
            * Dim
            + dims[None, None, :],
            valid[:, None, None],
            other=0,
        )
        scales = tl.load(
            Scales + tokens[:, None] * 2 + kv[None, :], valid[:, None], other=0
        )
        if held:
            tl.store(
                Codes
                + ((hot * 4 + lanes[:, None, None]) * 2 + kv[None, :, None]) * Dim
                + dims[None, None, :],
                codes,
            )
            tl.store(HotScales + (hot * 4 + lanes[:, None]) * 2 + kv[None, :], scales)
            tl.debug_barrier()
            tl.store(Stamps + hot, tl.load(Epoch))
            tl.atomic_xchg(Tags + hot, first, sem="release")
    values = fp8_e4m3fn_bits_to_fp32_bitcast(codes) * scales[:, :, None]
    tl.store(
        Out
        + ((row * 2 + kv[None, :, None]) * Padded + columns[:, None, None]) * Dim
        + dims[None, None, :],
        values,
        columns[:, None, None] < Padded,
    )
    tl.store(
        Remapped + row * Width + columns, tl.where(valid, columns, -1), columns < Width
    )
    if held:
        tl.debug_barrier()
        tl.atomic_xchg(Locks + bucket, 0)
    # Each CTA owns its counters across ordered forwards. Avoid serializing
    # every page lookup through three global atomic counters.
    counter = (row * tl.cdiv(Width, 4) + group) * 3
    counters = tl.arange(0, 4)
    count = tl.sum(valid.to(tl.int64), 0)
    delta = tl.where(
        counters == 0,
        count * hit.to(tl.int64),
        tl.where(
            counters == 1,
            count * (~hit).to(tl.int64),
            (aligned & ~held & ~hit).to(tl.int64),
        ),
    )
    old = tl.load(Stats + counter + counters, counters < 3, other=0)
    tl.store(Stats + counter + counters, old + delta, counters < 3)


class HostQSAKV:
    """Own stable host history, a four-way hot cache and bounded staging."""

    def __init__(
        self,
        blocks: int,
        page_size: int,
        dim: int,
        device: torch.device,
        *,
        hot_tokens: int = 32768,
        rows: int = 32,
        width: int = 2051,
        history: torch.Tensor | None = None,
    ):
        if blocks <= 0 or page_size <= 0 or page_size % 4 or dim != 256:
            raise ValueError("Host QSA KV requires positive page4 geometry and D256")
        if hot_tokens <= 0 or hot_tokens % 16 or rows <= 0 or width <= 0:
            raise ValueError("Invalid hot cache or staging capacity")
        self.blocks, self.page_size, self.dim = blocks, page_size, dim
        self.rows, self.width = rows, width
        self.padded = triton.cdiv(width, 4) * 4
        self.sets = hot_tokens // 16
        self.host = None
        if history is None:
            self.host = torch.zeros(
                (blocks, 2, page_size, 1, dim), dtype=torch.uint8, pin_memory=True
            )
        elif history.shape != (blocks, 2, page_size, 1, dim) or (
            history.dtype != torch.uint8 or not history.is_contiguous()
        ):
            raise ValueError("Host history must use contiguous page-major E4M3 bytes")
        self.host_scales = torch.zeros(
            (blocks * page_size, 2), dtype=torch.float32, pin_memory=True
        )
        with torch.accelerator.device_index(device.index):
            self.history = (
                history
                if history is not None
                else (get_accelerator_view_from_cpu_tensor(self.host))
            )
            self.scales = get_accelerator_view_from_cpu_tensor(self.host_scales)
        self.codes = torch.empty((hot_tokens, 2, dim), dtype=torch.uint8, device=device)
        self.hot_scales = torch.empty(
            (hot_tokens, 2), dtype=torch.float32, device=device
        )
        self.tags = torch.full((self.sets, 4), -1, dtype=torch.int32, device=device)
        self.stamps = torch.zeros_like(self.tags)
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.locks = torch.zeros(self.sets, dtype=torch.int32, device=device)
        self.hands = torch.zeros_like(self.locks)
        self._stats = torch.zeros(
            (rows * triton.cdiv(width, 4), 3), dtype=torch.int64, device=device
        )
        # QSA layers execute serially on the model stream. Share staging across
        # owners while keeping each owner's hot pages and scales persistent.
        workspace_key = (device.index, rows, width, dim)
        if workspace_key not in _WORKSPACES:
            _WORKSPACES[workspace_key] = (
                torch.empty(
                    (rows, 2, self.padded, 1, dim), dtype=torch.float16, device=device
                ),
                torch.empty((rows, width), dtype=torch.int32, device=device),
                torch.arange(rows, dtype=torch.int32, device=device),
                torch.full((rows,), width - 1, dtype=torch.int64, device=device),
                torch.full((rows,), width, dtype=torch.int32, device=device),
            )
        self.staging, self.remapped, self.requests, self.positions, self.lengths = (
            _WORKSPACES[workspace_key]
        )
        self.table = self.requests.view(-1, 1)

    def write(self, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor):
        if key.shape[1:] != (1, self.dim) or value.shape != key.shape:
            raise ValueError("Host KV writer requires one local KV head")
        if slots.numel():
            _write[(slots.numel(),)](
                key,
                value,
                slots,
                self.history,
                self.scales,
                self.tags,
                self.epoch,
                key.stride(0),
                value.stride(0),
                self.blocks,
                self.page_size,
                self.dim,
                self.sets,
                num_warps=4,
            )

    def gather(self, indices, block_table, token_to_req, positions, lengths):
        rows = indices.shape[0]
        if indices.shape[1] != self.width or rows > self.rows:
            raise ValueError("Host KV selection exceeds fixed staging capacity")
        if block_table.shape[0] != lengths.numel():
            raise ValueError("Host KV request metadata disagrees")
        if rows:
            _protect[(rows, triton.cdiv(self.width, 256))](
                indices,
                block_table,
                token_to_req,
                positions,
                lengths,
                self.tags,
                self.stamps,
                self.epoch,
                self.width,
                block_table.shape[1],
                block_table.stride(0),
                indices.stride(0),
                block_table.shape[0],
                self.blocks,
                self.page_size,
                self.sets,
                num_warps=4,
            )
            _gather[(rows, triton.cdiv(self.width, 4))](
                self.history,
                self.scales,
                self.codes,
                self.hot_scales,
                self.tags,
                self.stamps,
                self.epoch,
                self.locks,
                self.hands,
                self._stats,
                indices,
                block_table,
                token_to_req,
                positions,
                lengths,
                self.staging,
                self.remapped,
                self.width,
                self.padded,
                block_table.shape[1],
                block_table.stride(0),
                indices.stride(0),
                block_table.shape[0],
                self.blocks,
                self.page_size,
                self.dim,
                self.sets,
                num_warps=4,
            )
        key, value = self.staging[:rows].unbind(1)
        return key, value, self.remapped[:rows]

    @property
    def stats(self):
        """Reduce diagnostic counters only when explicitly requested."""
        return self._stats.sum(0)
