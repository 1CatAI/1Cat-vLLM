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
    PageSlots,
    Hot,
    Epoch,
    Ks: tl.constexpr,
    Vs: tl.constexpr,
    Blocks: tl.constexpr,
    Page: tl.constexpr,
    Dim: tl.constexpr,
    Sets: tl.constexpr,
    FP8: tl.constexpr,
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
        if FP8:
            scales = tl.maximum(tl.div_rn(tl.max(tl.abs(values), 1), 448.0), 2.0**-126)
            codes = fp32_to_fp8_e4m3fn_bits(tl.div_rn(values, scales[:, None]))
            tl.store(Scales + slot * 2 + kv, scales)
        else:
            codes = values.to(tl.float16)
        block, token = slot // Page, slot % Page
        tl.store(
            History + ((block * 2 + kv[:, None]) * Page + token) * Dim + dims[None, :],
            codes,
        )
        # Update an existing hot copy from the encoded representation. Writes
        # and attention are stream ordered, including tentative overwrites.
        page = (slot // 4).to(tl.int32)
        resident = tl.load(PageSlots + page)
        tag = tl.load(Tags + tl.maximum(resident, 0), resident >= 0, other=-1)
        decoded = codes.to(tl.float16)
        if FP8:
            decoded = (fp8_e4m3fn_bits_to_fp32_bitcast(codes) * scales[:, None]).to(
                tl.float16
            )
        tl.store(
            Hot
            + ((tl.maximum(resident, 0) * 4 + slot % 4) * 2 + kv[:, None]) * Dim
            + dims[None, :],
            decoded,
            (resident >= 0) & (tag == page),
        )


@triton.jit
def _protect(
    Indices,
    Table,
    Requests,
    Positions,
    Lengths,
    Tags,
    PageSlots,
    Stamps,
    Epoch,
    VirtualPositions,
    VirtualLengths,
    Initial,
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
    logical = tl.load(Indices + row * IndexStride + columns, columns < Width, other=-1)
    req = tl.load(Requests + row)
    position = tl.load(Positions + row)
    length = tl.load(
        Lengths + tl.maximum(req, 0), (req >= 0) & (req < NumRequests), other=0
    )
    if tl.program_id(1) == 0:
        visible = tl.maximum(tl.minimum(position + 1, length), 0)
        selected_count = tl.minimum(visible // 4, Width // 4) * 4
        selected_count += tl.minimum(visible % 4, Width % 4)
        tl.store(VirtualPositions + row, selected_count - 1)
        tl.store(VirtualLengths + row, selected_count)
    valid = (columns < Width) & (logical >= 0) & (logical <= position)
    valid &= (logical < length) & (req >= 0) & (req < NumRequests)
    valid &= logical // Page < TableWidth
    blocks = tl.load(
        Table + tl.maximum(req, 0) * TableStride + tl.maximum(logical, 0) // Page,
        valid,
        other=-1,
    )
    valid &= (blocks >= 0) & (blocks < Blocks)
    if tl.sum(valid.to(tl.int32), 0) == 0:
        tl.store(Initial + row * Width + columns, -2, columns < Width)
        return
    tokens = tl.maximum(blocks, 0) * Page + tl.maximum(logical, 0) % Page
    pages = tokens // 4
    registered = tl.load(PageSlots + pages)
    slots = tl.maximum(registered, 0)
    tags = tl.load(Tags + slots, registered >= 0, other=-1)
    epoch = tl.load(Epoch)
    matched = (registered >= 0) & (tags == pages)
    tl.store(Stamps + slots, epoch, valid & matched)
    hot_token = slots * 4 + tokens % 4
    tl.store(
        Initial + row * Width + columns,
        tl.where(valid, tl.where(matched, hot_token, -1), -2),
        columns < Width,
    )


@triton.jit
def _gather(
    History,
    Scales,
    Hot,
    Tags,
    Stamps,
    Epoch,
    Hands,
    PageSlots,
    Stats,
    Indices,
    Table,
    Requests,
    Positions,
    Lengths,
    Out,
    Remapped,
    Initial,
    Resolved,
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
    FP8: tl.constexpr,
    STAGE: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    lanes = tl.arange(0, 16)
    dims = tl.arange(0, Dim)
    kv = tl.arange(0, 2)
    columns = tile * 16 + lanes
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
    tl.store(
        Remapped + row * Width + columns, tl.where(valid, columns, -1), columns < Width
    )
    if tl.sum(valid.to(tl.int32), 0) == 0:
        tl.store(Resolved + row * Width + columns, -2, columns < Width)
        return
    initial = tl.load(Initial + row * Width + columns, columns < Width, other=-2)
    initial_hit = valid & (initial >= 0)
    if tl.sum((initial_hit | ~valid).to(tl.int32), 0) == 16:
        # The preceding protection kernel pins these pages until attention
        # completes. Its stream dependency supplies visibility; no per-query
        # tag CAS, host decode or bucket lock is needed for an initial hit.
        tl.store(Resolved + row * Width + columns, initial, columns < Width)
        if STAGE == 1:
            values = tl.load(
                Hot
                + (tl.maximum(initial, 0)[:, None, None] * 2 + kv[None, :, None]) * Dim
                + dims[None, None, :],
                initial_hit[:, None, None],
                other=0,
            )
            tl.store(
                Out
                + ((row * 2 + kv[None, :, None]) * Padded + columns[:, None, None])
                * Dim
                + dims[None, None, :],
                values,
                columns[:, None, None] < Padded,
            )
        counter = (row * tl.cdiv(Width, 16) + tile) * 3
        tl.store(
            Stats + counter, tl.load(Stats + counter) + tl.sum(valid.to(tl.int64), 0)
        )
        return
    tokens = tl.maximum(blocks, 0) * Page + tl.maximum(logical, 0) % Page
    pages = tl.reshape(tokens // 4, (4, 4))
    page_lanes = tl.arange(0, 4)
    first = tl.sum(tl.where(page_lanes[None, :] == 0, pages, 0), 1)
    aligned = (
        tl.sum(
            tl.reshape(valid, (4, 4)).to(tl.int32)
            * (
                (pages == first[:, None])
                & (tl.reshape(tokens % 4, (4, 4)) == page_lanes[None, :])
            ),
            1,
        )
        == 4
    )
    registered = tl.load(PageSlots + first)
    slots = tl.maximum(registered, 0)
    tags = tl.load(Tags + slots, registered >= 0, other=-1)
    acquired = tl.atomic_cas(
        PageSlots + first,
        tl.where(registered >= 0, registered, -3),
        registered,
        sem="acquire",
    )
    hit = aligned & (registered >= 0) & (acquired == registered) & (tags == first)
    claim = (
        aligned
        & ~hit
        & (
            tl.atomic_cas(
                PageSlots + first,
                tl.where(aligned & ~hit, -1, -3),
                tl.full((4,), -2, tl.int32),
            )
            == -1
        )
    )
    reserved = tl.full((4,), False, tl.int1)
    epoch = tl.load(Epoch)
    probes = tl.arange(0, 16)
    # Bounded CLOCK probes: never wait for another CTA or evict a selected
    # page. Pending copies and capacity overflow use the exact host fallback.
    for _ in range(4):
        pending = claim & ~reserved
        if tl.sum(pending.to(tl.int32), 0) > 0:
            hand = tl.atomic_add(
                Hands + tl.zeros((4,), tl.int32), tl.where(pending, 16, 0)
            ).to(tl.uint32)
            candidates = (hand[:, None] + probes[None, :]) % (Sets * 4)
            stamps = tl.load(Stamps + candidates)
            chosen = tl.min(tl.where(stamps != epoch, probes[None, :], 16), 1)
            candidate = ((hand + tl.minimum(chosen, 15)) % (Sets * 4)).to(tl.int32)
            observed = tl.load(Stamps + candidate)
            attempt = pending & (chosen < 16) & (observed != epoch)
            won = attempt & (
                tl.atomic_cas(
                    Stamps + candidate,
                    tl.where(attempt, observed, -1),
                    tl.broadcast_to(epoch, (4,)),
                )
                == observed
            )
            slots = tl.where(won, candidate, slots)
            reserved |= won
    install = claim & reserved
    old_tag = tl.load(Tags + slots)
    tl.atomic_cas(
        PageSlots + tl.maximum(old_tag, 0),
        tl.where(install & (old_tag >= 0), slots, -3),
        tl.full((4,), -1, tl.int32),
    )
    hot_tokens = tl.reshape(slots[:, None] * 4 + page_lanes[None, :], (16,))
    token_hits = tl.reshape(tl.broadcast_to(hit[:, None], (4, 4)), (16,))
    cached = tl.load(
        Hot
        + (hot_tokens[:, None, None] * 2 + kv[None, :, None]) * Dim
        + dims[None, None, :],
        token_hits[:, None, None],
        other=0,
    )
    codes = tl.load(
        History
        + (
            (tl.maximum(blocks, 0).to(tl.int64)[:, None, None] * 2 + kv[None, :, None])
            * Page
            + (tl.maximum(logical, 0) % Page)[:, None, None]
        )
        * Dim
        + dims[None, None, :],
        (valid & ~token_hits)[:, None, None],
        other=0,
    )
    if FP8:
        scales = tl.load(
            Scales + tokens[:, None] * 2 + kv[None, :],
            (valid & ~token_hits)[:, None],
            other=0,
        )
        decoded = (fp8_e4m3fn_bits_to_fp32_bitcast(codes) * scales[:, :, None]).to(
            tl.float16
        )
    else:
        decoded = codes.to(tl.float16)
    values = tl.where(token_hits[:, None, None], cached, decoded)
    token_installs = tl.reshape(tl.broadcast_to(install[:, None], (4, 4)), (16,))
    tl.store(
        Hot
        + (hot_tokens[:, None, None] * 2 + kv[None, :, None]) * Dim
        + dims[None, None, :],
        values,
        token_installs[:, None, None],
    )
    tl.debug_barrier()
    tl.atomic_cas(Tags + slots, tl.where(install, old_tag, -2), first, sem="release")
    tl.atomic_cas(PageSlots + first, tl.where(install, -2, -3), slots, sem="release")
    tl.atomic_cas(
        PageSlots + first,
        tl.where(claim & ~install, -2, -3),
        tl.full((4,), -1, tl.int32),
    )
    tl.store(
        Resolved + row * Width + columns,
        tl.where(valid, tl.where(token_hits | token_installs, hot_tokens, -1), -2),
        columns < Width,
    )
    if STAGE > 0:
        store_values = columns < Padded
        if STAGE == 2:
            store_values &= valid & ~(token_hits | token_installs)
        tl.store(
            Out
            + ((row * 2 + kv[None, :, None]) * Padded + columns[:, None, None]) * Dim
            + dims[None, None, :],
            values,
            store_values[:, None, None],
        )
    counter = (row * tl.cdiv(Width, 16) + tile) * 3
    counters = tl.arange(0, 4)
    hits = tl.sum((valid & token_hits).to(tl.int64), 0)
    misses = tl.sum((valid & ~token_hits).to(tl.int64), 0)
    contended = tl.sum((aligned & ~hit & ~install).to(tl.int64), 0)
    delta = tl.where(counters == 0, hits, tl.where(counters == 1, misses, contended))
    old = tl.load(Stats + counter + counters, counters < 3, other=0)
    tl.store(Stats + counter + counters, old + delta, counters < 3)


class HostQSAKV:
    """Own stable host history, protected CLOCK slots and bounded miss staging."""

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
        dtype: torch.dtype = torch.uint8,
        device_reference: bool = False,
        direct_device: bool | None = None,
    ):
        if blocks <= 0 or page_size <= 0 or page_size % 4 or dim != 256:
            raise ValueError("Host QSA KV requires positive page4 geometry and D256")
        if hot_tokens <= 0 or hot_tokens % 16 or rows <= 0 or width <= 0:
            raise ValueError("Invalid hot cache or staging capacity")
        self.blocks, self.page_size, self.dim = blocks, page_size, dim
        self.rows, self.width = rows, width
        self.padded = triton.cdiv(width, 4) * 4
        self.sets = hot_tokens // 16
        dtype = history.dtype if history is not None else dtype
        if dtype not in (torch.uint8, torch.float16):
            raise ValueError("Host history requires E4M3 bytes or FP16 values")
        self.fp8 = dtype == torch.uint8
        self.device_reference = device_reference
        self.host = None
        if history is None:
            self.host = torch.zeros(
                (blocks, 2, page_size, 1, dim), dtype=dtype, pin_memory=True
            )
        elif history.shape != (blocks, 2, page_size, 1, dim) or (
            not history.is_contiguous()
        ):
            raise ValueError("Host history must use contiguous page-major E4M3 bytes")
        scale_shape = (blocks * page_size if self.fp8 else 1, 2)
        if device_reference:
            # Preserve the host allocator geometry and byte layout for the
            # placement experiment; only the writer/reader backing changes.
            self.history = torch.zeros(
                (blocks, 2, page_size, 1, dim), dtype=dtype, device=device
            )
            self.host_scales = torch.zeros(
                scale_shape, dtype=torch.float32, device=device
            )
            self.scales = self.host_scales
        else:
            self.host_scales = torch.zeros(
                scale_shape, dtype=torch.float32, pin_memory=True
            )
            with torch.accelerator.device_index(device.index):
                self.history = (
                    history
                    if history is not None
                    else get_accelerator_view_from_cpu_tensor(self.host)
                )
                self.scales = get_accelerator_view_from_cpu_tensor(self.host_scales)
        self.hot_values = torch.empty(
            (hot_tokens, 2, dim), dtype=torch.float16, device=device
        )
        self.tags = torch.full((self.sets, 4), -1, dtype=torch.int32, device=device)
        self.stamps = torch.zeros_like(self.tags)
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.hands = torch.zeros(1, dtype=torch.int32, device=device)
        self.page_slots = torch.full(
            (blocks * page_size // 4,), -1, dtype=torch.int32, device=device
        )
        self._stats = torch.zeros(
            (rows * triton.cdiv(width, 16), 3), dtype=torch.int64, device=device
        )
        # QSA layers execute serially on the model stream. Share staging across
        # owners while keeping each owner's hot pages and scales persistent.
        workspace_key = (device.index, rows, width, dim)
        if workspace_key not in _WORKSPACES:
            _WORKSPACES[workspace_key] = (
                torch.zeros(
                    (rows, 2, self.padded, 1, dim),
                    dtype=torch.float16,
                    device=device,
                ),
                torch.empty((rows, width), dtype=torch.int32, device=device),
                torch.arange(rows, dtype=torch.int32, device=device),
                torch.full((rows,), width - 1, dtype=torch.int64, device=device),
                torch.full((rows,), width, dtype=torch.int32, device=device),
                torch.empty((rows, width), dtype=torch.int32, device=device),
                torch.empty((rows, width), dtype=torch.int32, device=device),
            )
        (
            self.staging,
            self.remapped,
            self.requests,
            self.positions,
            self.lengths,
            self.initial,
            self.resolved,
        ) = _WORKSPACES[workspace_key]
        self.table = self.requests.view(-1, 1)

        from .device_kv_attention import initialize_device_history_attention

        self.device_history_workspace: tuple[torch.Tensor, torch.Tensor] | None = None
        self.device_history_reason: str | None = None
        initialize_device_history_attention(self, direct_device)

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
                self.page_slots,
                self.hot_values,
                self.epoch,
                key.stride(0),
                value.stride(0),
                self.blocks,
                self.page_size,
                self.dim,
                self.sets,
                self.fp8,
                num_warps=4,
            )

    def gather(self, indices, block_table, token_to_req, positions, lengths):
        self._resolve(indices, block_table, token_to_req, positions, lengths, 1)
        key, value = self.staging[: indices.shape[0]].unbind(1)
        return key, value, self.remapped[: indices.shape[0]]

    def resolve(self, indices, block_table, token_to_req, positions, lengths):
        self._resolve(indices, block_table, token_to_req, positions, lengths, 2)
        return self.resolved[: indices.shape[0]]

    def _resolve(self, indices, block_table, token_to_req, positions, lengths, stage):
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
                self.page_slots,
                self.stamps,
                self.epoch,
                self.positions,
                self.lengths,
                self.initial,
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
            _gather[(rows, triton.cdiv(self.width, 16))](
                self.history,
                self.scales,
                self.hot_values,
                self.tags,
                self.stamps,
                self.epoch,
                self.hands,
                self.page_slots,
                self._stats,
                indices,
                block_table,
                token_to_req,
                positions,
                lengths,
                self.staging,
                self.remapped,
                self.initial,
                self.resolved,
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
                self.fp8,
                stage,
                num_warps=4,
            )

    @property
    def stats(self):
        """Reduce diagnostic counters only when explicitly requested."""
        return self._stats.sum(0)
