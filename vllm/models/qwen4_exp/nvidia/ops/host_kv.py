# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned, per-vector E4M3 QSA history with a bounded device hot-page cache.

The selector's positions and masks are preserved. Cache contention falls back
to authoritative host bytes, never spins or drops selected pages. All storage
and workspaces are allocated before graph capture.
"""

import numpy as np
import torch

from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32_bitcast,
    fp32_to_fp8_e4m3fn_bits,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

_WORKSPACES: dict[tuple, tuple[torch.Tensor, ...]] = {}
# Prefill history stages, shared by all QSA layers of a device.
_STAGES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


@triton.jit
def _stage_pages(
    src,
    dst,
    src_scales,
    dst_scales,
    blocks,
    WORDS: tl.constexpr,
    SCALES: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    """dst page p = src page blocks[p] (int64 words), plus its token scales.

    src/src_scales may be device-mapped pinned host memory: one launch reads
    every staged page over PCIe instead of one memcpy per page run.
    """
    page = tl.program_id(0)
    chunk = tl.program_id(1)
    block = tl.load(blocks + page).to(tl.int64)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < WORDS
    value = tl.load(src + block * WORDS + offs, mask=mask)
    tl.store(dst + page.to(tl.int64) * WORDS + offs, value, mask=mask)
    if SCALES > 0 and chunk == 0:
        s = tl.arange(0, SCALE_BLOCK)
        smask = s < SCALES
        scale = tl.load(src_scales + block * SCALES + s, mask=smask)
        tl.store(dst_scales + page.to(tl.int64) * SCALES + s, scale, mask=smask)


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
    HOT: tl.constexpr = True,
):
    # HOT=False: a plain history copy (a prefill stage buffer): no epoch,
    # no hot-cache update.
    row = tl.program_id(0)
    if HOT and row == 0:
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
        if HOT:
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
        is_speculative_draft: bool = False,
        resolve_rows: int = 32,
    ):
        if blocks <= 0 or page_size <= 0 or page_size % 4 or dim != 256:
            raise ValueError("Host QSA KV requires positive page4 geometry and D256")
        if hot_tokens <= 0 or hot_tokens % 16 or rows <= 0 or width <= 0:
            raise ValueError("Invalid hot cache or staging capacity")
        self.blocks, self.page_size, self.dim = blocks, page_size, dim
        self.rows, self.width = rows, width
        # Rows one hot-cache resolve may cover. Staging (``rows``) bounds only
        # the Triton miss path; direct readers resolve larger prefill slices.
        self.resolve_rows = max(rows, resolve_rows)
        self.padded = triton.cdiv(width, 4) * 4
        self.sets = hot_tokens // 16
        dtype = history.dtype if history is not None else dtype
        if dtype not in (torch.uint8, torch.float16):
            raise ValueError("Host history requires E4M3 bytes or FP16 values")
        self.fp8 = dtype == torch.uint8
        self.device_reference = device_reference
        self.is_speculative_draft = is_speculative_draft
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
            (self.resolve_rows * triton.cdiv(width, 16), 3),
            dtype=torch.int64,
            device=device,
        )
        # QSA layers execute serially on the model stream. Share staging across
        # owners while keeping each owner's hot pages and scales persistent.
        workspace_key = (device.index, rows, self.resolve_rows, width, dim)
        if workspace_key not in _WORKSPACES:
            meta = self.resolve_rows
            _WORKSPACES[workspace_key] = (
                torch.zeros(
                    (rows, 2, self.padded, 1, dim),
                    dtype=torch.float16,
                    device=device,
                ),
                torch.empty((meta, width), dtype=torch.int32, device=device),
                torch.arange(meta, dtype=torch.int32, device=device),
                torch.full((meta,), width - 1, dtype=torch.int64, device=device),
                torch.full((meta,), width, dtype=torch.int32, device=device),
                torch.empty((meta, width), dtype=torch.int32, device=device),
                torch.empty((meta, width), dtype=torch.int32, device=device),
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
        self.device_history_hot = False
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

    def write_staged(self, key, value, slots, plan, history, scales):
        """Write this step's K/V straight into a prefill stage buffer
        (`history`/`scales` from stage_history): the same encoding as
        `write`, at each token's staged page, instead of copying the pages
        back from the host."""
        _, k, _, pages, _ = plan
        if not slots.numel() or not k:
            return
        page = self.page_size
        staged_of = torch.full(
            (self.blocks,), -1, dtype=torch.int64, device=slots.device
        )
        staged_of[pages.long()] = torch.arange(k, device=slots.device)
        source = slots.long()
        target = staged_of[(source // page).clamp(0, self.blocks - 1)]
        target = torch.where(
            (source >= 0) & (target >= 0), target * page + source % page, -1
        )
        _write[(slots.numel(),)](
            key,
            value,
            target,
            history,
            scales,
            self.tags,
            self.page_slots,
            self.hot_values,
            self.epoch,
            key.stride(0),
            value.stride(0),
            # Blocks is constexpr: the fixed pool size (a loose bound; targets
            # are already < k pages), not k, which changes every step and
            # would recompile the kernel each time.
            self.blocks,
            page,
            self.dim,
            self.sets,
            self.fp8,
            HOT=False,
            num_warps=4,
        )

    def stage_plan(self, block_table, lengths, capacity_tokens):
        """Host plan for stage_history: (runs, pages, device table) or None.

        Depends only on the step's block table and lengths, so one plan
        (one host sync) serves every layer of a step with this geometry.
        """
        page = self.page_size
        capacity = self._stage_capacity(capacity_tokens)
        table = block_table.to("cpu", non_blocking=False).numpy()
        lengths_host = [int(n) for n in lengths.tolist()]
        sizes = [(n + page - 1) // page for n in lengths_host]
        if sum(sizes) > capacity or not sum(sizes):
            return None
        # Vectorized per request: valid pages keep their order and get
        # consecutive staged indices.
        staged = np.full(table.shape, -1, dtype=np.int32)
        sources = []
        k = 0
        for request, count in enumerate(sizes):
            row = table[request, :count]
            valid = np.flatnonzero((row >= 0) & (row < self.blocks))
            staged[request, valid] = np.arange(k, k + valid.size, dtype=np.int32)
            sources.append(row[valid])
            k += valid.size
        sources = np.concatenate(sources).astype(np.int32)
        # Runs of consecutive source blocks (first block, first staged, count)
        # for the host-tensor copy path.
        breaks = np.flatnonzero(np.diff(sources) != 1) + 1
        starts = np.concatenate(([0], breaks))
        counts = np.diff(np.concatenate((starts, [sources.size])))
        runs = [[int(sources[a]), int(a), int(n)] for a, n in zip(starts, counts)]
        device = self.history.device
        staged = torch.from_numpy(staged).pin_memory()
        pages = torch.from_numpy(sources).pin_memory()
        return (
            runs,
            k,
            staged.to(device, non_blocking=True),
            pages.to(device, non_blocking=True),
            lengths_host,
        )

    def _stage_capacity(self, capacity_tokens):
        # One byte budget (capacity_tokens E4M3 K/V tokens) shared by every
        # layer and dtype: FP16 draft history stages half as many tokens.
        page_bytes = 2 * self.page_size * self.dim * self.history.element_size()
        return capacity_tokens * 2 * self.dim // page_bytes

    def _stage_views(self, capacity_tokens, slot):
        """(history, scales) views of stage buffer `slot` (0/1)."""
        page = self.page_size
        capacity = self._stage_capacity(capacity_tokens)
        budget = capacity_tokens * 2 * self.dim
        device = self.history.device
        key = (device.index, budget, slot)
        if key not in _STAGES:
            _STAGES[key] = (
                torch.empty(budget, dtype=torch.uint8, device=device),
                torch.empty((capacity_tokens, 2), dtype=torch.float32, device=device),
            )
        raw, raw_scales = _STAGES[key]
        source = self.host if self.host is not None else self.history
        history = raw[: capacity * 2 * page * self.dim * source.element_size()]
        history = history.view(source.dtype).view(capacity, 2, page, 1, self.dim)
        scales = raw_scales[: capacity * page] if self.fp8 else raw_scales[:1]
        return history, scales

    def _stage_range(self, plan, history, scales, first, last):
        """Copy staged pages [first, last) of `plan` into the stage views."""
        runs, _, _, pages, _ = plan
        if last <= first:
            return
        page = self.page_size
        source = self.host if self.host is not None else self.history
        if source.is_cuda:
            # Device-addressable history (the mapped serving pool): one
            # gather launch for the pages and their scales.
            words = 2 * page * self.dim * source.element_size() // 8
            block = min(4096, triton.next_power_of_2(words))
            scale_count = 2 * page if self.fp8 else 0
            _stage_pages[(last - first, triton.cdiv(words, block))](
                source.view(-1).view(torch.int64),
                history[first:].reshape(-1).view(torch.int64),
                self.scales,
                scales[first * page :] if self.fp8 else scales,
                pages[first:last],
                WORDS=words,
                SCALES=scale_count,
                BLOCK=block,
                SCALE_BLOCK=triton.next_power_of_2(max(scale_count, 1)),
            )
            return
        for source_block, target, count in runs:
            lo, hi = max(target, first), min(target + count, last)
            if lo >= hi:
                continue
            src = source_block + lo - target
            history[lo:hi].copy_(source[src : src + hi - lo], non_blocking=True)
            if self.fp8:
                scales[lo * page : hi * page].copy_(
                    self.host_scales[src * page : (src + hi - lo) * page],
                    non_blocking=True,
                )

    def prefix_pages(self, plan, new_tokens):
        """Staged pages holding only tokens written before this step (one
        request), or 0 when the plan covers several requests."""
        lengths = plan[4]
        if len(lengths) != 1:
            return 0
        return max(0, lengths[0] - new_tokens) // self.page_size

    def prefetch_prefix(
        self, plan, new_tokens, capacity_tokens, slot, stream, buffer_free
    ):
        """Stage this owner's prefix pages into buffer `slot` on `stream`
        (after `buffer_free`); returns the completion event."""
        prefix = self.prefix_pages(plan, new_tokens)
        stream.wait_stream(torch.cuda.current_stream())
        if buffer_free is not None:
            stream.wait_event(buffer_free)
        with torch.cuda.stream(stream):
            history, scales = self._stage_views(capacity_tokens, slot)
            self._stage_range(plan, history, scales, 0, prefix)
            if not self.fp8:
                scales.copy_(self.host_scales, non_blocking=True)
            done = torch.cuda.Event()
            done.record(stream)
        return done, prefix

    def stage_history(
        self,
        block_table,
        lengths,
        capacity_tokens,
        plan=None,
        slot=0,
        staged_pages=0,
        upto=None,
    ):
        """Copy every request's history pages to a shared device buffer.

        Returns (history, scales, table) for the device-history reader, the
        table remapped to the staged pages, or None when the history does not
        fit (or is not on the host). Copies run on the current stream after
        this step's writes, so the current chunk's own tokens are included.
        The first `staged_pages` pages are already in buffer `slot`; with
        `upto`, pages from `upto` on are left for write_staged.
        """
        if self.device_reference or capacity_tokens <= 0:
            return None
        if plan is None:
            plan = self.stage_plan(block_table, lengths, capacity_tokens)
        if plan is None:
            return None
        _, k, staged_table, _, _ = plan
        page = self.page_size
        history, scales = self._stage_views(capacity_tokens, slot)
        self._stage_range(
            plan, history, scales, staged_pages, k if upto is None else min(upto, k)
        )
        if not self.fp8 and not staged_pages:
            scales.copy_(self.host_scales, non_blocking=True)
        return history[:k], scales[: k * page] if self.fp8 else scales, staged_table

    def gather(self, indices, block_table, token_to_req, positions, lengths):
        self._resolve(indices, block_table, token_to_req, positions, lengths, 1)
        key, value = self.staging[: indices.shape[0]].unbind(1)
        return key, value, self.remapped[: indices.shape[0]]

    def resolve(self, indices, block_table, token_to_req, positions, lengths):
        self._resolve(indices, block_table, token_to_req, positions, lengths, 2)
        return self.resolved[: indices.shape[0]]

    def resolve_hot(self, indices, block_table, token_to_req, positions, lengths):
        """Install selected pages in the hot cache without staging values.

        Returns hot token ids (-1: read the authoritative history, -2: not
        selected) for readers that also address the history directly.
        """
        self._resolve(indices, block_table, token_to_req, positions, lengths, 0)
        return self.resolved[: indices.shape[0]]

    def _resolve(self, indices, block_table, token_to_req, positions, lengths, stage):
        rows = indices.shape[0]
        capacity = self.rows if stage else self.resolve_rows
        if indices.shape[1] != self.width or rows > capacity:
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
