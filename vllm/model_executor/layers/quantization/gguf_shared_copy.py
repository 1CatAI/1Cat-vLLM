# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw GGUF copies between a canonical shared pool and TP device carriers.

The coordinator validates allocation ranges and owns source read leases. Load
and save offsets are int64 byte offsets, never pointers from another process.
Each original down-projection block has exactly one writer; duplicated carrier
bytes on the other ranks are read-only copies. No format conversion occurs.
"""

import torch

from vllm.triton_utils import tl, triton

from .gguf_shared_layout import SharedExpertLayout


@triton.jit
def _swap_original(
    bank,
    host,
    slots,
    loads,
    saves,
    count,
    SOURCE_ROW_BYTES: tl.constexpr,
    SOURCE_ROW_START: tl.constexpr,
    FIRST_BYTE: tl.constexpr,
    DEVICE_ROW_BYTES: tl.constexpr,
    WRITE_BYTES: tl.constexpr,
    ROWS: tl.constexpr,
    PROJECTION_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    elements = ROWS * DEVICE_ROW_BYTES
    total = tl.load(count).to(tl.int64) * elements
    # A bounded persistent grid avoids dispatching a CTA for every byte tile
    # of every *possible* swap when the device-side plan is empty. The count
    # stays on device and may change between CUDA graph replays.
    for tile in range(tl.program_id(0), tl.cdiv(total, BLOCK), tl.num_programs(0)):
        linear = tl.cast(tile, tl.int64) * BLOCK + tl.arange(0, BLOCK)
        active = linear < total
        op, byte = linear // elements, linear % elements
        slot = tl.load(slots + op, mask=active, other=0).to(tl.int64)
        load = tl.load(loads + op, mask=active, other=-1).to(tl.int64)
        save = tl.load(saves + op, mask=active, other=-1).to(tl.int64)
        row, col = byte // DEVICE_ROW_BYTES, byte % DEVICE_ROW_BYTES
        device = bank + slot * elements + byte
        original = (
            (row + SOURCE_ROW_START) * SOURCE_ROW_BYTES
            + FIRST_BYTE
            + col
            + PROJECTION_OFFSET
        )
        # Save old bytes before overwriting the same portion of this slot.
        # Independent host destinations and exclusive whole-block writers
        # preserve the existing coordinator's non-aliasing copy contract.
        old = tl.load(device, mask=active & (save >= 0) & (col < WRITE_BYTES), other=0)
        tl.store(
            host + save + original,
            old,
            mask=active & (save >= 0) & (col < WRITE_BYTES),
        )
        new = tl.load(host + load + original, mask=active & (load >= 0), other=0)
        tl.store(device, new, mask=active & (load >= 0))


def swap_original_experts(
    bank: torch.Tensor,
    host: torch.Tensor,
    slots: torch.Tensor,
    loads: torch.Tensor,
    saves: torch.Tensor,
    count: torch.Tensor,
    layout: SharedExpertLayout,
    *,
    projection_offset: int = 0,
) -> None:
    """Execute an already validated plan; ``-1`` disables a load or save.

    ``bank`` is [GPU slots, local rows, carrier bytes] with an optional bank
    tail outside its view. ``host`` is a mapped CUDA alias of the shared byte
    arena. The caller publishes completion on this stream with a system-scope
    release only after this kernel. Returning from a launch is not completion.
    """
    tensors = (bank, host, slots, loads, saves, count)
    if projection_offset < 0:
        raise ValueError("Projection offset must be nonnegative")
    if not all(
        t.is_cuda and t.device == bank.device and t.is_contiguous() for t in tensors
    ):
        raise ValueError("Shared copies require contiguous tensors on one GPU")
    if (
        bank.dtype != torch.uint8
        or host.dtype != torch.uint8
        or slots.dtype != torch.int32
        or count.dtype != torch.int32
        or loads.dtype != torch.int64
        or saves.dtype != torch.int64
    ):
        raise ValueError("Invalid shared expert copy plan dtypes")
    if (
        bank.ndim != 3
        or bank.shape[1:] != (layout.device_rows, layout.device_row_bytes)
        or host.ndim != 1
        or count.numel() != 1
        or slots.ndim != 1
        or loads.shape != slots.shape
        or saves.shape != slots.shape
        or slots.numel() == 0
    ):
        raise ValueError("Invalid shared expert copy plan geometry")
    block = 512
    _swap_original[(256,)](
        bank,
        host,
        slots,
        loads,
        saves,
        count,
        SOURCE_ROW_BYTES=layout.source_row_bytes,
        SOURCE_ROW_START=layout.source_row_start,
        FIRST_BYTE=layout.first_block * layout.block_bytes,
        DEVICE_ROW_BYTES=layout.device_row_bytes,
        WRITE_BYTES=(layout.write_end_block - layout.first_block) * layout.block_bytes,
        ROWS=layout.device_rows,
        PROJECTION_OFFSET=projection_offset,
        BLOCK=block,
    )


@triton.jit
def _swap_shard(
    bank,
    host,
    slots,
    loads,
    saves,
    linear,
    active,
    SOURCE_ROW_BYTES: tl.constexpr,
    SOURCE_ROW_START: tl.constexpr,
    FIRST_BYTE: tl.constexpr,
    DEVICE_ROW_BYTES: tl.constexpr,
    WRITE_BYTES: tl.constexpr,
    ROWS: tl.constexpr,
    PROJECTION_OFFSET: tl.constexpr,
):
    elements = ROWS * DEVICE_ROW_BYTES
    op, byte = linear // elements, linear % elements
    slot = tl.load(slots + op, mask=active, other=0).to(tl.int64)
    load = tl.load(loads + op, mask=active, other=-1).to(tl.int64)
    save = tl.load(saves + op, mask=active, other=-1).to(tl.int64)
    row, col = byte // DEVICE_ROW_BYTES, byte % DEVICE_ROW_BYTES
    device = bank + slot * elements + byte
    original = (
        (row + SOURCE_ROW_START) * SOURCE_ROW_BYTES
        + FIRST_BYTE
        + col
        + PROJECTION_OFFSET
    )
    old = tl.load(device, mask=active & (save >= 0) & (col < WRITE_BYTES), other=0)
    tl.store(
        host + save + original, old, mask=active & (save >= 0) & (col < WRITE_BYTES)
    )
    new = tl.load(host + load + original, mask=active & (load >= 0), other=0)
    tl.store(device, new, mask=active & (load >= 0))


@triton.jit
def _swap_original3(
    b0,
    b1,
    b2,
    host,
    slots,
    loads,
    saves,
    count,
    S0: tl.constexpr,
    R0: tl.constexpr,
    F0: tl.constexpr,
    D0: tl.constexpr,
    W0: tl.constexpr,
    N0: tl.constexpr,
    P0: tl.constexpr,
    S1: tl.constexpr,
    R1: tl.constexpr,
    F1: tl.constexpr,
    D1: tl.constexpr,
    W1: tl.constexpr,
    N1: tl.constexpr,
    P1: tl.constexpr,
    S2: tl.constexpr,
    R2: tl.constexpr,
    F2: tl.constexpr,
    D2: tl.constexpr,
    W2: tl.constexpr,
    N2: tl.constexpr,
    P2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The three projections of a layer in one bounded persistent launch."""
    ops = tl.load(count).to(tl.int64)
    t0 = ops * (N0 * D0)
    t1 = ops * (N1 * D1)
    t2 = ops * (N2 * D2)
    tiles0 = tl.cdiv(t0, BLOCK)
    tiles1 = tl.cdiv(t1, BLOCK)
    tiles2 = tl.cdiv(t2, BLOCK)
    for tile in range(tl.program_id(0), tiles0 + tiles1 + tiles2, tl.num_programs(0)):
        if tile < tiles0:
            linear = tl.cast(tile, tl.int64) * BLOCK + tl.arange(0, BLOCK)
            _swap_shard(
                b0,
                host,
                slots,
                loads,
                saves,
                linear,
                linear < t0,
                S0,
                R0,
                F0,
                D0,
                W0,
                N0,
                P0,
            )
        elif tile < tiles0 + tiles1:
            linear = tl.cast(tile - tiles0, tl.int64) * BLOCK + tl.arange(0, BLOCK)
            _swap_shard(
                b1,
                host,
                slots,
                loads,
                saves,
                linear,
                linear < t1,
                S1,
                R1,
                F1,
                D1,
                W1,
                N1,
                P1,
            )
        else:
            linear = tl.cast(tile - tiles0 - tiles1, tl.int64) * BLOCK + tl.arange(
                0, BLOCK
            )
            _swap_shard(
                b2,
                host,
                slots,
                loads,
                saves,
                linear,
                linear < t2,
                S2,
                R2,
                F2,
                D2,
                W2,
                N2,
                P2,
            )


def swap_original_experts3(banks, host, slots, loads, saves, count, layouts, offsets):
    """Fused three-shard form of swap_original_experts (same copy contract)."""
    meta = {}
    for i, (layout, offset) in enumerate(zip(layouts, offsets)):
        meta.update(
            {
                f"S{i}": layout.source_row_bytes,
                f"R{i}": layout.source_row_start,
                f"F{i}": layout.first_block * layout.block_bytes,
                f"D{i}": layout.device_row_bytes,
                f"W{i}": (layout.write_end_block - layout.first_block)
                * layout.block_bytes,
                f"N{i}": layout.device_rows,
                f"P{i}": offset,
            }
        )
    _swap_original3[(256,)](
        banks[0],
        banks[1],
        banks[2],
        host,
        slots,
        loads,
        saves,
        count,
        BLOCK=512,
        **meta,
    )
