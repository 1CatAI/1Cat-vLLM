# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""System-atomic sampled-key publication on mapped, registered host storage."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _publish_keys(
    history,
    mapping,
    seq_lens,
    sampled,
    counts,
    payload,
    flag,
    epoch,
    ACTIVE: tl.constexpr,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    HISTORY_ROWS: tl.constexpr,
    HISTORY_LENGTH: tl.constexpr,
    HISTORY_STRIDE: tl.constexpr,
    SAMPLE_STRIDE: tl.constexpr,
    EOS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    version = tl.load(epoch).to(tl.uint32) + 2
    # System-scope atomic stores, not host-memory read/modify/write atomics.
    # Every lane may store the same scalar version; payload stores are disjoint.
    tl.inline_asm_elementwise(
        "st.release.sys.global.u32 [$1], $2; mov.u32 $0, $2;",
        constraints="=r,l,r,~{memory}",
        args=[flag, version - 1],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    tl.inline_asm_elementwise(
        "membar.sys; mov.u32 $0, 0;",
        constraints="=r,~{memory}",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()
    word = tl.arange(0, BLOCK)
    row, column = word // COLUMNS, word % COLUMNS
    index = tl.load(mapping + row, row < ACTIVE, other=-1)
    length = tl.load(seq_lens + row, row < ACTIVE, other=-1)
    count = tl.load(counts + row, row < ACTIVE, other=0)
    valid = (
        (row < ACTIVE)
        & (index >= 0)
        & (index < HISTORY_ROWS)
        & (count == 1)
        & (length >= 0)
        & (length <= HISTORY_LENGTH)
    )
    position = length - (COLUMNS - 1) + column
    old_token = tl.load(
        history + index * HISTORY_STRIDE + position,
        valid
        & (column > 0)
        & (column < COLUMNS - 1)
        & (position >= 0)
        & (position < HISTORY_LENGTH),
        other=EOS,
    )
    next_token = tl.load(sampled + row * SAMPLE_STRIDE, valid, other=0).to(tl.int32)
    token = tl.where(column == COLUMNS - 1, next_token, old_token)
    value = tl.where(column == 0, valid.to(tl.int32), tl.where(valid, token, 0))
    tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.u32 p, $3, 0; "
        "@p st.relaxed.sys.global.u32 [$1], $2; mov.u32 $0, $2; }",
        constraints="=r,l,r,r,~{memory}",
        args=[payload + word, value, (word < ROWS * COLUMNS).to(tl.uint32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.inline_asm_elementwise(
        "membar.sys; mov.u32 $0, 0;",
        constraints="=r,~{memory}",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()
    tl.inline_asm_elementwise(
        "st.release.sys.global.u32 [$1], $2; mov.u32 $0, $2;",
        constraints="=r,l,r,~{memory}",
        args=[flag, version],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    tl.store(epoch, version)


def publish_sampled_keys(
    history, mapping, seq_lens, sampled, counts, payload, flag, epoch, eos
):
    tensors = (history, mapping, seq_lens, sampled, counts, payload, flag, epoch)
    if any(
        not t.is_cuda or t.device != history.device or not t.is_contiguous()
        for t in tensors
    ):
        raise ValueError("Sampled-key publication requires same-device CUDA tensors")
    if (
        history.ndim != 2
        or payload.ndim != 2
        or payload.shape[1] < 3
        or mapping.ndim != 1
        or len(mapping) > len(payload)
        or any(
            t.dtype != torch.int32
            for t in (history, mapping, seq_lens, counts, payload, flag, epoch)
        )
        or sampled.dtype not in (torch.int32, torch.int64)
        or sampled.ndim != 2
        or sampled.shape[1] != 1
        or min(seq_lens.numel(), counts.numel(), len(sampled)) < len(mapping)
        or flag.numel() < 1
        or epoch.numel() != 1
    ):
        raise ValueError("Invalid sampled-key publication dtype or dimensions")
    _publish_keys[(1,)](
        history,
        mapping,
        seq_lens,
        sampled,
        counts,
        payload,
        flag,
        epoch,
        ACTIVE=len(mapping),
        ROWS=len(payload),
        COLUMNS=payload.shape[1],
        HISTORY_ROWS=len(history),
        HISTORY_LENGTH=history.shape[1],
        HISTORY_STRIDE=history.stride(0),
        SAMPLE_STRIDE=sampled.stride(0),
        EOS=eos,
        BLOCK=triton.next_power_of_2(payload.numel()),
        num_warps=1,
    )
