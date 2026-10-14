# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dilated PLE prefill convolution without a padded channels-first history."""

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton


@triton.jit
def _dilated_prefill_conv(
    X,
    State,
    Weight,
    Starts,
    Slots,
    Initial,
    Out,
    D: tl.constexpr,
    X_STRIDE: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    DECODE_TOKENS: tl.constexpr,
    STATE_AVAILABLE: tl.constexpr,
    NULL_SLOT: tl.constexpr,
    K: tl.constexpr,
    DILATION: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    req = tl.program_id(2)
    first = tl.load(Starts + req) - DECODE_TOKENS
    length = tl.load(Starts + req + 1) - DECODE_TOKENS - first
    slot = tl.load(Slots + req).to(tl.int64)
    valid = slot != NULL_SLOT
    initial = tl.load(Initial + req).to(tl.int1) & valid
    times = tl.program_id(0) * BT + tl.arange(0, BT)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    active = (times[:, None] < length) & (cols[None, :] < D)
    acc = tl.full((BT, BD), 0.0, tl.float32)
    for tap in tl.static_range(K):
        pos = times - (K - 1 - tap) * DILATION
        value = tl.load(
            X + (first + pos[:, None]) * X_STRIDE + cols[None, :],
            active & (pos[:, None] >= 0) & valid,
            other=0.0,
        ).to(tl.float32)
        if STATE_AVAILABLE:
            old = tl.load(
                State
                + slot * S0
                + cols[None, :] * S1
                + ((K - 1) * DILATION + pos[:, None]) * S2,
                active & (pos[:, None] < 0) & initial,
                other=0.0,
            ).to(tl.float32)
            value += old
        weight = tl.load(Weight + cols * W0 + tap * W1, cols < D, 0).to(tl.float32)
        acc = tl.fma(value, weight[None, :], acc)
    tl.store(Out + (first + times[:, None]) * D + cols[None, :], acc, active)


@triton.jit
def _dilated_prefill_state(
    X,
    State,
    Starts,
    Slots,
    Initial,
    D: tl.constexpr,
    X_STRIDE: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    DECODE_TOKENS: tl.constexpr,
    NULL_SLOT: tl.constexpr,
    LENGTH: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
):
    req = tl.program_id(1)
    first = tl.load(Starts + req) - DECODE_TOKENS
    length = tl.load(Starts + req + 1) - DECODE_TOKENS - first
    slot = tl.load(Slots + req).to(tl.int64)
    valid = (slot != NULL_SLOT) & (length > 0)
    initial = tl.load(Initial + req).to(tl.int1)
    cols = tl.program_id(0) * BD + tl.arange(0, BD)
    offsets = tl.arange(0, BL)
    pos = length - LENGTH + offsets
    mask = (cols[:, None] < D) & (offsets[None, :] < LENGTH) & valid
    recent = tl.load(
        X + (first + pos[None, :]) * X_STRIDE + cols[:, None],
        mask & (pos[None, :] >= 0),
        other=0.0,
    )
    # One CTA owns all history positions for a channel. Read the old window
    # before its in-place update; other CTAs never write these channels.
    old = tl.load(
        State + slot * S0 + cols[:, None] * S1 + (LENGTH + pos[None, :]) * S2,
        mask & (pos[None, :] < 0) & initial,
        other=0.0,
    )
    next_state = tl.where(pos[None, :] >= 0, recent, old)
    tl.store(
        State + slot * S0 + cols[:, None] * S1 + offsets[None, :] * S2,
        next_state,
        mask,
    )


def dilated_prefill_short_conv(
    x,
    state,
    weight,
    starts,
    slots,
    initial,
    max_length,
    decode_tokens,
    dilation,
    null_slot,
    *,
    token_tile=8,
    channel_tile=128,
):
    """Compute request-local convolution, then commit each valid base state.

    Callers provide contiguous channels, unique non-null state slots, and the
    prefill suffix of query starts. Extra speculative state positions remain
    unchanged. This stage is screened independently before runtime admission.
    """
    rows, hidden = x.shape
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if rows == 0:
        return out
    kernel = weight.shape[1]
    common = dict(
        D=hidden,
        X_STRIDE=x.stride(0),
        S0=state.stride(0),
        S1=state.stride(1),
        S2=state.stride(2),
        DECODE_TOKENS=decode_tokens,
        NULL_SLOT=null_slot,
    )
    _dilated_prefill_conv[
        (
            triton.cdiv(max_length, token_tile),
            triton.cdiv(hidden, channel_tile),
            slots.numel(),
        )
    ](
        x,
        state,
        weight,
        starts,
        slots,
        initial,
        out,
        **common,
        W0=weight.stride(0),
        W1=weight.stride(1),
        STATE_AVAILABLE=state.shape[0] > 0,
        K=kernel,
        DILATION=dilation,
        BT=token_tile,
        BD=channel_tile,
        num_warps=4,
    )
    state_length = (kernel - 1) * dilation
    if state_length and state.shape[0]:
        _dilated_prefill_state[(triton.cdiv(hidden, 64), slots.numel())](
            x,
            state,
            starts,
            slots,
            initial,
            **common,
            LENGTH=state_length,
            BL=triton.next_power_of_2(state_length),
            BD=64,
            num_warps=4,
        )
    # Retain the native FP16 conv -> SiLU rounding boundary.
    return F.silu(out, inplace=True)
