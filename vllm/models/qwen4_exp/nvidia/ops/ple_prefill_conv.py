# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dilated PLE prefill convolution without padded channels-first history."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def prefill_conv_reason(x, state, weight, requests, dilation):
    if not x.is_cuda or not current_platform.is_device_capability(70):
        return "requires_sm70"
    if x.dtype != torch.float16 or weight.dtype != x.dtype:
        return "requires_fp16_operands"
    if state.dtype not in (torch.float16, torch.float32):
        return "unsupported_state_dtype"
    if x.ndim != 2 or weight.ndim != 2 or state.ndim != 3:
        return "invalid_tensor_rank"
    if x.shape[0] < 512:
        return "query_rows_below_prefill_band"
    if not 1 <= requests <= 16 or not 1 <= dilation <= 8:
        return "unqualified_request_count_or_dilation"
    if not 2 <= weight.shape[1] <= 8:
        return "unqualified_kernel_width"
    if (
        weight.shape[0] != x.shape[1]
        or state.shape[1] != x.shape[1]
        or not state.shape[0]
        or state.shape[2] < (weight.shape[1] - 1) * dilation
        or not x.device == state.device == weight.device
    ):
        return "invalid_channel_state_or_device_layout"
    return None


@triton.jit
def _prefill_conv(
    X,
    Initial,
    Weight,
    Starts,
    States,
    HasInitial,
    Output,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    IS2: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    Rows: tl.constexpr,
    Hidden: tl.constexpr,
    Requests: tl.constexpr,
    Width: tl.constexpr,
    Dilation: tl.constexpr,
    Null: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    channels = tl.program_id(1) * BH + tl.arange(0, BH)
    request = tl.full((BT,), 0, tl.int32)
    for req in range(Requests):
        end = tl.load(Starts + req + 1)
        request += rows >= end
    safe_request = tl.minimum(request, Requests - 1)
    begin = tl.load(Starts + safe_request)
    position = rows - begin
    state = tl.load(States + safe_request)
    initial = tl.load(HasInitial + safe_request)
    valid = (rows < Rows) & (request < Requests) & (state != Null)
    state_len: tl.constexpr = (Width - 1) * Dilation
    acc = tl.full((BT, BH), 0, tl.float32)
    for tap in tl.static_range(Width):
        source = position - state_len + tap * Dilation
        current = tl.load(
            X
            + (begin + tl.maximum(source, 0))[:, None] * XS0
            + channels[None, :] * XS1,
            valid[:, None] & (source >= 0)[:, None] & (channels < Hidden)[None, :],
            other=0,
        ).to(tl.float32)
        previous = tl.load(
            Initial
            + safe_request[:, None] * IS0
            + channels[None, :] * IS1
            + tl.maximum(source + state_len, 0)[:, None] * IS2,
            valid[:, None]
            & initial[:, None]
            & (source < 0)[:, None]
            & (channels < Hidden)[None, :],
            other=0,
        ).to(tl.float32)
        coeff = tl.load(
            Weight + channels * WS0 + tap * WS1, channels < Hidden, other=0
        ).to(tl.float32)
        acc = tl.fma(
            tl.where(source[:, None] >= 0, current, previous), coeff[None, :], acc
        )
    # Retain the convolution's FP16 boundary before FP32 SiLU arithmetic.
    rounded = acc.to(tl.float16).to(tl.float32)
    activated = rounded / (1 + tl.extra.cuda.libdevice.exp(-rounded))
    tl.store(
        Output + rows[:, None] * Hidden + channels[None, :],
        tl.where(valid[:, None], activated, 0),
        (rows < Rows)[:, None] & (channels < Hidden)[None, :],
    )


@triton.jit
def _commit_prefill_state(
    X,
    Initial,
    Starts,
    States,
    HasInitial,
    State,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    IS2: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    Hidden: tl.constexpr,
    Length: tl.constexpr,
    Null: tl.constexpr,
    BH: tl.constexpr,
    BL: tl.constexpr,
):
    request = tl.program_id(0)
    channels = tl.program_id(1) * BH + tl.arange(0, BH)
    slots = tl.arange(0, BL)
    begin = tl.load(Starts + request)
    count = tl.load(Starts + request + 1) - begin
    state = tl.load(States + request)
    initial = tl.load(HasInitial + request)
    valid = (state != Null) & (count > 0)
    source = count - Length + slots
    current = tl.load(
        X + (begin + tl.maximum(source, 0))[None, :] * XS0 + channels[:, None] * XS1,
        valid
        & (channels < Hidden)[:, None]
        & (slots < Length)[None, :]
        & (source >= 0)[None, :],
        other=0,
    )
    previous = tl.load(
        Initial
        + request * IS0
        + channels[:, None] * IS1
        + tl.maximum(source + Length, 0)[None, :] * IS2,
        valid
        & initial
        & (channels < Hidden)[:, None]
        & (slots < Length)[None, :]
        & (source < 0)[None, :],
        other=0,
    )
    tl.store(
        State
        + tl.maximum(state, 0) * SS0
        + channels[:, None] * SS1
        + slots[None, :] * SS2,
        tl.where(source[None, :] >= 0, current, previous),
        valid & (channels < Hidden)[:, None] & (slots < Length)[None, :],
    )


def prefill_conv(x, state, weight, starts, states, has_initial, dilation):
    """Preserve null/empty rows and commit base history after reading its snapshot."""
    count = states.numel()
    length = (weight.shape[1] - 1) * dilation
    indices = states.to(device=state.device, dtype=torch.int64)
    indices = torch.where(indices == NULL_BLOCK_ID, 0, indices)
    initial = state.index_select(0, indices)[..., :length].to(x.dtype).contiguous()
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    _prefill_conv[(triton.cdiv(x.shape[0], 16), triton.cdiv(x.shape[1], 128))](
        x,
        initial,
        weight,
        starts,
        states,
        has_initial,
        out,
        *x.stride(),
        *initial.stride(),
        *weight.stride(),
        x.shape[0],
        x.shape[1],
        count,
        weight.shape[1],
        dilation,
        NULL_BLOCK_ID,
        16,
        128,
        num_warps=4,
    )
    _commit_prefill_state[(count, triton.cdiv(x.shape[1], 128))](
        x,
        initial,
        starts,
        states,
        has_initial,
        state,
        *x.stride(),
        *initial.stride(),
        *state.stride(),
        x.shape[1],
        length,
        NULL_BLOCK_ID,
        128,
        triton.next_power_of_2(length),
        num_warps=4,
    )
    return out
