# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 verifier convolution, gating and output initialization across requests."""

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    _causal_conv1d_update_kernel,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _conv_gate_zero_kernel(
    X,
    W,
    State,
    StateIndices,
    Accepted,
    Cu,
    G,
    Beta,
    ALog,
    A,
    B,
    Bias,
    Core,
    BATCH,
    INDEX_STRIDE: tl.constexpr,
    X_ROW: tl.constexpr,
    W_ROW: tl.constexpr,
    W_COL: tl.constexpr,
    STATE_SEQ: tl.constexpr,
    STATE_DIM: tl.constexpr,
    STATE_TOKEN: tl.constexpr,
    CACHE_LINES: tl.constexpr,
    QKV_WIDTH: tl.constexpr,
    VALUE_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
    STATE_LENGTH: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    NP2_STATE: tl.constexpr,
    CHANNEL_TILE: tl.constexpr,
    COMPUTE_GATING: tl.constexpr = True,
):
    request = tl.program_id(0)
    feature = tl.program_id(1) * CHANNEL_TILE + tl.arange(0, CHANNEL_TILE)
    token = tl.arange(0, BLOCK_TOKENS)
    row = request * TOKENS + token
    # The old allocation zeroes padded output rows too. Keep that contract,
    # including invalid state slots, before the convolution's early returns.
    tl.store(
        Core + row[:, None] * VALUE_WIDTH + feature[None, :],
        0,
        (token[:, None] < TOKENS) & (feature[None, :] < VALUE_WIDTH),
    )
    if COMPUTE_GATING:  # noqa: SIM102
        if tl.program_id(1) < TOKENS:
            head = tl.arange(0, BLOCK_HEADS)
            mask = head < HEADS
            offset = (request * TOKENS + tl.program_id(1)) * HEADS + head
            a_log = tl.load(ALog + head, mask=mask, other=0)
            a = tl.load(A + offset, mask=mask, other=0)
            b = tl.load(B + offset, mask=mask, other=0)
            bias = tl.load(Bias + head, mask=mask, other=0)
            x = a.to(tl.float32) + bias.to(tl.float32)
            softplus = tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x)
            g = -tl.exp(a_log.to(tl.float32)) * softplus
            tl.store(G + offset, g, mask=mask)
            tl.store(Beta + offset, tl.sigmoid(b.to(tl.float32)), mask=mask)
    # Inline the existing convolution body, including accepted-history offset,
    # in-place QKV writes, and state-layout strides. Delta is a separate launch.
    _causal_conv1d_update_kernel(
        X,
        W,
        None,
        State,
        StateIndices,
        Accepted,
        Cu,
        None,
        None,
        X,
        BATCH,
        QKV_WIDTH,
        TOKENS,
        STATE_LENGTH,
        CACHE_LINES,
        0,
        1,
        X_ROW,
        W_ROW,
        W_COL,
        STATE_SEQ,
        STATE_DIM,
        STATE_TOKEN,
        INDEX_STRIDE,
        0,
        1,
        X_ROW,
        -1,
        HAS_BIAS=False,
        KERNEL_WIDTH=4,
        SILU_ACTIVATION=True,
        IS_VARLEN=True,
        IS_APC_ENABLED=False,
        IS_SPEC_DECODING=True,
        NP2_STATELEN=NP2_STATE,
        USE_PAD_SLOT=True,
        BLOCK_N=CHANNEL_TILE,
    )


def conv_gate_zero(
    qkv,
    state,
    weight,
    state_indices,
    accepted,
    cu,
    a_log,
    a,
    b,
    bias,
    core_out,
    channel_tile=64,
    num_warps=2,
    *,
    compute_gating: bool = True,
):
    """Dispatch proves a complete verifier batch and its state layouts."""
    rows = qkv.shape[0]
    channels = qkv.shape[1]
    heads = core_out.shape[1]
    value_width = heads * core_out.shape[2]
    batch = state_indices.numel()
    assert batch > 0 and rows % batch == 0
    tokens = rows // batch
    assert 1 <= tokens <= 8 and channels > 0 and qkv.stride(1) == 1
    assert qkv.dtype == state.dtype == weight.dtype == torch.float16
    assert weight.shape == (channels, 4)
    assert core_out.shape == (rows, heads, 128) and core_out.is_contiguous()
    assert 0 < value_width <= channels
    assert state.shape[1] == channels and state.shape[2] >= tokens + 2
    assert (
        a.shape == b.shape == (rows, heads) and a.is_contiguous() and b.is_contiguous()
    )
    assert a_log.shape == bias.shape == (heads,)
    assert state_indices.shape == accepted.shape == (batch,)
    assert cu.shape == (batch + 1,) and accepted.is_contiguous() and cu.is_contiguous()
    # Match causal_conv1d_update: speculative history uses width-1 plus
    # tokens-1 entries even when the physical cache has a longer stride.
    if compute_gating:
        g = torch.empty((1, rows, heads), dtype=torch.float32, device=qkv.device)
        beta = torch.empty_like(g)
        grid_y = max(tokens, triton.cdiv(channels, channel_tile))
    else:
        g = None
        beta = None
        grid_y = triton.cdiv(channels, channel_tile)
    _conv_gate_zero_kernel[(batch, grid_y)](
        qkv,
        weight,
        state,
        state_indices,
        accepted,
        cu,
        g,
        beta,
        a_log,
        a,
        b,
        bias,
        core_out,
        batch,
        state_indices.stride(0),
        qkv.stride(0),
        *weight.stride(),
        *state.stride(),
        state.shape[0],
        channels,
        value_width,
        heads,
        triton.next_power_of_2(heads),
        tokens,
        tokens + 2,
        triton.next_power_of_2(tokens),
        triton.next_power_of_2(tokens + 2),
        channel_tile,
        COMPUTE_GATING=compute_gating,
        num_warps=num_warps,
    )
    return qkv, g, beta
