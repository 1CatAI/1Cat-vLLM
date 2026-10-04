# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 verifier preprocessing prototype; production dispatch is not connected."""

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
    X_ROW: tl.constexpr,
    W_ROW: tl.constexpr,
    W_COL: tl.constexpr,
    STATE_SEQ: tl.constexpr,
    STATE_DIM: tl.constexpr,
    STATE_TOKEN: tl.constexpr,
    CACHE_LINES: tl.constexpr,
    CHANNEL_TILE: tl.constexpr,
):
    feature = tl.program_id(1) * CHANNEL_TILE + tl.arange(0, CHANNEL_TILE)
    token = tl.arange(0, 8)
    # The old allocation zeroes padded output rows too. Keep that contract,
    # including invalid state slots, before the convolution's early returns.
    tl.store(
        Core + token[:, None] * 1536 + feature[None, :], 0, feature[None, :] < 1536
    )
    if tl.program_id(1) < 8:
        head = tl.arange(0, 16)
        mask = head < 12
        offset = tl.program_id(1) * 12 + head
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
        1,
        2560,
        8,
        10,
        CACHE_LINES,
        0,
        1,
        X_ROW,
        W_ROW,
        W_COL,
        STATE_SEQ,
        STATE_DIM,
        STATE_TOKEN,
        1,
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
        NP2_STATELEN=16,
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
    channel_tile=256,
    num_warps=4,
):
    """Exact-shape research entry; the caller proves the verifier layout."""
    assert qkv.shape == (8, 2560) and qkv.stride(1) == 1
    assert qkv.dtype == state.dtype == weight.dtype == torch.float16
    assert weight.shape == (2560, 4)
    assert core_out.shape == (8, 12, 128) and core_out.is_contiguous()
    assert state.shape[1] == 2560 and state.shape[2] >= 10
    assert a.shape == b.shape == (8, 12) and a.is_contiguous() and b.is_contiguous()
    assert state_indices.shape == accepted.shape == (1,) and cu.shape == (2,)
    g = torch.empty((1, 8, 12), dtype=torch.float32, device=qkv.device)
    beta = torch.empty_like(g)
    _conv_gate_zero_kernel[(1, triton.cdiv(2560, channel_tile))](
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
        qkv.stride(0),
        *weight.stride(),
        *state.stride(),
        state.shape[0],
        channel_tile,
        num_warps=num_warps,
    )
    return qkv, g, beta
