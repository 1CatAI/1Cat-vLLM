# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research Q/K normalization once in convolution, without cross-CTA polling.

Convolution owns complete K128 heads. Its original FP16 result is normalized
in FP32 and cached for the unchanged BV2 delta recurrence. This may change
the reduction layout, so whole-model numerical admission remains required.
"""

import torch

from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update_kernel,
)
from vllm.model_executor.layers.mamba.gdn.sm70_preprocess import (
    _conv_gate_zero_kernel,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _prepare(
    X,
    Prepared,
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
):
    _conv_gate_zero_kernel(
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
        X_ROW,
        W_ROW,
        W_COL,
        STATE_SEQ,
        STATE_DIM,
        STATE_TOKEN,
        CACHE_LINES,
        8,
        10,
        8,
        16,
        128,
    )
    # No other CTA writes this complete K128 head. Preserve the FP16
    # convolution result, then communicate FP32 normalized Q/K to delta.
    tl.debug_barrier()
    feature = tl.program_id(1) * 128 + tl.arange(0, 128)
    token = tl.arange(0, 8)
    value = tl.load(X + token[:, None] * X_ROW + feature[None, :]).to(tl.float32)
    if tl.program_id(1) < 8:
        value = value / tl.sqrt(tl.sum(value * value, axis=1)[:, None] + 1e-6)
    tl.store(Prepared + token[:, None] * 2560 + feature[None, :], value)


def conv_prepare(
    qkv, state, weight, state_indices, accepted, cu, a_log, a, b, bias, core_out
):
    assert qkv.shape == (8, 2560) and qkv.stride(1) == 1
    assert qkv.dtype == state.dtype == weight.dtype == torch.float16
    g = torch.empty((1, 8, 12), device=qkv.device, dtype=torch.float32)
    beta = torch.empty_like(g)
    prepared = torch.empty_like(qkv, dtype=torch.float32)
    _prepare[(1, 20)](
        qkv,
        prepared,
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
        num_warps=1,
    )
    return prepared, g, beta


def prepared_delta(
    a_log,
    a,
    b,
    bias,
    mixed_qkv,
    h,
    hv,
    k,
    v,
    out,
    *,
    initial_state,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    precomputed_g,
    precomputed_beta,
    **ignored,
):
    assert (h, hv, k, v) == (4, 12, 128, 128)
    assert mixed_qkv.shape == (8, 2560) and mixed_qkv.dtype == torch.float32
    assert initial_state.dtype == torch.float32
    assert ssm_state_indices.shape == (1, 8)
    fused_sigmoid_gating_delta_rule_update_kernel[(1, 64, 12)](
        A_log=a_log,
        a=precomputed_g,
        b=precomputed_beta,
        dt_bias=bias,
        beta=1.0,
        threshold=20.0,
        q=mixed_qkv,
        k=mixed_qkv,
        v=mixed_qkv,
        mixed_qkv=mixed_qkv,
        o=out,
        h0=initial_state,
        ht=initial_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        ddtree_parent_ids=None,
        scale=128**-0.5,
        N=1,
        T=8,
        B=1,
        H=4,
        HV=12,
        K=128,
        V=128,
        BK=128,
        BV=2,
        Q_OFFSET=0,
        K_OFFSET=512,
        V_OFFSET=1024,
        QKV_STRIDE=mixed_qkv.stride(0),
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        stride_indices_tok=ssm_state_indices.stride(1),
        stride_parent_ids_seq=1,
        stride_parent_ids_tok=1,
        INPLACE_FINAL_STATE=True,
        USE_QK_L2NORM_IN_KERNEL=False,
        IS_KDA=False,
        MIXED_QKV=True,
        PRECOMPUTED_GATING=True,
        QUANTIZE_BETA=False,
        QUANTIZE_STATE_EACH_STEP=False,
        MATCH_RECURRENT_NUMERICS=True,
        num_warps=1,
        num_stages=3,
    )
    return out, initial_state
