# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared GDN compute stages with explicit weights, shapes and state views."""

from dataclasses import dataclass

from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update_mixed_qkv,
    fused_sigmoid_gating_delta_rule_update_mixed_qkv_out,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update


@dataclass(frozen=True)
class GdnHeadContract:
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    tp_size: int

    @property
    def local_k_heads(self) -> int:
        return self.num_k_heads // self.tp_size

    @property
    def local_v_heads(self) -> int:
        return self.num_v_heads // self.tp_size


def convolve_decode(
    mixed_qkv,
    state,
    weight,
    bias,
    activation,
    *,
    state_indices,
    validate_data,
    num_accepted_tokens=None,
    query_start_loc=None,
    max_query_len=None,
):
    """Keep each caller's validation and speculative slot semantics explicit."""
    if weight.ndim != 2:
        weight = weight.view(weight.size(0), weight.size(-1))
    if max_query_len is None:
        return causal_conv1d_update(
            mixed_qkv,
            state,
            weight,
            bias,
            activation,
            conv_state_indices=state_indices,
            validate_data=validate_data,
        )
    return causal_conv1d_update(
        mixed_qkv,
        state,
        weight,
        bias,
        activation,
        conv_state_indices=state_indices,
        validate_data=validate_data,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        max_query_len=max_query_len,
    )


def mixed_qkv_recurrence(
    contract: GdnHeadContract,
    *,
    A_log,
    dt_bias,
    a,
    b,
    mixed_qkv,
    initial_state,
    cu_seqlens,
    state_indices,
    out=None,
    inplace_final_state=True,
    schedule=None,
):
    """One recurrence dispatch; output allocation and state writes stay explicit.

    The out variant keeps its original scale argument and [T,1,H,V] layout.
    The allocating variant keeps [1,T,H,V] and optional comparison state.
    """
    # Fixed arguments avoid constructing and unpacking a dispatch dictionary
    # on each eager decode step. The two operators own different output layouts.
    if out is not None:
        fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
            A_log,
            a,
            b,
            dt_bias,
            mixed_qkv,
            contract.local_k_heads,
            contract.local_v_heads,
            contract.head_k_dim,
            contract.head_v_dim,
            out,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
            scale=contract.head_k_dim**-0.5,
            schedule=schedule,
        )
        return out, initial_state
    return fused_sigmoid_gating_delta_rule_update_mixed_qkv(
        A_log,
        a,
        b,
        dt_bias,
        mixed_qkv,
        num_q_heads=contract.local_k_heads,
        num_v_heads=contract.local_v_heads,
        head_k_dim=contract.head_k_dim,
        head_v_dim=contract.head_v_dim,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
        inplace_final_state=inplace_final_state,
        schedule=schedule,
    )


def normalize_qk(q, k):
    """Shared external normalization; callers preserve gate-conversion order."""
    return l2norm_fwd(q), l2norm_fwd(k)
