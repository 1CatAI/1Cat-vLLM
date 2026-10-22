# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional SM70 single-request GDN verification provider."""

from functools import partial

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fla.ops.gdn_stages import GdnHeadContract

logger = init_logger(__name__)


def bind_native_verifier(contract, *, enabled):
    if not enabled or not hasattr(torch.ops._C, "sm70_gdn_verify_out"):
        return None
    # Resolve tensor operands at the layer call, so reload/replacement never
    # leaves the provider pointing at an old parameter or AOT address.
    return partial(verify_bound, contract)


def verify_bound(
    contract: GdnHeadContract,
    A_log,
    dt_bias,
    requests,
    mixed_qkv,
    a,
    b,
    state,
    out,
    num_actual_tokens,
    direct_out,
    query_start,
    state_indices,
    slot_selectors,
    replay=None,
):
    """Native verification over ``requests`` sequences of at most
    ``state_indices.shape[1]`` tokens each; ``replay`` commits by replay."""
    tokens = mixed_qkv.shape[0]
    if replay is not None:
        # Replay reads only the committed column; the index width does not
        # bound the round length.
        *replay, per_request = replay
    else:
        per_request = state_indices.shape[1] if state_indices.dim() == 2 else 1
    if not (
        requests >= 1
        and per_request <= 8
        and tokens <= requests * per_request
        and mixed_qkv.dtype == torch.float16
        and mixed_qkv.stride(1) == 1
        and state.dtype == torch.float32
        and contract.head_k_dim == 128
        and contract.head_v_dim == 128
    ):
        return None
    if replay is None:
        logger.info_once("SM70 CUDA GDN verification route hit.")
    else:
        logger.info_once("SM70 CUDA GDN verification with deferred state commit.")
    heads = contract.local_v_heads
    verify_out = (
        out[:num_actual_tokens].unsqueeze(0)
        if direct_out
        else mixed_qkv.new_empty((1, tokens, heads, 128))
    )
    rbuf = rn = rmap = ckt = cks = None
    if replay is not None:
        rbuf, rn, rmap, ckt, cks = replay
    torch.ops._C.sm70_gdn_verify_out(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        verify_out.view(-1, heads, 128),
        query_start[: requests + 1],
        state_indices,
        None if replay is not None else slot_selectors,
        contract.local_k_heads,
        heads,
        contract.head_k_dim**-0.5,
        1,
        rbuf,
        rn,
        None,
        per_request if requests > 1 or replay is not None else tokens,
        rmap,
        ckt,
        cks,
    )
    return verify_out, state


def commit_replayed(
    contract: GdnHeadContract,
    A_log,
    dt_bias,
    state,
    starts,
    state_indices,
    replay_buffer,
    replay_counts,
    replay_map,
    stride,
):
    """Commit the previous verification of rows that verify no drafts this
    step: replay their accepted tokens into the committed state (T = 0)."""
    k_heads, v_heads = contract.local_k_heads, contract.local_v_heads
    width = 2 * k_heads * contract.head_k_dim + v_heads * contract.head_v_dim
    empty = state.new_empty((0, width), dtype=torch.float16)
    gates = state.new_empty((0, v_heads), dtype=torch.float16)
    out = state.new_empty((0, v_heads, contract.head_v_dim), dtype=torch.float16)
    torch.ops._C.sm70_gdn_verify_out(
        empty,
        gates,
        gates,
        A_log,
        dt_bias,
        state,
        out,
        starts,
        state_indices,
        None,
        k_heads,
        v_heads,
        contract.head_k_dim**-0.5,
        1,
        replay_buffer,
        replay_counts,
        None,
        stride,
        replay_map,
        None,
        None,
    )
