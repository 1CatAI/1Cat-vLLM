# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use the shared QSA arithmetic directly on protected hot/host values."""

import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    _qsa_merge_splitk_kernel,
    _qsa_sparse_launch_profile,
    _qsa_sparse_paged_gqa_splitk_kernel,
    _use_sm70_qsa_two_warp_partial,
)
from vllm.triton_utils import triton


def host_qsa_attention(
    query,
    state,
    indices,
    table,
    requests,
    positions,
    lengths,
    out,
    gate=None,
    *,
    launch_rows=None,
):
    if query.dtype != torch.float16 or query.shape[2] != state.dim:
        raise ValueError("Direct host QSA requires FP16 queries matching D256")
    if out.shape != query.shape or indices.shape[0] != query.shape[0]:
        raise ValueError("Direct host QSA row counts disagree")
    from .device_kv_attention import device_history_attention

    if device_history_attention(
        query, state, indices, table, requests, positions, lengths, out, gate
    ):
        return out
    resolved = state.resolve(indices, table, requests, positions, lengths)
    group = query.shape[1]  # Host admission requires one TP-local KV head.
    block_m = triton.next_power_of_2(group)
    # Smaller staging batches retain the original logical batch's split-K
    # and warp choices, so the per-row accumulation order does not change.
    launch_rows = query.shape[0] if launch_rows is None else launch_rows
    block_n, target, warps = _qsa_sparse_launch_profile(launch_rows, block_m, True)
    if _use_sm70_qsa_two_warp_partial(launch_rows, group, state.dim):
        warps = 2
    tiles = triton.cdiv(indices.shape[1], block_n)
    splits = min(1 << (tiles.bit_length() - 1), target)
    if splits == 1:
        partial, lse = out, out
    else:
        partial = torch.empty(
            (splits, *query.shape), dtype=torch.float32, device=query.device
        )
        lse = torch.empty(
            (splits, *query.shape[:2]), dtype=torch.float32, device=query.device
        )
    gate = gate.view_as(query) if gate is not None else None
    _qsa_sparse_paged_gqa_splitk_kernel[(query.shape[0], 1, splits)](
        query,
        state.hot_values,
        state.staging,
        indices,
        table,
        requests,
        partial,
        lse,
        out,
        None,
        gate,
        query.stride(0),
        query.stride(1),
        0,
        0,
        0,
        0,
        0,
        0,
        indices.stride(0),
        table.stride(0),
        out.stride(0),
        out.stride(1),
        gate.stride(0) if gate is not None else 0,
        gate.stride(1) if gate is not None else 0,
        query.shape[0],
        state.blocks,
        table.shape[0],
        1.0,
        1.0,
        TOPK=indices.shape[1],
        PAGE_SIZE=state.page_size,
        PAGE_TABLE_WIDTH=table.shape[1],
        GROUP_SIZE=group,
        HEAD_DIM=state.dim,
        NUM_QUERY_HEADS=group,
        NUM_SPLITS=splits,
        NUM_TILES=tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        KV_E4M3=False,
        HOST_INDICES=resolved,
        HOST_VALID_COUNTS=state.lengths,
        HOST_CACHE=True,
        num_warps=warps,
        num_stages=2,
    )
    if splits > 1:
        _qsa_merge_splitk_kernel[(query.shape[0], group)](
            partial,
            lse,
            out,
            None,
            gate,
            out.stride(0),
            out.stride(1),
            gate.stride(0) if gate is not None else 0,
            gate.stride(1) if gate is not None else 0,
            query.shape[0],
            1.0,
            HEAD_DIM=state.dim,
            NUM_QUERY_HEADS=group,
            NUM_SPLITS=splits,
            BLOCK_SPLITS=triton.next_power_of_2(splits),
            KV_E4M3=False,
            num_warps=2,
            num_stages=1,
        )
    return out
