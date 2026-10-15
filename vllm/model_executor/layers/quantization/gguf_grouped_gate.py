# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped original integer experts, with live sparsity admission on device."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _sort_routes(
    Ids,
    Sorted,
    Tokens,
    Order,
    Mode,
    Routes: tl.constexpr,
    TopK: tl.constexpr,
    Block: tl.constexpr,
):
    i = tl.arange(0, Block)
    ids = tl.load(Ids + i, i < Routes, other=2147483647)
    packed = tl.where(i < Routes, ids * Block + i, 2147483647)
    ranked = tl.sort(packed, descending=False)
    experts, original = ranked // Block, ranked % Block
    previous = tl.gather(experts, tl.maximum(i - 1, 0), 0)
    distinct = tl.sum(
        ((i < Routes) & ((i == 0) | (experts != previous))).to(tl.int32), 0
    )
    # At least one third of routes must reuse an expert before grouping pays.
    grouped = distinct * 3 <= Routes * 2
    tl.store(Mode, grouped.to(tl.int32))
    tl.store(Sorted + i, experts, i < Routes)
    tl.store(Tokens + i, original // TopK, i < Routes)
    tl.store(Order + i, original, i < Routes)


def grouped_gate_up(
    out, activation, ids, gate, up, source_type, bank_aware, max_rows=20
):
    """Return False before launch when the native grouping contract is absent."""
    if (
        max_rows not in (20, 40)
        or not 9 <= ids.shape[0] <= max_rows
        or ids.dtype not in (torch.int32, torch.int64)
        or source_type not in (18, 21, 22)
        or out.dtype != torch.uint8
        or not hasattr(torch.ops._C, "gguf_dp4a_gate_up_grouped_sm70_out")
        or (
            max_rows == 40
            and not hasattr(torch.ops._C, "gguf_dp4a_gate_up_grouped_batch_sm70_out")
        )
    ):
        return False
    # Router IDs are bounded by the expert bank. Keep the native route mapping
    # int32 without reading device sparsity or IDs on the host.
    ids = ids.to(dtype=torch.int32).contiguous()
    sorted_ids = torch.empty_like(ids)
    tokens = torch.empty_like(ids)
    order = torch.empty_like(ids)
    mode = torch.empty(1, dtype=torch.int32, device=ids.device)
    _sort_routes[(1,)](
        ids,
        sorted_ids,
        tokens,
        order,
        mode,
        ids.numel(),
        ids.shape[1],
        triton.next_power_of_2(ids.numel()),
        num_warps=4,
    )
    operation = (
        torch.ops._C.gguf_dp4a_gate_up_grouped_batch_sm70_out
        if max_rows == 40
        else torch.ops._C.gguf_dp4a_gate_up_grouped_sm70_out
    )
    operation(
        out,
        activation,
        sorted_ids,
        gate,
        up,
        source_type,
        bank_aware,
        tokens,
        order,
        ids,
        mode,
    )
    return True


def grouped_down(out, activation, ids, probabilities, weight, source_type, left):
    """Retain the twenty-row down path outside the admitted grouped range."""
    if (
        not 25 <= ids.shape[0] <= 40
        or ids.dtype != torch.int32
        or source_type not in (20, 42)
        or activation.dtype != torch.uint8
        or not hasattr(torch.ops._C, "gguf_dp4a_raw_down_grouped_sm70_out")
    ):
        return False
    sorted_ids = torch.empty_like(ids)
    tokens = torch.empty_like(ids)
    order = torch.empty_like(ids)
    mode = torch.empty(1, dtype=torch.int32, device=ids.device)
    _sort_routes[(1,)](
        ids,
        sorted_ids,
        tokens,
        order,
        mode,
        ids.numel(),
        ids.shape[1],
        triton.next_power_of_2(ids.numel()),
        num_warps=4,
    )
    projected = out.new_empty((*ids.shape, out.shape[1]))
    torch.ops._C.gguf_dp4a_raw_down_grouped_sm70_out(
        out,
        projected,
        activation,
        sorted_ids,
        order,
        probabilities,
        weight,
        source_type,
        left,
    )
    return True
