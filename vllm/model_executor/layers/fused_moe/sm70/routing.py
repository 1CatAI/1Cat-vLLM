# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing groups of the grouped skinny MoE calls.

The slots of a routing (token x top-k) are sorted by expert, and consecutive
equal experts form a group: ``gids[g]`` is the group's expert, ``goff[g]`` its
first slot in the sorted order. Unused group slots stay empty (``goff == S``,
``gids == 0``). All shapes are fixed, so the launch is CUDA-graph safe.
"""

import torch

from vllm.triton_utils import tl, triton

# Slots one program sorts by pairwise ranking (an S x S comparison in registers):
# one decode token has 8, a speculative verify of 16 tokens 128.
_MAX_KERNEL_SLOTS = 128


def group_routing_reference(
    topk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The routing groups as torch ops; defines the result for any slot count."""
    device = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    num_slots = flat.numel()
    perm = torch.argsort(flat, stable=True).to(torch.int32)
    sorted_e = flat[perm.long()]
    new_group = torch.ones(num_slots, dtype=torch.bool, device=device)
    new_group[1:] = sorted_e[1:] != sorted_e[:-1]
    gidx = torch.cumsum(new_group, 0) - 1
    goff = torch.full((num_slots + 1,), num_slots, dtype=torch.int64, device=device)
    goff.scatter_reduce_(
        0,
        gidx,
        torch.arange(num_slots, dtype=torch.int64, device=device),
        reduce="amin",
    )
    gids = torch.zeros(num_slots, dtype=torch.int64, device=device)
    gids.scatter_(0, gidx, sorted_e)
    return perm, gids.to(torch.int32), goff.to(torch.int32)


@triton.jit
def _group_routing_kernel(
    ids_ptr,
    perm_ptr,
    gids_ptr,
    goff_ptr,
    num_slots,
    BLOCK: tl.constexpr,
):
    slot = tl.arange(0, BLOCK)
    valid = slot < num_slots
    # Padding slots carry the largest key and rank behind every real slot.
    expert = tl.load(ids_ptr + slot, mask=valid, other=0x7FFFFFFF).to(tl.int32)
    # Stable rank: slot b sorts before slot a if its expert is smaller, or equal
    # and it comes first.
    before = (expert[None, :] < expert[:, None]) | (
        (expert[None, :] == expert[:, None]) & (slot[None, :] < slot[:, None])
    )
    rank = tl.sum(before.to(tl.int32), axis=1)
    # Row r of `at_rank` selects the slot of rank r.
    at_rank = rank[None, :] == slot[:, None]
    sorted_expert = tl.sum(tl.where(at_rank, expert[None, :], 0), axis=1)
    perm = tl.sum(tl.where(at_rank, slot[None, :], 0), axis=1)
    previous_expert = tl.sum(
        tl.where(rank[None, :] == slot[:, None] - 1, expert[None, :], 0), axis=1
    )
    starts = valid & ((slot == 0) | (sorted_expert != previous_expert))
    group = tl.cumsum(starts.to(tl.int32), axis=0) - 1
    # Row g of `is_start` selects the sorted position that opens group g.
    is_start = starts[None, :] & (group[None, :] == slot[:, None])
    opens = tl.sum(is_start.to(tl.int32), axis=1) > 0
    first = tl.sum(tl.where(is_start, slot[None, :], 0), axis=1)
    group_expert = tl.sum(tl.where(is_start, sorted_expert[None, :], 0), axis=1)
    tl.store(perm_ptr + slot, perm, mask=valid)
    tl.store(goff_ptr + slot, tl.where(opens, first, num_slots), mask=valid)
    tl.store(gids_ptr + slot, tl.where(opens, group_expert, 0), mask=valid)
    tl.store(goff_ptr + num_slots, num_slots)


def group_routing(
    topk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort order, group experts and group offsets of a routing, int32 tensors
    of ``S``, ``S`` and ``S + 1`` entries (``S`` = token x top-k slots)."""
    num_slots = topk_ids.numel()
    if num_slots > _MAX_KERNEL_SLOTS:
        return group_routing_reference(topk_ids)
    device = topk_ids.device
    perm = torch.empty(num_slots, dtype=torch.int32, device=device)
    gids = torch.empty(num_slots, dtype=torch.int32, device=device)
    goff = torch.empty(num_slots + 1, dtype=torch.int32, device=device)
    _group_routing_kernel[(1,)](
        topk_ids.reshape(-1),
        perm,
        gids,
        goff,
        num_slots,
        BLOCK=triton.next_power_of_2(num_slots),
        num_warps=4,
    )
    return perm, gids, goff
