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

# Slots one program sorts (one warp, a register/shuffle bitonic sort): one decode
# token has 8, a speculative verify of 16 tokens 128.
_MAX_KERNEL_SLOTS = 128
# A single warp keeps the sort and the barrier below inside shuffles; more warps
# only add shared-memory round trips for at most 128 keys.
_NUM_WARPS = 1


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
    LOG2_BLOCK: tl.constexpr,
):
    slot = tl.arange(0, BLOCK)
    valid = slot < num_slots
    expert = tl.load(ids_ptr + slot, mask=valid, other=0).to(tl.int32)
    # One sort key per slot, expert-major and slot-minor: the keys are distinct,
    # so sorting them is the stable sort by expert and the slot index rides in
    # the low bits (needs |expert| < 2**(31 - LOG2_BLOCK), far above any expert
    # count). Padding lanes take the largest key and end up behind every slot.
    key = tl.where(valid, expert * BLOCK + slot, 0x7FFFFFFF)
    ordered = tl.sort(key)
    perm = ordered & (BLOCK - 1)
    sorted_expert = ordered >> LOG2_BLOCK
    previous_expert = tl.gather(sorted_expert, tl.maximum(slot - 1, 0), axis=0)
    starts = valid & ((slot == 0) | (sorted_expert != previous_expert))
    group = tl.cumsum(starts.to(tl.int32), axis=0) - 1
    tl.store(perm_ptr + slot, perm, mask=valid)
    # Unused group rows first, then each group start scatters into its row; the
    # barrier orders the two writes to the same address.
    tl.store(goff_ptr + slot, num_slots, mask=valid)
    tl.store(gids_ptr + slot, 0, mask=valid)
    tl.debug_barrier()
    tl.store(goff_ptr + group, slot, mask=starts)
    tl.store(gids_ptr + group, sorted_expert, mask=starts)
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
    # tl.sort needs at least two lanes.
    block = max(2, triton.next_power_of_2(num_slots))
    perm = torch.empty(num_slots, dtype=torch.int32, device=device)
    gids = torch.empty(num_slots, dtype=torch.int32, device=device)
    goff = torch.empty(num_slots + 1, dtype=torch.int32, device=device)
    _group_routing_kernel[(1,)](
        topk_ids.reshape(-1),
        perm,
        gids,
        goff,
        num_slots,
        BLOCK=block,
        LOG2_BLOCK=block.bit_length() - 1,
        num_warps=_NUM_WARPS,
    )
    return perm, gids, goff
