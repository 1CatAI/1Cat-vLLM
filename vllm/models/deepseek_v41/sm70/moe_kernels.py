# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small Triton kernels of the DeepSeek-V4.1 SM70 MoE (lane L-MOE, PORT_DESIGN §3.4, §4.1).

* ``route_prep``: slot routing for the grouped skinny MXFP4 kernel. Slots (token-major, ``slot = t * top_k + j``)
  are sorted by physical expert id; consecutive equal experts form a group (``gids[g]`` = expert,
  ``goff[g]..goff[g+1]`` = slot range, unused groups empty at the end). With an expert spill the resident
  experts (physical ids ``< n_resident``) and the spilled ones get separate group tables over one shared
  permutation, so each partition runs its own launch on its own weight stack.
* ``swiglu_fp32``: the reference SwiGLU with the training clamps (up to ``[-limit, limit]``, gate to
  ``<= limit``), computed in FP32 and rounded to FP16 once (ref:m.py:840-848; §4.1 "clamp + SiLU in FP32").
* ``combine_fp32``: ``out = sum_j w[t, j] * y[t * top_k + j] + shared[t]`` in FP32, slot order, separate
  multiply and add (no FMA), so it reproduces :func:`combine_reference` bit for bit.

Every kernel has a torch twin (``*_reference``) that defines its exact semantics; tests compare them.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

# Above this many slots the routing tables are built with torch sorts (eager prefill); at or below it one
# Triton program does it with fixed shapes (CUDA-graph decode, T <= 8 at top-6 = 48 slots).
ROUTE_PREP_MAX_SMALL_SLOTS = 64
_BIG_KEY = tl.constexpr(1 << 30)


@triton.jit
def _route_prep_small_kernel(
    ids_ptr,
    phys_ptr,
    perm_ptr,
    gids0_ptr,
    goff0_ptr,
    gids1_ptr,
    goff1_ptr,
    S,
    n_resident,
    HAS_PHYS: tl.constexpr,
    SPILL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    valid = i < S
    expert = tl.load(ids_ptr + i, mask=valid, other=0)
    if HAS_PHYS:
        key = tl.load(phys_ptr + expert, mask=valid, other=0)
    else:
        key = expert
    key = tl.where(valid, key, _BIG_KEY)
    ki = key[:, None]
    kj = key[None, :]
    ii = i[:, None]
    jj = i[None, :]
    same_before = (kj == ki) & (jj < ii)
    rank = tl.sum(((kj < ki) | same_before).to(tl.int32), axis=1)
    first = valid & (tl.sum(same_before.to(tl.int32), axis=1) == 0)
    group = tl.sum(((kj < ki) & first[None, :]).to(tl.int32), axis=1)
    tl.store(perm_ptr + rank, i, mask=valid)
    resident = key < n_resident
    if SPILL:
        n_res_slots = tl.sum((valid & resident).to(tl.int32), axis=0)
        n_res_groups = tl.sum((first & resident).to(tl.int32), axis=0)
        tl.store(goff0_ptr + i, n_res_slots + 0 * i, mask=i <= S)
        tl.store(goff1_ptr + i, S + 0 * i, mask=i <= S)
        tl.store(gids0_ptr + i, 0 * i, mask=valid)
        tl.store(gids1_ptr + i, 0 * i, mask=valid)
        tl.debug_barrier()
        tl.store(goff0_ptr + group, rank, mask=first & resident)
        tl.store(gids0_ptr + group, key, mask=first & resident)
        spilled = first & (key >= n_resident)
        tl.store(goff1_ptr + group - n_res_groups, rank, mask=spilled)
        tl.store(gids1_ptr + group - n_res_groups, key - n_resident, mask=spilled)
    else:
        tl.store(goff0_ptr + i, S + 0 * i, mask=i <= S)
        tl.store(gids0_ptr + i, 0 * i, mask=valid)
        tl.debug_barrier()
        tl.store(goff0_ptr + group, rank, mask=first)
        tl.store(gids0_ptr + group, key, mask=first)


RouteTables = tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]


def route_prep_reference(
    topk_ids: torch.Tensor, phys_map: torch.Tensor | None, n_resident: int, spill: bool
) -> RouteTables:
    """Torch twin of the routing tables: ``(perm, [(gids0, goff0)] + ([(gids1, goff1)] if spill))``.

    ``perm`` lists slot indices sorted by (physical expert, slot); group ``g`` of a partition covers
    ``perm[goff[g]:goff[g+1]]`` and names its partition-local expert ``gids[g]``. Fixed shapes and no host
    synchronisation, so it also serves the large-slot path inside piecewise CUDA graphs.
    """
    flat = topk_ids.reshape(-1).to(torch.int64)
    key = phys_map.to(torch.int64)[flat] if phys_map is not None else flat
    num_slots = key.numel()
    device = key.device
    perm = torch.sort(key, stable=True)[1]
    skey = key[perm]
    first = torch.ones(num_slots, dtype=torch.bool, device=device)
    first[1:] = skey[1:] != skey[:-1]
    group = torch.cumsum(first.to(torch.int64), 0) - 1
    slots = torch.arange(num_slots, dtype=torch.int64, device=device)
    goff = torch.full((num_slots + 1,), num_slots, dtype=torch.int64, device=device)
    goff.scatter_reduce_(0, group, slots, reduce="amin")
    gids = torch.zeros(num_slots, dtype=torch.int64, device=device)
    gids.scatter_(0, group, skey)
    if not spill:
        return perm.to(torch.int32), [(gids.to(torch.int32), goff.to(torch.int32))]
    resident = skey < n_resident
    n_res_slots = resident.sum()
    n_res_groups = (first & resident).sum()
    idx1 = torch.arange(num_slots + 1, dtype=torch.int64, device=device)
    goff0 = torch.where(idx1 < n_res_groups, goff, n_res_slots)
    gids0 = torch.where(idx1[:-1] < n_res_groups, gids, 0)
    shifted = torch.clamp(idx1 + n_res_groups, max=num_slots)
    goff1 = goff.gather(0, shifted)
    n_groups = first.sum()
    gids1 = torch.where(idx1[:-1] + n_res_groups < n_groups,
                        gids.gather(0, torch.clamp(idx1[:-1] + n_res_groups, max=num_slots - 1)) - n_resident, 0)
    return perm.to(torch.int32), [(gids0.to(torch.int32), goff0.to(torch.int32)),
                                  (gids1.to(torch.int32), goff1.to(torch.int32))]


def route_prep(
    topk_ids: torch.Tensor, phys_map: torch.Tensor | None, n_resident: int, spill: bool
) -> RouteTables:
    """Routing tables for the grouped skinny kernel (see module docstring). ``topk_ids`` [T, k] int32."""
    if topk_ids.dtype != torch.int32 or topk_ids.ndim != 2 or not topk_ids.is_contiguous():
        raise TypeError(f"route_prep needs contiguous int32 [T, k] ids, got {topk_ids.dtype} {tuple(topk_ids.shape)}")
    num_slots = topk_ids.numel()
    if num_slots > ROUTE_PREP_MAX_SMALL_SLOTS:
        return route_prep_reference(topk_ids, phys_map, n_resident, spill)
    device = topk_ids.device
    perm = torch.empty(num_slots, dtype=torch.int32, device=device)
    gids0 = torch.empty(num_slots, dtype=torch.int32, device=device)
    goff0 = torch.empty(num_slots + 1, dtype=torch.int32, device=device)
    gids1 = torch.empty(num_slots if spill else 1, dtype=torch.int32, device=device)
    goff1 = torch.empty(num_slots + 1 if spill else 1, dtype=torch.int32, device=device)
    block = max(16, triton.next_power_of_2(num_slots + 1))
    _route_prep_small_kernel[(1,)](
        topk_ids,
        phys_map if phys_map is not None else topk_ids,
        perm,
        gids0,
        goff0,
        gids1,
        goff1,
        num_slots,
        n_resident,
        HAS_PHYS=phys_map is not None,
        SPILL=spill,
        BLOCK=block,
        num_warps=4,
    )
    tables = [(gids0, goff0)] + ([(gids1, goff1)] if spill else [])
    return perm, tables


@triton.jit
def _swiglu_fp32_kernel(in_ptr, out_ptr, I, limit, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < I
    gate = tl.load(in_ptr + row * 2 * I + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(in_ptr + row * 2 * I + I + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.minimum(gate, limit)
    up = tl.minimum(tl.maximum(up, -limit), limit)
    silu = tl.div_rn(gate, 1.0 + tl.extra.cuda.libdevice.exp(-gate))
    tl.store(out_ptr + row * I + cols, (silu * up).to(tl.float16), mask=mask)


def swiglu_fp32_reference(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    inter = gate_up.shape[-1] // 2
    gate = gate_up[..., :inter].float().clamp(max=limit)
    up = gate_up[..., inter:].float().clamp(min=-limit, max=limit)
    return (gate / (1.0 + torch.exp(-gate)) * up).to(torch.float16)


def swiglu_fp32(gate_up: torch.Tensor, limit: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """[M, 2I] fp16 (gate | up) -> [M, I] fp16, FP32 math, one rounding."""
    if gate_up.dtype != torch.float16 or gate_up.ndim != 2 or not gate_up.is_contiguous():
        raise TypeError("swiglu_fp32 needs a contiguous fp16 [M, 2I] input")
    rows, two_i = gate_up.shape
    inter = two_i // 2
    if out is None:
        out = torch.empty((rows, inter), dtype=torch.float16, device=gate_up.device)
    if rows == 0:
        return out
    block = 1024
    _swiglu_fp32_kernel[(rows, triton.cdiv(inter, block))](gate_up, out, inter, float(limit), BLOCK=block, num_warps=4)
    return out


@triton.jit
def _combine_fp32_kernel(
    y_ptr,
    w_ptr,
    shared_ptr,
    out_ptr,
    H,
    TOP_K: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(TOP_K):
        w = tl.load(w_ptr + t * TOP_K + j)
        y = tl.load(y_ptr + (t * TOP_K + j) * H + cols, mask=mask, other=0.0).to(tl.float32)
        if j == 0:
            acc = w * y
        else:
            acc = acc + w * y
    if HAS_SHARED:
        acc = acc + tl.load(shared_ptr + t * H + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + t * H + cols, acc, mask=mask)


def combine_reference(y_slots: torch.Tensor, topk_weights: torch.Tensor, shared: torch.Tensor | None) -> torch.Tensor:
    """FP32 combine in slot order: ((w0*y0 + w1*y1) + ...) + shared."""
    num_tokens, top_k = topk_weights.shape
    y = y_slots.view(num_tokens, top_k, -1).float()
    w = topk_weights.float()
    acc = w[:, 0:1] * y[:, 0]
    for j in range(1, top_k):
        acc = acc + w[:, j : j + 1] * y[:, j]
    if shared is not None:
        acc = acc + shared.float()
    return acc


def combine_fp32(
    y_slots: torch.Tensor,
    topk_weights: torch.Tensor,
    shared: torch.Tensor | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """y_slots [T*k, H] fp16, topk_weights [T, k] f32, shared [T, H] (f32/fp16) or None -> [T, H] f32."""
    num_tokens, top_k = topk_weights.shape
    hidden = y_slots.shape[1]
    if y_slots.dtype != torch.float16 or y_slots.shape[0] != num_tokens * top_k or not y_slots.is_contiguous():
        raise TypeError(f"combine_fp32: y_slots must be contiguous fp16 [{num_tokens * top_k}, H]")
    if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
        raise TypeError("combine_fp32: topk_weights must be contiguous float32")
    if shared is not None and (shared.shape != (num_tokens, hidden) or not shared.is_contiguous()):
        raise TypeError(f"combine_fp32: shared must be contiguous [{num_tokens}, {hidden}]")
    if out is None:
        out = torch.empty((num_tokens, hidden), dtype=torch.float32, device=y_slots.device)
    if num_tokens == 0:
        return out
    block = 1024
    _combine_fp32_kernel[(num_tokens, triton.cdiv(hidden, block))](
        y_slots,
        topk_weights,
        shared if shared is not None else y_slots,
        out,
        hidden,
        TOP_K=top_k,
        HAS_SHARED=shared is not None,
        BLOCK=block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
