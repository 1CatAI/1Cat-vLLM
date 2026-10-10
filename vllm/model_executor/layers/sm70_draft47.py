# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ONECAT_DRAFT47: SM70 MTP drafter shortcuts that leave every real row unchanged.

Units (a comma list, or ``all``; unset or ``0`` keeps every path unchanged):

* ``d2a``: a drafter decode CUDA graph captured for ``R`` requests also serves
  fewer requests. After routing, the padded rows get expert id -1, so
  ``fused_moe_kernel`` writes zeros for their blocks without reading expert
  weights. The real rows keep the same kernels and the same M. The padded rows'
  MoE output becomes 0 instead of a value computed from stale inputs; only
  those rows read it.
* ``d2b``: the drafter decode graph manager also captures FULL graphs for 6, 7
  and 8 requests. Without them, these batches run the decode steps eagerly with
  the same M.
* ``d1a``: the drafter's vocab-parallel argmax
  (``LogitsProcessor.get_top_tokens``) in two kernels. Each rank writes its
  (max, index) pair with ``torch.max``'s tie and NaN rules. The pairs are
  all-gathered along dim 0, and one kernel picks the winner with
  ``torch.argmax``'s rules.

The engine resolves the legacy switch into KernelConfig.sm70_draft once.
Active drafter policy participates in the engine's graph cache fingerprint.
"""

import os

import torch

from vllm.config.execution_policy import communication_policy
from vllm.config.sm70_draft import UNITS as UNITS
from vllm.config.sm70_draft import parse_legacy_units
from vllm.diagnostics import diagnostic_channel
from vllm.triton_utils import tl, triton

ENV = "ONECAT_DRAFT47"
DECODE_GRAPH_SIZES = (6, 7, 8)


def units() -> frozenset[str]:
    return frozenset(parse_legacy_units(os.getenv(ENV, "")))


def enabled(unit: str) -> bool:
    return unit in units()


def note_route(unit: str, route: str) -> None:
    from vllm.model_executor.layers import sm70_fuse47

    sm70_fuse47.note_route(unit, route)


# --------------------------------------------------------------------------
# d2b: FULL drafter decode graphs for 6-8 requests
# --------------------------------------------------------------------------
def extend_draft_decode_graphs(manager, *, requested: bool | None = None) -> None:
    """Add the 6-8 request shapes to the drafter decode graph manager.

    Only this manager's own capture list changes; the shared
    ``compilation_config.cudagraph_capture_sizes`` and the target and draft
    prefill managers keep theirs.
    """
    if not (enabled("d2b") if requested is None else requested):
        return
    if manager is None or not manager.cudagraph_mode:
        note_route("d2b", "fallback:no_graphs")
        return
    if manager.decode_query_len != 1:
        note_route("d2b", "fallback:query_len")
        return
    sizes = set(manager._capture_sizes)
    extra = [
        s for s in DECODE_GRAPH_SIZES if s <= manager.max_num_reqs and s not in sizes
    ]
    if not extra:
        note_route("d2b", "fallback:no_new_sizes")
        return
    manager._capture_sizes = sorted(sizes | set(extra))
    manager._candidates = []
    manager._capture_descs = {}
    manager._init_candidates()
    note_route("d2b", "extended")


# --------------------------------------------------------------------------
# d2a: padded rows skip the drafter MoE
# --------------------------------------------------------------------------
@triton.jit
def _mask_pad_rows_kernel(
    topk_ids_ptr,
    num_valid_ptr,
    stride_row,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    num_valid = tl.load(num_valid_ptr)
    if row >= num_valid:
        offs = tl.arange(0, BLOCK_K)
        pad = tl.full([BLOCK_K], -1, topk_ids_ptr.dtype.element_ty)
        tl.store(topk_ids_ptr + row * stride_row + offs, pad, mask=offs < K)


def mask_pad_rows_(topk_ids: torch.Tensor, num_valid: torch.Tensor) -> None:
    """Set the expert ids of rows >= num_valid[0] to -1, in place."""
    if (
        topk_ids.dim() != 2
        or topk_ids.stride(1) != 1
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or not topk_ids.is_cuda
    ):
        note_route("d2a", "fallback:topk_ids")
        return
    rows, k = topk_ids.shape
    if rows == 0:
        return
    _mask_pad_rows_kernel[(rows,)](
        topk_ids,
        num_valid,
        topk_ids.stride(0),
        K=k,
        BLOCK_K=triton.next_power_of_2(k),
        num_warps=1,
    )
    note_route("d2a", "masked")


# --------------------------------------------------------------------------
# d1a: drafter vocab-parallel argmax
# --------------------------------------------------------------------------
@triton.jit
def _local_top1_pair_kernel(
    logits_ptr,
    stride_row,
    num_cols,
    vocab_start,
    pair_ptr,
    BLOCK: tl.constexpr,
):
    """One row per program: torch.max(dim=-1) of FP16 logits, written as the
    float32 pair (value, vocab_start + index).

    Each element maps to one int64 key: the 16-bit order-preserving key of its
    value in the high half (NaN above +inf, -0 folded onto +0 so that both
    zeros tie) and 0x7FFFFFFF - column in the low half. The largest key is
    torch's winner: the first NaN, else the first maximum. The value is read
    back at the winning column, so -0.0 and NaN payloads come out unchanged.
    """
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * stride_row
    offs = tl.arange(0, BLOCK)
    best = tl.full([BLOCK], -1, tl.int64)
    for start in range(0, num_cols, BLOCK):
        cols = start + offs
        in_row = cols < num_cols
        x = tl.load(base + cols, mask=in_row, other=0.0)
        bits = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        is_nan = ((bits & 0x7C00) == 0x7C00) & ((bits & 0x03FF) != 0)
        bits = tl.where(bits == 0x8000, 0, bits)
        key = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits | 0x8000)
        key = tl.where(is_nan, 0x10000, key)
        packed = (key.to(tl.int64) << 32) | (0x7FFFFFFF - cols).to(tl.int64)
        best = tl.maximum(best, tl.where(in_row, packed, -1))
    win = tl.max(best, axis=0)
    col = 0x7FFFFFFF - (win & 0xFFFFFFFF)
    value = tl.load(base + col).to(tl.float32)
    tl.store(pair_ptr + row * 2, value)
    tl.store(pair_ptr + row * 2 + 1, (col + vocab_start).to(tl.float32))


@triton.jit
def _global_top1_kernel(
    gathered_ptr,
    out_ptr,
    num_rows,
    TP: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """gathered is [TP * num_rows, 2] (all-gather along dim 0). Per row, keep
    the first rank whose value wins under torch.argmax's rule (NaN first, then
    the larger value, ties to the lower rank); write its index as int64."""
    rows = tl.arange(0, BLOCK_M)
    live = rows < num_rows
    best_v = tl.load(gathered_ptr + rows * 2, mask=live, other=0.0)
    best_i = tl.load(gathered_ptr + rows * 2 + 1, mask=live, other=0.0)
    for r in tl.static_range(1, TP):
        v = tl.load(gathered_ptr + (r * num_rows + rows) * 2, mask=live, other=0.0)
        i = tl.load(gathered_ptr + (r * num_rows + rows) * 2 + 1, mask=live, other=0.0)
        take = (best_v == best_v) & ((v != v) | (v > best_v))
        best_v = tl.where(take, v, best_v)
        best_i = tl.where(take, i, best_i)
    tl.store(out_ptr + rows, best_i.to(tl.int64), mask=live)


def top1_block_reason(
    logits: torch.Tensor, tp_size: int, *, custom_ar: bool | None = None
) -> str | None:
    if tp_size <= 1:
        return "tp1"
    if not logits.is_cuda:
        return "device"
    if logits.dtype != torch.float16:
        return "dtype"
    if logits.dim() != 2 or logits.shape[0] == 0 or logits.stride(1) != 1:
        return "shape"
    if logits.shape[0] > 4096 or logits.shape[1] >= 2**31 - 1:
        return "size"
    if custom_ar is None:
        custom_ar = communication_policy().top1_custom_ar
    if diagnostic_channel("top1_sync").policy.steps or custom_ar:
        return "top1_env"
    return None


def local_top1_pair(logits: torch.Tensor, vocab_start: int) -> torch.Tensor:
    rows, cols = logits.shape
    pair = torch.empty((rows, 2), dtype=torch.float32, device=logits.device)
    _local_top1_pair_kernel[(rows,)](
        logits,
        logits.stride(0),
        cols,
        vocab_start,
        pair,
        BLOCK=4096,
        num_warps=8,
    )
    return pair


def global_top1(gathered: torch.Tensor, tp_size: int, num_rows: int) -> torch.Tensor:
    out = torch.empty((num_rows,), dtype=torch.int64, device=gathered.device)
    _global_top1_kernel[(1,)](
        gathered,
        out,
        num_rows,
        TP=tp_size,
        BLOCK_M=triton.next_power_of_2(num_rows),
        num_warps=1,
    )
    return out


def top_tokens(
    logits: torch.Tensor,
    vocab_start: int,
    tp_size: int,
    *,
    custom_ar: bool | None = None,
) -> torch.Tensor | None:
    """d1a: the global argmax token per row, or None (with a counted reason)
    when the fused path does not apply."""
    reason = top1_block_reason(logits, tp_size, custom_ar=custom_ar)
    if reason is not None:
        note_route("d1a", f"fallback:{reason}")
        return None
    from vllm.distributed import tensor_model_parallel_all_gather

    pair = local_top1_pair(logits, vocab_start)
    gathered = tensor_model_parallel_all_gather(pair, dim=0)
    note_route("d1a", "fused")
    return global_top1(gathered, tp_size, logits.shape[0])
