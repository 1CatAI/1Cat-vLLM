# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Level one of DeepSeek-V4.1's two-level top-k: candidate blocks (PORT_DESIGN §1 "Candidate block mask"; L-ATTN).

Port of upstream ``vllm/model_executor/kernels/attention/dsa/candidate_blocks.py`` (b6d8e8af; pure Triton,
no ``tl.dot``) with the reference semantics of ref:m.py:583-610 (``select_candidate_blocks``):

* block score = max of the indexer scores of the block's ``block_size`` (8) positions; positions a query
  cannot see (``>= end``) count as ``-inf``;
* the block holding the query's newest visible position, ``(end - 1) // block_size``, is pinned to ``+inf``;
* the ``topk_blocks`` (2048) best blocks are kept; picks whose score is ``-inf`` are dropped;
* Reindex layers (24, 28, 32, 36) set every position outside the kept blocks to ``-inf``.

Adaptations to the cross-lane contract (``SharedAttnBuffers.candidate_blocks``): rows hold the kept block
ids **ascending**, ``-1`` padded, and ``row[0] == CAND_ALL`` when the query sees at most ``topk_blocks``
blocks -- then every reachable block is kept and the mask is a no-op (the reference's behaviour at
<= 16,384 positions), so the masking kernel leaves such rows causal-only. Row bounds start at 0 (scores are
indexed by logical compressed position). NaN propagates through the block max.
"""

from __future__ import annotations

import torch

from vllm.models.deepseek_v41.common.contracts import CAND_ALL
from vllm.triton_utils import tl, triton


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(logits, ends, scores, stride_row, stride_col, width, nblocks,
                         BLOCK_SIZE: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    end = tl.load(ends + row)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    cols = blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    values = tl.load(
        logits + row * stride_row + cols * stride_col,
        (blocks[:, None] < nblocks) & (offsets[None, :] < BLOCK_SIZE) & (cols < end) & (cols < width),
        other=-float("inf"))
    reduced = tl.reduce(values, 1, _max_with_nan)
    reduced = tl.where((end > 0) & (blocks == (end - 1) // BLOCK_SIZE), float("inf"), reduced)
    tl.store(scores + row * nblocks + blocks, reduced, blocks < nblocks)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _candidate_flags_kernel(candidates, flags, stride_row, stride_col, width, nblocks,
                            BLOCK_SIZE: tl.constexpr, K: tl.constexpr, CAND_ALL_ID: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    first = tl.load(candidates + row * stride_row)
    keep_all = first == CAND_ALL_ID
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(nblocks, 1024)):
        slots = tile * 1024 + offsets
        tl.store(flags + row * nblocks + slots, keep_all.to(tl.uint8), slots < nblocks)
    tl.debug_barrier()
    if first != CAND_ALL_ID:
        cols = tl.arange(0, triton.next_power_of_2(K))
        block = tl.load(candidates + row * stride_row + cols * stride_col, cols < K, other=-1).to(tl.int64)
        tl.store(flags + row * nblocks + block, 1, (cols < K) & (block >= 0) & (block < nblocks))


@triton.jit(do_not_specialize=["width", "nblocks"])
def _mask_candidates_kernel(logits, ends, flags, stride_row, stride_col, width, nblocks,
                            BLOCK_SIZE: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    end = tl.load(ends + row)
    valid = (cols < end) & (cols < width)
    keep = tl.load(flags + row * nblocks + cols // BLOCK_SIZE, valid, other=0)
    tl.store(logits + row * stride_row + cols * stride_col, -float("inf"), (cols < width) & ~(valid & (keep != 0)))


def select_candidate_blocks(logits: torch.Tensor, ends: torch.Tensor, topk_blocks: int, block_size: int,
                            out: torch.Tensor) -> None:
    """logits [rows, width] FP32 scores by logical position; ends [rows] int (visible count per row).
    Writes ``out`` [rows, topk_blocks] int32: ascending kept block ids, -1 padded; ``CAND_ALL`` in column 0
    for rows that see <= topk_blocks blocks."""
    assert logits.is_cuda and logits.dtype == torch.float32 and out.dtype == torch.int32
    rows, width = logits.shape
    if out.shape != (rows, topk_blocks):
        raise ValueError(f"out must be [{rows}, {topk_blocks}], got {tuple(out.shape)}")
    if not rows:
        return
    out.fill_(-1)
    ends64 = ends.to(torch.int64)
    reach = torch.div(ends64 + block_size - 1, block_size, rounding_mode="floor")
    if width:
        nblocks = triton.cdiv(width, block_size)
        scores = logits.new_empty((rows, nblocks))
        _block_scores_kernel[(rows, triton.cdiv(nblocks, 128))](
            logits, ends64, scores, *logits.stride(), width, nblocks, block_size, 128)
        k = min(topk_blocks, nblocks)
        top = scores.topk(k, dim=-1)                       # same tie behaviour as the reference
        ids = torch.where(top.values != -float("inf"), top.indices, torch.full_like(top.indices, -1))
        ids = torch.where(ids < 0, torch.full_like(ids, torch.iinfo(torch.int64).max), ids)
        ids = ids.sort(dim=-1).values
        ids = torch.where(ids == torch.iinfo(torch.int64).max, torch.full_like(ids, -1), ids)
        out[:, :k] = ids.to(torch.int32)
    all_rows = reach <= topk_blocks
    out[:, 0] = torch.where(all_rows, torch.full_like(out[:, 0], CAND_ALL), out[:, 0])


def apply_candidate_mask(logits: torch.Tensor, ends: torch.Tensor, candidate_blocks: torch.Tensor,
                         block_size: int) -> None:
    """In place: positions >= ends[row] and positions outside the row's candidate blocks -> -inf
    (rows whose column 0 is ``CAND_ALL`` get the causal bound only)."""
    assert logits.is_cuda and logits.dtype == torch.float32
    rows, width = logits.shape
    if not rows or not width:
        return
    nblocks = triton.cdiv(width, block_size)
    flags = torch.empty((rows, nblocks), device=logits.device, dtype=torch.uint8)
    _candidate_flags_kernel[(rows,)](candidate_blocks, flags, *candidate_blocks.stride(), width, nblocks,
                                     block_size, candidate_blocks.shape[1], CAND_ALL)
    _mask_candidates_kernel[(rows, triton.cdiv(width, 1024))](
        logits, ends.to(torch.int64), flags, *logits.stride(), width, nblocks, block_size, 1024)


# ----------------------------------------------------------------------------- torch reference twin
def select_candidate_blocks_torch(logits: torch.Tensor, ends: torch.Tensor, topk_blocks: int,
                                  block_size: int) -> torch.Tensor:
    """ref:m.py:583-610 on rows with per-row ``ends`` -> bool mask [rows, width] (True = kept)."""
    width = logits.size(-1)
    cols = torch.arange(width, device=logits.device)
    masked = logits.masked_fill(cols[None, :] >= ends[:, None], -torch.inf)
    scores = torch.nn.functional.pad(masked, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = torch.div(ends - 1, block_size, rounding_mode="floor")
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device)[None, :] == last[:, None], torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
