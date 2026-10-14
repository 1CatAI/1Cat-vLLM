# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compose existing exact QSA selectors over bounded score partitions.

Every globally selected index belongs to its partition's local Top512. Local
and merged stages retain FP32 scores and lower-index ties. Their increasing
candidate order is also increasing global index order, including a short
causal tail, so the existing exact selector preserves the canonical set/order.
"""

import math

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _qsa_partition_prepare(
    Scores,
    Lengths,
    PartitionScores,
    PartitionLengths,
    STRIDE: tl.constexpr,
    Selected: tl.constexpr,
    PARTS: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    index = part * TILE + tl.arange(0, BLOCK)
    length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), Selected)
    value = tl.load(
        Scores + row * STRIDE + index,
        (index < Selected) & (index < length),
        float("-inf"),
    )
    tl.store(PartitionScores + (row * PARTS + part) * TILE + tl.arange(0, BLOCK), value)
    tl.store(
        PartitionLengths + row * PARTS + part,
        tl.minimum(tl.maximum(length - part * TILE, 0), TILE),
    )


@triton.jit
def _qsa_partition_gather(
    Scores,
    Lengths,
    LocalIndices,
    CandidateScores,
    GlobalIndices,
    CandidateLengths,
    STRIDE: tl.constexpr,
    Selected: tl.constexpr,
    PARTS: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    offset = tl.arange(0, BLOCK)
    local = tl.load(LocalIndices + (row * PARTS + part) * 512 + offset)
    index = part * TILE + local
    valid = (local >= 0) & (index < Selected)
    score = tl.load(Scores + row * STRIDE + index, valid, float("-inf"))
    tl.store(CandidateScores + (row * PARTS + part) * 512 + offset, score)
    tl.store(
        GlobalIndices + (row * PARTS + part) * 512 + offset, tl.where(valid, index, -1)
    )
    length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), Selected)
    full = length // TILE
    tail = length % TILE
    if part == 0:
        tl.store(CandidateLengths + row, full * 512 + tl.minimum(tail, 512))


@triton.jit
def _qsa_partition_remap(
    Selected, GlobalIndices, Output, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    index = tl.load(Selected + row * 512 + offset)
    value = tl.load(GlobalIndices + row * WIDTH + tl.maximum(index, 0), index >= 0, -1)
    tl.store(Output + row * 512 + offset, value)


def qsa_partition_topk(x, lengths, out=None, *, tile=8192):
    rows, cols = x.shape
    parts = math.ceil(cols / tile)
    if (
        x.dtype != torch.float32
        or x.stride(1) != 1
        or not x.is_cuda
        or lengths.dtype != torch.int32
        or not lengths.is_contiguous()
        or lengths.shape != (rows,)
        or lengths.device != x.device
    ):
        raise ValueError("Invalid exact partition Top-K operands")
    if out is not None and (
        out.shape != (rows, 512)
        or out.dtype != torch.int32
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError("Invalid exact partition Top-K output")
    partial = torch.empty((rows * parts, tile), device=x.device, dtype=torch.float32)
    pl = torch.empty(rows * parts, device=x.device, dtype=torch.int32)
    indices = torch.empty((rows * parts, 512), device=x.device, dtype=torch.int32)
    scores = torch.empty((rows, parts * 512), device=x.device, dtype=torch.float32)
    global_ids = torch.empty((rows, parts * 512), device=x.device, dtype=torch.int32)
    counts = torch.empty(rows, device=x.device, dtype=torch.int32)
    chosen = torch.empty((rows, 512), device=x.device, dtype=torch.int32)
    out = torch.empty_like(chosen) if out is None else out
    _qsa_partition_prepare[(rows, parts)](
        x, lengths, partial, pl, x.stride(0), cols, parts, tile, tile, num_warps=8
    )
    torch.ops._C.qsa_lexicographic_topk(partial, pl, indices, 512, True)
    _qsa_partition_gather[(rows, parts)](
        x,
        lengths,
        indices,
        scores,
        global_ids,
        counts,
        x.stride(0),
        cols,
        parts,
        tile,
        512,
        num_warps=4,
    )
    torch.ops._C.qsa_lexicographic_topk(scores, counts, chosen, 512, True)
    _qsa_partition_remap[(rows,)](
        chosen, global_ids, out, parts * 512, 512, num_warps=4
    )
    return out
