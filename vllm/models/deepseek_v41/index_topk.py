# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic-length top-k for the V4.1 indexer (fp16 scores, ascending output)."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _orderable(x):
    bits = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    return tl.where(bits >= 0x8000, 0xFFFF - bits, bits | 0x8000)


@triton.jit(do_not_specialize=["width"])
def _index_topk(
    scores,
    out,
    positions,
    valid,
    width,
    stride,
    out_stride,
    RATIO: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.load(positions + row).to(tl.int32)
    n = tl.minimum((pos + 1) // RATIO, width)
    n = tl.where(tl.load(valid + row) != 0, n, 0)
    lanes = tl.arange(0, K)
    dst = out + row * out_stride
    if n <= K:
        tl.store(dst + lanes, tl.where(lanes < n, lanes, -1))
    else:
        src = scores + row.to(tl.int64) * stride
        bins = tl.arange(0, 256)
        # pass 1: high byte
        hist = tl.zeros((256,), dtype=tl.int32)
        for start in range(0, n, BLOCK):
            idx = start + tl.arange(0, BLOCK)
            m = idx < n
            u = _orderable(tl.load(src + idx, mask=m, other=0.0))
            hist += tl.histogram(u >> 8, 256, mask=m)
        total = tl.sum(hist, 0)
        suffix = total - (tl.cumsum(hist, 0) - hist)
        hi = tl.max(tl.where(suffix >= K, bins, -1), 0)
        want = K - tl.sum(tl.where(bins > hi, hist, 0), 0)
        # pass 2: low byte among the high-byte matches
        hist = tl.zeros((256,), dtype=tl.int32)
        for start in range(0, n, BLOCK):
            idx = start + tl.arange(0, BLOCK)
            m = idx < n
            u = _orderable(tl.load(src + idx, mask=m, other=0.0))
            hist += tl.histogram(u & 255, 256, mask=m & ((u >> 8) == hi))
        total = tl.sum(hist, 0)
        suffix = total - (tl.cumsum(hist, 0) - hist)
        lo = tl.max(tl.where(suffix >= want, bins, -1), 0)
        want = want - tl.sum(tl.where(bins > lo, hist, 0), 0)
        threshold = hi * 256 + lo
        written = 0
        ties = 0
        for start in range(0, n, BLOCK):
            idx = start + tl.arange(0, BLOCK)
            m = idx < n
            u = _orderable(tl.load(src + idx, mask=m, other=0.0))
            gt = (m & (u > threshold)).to(tl.int32)
            eq = (m & (u == threshold)).to(tl.int32)
            rank = ties + tl.cumsum(eq, 0) - eq
            take = gt | (eq & (rank < want).to(tl.int32))
            at = written + tl.cumsum(take, 0) - take
            tl.store(dst + at, idx, mask=take != 0)
            written += tl.sum(take, 0)
            ties += tl.sum(eq, 0)


def index_topk(out, scores, positions, valid, ratio, k=512):
    """out[r] = ascending indices of the top-k of scores[r, :n_r], -1 padded,
    n_r = (positions[r] + 1) // ratio (0 for invalid rows)."""
    assert scores.dtype == torch.float16 and scores.stride(1) == 1 and out.shape[1] == k
    _index_topk[(scores.shape[0],)](
        scores,
        out,
        positions,
        valid,
        scores.shape[1],
        scores.stride(0),
        out.stride(0),
        RATIO=ratio,
        K=k,
        BLOCK=1024,
        num_warps=8,
    )
    return out


SEG = 8192


@triton.jit
def _row_length(positions, valid, width, row, RATIO: tl.constexpr):
    pos = tl.load(positions + row).to(tl.int32)
    n = tl.minimum((pos + 1) // RATIO, width)
    return tl.where(tl.load(valid + row) != 0, n, 0)


@triton.jit(do_not_specialize=["width"])
def _seg_hist(
    scores,
    positions,
    valid,
    width,
    stride,
    hist,
    meta,
    PASS: tl.constexpr,
    RATIO: tl.constexpr,
    K: tl.constexpr,
    SEGMENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, seg = tl.program_id(0), tl.program_id(1)
    n = _row_length(positions, valid, width, row, RATIO)
    first = seg * SEGMENT
    if (n > K) & (first < n):
        src = scores + row.to(tl.int64) * stride
        hi = tl.load(meta + row * 4)
        acc = tl.zeros((256,), dtype=tl.int32)
        for start in range(first, tl.minimum(first + SEGMENT, n), BLOCK):
            idx = start + tl.arange(0, BLOCK)
            m = idx < tl.minimum(first + SEGMENT, n)
            u = _orderable(tl.load(src + idx, mask=m, other=0.0))
            if PASS == 1:
                acc += tl.histogram(u >> 8, 256, mask=m)
            else:
                acc += tl.histogram(u & 255, 256, mask=m & ((u >> 8) == hi))
        tl.atomic_add(hist + row * 256 + tl.arange(0, 256), acc)


@triton.jit(do_not_specialize=["width"])
def _seg_select(
    hist,
    meta,
    out,
    positions,
    valid,
    width,
    out_stride,
    PASS: tl.constexpr,
    RATIO: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    n = _row_length(positions, valid, width, row, RATIO)
    lanes = tl.arange(0, K)
    if n <= K:
        if PASS == 1:
            tl.store(out + row * out_stride + lanes, tl.where(lanes < n, lanes, -1))
    else:
        bins = tl.arange(0, 256)
        h = tl.load(hist + row * 256 + bins)
        total = tl.sum(h, 0)
        suffix = total - (tl.cumsum(h, 0) - h)
        want = K if PASS == 1 else tl.load(meta + row * 4 + 1)
        pick = tl.max(tl.where(suffix >= want, bins, -1), 0)
        left = want - tl.sum(tl.where(bins > pick, h, 0), 0)
        if PASS == 1:
            tl.store(meta + row * 4, pick)
            tl.store(meta + row * 4 + 1, left)
        else:
            tl.store(meta + row * 4 + 2, tl.load(meta + row * 4) * 256 + pick)
            tl.store(meta + row * 4 + 3, left)


@triton.jit(do_not_specialize=["width"])
def _seg_count(
    scores,
    positions,
    valid,
    width,
    stride,
    meta,
    counts,
    NSEG,
    RATIO: tl.constexpr,
    K: tl.constexpr,
    SEGMENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, seg = tl.program_id(0), tl.program_id(1)
    n = _row_length(positions, valid, width, row, RATIO)
    first = seg * SEGMENT
    gt = 0
    eq = 0
    if (n > K) & (first < n):
        src = scores + row.to(tl.int64) * stride
        threshold = tl.load(meta + row * 4 + 2)
        stop = tl.minimum(first + SEGMENT, n)
        for start in range(first, stop, BLOCK):
            idx = start + tl.arange(0, BLOCK)
            m = idx < stop
            u = _orderable(tl.load(src + idx, mask=m, other=0.0))
            gt += tl.sum((m & (u > threshold)).to(tl.int32), 0)
            eq += tl.sum((m & (u == threshold)).to(tl.int32), 0)
    tl.store(counts + (row * NSEG + seg) * 2, gt)
    tl.store(counts + (row * NSEG + seg) * 2 + 1, eq)


@triton.jit(do_not_specialize=["width"])
def _seg_write(
    scores,
    out,
    positions,
    valid,
    width,
    stride,
    out_stride,
    meta,
    counts,
    NSEG,
    RATIO: tl.constexpr,
    K: tl.constexpr,
    SEGMENT: tl.constexpr,
    BLOCK: tl.constexpr,
    SEGS: tl.constexpr,
):
    row, seg = tl.program_id(0), tl.program_id(1)
    n = _row_length(positions, valid, width, row, RATIO)
    first = seg * SEGMENT
    if (n > K) & (first < n):
        j = tl.arange(0, SEGS)
        before = (j < seg) & (j < NSEG)
        gt_before = tl.sum(
            tl.where(before, tl.load(counts + (row * NSEG + j) * 2, j < NSEG, 0), 0), 0
        )
        eq_before = tl.sum(
            tl.where(
                before, tl.load(counts + (row * NSEG + j) * 2 + 1, j < NSEG, 0), 0
            ),
            0,
        )
        threshold = tl.load(meta + row * 4 + 2)
        want = tl.load(meta + row * 4 + 3)
        src = scores + row.to(tl.int64) * stride
        dst = out + row * out_stride
        stop = tl.minimum(first + SEGMENT, n)
        for start in range(first, stop, BLOCK):
            idx = start + tl.arange(0, BLOCK)
            m = idx < stop
            u = _orderable(tl.load(src + idx, mask=m, other=0.0))
            g = (m & (u > threshold)).to(tl.int32)
            e = (m & (u == threshold)).to(tl.int32)
            ge = gt_before + tl.cumsum(g, 0) - g
            ee = eq_before + tl.cumsum(e, 0) - e
            take = (g != 0) | ((e != 0) & (ee < want))
            tl.store(dst + ge + tl.minimum(ee, want), idx, mask=take)
            gt_before += tl.sum(g, 0)
            eq_before += tl.sum(e, 0)


_SEG_WORK = {}


def index_topk_parallel(out, scores, positions, valid, ratio, k=512):
    """Same output as index_topk; segments of SEG scores run in parallel (few rows)."""
    assert scores.dtype == torch.float16 and scores.stride(1) == 1 and out.shape[1] == k
    rows, width = scores.shape
    nseg = triton.cdiv(width, SEG)
    key = (scores.device, rows, nseg)
    if key not in _SEG_WORK:
        _SEG_WORK[key] = (
            torch.zeros(2, rows, 256, dtype=torch.int32, device=scores.device),
            torch.zeros(rows, 4, dtype=torch.int32, device=scores.device),
            torch.zeros(rows, nseg, 2, dtype=torch.int32, device=scores.device),
        )
    hist, meta, counts = _SEG_WORK[key]
    hist.zero_()
    grid = (rows, nseg)
    common = dict(RATIO=ratio, K=k)
    _seg_hist[grid](
        scores,
        positions,
        valid,
        width,
        scores.stride(0),
        hist[0],
        meta,
        PASS=1,
        SEGMENT=SEG,
        BLOCK=1024,
        **common,
    )
    _seg_select[(rows,)](
        hist[0], meta, out, positions, valid, width, out.stride(0), PASS=1, **common
    )
    _seg_hist[grid](
        scores,
        positions,
        valid,
        width,
        scores.stride(0),
        hist[1],
        meta,
        PASS=2,
        SEGMENT=SEG,
        BLOCK=1024,
        **common,
    )
    _seg_select[(rows,)](
        hist[1], meta, out, positions, valid, width, out.stride(0), PASS=2, **common
    )
    _seg_count[grid](
        scores,
        positions,
        valid,
        width,
        scores.stride(0),
        meta,
        counts,
        nseg,
        SEGMENT=SEG,
        BLOCK=1024,
        **common,
    )
    _seg_write[grid](
        scores,
        out,
        positions,
        valid,
        width,
        scores.stride(0),
        out.stride(0),
        meta,
        counts,
        nseg,
        SEGMENT=SEG,
        BLOCK=1024,
        SEGS=triton.next_power_of_2(nseg),
        **common,
    )
    return out
