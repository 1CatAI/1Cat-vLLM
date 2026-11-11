# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 paged latent caches and bounded sparse attention for local TP4."""

import contextlib

import torch

import vllm.envs as envs
from vllm.models.deepseek_v4.common.ops.sparse_prefill_bmm import attend_gathered_keys
from vllm.triton_utils import tl, triton


@triton.jit
def _insert(
    values,
    positions,
    valid_tokens,
    block_table,
    cache,
    STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    position = tl.load(positions + token).to(tl.int64)
    valid = tl.load(valid_tokens + token) & ((position + 1) % RATIO == 0)
    page = tl.load(block_table + position // BLOCK_SIZE, valid, -1).to(tl.int64)
    valid &= page >= 0
    col = tl.arange(0, BLOCK)
    data = tl.load(values + token * D + col, valid & (col < D), 0)
    row = (position % BLOCK_SIZE) // RATIO
    tl.store(cache + page * STRIDE + row * D + col, data, valid & (col < D))


def insert_latents(
    cache, values, positions, valid_tokens, block_table, *, block_size, ratio=1
):
    """Insert only complete groups; values have already passed RoPE and QDQ."""
    _validate_cache(cache, block_table, block_size, ratio)
    if (
        values.shape != (positions.numel(), cache.shape[-1])
        or values.dtype != torch.float16
        or valid_tokens.shape != positions.shape
    ):
        raise ValueError("V4.1 latent insertion shapes do not match")
    _insert[(values.shape[0],)](
        values,
        positions,
        valid_tokens,
        block_table,
        cache,
        STRIDE=cache.stride(0),
        BLOCK_SIZE=block_size,
        RATIO=ratio,
        D=values.shape[1],
        BLOCK=triton.next_power_of_2(values.shape[1]),
    )


def _validate_cache(cache, block_table, block_size, ratio):
    if (
        not cache.is_cuda
        or cache.dtype != torch.float16
        or cache.ndim != 3
        or cache.shape[1] != block_size // ratio
        or block_size % ratio
        or cache.shape[2] not in (128, 512)
        or cache.stride(1) != cache.shape[2]
        or cache.stride(2) != 1
        or block_table.ndim != 2
        or block_table.shape[0] != 1
        or block_table.dtype != torch.int32
        or block_table.device != cache.device
    ):
        raise ValueError("V4.1 caches require one physical block table and FP16 rows")


@triton.jit(do_not_specialize=["TOKENS"])
def _gather_index(
    cache,
    block_table,
    positions,
    valid_tokens,
    out,
    STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    TOKENS,
):
    index = tl.program_id(0).to(tl.int64)
    end = tl.load(positions + TOKENS - 1).to(tl.int64) + 1
    valid = tl.load(valid_tokens) & (index < end // RATIO)
    page = tl.load(block_table + index // (BLOCK_SIZE // RATIO), valid, -1).to(tl.int64)
    col = tl.arange(0, 128)
    value = tl.load(
        cache + page * STRIDE + (index % (BLOCK_SIZE // RATIO)) * 128 + col,
        valid & (page >= 0),
        0,
    )
    tl.store(out + index * 128 + col, value)


def gather_index_keys(
    cache, block_table, positions, valid_tokens, *, block_size, ratio, max_context=16384
):
    """One bounded contiguous key matrix; unreachable rows are zero-filled."""
    _validate_cache(cache, block_table, block_size, ratio)
    if cache.shape[-1] != 128 or max_context % ratio:
        raise ValueError("V4.1 index keys require 128 dimensions")
    result = torch.empty(
        max_context // ratio, 128, dtype=torch.float16, device=cache.device
    )
    _gather_index[(result.shape[0],)](
        cache,
        block_table,
        positions,
        valid_tokens,
        result,
        STRIDE=cache.stride(0),
        BLOCK_SIZE=block_size,
        RATIO=ratio,
        TOKENS=positions.numel(),
    )
    return result


_FUSED_TOPK = envs.VLLM_V41_INDEX_TOPK == "1"
# VLLM_V41_INDEX_SHARD=1 (default off): batched decode indexer sharded by key
# tiles across TP ranks instead of by heads (_sharded_index_topk).
_INDEX_SHARD = envs.VLLM_V41_INDEX_SHARD == "1"


_INDEX_TILE_BYTES = int(envs.VLLM_V41_INDEX_TILE_MB) << 20

# L4: with this knob on, select_compressed_positions hands
# off to _select_compressed_positions_overlap, where the TP reduction and
# top-k of score tile i run on a side stream while the main stream scores
# tile i + 1. Same kernels on the same inputs, and every rank issues the same
# collectives in the same order. Off by default (the port maps
# VLLM_V41_INDEX_OVERLAP here); off, none of the code below runs.
_INDEX_OVERLAP = envs.VLLM_V41_INDEX_OVERLAP == "1"
_INDEX_SIDE: dict = {}


def _index_overlap(q, tile):
    return (
        _INDEX_OVERLAP
        and q.is_cuda
        and q.shape[0] > tile
        and not torch.cuda.is_current_stream_capturing()
    )


def _index_side_stream(device):
    stream = _INDEX_SIDE.get(device)
    if stream is None:
        stream = _INDEX_SIDE[device] = torch.cuda.Stream(device=device)
    return stream


@contextlib.contextmanager
def _on_side_stream(head):
    """Fork: the side stream waits for everything queued so far on the main
    stream, and the allocator keeps ``head`` (allocated on the main stream)
    until the side stream's work on it is done."""
    side = _index_side_stream(head.device)
    side.wait_stream(torch.cuda.current_stream())
    head.record_stream(side)
    with torch.cuda.stream(side):
        yield


def _join_side_stream(device):
    torch.cuda.current_stream().wait_stream(_index_side_stream(device))


def _index_tile_topk(
    head, all_reduce, result, column, positions, valid_tokens, first, last, ratio, top_k
):
    """TP-reduce one tile's head sums and write its top-k rows of ``result``
    (the tail of select_compressed_positions' tile loop)."""
    width = result.shape[1]
    scores = all_reduce(head)
    if _FUSED_TOPK and width == top_k and scores.dtype == torch.float16:
        from vllm.models.deepseek_v41.index_topk import index_topk, index_topk_parallel

        # One CTA per row scans the whole row three times: past 16K columns
        # the segmented version (rows x 8K segments; segments past a row's
        # length exit) keeps long contexts from serializing on it.
        select = index_topk_parallel if scores.shape[1] > 16384 else index_topk
        select(
            result[first:last],
            scores.contiguous(),
            positions[first:last],
            valid_tokens[first:last],
            ratio,
            top_k,
        )
        return
    length = ((positions[first:last] + 1) // ratio)[:, None]
    scores.masked_fill_(
        (column >= length) | ~valid_tokens[first:last, None], float("-inf")
    )
    indices = scores.topk(width, dim=-1, sorted=False).indices.sort(dim=-1).values
    indices = torch.where(indices < length, indices, -1)
    indices = torch.where(valid_tokens[first:last, None], indices, -1)
    result[first:last].copy_(indices)


def select_compressed_positions(
    q, weights, keys, positions, valid_tokens, ratio, all_reduce, *, top_k=512
):
    """Keep top-512 even below 16K: only candidate-block prefilter is bypassed.

    Score tiles hold at most 32 query tokens. The official FP16 einsum,
    ReLU/weight and head-sum boundaries are retained before the TP reduction.
    No full [prefill, heads, context] tensor is materialized.
    """
    if _INDEX_OVERLAP:
        return _select_compressed_positions_overlap(
            q, weights, keys, positions, valid_tokens, ratio, all_reduce, top_k=top_k
        )
    if (
        q.dtype != torch.float16
        or q.shape[2] != 128
        or weights.shape != q.shape[:2]
        or weights.dtype != torch.float16
        or keys.shape[1] != 128
    ):
        raise ValueError("V4.1 indexer requires FP16 queries, keys and weights")
    width = min(top_k, keys.shape[0])
    result = torch.empty(q.shape[0], width, device=q.device, dtype=torch.int32)
    column = torch.arange(keys.shape[0], device=q.device)[None]
    # Long chunks: size query tiles to a fixed FP16 score budget instead of 32
    # rows, so launches and TP reductions stay O(chunk / tile).
    budget = _INDEX_TILE_BYTES // max(1, q.shape[1] * keys.shape[0] * 2)
    # 32-row multiples up to 1024; past ~64K keys the FP16 score budget
    # allows fewer than 32 rows per tile.
    tile = max(1, min(1024, budget // 32 * 32 if budget >= 32 else budget))
    for first in range(0, q.shape[0], tile):
        last = min(first + tile, q.shape[0])
        heads = q.shape[1]
        scores = torch.mm(q[first:last].reshape(-1, 128), keys.T).view(
            last - first, heads, -1
        )
        scores.relu_().mul_(weights[first:last, :, None])
        scores = all_reduce(scores.sum(1))
        if _FUSED_TOPK and width == top_k and scores.dtype == torch.float16:
            from vllm.models.deepseek_v41.index_topk import index_topk

            index_topk(
                result[first:last],
                scores.contiguous(),
                positions[first:last],
                valid_tokens[first:last],
                ratio,
                top_k,
            )
            continue
        length = ((positions[first:last] + 1) // ratio)[:, None]
        scores.masked_fill_(
            (column >= length) | ~valid_tokens[first:last, None], float("-inf")
        )
        indices = scores.topk(width, dim=-1, sorted=False).indices.sort(dim=-1).values
        indices = torch.where(indices < length, indices, -1)
        indices = torch.where(valid_tokens[first:last, None], indices, -1)
        result[first:last].copy_(indices)
    return result


def _select_compressed_positions_overlap(
    q, weights, keys, positions, valid_tokens, ratio, all_reduce, *, top_k=512
):
    """select_compressed_positions with L4: the same score tiles, and tile i's
    TP reduction and top-k (_index_tile_topk) run on a side stream while the
    main stream scores tile i + 1. For CPU tensors, a single tile and under
    stream capture it all stays on the current stream."""
    if (
        q.dtype != torch.float16
        or q.shape[2] != 128
        or weights.shape != q.shape[:2]
        or weights.dtype != torch.float16
        or keys.shape[1] != 128
    ):
        raise ValueError("V4.1 indexer requires FP16 queries, keys and weights")
    width = min(top_k, keys.shape[0])
    result = torch.empty(q.shape[0], width, device=q.device, dtype=torch.int32)
    column = torch.arange(keys.shape[0], device=q.device)[None]
    budget = _INDEX_TILE_BYTES // max(1, q.shape[1] * keys.shape[0] * 2)
    # 32-row multiples up to 1024; past ~64K keys the FP16 score budget
    # allows fewer than 32 rows per tile.
    tile = max(1, min(1024, budget // 32 * 32 if budget >= 32 else budget))
    overlap = _index_overlap(q, tile)
    for first in range(0, q.shape[0], tile):
        last = min(first + tile, q.shape[0])
        heads = q.shape[1]
        scores = torch.mm(q[first:last].reshape(-1, 128), keys.T).view(
            last - first, heads, -1
        )
        scores.relu_().mul_(weights[first:last, :, None])
        head = scores.sum(1)
        del scores  # the [tile, heads, N] buffer is freed on the main stream
        with _on_side_stream(head) if overlap else contextlib.nullcontext():
            _index_tile_topk(
                head,
                all_reduce,
                result,
                column,
                positions,
                valid_tokens,
                first,
                last,
                ratio,
                top_k,
            )
    if overlap:
        _join_side_stream(q.device)
    return result


@triton.jit
def _gather_attention(
    window_cache,
    window_table,
    compressed_cache,
    compressed_table,
    positions,
    valid_tokens,
    topk,
    keys,
    unused,
    WINDOW_STRIDE: tl.constexpr,
    COMPRESSED_STRIDE: tl.constexpr,
    WINDOW_BLOCK: tl.constexpr,
    COMPRESSED_BLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
):
    token, key = tl.program_id(0), tl.program_id(1)
    position = tl.load(positions + token).to(tl.int64)
    valid = tl.load(valid_tokens + token)
    col = tl.arange(0, 512)
    if key < WINDOW:
        at = position - WINDOW + 1 + key
        valid &= at >= 0
        page = tl.load(window_table + at // WINDOW_BLOCK, valid, -1).to(tl.int64)
        valid &= page >= 0
        row = at % WINDOW_BLOCK
        value = tl.load(window_cache + page * WINDOW_STRIDE + row * 512 + col, valid, 0)
    else:
        index = tl.load(topk + token * TOPK + key - WINDOW).to(tl.int64)
        valid &= (index >= 0) & (index < (position + 1) // RATIO)
        page = tl.load(
            compressed_table + index // (COMPRESSED_BLOCK // RATIO), valid, -1
        ).to(tl.int64)
        valid &= page >= 0
        row = index % (COMPRESSED_BLOCK // RATIO)
        value = tl.load(
            compressed_cache + page * COMPRESSED_STRIDE + row * 512 + col, valid, 0
        )
    tl.store(keys + (token * (WINDOW + TOPK) + key) * 512 + col, value)
    tl.store(unused + token * (WINDOW + TOPK) + key, ~valid)


WINDOW_RING = 256  # positions per layer ring; > window + 6 draft rows


@triton.jit
def _gather_attention_ring(
    ring,
    values,
    starts,
    compressed_cache,
    compressed_table,
    positions,
    valid_tokens,
    topk,
    keys,
    unused,
    COMPRESSED_STRIDE: tl.constexpr,
    COMPRESSED_BLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    RING: tl.constexpr,
):
    """Window keys of this forward come from its latents, older ones from the ring."""
    token, key = tl.program_id(0), tl.program_id(1)
    position = tl.load(positions + token).to(tl.int64)
    valid = tl.load(valid_tokens + token)
    col = tl.arange(0, 512)
    if key < WINDOW:
        at = position - WINDOW + 1 + key
        valid &= at >= 0
        start = tl.load(starts).to(tl.int64)
        inside = at >= start
        old = tl.load(ring + (at % RING) * 512 + col, valid & ~inside, 0)
        new = tl.load(values + (at - start) * 512 + col, valid & inside, 0)
        value = tl.where(inside, new, old)
    else:
        index = tl.load(topk + token * TOPK + key - WINDOW).to(tl.int64)
        valid &= (index >= 0) & (index < (position + 1) // RATIO)
        page = tl.load(
            compressed_table + index // (COMPRESSED_BLOCK // RATIO), valid, -1
        ).to(tl.int64)
        valid &= page >= 0
        row = index % (COMPRESSED_BLOCK // RATIO)
        value = tl.load(
            compressed_cache + page * COMPRESSED_STRIDE + row * 512 + col, valid, 0
        )
    tl.store(keys + (token * (WINDOW + TOPK) + key) * 512 + col, value)
    tl.store(unused + token * (WINDOW + TOPK) + key, ~valid)


@triton.jit(do_not_specialize=["FIRST"])
def _ring_insert(values, positions, valid_tokens, ring, FIRST, RING: tl.constexpr):
    token = tl.program_id(0) + FIRST
    position = tl.load(positions + token).to(tl.int64)
    valid = tl.load(valid_tokens + token) & (position >= 0)
    col = tl.arange(0, 512)
    data = tl.load(values + token * 512 + col, valid, 0)
    tl.store(ring + (position % RING) * 512 + col, data, valid)


def insert_window_ring(ring, values, positions, valid_tokens):
    """After attention: keep the last RING positions (no in-launch slot reuse)."""
    tokens = values.shape[0]
    if values.shape != (tokens, 512) or ring.shape != (WINDOW_RING, 512):
        raise ValueError("V4.1 window ring shapes do not match")
    first = max(0, tokens - WINDOW_RING)
    if tokens > first:
        _ring_insert[(tokens - first,)](
            values, positions, valid_tokens, ring, first, RING=WINDOW_RING
        )


@triton.jit(do_not_specialize=["window_cache", "ring", "values", "compressed_cache"])
def _attention_addresses(
    window_cache,
    window_table,
    ring,
    values,
    compressed_cache,
    compressed_table,
    positions,
    valid_tokens,
    topk,
    addresses,
    unused,
    WINDOW_STRIDE: tl.constexpr,
    COMPRESSED_STRIDE: tl.constexpr,
    WINDOW_BLOCK: tl.constexpr,
    COMPRESSED_BLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    RING: tl.constexpr,
    USE_RING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Element addresses of each query's window + selected compressed rows."""
    token, part = tl.program_id(0), tl.program_id(1)
    key = part * BLOCK + tl.arange(0, BLOCK)
    width: tl.constexpr = WINDOW + TOPK
    inside_width = key < width
    position = tl.load(positions + token).to(tl.int64)
    valid = tl.load(valid_tokens + token) & inside_width
    is_window = key < WINDOW
    at = position - WINDOW + 1 + key
    wvalid = valid & is_window & (at >= 0)
    if USE_RING:
        start = tl.load(positions).to(tl.int64)
        inside = at >= start
        waddr = tl.where(
            inside,
            values + (at - start) * 1024,
            ring + (at % RING) * 1024,
        )
    else:
        page = tl.load(window_table + at // WINDOW_BLOCK, wvalid, -1).to(tl.int64)
        wvalid &= page >= 0
        waddr = window_cache + (page * WINDOW_STRIDE + (at % WINDOW_BLOCK) * 512) * 2
    index = tl.load(topk + token * TOPK + key - WINDOW, valid & ~is_window, -1).to(
        tl.int64
    )
    cvalid = valid & ~is_window & (index >= 0) & (index < (position + 1) // RATIO)
    cpage = tl.load(
        compressed_table + index // (COMPRESSED_BLOCK // RATIO), cvalid, -1
    ).to(tl.int64)
    cvalid &= cpage >= 0
    caddr = (
        compressed_cache
        + (cpage * COMPRESSED_STRIDE + (index % (COMPRESSED_BLOCK // RATIO)) * 512) * 2
    )
    ok = tl.where(is_window, wvalid, cvalid)
    addr = tl.where(is_window, waddr, caddr)
    tl.store(addresses + token * width + key, tl.where(ok, addr, 0), inside_width)
    tl.store(unused + token * width + key, ~ok, inside_width)


@triton.jit
def _sparse_attention_fused(
    q,
    addresses,
    unused,
    sink,
    out,
    scale,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BK: tl.constexpr,
):
    """One query token, all heads: online softmax with the attention sink, K = V."""
    token = tl.program_id(0)
    h = tl.arange(0, HEADS)
    d = tl.arange(0, 512)
    query = tl.load(q + (token * HEADS + h[:, None]) * 512 + d[None, :])
    m = tl.full((HEADS,), float("-inf"), tl.float32)
    l = tl.zeros((HEADS,), tl.float32)  # noqa: E741
    acc = tl.zeros((HEADS, 512), tl.float32)
    for kb in range(0, WIDTH, BK):
        k = kb + tl.arange(0, BK)
        live = k < WIDTH
        addr = tl.load(addresses + token * WIDTH + k, live, 0)
        dead = (tl.load(unused + token * WIDTH + k, live, 1) != 0) | ~live
        rows = addr.to(tl.pointer_type(tl.float16))
        kv = tl.load(rows[:, None] + d[None, :], ~dead[:, None], 0.0)
        s = tl.dot(query, tl.trans(kv)) * scale
        s = tl.where(dead[None, :], float("-inf"), s)
        m_new = tl.maximum(m, tl.max(s, 1))
        safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        prob = tl.exp(s - safe[:, None])
        alpha = tl.exp(m - safe)
        l = l * alpha + tl.sum(prob, 1)  # noqa: E741
        acc = acc * alpha[:, None] + tl.dot(prob.to(tl.float16), kv)
        m = m_new
    z = tl.load(sink + h)
    top = tl.maximum(m, z)
    factor = tl.exp(m - top)
    total = l * factor + tl.exp(z - top)
    result = acc * (factor / total)[:, None]
    tl.store(
        out + (token * HEADS + h[:, None]) * 512 + d[None, :], result.to(tl.float16)
    )


_FUSED_SPARSE = envs.VLLM_V41_FUSED_SPARSE == "1"
_FUSED_MIN_TOKENS = int(envs.VLLM_V41_FUSED_SPARSE_MIN)


def _attend_sparse_fused(
    q,
    positions,
    valid_tokens,
    window_cache,
    window_table,
    attn_sink,
    window_block,
    compressed_cache,
    compressed_table,
    compressed_block,
    ratio,
    topk,
    window,
    window_ring,
    window_values,
):
    tokens, heads = q.shape[0], q.shape[1]
    compressed = compressed_cache is not None
    count = topk.shape[1] if compressed else 0
    width = window + count
    addresses = torch.empty(tokens, width, dtype=torch.int64, device=q.device)
    unused = torch.empty(tokens, width, dtype=torch.bool, device=q.device)
    ring = window_ring is not None
    _attention_addresses[(tokens, triton.cdiv(width, 128))](
        0 if ring else window_cache.data_ptr(),
        window_table,
        window_ring.data_ptr() if ring else 0,
        window_values.data_ptr() if ring else 0,
        compressed_cache.data_ptr() if compressed else 0,
        compressed_table if compressed else window_table,
        positions,
        valid_tokens,
        topk if compressed else positions,
        addresses,
        unused,
        WINDOW_STRIDE=0 if ring else window_cache.stride(0),
        COMPRESSED_STRIDE=compressed_cache.stride(0) if compressed else 0,
        WINDOW_BLOCK=window_block,
        COMPRESSED_BLOCK=compressed_block,
        RATIO=max(ratio, 1),
        WINDOW=window,
        TOPK=count,
        RING=WINDOW_RING,
        USE_RING=ring,
        BLOCK=128,
    )
    result = torch.empty_like(q)
    _sparse_attention_fused[(tokens,)](
        q,
        addresses,
        unused,
        attn_sink,
        result,
        512**-0.5,
        HEADS=heads,
        WIDTH=width,
        BK=32,
        num_warps=8,
    )
    return result


# ---- G4b packed KV rows (see kv_pack.py) ------------------------------------
from vllm.models.deepseek_v41.kv_pack import (  # noqa: E402
    C_ROW,
    I_ROW,
    pack_c_group,
    pack_i_group,
    unpack_c_row,
)


@triton.jit(do_not_specialize=["FIRST"])
def _insert_packed(
    values,
    positions,
    valid_tokens,
    rows,
    FIRST,
    RATIO: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    ROW: tl.constexpr,
    COMPRESSED: tl.constexpr,
):
    """Pack complete compression groups into their position-addressed rows."""
    token = tl.program_id(0) + FIRST
    group = tl.program_id(1)
    position = tl.load(positions + token).to(tl.int64)
    valid = (
        tl.load(valid_tokens + token) & ((position + 1) % RATIO == 0) & (position >= 0)
    )
    if valid:
        row = rows + (position // RATIO) * ROW
        if COMPRESSED:
            pack_c_group(values + token.to(tl.int64) * DIM, row, group)
        else:
            pack_i_group(values + token.to(tl.int64) * DIM, row, group)


def insert_packed_rows(rows, values, positions, valid_tokens, *, ratio, compressed):
    """values: pre-QDQ FP16 rows [T, 512|128] (RoPE applied); rows: [R, ROW] uint8."""
    dim, group, size = (512, 16, C_ROW) if compressed else (128, 32, I_ROW)
    if (
        values.dtype != torch.float16
        or values.shape != (positions.numel(), dim)
        or rows.dtype != torch.uint8
        or rows.shape[1] != size
        or not values.is_contiguous()
    ):
        raise ValueError("V4.1 packed KV insertion shapes do not match")
    if values.shape[0]:
        _insert_packed[(values.shape[0], dim // group)](
            values,
            positions,
            valid_tokens,
            rows,
            0,
            RATIO=ratio,
            DIM=dim,
            GROUP=group,
            ROW=size,
            COMPRESSED=compressed,
        )


@triton.jit(do_not_specialize=["T", "N_OUT"])
def _index_scores(
    q,
    w,
    keys,
    positions,
    out,
    T,
    N_OUT,
    RATIO: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
):
    """out[t, n] = sum_h fp16(fp16(relu(fp16(q[t,h].k[n]))) * w[t,h]) for live keys."""
    tb, nb = tl.program_id(0), tl.program_id(1)
    end = (tl.load(positions + T - 1).to(tl.int64) + 1) // RATIO
    n0 = nb * BN
    if n0 < end:
        r = tl.arange(0, BT * H)
        tt = tb * BT + r // H
        hh = r % H
        d = tl.arange(0, 128)
        qv = tl.load(
            q + (tt * H + hh)[:, None] * 128 + d[None, :], (tt < T)[:, None], 0.0
        )
        n = n0 + tl.arange(0, BN)
        live = n < end
        base = keys + n.to(tl.int64)[:, None] * 68
        byte = tl.load(base + (d // 2)[None, :], live[:, None], 0).to(tl.int32)
        code = tl.where((d % 2 == 0)[None, :], byte & 15, (byte >> 4) & 15)
        exponent = tl.load(base + 64 + (d // 32)[None, :], live[:, None], 0).to(
            tl.int32
        )
        m = code & 7
        mag = tl.where(m == 1, 0.5, 0.0)
        mag = tl.where(m == 2, 1.0, mag)
        mag = tl.where(m == 3, 1.5, mag)
        mag = tl.where(m == 4, 2.0, mag)
        mag = tl.where(m == 5, 3.0, mag)
        mag = tl.where(m == 6, 4.0, mag)
        mag = tl.where(m == 7, 6.0, mag)
        sign = ((code >> 3) & 1).to(tl.uint32) << 31
        mag = (mag.to(tl.uint32, bitcast=True) | sign).to(tl.float32, bitcast=True)
        kv = (mag * (exponent << 23).to(tl.float32, bitcast=True)).to(tl.float16)
        score = tl.dot(qv, tl.trans(kv)).to(tl.float16).to(tl.float32)
        score = tl.maximum(score, 0.0)
        wv = tl.load(w + tt * H + hh, tt < T, 0.0).to(tl.float32)
        score = (score * wv[:, None]).to(tl.float16).to(tl.float32)
        total = tl.sum(tl.reshape(score, (BT, H, BN)), 1)
        t = tb * BT + tl.arange(0, BT)
        tl.store(
            out + t.to(tl.int64)[:, None] * N_OUT + n[None, :],
            total.to(tl.float16),
            (t < T)[:, None] & live[None, :],
        )


def index_scores(q, w, keys, positions, *, ratio, out=None):
    """Per-rank head sums of the V4.1 indexer over packed keys (no all-reduce)."""
    tokens, heads = q.shape[0], q.shape[1]
    if (
        q.dtype != torch.float16
        or q.shape[2] != 128
        or w.shape != (tokens, heads)
        or w.dtype != torch.float16
        or keys.dtype != torch.uint8
        or keys.shape[1] != I_ROW
    ):
        raise ValueError("V4.1 packed indexer requires FP16 queries and packed keys")
    if out is None:
        out = torch.empty(tokens, keys.shape[0], dtype=torch.float16, device=q.device)
    bt = 128 // heads
    _index_scores[(triton.cdiv(tokens, bt), triton.cdiv(keys.shape[0], 64))](
        q.contiguous(),
        w.contiguous(),
        keys,
        positions,
        out,
        tokens,
        out.shape[1],
        RATIO=ratio,
        H=heads,
        BT=bt,
        BN=64,
        num_warps=8,
    )
    return out


@triton.jit
def _e2m1_bits(m):
    """FP32 bits of the E2M1 magnitude index m (0..7): 0,.5,1,1.5,2,3,4,6."""
    e = m >> 1
    normal = ((e + 126) << 23) | ((m & 1) << 22)
    return tl.where(m >= 2, normal, tl.where(m == 1, 126 << 23, 0)).to(tl.uint32)


@triton.jit
def _unpack_i_block(base, live, d):
    """[BN, 128] FP16 index keys of packed rows at `base` (power-of-two scales)."""
    byte = tl.load(base + (d // 2)[None, :], live[:, None], 0).to(tl.int32)
    code = tl.where((d % 2 == 0)[None, :], byte & 15, (byte >> 4) & 15)
    exponent = tl.load(base + 64 + (d // 32)[None, :], live[:, None], 127).to(tl.int32)
    bits = _e2m1_bits(code & 7)
    # Exact power-of-two scaling of a nonzero E2M1 value is an exponent add.
    bits = tl.where(bits != 0, bits + ((exponent - 127) << 23).to(tl.uint32), bits)
    bits = bits | (((code >> 3) & 1).to(tl.uint32) << 31)
    return bits.to(tl.float32, bitcast=True).to(tl.float16)


@triton.jit(do_not_specialize=["ROWS"])
def _unpack_index_keys(
    keys, positions, out, ROWS, TOKENS, RATIO: tl.constexpr, BN: tl.constexpr
):
    end = (tl.load(positions + TOKENS - 1).to(tl.int64) + 1) // RATIO
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    live = (n < end) & (n < ROWS)
    d = tl.arange(0, 128)
    value = _unpack_i_block(keys + n.to(tl.int64)[:, None] * 68, live, d)
    tl.store(
        out + n.to(tl.int64)[:, None] * 128 + d[None, :], value, (n < ROWS)[:, None]
    )


@triton.jit
def _decode_block(
    qv0,
    qv1,
    qv2,
    qv3,
    wv,
    keys,
    out,
    n0,
    end,
    T,
    N_OUT,
    R: tl.constexpr,
    H: tl.constexpr,
    BN: tl.constexpr,
):
    n = n0 + tl.arange(0, BN)
    live = n < end
    base = keys + n.to(tl.int64) * 68
    b16 = tl.arange(0, 16)
    tt = tl.arange(0, R) // H
    acc = tl.zeros((R, BN), tl.float32)
    for g in tl.static_range(4):
        qv = qv0 if g == 0 else (qv1 if g == 1 else (qv2 if g == 2 else qv3))
        byte = tl.load(base[:, None] + g * 16 + b16[None, :], live[:, None], 0).to(
            tl.int32
        )
        lo = _e2m1_bits(byte & 7) | (((byte >> 3) & 1).to(tl.uint32) << 31)
        hi = _e2m1_bits((byte >> 4) & 7) | (((byte >> 7) & 1).to(tl.uint32) << 31)
        pair = tl.join(lo.to(tl.float32, bitcast=True), hi.to(tl.float32, bitcast=True))
        kv = tl.reshape(pair, (BN, 32)).to(tl.float16)  # unscaled E2M1, exact
        part = tl.dot(qv, tl.trans(kv))
        exponent = tl.load(base + 64 + g, live, 127).to(tl.int32)
        acc += part * (exponent << 23).to(tl.float32, bitcast=True)[None, :]
    score = tl.maximum(acc.to(tl.float16).to(tl.float32), 0.0)
    score = (score * wv[:, None]).to(tl.float16).to(tl.float32)
    for t in tl.static_range(R // H):
        total = tl.sum(tl.where((tt == t)[:, None], score, 0.0), 0)
        tl.store(out + t * N_OUT + n, total.to(tl.float16), live & (t < T))


@triton.jit(do_not_specialize=["T", "N_OUT"])
def _index_scores_decode(
    q,
    w,
    keys,
    positions,
    out,
    T,
    N_OUT,
    RATIO: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    BN: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    """Decode: q/w padded to R rows; each program scores BLOCKS live key blocks."""
    end = (tl.load(positions + T - 1).to(tl.int64) + 1) // RATIO
    first = tl.program_id(0) * BLOCKS * BN
    # 0/1-trip loops instead of branches around the dots (sm70 codegen).
    for _ in range(0, tl.where(first < end, 1, 0)):
        r = tl.arange(0, R)
        d = tl.arange(0, 32)
        qv0 = tl.load(q + r[:, None] * 128 + d[None, :])
        qv1 = tl.load(q + r[:, None] * 128 + 32 + d[None, :])
        qv2 = tl.load(q + r[:, None] * 128 + 64 + d[None, :])
        qv3 = tl.load(q + r[:, None] * 128 + 96 + d[None, :])
        wv = tl.load(w + r).to(tl.float32)
        stop = tl.minimum(first + BLOCKS * BN, end)
        for n0 in range(first, stop, BN):
            _decode_block(
                qv0, qv1, qv2, qv3, wv, keys, out, n0, end, T, N_OUT, R=R, H=H, BN=BN
            )


def unpack_index_keys(keys, positions, out=None):
    """FP16 [rows, 128] keys up to the forward's last complete group (rest unset)."""
    if out is None:
        out = torch.empty(keys.shape[0], 128, dtype=torch.float16, device=keys.device)
    rows = out.shape[0]
    _unpack_index_keys[(triton.cdiv(rows, 64),)](
        keys, positions, out, rows, positions.numel(), RATIO=1, BN=64
    )
    return out


_DECODE_BLOCKS = int(envs.VLLM_V41_INDEX_DECODE_BLOCKS)


_CUDA_INDEX = {}


def _cuda_index_scores():
    if "ok" not in _CUDA_INDEX:
        try:
            from vllm.model_executor.layers.quantization.gguf_v41_moe import (
                _expert_major_ready,
            )

            _CUDA_INDEX["ok"] = bool(
                _expert_major_ready() and hasattr(torch.ops.v41k, "index_scores_decode")
            )
        except Exception:
            _CUDA_INDEX["ok"] = False
    return _CUDA_INDEX["ok"]


def select_packed_positions(
    q, weights, keys, positions, valid_tokens, ratio, all_reduce, *, top_k=512
):
    """select_compressed_positions over packed, position-addressed index keys."""
    from vllm.models.deepseek_v41.index_topk import index_topk

    tokens, heads = q.shape[0], q.shape[1]
    if tokens * heads <= 64:
        # Decode: bounded fused scores over the live keys only (graph-safe).
        scores = torch.empty(
            tokens, keys.shape[0], dtype=torch.float16, device=q.device
        )
        if _cuda_index_scores():
            # Volta: Triton dots fall back to FMA; the CUDA kernel is bounded too.
            torch.ops.v41k.index_scores_decode(
                scores,
                q.reshape(-1, 128).contiguous(),
                weights.reshape(-1).contiguous(),
                keys,
                positions.long(),
                heads,
                ratio,
            )
        else:
            qp = q.new_zeros(64, 128)
            qp[: tokens * heads].copy_(q.reshape(-1, 128))
            wp = weights.new_zeros(64)
            wp[: tokens * heads].copy_(weights.reshape(-1))
            _index_scores_decode[(triton.cdiv(keys.shape[0], 64 * _DECODE_BLOCKS),)](
                qp,
                wp,
                keys,
                positions,
                scores,
                tokens,
                scores.shape[1],
                RATIO=ratio,
                H=heads,
                R=64,
                BN=64,
                BLOCKS=_DECODE_BLOCKS,
                num_warps=4,
            )
        scores = all_reduce(scores)
        result = torch.empty(tokens, top_k, dtype=torch.int32, device=q.device)
        if keys.shape[0] > 16384:
            from vllm.models.deepseek_v41.index_topk import index_topk_parallel

            index_topk_parallel(result, scores, positions, valid_tokens, ratio, top_k)
        else:
            index_topk(result, scores, positions, valid_tokens, ratio, top_k)
        return result
    # Prefill: unpack the live keys once, then the reference cuBLAS path.
    end = int(positions[-1].item() + 1) // ratio
    # At least top_k rows: a fixed result width (no per-length recompiles).
    fp16 = torch.empty(max(end, top_k), 128, dtype=torch.float16, device=q.device)
    _unpack_index_keys[(triton.cdiv(fp16.shape[0], 64),)](
        keys, positions, fp16, fp16.shape[0], positions.numel(), RATIO=ratio, BN=64
    )
    return select_compressed_positions(
        q, weights, fp16, positions, valid_tokens, ratio, all_reduce, top_k=top_k
    )


@triton.jit(do_not_specialize=["HOT_ROWS", "TOKENS"])
def _gather_attention_packed(
    ring,
    values,
    starts,
    selected,
    hot,
    staging,
    positions,
    valid_tokens,
    topk,
    keys,
    unused,
    HOT_ROWS,
    TOKENS,
    RATIO: tl.constexpr,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    RING: tl.constexpr,
    SELECTED: tl.constexpr,
):
    """Window keys from the chunk/ring; compressed keys from packed rows."""
    token, key = tl.program_id(0), tl.program_id(1)
    position = tl.load(positions + token).to(tl.int64)
    valid = tl.load(valid_tokens + token)
    col = tl.arange(0, 512)
    if key < WINDOW:
        at = position - WINDOW + 1 + key
        valid &= at >= 0
        start = tl.load(starts).to(tl.int64)
        inside = at >= start
        old = tl.load(ring + (at % RING) * 512 + col, valid & ~inside, 0)
        new = tl.load(values + (at - start) * 512 + col, valid & inside, 0)
        value = tl.where(inside, new, old)
    else:
        index = tl.load(topk + token * TOPK + key - WINDOW).to(tl.int64)
        valid &= (index >= 0) & (index < (position + 1) // RATIO)
        if SELECTED:
            row = selected + (token * TOPK + key - WINDOW).to(tl.int64) * 288
        else:
            # Unused slots (-1) must still form an in-bounds address: unpack
            # loads unconditionally and the value is masked afterwards.
            safe = tl.where(valid, index, 0)
            newest = (tl.load(starts + TOKENS - 1).to(tl.int64) + 1) // RATIO - 1
            in_hot = tl.where(valid, safe > newest - HOT_ROWS, True)
            row = tl.where(in_hot, hot + (safe % HOT_ROWS) * 288, staging + safe * 288)
        value = tl.where(valid, unpack_c_row(row), 0.0).to(tl.float16)
    tl.store(keys + (token * (WINDOW + TOPK) + key) * 512 + col, value)
    tl.store(unused + token * (WINDOW + TOPK) + key, ~valid)


_ATTN_FUSED = envs.VLLM_V41_ATTN_FUSED == "1"
# Draft attention (width 128 + n) on v41sk.attn_dec instead of bmm + softmax
# + bmm. Default off.
_ATTN_DEC = envs.VLLM_V41_ATTN_DEC == "1"
# Target decode attention with the ring gather fused (v41sk.attn_ring_dec).
_ATTN_RING = envs.VLLM_V41_ATTN_RING == "1"
# VLLM_V41_BINDEX_CUDA: batched index scores by the v41sk WMMA kernel.
_BINDEX_CUDA = envs.VLLM_V41_BINDEX_CUDA == "1"
_ATTN_RING_POST = envs.VLLM_V41_ATTN_RING_POST == "1"
# The same kernel for prefill chunks (default off).
_ATTN_RING_PREFILL = envs.VLLM_V41_ATTN_RING_PREFILL == "1"


def nchunk_ok(width):
    return 0 < width <= 64 * 32


@triton.jit
def _sink_softmax(
    scores,
    unused,
    sink,
    probs,
    scale,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # FP32 scale + unused mask + sink softmax, FP16 probabilities (the PV
    # boundary of attend_gathered_keys), in one pass per (token, head).
    row = tl.program_id(0)
    token = row // HEADS
    col = tl.arange(0, BLOCK)
    inside = col < WIDTH
    s = tl.load(scores + row * WIDTH + col, mask=inside, other=0.0) * scale
    dead = tl.load(unused + token * WIDTH + col, mask=inside, other=1) != 0
    s = tl.where(dead | ~inside, float("-inf"), s)
    z = tl.load(sink + row % HEADS)
    m = tl.maximum(tl.max(s, 0), z)
    e = tl.exp(s - m)
    p = e / (tl.sum(e, 0) + tl.exp(z - m))
    tl.store(probs + row * WIDTH + col, p.to(tl.float16), mask=inside)


def _attend_fused(q, keys, unused, scale, attn_sink, output, scores):
    """attend_gathered_keys for gathers that already zero unused keys."""
    n, heads, width = scores.shape
    torch.bmm(q, keys.transpose(1, 2), out=scores, out_dtype=torch.float32)
    probs = torch.empty(n, heads, width, dtype=torch.float16, device=q.device)
    _sink_softmax[(n * heads,)](
        scores,
        unused,
        attn_sink,
        probs,
        scale,
        HEADS=heads,
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
        num_warps=4,
    )
    torch.bmm(probs, keys, out=output)


def attend_ring_post(
    q,
    positions,
    valid_tokens,
    attn_sink,
    rope,
    *,
    window_block,
    window_ring=None,
    window_values=None,
    compressed_cache=None,
    compressed_table=None,
    compressed_block=16,
    ratio=1,
    topk=None,
    window=128,
    packed=None,
):
    """Decode attention, inverse RoPE, FP8 QDQ and ring insertion in two launches.

    Return None when the existing attention/post paths should handle the call.
    """
    compressed = compressed_cache is not None
    count = topk.shape[1] if compressed else 0
    if not (
        _ATTN_RING_POST
        and _ATTN_RING
        and window_ring is not None
        and packed is None
        and 1 <= q.shape[0] <= 8
        and q.shape[1:] == (16, 512)
        and q.dtype == torch.float16
        and q.is_contiguous()
        and positions.dtype == torch.int64
        and (not compressed or topk.dtype == torch.int32)
        and (compressed or topk is None)
        and nchunk_ok(window + count)
        and rope.dtype == torch.float32
        and rope.ndim == 2
        and rope.shape[1] == 64
        and rope.is_contiguous()
    ):
        return None
    if compressed:
        _validate_cache(compressed_cache, compressed_table, compressed_block, ratio)
    from vllm.models.deepseek_v41.native import load_v41sk

    load_v41sk()
    result = torch.empty_like(q)
    torch.ops.v41sk.attn_ring_dec(
        result,
        q,
        window_ring,
        window_values,
        positions,
        compressed_cache if compressed else window_ring,
        compressed_table if compressed else positions,
        positions,
        valid_tokens,
        topk if compressed else positions,
        attn_sink,
        512**-0.5,
        compressed_block if compressed else 1,
        ratio if compressed else 1,
        window,
        count,
        rope,
    )
    return result.view(q.shape[0], 2, 4096)


def attend_paged_latents(
    q,
    positions,
    valid_tokens,
    window_cache,
    window_table,
    attn_sink,
    *,
    window_block,
    compressed_cache=None,
    compressed_table=None,
    compressed_block=16,
    ratio=1,
    topk=None,
    window=128,
    tokens_per_pass=64,
    window_ring=None,
    window_values=None,
    packed=None,
):
    """Gather each query's window and selected compressed rows, then use HMMA.

    packed: (store, source) - compressed rows come from the packed KV store:
    its selected-row buffer in decode, its hot ring + staging in prefill.
    """
    ring = window_ring is not None
    if not ring:
        _validate_cache(window_cache, window_table, window_block, 1)
    compressed = compressed_cache is not None
    if compressed:
        _validate_cache(compressed_cache, compressed_table, compressed_block, ratio)
    elif topk is not None and packed is None:
        raise ValueError("Compressed positions require their source cache")
    if (
        q.dtype != torch.float16
        or q.ndim != 3
        or q.shape[2] != 512
        or attn_sink.dtype != torch.float32
        or attn_sink.shape != (q.shape[1],)
    ):
        raise ValueError("V4.1 sparse attention requires FP16 queries and FP32 sinks")
    if packed is not None:
        compressed, count = True, topk.shape[1]
    else:
        count = topk.shape[1] if compressed else 0
    width = window + count
    if (
        packed is None
        and _FUSED_SPARSE
        and q.shape[0] >= _FUSED_MIN_TOKENS
        and q.shape[1] == 16
    ):
        return _attend_sparse_fused(
            q,
            positions,
            valid_tokens,
            window_cache,
            window_table,
            attn_sink,
            window_block,
            compressed_cache,
            compressed_table,
            compressed_block,
            ratio,
            topk,
            window,
            window_ring,
            window_values,
        )
    result = torch.empty_like(q)
    if (
        _ATTN_RING_PREFILL
        and ring
        and packed is None
        and q.shape[0] > 8
        and q.shape[1] == 16
        and nchunk_ok(width)
    ):
        # Prefill: one CTA per token walks all its keys (gather + QK + online
        # sink softmax + PV, v41sk.attn_ring_tok); no key tensor, no combine.
        from vllm.models.deepseek_v41.native import load_v41sk

        load_v41sk()
        # Compressed layers read int32 top-k rows; window layers pass positions.
        top32 = (
            (topk if topk.dtype == torch.int32 else topk.to(torch.int32))
            if compressed
            else positions
        )
        for first in range(0, q.shape[0], 4096):
            last = min(first + 4096, q.shape[0])
            torch.ops.v41sk.attn_ring_tok(
                result[first:last],
                q[first:last].contiguous(),
                window_ring,
                window_values,
                positions,
                compressed_cache if compressed else window_ring,
                compressed_table if compressed else positions,
                positions[first:last],
                valid_tokens[first:last],
                top32[first:last].contiguous() if compressed else top32,
                attn_sink,
                512**-0.5,
                compressed_block if compressed else 1,
                ratio if compressed else 1,
                window,
                count,
            )
        return result
    for first in range(0, q.shape[0], tokens_per_pass):
        last = min(first + tokens_per_pass, q.shape[0])
        n, heads = last - first, q.shape[1]
        if (
            _ATTN_RING
            and ring
            and packed is None
            and n <= 8
            and heads == 16
            and (not compressed or topk.dtype == torch.int32)
        ):
            # Gather + QK + sink softmax + PV in one split-key launch pair
            # (v41sk.attn_ring_dec): no [n, width, 512] key tensor.
            from vllm.models.deepseek_v41.native import load_v41sk

            load_v41sk()
            torch.ops.v41sk.attn_ring_dec(
                result[first:last],
                q[first:last].contiguous(),
                window_ring,
                window_values,
                positions,
                compressed_cache if compressed else window_ring,
                compressed_table if compressed else positions,
                positions[first:last],
                valid_tokens[first:last],
                topk[first:last] if compressed else positions,
                attn_sink,
                512**-0.5,
                compressed_block if compressed else 1,
                ratio if compressed else 1,
                window,
                count,
            )
            continue
        keys = torch.empty(n, width, 512, dtype=torch.float16, device=q.device)
        unused = torch.empty(n, width, dtype=torch.bool, device=q.device)
        if packed is not None:
            store, source = packed
            decode = q.shape[0] <= 8
            _gather_attention_packed[(n, width)](
                window_ring,
                window_values,
                positions,
                store.selected[first:last] if decode else store.selected,
                store.hot[source],
                store.staging if store.staging is not None else store.hot[source],
                positions[first:last],
                valid_tokens[first:last],
                topk[first:last],
                keys,
                unused,
                store.hot_rows[source],
                positions.numel(),
                RATIO=ratio,
                WINDOW=window,
                TOPK=count,
                RING=WINDOW_RING,
                SELECTED=decode,
            )
        elif ring:
            _gather_attention_ring[(n, width)](
                window_ring,
                window_values,
                positions,
                compressed_cache if compressed else window_ring,
                compressed_table if compressed else positions,
                positions[first:last],
                valid_tokens[first:last],
                topk[first:last] if compressed else positions,
                keys,
                unused,
                COMPRESSED_STRIDE=compressed_cache.stride(0) if compressed else 0,
                COMPRESSED_BLOCK=compressed_block,
                RATIO=ratio,
                WINDOW=window,
                TOPK=count,
                RING=WINDOW_RING,
            )
        else:
            _gather_attention[(n, width)](
                window_cache,
                window_table,
                compressed_cache if compressed else window_cache,
                compressed_table if compressed else window_table,
                positions[first:last],
                valid_tokens[first:last],
                topk[first:last] if compressed else positions,
                keys,
                unused,
                WINDOW_STRIDE=window_cache.stride(0),
                COMPRESSED_STRIDE=compressed_cache.stride(0) if compressed else 0,
                WINDOW_BLOCK=window_block,
                COMPRESSED_BLOCK=compressed_block,
                RATIO=ratio,
                WINDOW=window,
                TOPK=count,
            )
        scores = torch.empty(n, heads, width, dtype=torch.float32, device=q.device)
        if _ATTN_FUSED:
            _attend_fused(
                q[first:last],
                keys,
                unused,
                512**-0.5,
                attn_sink,
                result[first:last],
                scores,
            )
            continue
        logits = torch.empty(n, heads, width + 1, dtype=torch.float32, device=q.device)
        probabilities = torch.empty_like(logits)
        attend_gathered_keys(
            q[first:last],
            keys,
            unused,
            512**-0.5,
            attn_sink,
            result[first:last],
            scores,
            logits,
            probabilities,
        )
    return result


@triton.jit
def _insert_slots(values, slots, cache, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    slot = tl.load(slots + token).to(tl.int64)
    col = tl.arange(0, 512)
    value = tl.load(values + token * 512 + col)
    tl.store(
        cache + (slot // BLOCK) * STRIDE + (slot % BLOCK) * 512 + col,
        value,
        slot >= 0,
    )


def insert_context_slots(cache, values, slots):
    """The proposer supplies accepted target context's physical draft slots."""
    if (
        cache.ndim != 3
        or cache.shape[-1] != 512
        or cache.stride(1) != 512
        or values.shape != (slots.numel(), 512)
        or values.dtype != torch.float16
        or slots.dtype != torch.int64
        or values.device != cache.device
        or slots.device != cache.device
    ):
        raise ValueError("V4.1 draft context requires FP16 latents and physical slots")
    _insert_slots[(slots.numel(),)](
        values, slots, cache, STRIDE=cache.stride(0), BLOCK=cache.shape[1]
    )


@triton.jit
def _gather_draft(
    cache,
    table,
    query_kv,
    positions,
    valid_queries,
    keys,
    unused,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    QUERIES: tl.constexpr,
):
    token, key = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, 512)
    valid = tl.load(valid_queries + token)
    if key < 128:
        # All five queries see the same accepted context and each other.
        at = tl.load(positions).to(tl.int64) - 128 + key
        valid &= at >= 0
        page = tl.load(table + at // BLOCK, valid, -1).to(tl.int64)
        valid &= page >= 0
        value = tl.load(cache + page * STRIDE + (at % BLOCK) * 512 + col, valid, 0)
    else:
        index = key - 128
        valid &= tl.load(valid_queries + index)
        value = tl.load(query_kv + index * 512 + col, valid, 0)
    tl.store(keys + (token * (128 + QUERIES) + key) * 512 + col, value)
    tl.store(unused + token * (128 + QUERIES) + key, ~valid)


def attend_draft_latents(q, kv, positions, valid, cache, table, sink, block_size):
    """Non-causal N=5 draft attention; query KVs never overwrite context."""
    _validate_cache(cache, table, block_size, 1)
    n, heads, dimension = q.shape
    if (
        not 1 <= n <= 5
        or dimension != 512
        or kv.shape != (n, 512)
        or q.dtype != torch.float16
        or kv.dtype != torch.float16
        or sink.shape != (heads,)
        or sink.dtype != torch.float32
        or valid.shape != (n,)
    ):
        raise ValueError("V4.1 draft attention admits one to five query rows")
    width = 128 + n
    keys = torch.empty(n, width, 512, device=q.device, dtype=torch.float16)
    unused = torch.empty(n, width, device=q.device, dtype=torch.bool)
    _gather_draft[(n, width)](
        cache,
        table,
        kv,
        positions,
        valid,
        keys,
        unused,
        STRIDE=cache.stride(0),
        BLOCK=block_size,
        QUERIES=n,
    )
    scores = torch.empty(n, heads, width, dtype=torch.float32, device=q.device)
    result = torch.empty_like(q)
    if _ATTN_DEC and heads == 16 and q.is_cuda:
        # v41sk.attn_dec zeroes unused rows itself; cuBLAS picks a slow path
        # for these odd widths (128 + n): 32 -> 16 us per draft layer.
        from vllm.models.deepseek_v41.native import load_v41sk

        load_v41sk()
        torch.ops.v41sk.attn_dec(result, q.contiguous(), keys, unused, sink, 512**-0.5)
        return result
    if _ATTN_FUSED:
        keys.masked_fill_(unused[:, :, None], 0)
        _attend_fused(q, keys, unused, 512**-0.5, sink, result, scores)
        return result
    logits = torch.empty(n, heads, width + 1, dtype=torch.float32, device=q.device)
    attend_gathered_keys(
        q,
        keys,
        unused,
        512**-0.5,
        sink,
        result,
        scores,
        logits,
        torch.empty_like(logits),
    )
    return result


# ---- VLLM_V41_BATCH: several requests per forward ---------------------------
# Token t belongs to request row rows[t] (its block-table row); its request's
# first token in this forward is bases[t]. Block tables are [rows, blocks].


def _validate_batch_cache(cache, block_table, block_size, ratio):
    if (
        not cache.is_cuda
        or cache.dtype != torch.float16
        or cache.ndim != 3
        or cache.shape[1] != block_size // ratio
        or block_size % ratio
        or cache.shape[2] not in (128, 512)
        or cache.stride(1) != cache.shape[2]
        or cache.stride(2) != 1
        or block_table.ndim != 2
        or block_table.stride(1) != 1
        or block_table.dtype != torch.int32
        or block_table.device != cache.device
    ):
        raise ValueError("V4.1 batched caches require int32 block tables, FP16 rows")


@triton.jit
def _binsert(
    values,
    positions,
    valid_tokens,
    rows,
    block_table,
    cache,
    TABLE_STRIDE,
    STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    position = tl.load(positions + token).to(tl.int64)
    valid = tl.load(valid_tokens + token) & ((position + 1) % RATIO == 0)
    table = block_table + tl.load(rows + token).to(tl.int64) * TABLE_STRIDE
    page = tl.load(table + position // BLOCK_SIZE, valid, -1).to(tl.int64)
    valid &= page >= 0
    col = tl.arange(0, BLOCK)
    data = tl.load(values + token * D + col, valid & (col < D), 0)
    row = (position % BLOCK_SIZE) // RATIO
    tl.store(cache + page * STRIDE + row * D + col, data, valid & (col < D))


def insert_latents_batch(
    cache, values, positions, valid_tokens, block_table, layout, *, block_size, ratio=1
):
    """insert_latents for a batch: each token uses its request's table row."""
    _validate_batch_cache(cache, block_table, block_size, ratio)
    if (
        values.shape != (positions.numel(), cache.shape[-1])
        or values.dtype != torch.float16
        or valid_tokens.shape != positions.shape
    ):
        raise ValueError("V4.1 latent insertion shapes do not match")
    if values.shape[0]:
        _binsert[(values.shape[0],)](
            values,
            positions,
            valid_tokens,
            layout.rows,
            block_table,
            cache,
            block_table.stride(0),
            STRIDE=cache.stride(0),
            BLOCK_SIZE=block_size,
            RATIO=ratio,
            D=values.shape[1],
            BLOCK=triton.next_power_of_2(values.shape[1]),
        )


@triton.jit(do_not_specialize=["TABLE_STRIDE", "N_OUT"])
def _bindex_scores(
    q,
    w,
    cache,
    block_table,
    positions,
    query_start_loc,
    out,
    TABLE_STRIDE,
    N_OUT,
    STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
):
    """Request row r, key tile nb: out[t, n] = sum_h fp16(fp16(relu(fp16(q.k)))
    * w) for every token t of the row, keys read once from the paged cache."""
    r, nb = tl.program_id(0), tl.program_id(1)
    first = tl.load(query_start_loc + r)
    count = tl.load(query_start_loc + r + 1) - first
    if count > 0:
        end = (tl.load(positions + first + count - 1).to(tl.int64) + 1) // RATIO
        n0 = nb * BN
        if n0 < end:
            n = n0 + tl.arange(0, BN)
            live = n < end
            table = block_table + r.to(tl.int64) * TABLE_STRIDE
            page = tl.load(table + n // (BLOCK_SIZE // RATIO), live, -1).to(tl.int64)
            live &= page >= 0
            d = tl.arange(0, 128)
            kv = tl.load(
                cache
                + (page * STRIDE + (n % (BLOCK_SIZE // RATIO)) * 128)[:, None]
                + d[None, :],
                live[:, None],
                0.0,
            )
            for c in range(0, count, BT):
                lane = tl.arange(0, BT * H)
                tt = first + c + lane // H
                inside = (c + lane // H) < count
                hh = lane % H
                qv = tl.load(
                    q + (tt.to(tl.int64) * H + hh)[:, None] * 128 + d[None, :],
                    inside[:, None],
                    0.0,
                )
                score = tl.dot(qv, tl.trans(kv)).to(tl.float16).to(tl.float32)
                score = tl.maximum(score, 0.0)
                wv = tl.load(w + tt * H + hh, inside, 0.0).to(tl.float32)
                score = (score * wv[:, None]).to(tl.float16).to(tl.float32)
                total = tl.sum(tl.reshape(score, (BT, H, BN)), 1)
                t = first + c + tl.arange(0, BT)
                tl.store(
                    out + t.to(tl.int64)[:, None] * N_OUT + n[None, :],
                    total.to(tl.float16),
                    ((c + tl.arange(0, BT)) < count)[:, None] & (n < end)[None, :],
                )


def select_compressed_positions_batch(
    q,
    weights,
    cache,
    block_table,
    positions,
    valid_tokens,
    layout,
    ratio,
    all_reduce,
    *,
    block_size,
    max_context=16384,
    top_k=512,
    all_gather=None,
    tp_rank=0,
    tp_size=1,
):
    """select_compressed_positions over paged index keys, several requests."""
    _validate_batch_cache(cache, block_table, block_size, ratio)
    tokens, heads = q.shape[0], q.shape[1]
    if (
        q.dtype != torch.float16
        or q.shape[2] != 128
        or weights.shape != (tokens, heads)
        or weights.dtype != torch.float16
        or cache.shape[-1] != 128
        or 128 % heads
    ):
        raise ValueError("V4.1 indexer requires FP16 queries, keys and weights")
    columns = max_context // ratio
    width = min(top_k, columns)
    starts = layout.query_start_loc_cpu[: layout.num_reqs + 1].tolist()
    if any(b - a > 8 for a, b in zip(starts, starts[1:])):
        # A prefill chunk in the step: every request goes through the
        # single-request gather + cuBLAS scores, row by row (decode-only
        # steps, and so every graph, take the batched kernel below).
        result = torch.full((tokens, width), -1, device=q.device, dtype=torch.int32)
        last_positions = positions[
            torch.tensor(starts[1:], device=positions.device).clamp(min=1) - 1
        ].tolist()
        if (
            _INDEX_SHARD
            and all_gather is not None
            and tp_size > 1
            and heads * tp_size <= 32
            and width == top_k
        ):
            _sharded_index_prefill(
                result,
                q,
                weights,
                cache,
                block_table,
                positions,
                valid_tokens,
                starts,
                last_positions,
                ratio,
                top_k,
                all_gather,
                tp_rank,
                tp_size,
            )
            return result
        for r, (a, b) in enumerate(zip(starts, starts[1:])):
            if b <= a:
                continue
            # Keys up to this request's live length (not max_context): long
            # contexts gather and score only what exists.
            live = -(-(int(last_positions[r]) + 1) // ratio) * ratio
            keys = gather_index_keys(
                cache,
                block_table[r : r + 1],
                positions[a:b],
                valid_tokens[a:b],
                block_size=block_size,
                ratio=ratio,
                max_context=min(max_context, max(live, top_k * ratio)),
            )
            result[a:b].copy_(
                select_compressed_positions(
                    q[a:b],
                    weights[a:b],
                    keys,
                    positions[a:b],
                    valid_tokens[a:b],
                    ratio,
                    all_reduce,
                    top_k=top_k,
                )
            )
        return result
    if (
        _INDEX_SHARD
        and all_gather is not None
        and tp_size > 1
        and heads * tp_size <= 32
        and width == top_k
    ):
        return _sharded_index_topk(
            q,
            weights,
            cache,
            block_table,
            positions,
            valid_tokens,
            layout,
            ratio,
            columns,
            top_k,
            all_gather,
            tp_rank,
            tp_size,
        )
    head = torch.empty(tokens, columns, dtype=torch.float16, device=q.device)
    if _BINDEX_CUDA and 8 * heads <= 64:
        # VLLM_V41_BINDEX_CUDA: the WMMA kernel (v41sk.cu), same rounding.
        from vllm.models.deepseek_v41.native import load_v41sk

        load_v41sk()
        torch.ops.v41sk.index_scores(
            head,
            q.contiguous(),
            weights.contiguous(),
            cache,
            block_table,
            positions if positions.dtype == torch.int64 else positions.long(),
            layout.query_start_loc,
            layout.num_reqs,
            ratio,
            # Key-tile CTAs per request: 32 cover 16K; longer contexts spread
            # over the 80 SMs (a CTA past its request's length exits at once).
            min(triton.cdiv(columns, 32), max(32, 320 // max(1, layout.num_reqs))),
        )
    else:
        _bindex_scores_launch(
            head,
            q,
            weights,
            cache,
            block_table,
            positions,
            layout,
            columns,
            ratio,
            heads,
            block_size,
        )
    result = torch.empty(tokens, width, device=q.device, dtype=torch.int32)
    column = torch.arange(columns, device=q.device)[None]
    _index_tile_topk(
        head,
        all_reduce,
        result,
        column,
        positions,
        valid_tokens,
        0,
        tokens,
        ratio,
        top_k,
    )
    return result


def _gather_queries(q, weights, all_gather):
    """All heads' index queries and weights from every TP rank, one gather."""
    tokens, heads = q.shape[0], q.shape[1]
    packed = torch.cat([q.reshape(tokens, heads * 128), weights], dim=1)
    world = all_gather(packed.contiguous(), 1).view(tokens, -1, heads * 129)
    q_all = world[:, :, : heads * 128].reshape(tokens, -1, 128).contiguous()
    w_all = world[:, :, heads * 128 :].reshape(tokens, -1).contiguous()
    return q_all, w_all


def _sharded_index_topk(
    q,
    weights,
    cache,
    block_table,
    positions,
    valid_tokens,
    layout,
    ratio,
    columns,
    top_k,
    all_gather,
    rank,
    world,
):
    """VLLM_V41_INDEX_SHARD: the decode indexer sharded by key tiles.

    The head-sharded path has every rank read all (replicated) index keys for
    its heads, then all-reduces a [tokens, columns] score matrix sized by the
    longest admitted context. Here the ranks gather the (small) queries and
    weights, score all heads on their own 32-key tiles (tile t on rank
    t % world: a quarter of the key reads), take their tiles' top-k and
    exchange only those candidates: no score all-reduce, and every cost
    follows the live length. Same scores up to FP32 summation order."""
    q_all, w_all = _gather_queries(q, weights, all_gather)
    return _sharded_index_core(
        q_all,
        w_all,
        cache,
        block_table,
        positions,
        valid_tokens,
        layout,
        ratio,
        columns,
        top_k,
        all_gather,
        rank,
        world,
    )


def _sharded_index_core(
    q_all,
    w_all,
    cache,
    block_table,
    positions,
    valid_tokens,
    layout,
    ratio,
    columns,
    top_k,
    all_gather,
    rank,
    world,
):
    """_sharded_index_topk after the query / weight gather (all heads)."""
    from vllm.models.deepseek_v41.index_topk import index_topk, index_topk_parallel
    from vllm.models.deepseek_v41.native import load_v41sk

    load_v41sk()
    tokens, kt = q_all.shape[0], 32
    q = q_all
    local_columns = triton.cdiv(columns, kt * world) * kt
    head = torch.empty(tokens, local_columns, dtype=torch.float16, device=q.device)
    torch.ops.v41sk.index_scores_shard(
        head,
        q_all,
        w_all,
        cache,
        block_table,
        positions if positions.dtype == torch.int64 else positions.long(),
        layout.query_start_loc,
        layout.num_reqs,
        ratio,
        min(triton.cdiv(local_columns, kt), max(32, 320 // max(1, layout.num_reqs))),
        rank,
        world,
    )
    # This rank's keys below each row's length (whole tiles t % world == rank,
    # plus the partial last tile when it is ours), as the select kernels'
    # positions: p = mine * ratio - 1 makes their (p + 1) // ratio == mine.
    mine = torch.empty(tokens, dtype=torch.int64, device=q.device)
    _shard_lengths[(triton.cdiv(tokens, 128),)](
        positions,
        valid_tokens,
        mine,
        tokens,
        columns,
        RATIO=ratio,
        KT=kt,
        RANK=rank,
        WORLD=world,
        BLOCK=128,
    )
    local = torch.empty(tokens, top_k, dtype=torch.int32, device=q.device)
    select = index_topk_parallel if local_columns > 16384 else index_topk
    select(local, head, mine, valid_tokens, ratio, top_k)
    # One gather: [global indices | FP16 score bits] per rank.
    packed = torch.empty(tokens, 2 * top_k, dtype=torch.int32, device=q.device)
    _shard_pack[(tokens,)](
        local, head, packed, head.stride(0), KT=kt, RANK=rank, WORLD=world, K=top_k
    )
    # (bit-cast to FP32: the one-shot TP gather moves FP16/FP32 rows)
    packed = all_gather(packed.view(torch.float32), 1).view(torch.int32)
    result = torch.empty(tokens, top_k, dtype=torch.int32, device=q.device)
    _shard_merge[(tokens,)](
        packed, result, packed.stride(0), K=top_k, N=world * top_k, num_warps=8
    )
    return result


@triton.jit
def _shard_lengths(
    positions,
    valid,
    out,
    rows,
    columns,
    RATIO: tl.constexpr,
    KT: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = row < rows
    pos = tl.load(positions + row, m, 0).to(tl.int64)
    ok = tl.load(valid + row, m, 0) != 0
    n = tl.where(ok, tl.minimum((pos + 1) // RATIO, columns), 0)
    full, rest = n // KT, n % KT
    mine = (full - RANK + WORLD - 1) // WORLD * KT + tl.where(
        full % WORLD == RANK, rest, 0
    )
    tl.store(out + row, mine * RATIO - 1, m)


@triton.jit
def _shard_pack(
    local,
    head,
    packed,
    stride,
    KT: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    K: tl.constexpr,
):
    """This rank's top-k (local tile columns) as [global index | score bits],
    -1 / -inf past the row's candidates."""
    row = tl.program_id(0)
    lanes = tl.arange(0, K)
    at = tl.load(local + row * K + lanes)
    ok = at >= 0
    at = tl.maximum(at, 0)
    score = tl.load(head + row.to(tl.int64) * stride + at, ok, float("-inf"))
    bits = score.to(tl.int16, bitcast=True).to(tl.int32)
    index = tl.where(ok, ((at // KT) * WORLD + RANK) * KT + at % KT, -1)
    tl.store(packed + row * 2 * K + lanes, index)
    tl.store(packed + row * 2 * K + K + lanes, bits)


@triton.jit
def _shard_merge(packed, out, stride, K: tl.constexpr, N: tl.constexpr):
    """Top-K of the WORLD ranks' candidates by score, as ascending positions
    with -1 after them (index_topk's layout). Ties at the threshold keep the
    lowest candidate slots."""
    row = tl.program_id(0)
    j = tl.arange(0, N)
    at = packed + row.to(tl.int64) * stride + (j // K) * 2 * K + j % K
    index = tl.load(at)
    u = tl.load(at + K) & 0xFFFF
    u = tl.where(u >= 0x8000, 0xFFFF - u, u | 0x8000)
    bins = tl.arange(0, 256)
    hist = tl.histogram(u >> 8, 256)
    suffix = N - (tl.cumsum(hist, 0) - hist)
    hi = tl.max(tl.where(suffix >= K, bins, -1), 0)
    want = K - tl.sum(tl.where(bins > hi, hist, 0), 0)
    hist = tl.histogram(u & 255, 256, mask=(u >> 8) == hi)
    total = tl.sum(hist, 0)
    suffix = total - (tl.cumsum(hist, 0) - hist)
    lo = tl.max(tl.where(suffix >= want, bins, -1), 0)
    want = want - tl.sum(tl.where(bins > lo, hist, 0), 0)
    threshold = hi * 256 + lo
    eq = (u == threshold).to(tl.int32)
    take = ((u > threshold) | ((eq != 0) & (tl.cumsum(eq, 0) - eq < want))).to(tl.int32)
    # Exactly K taken: compact them into the row, then sort those K only.
    last = 2147483647
    at = tl.cumsum(take, 0) - take
    dst = out + row * K
    tl.store(dst + at, tl.where(index >= 0, index, last), take != 0)
    tl.debug_barrier()
    lanes = tl.arange(0, K)
    key = tl.sort(tl.load(dst + lanes))
    tl.debug_barrier()
    tl.store(dst + lanes, tl.where(key == last, -1, key))


def _sharded_index_prefill(
    result,
    q,
    weights,
    cache,
    block_table,
    positions,
    valid_tokens,
    starts,
    last_positions,
    ratio,
    top_k,
    all_gather,
    rank,
    world,
):
    """VLLM_V41_INDEX_SHARD for prefill chunks: the decode kernel over groups
    of 6 rows (each group one pseudo-request on the request's block table),
    in row tiles whose [rows, this rank's columns] FP16 scores fit 64 MB.
    No [rows, heads, keys] product and no score all-reduce."""
    from types import SimpleNamespace

    q_all, w_all = _gather_queries(q, weights, all_gather)
    group, kt = 6, 32
    for r, (a, b) in enumerate(zip(starts, starts[1:])):
        if b <= a:
            continue
        columns = max(-(-(int(last_positions[r]) + 1) // ratio), top_k)
        local_columns = triton.cdiv(columns, kt * world) * kt
        tile = max(group, min(1536, (64 << 20) // (2 * local_columns) // group * group))
        for first in range(a, b, tile):
            last = min(first + tile, b)
            groups = triton.cdiv(last - first, group)
            qsl = (
                torch.arange(groups + 1, device=q.device, dtype=torch.int32) * group
            ).clamp_(max=last - first)
            result[first:last] = _sharded_index_core(
                q_all[first:last],
                w_all[first:last],
                cache,
                block_table[r : r + 1].expand(groups, -1),
                positions[first:last],
                valid_tokens[first:last],
                SimpleNamespace(query_start_loc=qsl, num_reqs=groups),
                ratio,
                columns,
                top_k,
                all_gather,
                rank,
                world,
            )


def _bindex_scores_launch(
    head,
    q,
    weights,
    cache,
    block_table,
    positions,
    layout,
    columns,
    ratio,
    heads,
    block_size,
):
    bt = 128 // heads
    _bindex_scores[(layout.num_reqs, triton.cdiv(columns, 64))](
        q.contiguous(),
        weights.contiguous(),
        cache,
        block_table,
        positions,
        layout.query_start_loc,
        head,
        block_table.stride(0),
        columns,
        STRIDE=cache.stride(0),
        BLOCK_SIZE=block_size,
        RATIO=ratio,
        H=heads,
        BT=bt,
        BN=64,
        num_warps=8,
    )


@triton.jit
def _bgather_attention(
    window_cache,
    window_table,
    compressed_cache,
    compressed_table,
    rows,
    positions,
    valid_tokens,
    topk,
    keys,
    unused,
    WINDOW_TABLE_STRIDE,
    COMPRESSED_TABLE_STRIDE,
    WINDOW_STRIDE: tl.constexpr,
    COMPRESSED_STRIDE: tl.constexpr,
    WINDOW_BLOCK: tl.constexpr,
    COMPRESSED_BLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
):
    token, key = tl.program_id(0), tl.program_id(1)
    position = tl.load(positions + token).to(tl.int64)
    valid = tl.load(valid_tokens + token)
    row = tl.load(rows + token).to(tl.int64)
    col = tl.arange(0, 512)
    if key < WINDOW:
        at = position - WINDOW + 1 + key
        valid &= at >= 0
        table = window_table + row * WINDOW_TABLE_STRIDE
        page = tl.load(table + at // WINDOW_BLOCK, valid, -1).to(tl.int64)
        valid &= page >= 0
        value = tl.load(
            window_cache + page * WINDOW_STRIDE + (at % WINDOW_BLOCK) * 512 + col,
            valid,
            0,
        )
    else:
        index = tl.load(topk + token * TOPK + key - WINDOW).to(tl.int64)
        valid &= (index >= 0) & (index < (position + 1) // RATIO)
        table = compressed_table + row * COMPRESSED_TABLE_STRIDE
        page = tl.load(table + index // (COMPRESSED_BLOCK // RATIO), valid, -1).to(
            tl.int64
        )
        valid &= page >= 0
        value = tl.load(
            compressed_cache
            + page * COMPRESSED_STRIDE
            + (index % (COMPRESSED_BLOCK // RATIO)) * 512
            + col,
            valid,
            0,
        )
    tl.store(keys + (token * (WINDOW + TOPK) + key) * 512 + col, value)
    tl.store(unused + token * (WINDOW + TOPK) + key, ~valid)


def _has_prefill_chunk(layout):
    """A request with more than 8 query rows (a prefill chunk) in the step.

    Decode steps (and so every graph) keep the split-K decode kernels: one CTA
    per token (attn_paged_tok) leaves most SMs idle at decode batch sizes."""
    starts = layout.query_start_loc_cpu[: layout.num_reqs + 1]
    return layout.num_reqs > 0 and bool((starts[1:] - starts[:-1]).max() > 8)


def attend_paged_latents_batch(
    q,
    positions,
    valid_tokens,
    window_cache,
    window_table,
    attn_sink,
    layout,
    *,
    window_block,
    compressed_cache=None,
    compressed_table=None,
    compressed_block=16,
    ratio=1,
    topk=None,
    window=128,
    tokens_per_pass=64,
):
    """attend_paged_latents for a batch (paged window cache, no rings)."""
    _validate_batch_cache(window_cache, window_table, window_block, 1)
    compressed = compressed_cache is not None
    if compressed:
        _validate_batch_cache(
            compressed_cache, compressed_table, compressed_block, ratio
        )
    if (
        q.dtype != torch.float16
        or q.ndim != 3
        or q.shape[2] != 512
        or attn_sink.dtype != torch.float32
        or attn_sink.shape != (q.shape[1],)
    ):
        raise ValueError("V4.1 sparse attention requires FP16 queries and FP32 sinks")
    count = topk.shape[1] if compressed else 0
    width = window + count
    result = torch.empty_like(q)
    if (
        _ATTN_RING_PREFILL
        and _has_prefill_chunk(layout)
        and q.shape[1] == 16
        and (not compressed or topk.dtype == torch.int32)
        and nchunk_ok(width)
    ):
        # Prefill (the window rows are already inserted): v41sk.attn_paged_tok,
        # one CTA per token gathering its keys from the paged caches, online
        # sink softmax + PV; no [T, width, 512] key tensor, no combine.
        from vllm.models.deepseek_v41.native import load_v41sk

        load_v41sk()
        torch.ops.v41sk.attn_paged_tok(
            result,
            q.contiguous(),
            window_cache,
            window_table,
            compressed_cache if compressed else window_cache,
            compressed_table if compressed else window_table,
            layout.rows,
            positions if positions.dtype == torch.int64 else positions.long(),
            valid_tokens,
            topk.contiguous() if compressed else window_table,
            attn_sink,
            512**-0.5,
            window_block,
            compressed_block if compressed else 1,
            ratio if compressed else 1,
            window,
            count,
        )
        return result
    for first in range(0, q.shape[0], tokens_per_pass):
        last = min(first + tokens_per_pass, q.shape[0])
        n, heads = last - first, q.shape[1]
        keys = torch.empty(n, width, 512, dtype=torch.float16, device=q.device)
        unused = torch.empty(n, width, dtype=torch.bool, device=q.device)
        _bgather_attention[(n, width)](
            window_cache,
            window_table,
            compressed_cache if compressed else window_cache,
            compressed_table if compressed else window_table,
            layout.rows[first:last],
            positions[first:last],
            valid_tokens[first:last],
            topk[first:last] if compressed else positions,
            keys,
            unused,
            window_table.stride(0),
            (compressed_table if compressed else window_table).stride(0),
            WINDOW_STRIDE=window_cache.stride(0),
            COMPRESSED_STRIDE=compressed_cache.stride(0) if compressed else 0,
            WINDOW_BLOCK=window_block,
            COMPRESSED_BLOCK=compressed_block,
            RATIO=ratio,
            WINDOW=window,
            TOPK=count,
        )
        if _ATTN_DEC and heads == 16:
            from vllm.models.deepseek_v41.native import load_v41sk

            load_v41sk()
            torch.ops.v41sk.attn_dec(
                result[first:last],
                q[first:last].contiguous(),
                keys,
                unused,
                attn_sink,
                512**-0.5,
            )
            continue
        scores = torch.empty(n, heads, width, dtype=torch.float32, device=q.device)
        keys.masked_fill_(unused[:, :, None], 0)
        _attend_fused(
            q[first:last],
            keys,
            unused,
            512**-0.5,
            attn_sink,
            result[first:last],
            scores,
        )
    return result


@triton.jit
def _bgather_draft(
    cache,
    table,
    query_kv,
    rows,
    bases,
    positions,
    valid_queries,
    keys,
    unused,
    TABLE_STRIDE,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    QUERIES: tl.constexpr,
):
    token, key = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, 512)
    valid = tl.load(valid_queries + token)
    base = tl.load(bases + token)
    if key < 128:
        # The request's queries see the same accepted context and each other.
        at = tl.load(positions + base).to(tl.int64) - 128 + key
        valid &= at >= 0
        own = table + tl.load(rows + token).to(tl.int64) * TABLE_STRIDE
        page = tl.load(own + at // BLOCK, valid, -1).to(tl.int64)
        valid &= page >= 0
        value = tl.load(cache + page * STRIDE + (at % BLOCK) * 512 + col, valid, 0)
    else:
        index = base + key - 128
        valid &= tl.load(valid_queries + index)
        value = tl.load(query_kv + index.to(tl.int64) * 512 + col, valid, 0)
    tl.store(keys + (token * (128 + QUERIES) + key) * 512 + col, value)
    tl.store(unused + token * (128 + QUERIES) + key, ~valid)


def attend_draft_latents_batch(
    q, kv, positions, valid, cache, table, sink, block_size, layout, queries
):
    """attend_draft_latents for a batch of requests with `queries` rows each."""
    _validate_batch_cache(cache, table, block_size, 1)
    n, heads, dimension = q.shape
    if (
        n % queries
        or dimension != 512
        or kv.shape != (n, 512)
        or q.dtype != torch.float16
        or kv.dtype != torch.float16
        or sink.shape != (heads,)
        or valid.shape != (n,)
    ):
        raise ValueError("V4.1 batched draft attention: rows are request blocks")
    width = 128 + queries
    keys = torch.empty(n, width, 512, device=q.device, dtype=torch.float16)
    unused = torch.empty(n, width, device=q.device, dtype=torch.bool)
    _bgather_draft[(n, width)](
        cache,
        table,
        kv,
        layout.rows,
        layout.bases,
        positions,
        valid,
        keys,
        unused,
        table.stride(0),
        STRIDE=cache.stride(0),
        BLOCK=block_size,
        QUERIES=queries,
    )
    result = torch.empty_like(q)
    if _ATTN_DEC and heads == 16:
        from vllm.models.deepseek_v41.native import load_v41sk

        load_v41sk()
        for first in range(0, n, 8):
            last = min(first + 8, n)
            torch.ops.v41sk.attn_dec(
                result[first:last],
                q[first:last].contiguous(),
                keys[first:last],
                unused[first:last],
                sink,
                512**-0.5,
            )
        return result
    scores = torch.empty(n, heads, width, dtype=torch.float32, device=q.device)
    keys.masked_fill_(unused[:, :, None], 0)
    _attend_fused(q, keys, unused, 512**-0.5, sink, result, scores)
    return result
