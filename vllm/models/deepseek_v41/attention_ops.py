# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 paged latent caches and bounded sparse attention for local TP4."""

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
    if cache.shape[-1] != 128 or max_context > 16384 or max_context % ratio:
        raise ValueError("V4.1 short-context index keys require 128 dimensions")
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


_INDEX_TILE_BYTES = int(envs.VLLM_V41_INDEX_TILE_MB) << 20


def select_compressed_positions(
    q, weights, keys, positions, valid_tokens, ratio, all_reduce, *, top_k=512
):
    """Keep top-512 even below 16K: only candidate-block prefilter is bypassed.

    Score tiles hold at most 32 query tokens. The official FP16 einsum,
    ReLU/weight and head-sum boundaries are retained before the TP reduction.
    No full [prefill, heads, context] tensor is materialized.
    """
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
    tile = max(32, min(1024, budget // 32 * 32))
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
    for first in range(0, q.shape[0], tokens_per_pass):
        last = min(first + tokens_per_pass, q.shape[0])
        n, heads = last - first, q.shape[1]
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
