# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shard-local rejection sampling for Inkling2 DSpark on TP (SM70).

The dense path all-gathers the target's vocabulary (and the draft's dense
logits) over TP and then runs `rejection_sample` on every rank: block stats
over both vocabularies, the accept loop, a full-vocabulary residual resample.
Everything it needs is separable over the vocabulary shards:

- log-sum-exp of the target and of the draft distribution: per-shard
  (max, sum exp) pairs, combined in rank order;
- p(x) and q(x) of a drafted token x: known to the rank that owns x;
- the token resampled at the first rejected position (or the bonus token):
  the Gumbel-max over the vocabulary of the residual log(max(p - q, 0)) (the
  noise is keyed by seed, position and token id) is the best of the shards'
  own Gumbel-maxima.

So every rank works on its own shard and the ranks meet twice, each time in
one small all-reduce of a zero-padded stack of exact fp16 integers (as
`ops/dspark_markov.py`): first the statistics (`_stats_kernel`,
`_stats_pack_kernel`), then, after every rank has run the identical accept
loop on the gathered numbers (`_decide_kernel`), the resampled pair
(`_resample_kernel`, `_pick_pack_kernel`, `_insert_kernel`).

Same acceptance draws and the same noise keys as `rejection_sample`; only the
order of the fp32 sums inside the log-sum-exps differs. Requests at
temperature 0 take the argmax of the same statistics. Requests that need the
whole vocabulary at once (top-k, top-p, min-p, penalties, logit bias, bad
words, logprobs, grammar) must use the dense path.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, tl_rand32

from .dspark_markov import _winner, pick_pack

CHUNK = 1024
# fp16 words per (rank, row) of the statistics: six fp32 values as 11 + 11 + 10
# bits each (target max, target sum exp, draft max, draft sum exp, p-logit and
# q-logit of the drafted token) and the greedy argmax id as 11 + 7 bits.
STAT_WORDS = 20


@triton.jit
def _softcapped(raw, softcap, temp, HAS_CAP: tl.constexpr):
    x = raw
    if HAS_CAP:
        # The op order of LogitsProcessor: divide, tanh, multiply.
        x = x / softcap
        x = tldevice.tanh(x)
        x = x * softcap
    if temp != 0.0:  # noqa: SIM102  (Triton cannot compile `in` or a merged condition)
        if temp != 1.0:
            x = x / temp
    return x


@triton.jit
def _stats_kernel(
    LOGITS,
    logits_stride,
    DENSE,
    INPUT_IDS,
    LOGITS_INDICES,
    EXP_IDX,
    LOCAL_POS,
    TEMP,
    P_MAX,
    P_SUM,
    P_ARG,
    Q_MAX,
    Q_SUM,
    PX,
    QX,
    num_valid,
    id_base,
    n_chunks,
    num_spec,
    num_rows,
    softcap,
    V_LOCAL: tl.constexpr,
    CHUNK: tl.constexpr,
    HAS_CAP: tl.constexpr,
):
    """One chunk of the shard for one target row: max and sum exp of the
    processed target logits and of the draft logits of that position, the
    chunk argmax (greedy), and the two logits of the drafted token if it
    lives in this chunk. Bonus rows have no draft."""
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    part = row * n_chunks + chunk
    step = tl.load(LOCAL_POS + row)
    rs = tl.load(EXP_IDX + row).to(tl.int64)
    temp = tl.load(TEMP + rs).to(tl.float32)
    # A row has a draft iff the next sampled row is the same request's.
    nxt = tl.minimum(row + 1, num_rows - 1)
    has_draft = (
        (row + 1 < num_rows)
        & (tl.load(LOCAL_POS + nxt) == step + 1)
        & (tl.load(EXP_IDX + nxt).to(tl.int64) == rs)
        & (step < num_spec)
    )
    cols = chunk * CHUNK + tl.arange(0, CHUNK)
    mask = cols < num_valid
    raw = tl.load(LOGITS + row * logits_stride + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    x = tl.where(mask, _softcapped(raw, softcap, temp, HAS_CAP), float("-inf"))
    p_max, p_arg = tl.max(x, axis=0, return_indices=True)
    tl.store(P_MAX + part, p_max)
    tl.store(P_ARG + part, id_base + chunk * CHUNK + p_arg)
    p_sum = tl.sum(
        tl.where(mask & (p_max > float("-inf")), tl.exp(x - p_max), 0.0), axis=0
    )
    tl.store(P_SUM + part, p_sum)
    if has_draft:
        q = tl.load(
            DENSE + (rs * num_spec + step) * V_LOCAL + cols,
            mask=mask,
            other=float("-inf"),
        )
        q_max = tl.max(q, axis=0)
        q_sum = tl.sum(
            tl.where(mask & (q_max > float("-inf")), tl.exp(q - q_max), 0.0), axis=0
        )
        tl.store(Q_MAX + part, q_max)
        tl.store(Q_SUM + part, q_sum)
        # The drafted token of this position is the input of the next row.
        draft = (
            tl.load(INPUT_IDS + tl.load(LOGITS_INDICES + nxt)).to(tl.int64) - id_base
        )
        if (
            (draft >= chunk * CHUNK)
            & (draft < chunk * CHUNK + CHUNK)
            & (draft < num_valid)
        ):
            mine = cols == draft
            tl.store(PX + row, tl.sum(tl.where(mine, x, 0.0), axis=0))
            tl.store(QX + row, tl.sum(tl.where(mine, q, 0.0), axis=0))
    else:
        tl.store(Q_MAX + part, float("-inf"))
        tl.store(Q_SUM + part, 0.0)


@triton.jit
def _put32(dst, value, mine):
    bits = value.to(tl.uint32, bitcast=True)
    tl.store(dst, tl.where(mine, (bits & 0x7FF).to(tl.float16), 0.0))
    tl.store(dst + 1, tl.where(mine, ((bits >> 11) & 0x7FF).to(tl.float16), 0.0))
    tl.store(dst + 2, tl.where(mine, (bits >> 22).to(tl.float16), 0.0))


@triton.jit
def _get32(src, mask):
    bits = tl.load(src, mask=mask, other=0.0).to(tl.uint32)
    bits |= tl.load(src + 1, mask=mask, other=0.0).to(tl.uint32) << 11
    bits |= tl.load(src + 2, mask=mask, other=0.0).to(tl.uint32) << 22
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _stats_pack_kernel(
    P_MAX,
    P_SUM,
    P_ARG,
    Q_MAX,
    Q_SUM,
    PX,
    QX,
    OUT,
    n_chunks,
    num_rows,
    rank,
    NP: tl.constexpr,
    WORLD: tl.constexpr,
):
    """A row's chunk statistics reduced to the shard's (max, sum exp) pairs
    and argmax, packed into this rank's slice of the `(WORLD, rows, 20)`
    stack (zeros in the other slices; PX / QX are zero unless this rank
    owns the drafted token)."""
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, NP)
    m = c < n_chunks
    base = row * n_chunks + c
    p_max_c = tl.load(P_MAX + base, mask=m, other=float("-inf"))
    p_max, j = tl.max(p_max_c, axis=0, return_indices=True)
    arg = tl.sum(tl.where(c == j, tl.load(P_ARG + base, mask=m, other=0), 0), axis=0)
    p_sum = tl.sum(
        tl.where(
            m & (p_max_c > float("-inf")),
            tl.load(P_SUM + base, mask=m, other=0.0) * tl.exp(p_max_c - p_max),
            0.0,
        ),
        axis=0,
    )
    q_max_c = tl.load(Q_MAX + base, mask=m, other=float("-inf"))
    q_max = tl.max(q_max_c, axis=0)
    q_sum = tl.sum(
        tl.where(
            m & (q_max_c > float("-inf")),
            tl.load(Q_SUM + base, mask=m, other=0.0) * tl.exp(q_max_c - q_max),
            0.0,
        ),
        axis=0,
    )
    px = tl.load(PX + row)
    qx = tl.load(QX + row)
    uarg = arg.to(tl.uint32)
    for r in tl.static_range(WORLD):
        dst = OUT + (r * num_rows + row) * 20
        mine = r == rank
        _put32(dst, p_max, mine)
        _put32(dst + 3, p_sum, mine)
        _put32(dst + 6, q_max, mine)
        _put32(dst + 9, q_sum, mine)
        _put32(dst + 12, px, mine)
        _put32(dst + 15, qx, mine)
        tl.store(dst + 18, tl.where(mine, (uarg & 0x7FF).to(tl.float16), 0.0))
        tl.store(dst + 19, tl.where(mine, (uarg >> 11).to(tl.float16), 0.0))


@triton.jit
def _lse(G, row, num_rows, field, WORLD: tl.constexpr, WP: tl.constexpr):
    """log-sum-exp from the ranks' (max, sum exp) pairs at word `field`."""
    r = tl.arange(0, WP)
    m = r < WORLD
    p = G + (r * num_rows + row) * 20 + field
    mx = tl.where(m, _get32(p, m), float("-inf"))
    sm = tl.where(m, _get32(p + 3, m), 0.0)
    top = tl.max(mx, axis=0)
    total = tl.sum(tl.where(mx > float("-inf"), sm * tl.exp(mx - top), 0.0), axis=0)
    return top + tl.log(total)


@triton.jit
def _decide_kernel(
    G,
    SAMPLED,
    sampled_stride,
    NUM_ACCEPTED,
    P_LSE,
    Q_LSE,
    INPUT_IDS,
    LOGITS_INDICES,
    CU_NUM_LOGITS,
    IDX_MAPPING,
    TEMP,
    SEEDS,
    POS,
    num_rows,
    WORLD: tl.constexpr,
    WP: tl.constexpr,
):
    """The accept loop of `_rejection_kernel` on the gathered statistics:
    the same on every rank. Stores the accepted prefix (greedy: the target
    argmax at the first mismatch), the number of accepted drafts and the two
    log-sum-exps of the position that will be resampled."""
    req = tl.program_id(0)
    rs = tl.load(IDX_MAPPING + req).to(tl.int64)
    start = tl.load(CU_NUM_LOGITS + req).to(tl.int64)
    end = tl.load(CU_NUM_LOGITS + req + 1).to(tl.int64)
    seed = tl.load(SEEDS + rs)
    temp = tl.load(TEMP + rs).to(tl.float32)
    r = tl.arange(0, WP)
    m = r < WORLD
    accepted = True
    count = 0
    p_lse = 0.0
    q_lse = 0.0
    for i in range(end - start - 1):
        if accepted:
            row = start + i
            draft = tl.load(INPUT_IDS + tl.load(LOGITS_INDICES + row + 1)).to(tl.int64)
            p = G + (r * num_rows + row) * 20
            if temp == 0.0:
                top = tl.where(m, _get32(p, m), float("-inf"))
                _, best = tl.max(top, axis=0, return_indices=True)
                ids = tl.load(p + 18, mask=m, other=0.0).to(tl.int64)
                ids |= tl.load(p + 19, mask=m, other=0.0).to(tl.int64) << 11
                target = tl.sum(tl.where(r == best, ids, 0), axis=0)
                accepted &= target == draft
                tl.store(
                    SAMPLED + req * sampled_stride + i,
                    tl.where(accepted, draft, target),
                )
            else:
                p_lse = _lse(G, row, num_rows, 0, WORLD, WP)
                q_lse = _lse(G, row, num_rows, 6, WORLD, WP)
                # Exactly one rank sent the two logits of the drafted token.
                px = tl.sum(tl.where(m, _get32(p + 12, m), 0.0), axis=0)
                qx = tl.sum(tl.where(m, _get32(p + 15, m), 0.0), axis=0)
                u = tl_rand32(seed, tl.load(POS + row), includes_zero=False)
                # p(x) > u * q(x)
                accepted &= (px - p_lse) > tl.log(u) + (qx - q_lse)
                tl.store(SAMPLED + req * sampled_stride + i, draft)
            count += accepted.to(tl.int32)
    tl.store(NUM_ACCEPTED + req, count)
    tl.store(P_LSE + req, p_lse)
    tl.store(Q_LSE + req, q_lse)


@triton.jit
def _resample_kernel(
    LOGITS,
    logits_stride,
    DENSE,
    PART_V,
    PART_I,
    NUM_ACCEPTED,
    P_LSE,
    Q_LSE,
    CU_NUM_LOGITS,
    IDX_MAPPING,
    TEMP,
    SEEDS,
    POS,
    num_valid,
    id_base,
    n_chunks,
    num_spec,
    softcap,
    V_LOCAL: tl.constexpr,
    CHUNK: tl.constexpr,
    HAS_CAP: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """One chunk of the shard for a request's resampled position: the
    Gumbel-max (argmax at temperature 0) of the residual log(max(p - q, 0)),
    or of the target itself for the bonus token. A greedy request rejected
    before its bonus token already has its token: -inf here."""
    req = tl.program_id(0)
    chunk = tl.program_id(1)
    part = req * n_chunks + chunk
    rs = tl.load(IDX_MAPPING + req).to(tl.int64)
    step = tl.load(NUM_ACCEPTED + req)
    start = tl.load(CU_NUM_LOGITS + req)
    end = tl.load(CU_NUM_LOGITS + req + 1)
    row = (start + step).to(tl.int64)
    is_bonus = row == end - 1
    temp = tl.load(TEMP + rs).to(tl.float32)
    if (temp == 0.0) & (row != end - 1):
        tl.store(PART_V + part, float("-inf"))
        tl.store(PART_I + part, 0)
        return
    cols = chunk * CHUNK + tl.arange(0, CHUNK)
    mask = cols < num_valid
    raw = tl.load(LOGITS + row * logits_stride + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    x = _softcapped(raw, softcap, temp, HAS_CAP)
    if is_bonus:
        residual = x
    else:
        q = tl.load(
            DENSE + (rs * num_spec + step) * V_LOCAL + cols,
            mask=mask,
            other=float("-inf"),
        )
        p_lp = x - tl.load(P_LSE + req)
        q_lp = q - tl.load(Q_LSE + req)
        # log(max(p - q, 0)) = log p + log(max(1 - q / p, 0))
        ratio = tl.exp(q_lp - p_lp)
        residual = tl.where(ratio < 1.0, p_lp + tldevice.log1p(-ratio), float("-inf"))
    residual = tl.where(mask, residual, float("-inf")).to(tl.float32)
    value, idx = gumbel_noised_argmax(
        residual,
        id_base + cols,
        mask,
        tl.load(SEEDS + rs),
        tl.load(POS + row),
        temp,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    tl.store(PART_V + part, value)
    tl.store(PART_I + part, id_base + chunk * CHUNK + idx)


@triton.jit
def _insert_kernel(
    G,
    SAMPLED,
    sampled_stride,
    NUM_ACCEPTED,
    NUM_SAMPLED,
    CU_NUM_LOGITS,
    IDX_MAPPING,
    TEMP,
    num_reqs,
    WORLD: tl.constexpr,
    WP: tl.constexpr,
):
    req = tl.program_id(0)
    step = tl.load(NUM_ACCEPTED + req)
    rs = tl.load(IDX_MAPPING + req).to(tl.int64)
    temp = tl.load(TEMP + rs).to(tl.float32)
    is_bonus = (
        step == tl.load(CU_NUM_LOGITS + req + 1) - tl.load(CU_NUM_LOGITS + req) - 1
    )
    if (temp != 0.0) | is_bonus:
        token = _winner(G, req.to(tl.int64), num_reqs, WORLD, WP)
        tl.store(SAMPLED + req * sampled_stride + step, token)
    tl.store(NUM_SAMPLED + req, step + 1)


def shard_rejection_sample(
    logits: torch.Tensor,
    dense: torch.Tensor | None,
    input_ids: torch.Tensor,
    logits_indices: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    num_spec: int,
    softcap: float | None,
    num_valid: int,
    id_base: int,
    rank: int,
    world: int,
    use_fp64: bool,
    all_reduce,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`rejection_sample` (probabilistic drafts) without the gathered
    vocabularies: `(sampled (reqs, num_spec + 1) int64, num_sampled (reqs,)
    int32)`, the same on every rank.

    `logits` `(rows, shard)` fp32: this rank's RAW target logits of the
    sampled rows (softcap and temperature are applied here); `dense`
    `(states * num_spec, shard)` fp32: this rank's temperature-applied draft
    logits, row = state row * num_spec + draft step (None when no row has a
    draft); `input_ids` / `logits_indices`: the target batch's tokens and
    the rows sampled from (the drafted token of a row is the input after
    it); `pos` `(rows,)`: their positions; the rest as `rejection_sample`.
    The first `num_valid` shard rows are tokens `id_base + row`."""
    rows, shard = logits.shape
    num_reqs = cu_num_logits.shape[0] - 1
    assert logits.stride(1) == 1 and logits.dtype == torch.float32
    assert id_base + shard < (1 << 18) and pos.is_contiguous()
    if dense is None:
        dense = logits
    else:
        assert dense.shape[1] == shard and dense.is_contiguous()
        assert dense.dtype == torch.float32
    device = logits.device
    n_chunks = triton.cdiv(shard, CHUNK)
    wp = triton.next_power_of_2(world)
    has_cap = softcap is not None
    cap = softcap if has_cap else 1.0

    def f32(*shape):
        return torch.empty(shape, dtype=torch.float32, device=device)

    p_max, p_sum, q_max, q_sum = (f32(rows, n_chunks) for _ in range(4))
    p_arg = torch.empty((rows, n_chunks), dtype=torch.int32, device=device)
    px = torch.zeros(rows, dtype=torch.float32, device=device)
    qx = torch.zeros(rows, dtype=torch.float32, device=device)
    _stats_kernel[(rows, n_chunks)](
        logits,
        logits.stride(0),
        dense,
        input_ids,
        logits_indices,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        p_max,
        p_sum,
        p_arg,
        q_max,
        q_sum,
        px,
        qx,
        num_valid,
        id_base,
        n_chunks,
        num_spec,
        rows,
        cap,
        V_LOCAL=shard,
        CHUNK=CHUNK,
        HAS_CAP=has_cap,
        num_warps=4,
    )
    stats = torch.empty((world, rows, STAT_WORDS), dtype=torch.float16, device=device)
    _stats_pack_kernel[(rows,)](
        p_max,
        p_sum,
        p_arg,
        q_max,
        q_sum,
        px,
        qx,
        stats,
        n_chunks,
        rows,
        rank,
        NP=triton.next_power_of_2(n_chunks),
        WORLD=world,
        num_warps=2,
    )
    if world > 1:
        stats = all_reduce(stats)
    sampled = torch.empty((num_reqs, num_spec + 1), dtype=torch.int64, device=device)
    num_accepted = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=device)
    p_lse, q_lse = f32(num_reqs), f32(num_reqs)
    _decide_kernel[(num_reqs,)](
        stats,
        sampled,
        sampled.stride(0),
        num_accepted,
        p_lse,
        q_lse,
        input_ids,
        logits_indices,
        cu_num_logits,
        idx_mapping,
        temperature,
        seeds,
        pos,
        rows,
        WORLD=world,
        WP=wp,
        num_warps=1,
    )
    part_v = torch.empty(
        (num_reqs, n_chunks),
        dtype=torch.float64 if use_fp64 else torch.float32,
        device=device,
    )
    part_i = torch.empty((num_reqs, n_chunks), dtype=torch.int32, device=device)
    _resample_kernel[(num_reqs, n_chunks)](
        logits,
        logits.stride(0),
        dense,
        part_v,
        part_i,
        num_accepted,
        p_lse,
        q_lse,
        cu_num_logits,
        idx_mapping,
        temperature,
        seeds,
        pos,
        num_valid,
        id_base,
        n_chunks,
        num_spec,
        cap,
        V_LOCAL=shard,
        CHUNK=CHUNK,
        HAS_CAP=has_cap,
        USE_FP64=use_fp64,
        num_warps=4,
    )
    packed = pick_pack(part_v, part_i, rank, world)
    if world > 1:
        packed = all_reduce(packed)
    _insert_kernel[(num_reqs,)](
        packed,
        sampled,
        sampled.stride(0),
        num_accepted,
        num_sampled,
        cu_num_logits,
        idx_mapping,
        temperature,
        num_reqs,
        WORLD=world,
        WP=wp,
        num_warps=1,
    )
    return sampled, num_sampled
