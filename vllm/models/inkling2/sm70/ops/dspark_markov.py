# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Shard-local Markov head of the Inkling2 DSpark draft and shard-local argmax
of the target head (SM70).

The unfused head runs, per draft step, about fifteen small kernels, a cuBLAS
GEMV over the folded LM-head shard and a TP all-gather of the whole softcapped
vocabulary (NCCL), then samples from the gathered logits. Here every rank
keeps the vocabulary pass on its own shard and the ranks exchange one pair per
request and step:

- `_markov_rnn_kernel`: the recurrence step (rnn_proj GEMV + gate);
- `_markov_vocab_kernel`: for the rank's shard, logits = softcap(base +
  fold_bias + latent @ fold_W) / temperature, written to the rank's dense
  shard buffer, and the token-keyed Gumbel-max (argmax at temperature 0) of
  every chunk of rows: the same logits and the same noise as the unfused path
  (`gumbel_sample` keys the noise by seed, position and token id, so the
  maximum over shards of the shard maxima IS the unfused sample);
- `_pick_pack_kernel`: the shard's best (value, token) as eight exact fp16
  integers (64 value bits + 18 id bits in 11-bit words) in this rank's slice
  of a zero `(world, rows, 8)` stack. One all-reduce of the stacks is a gather
  (x + 0 == x in any summation order) and stays on the fork's small-message
  push all-reduce inside CUDA graphs;
- the next step (or `_markov_last_kernel`) takes the best pair over the ranks.

Only the steps the scheduler will verify run. The dense draft logits that
probabilistic rejection needs are all-gathered once after the loop, and only
for batches with a request at temperature > 0 (`dense_scatter`).

`shard_argmax` gives the target's greedy tokens the same way (softcap, chunk
argmax, pick, one small all-reduce) instead of all-gathering the vocabulary.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax

# fp16 words of one (value, token) pair: 5 x 11 + 9 bits of the fp64 value, 11 + 7 of the id.
WORDS = 8
# Shard rows per program of the vocabulary pass / of the target argmax.
VOCAB_ROWS = 32
ARGMAX_CHUNK = 1024


@triton.jit
def _winner(G, row, num_rows, WORLD: tl.constexpr, WP: tl.constexpr):
    """Token of the best pair over the ranks (ties: lowest rank, i.e. lowest
    token id) in a gathered `(WORLD, num_rows, 8)` stack."""
    r = tl.arange(0, WP)
    m = r < WORLD
    p = G + (r * num_rows + row) * 8
    bits = tl.load(p, mask=m, other=0.0).to(tl.uint64)
    bits |= tl.load(p + 1, mask=m, other=0.0).to(tl.uint64) << 11
    bits |= tl.load(p + 2, mask=m, other=0.0).to(tl.uint64) << 22
    bits |= tl.load(p + 3, mask=m, other=0.0).to(tl.uint64) << 33
    bits |= tl.load(p + 4, mask=m, other=0.0).to(tl.uint64) << 44
    bits |= tl.load(p + 5, mask=m, other=0.0).to(tl.uint64) << 55
    value = tl.where(m, bits.to(tl.float64, bitcast=True), float("-inf"))
    token = tl.load(p + 6, mask=m, other=0.0).to(tl.int64)
    token |= tl.load(p + 7, mask=m, other=0.0).to(tl.int64) << 11
    _, best = tl.max(value, axis=0, return_indices=True)
    return tl.sum(tl.where(r == best, token, 0), axis=0)


@triton.jit
def _pick_pack_kernel(
    PART_V,
    PART_I,
    OUT,
    n_parts,
    num_rows,
    rank,
    NP: tl.constexpr,
    WORLD: tl.constexpr,
):
    """Best of a row's chunk maxima (ties: lowest chunk), packed into this
    rank's slice of the `(WORLD, num_rows, 8)` stack; zeros in the others."""
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, NP)
    m = c < n_parts
    v = tl.load(PART_V + row * n_parts + c, mask=m, other=float("-inf"))
    ids = tl.load(PART_I + row * n_parts + c, mask=m, other=0).to(tl.int64)
    best, j = tl.max(v, axis=0, return_indices=True)
    token = tl.sum(tl.where(c == j, ids, 0), axis=0).to(tl.uint64)
    bits = best.to(tl.float64).to(tl.uint64, bitcast=True)
    for r in tl.static_range(WORLD):
        dst = OUT + (r * num_rows + row) * 8
        mine = r == rank
        tl.store(dst, tl.where(mine, (bits & 0x7FF).to(tl.float16), 0.0))
        tl.store(dst + 1, tl.where(mine, ((bits >> 11) & 0x7FF).to(tl.float16), 0.0))
        tl.store(dst + 2, tl.where(mine, ((bits >> 22) & 0x7FF).to(tl.float16), 0.0))
        tl.store(dst + 3, tl.where(mine, ((bits >> 33) & 0x7FF).to(tl.float16), 0.0))
        tl.store(dst + 4, tl.where(mine, ((bits >> 44) & 0x7FF).to(tl.float16), 0.0))
        tl.store(dst + 5, tl.where(mine, (bits >> 55).to(tl.float16), 0.0))
        tl.store(dst + 6, tl.where(mine, (token & 0x7FF).to(tl.float16), 0.0))
        tl.store(dst + 7, tl.where(mine, (token >> 11).to(tl.float16), 0.0))


def pick_pack(
    part_v: torch.Tensor, part_i: torch.Tensor, rank: int, world: int
) -> torch.Tensor:
    """`(world, rows, 8)` fp16 stack holding, in slice `rank`, the best
    (value, id) of each row of the `(rows, parts)` chunk maxima."""
    rows, parts = part_v.shape
    assert (
        part_v.is_contiguous()
        and part_i.is_contiguous()
        and part_i.dtype == torch.int32
    )
    out = torch.empty((world, rows, WORDS), dtype=torch.float16, device=part_v.device)
    if rows:
        _pick_pack_kernel[(rows,)](
            part_v,
            part_i,
            out,
            parts,
            rows,
            rank,
            NP=triton.next_power_of_2(parts),
            WORLD=world,
            num_warps=8 if parts > 512 else 4,
        )
    return out


@triton.jit
def _unpack_kernel(G, OUT, num_rows, WORLD: tl.constexpr, WP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tl.store(OUT + row, _winner(G, row, num_rows, WORLD, WP))


def unpack_tokens(gathered: torch.Tensor) -> torch.Tensor:
    """int64 token of the best pair of every row of a gathered stack."""
    world, rows, _ = gathered.shape
    assert gathered.is_contiguous() and gathered.dtype == torch.float16
    out = torch.empty(rows, dtype=torch.int64, device=gathered.device)
    if rows:
        _unpack_kernel[(rows,)](
            gathered,
            out,
            rows,
            WORLD=world,
            WP=triton.next_power_of_2(world),
            num_warps=1,
        )
    return out


@triton.jit
def _markov_rnn_kernel(
    STATE,
    LATENT,
    CTX,
    W,
    W1,
    PREV,
    TOKENS,
    tok_stride,
    ANCHORS,
    anchor_stride,
    step,
    num_reqs,
    RANK: tl.constexpr,
    U: tl.constexpr,
    WORLD: tl.constexpr,
    WP: tl.constexpr,
):
    """`U` units of recurrence step `step` of one request:
    raw = context + W[:, :r] @ state + W[:, r:] @ w1[previous token];
    state' = sig(raw_g) * state + (1 - sig(raw_g)) * tanh(raw_c), latent =
    tanh(raw_o). The previous token is the anchor at step 0, else the best
    pair of the gathered stack `PREV` of step - 1 (program 0 also stores it)."""
    b = tl.program_id(0).to(tl.int64)
    if step == 0:
        prev = tl.load(ANCHORS + b * anchor_stride).to(tl.int64)
    else:
        prev = _winner(PREV, b, num_reqs, WORLD, WP)
        if tl.program_id(1) == 0:
            tl.store(TOKENS + b * tok_stride + (step - 1), prev)
    u = tl.program_id(1) * U + tl.arange(0, U)
    cols = tl.arange(0, RANK)
    xp = tl.load(W1 + prev * RANK + cols).to(tl.float32)
    # `STATE[0]` is zero. The state enters rnn_proj rounded to the weights'
    # dtype (as the unfused path); the gate blends the unrounded state.
    src = STATE + (step * num_reqs + b) * RANK
    xs = tl.load(src + cols).to(W.dtype.element_ty).to(tl.float32)
    state = tl.load(src + u)
    ctx = CTX + (step * num_reqs + b) * (3 * RANK)
    # The three row blocks of rnn_proj: gate, candidate, output.
    wg = W + u[:, None] * (2 * RANK) + cols[None, :]
    wc = W + (RANK + u)[:, None] * (2 * RANK) + cols[None, :]
    wo = W + (2 * RANK + u)[:, None] * (2 * RANK) + cols[None, :]
    raw_g = tl.sum(
        tl.load(wg).to(tl.float32) * xs[None, :]
        + tl.load(wg + RANK).to(tl.float32) * xp[None, :],
        axis=1,
    ) + tl.load(ctx + u)
    raw_c = tl.sum(
        tl.load(wc).to(tl.float32) * xs[None, :]
        + tl.load(wc + RANK).to(tl.float32) * xp[None, :],
        axis=1,
    ) + tl.load(ctx + RANK + u)
    raw_o = tl.sum(
        tl.load(wo).to(tl.float32) * xs[None, :]
        + tl.load(wo + RANK).to(tl.float32) * xp[None, :],
        axis=1,
    ) + tl.load(ctx + 2 * RANK + u)
    gate = 1.0 / (1.0 + tldevice.exp(-raw_g))
    new_state = gate * state + (1.0 - gate) * tldevice.tanh(raw_c)
    tl.store(STATE + ((step + 1) * num_reqs + b) * RANK + u, new_state)
    tl.store(LATENT + b * RANK + u, tldevice.tanh(raw_o).to(LATENT.dtype.element_ty))


@triton.jit
def _markov_vocab_kernel(
    BASE,
    base_stride,
    FOLD_B,
    FW,
    LATENT,
    DENSE,
    dense_stride,
    PART_V,
    PART_I,
    IDX_MAP,
    POS,
    TEMP,
    SEEDS,
    step,
    num_reqs,
    num_valid,
    id_base,
    n_chunks,
    softcap,
    steps,
    S_ALL: tl.constexpr,
    RANK: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_DENSE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """One chunk of the shard for step `step` of every request: the folded
    rows are loaded once, then per request logits = softcap(base + (latent @
    fold_W + fold_bias)) (/ temperature), their dense shard row, and the
    chunk's Gumbel-max / argmax pair."""
    chunk = tl.program_id(0)
    rows = chunk * ROWS + tl.arange(0, ROWS)
    mask = rows < num_valid
    cols = tl.arange(0, RANK)
    w = tl.load(
        FW + rows.to(tl.int64)[:, None] * RANK + cols[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    fold_b = tl.load(FOLD_B + rows, mask=mask, other=0.0).to(tl.float32)
    keys = id_base + rows
    for b in range(num_reqs):
        lat = tl.load(LATENT + b * RANK + cols).to(tl.float32)
        bias = tl.sum(w * lat[None, :], axis=1) + fold_b
        row = (b * steps + step).to(tl.int64)
        base = tl.load(BASE + row * base_stride + rows, mask=mask, other=0.0)
        logits = tldevice.tanh((base.to(tl.float32) + bias) / softcap) * softcap
        sample = b * S_ALL + step
        rs = tl.load(IDX_MAP + sample).to(tl.int64)
        valid = rs >= 0
        temp = tl.load(TEMP + rs, mask=valid, other=0.0).to(tl.float32)
        if temp != 0.0:
            # The proposal handed to rejection sampling has the request
            # temperature applied (the MRV2 contract, as `gumbel_sample`).
            logits = logits / temp
        logits = tl.where(mask, logits, float("-inf"))
        if HAS_DENSE:
            # Kept per request state (not batch row): the verification of
            # the next step reads it, whatever the batch then looks like.
            tl.store(
                DENSE + (rs * S_ALL + step) * dense_stride + rows,
                logits,
                mask=mask & valid,
            )
        seed = tl.load(SEEDS + rs, mask=valid, other=0)
        # Draws are keyed by the position before the sampled token.
        pos = tl.load(POS + sample) - 1
        value, idx = gumbel_noised_argmax(
            logits,
            keys,
            mask,
            seed,
            pos,
            temp,
            IS_DRAFTING=True,
            USE_FP64=USE_FP64,
            APPLY_TEMPERATURE=False,
        )
        tl.store(PART_V + b * n_chunks + chunk, value)
        tl.store(PART_I + b * n_chunks + chunk, id_base + chunk * ROWS + idx)


@triton.jit
def _markov_last_kernel(
    PREV, TOKENS, tok_stride, step, num_reqs, WORLD: tl.constexpr, WP: tl.constexpr
):
    b = tl.program_id(0).to(tl.int64)
    tl.store(TOKENS + b * tok_stride + step, _winner(PREV, b, num_reqs, WORLD, WP))


def markov_sample(
    base: torch.Tensor,
    context: torch.Tensor,
    w_state: torch.Tensor,
    w1: torch.Tensor,
    fold_weight: torch.Tensor,
    fold_bias: torch.Tensor,
    tokens: torch.Tensor,
    anchors: torch.Tensor,
    anchor_stride: int,
    idx_map: torch.Tensor,
    sample_pos: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    dense: torch.Tensor | None,
    num_spec: int,
    num_valid: int,
    id_base: int,
    rank: int,
    world: int,
    softcap: float,
    use_fp64: bool,
    all_reduce,
) -> None:
    """Sample `steps` draft tokens per request into `tokens[request, step]`.

    `base` `(reqs * steps, shard)` fp32: this rank's base logits, rows in
    (request, step) order; `context` `(steps, reqs, 3r)` fp32: rnn_proj bias
    plus its `[hidden, f_rnn]` term; `w_state` `(3r, 2r)`, `w1` `(vocab, r)`;
    `fold_weight` `(shard, r)` / `fold_bias` `(shard,)`: this rank's folded
    head, whose first `num_valid` rows are tokens `id_base + row`;
    `anchors[request * anchor_stride]`: the token before step 0; `idx_map` /
    `sample_pos` `(>= reqs * num_spec,)`: request-state row (-1: inert) and
    position of draft `request * num_spec + step`; `dense` `(states *
    num_spec, shard)` fp32 or None: receives the temperature-applied logits
    of this shard (row = state row * num_spec + step). `all_reduce` sums a
    tensor over the TP ranks (unused when `world == 1`)."""
    steps, num_reqs, width = context.shape
    rank_r = width // 3
    shard = base.shape[1]
    assert base.shape[0] == num_reqs * steps and base.stride(1) == 1
    assert context.is_contiguous() and context.dtype == torch.float32
    assert w_state.shape == (3 * rank_r, 2 * rank_r) and w_state.is_contiguous()
    assert w1.is_contiguous() and fold_weight.is_contiguous()
    assert fold_weight.shape == (shard, rank_r) and fold_bias.shape == (shard,)
    assert rank_r & (rank_r - 1) == 0 and tokens.dtype == torch.int64
    assert steps <= num_spec and id_base + shard < (1 << 18)
    if not num_reqs:
        return
    device = base.device
    n_chunks = triton.cdiv(shard, VOCAB_ROWS)
    state = torch.zeros(
        (steps + 1, num_reqs, rank_r), dtype=torch.float32, device=device
    )
    latent = torch.empty((num_reqs, rank_r), dtype=w_state.dtype, device=device)
    part_v = torch.empty(
        (num_reqs, n_chunks),
        dtype=torch.float64 if use_fp64 else torch.float32,
        device=device,
    )
    part_i = torch.empty((num_reqs, n_chunks), dtype=torch.int32, device=device)
    has_dense = dense is not None
    if has_dense:
        assert dense.shape[1] == shard and dense.stride(1) == 1
        assert dense.dtype == torch.float32 and dense.shape[0] % num_spec == 0
    wp = triton.next_power_of_2(world)
    units = 8
    prev = latent  # unused at step 0
    for step in range(steps):
        _markov_rnn_kernel[(num_reqs, rank_r // units)](
            state,
            latent,
            context,
            w_state,
            w1,
            prev,
            tokens,
            tokens.stride(0),
            anchors,
            anchor_stride,
            step,
            num_reqs,
            RANK=rank_r,
            U=units,
            WORLD=world,
            WP=wp,
            num_warps=4,
        )
        _markov_vocab_kernel[(n_chunks,)](
            base,
            base.stride(0),
            fold_bias,
            fold_weight,
            latent,
            dense if has_dense else base,
            dense.stride(0) if has_dense else 0,
            part_v,
            part_i,
            idx_map,
            sample_pos,
            temperature,
            seeds,
            step,
            num_reqs,
            num_valid,
            id_base,
            n_chunks,
            softcap,
            steps,
            S_ALL=num_spec,
            RANK=rank_r,
            ROWS=VOCAB_ROWS,
            HAS_DENSE=has_dense,
            USE_FP64=use_fp64,
            num_warps=4,
        )
        prev = pick_pack(part_v, part_i, rank, world)
        if world > 1:
            prev = all_reduce(prev)
    _markov_last_kernel[(num_reqs,)](
        prev,
        tokens,
        tokens.stride(0),
        steps - 1,
        num_reqs,
        WORLD=world,
        WP=wp,
        num_warps=1,
    )


@triton.jit
def _dense_rows_kernel(IDX_MAP, OUT, steps, S_ALL: tl.constexpr):
    """Row of the per-state dense buffer for draft `row = request * steps +
    step` (0 for an inert request)."""
    row = tl.program_id(0)
    step = row % steps
    rs = tl.load(IDX_MAP + (row // steps) * S_ALL + step).to(tl.int64)
    tl.store(OUT + row, tl.maximum(rs, 0) * S_ALL + step)


@triton.jit
def _dense_scatter_kernel(
    SRC,
    src_stride,
    DST,
    dst_s0,
    dst_s1,
    IDX_MAP,
    vocab_size,
    steps,
    S_ALL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    step = row % steps
    rs = tl.load(IDX_MAP + (row // steps) * S_ALL + step).to(tl.int64)
    if rs < 0:
        return
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < vocab_size
    x = tl.load(SRC + row * src_stride + cols, mask=m, other=0.0)
    tl.store(DST + rs * dst_s0 + step * dst_s1 + cols, x, mask=m)


def dense_rows(
    idx_map: torch.Tensor, num_reqs: int, steps: int, num_spec: int
) -> torch.Tensor:
    """int64 `(num_reqs * steps,)`: rows of the per-state dense buffer
    holding draft `request * steps + step` (`idx_map[request * num_spec +
    step]` is the request's state row)."""
    out = torch.empty(num_reqs * steps, dtype=torch.int64, device=idx_map.device)
    if num_reqs * steps:
        _dense_rows_kernel[(num_reqs * steps,)](
            idx_map, out, steps, S_ALL=num_spec, num_warps=1
        )
    return out


def dense_scatter(
    gathered: torch.Tensor,
    draft_logits: torch.Tensor,
    idx_map: torch.Tensor,
    steps: int,
) -> None:
    """Rows `request * steps + step` of the all-gathered `(reqs * steps,
    >= vocab)` dense logits into `draft_logits[state row, step, :vocab]`
    (`idx_map[request * num_spec + step]`: the state row, -1: skip)."""
    rows = gathered.shape[0]
    _, num_spec, vocab = draft_logits.shape
    assert gathered.stride(1) == 1 and gathered.shape[1] >= vocab
    assert gathered.dtype == draft_logits.dtype == torch.float32
    block = 4096
    if rows:
        _dense_scatter_kernel[(rows, triton.cdiv(vocab, block))](
            gathered,
            gathered.stride(0),
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            idx_map,
            vocab,
            steps,
            S_ALL=num_spec,
            BLOCK=block,
            num_warps=4,
        )


@triton.jit
def _argmax_chunk_kernel(
    LOGITS,
    stride,
    PART_V,
    PART_I,
    num_valid,
    id_base,
    n_chunks,
    softcap,
    CHUNK: tl.constexpr,
    HAS_CAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    cols = chunk * CHUNK + tl.arange(0, CHUNK)
    m = cols < num_valid
    x = tl.load(LOGITS + row * stride + cols, mask=m, other=0.0).to(tl.float32)
    if HAS_CAP:
        # The op order of LogitsProcessor: divide, tanh, multiply.
        x = x / softcap
        x = tldevice.tanh(x)
        x = x * softcap
    x = tl.where(m, x, float("-inf"))
    value, idx = tl.max(x, axis=0, return_indices=True)
    tl.store(PART_V + row * n_chunks + chunk, value)
    tl.store(PART_I + row * n_chunks + chunk, id_base + chunk * CHUNK + idx)


def shard_argmax(
    logits: torch.Tensor,
    softcap: float | None,
    num_valid: int,
    id_base: int,
    rank: int,
    world: int,
    all_reduce,
) -> torch.Tensor:
    """Greedy token of every row from the TP-local fp32 `logits` `(rows,
    shard)` (softcapped first, as the logits processor does; ties go to the
    lowest token id, as the dense argmax): int64 `(rows,)`, the same on all
    ranks. `num_valid` rows of the shard are tokens `id_base + row`."""
    rows, shard = logits.shape
    assert logits.stride(1) == 1 and id_base + shard < (1 << 18)
    n_chunks = triton.cdiv(shard, ARGMAX_CHUNK)
    part_v = torch.empty((rows, n_chunks), dtype=torch.float32, device=logits.device)
    part_i = torch.empty((rows, n_chunks), dtype=torch.int32, device=logits.device)
    if rows:
        _argmax_chunk_kernel[(rows, n_chunks)](
            logits,
            logits.stride(0),
            part_v,
            part_i,
            num_valid,
            id_base,
            n_chunks,
            softcap if softcap is not None else 1.0,
            CHUNK=ARGMAX_CHUNK,
            HAS_CAP=softcap is not None,
            num_warps=4,
        )
    packed = pick_pack(part_v, part_i, rank, world)
    if world > 1:
        packed = all_reduce(packed)
    return unpack_tokens(packed)
