# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coupled (keyed) DSpark Markov draft steps without [vocab] tensors.

The sampled-draft twin of ``markov_fused.MarkovGreedy`` for any batch: per
step, one ``_markov_keyed_block`` launch scores every (row, 128-token block)
with the BF16 Markov bias summed in registers exactly as ``_markov_block``
does, divides by the row's temperature x scale and adds the row's keyed
Gumbel draw computed inline (``coupled_draw.gumbel``, the same draws as
``coupled_draw.keyed_gumbel``); one ``_markov_keyed_reduce`` launch picks each
row's token (lowest id on ties) and gathers its next Markov embedding row.
Greedy rows (temperature 0) take the plain argmax of logits + bias.

The drafter only proposes; verification keeps the output exact whatever
these tokens are.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.spec_decode.coupled_draw import gumbel, row_seed

from .markov_fused import ROWS
from .markov_shard import _markov_shard_slot


@triton.jit
def _markov_keyed_block(
    weight,
    embed_rows,
    logits,
    l_stride,
    temperature,
    scale,
    keys,
    positions,
    p_stride,
    best_val,
    best_idx,
    start,
    V: tl.constexpr,
    K: tl.constexpr,
    NB: tl.constexpr,
    ROWS: tl.constexpr,
):
    # rows v of a vocabulary slice starting at global id `start` (0: full)
    pid, b = tl.program_id(0), tl.program_id(1)
    v = pid * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, K)
    valid = v < V
    raw = tl.load(weight + v[:, None] * K + k[None, :], valid[:, None], 0)
    w = (raw.to(tl.int32) << 16).to(tl.float32, bitcast=True)
    e = tl.load(embed_rows + b * K + k).to(tl.float32)
    bias = tl.sum(w * e[None, :], 1)
    x = tl.load(logits + b.to(tl.int64) * l_stride + v, valid, float("-inf")) + bias
    t = tl.load(temperature + b)
    if t > 0:
        seed = row_seed(tl.load(keys + b), tl.load(positions + b * p_stride))
        x = x / (t * scale) + gumbel(seed, (v + start).to(tl.int64))
    x = tl.where(valid, x, float("-inf"))
    m = tl.max(x, 0)
    tl.store(best_val + b * NB + pid, m)
    tl.store(best_idx + b * NB + pid, tl.min(tl.where(x == m, v, 2147483647), 0))


@triton.jit
def _markov_keyed_reduce(
    best_val,
    best_idx,
    embed,
    embed_rows,
    tokens,
    t_stride,
    V: tl.constexpr,
    K: tl.constexpr,
    NB: tl.constexpr,
    PB: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.arange(0, PB)
    valid = i < NB
    val = tl.load(best_val + b * NB + i, valid, float("-inf"))
    idx = tl.load(best_idx + b * NB + i, valid, 2147483647)
    m = tl.max(val, 0)
    token = tl.min(tl.where(val == m, idx, 2147483647), 0)
    tl.store(tokens + b * t_stride, token.to(tl.int64))
    # next step's row, clamped like markov_embed (an all-NaN step reports 2**31-1)
    row = tl.minimum(token, V - 1).to(tl.int64)
    k = tl.arange(0, K)
    tl.store(embed_rows + b * K + k, tl.load(embed + row * K + k))


def coupled_markov_tokens(
    greedy,
    base_logits: torch.Tensor,
    prev: torch.Tensor,
    temperature: torch.Tensor,
    keys: torch.Tensor,
    positions: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """[B, steps] int64 coupled drafts.

    greedy: the model's ``MarkovGreedy`` (BF16 head as int16, FP16 embedding);
    base_logits: [B, steps, vocab] draft logits; prev: [B] anchor tokens;
    temperature: [B] (0 = greedy row); keys: [B] int64 stream keys;
    positions: [B, steps] int64 output positions of the draft tokens.
    """
    batch, steps, vocab = base_logits.shape
    device = base_logits.device
    logits = base_logits.float().contiguous()
    tokens = torch.empty((batch, steps), dtype=torch.int64, device=device)
    if batch == 0:
        return tokens
    rows = torch.index_select(
        greedy.embed, 0, prev.reshape(-1).long().clamp(0, greedy.vocab - 1)
    ).contiguous()
    nb = greedy.blocks
    best_val = torch.empty((batch, nb), dtype=torch.float32, device=device)
    best_idx = torch.empty((batch, nb), dtype=torch.int32, device=device)
    temperature = temperature.to(torch.float32).contiguous()
    for step in range(steps):
        _markov_keyed_block[(nb, batch)](
            greedy.weight,
            rows,
            logits[:, step],
            logits.stride(0),
            temperature,
            float(scale),
            keys,
            positions[:, step],
            positions.stride(0),
            best_val,
            best_idx,
            0,
            V=vocab,
            K=greedy.rank,
            NB=nb,
            ROWS=ROWS,
            num_warps=4,
        )
        _markov_keyed_reduce[(batch,)](
            best_val,
            best_idx,
            greedy.embed,
            rows,
            tokens[:, step],
            tokens.stride(0),
            V=vocab,
            K=greedy.rank,
            NB=nb,
            PB=triton.next_power_of_2(nb),
            num_warps=4,
        )
    return tokens


def coupled_shard_tokens(
    shard,
    local_logits: torch.Tensor,
    anchor: torch.Tensor,
    temperature: torch.Tensor,
    keys: torch.Tensor,
    positions: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """[steps] int64 coupled drafts for one request from vocabulary shards.

    The keyed twin of ``MarkovShardGreedy.__call__``: each rank scores its
    rows ``[start, start + n)`` with ``_markov_keyed_block`` (keyed draws at
    the global ids), reduces them into its exchange slot, and one all-reduce
    per step picks the token (largest value, lowest global id) and its next
    Markov row on every rank, exactly as the greedy shard steps do. The
    drafts equal ``coupled_markov_tokens`` on the gathered logits. Every rank
    must call it with the same inputs (the exchanges pair up).

    local_logits: this rank's [steps, n] logits; anchor: [1] token id;
    temperature: [1]; keys: [1] int64; positions: [steps] int64.
    """
    tokens = torch.empty(shard.steps, dtype=torch.int64, device=local_logits.device)
    logits = local_logits.float().contiguous()
    first = anchor.reshape(-1)[:1].long().clamp(0, shard.vocab - 1)
    shard.row.copy_(torch.index_select(shard.embed, 0, first).reshape(-1))
    best_val = shard.best_val.view(1, -1)
    best_idx = shard.best_idx.view(1, -1)
    for step in range(shard.steps):
        _markov_keyed_block[(shard.blocks, 1)](
            shard.weight,
            shard.row,
            logits[step],
            logits.stride(0),
            temperature,
            float(scale),
            keys,
            positions[step:],
            1,
            best_val,
            best_idx,
            shard.start,
            V=shard.n,
            K=shard.rank,
            NB=shard.blocks,
            ROWS=ROWS,
            num_warps=4,
        )
        _markov_shard_slot[(1,)](
            shard.best_val,
            shard.best_idx,
            shard.slots[step],
            shard.start,
            shard.tp_rank,
            N=shard.n,
            BLOCKS=shard.blocks,
            PB=triton.next_power_of_2(shard.blocks),
            TP=shard.tp,
            TPP=triton.next_power_of_2(shard.tp),
            num_warps=4,
        )
        shard._pick(shard.exchange(shard.slots[step]), tokens[step : step + 1])
    return tokens
