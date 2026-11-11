# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark greedy Markov steps on vocabulary shards (no full-vocabulary logits).

Each TP rank keeps its shard of the draft logits (the vocab-parallel head
without the all-gather) and scores only its rows of the BF16 Markov head with
upstream's own ``_markov_block`` kernel (``markov_fused``, 1Cat #1277), so
every row's bias is summed exactly as on the full path. ``_markov_shard_slot``
reduces the rank's block maxima like ``_markov_reduce`` and writes the rank's
exchange slot (maximum, global id); the other slots are zero, so one FP32
all-reduce copies all slots to every rank (ids stay below 2**24: exact).
``_markov_shard_pick`` then takes the largest maximum, lowest id on ties, and
reads the next step's row from the replicated Markov embedding.

The tokens are ``MarkovGreedy``'s: the maximum skips NaN as upstream's
``tl.max`` does, ties go to the lowest id (the shards are in vocabulary
order), and a shard whose rows are all NaN reports NaN, so an all-NaN step
yields upstream's 2**31 - 1. Only the row read differs there: upstream reads
past the table, this reads its last row.

Port of our earlier patch (cut against an older engine, where
the Markov embedding is split and the exchange also carries its row) onto
1Cat #1277 (899e9ed), where the embedding is replicated.
"""

from typing import Any

import torch

from vllm.triton_utils import tl, triton

from .markov_fused import ROWS, _markov_block


@triton.jit
def _markov_shard_slot(
    best_val,
    best_idx,
    slots,
    start,
    rank,
    N: tl.constexpr,
    BLOCKS: tl.constexpr,
    PB: tl.constexpr,
    TP: tl.constexpr,
    TPP: tl.constexpr,
):
    b = tl.arange(0, PB)
    valid = b < BLOCKS
    val = tl.load(best_val + b, valid, float("-inf"))
    idx = tl.load(best_idx + b, valid, 2147483647)
    m = tl.max(val, 0)
    token = tl.min(tl.where(val == m, idx, 2147483647), 0)
    # Only a shard whose real rows are all NaN ends past them: its last
    # partial block reports -inf from the padding rows. Upstream's aligned
    # blocks report NaN for those rows, so report NaN.
    real = token < N
    m = tl.where(real, m, float("nan"))
    gid = (tl.where(real, token, 0) + start).to(tl.float32)
    r = tl.arange(0, TPP)
    mine = r == rank
    tl.store(slots + 2 * r, tl.where(mine, m, 0.0), r < TP)
    tl.store(slots + 2 * r + 1, tl.where(mine, gid, 0.0), r < TP)


@triton.jit
def _markov_shard_pick(
    slots,
    embed,
    embed_out,
    token_out,
    V: tl.constexpr,
    K: tl.constexpr,
    TP: tl.constexpr,
    TPP: tl.constexpr,
):
    r = tl.arange(0, TPP)
    valid = r < TP
    val = tl.load(slots + 2 * r, valid, float("nan"))  # NaN: never selected
    gid = tl.load(slots + 2 * r + 1, valid, 0.0).to(tl.int32)
    m = tl.max(val, 0)
    token = tl.min(tl.where(val == m, gid, 2147483647), 0)
    tl.store(token_out, token.to(tl.int64))
    row = tl.minimum(token, V - 1)
    k = tl.arange(0, K)
    tl.store(embed_out + k, tl.load(embed + row.to(tl.int64) * K + k))


@triton.jit
def _markov_block_b(
    weight,
    rows,
    logits,
    best_val,
    best_idx,
    LSTRIDE,
    V: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    R: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    """_markov_block for R requests: the weight tile is read once."""
    pid = tl.program_id(0)
    v = pid * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, K)
    valid = v < V
    raw = tl.load(weight + v[:, None] * K + k[None, :], valid[:, None], 0)
    w = (raw.to(tl.int32) << 16).to(tl.float32, bitcast=True)
    for r in tl.static_range(R):
        e = tl.load(rows + r * K + k).to(tl.float32)
        bias = tl.sum(w * e[None, :], 1)
        value = tl.where(
            valid,
            tl.load(logits + r * LSTRIDE + v, valid, float("-inf")) + bias,
            float("-inf"),
        )
        m = tl.max(value, 0)
        idx = tl.min(tl.where(value == m, v, 2147483647), 0)
        tl.store(best_val + r * BLOCKS + pid, m)
        tl.store(best_idx + r * BLOCKS + pid, idx)


@triton.jit
def _markov_shard_slot_b(
    best_val,
    best_idx,
    slots,
    start,
    rank,
    N: tl.constexpr,
    BLOCKS: tl.constexpr,
    PB: tl.constexpr,
    TP: tl.constexpr,
    TPP: tl.constexpr,
):
    q = tl.program_id(0)
    b = tl.arange(0, PB)
    valid = b < BLOCKS
    val = tl.load(best_val + q * BLOCKS + b, valid, float("-inf"))
    idx = tl.load(best_idx + q * BLOCKS + b, valid, 2147483647)
    m = tl.max(val, 0)
    token = tl.min(tl.where(val == m, idx, 2147483647), 0)
    real = token < N
    m = tl.where(real, m, float("nan"))
    gid = (tl.where(real, token, 0) + start).to(tl.float32)
    r = tl.arange(0, TPP)
    mine = r == rank
    base = slots + q * TP * 2
    tl.store(base + 2 * r, tl.where(mine, m, 0.0), r < TP)
    tl.store(base + 2 * r + 1, tl.where(mine, gid, 0.0), r < TP)


@triton.jit
def _markov_shard_pick_b(
    slots,
    embed,
    rows_out,
    token_out,
    TSTRIDE,
    V: tl.constexpr,
    K: tl.constexpr,
    TP: tl.constexpr,
    TPP: tl.constexpr,
):
    q = tl.program_id(0)
    r = tl.arange(0, TPP)
    valid = r < TP
    base = slots + q * TP * 2
    val = tl.load(base + 2 * r, valid, float("nan"))
    gid = tl.load(base + 2 * r + 1, valid, 0.0).to(tl.int32)
    m = tl.max(val, 0)
    token = tl.min(tl.where(val == m, gid, 2147483647), 0)
    tl.store(token_out + q * TSTRIDE, token.to(tl.int64))
    row = tl.minimum(token, V - 1)
    k = tl.arange(0, K)
    tl.store(rows_out + q * K + k, tl.load(embed + row.to(tl.int64) * K + k))


class MarkovShardGreedy:
    """Graph-capturable greedy Markov drafting for one request on one TP rank.

    ``greedy`` is the rank's ``MarkovGreedy``: its BF16 head and replicated
    embedding are shared (views, no copy). This rank scores vocabulary rows
    ``[rank * n, (rank + 1) * n)``. ``exchange`` sums a [tp, 2] FP32 tensor
    over the TP group (the engine passes ``tensor_model_parallel_all_reduce``);
    every rank must call with the same anchor, so the exchanges pair up.
    """

    def __init__(self, greedy, rank, tp, exchange):
        vocab = greedy.vocab
        if tp < 1 or not 0 <= rank < tp or vocab % tp:
            raise ValueError(f"vocabulary {vocab} is not {tp} equal shards")
        if vocab >= 1 << 24:
            raise ValueError("token ids must stay below 2**24 (exact in FP32)")
        n = vocab // tp
        self.vocab, self.rank, self.steps = vocab, greedy.rank, greedy.steps
        self.n, self.start, self.tp, self.tp_rank = n, rank * n, tp, rank
        self.weight = greedy.weight[self.start : self.start + n]
        self.embed = greedy.embed
        self.exchange = exchange
        device = self.weight.device
        self.blocks = triton.cdiv(n, ROWS)
        self.best_val = torch.empty(self.blocks, dtype=torch.float32, device=device)
        self.best_idx = torch.empty(self.blocks, dtype=torch.int32, device=device)
        self.row = torch.empty(self.rank, dtype=self.embed.dtype, device=device)
        # Persistent exchange inputs, one per step (fixed addresses in graphs).
        self.slots = torch.empty(
            (self.steps, tp, 2), dtype=torch.float32, device=device
        )

    def _score(self, logits, slot):
        """This rank's best row for one step's FP32 logits [n] into ``slot``."""
        _markov_block[(self.blocks,)](
            self.weight,
            self.row,
            logits,
            self.best_val,
            self.best_idx,
            V=self.n,
            K=self.rank,
            ROWS=ROWS,
            num_warps=4,
        )
        _markov_shard_slot[(1,)](
            self.best_val,
            self.best_idx,
            slot,
            self.start,
            self.tp_rank,
            N=self.n,
            BLOCKS=self.blocks,
            PB=triton.next_power_of_2(self.blocks),
            TP=self.tp,
            TPP=triton.next_power_of_2(self.tp),
            num_warps=4,
        )

    def _pick(self, slots, token_out):
        _markov_shard_pick[(1,)](
            slots,
            self.embed,
            self.row,
            token_out,
            V=self.vocab,
            K=self.rank,
            TP=self.tp,
            TPP=triton.next_power_of_2(self.tp),
            num_warps=4,
        )

    def warm(self):
        """Compile and launch every kernel once, without any collective."""
        logits = torch.zeros(self.n, dtype=torch.float32, device=self.weight.device)
        token = torch.empty(1, dtype=torch.int64, device=self.weight.device)
        self.row.zero_()
        self._score(logits, self.slots[0])
        self._pick(self.slots[0], token)

    def __call__(self, local_logits, anchor):
        """local_logits: this rank's [steps, n] logits; anchor: [1] token id."""
        tokens = torch.empty(self.steps, dtype=torch.int64, device=local_logits.device)
        logits = local_logits.float().contiguous()
        # Non-sampling steps may carry an unfilled anchor; clamp as upstream.
        first = anchor.reshape(-1)[:1].long().clamp(0, self.vocab - 1)
        self.row.copy_(torch.index_select(self.embed, 0, first).reshape(-1))
        for step in range(self.steps):
            self._score(logits[step], self.slots[step])
            self._pick(self.exchange(self.slots[step]), tokens[step : step + 1])
        return tokens

    def call_batch(self, local_logits, anchors):
        """VLLM_V41_BATCH: R requests at once. local_logits: [R * steps, n]
        (request-major), anchors: [R]. Same tokens as R single calls (each
        request's scores, exchange and pick are independent); returns
        [R * steps] request-major."""
        r = anchors.numel()
        if r == 1:
            return self(local_logits, anchors)
        device = local_logits.device
        n, steps, k = self.n, self.steps, self.rank
        batch: dict[int, Any] = self.__dict__.setdefault("_batch", {})
        state = batch.get(r)
        if state is None:
            state = batch[r] = (
                torch.empty(r * self.blocks, dtype=torch.float32, device=device),
                torch.empty(r * self.blocks, dtype=torch.int32, device=device),
                torch.empty((r, k), dtype=self.embed.dtype, device=device),
                torch.empty((steps, r, self.tp, 2), dtype=torch.float32, device=device),
            )
        best_val, best_idx, rows, slots = state
        tokens = torch.empty((r, steps), dtype=torch.int64, device=device)
        logits = local_logits.float().contiguous()
        first = anchors.reshape(-1).long().clamp(0, self.vocab - 1)
        rows.copy_(torch.index_select(self.embed, 0, first))
        tpp = triton.next_power_of_2(self.tp)
        for step in range(steps):
            _markov_block_b[(self.blocks,)](
                self.weight,
                rows,
                logits[step],
                best_val,
                best_idx,
                steps * n,
                V=n,
                K=k,
                ROWS=ROWS,
                R=r,
                BLOCKS=self.blocks,
                num_warps=4,
            )
            _markov_shard_slot_b[(r,)](
                best_val,
                best_idx,
                slots[step],
                self.start,
                self.tp_rank,
                N=n,
                BLOCKS=self.blocks,
                PB=triton.next_power_of_2(self.blocks),
                TP=self.tp,
                TPP=tpp,
                num_warps=4,
            )
            exchanged = self.exchange(slots[step])
            _markov_shard_pick_b[(r,)](
                exchanged,
                self.embed,
                rows,
                tokens[:, step],
                steps,
                V=self.vocab,
                K=k,
                TP=self.tp,
                TPP=tpp,
                num_warps=4,
            )
        return tokens.reshape(-1)
