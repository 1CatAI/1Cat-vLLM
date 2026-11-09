# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DSpark Markov greedy step: bias = W_bf16 @ embed(prev), argmax(logits + bias).

W stays in the checkpoint's BF16 (bit-shifted to FP32 in registers, exact), so
each step reads half the bytes of the FP32 copy and the [vocab] bias never
materializes. A second tiny kernel reduces block maxima (lowest index on ties,
as torch.argmax) and gathers the next step's embedding row.
"""

import torch

from vllm.triton_utils import tl, triton

ROWS = 128


@triton.jit
def _markov_block(
    weight,
    embed_row,
    logits,
    best_val,
    best_idx,
    V: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    v = pid * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, K)
    valid = v < V
    raw = tl.load(weight + v[:, None] * K + k[None, :], valid[:, None], 0)
    w = (raw.to(tl.int32) << 16).to(tl.float32, bitcast=True)
    e = tl.load(embed_row + k).to(tl.float32)
    bias = tl.sum(w * e[None, :], 1)
    value = tl.where(
        valid, tl.load(logits + v, valid, float("-inf")) + bias, float("-inf")
    )
    m = tl.max(value, 0)
    idx = tl.min(tl.where(value == m, v, 2147483647), 0)
    tl.store(best_val + pid, m)
    tl.store(best_idx + pid, idx)


@triton.jit
def _markov_reduce(
    best_val,
    best_idx,
    embed,
    embed_out,
    token_out,
    BLOCKS: tl.constexpr,
    PB: tl.constexpr,
    K: tl.constexpr,
):
    b = tl.arange(0, PB)
    valid = b < BLOCKS
    val = tl.load(best_val + b, valid, float("-inf"))
    idx = tl.load(best_idx + b, valid, 2147483647)
    m = tl.max(val, 0)
    token = tl.min(tl.where(val == m, idx, 2147483647), 0)
    tl.store(token_out, token.to(tl.int64))
    k = tl.arange(0, K)
    tl.store(embed_out + k, tl.load(embed + token.to(tl.int64) * K + k))


class MarkovGreedy:
    """Graph-capturable 5-step greedy Markov drafting for one request."""

    def __init__(self, weight_bf16, embed_fp16, steps=5):
        vocab, rank = weight_bf16.shape
        self.weight = weight_bf16.contiguous().view(torch.int16)
        self.embed = embed_fp16.contiguous()
        self.vocab, self.rank, self.steps = vocab, rank, steps
        device = weight_bf16.device
        self.blocks = triton.cdiv(vocab, ROWS)
        self.best_val = torch.empty(self.blocks, dtype=torch.float32, device=device)
        self.best_idx = torch.empty(self.blocks, dtype=torch.int32, device=device)
        self.row = torch.empty(rank, dtype=embed_fp16.dtype, device=device)

    def __call__(self, logits, anchor):
        tokens = torch.empty(self.steps, dtype=torch.int64, device=logits.device)
        logits = logits.float().contiguous()
        # Non-sampling steps may carry an unfilled anchor; their drafts are discarded.
        first = anchor.reshape(-1)[:1].long().clamp(0, self.vocab - 1)
        self.row.copy_(torch.index_select(self.embed, 0, first).reshape(-1))
        for step in range(self.steps):
            _markov_block[(self.blocks,)](
                self.weight,
                self.row,
                logits[step],
                self.best_val,
                self.best_idx,
                V=self.vocab,
                K=self.rank,
                ROWS=ROWS,
                num_warps=4,
            )
            _markov_reduce[(1,)](
                self.best_val,
                self.best_idx,
                self.embed,
                self.row,
                tokens[step : step + 1],
                BLOCKS=self.blocks,
                PB=triton.next_power_of_2(self.blocks),
                K=self.rank,
                num_warps=4,
            )
        return tokens


@triton.jit
def _markov_bias(
    weight, embed_rows, out, V: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr
):
    pid, n = tl.program_id(0), tl.program_id(1)
    v = pid * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, K)
    valid = v < V
    raw = tl.load(weight + v[:, None] * K + k[None, :], valid[:, None], 0)
    w = (raw.to(tl.int32) << 16).to(tl.float32, bitcast=True)
    e = tl.load(embed_rows + n * K + k).to(tl.float32)
    tl.store(out + n * V + v, tl.sum(w * e[None, :], 1), valid)


def markov_bias(greedy, embedding):
    """FP32 [n, vocab] Markov bias for the sampling path, from BF16 weights."""
    rows = embedding.reshape(-1, greedy.rank).contiguous()
    out = torch.empty(
        rows.shape[0], greedy.vocab, dtype=torch.float32, device=rows.device
    )
    if rows.shape[0]:
        _markov_bias[(greedy.blocks, rows.shape[0])](
            greedy.weight,
            rows,
            out,
            V=greedy.vocab,
            K=greedy.rank,
            ROWS=ROWS,
            num_warps=4,
        )
    return out.view(*embedding.shape[:-1], greedy.vocab)
