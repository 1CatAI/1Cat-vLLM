# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 n-gram hashing with explicit accepted-input-prefix state commit.

This module holds small integer hash auxiliaries and three compressed token
ids, never an embedding table. A call produces candidate histories for every
prefix; ``commit`` selects the accepted input count, including the anchor.
"""

import torch
from torch import nn

from vllm.triton_utils import tl, triton

from .quantization.gguf_shared_protocol import _fail_if


@triton.jit
def _compressed(ids, token_map, mask, t, N, VOCAB: tl.constexpr, MASKED: tl.constexpr):
    valid = (t >= 0) & (t < N)
    token = tl.load(ids + t, valid, 0).to(tl.int64)
    _fail_if(valid & ((token < 0) | (token >= VOCAB)))
    compressed = tl.load(token_map + token, valid, 0).to(tl.int64)
    if MASKED:
        live = tl.load(mask + t, valid, 0)
        compressed = tl.where(live, compressed, -1)
    return compressed


@triton.jit(do_not_specialize=["N"])
def _hash_request(
    ids,
    positions,
    token_map,
    mask,
    history,
    metadata,
    N,
    VOCAB: tl.constexpr,
    MAX_CONTEXT: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    start = tl.load(positions).to(tl.int64)
    previous = tl.load(history + 3)
    pending = tl.load(metadata + 2)
    _fail_if(
        (start < 0)
        | (start + N > MAX_CONTEXT)
        | ((start != 0) & ((start != previous) | (pending != 0)))
    )
    t = tl.arange(0, BLOCK)
    pos = tl.load(positions + t, t < N, 0).to(tl.int64)
    _fail_if(tl.sum(((t < N) & (pos != start + t)).to(tl.int32), 0))
    # Validate tokens before any token-map dereference on following launches.
    token = tl.load(ids + t, t < N, 0).to(tl.int64)
    _fail_if(tl.sum(((t < N) & ((token < 0) | (token >= VOCAB))).to(tl.int32), 0))
    tl.store(metadata, start)
    tl.store(metadata + 1, N)
    tl.store(metadata + 2, 1)


@triton.jit(do_not_specialize=["N"])
def _hash_rows(
    ids,
    positions,
    mask,
    token_map,
    primes,
    offsets,
    multipliers,
    pad,
    history,
    hashes,
    N,
    VOCAB: tl.constexpr,
    MASKED: tl.constexpr,
):
    t, layer = tl.program_id(0), tl.program_id(1)
    heads = tl.arange(0, 32)
    start = tl.load(positions)
    blocked = tl.full((), False, tl.int1)
    products = tl.full((4,), 0, tl.int64)
    for lag in tl.static_range(4):
        at = t - lag
        current = _compressed(ids, token_map, mask, at, N, VOCAB, MASKED)
        old = tl.load(history + at + 3, (at < 0) & (at >= -3), -1)
        old = tl.where(start == 0, -1, old)
        current = tl.where(at >= 0, current, old)
        blocked |= current == -1
        current = tl.where(blocked, tl.load(pad), current)
        product = current * tl.load(multipliers + layer * 4 + lag)
        products = tl.where(tl.arange(0, 4) == lag, product, products)
    p0 = tl.sum(tl.where(tl.arange(0, 4) == 0, products, 0), 0)
    p1 = tl.sum(tl.where(tl.arange(0, 4) == 1, products, 0), 0)
    p2 = tl.sum(tl.where(tl.arange(0, 4) == 2, products, 0), 0)
    p3 = tl.sum(tl.where(tl.arange(0, 4) == 3, products, 0), 0)
    value = p0 ^ p1
    value = tl.where(heads >= 8, value ^ p2, value)
    value = tl.where(heads >= 16, value ^ p3, value)
    prime = tl.load(primes + layer * 24 + heads, heads < 24, 1)
    base = tl.load(offsets + layer * 24 + heads, heads < 24, 0)
    tl.store(hashes + (t * 2 + layer) * 24 + heads, value % prime + base, heads < 24)


@triton.jit(do_not_specialize=["N"])
def _history_candidates(
    ids,
    positions,
    mask,
    token_map,
    history,
    candidates,
    N,
    VOCAB: tl.constexpr,
    MASKED: tl.constexpr,
):
    prefix = tl.program_id(0)
    start = tl.load(positions)
    for column in tl.static_range(3):
        at = prefix - 3 + column
        current = _compressed(ids, token_map, mask, at, N, VOCAB, MASKED)
        old = tl.load(history + at + 3, (at < 0) & (at >= -3), -1)
        old = tl.where(start == 0, -1, old)
        value = tl.where(at >= 0, current, old)
        tl.store(candidates + prefix * 3 + column, value)


@triton.jit
def _commit_history(candidates, metadata, history, selector):
    count = tl.load(selector).to(tl.int64)
    length, start = tl.load(metadata + 1), tl.load(metadata)
    _fail_if((tl.load(metadata + 2) != 1) | (count < 0) | (count > length))
    col = tl.arange(0, 4)
    value = tl.load(candidates + count * 3 + col, col < 3, 0)
    value = tl.where(col < 3, value, start + count)
    tl.store(history + col, value)
    tl.store(metadata + 2, 0)


class V41EngramHashState(nn.Module):
    """Single-request CUDA history; the runner owns the commit selector."""

    def __init__(
        self,
        token_map,
        primes,
        offsets,
        multipliers,
        *,
        pad_token_id=2,
        max_tokens=2048,
        max_context=16384,
    ):
        super().__init__()
        device = token_map.device
        if (
            not token_map.is_cuda
            or token_map.ndim != 1
            or token_map.dtype not in (torch.int32, torch.int64)
            or primes.shape != (2, 24)
            or offsets.shape != (2, 24)
            or multipliers.shape != (2, 4)
            or any(
                t.dtype != torch.int64 or t.device != device
                for t in (primes, offsets, multipliers)
            )
            or not 0 <= pad_token_id < token_map.numel()
            or not 1 <= max_tokens <= max_context <= 1048576
        ):
            raise ValueError("V4.1 Engram requires two integer hash layouts on one GPU")
        if bool((primes <= 0).any()) or bool((token_map < 0).any()):
            raise ValueError("Invalid V4.1 hash modulus or compressed token id")
        for name, value in (
            ("token_map", token_map),
            ("primes", primes),
            ("offsets", offsets),
            ("multipliers", multipliers),
        ):
            self.register_buffer(name, value.contiguous(), persistent=False)
        self.register_buffer(
            "pad", token_map[pad_token_id : pad_token_id + 1].long(), persistent=False
        )
        self.register_buffer(
            "history",
            torch.tensor([-1, -1, -1, 0], dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "candidates",
            torch.empty(max_tokens + 1, 3, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "metadata",
            torch.zeros(3, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.max_tokens, self.max_context = max_tokens, max_context

    def forward(self, input_ids, positions, token_mask=None):
        n = input_ids.numel()
        if (
            input_ids.ndim != 1
            or positions.shape != input_ids.shape
            or not 0 < n <= self.max_tokens
            or input_ids.dtype not in (torch.int32, torch.int64)
            or positions.dtype not in (torch.int32, torch.int64)
            or any(
                t.device != self.history.device or not t.is_contiguous()
                for t in (input_ids, positions)
            )
            or (
                token_mask is not None
                and (
                    token_mask.shape != input_ids.shape
                    or token_mask.dtype != torch.bool
                    or token_mask.device != input_ids.device
                )
            )
        ):
            raise ValueError("Engram hashing requires one contiguous token chain")
        result = torch.empty(n, 2, 24, dtype=torch.int64, device=input_ids.device)
        mask = token_mask if token_mask is not None else input_ids
        meta = dict(N=n, VOCAB=self.token_map.numel(), MASKED=token_mask is not None)
        _hash_request[(1,)](
            input_ids,
            positions,
            self.token_map,
            mask,
            self.history,
            self.metadata,
            MAX_CONTEXT=self.max_context,
            BLOCK=triton.next_power_of_2(n),
            **meta,
        )
        _hash_rows[(n, 2)](
            input_ids,
            positions,
            mask,
            self.token_map,
            self.primes,
            self.offsets,
            self.multipliers,
            self.pad,
            self.history,
            result,
            **meta,
        )
        _history_candidates[(n + 1,)](
            input_ids,
            positions,
            mask,
            self.token_map,
            self.history,
            self.candidates,
            **meta,
        )
        return result

    def commit(self, accepted_inputs):
        if (
            accepted_inputs.numel() != 1
            or accepted_inputs.dtype not in (torch.int32, torch.int64)
            or accepted_inputs.device != self.history.device
        ):
            raise ValueError("Engram commit selector must be one device integer")
        _commit_history[(1,)](
            self.candidates, self.metadata, self.history, accepted_inputs
        )


@triton.jit
def _engram_gate(
    residual,
    kv,
    q_weight,
    k_weight,
    mask,
    out,
    H: tl.constexpr,
    EPS: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token, stream = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, BLOCK)
    valid = col < H
    h = tl.load(residual + (token * 4 + stream) * H + col, valid, 0).to(tl.float32)
    key = tl.load(kv + (token * 5 + stream) * H + col, valid, 0).to(tl.float32)
    weight = tl.load(q_weight + stream * H + col, valid, 0) * tl.load(
        k_weight + stream * H + col, valid, 0
    )
    dot = tl.sum((h * weight) * key, 0)
    rstd = tl.rsqrt(tl.sum(h * h, 0) / H + EPS) * tl.rsqrt(
        tl.sum(key * key, 0) / H + EPS
    )
    dot = (dot * rstd) * (H**-0.5)
    root = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
    # copysign also preserves the sign of a negative zero.
    signed = (
        root.to(tl.uint32, bitcast=True)
        | (dot.to(tl.uint32, bitcast=True) & 0x80000000)
    ).to(tl.float32, bitcast=True)
    gate = tl.sigmoid(signed)
    if MASKED:
        gate = tl.where(tl.load(mask + token), gate, 0)
    value = tl.load(kv + (token * 5 + 4) * H + col, valid, 0).to(tl.float32)
    tl.store(out + (token * 4 + stream) * H + col, h + gate * value, valid)


def apply_engram_gate(
    residual,
    projected_kv,
    q_weight,
    k_weight,
    token_mask=None,
    *,
    inplace=False,
    eps=1e-20,
):
    """FP32 per-stream gate; preserve the caller's HC storage without a copy."""
    n, hc, hidden = residual.shape
    if (
        not residual.is_cuda
        or residual.dtype not in (torch.float16, torch.float32)
        or hc != 4
        or projected_kv.shape != (n, 5 * hidden)
        or projected_kv.dtype != torch.float16
        or q_weight.shape != (4, hidden)
        or k_weight.shape != (4, hidden)
        or q_weight.dtype != torch.float32
        or k_weight.dtype != torch.float32
        or any(
            t.device != residual.device or not t.is_contiguous()
            for t in (residual, projected_kv, q_weight, k_weight)
        )
        or (
            token_mask is not None
            and (
                token_mask.shape != (n,)
                or token_mask.dtype != torch.bool
                or token_mask.device != residual.device
            )
        )
    ):
        raise ValueError("V4.1 Engram gate requires FP16 values and FP32 q/k weights")
    result = residual if inplace else torch.empty_like(residual)
    _engram_gate[(n, 4)](
        residual,
        projected_kv,
        q_weight,
        k_weight,
        token_mask if token_mask is not None else residual,
        result,
        H=hidden,
        EPS=eps,
        MASKED=token_mask is not None,
        BLOCK=triton.next_power_of_2(hidden),
        num_warps=8,
        enable_fp_fusion=False,
    )
    return result
