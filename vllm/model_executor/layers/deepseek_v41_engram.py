# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 n-gram hashing with explicit accepted-input-prefix state commit.

This module holds small integer hash auxiliaries and three compressed token
ids, never an embedding table. A call produces candidate histories for every
prefix; ``commit`` selects the accepted input count, including the anchor.
"""

import torch
from torch import nn

import vllm.envs as envs
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


# ---- VLLM_V41_BATCH: per-request hash state slots --------------------------
# Same hashes and commit rule as V41EngramHashState, for a batch of requests.
# Token t belongs to request row rows[t] whose first token in this forward is
# bases[t]; the row's state lives in slot slots[row]. A slot keeps its own
# candidate histories (prefix 0 .. N of its pending forward), so a commit needs
# only (slot, count) and may run after later forwards of other slots.


@triton.jit
def _bcompressed(ids, token_map, mask, at, VOCAB: tl.constexpr):
    token = tl.load(ids + at).to(tl.int64)
    _fail_if((token < 0) | (token >= VOCAB))
    compressed = tl.load(token_map + token).to(tl.int64)
    return tl.where(tl.load(mask + at), compressed, -1)


@triton.jit
def _bhash_request(
    ids,
    positions,
    mask,
    query_start_loc,
    slots,
    history,
    metadata,
    scratch,
    errors,
    VOCAB: tl.constexpr,
    MAX_CONTEXT: tl.constexpr,
    RECORD: tl.constexpr,
):
    r = tl.program_id(0)
    first = tl.load(query_start_loc + r).to(tl.int64)
    n = tl.load(query_start_loc + r + 1).to(tl.int64) - first
    slot = tl.load(slots + r).to(tl.int64)
    # Triton: nested ifs on tensors (no short-circuit `and`).
    if (n > 0) & (slot != scratch):  # noqa: SIM102 (scratch = graph padding)
        if tl.load(mask + first):
            start = tl.load(positions + first).to(tl.int64)
            previous = tl.load(history + slot * 4 + 3)
            pending = tl.load(metadata + slot * 3 + 2)
            bad = (
                (start < 0)
                | (start + n > MAX_CONTEXT)
                | ((start != 0) & ((start != previous) | (pending != 0)))
            )
            if RECORD:
                # VLLM_V41_BATCH_DEBUG: record the first failure, do not trap.
                if bad:  # noqa: SIM102
                    if tl.atomic_cas(errors, 0, 1) == 0:
                        tl.store(errors + 1, slot.to(tl.int32))
                        tl.store(errors + 2, start.to(tl.int32))
                        tl.store(errors + 3, previous.to(tl.int32))
                        tl.store(errors + 4, pending.to(tl.int32))
                        tl.store(errors + 5, n.to(tl.int32))
                        tl.store(errors + 6, r)
            else:
                _fail_if(bad)
            for i in range(0, n):
                pos = tl.load(positions + first + i).to(tl.int64)
                token = tl.load(ids + first + i).to(tl.int64)
                _fail_if((pos != start + i) | (token < 0) | (token >= VOCAB))
            tl.store(metadata + slot * 3, start)
            tl.store(metadata + slot * 3 + 1, n)
            tl.store(metadata + slot * 3 + 2, 1)


@triton.jit
def _bhash_rows(
    ids,
    positions,
    mask,
    rows,
    bases,
    slots,
    token_map,
    primes,
    offsets,
    multipliers,
    pad,
    history,
    hashes,
    VOCAB: tl.constexpr,
):
    t, layer = tl.program_id(0), tl.program_id(1)
    heads = tl.arange(0, 32)
    base = tl.load(bases + t)
    slot = tl.load(slots + tl.load(rows + t)).to(tl.int64)
    local = t - base
    start = tl.load(positions + base)
    blocked = tl.full((), False, tl.int1)
    products = tl.full((4,), 0, tl.int64)
    live = tl.load(mask + t)
    for lag in tl.static_range(4):
        at = local - lag
        inside = live & (at >= 0)
        token = tl.load(ids + base + at, inside, 0).to(tl.int64)
        _fail_if(inside & ((token < 0) | (token >= VOCAB)))
        current = tl.load(token_map + token, inside, 0).to(tl.int64)
        current = tl.where(tl.load(mask + base + at, inside, False), current, -1)
        old = tl.load(history + slot * 4 + at + 3, live & (at < 0) & (at >= -3), -1)
        old = tl.where(start == 0, -1, old)
        current = tl.where(at >= 0, current, old)
        current = tl.where(live, current, -1)
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
    offset = tl.load(offsets + layer * 24 + heads, heads < 24, 0)
    tl.store(hashes + (t * 2 + layer) * 24 + heads, value % prime + offset, heads < 24)


@triton.jit
def _bhistory_candidates(
    ids,
    positions,
    mask,
    rows,
    bases,
    slots,
    token_map,
    history,
    candidates,
    PREFIXES: tl.constexpr,
    VOCAB: tl.constexpr,
):
    # Token t writes its row's prefix local + 1; a row's first token also
    # writes prefix 0. Padding tokens (mask false) write nothing.
    t = tl.program_id(0)
    if tl.load(mask + t):
        row = tl.load(rows + t)
        base = tl.load(bases + t)
        slot = tl.load(slots + row).to(tl.int64)
        local = t - base
        start = tl.load(positions + base)
        for extra in tl.static_range(2):
            prefix = local + 1 - extra
            if (extra == 0) | (local == 0):
                for column in tl.static_range(3):
                    at = prefix - 3 + column
                    inside = at >= 0
                    token = tl.load(ids + base + at, inside, 0).to(tl.int64)
                    current = tl.load(token_map + token, inside, 0).to(tl.int64)
                    current = tl.where(
                        tl.load(mask + base + at, inside, False), current, -1
                    )
                    old = tl.load(history + slot * 4 + at + 3, ~inside, -1)
                    old = tl.where(start == 0, -1, old)
                    value = tl.where(inside, current, old)
                    tl.store(
                        candidates + (slot * PREFIXES + prefix) * 3 + column, value
                    )


@triton.jit
def _bcommit_history(
    candidates,
    metadata,
    history,
    selector,
    slots,
    errors,
    PREFIXES: tl.constexpr,
    RECORD: tl.constexpr,
):
    # selector[i] < 0: entry i has no forward to commit (it is skipped).
    i = tl.program_id(0)
    count = tl.load(selector + i).to(tl.int64)
    if count >= 0:
        slot = tl.load(slots + i).to(tl.int64)
        length, start = tl.load(metadata + slot * 3 + 1), tl.load(metadata + slot * 3)
        pending = tl.load(metadata + slot * 3 + 2)
        bad = (pending != 1) | (count > length)
        if RECORD:
            if bad:
                if tl.atomic_cas(errors, 0, 2) == 0:
                    tl.store(errors + 1, slot.to(tl.int32))
                    tl.store(errors + 2, start.to(tl.int32))
                    tl.store(errors + 3, count.to(tl.int32))
                    tl.store(errors + 4, pending.to(tl.int32))
                    tl.store(errors + 5, length.to(tl.int32))
                    tl.store(errors + 6, i)
                return
        else:
            _fail_if(bad)
        col = tl.arange(0, 4)
        value = tl.load(candidates + (slot * PREFIXES + count) * 3 + col, col < 3, 0)
        value = tl.where(col < 3, value, start + count)
        tl.store(history + slot * 4 + col, value)
        tl.store(metadata + slot * 3 + 2, 0)


_DEBUG_LEVEL = int(envs.VLLM_V41_BATCH_DEBUG)
# Batched state checks record the first protocol violation on device and the
# host raises with it at the next step (check_errors), instead of a device
# trap: same detection, and the trap path failed under PP micro-batches with
# no violation ever recorded by this path (2026-11-08, 48-64 requests).
_DEBUG = True
_DEBUG_SYNC = _DEBUG_LEVEL >= 2  # also synchronous host checks and commit logs


class V41EngramHashBatch(nn.Module):
    """V41EngramHashState with one history per request slot."""

    def __init__(self, single, slots):
        super().__init__()
        device = single.history.device
        self.single = single
        self.history = torch.tensor(
            [-1, -1, -1, 0], dtype=torch.int64, device=device
        ).repeat(slots, 1)
        self.metadata = torch.zeros(slots, 3, dtype=torch.int64, device=device)
        self.prefixes = single.max_tokens + 1
        self.candidates = torch.empty(
            slots * self.prefixes, 3, dtype=torch.int64, device=device
        )
        self.max_context = single.max_context
        # The commit reads the forward's query_start_loc and slots after the
        # runner may have moved on: keep the ones that forward used.
        self.query_start_loc = torch.zeros(slots + 1, dtype=torch.int32, device=device)
        self.forward_slots = torch.zeros(slots, dtype=torch.int32, device=device)
        self.errors = torch.zeros(8, dtype=torch.int32, device=device)

    def check_errors(self):
        """VLLM_V41_BATCH_DEBUG: raise with the first recorded begin failure
        (reads the copy started one step earlier; never synchronizes)."""
        if not hasattr(self, "_errors_host"):
            self._errors_host = torch.zeros(8, dtype=torch.int32).pin_memory()
            self._errors_event = torch.cuda.Event()
            self._errors_event.record()
        if not self._errors_event.query():
            return
        values = self._errors_host.tolist()
        self._errors_host.copy_(self.errors, non_blocking=True)
        self._errors_event.record()
        if values[0] == 2:
            raise RuntimeError(
                f"V4.1 batch debug (commit): entry {values[6]} slot {values[1]} "
                f"pending forward starts at {values[2]} with {values[5]} tokens, "
                f"pending flag {values[4]}, commit count {values[3]}"
            )
        if values[0]:
            slot = values[1]
            raise RuntimeError(
                f"V4.1 batch debug (graph): row {values[6]} slot {slot} starts at "
                f"{values[2]} ({values[5]} tokens), committed {values[3]}, pending "
                f"{values[4]}"
            )

    def reset(self, slot):
        self.history[slot] = torch.tensor([-1, -1, -1, 0], device=self.history.device)
        self.metadata[slot].zero_()

    def forward(self, input_ids, positions, token_mask, layout, slots):
        s = self.single
        n = input_ids.numel()
        reqs = layout.num_reqs
        result = torch.empty(n, 2, 24, dtype=torch.int64, device=input_ids.device)
        vocab = s.token_map.numel()
        self.query_start_loc[: reqs + 1].copy_(layout.query_start_loc[: reqs + 1])
        self.forward_slots[:reqs].copy_(slots[:reqs])
        if _DEBUG_SYNC:
            self._debug_check(positions, token_mask, reqs)
        _bhash_request[(reqs,)](
            input_ids,
            positions,
            token_mask,
            self.query_start_loc,
            self.forward_slots,
            self.history,
            self.metadata,
            self.history.shape[0] - 1,
            self.errors,
            VOCAB=vocab,
            MAX_CONTEXT=self.max_context,
            RECORD=_DEBUG,
        )
        _bhash_rows[(n, 2)](
            input_ids,
            positions,
            token_mask,
            layout.rows,
            layout.bases,
            self.forward_slots,
            s.token_map,
            s.primes,
            s.offsets,
            s.multipliers,
            s.pad,
            self.history,
            result,
            VOCAB=vocab,
        )
        _bhistory_candidates[(n,)](
            input_ids,
            positions,
            token_mask,
            layout.rows,
            layout.bases,
            self.forward_slots,
            s.token_map,
            self.history,
            self.candidates,
            PREFIXES=self.prefixes,
            VOCAB=vocab,
        )
        return result

    def _debug_check(self, positions, token_mask, reqs):
        torch.accelerator.synchronize()
        qsl = self.query_start_loc[: reqs + 1].tolist()
        slots = self.forward_slots[:reqs].tolist()
        for r in range(reqs):
            a, b = qsl[r], qsl[r + 1]
            if b <= a or not bool(token_mask[a]):
                continue
            start = int(positions[a])
            previous = int(self.history[slots[r], 3])
            pending = int(self.metadata[slots[r], 2])
            if start != 0 and (start != previous or pending != 0):
                raise RuntimeError(
                    f"V4.1 batch debug: row {r} slot {slots[r]} starts at {start} "
                    f"({b - a} tokens), committed {previous}, pending {pending}; "
                    f"slots {slots} qsl {qsl}"
                )

    def commit(self, selector, slots):
        """selector[i] accepted inputs of the pending forward of slots[i]."""
        if _DEBUG_SYNC:
            torch.accelerator.synchronize()
            print(
                f"[v41 batch debug] commit selector {selector.tolist()} slots "
                f"{slots.tolist()} pending "
                f"{[int(self.metadata[x, 2]) for x in slots.tolist()]}",
                flush=True,
            )
        _bcommit_history[(selector.numel(),)](
            self.candidates,
            self.metadata,
            self.history,
            selector,
            slots,
            self.errors,
            PREFIXES=self.prefixes,
            RECORD=_DEBUG,
        )
