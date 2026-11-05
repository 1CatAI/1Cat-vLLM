# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coupled sampled drafts with keyed-draw verification (exact).

A port of TensorFold's coupled drafts (Elevatormusic/TensorFold PR #25).
Every sampled token at output position t of a request is

    argmax(processed_target_logits + G(key, t))

where G(key, t)[v] is a Gumbel draw that is a pure function of the request
key, the position and the token id (a 32-bit integer hash, below). The
drafter picks its token at t with the *same* draw on its own logits at
temperature x DRAFT_TEMP_SCALE, so it agrees with the target far more often
than an independent draft. Verification accepts the longest draft prefix
that equals the target's keyed samples and emits the keyed sample at the
first mismatch (token matching), so every emitted token is the target's own
keyed sample: the output follows the target distribution exactly, whatever
the drafts were. A seeded request is reproducible regardless of drafting.

Exactness needs the draws to be independent across positions and tokens and
independent of everything emitted before t; G depends only on (key, t, v).
A position re-verified after a rejection reuses its draw, which is still
unseen by any emitted token.

Environment:
  VLLM_V41_COUPLED_DRAFTS     1 = on; default 0 (off)
  VLLM_V41_COUPLED_DRAFT_TEMP drafter temperature scale; default 0.5
"""

from __future__ import annotations

import functools
import hashlib
import os

import torch

from vllm.triton_utils import tl, triton

ENABLED = os.getenv("VLLM_V41_COUPLED_DRAFTS", "0") == "1"
DRAFT_TEMP_SCALE = float(os.getenv("VLLM_V41_COUPLED_DRAFT_TEMP", "0.5"))
if not DRAFT_TEMP_SCALE > 0.0:
    raise ValueError("VLLM_V41_COUPLED_DRAFT_TEMP must be > 0")
TRACE = os.getenv("VLLM_V41_COUPLED_TRACE", "")

M32 = 0xFFFFFFFF
_GOLDEN = 0x9E3779B9
_SALT = 0x6A09E667


def _mul32(x: torch.Tensor, c: int) -> torch.Tensor:
    """(x * c) mod 2**32 for x in [0, 2**32) without int64 overflow."""
    lo, hi = c & 0xFFFF, c >> 16
    return (x * lo + (((x * hi) & 0xFFFF) << 16)) & M32


def mix32(x: torch.Tensor) -> torch.Tensor:
    """lowbias32 (Wellons): a 32-bit avalanche hash on int64 lanes."""
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    return x ^ (x >> 16)


def row_seeds(keys: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """[R] 32-bit stream seeds for (key, output position) pairs."""
    keys = keys.to(torch.int64)
    hi = mix32(((keys >> 32) & M32) ^ _GOLDEN)
    s = mix32((keys & M32) ^ hi)
    return mix32(s ^ (positions.to(torch.int64) & M32))


def keyed_gumbel(
    keys: torch.Tensor, positions: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """[R, V] float32 Gumbel draws G(key_r, position_r)[v]."""
    s = row_seeds(keys, positions).unsqueeze(1)
    v = torch.arange(vocab_size, dtype=torch.int64, device=keys.device)
    h = mix32(mix32(s ^ v) ^ ((s + _SALT) & M32))
    # 23 bits: (k + 0.5) / 2**23 is exact in fp32 and strictly inside (0, 1).
    u = ((h >> 9).to(torch.float32) + 0.5) * (1.0 / (1 << 23))
    return -torch.log(-torch.log(u))


def keyed_sample(
    logits: torch.Tensor,
    keys: torch.Tensor,
    positions: torch.Tensor,
    random_rows: torch.Tensor,
) -> torch.Tensor:
    """[R] keyed samples of processed (temperature-scaled, masked) logits.

    Rows where `random_rows` is False take the plain argmax (greedy).
    """
    g = keyed_gumbel(keys, positions, logits.shape[-1])
    g = torch.where(random_rows.unsqueeze(1), g, torch.zeros_like(g))
    return (logits.to(torch.float32) + g).argmax(dim=-1)


def coupled_pick(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    gumbel: torch.Tensor,
    scale: float = DRAFT_TEMP_SCALE,
) -> torch.Tensor:
    """[R] draft tokens: argmax(logits / (T * scale) + G); greedy rows argmax.

    `temperature` is per row, 0 for greedy rows (vLLM's encoding).
    """
    random_rows = temperature > 0
    t = torch.where(random_rows, temperature * scale, torch.ones_like(temperature))
    g = torch.where(random_rows.unsqueeze(1), gumbel, torch.zeros_like(gumbel))
    return (logits.to(torch.float32) / t.unsqueeze(1) + g).argmax(dim=-1)


# Fused kernels: the same draws as keyed_gumbel computed inline, so no [R, V]
# noise tensor exists and a keyed argmax is two launches for any row count.
# The hash runs on int64 lanes holding uint32 values; a wrapping int64
# multiply keeps the low 32 bits exact, so the masked result equals _mul32.
@triton.jit
def _mix32(x):
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    return x ^ (x >> 16)


@triton.jit
def row_seed(key, position):
    """row_seeds for one (int64 key, int64 position)."""
    hi = _mix32(((key >> 32) & 0xFFFFFFFF) ^ 0x9E3779B9)
    s = _mix32((key & 0xFFFFFFFF) ^ hi)
    return _mix32(s ^ (position & 0xFFFFFFFF))


@triton.jit
def gumbel(seed, v):
    """keyed_gumbel values for one row seed at int64 token ids v."""
    h = _mix32(_mix32(seed ^ v) ^ ((seed + 0x6A09E667) & 0xFFFFFFFF))
    u = ((h >> 9).to(tl.float32) + 0.5) * (1.0 / 8388608.0)
    return -tl.log(-tl.log(u))


@triton.jit
def _keyed_block(
    logits,
    l_stride,
    temperature,
    scale,
    keys,
    positions,
    p_stride,
    best_val,
    best_idx,
    V: tl.constexpr,
    NB: tl.constexpr,
    DIVIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid, r = tl.program_id(0), tl.program_id(1)
    v = pid * BLOCK + tl.arange(0, BLOCK)
    valid = v < V
    x = tl.load(logits + r.to(tl.int64) * l_stride + v, valid, float("-inf"))
    x = x.to(tl.float32)
    t = tl.load(temperature + r)
    if t > 0:
        if DIVIDE:
            x = x / (t * scale)
        seed = row_seed(tl.load(keys + r), tl.load(positions + r * p_stride))
        x = x + gumbel(seed, v.to(tl.int64))
    x = tl.where(valid, x, float("-inf"))
    m = tl.max(x, 0)
    tl.store(best_val + r * NB + pid, m)
    tl.store(best_idx + r * NB + pid, tl.min(tl.where(x == m, v, 2147483647), 0))


@triton.jit
def _keyed_reduce(
    best_val, best_idx, out, o_stride, NB: tl.constexpr, PB: tl.constexpr
):
    r = tl.program_id(0)
    b = tl.arange(0, PB)
    valid = b < NB
    val = tl.load(best_val + r * NB + b, valid, float("-inf"))
    idx = tl.load(best_idx + r * NB + b, valid, 2147483647)
    m = tl.max(val, 0)
    token = tl.min(tl.where(val == m, idx, 2147483647), 0)
    tl.store(out + r * o_stride, token.to(tl.int64))


KEYED_BLOCK = 1024


def keyed_argmax(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    keys: torch.Tensor,
    positions: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """[R] int64 keyed picks in two launches, no host sync, graph safe.

    Rows with temperature > 0 take argmax(x + G(key, position)), where x is
    the row's logits divided by temperature * scale when `scale` is given
    (drafter) and the logits as they are otherwise (sampler: already
    temperature-scaled). Other rows take the plain argmax. Ties go to the
    lowest token id, as torch.argmax. `positions` may be a strided [R] view.
    """
    rows, vocab = logits.shape
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    out = torch.empty(rows, dtype=torch.int64, device=logits.device)
    if rows == 0:
        return out
    nb = triton.cdiv(vocab, KEYED_BLOCK)
    best_val = torch.empty((rows, nb), dtype=torch.float32, device=logits.device)
    best_idx = torch.empty((rows, nb), dtype=torch.int32, device=logits.device)
    _keyed_block[(nb, rows)](
        logits,
        logits.stride(0),
        temperature.to(torch.float32),
        1.0 if scale is None else float(scale),
        keys,
        positions,
        positions.stride(0),
        best_val,
        best_idx,
        V=vocab,
        NB=nb,
        DIVIDE=scale is not None,
        BLOCK=KEYED_BLOCK,
        num_warps=4,
    )
    _keyed_reduce[(rows,)](
        best_val, best_idx, out, 1, NB=nb, PB=triton.next_power_of_2(nb), num_warps=4
    )
    return out


# Target-side keyed picks from vocabulary shards (VLLM_V41_COUPLED_LOCAL):
# each TP rank scores only its shard of the verify logits at the global ids,
# then one [rows, tp, 2] FP32 all-reduce replaces the logits all-gather.
@triton.jit
def _keyed_shard_block(
    logits,
    l_stride,
    temperature,
    keys,
    positions,
    best_val,
    best_idx,
    start,
    N: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid, r = tl.program_id(0), tl.program_id(1)
    v = pid * BLOCK + tl.arange(0, BLOCK)
    valid = v < N
    x = tl.load(logits + r.to(tl.int64) * l_stride + v, valid, float("-inf"))
    x = x.to(tl.float32)
    if tl.load(temperature + r) > 0:
        seed = row_seed(tl.load(keys + r), tl.load(positions + r))
        x = x + gumbel(seed, v.to(tl.int64) + start)  # the global token id
    x = tl.where(valid, x, float("-inf"))
    m = tl.max(x, 0)
    tl.store(best_val + r * NB + pid, m)
    tl.store(best_idx + r * NB + pid, tl.min(tl.where(x == m, v, 2147483647), 0))


@triton.jit
def _keyed_shard_slot(
    best_val,
    best_idx,
    slots,
    start,
    rank,
    NB: tl.constexpr,
    PB: tl.constexpr,
    TP: tl.constexpr,
    TPP: tl.constexpr,
):
    """Row r's shard maximum and its lowest global id into slot [r, rank];
    the other slots of the row are zero, so a sum all-reduce copies every
    rank's pair to every rank exactly (ids below 2**24 are exact in FP32)."""
    r = tl.program_id(0)
    b = tl.arange(0, PB)
    valid = b < NB
    val = tl.load(best_val + r * NB + b, valid, float("-inf"))
    idx = tl.load(best_idx + r * NB + b, valid, 2147483647)
    m = tl.max(val, 0)
    gid = (tl.min(tl.where(val == m, idx, 2147483647), 0) + start).to(tl.float32)
    t = tl.arange(0, TPP)
    mine = t == rank
    tl.store(slots + r * (2 * TP) + 2 * t, tl.where(mine, m, 0.0), t < TP)
    tl.store(slots + r * (2 * TP) + 2 * t + 1, tl.where(mine, gid, 0.0), t < TP)


@triton.jit
def _keyed_shard_pick(slots, out, TP: tl.constexpr, TPP: tl.constexpr):
    """Largest shard maximum, lowest global id on ties (shards are in
    vocabulary order): keyed_argmax's pick over the gathered row."""
    r = tl.program_id(0)
    t = tl.arange(0, TPP)
    valid = t < TP
    val = tl.load(slots + r * (2 * TP) + 2 * t, valid, float("-inf"))
    gid = tl.load(slots + r * (2 * TP) + 2 * t + 1, valid, 0.0).to(tl.int32)
    m = tl.max(val, 0)
    token = tl.min(tl.where(val == m, gid, 2147483647), 0)
    tl.store(out + r, token.to(tl.int64))


class KeyedShard:
    """keyed_argmax for the target's verify rows on one TP rank's shard.

    ``local`` holds this rank's columns [start, start + n) of the processed
    (temperature-scaled) logits; the tokens equal keyed_argmax on the
    gathered rows whenever no row holds NaN. ``exchange`` sums a
    [rows, tp, 2] FP32 tensor over the TP group (the engine passes
    ``tensor_model_parallel_all_reduce``); every rank must call it with the
    same rows, so the exchanges pair up.
    """

    def __init__(self, start: int, n: int, rank: int, tp: int, exchange):
        if n <= 0 or tp < 1 or not 0 <= rank < tp:
            raise ValueError(f"bad shard start={start} n={n} rank={rank} tp={tp}")
        if start + n >= 1 << 24:
            raise ValueError("token ids must stay below 2**24 (exact in FP32)")
        self.start, self.n, self.rank, self.tp = start, n, rank, tp
        self.exchange = exchange

    def __call__(
        self,
        local: torch.Tensor,
        temperature: torch.Tensor,
        keys: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        rows, n = local.shape
        if n != self.n:
            raise ValueError(f"shard has {n} columns, expected {self.n}")
        if local.stride(1) != 1:
            local = local.contiguous()
        device = local.device
        out = torch.empty(rows, dtype=torch.int64, device=device)
        slots = torch.empty((rows, self.tp, 2), dtype=torch.float32, device=device)
        if rows == 0:
            return out
        nb = triton.cdiv(n, KEYED_BLOCK)
        best_val = torch.empty((rows, nb), dtype=torch.float32, device=device)
        best_idx = torch.empty((rows, nb), dtype=torch.int32, device=device)
        _keyed_shard_block[(nb, rows)](
            local,
            local.stride(0),
            temperature.to(torch.float32),
            keys.contiguous(),
            positions.contiguous(),
            best_val,
            best_idx,
            self.start,
            N=n,
            NB=nb,
            BLOCK=KEYED_BLOCK,
            num_warps=4,
        )
        tpp = triton.next_power_of_2(self.tp)
        _keyed_shard_slot[(rows,)](
            best_val,
            best_idx,
            slots,
            self.start,
            self.rank,
            NB=nb,
            PB=triton.next_power_of_2(nb),
            TP=self.tp,
            TPP=tpp,
            num_warps=4,
        )
        _keyed_shard_pick[(rows,)](
            self.exchange(slots), out, TP=self.tp, TPP=tpp, num_warps=1
        )
        return out


def local_fallback(sampling_metadata) -> str | None:
    """Why this step's verify rows must take the gathered path; None when
    the shard path gives the same tokens.

    The shard path covers plain temperature only: every row random (no
    greedy row in the batch), no top-k, top-p or active min_p, no logprobs,
    penalties, allowed or bad words, no thinking budget and no other active
    logits processor. Grammar (structured output) is the runner's check: it
    edits the gathered logits after the forward pass.
    """
    m = sampling_metadata
    if m.all_greedy or not m.all_random:
        return "greedy_rows"
    if m.temperature is None:
        return "no_temperature"
    if m.top_k is not None:
        return "top_k"
    if m.top_p is not None:
        return "top_p"
    if m.max_num_logprobs is not None or m.logprob_token_ids:
        return "logprobs"
    if not m.no_penalties:
        return "penalties"
    if m.allowed_token_ids_mask is not None:
        return "allowed_token_ids"
    if m.bad_words_token_ids:
        return "bad_words"
    holder = m.thinking_budget_state_holder
    if holder is not None and holder.has_tracked_requests():
        return "thinking_budget"
    for proc in m.logitsprocs.all:
        # With speculative decoding only min-tokens and min-p are built.
        if hasattr(proc, "min_p_count"):
            if proc.min_p_count:
                return "min_p"
        elif hasattr(proc, "min_toks"):
            if proc.min_toks:
                return "min_tokens"
        else:
            return "logits_processor"
    return None


_TRACE_STEP = [0]


def trace_rows(
    processed: torch.Tensor,
    temperature: torch.Tensor,
    keys: torch.Tensor,
    positions: torch.Tensor,
    sampled: torch.Tensor,
    path: str = TRACE,
) -> None:
    """Debug (VLLM_V41_COUPLED_TRACE): append one TSV line per keyed
    verification row to ``<path>.<pid>.tsv`` (host sync; off by default).

    Columns: step, row, key (low 16 bits), output position, sampled token,
    winner and runner-up of logits + G, margin (winner - runner-up), the
    processed logits of winner and runner-up as float hex (bit-exact across
    runs), and whether the fused pick equals the torch winner. The emitted
    token at a position is a function of (key, position, processed logits)
    only, so two runs that differ there show a changed logit (hex columns)
    or a near-tie margin.
    """
    noisy = (temperature > 0).unsqueeze(1)
    g = keyed_gumbel(keys, positions, processed.shape[-1])
    val = processed.to(torch.float32) + torch.where(noisy, g, torch.zeros_like(g))
    top = val.topk(2, dim=-1)
    win, run = top.indices[:, 0], top.indices[:, 1]
    margin = top.values[:, 0] - top.values[:, 1]
    lw = processed.to(torch.float32).gather(1, top.indices).cpu().tolist()
    cols = zip(
        (keys & 0xFFFF).tolist(),
        positions.tolist(),
        sampled.tolist(),
        win.tolist(),
        run.tolist(),
        margin.tolist(),
        lw,
    )
    step = _TRACE_STEP[0]
    _TRACE_STEP[0] += 1
    with open(f"{path}.{os.getpid()}.tsv", "a") as out:
        for row, (key, pos, tok, w, r, m, (a, b)) in enumerate(cols):
            out.write(
                f"{step}\t{row}\t{key}\t{pos}\t{tok}\t{w}\t{r}\t{m:.6g}\t"
                f"{float(a).hex()}\t{float(b).hex()}\t{int(tok == w)}\n"
            )


@functools.lru_cache(maxsize=8192)
def request_key(seed: int | None, req_id: str, engine_seed: int) -> int:
    """63-bit stream key: from the request seed when given (reproducible),
    else from the engine seed and the request id (identical on every rank)."""
    text = f"seed:{seed}" if seed is not None else f"req:{engine_seed}:{req_id}"
    digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


_KEYS: dict[str, torch.Tensor] = {}


def keys_buffer(device: torch.device, size: int) -> torch.Tensor:
    """The persistent per-batch-row key buffer shared by sampler and drafter."""
    name = str(device)
    buf = _KEYS.get(name)
    if buf is None or buf.numel() < size:
        buf = torch.zeros(size, dtype=torch.int64, device=device)
        _KEYS[name] = buf
    return buf
