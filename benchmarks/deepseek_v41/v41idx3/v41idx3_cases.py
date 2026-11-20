# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Operands for the tensor-core index scorer's tests, on the host and on one card (the same bits in both: torch's
generator from a fixed seed, made on the CPU and moved). No model output is involved.

Classes of operands (q rows, keys):
  close    the engine's own class (E2M1 values times a power of two per 32 dimensions) with close scales: query
           scales 2^-3..2^-1, key scale bytes 126..128. By arithmetic the guard passes every cell (A <= 5, B <= 2).
  spread   the same class with scales spread wide (query 2^-12..2^3, key bytes 121..131): some cells pass, the
           others take the fallback.
  plain    fp16 rows of any mantissa at exponents 2^-4..2^2, key bytes 121..131: the guard passes few or none.
  anybits  every fp16 bit pattern but NaN and infinity (subnormals included), every scale byte 0..255 (keys that
           are infinite, NaN or subnormal in fp16): rows and keys the guard does not take.
"""

from __future__ import annotations

import torch

POISON = 0x7E55
CLASSES = ("close", "spread", "plain", "anybits")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def halves(n: int, any_bits: bool, gen: torch.Generator) -> torch.Tensor:
    bits = torch.randint(0, 1 << 16, (n,), generator=gen, dtype=torch.int32)
    if any_bits:
        special = ((bits >> 10) & 31) == 31
        bits = torch.where(special, bits & 0xBFFF, bits)  # exponent 31 -> 15
    else:
        exp = torch.randint(11, 18, (n,), generator=gen, dtype=torch.int32)
        bits = (bits & 0x83FF) | (exp << 10)
    return bits.to(torch.int16).view(torch.float16)


def fp4_rows(rows: int, low: int, high: int, gen: torch.Generator) -> torch.Tensor:
    """What quant_rope._qdq (fp4, group 32) leaves: E2M1 values times 2^e per 32 dimensions, e in low..high."""
    codes = torch.randint(0, 16, (rows, 128), generator=gen)
    mag = E2M1[codes & 7]
    value = torch.where((codes & 8) != 0, -mag, mag)
    exponent = torch.randint(low, high + 1, (rows, 4), generator=gen)
    return (
        (value * torch.pow(2.0, exponent.float()).repeat_interleave(32, dim=1))
        .to(torch.float16)
        .contiguous()
    )


def make_keys(
    capacity: int, byte_low: int, byte_high: int, gen: torch.Generator
) -> torch.Tensor:
    keys = torch.randint(0, 256, (capacity, 68), generator=gen, dtype=torch.int32)
    keys[:, 64:] = torch.randint(
        byte_low, byte_high + 1, (capacity, 4), generator=gen, dtype=torch.int32
    )
    return keys.to(torch.uint8)


def operands(kind: str, tokens: int, capacity: int, gen: torch.Generator):
    """(q [tokens * 8, 128] fp16, w [tokens * 8] fp16, keys [capacity, 68] uint8), on the CPU."""
    rows = tokens * 8
    if kind == "close":
        q, keys = fp4_rows(rows, -3, -1, gen), make_keys(capacity, 126, 128, gen)
    elif kind == "spread":
        q, keys = fp4_rows(rows, -12, 3, gen), make_keys(capacity, 121, 131, gen)
    elif kind == "plain":
        q = halves(rows * 128, False, gen).view(rows, 128).contiguous()
        keys = make_keys(capacity, 121, 131, gen)
    elif kind == "anybits":
        q = halves(rows * 128, True, gen).view(rows, 128).contiguous()
        keys = make_keys(capacity, 0, 255, gen)
    else:
        raise ValueError(f"unknown class {kind!r}")
    return q, halves(rows, False, gen), keys


def poisoned(tokens: int, n_out: int, device) -> torch.Tensor:
    return torch.full((tokens, n_out), POISON, dtype=torch.int16, device=device).view(
        torch.float16
    )


def both(ref_op, new_op, q, w, keys, end: int, ratio: int, variant: int):
    """One case on the tensors' device: (cells that differ from the serving scorer, poison included; cells the
    new scorer's first look did not pass; heads of those it recomputed by the serving chain; problems, a list that
    is empty unless the serving scorer wrote past the live end (the test's premise) or the call's own count of its
    calls and cells is wrong). `end` may exceed the key capacity (a stale position): both scorers then stop at the
    capacity."""
    tokens = q.shape[0] // 8
    device = q.device
    positions = torch.full(
        (tokens,), end * ratio - 1, dtype=torch.int64, device=device
    )  # end = (last + 1) // ratio
    n_out = keys.shape[0] + 3
    ref = poisoned(tokens, n_out, device)
    ref_op(ref, q, w, keys, positions, 8, ratio)
    got = poisoned(tokens, n_out, device)
    counters = torch.zeros(4, dtype=torch.int64, device=device)
    new_op(got, q, w, keys, positions, 8, ratio, variant, counters)
    live = min(end, keys.shape[0])
    clean = bool((ref.view(torch.int16)[:, live:] == POISON).all())
    differ = int((ref.view(torch.int16) != got.view(torch.int16)).sum())
    problems = [] if clean else ["PREMISE: the serving scorer wrote past the live end"]
    # the call's own count of itself: one call and tokens * live cells if it has a live key, nothing if it has none
    want = (1, tokens * live) if live > 0 else (0, 0)
    if (int(counters[2]), int(counters[3])) != want:
        problems.append(
            f"COUNT: the call counted {int(counters[2])} call(s) and {int(counters[3])} cells, not "
            f"{want[0]} and {want[1]}"
        )
    return differ, int(counters[0]), int(counters[1]), problems


def boundary(tokens: int, capacity: int, a: int, b: int, gen: torch.Generator):
    """Operands whose guard sum is exactly a + b in every cell, by construction: in every row group 0 is all 6.0
    (largest exponent 2) and group 1 holds one 2^(2 - a) (lowest set bit 2 - a) and zeros, groups 2 and 3 zero: A =
    2 - (2 - a) = a. Every key's scale bytes are 127, 127, 127, 127 + b: B = b. Needs 3 <= a <= 15, 0 <= b <= 13."""
    rows = tokens * 8
    q = torch.zeros(rows, 128)
    q[:, :32] = 6.0
    q[:, 32] = 2.0 ** (2 - a)
    keys = torch.randint(0, 256, (capacity, 68), generator=gen, dtype=torch.int32)
    keys[:, 64:67] = 127
    keys[:, 67] = 127 + b
    return (
        q.to(torch.float16).contiguous(),
        halves(rows, False, gen),
        keys.to(torch.uint8),
    )
