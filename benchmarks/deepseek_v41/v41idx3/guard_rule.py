# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The v41idx3 guard's rule stated a second time, in numpy, independent of the kernel's code.

The tests and check_v41idx3.py take the expected counters (cells the first look does not pass; heads of those the
second look does not pass either) from here, never from the library under test.
"""

from __future__ import annotations

import numpy as np

NO_BIT = 10**6  # the lowest set bit's exponent of a group that is all zero
GUARD_LIMIT = 11  # the kernel's guard passes A + B <= 11
GUARD_BYTES = (
    114,
    140,
)  # scale bytes for which every E2M1 x 2^(byte - 127) is a NORMAL fp16 (0.5 x 2^-13 up)
OUT = 10**4  # A or B of a row or key the guard does not take


def group_bits(x: np.ndarray):
    """For fp16-valued float64 [n, 128]: per 32-dimension group, (the largest magnitude [n, 4], the exponent of
    the lowest set bit over the group's nonzero values [n, 4]; NO_BIT where the group is all zero)."""
    g = x.reshape(x.shape[0], 4, 32)
    m, e = np.frexp(np.abs(g))  # |x| = m * 2^e, m in [0.5, 1): m * 2^53 is an integer
    n = np.round(m * 2.0**53).astype(np.int64)
    low = n & -n
    bit = e - 53 + np.round(np.log2(np.maximum(low, 1))).astype(np.int64)
    return np.abs(g).max(-1), np.where(g != 0, bit, NO_BIT).min(-1)


def kernel_guard(q: np.ndarray, q_bits, raw: np.ndarray, tokens: int, heads: int):
    """The scale test as the kernel has it, in integers only.
    Row: over its nonzero 32-dimension groups, Eq = the exponent of the largest magnitude, lq = the exponent of
    the lowest set bit. Key: ek = its four scale bytes - 127 (its values are at most 6 x 2^ek, multiples of
    2^(ek - 1)), taken only if every byte is in GUARD_BYTES; a row only if all its elements are zero or normal
    fp16. Every partial sum of either chain is below 2^(max(Eq + ek) + 11) and a multiple of
    2^(min(lq + ek) - 1), so neither chain rounds if max_g(Eq + ek) - min_g(lq + ek) <= 11.
    Returns (pair [rows, keys]: that test per row and key;
             cell [tokens, keys]: the separable form the first look runs, A[token] + B[key] <= 11 with
               A = max over the token's heads of (max Eq - min lq), B = max ek - min ek;
             A + B per cell)."""
    mq, lq = q_bits
    nonzero = mq > 0
    # a row is taken only if every element is zero or a normal fp16 (no infinity, NaN or subnormal: the adder was
    # measured on normal operands)
    finite = (np.isfinite(q) & ((q == 0) | (np.abs(q) >= 2.0**-14))).all(-1)
    eq = np.where(nonzero, np.frexp(np.where(nonzero, mq, 1.0))[1] - 1, 0)
    ek = raw[:, 64:68].astype(np.int64) - 127
    taken = ((raw[:, 64:68] >= GUARD_BYTES[0]) & (raw[:, 64:68] <= GUARD_BYTES[1])).all(
        -1
    )
    both = nonzero[:, None, :]
    top = np.where(both, eq[:, None, :] + ek[None, :, :], -OUT).max(-1)
    low = np.where(both, lq[:, None, :] + ek[None, :, :], OUT).min(-1)
    pair = (
        ((top - low <= GUARD_LIMIT) | ~nonzero.any(-1)[:, None])
        & taken[None, :]
        & finite[:, None]
    )
    a_row = np.where(
        nonzero.any(-1),
        np.where(nonzero, eq, -OUT).max(-1) - np.where(nonzero, lq, OUT).min(-1),
        0,
    )
    a_row = np.where(finite, a_row, OUT)
    a_token = a_row.reshape(tokens, heads).max(-1)
    b_key = np.where(taken, ek.max(-1) - ek.min(-1), OUT)
    total = a_token[:, None] + b_key[None, :]
    return pair, total <= GUARD_LIMIT, np.minimum(total, OUT)


def expected_counts(q, keys, tokens: int, live: int) -> tuple[int, int]:
    """(cells the first look must not pass, heads of those the second look must not pass either) for `live`
    live keys, from the rule above."""
    if live == 0:
        return 0, 0
    q64 = q.cpu().numpy().astype(np.float64)
    pair, cell, _ = kernel_guard(
        q64, group_bits(q64), keys[:live].cpu().numpy(), tokens, 8
    )
    return int((~cell).sum()), int((~pair & ~np.repeat(cell, 8, axis=0)).sum())
