# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The tensor-core index scorer on the HOST (no card): v41idx3.cu's own bodies, with each mma replaced by the
card's measured layout and adder, against ../v41idx/v41idx.cu's own bodies (#1372, the reference).

What decides each expected value:
- the adder: the bits a V100 recorded (mma_probe_v100.json.gz: 6,500 single instructions, 32 chains);
- the scores: the reference scorer's own host bodies (torch.ops.v41idx.index_scores_tiled of the same CPU
  library), bit for bit, columns past the live end included;
- the two counts (cells the first look did not pass; heads of those recomputed by the serving chain): arithmetic
  on operands built for it (v41idx3_cases.boundary; keys with scale bytes out of range; a row with a subnormal),
  and for random operands an independent statement of the guard's rule (guard_rule.py, numpy);
- each mutant (the guard's limit raised, the scale bytes' range unchecked, the fallback's chain backwards) must
  be caught by the test the plain library passes.

Run (no card; nvcc and torch are needed to build the CPU library):
    ./build_cpu_ops.sh python DIR/v41idx3_cpu_ops.so
    for m in limit range fallback; do
      ./build_cpu_ops.sh python DIR/v41idx3_cpu_ops_$m.so -DV41IDX3_MUTANT_$(echo $m | tr a-z A-Z); done
    V41IDX3_CPU_OPS_DIR=DIR python -m pytest -q test_v41idx3_host.py
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import guard_rule  # noqa: E402
import v41idx3_cases as cases  # noqa: E402

DIR = os.environ["V41IDX3_CPU_OPS_DIR"]
PROBE = os.path.join(HERE, "mma_probe_v100.json.gz")
LIB = os.path.join(DIR, "v41idx3_cpu_ops.so")
MUTANTS = {
    name: os.path.join(DIR, f"v41idx3_cpu_ops_{name}.so")
    for name in ("limit", "range", "fallback")
}
MUTANT = os.environ.get(
    "V41IDX3_TEST_MUTANT"
)  # set only in the child process that loads a mutant

torch.ops.load_library(MUTANTS[MUTANT] if MUTANT else LIB)
REF = torch.ops.v41idx.index_scores_tiled
NEW = torch.ops.v41idx3.index_scores_tc

plain_only = pytest.mark.skipif(MUTANT is not None, reason="the plain library's test")


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def guard_expected(
    q: torch.Tensor, keys: torch.Tensor, tokens: int, end: int
) -> tuple[int, int]:
    """(cells the first look must not pass, heads of those the second look must not pass either), from the rule
    stated a second time (numpy), over the live keys (the end clamped to the capacity)."""
    return guard_rule.expected_counts(q, keys, tokens, min(end, keys.shape[0]))


@plain_only
def test_the_host_adder_reproduces_every_recorded_bit_of_the_card():
    def half(bits):
        return torch.tensor(
            np.array(bits, dtype=np.uint16).view(np.float16).astype(np.float32)
        )

    def f32(bits):
        return torch.tensor(np.array(bits, dtype=np.uint32).view(np.float32))

    d = json.load(gzip.open(PROBE))
    rows = [
        (c, a, b, got)
        for c, a, b, got in d["single"]
        if np.isfinite(np.array([got], dtype=np.uint32).view(np.float32))[0]
    ]
    assert len(rows) == 6500
    mine = torch.ops.v41idx3.tc_cell(
        f32([r[0] for r in rows]),
        half([r[1] for r in rows]),
        half([r[2] for r in rows]),
    )
    assert (
        int((mine != f32([r[3] for r in rows])).sum()) == 0
    )  # +0 and -0 are one value to the model
    chains = 0
    for chain in d["chains"]:
        for steps, got in zip(chain["operands"], chain["got"]):
            acc = torch.zeros(1)
            for word in steps:
                h = half([int(word[4 * i : 4 * i + 4], 16) for i in range(8)])
                acc = torch.ops.v41idx3.tc_cell(
                    acc, h[:4].view(1, 4).contiguous(), h[4:].view(1, 4).contiguous()
                )
            assert float(acc[0]) == float(f32([got])[0])
            chains += 1
    assert chains == 32


MATRIX = [
    (kind, tokens, ratio, variant)
    for kind in cases.CLASSES
    for tokens in (1, 3, 6, 8)
    for ratio in (1, 4)
    for variant in (0, 2, 4)
]


@plain_only
@pytest.mark.parametrize("kind,tokens,ratio,variant", MATRIX)
def test_scores_are_the_serving_scorer_s_bit_for_bit(kind, tokens, ratio, variant):
    """Guard on (0), every cell through the fallback (2), three CTAs an SM (4)."""
    capacity = 8192
    q, w, keys = cases.operands(
        kind, tokens, capacity, gen(10 * cases.CLASSES.index(kind) + tokens)
    )
    for end in (0, 1, 63, 64, 65, 4097 // ratio, capacity // ratio, capacity + 70):
        differ, cells, heads, problems = cases.both(
            REF, NEW, q, w, keys, end, ratio, variant
        )
        assert not problems, (
            problems
        )  # the premise, and the call's own count of its calls and cells
        assert differ == 0, (kind, tokens, ratio, variant, end)
        live = min(end, capacity)
        expected = (
            (tokens * live, 8 * tokens * live)
            if variant & 2
            else guard_expected(q, keys, tokens, end)
        )
        assert (cells, heads) == expected, (kind, tokens, ratio, variant, end)
        if kind == "close" and not variant & 2:
            assert cells == 0  # by arithmetic: A <= 5 and B <= 2 in every cell


@plain_only
def test_a_long_block_walk_with_both_kinds_of_cell():
    """62,000 live keys of the close class, every fifth with scale bytes 122, 127, 127, 132 (in range, B = 10;
    a close row has A >= 3, in fact 3 to 5: it cannot pass the first look): every CTA walks several blocks, and
    each block has cells that pass and cells that do not. By arithmetic 6 tokens x 12,400 keys get the second
    look. A head passes it when its row's group 3 is scaled 4 times below its group 0 (about one row in nine):
    how many do is the numpy rule's count, and both kinds must be there."""
    q, w, keys = cases.operands("close", 6, 62000, gen(7))
    keys = keys.clone()
    keys[0::5, 64:] = torch.tensor([122, 127, 127, 132], dtype=torch.uint8)
    differ, cells, heads, _ = cases.both(REF, NEW, q, w, keys, 62000, 1, 0)
    assert differ == 0
    assert (cells, heads) == guard_expected(q, keys, 6, 62000)
    assert cells == 6 * 12400 and 0 < heads < 8 * cells


@plain_only
@pytest.mark.parametrize(
    "a,b,cells_all,heads_all",
    [
        (7, 4, False, False),
        (11, 0, False, False),
        (3, 8, False, False),
        (12, 0, True, True),
        (7, 5, True, False),
        (3, 9, True, False),
    ],
)
def test_the_guard_s_limit_is_eleven(a, b, cells_all, heads_all):
    """Operands built so that A + B is the same in every cell (v41idx3_cases.boundary). At 11 the first look
    passes. At 12 it does not; then the second look: with A = 12 a row alone is over the limit (every head is the
    serving chain's); with A = 7 or 3 the 12 comes from the key's wide byte, which sits on a group where the rows
    are zero, so every head passes the second look and none is recomputed."""
    tokens, end = 6, 3000
    q, w, keys = cases.boundary(tokens, 4096, a, b, gen(100 + a + b))
    differ, cells, heads, _ = cases.both(REF, NEW, q, w, keys, end, 1, 0)
    assert differ == 0
    assert cells == (tokens * end if cells_all else 0)
    assert heads == (8 * tokens * end if heads_all else 0)


@plain_only
def test_one_row_over_the_limit_costs_its_own_head_only():
    """Every row has A = 7 but head 3 of token 4, which has A = 12 (v41idx3_cases.boundary's group 1 element made
    2^-10); every key has B = 0. Token 4's cells do not pass the first look (its worst row decides); of their
    eight heads one is recomputed. The other tokens' cells pass."""
    tokens, end = 6, 3000
    q, w, keys = cases.boundary(tokens, 4096, 7, 0, gen(77))
    q = q.clone()
    q[4 * 8 + 3, 32] = 2.0 ** (2 - 12)
    differ, cells, heads, _ = cases.both(REF, NEW, q, w, keys, end, 1, 0)
    assert differ == 0
    assert (cells, heads) == (end, end)


def out_of_range_case():
    """The close class; of 3,000 live keys, key 0, 3, 6, ... has all four scale bytes 113 (its smallest values
    are subnormal fp16) and key 1, 4, 7, ... all four 141 (its largest overflow fp16): 2,000 keys out of the range,
    each with B = 0, so only the range check can keep them from passing."""
    tokens, end = 6, 3000
    q, w, keys = cases.operands("close", tokens, 4096, gen(31))
    keys = keys.clone()
    keys[0:end:3, 64:] = 113
    keys[1:end:3, 64:] = 141
    return q, w, keys, tokens, end, 2000


def test_keys_outside_the_scale_range_take_the_fallback():
    q, w, keys, tokens, end, marked = out_of_range_case()
    differ, cells, heads, _ = cases.both(REF, NEW, q, w, keys, end, 1, 0)
    if MUTANT == "range":
        assert cells != tokens * marked, "the mutant (no range check) was not caught"
        return
    assert differ == 0
    assert (cells, heads) == (tokens * marked, 8 * tokens * marked)


@plain_only
@pytest.mark.parametrize("bits", [0x0001, 0x03FF, 0x7C00, 0x7E00])
def test_a_row_with_a_subnormal_an_infinity_or_a_nan_is_the_serving_chain_s(bits):
    """One element of head 5 of token 2 is a subnormal (0x0001, 0x03FF), an infinity or a NaN: none of that
    token's cells passes the first look, and in each that one head is recomputed (the other seven pass the second
    look: close class, at most 5 + 2). No other token's cell is touched."""
    tokens, end = 6, 2000
    q, w, keys = cases.operands("close", tokens, 4096, gen(41))
    q = q.clone()
    q.view(torch.int16)[2 * 8 + 5, 77] = bits
    differ, cells, heads, _ = cases.both(REF, NEW, q, w, keys, end, 1, 0)
    assert differ == 0
    assert (cells, heads) == (end, end)


def wide_case():
    q, w, keys = cases.operands("plain", 6, 20000, gen(51))
    return q, w, keys, 20000


def test_cells_whose_chains_can_round_are_not_passed():
    """fp16 rows of any mantissa: the two chains' fp32 values differ in about 44 % of the dot products (measured on such
    operands). The plain library is bitwise all the same (those heads are the serving chain's); the
    mutant with the limit raised passes them and must differ."""
    q, w, keys, end = wide_case()
    differ, cells, heads, _ = cases.both(REF, NEW, q, w, keys, end, 1, 0)
    if MUTANT == "limit":
        assert differ > 0, "the mutant (limit 40) was not caught"
        return
    assert differ == 0 and (cells, heads) == guard_expected(q, keys, 6, end)


def test_the_fallback_is_the_serving_chain():
    q, w, keys, end = wide_case()
    differ, cells, heads, _ = cases.both(REF, NEW, q, w, keys, end, 1, 2)
    if MUTANT == "fallback":
        assert differ > 0, "the mutant (the chain backwards) was not caught"
        return
    assert differ == 0 and (cells, heads) == (6 * end, 48 * end)


@plain_only
@pytest.mark.parametrize(
    "name,test",
    [
        ("limit", "test_cells_whose_chains_can_round_are_not_passed"),
        ("range", "test_keys_outside_the_scale_range_take_the_fallback"),
        ("fallback", "test_the_fallback_is_the_serving_chain"),
    ],
)
def test_each_mutant_is_caught(name, test):
    """The same test in a child process that loads the mutant library: there it takes its mutant branch, which
    passes only if the difference was seen. In this process the same tests run on the plain library and find
    none."""
    env = dict(os.environ, V41IDX3_TEST_MUTANT=name)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"{__file__}::{test}",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert done.returncode == 0 and "1 passed" in done.stdout, (
        done.stdout[-1500:] + done.stderr[-500:]
    )


@plain_only
def test_what_the_op_refuses():
    q, w, keys = cases.operands("close", 2, 256, gen(61))
    out = cases.poisoned(2, 256, "cpu")
    positions = torch.full((2,), 99, dtype=torch.int64)
    counters = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="8 index heads per rank only"):
        NEW(out, q[:8], w[:8], keys, positions, 4, 1, 0, counters)
    with pytest.raises(RuntimeError, match="not NOGUARD with FORCE"):
        NEW(out, q, w, keys, positions, 8, 1, 3, counters)
    with pytest.raises(RuntimeError, match="counters"):
        NEW(out, q, w, keys, positions, 8, 1, 0, torch.zeros(4, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="counters"):
        NEW(out, q, w, keys, positions, 8, 1, 0, torch.zeros(3, dtype=torch.int64))
    with pytest.raises(RuntimeError, match="output width must cover key capacity"):
        NEW(cases.poisoned(2, 255, "cpu"), q, w, keys, positions, 8, 1, 0, counters)


@plain_only
@pytest.mark.parametrize(
    "capacity,end", [(4096, 4097), (4096, 5000), (4000, 4096), (100, 1 << 40)]
)
def test_a_stale_position_stops_at_the_key_capacity(capacity, end):
    """A position past the allocation (a stale or padded graph row): both scorers read and write only the
    capacity's keys (v41idx.cu's live_end since #1372), and the counters count capacity cells. The output is one
    column wider than the capacity, so a write past it would show as a changed poison cell."""
    tokens = 6
    q, w, keys = cases.operands("close", tokens, capacity, gen(capacity + 3))
    differ, cells, heads, problems = cases.both(REF, NEW, q, w, keys, end, 1, 0)
    assert not problems, problems
    assert differ == 0 and (cells, heads) == (0, 0)
    counters = torch.zeros(4, dtype=torch.int64)
    out = cases.poisoned(tokens, capacity + 1, "cpu")
    NEW(
        out,
        q,
        w,
        keys,
        torch.full((tokens,), end - 1, dtype=torch.int64),
        8,
        1,
        0,
        counters,
    )
    assert counters.tolist() == [0, 0, 1, tokens * capacity]
    assert bool((out.view(torch.int16)[:, capacity] == cases.POISON).all())


@plain_only
def test_the_counters_add_up_over_calls():
    """One counter tensor over three calls (100 live keys, none, 4,096): calls with a live key 2, cells
    2 * 100 + 2 * 4096; the close class: nothing over the limit."""
    q, w, keys = cases.operands("close", 2, 4096, gen(62))
    counters = torch.zeros(4, dtype=torch.int64)
    for end in (100, 0, 4096):
        NEW(
            cases.poisoned(2, 4099, "cpu"),
            q,
            w,
            keys,
            torch.full((2,), end - 1, dtype=torch.int64),
            8,
            1,
            0,
            counters,
        )
    assert counters.tolist() == [0, 0, 2, 2 * 100 + 2 * 4096]
