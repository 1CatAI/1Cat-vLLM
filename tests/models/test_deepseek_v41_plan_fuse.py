# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_PLAN_FUSE: the register plan must write moe_plan's plan (CPU).

``plan_warp`` (v41k.cu, run inside the q8_plan launch) replaces the
shared-memory ``moe_plan`` kernel. Both are modelled here lane by lane as
written in CUDA and must write the same entries with the same values for any
ids (repeats across rows, dropped routes id < 0, 0..36 routes). The dispatch
test runs ``_q8_and_plan`` against recorded fake ops.

Scope: a Python model of the two kernels, not the GPU. The GPU check is
benchmarks/deepseek_v41/plan_fuse_check.py.

    python tests/models/test_deepseek_v41_plan_fuse.py   # or pytest
"""

import ast
import random
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CU = ROOT / "benchmarks/deepseek_v41/v41k/v41k.cu"
MOE = ROOT / "vllm/model_executor/layers/quantization/gguf_v41_moe.py"


def _popc(x):
    return bin(x & 0xFFFFFFFF).count("1")


def moe_plan(ids, routes):
    """v41k::moe_plan, one warp, as written (shared s_ids, serial scans)."""
    plan = {}
    s_ids = [ids[r] if r < routes else -1 for r in range(64)]
    e = {(lane, h): s_ids[lane + 32 * h] for lane in range(32) for h in range(2)}
    first = {}
    for lane in range(32):
        for h in range(2):
            r = lane + 32 * h
            f = r < routes and e[lane, h] >= 0
            q = 0
            while q < r and f:
                f = s_ids[q] != e[lane, h]
                q += 1
            first[lane, h] = f
    b0 = sum(1 << lane for lane in range(32) if first[lane, 0])
    b1 = sum(1 << lane for lane in range(32) if first[lane, 1])
    s_first_u = {}
    for lane in range(32):
        below = (1 << lane) - 1
        for h in range(2):
            r = lane + 32 * h
            u = _popc(b0 & below) if h == 0 else _popc(b0) + _popc(b1 & below)
            if first[lane, h]:
                s_first_u[r] = u
    for lane in range(32):
        for h in range(2):
            r = lane + 32 * h
            if r >= routes or e[lane, h] < 0:
                continue
            q0 = 0
            while s_ids[q0] != e[lane, h]:
                q0 += 1
            c = sum(s_ids[q] == e[lane, h] for q in range(q0, r))
            u = s_first_u[q0]
            if c < 6:
                plan[73 + 6 * u + c] = r
            if first[lane, h]:
                cnt = sum(s_ids[q] == e[lane, h] for q in range(r, routes))
                plan[1 + u] = e[lane, h]
                plan[37 + u] = min(cnt, 6)
    plan[0] = _popc(b0) + _popc(b1)
    return plan


def plan_warp(ids, routes):
    """v41k::plan_warp, one warp, as written (registers and shuffles)."""
    plan = {}
    lanes = range(32)
    e = {
        (ln, h): ids[ln + 32 * h] if ln + 32 * h < routes else -1
        for ln in lanes
        for h in range(2)
    }
    live = {k: k[0] + 32 * k[1] < routes and e[k] >= 0 for k in e}
    c = dict.fromkeys(e, 0)
    cnt = dict.fromkeys(e, 0)
    q0 = {k: k[0] + 32 * k[1] for k in e}
    for q in range(routes):
        src = {ln: e[ln, 0] if q < 32 else e[ln, 1] for ln in lanes}
        v = src[q & 31]  # __shfl_sync(src, q & 31): same value in every lane
        for ln, h in e:
            r = ln + 32 * h
            if live[ln, h] and v == e[ln, h]:
                if q < r:
                    c[ln, h] += 1
                    q0[ln, h] = min(q0[ln, h], q)
                else:
                    cnt[ln, h] += 1
    b0 = sum(1 << ln for ln in lanes if live[ln, 0] and c[ln, 0] == 0)
    b1 = sum(1 << ln for ln in lanes if live[ln, 1] and c[ln, 1] == 0)
    own0 = {ln: _popc(b0 & ((1 << ln) - 1)) for ln in lanes}
    own1 = {ln: _popc(b0) + _popc(b1 & ((1 << ln) - 1)) for ln in lanes}
    for ln, h in e:
        u0, u1 = own0[q0[ln, h] & 31], own1[q0[ln, h] & 31]
        u, r = (u0 if q0[ln, h] < 32 else u1), ln + 32 * h
        if not live[ln, h]:
            continue
        if c[ln, h] < 6:
            plan[73 + 6 * u + c[ln, h]] = r
        if c[ln, h] == 0:
            plan[1 + u] = e[ln, h]
            plan[37 + u] = min(cnt[ln, h], 6)
    plan[0] = _popc(b0) + _popc(b1)
    return plan


def _cases():
    rng = random.Random(0)
    yield [], 0
    yield [-1] * 36, 36
    yield [5] * 36, 36  # one expert: 36 routes, only 6 recorded
    yield list(range(6)) * 6, 36
    for _ in range(400):
        rows = rng.randint(1, 6)
        pool = rng.choice([6, 7, 8, 12, 24, 96, 384])
        experts = rng.sample(range(384), pool)
        ids = []
        for _ in range(rows):
            row = rng.sample(experts, 6)
            keep = rng.choice([6, 6, 6, 5, 3, 1])
            ids += row[:keep] + [-1] * (6 - keep)
        yield ids, len(ids)


@pytest.mark.parametrize("case", list(_cases()), ids=lambda c: f"r{c[1]}")
def test_register_plan_writes_moe_plan_entries(case):
    ids, routes = case
    want = moe_plan(ids, routes)
    got = plan_warp(ids, routes)
    assert got == want
    assert all(0 <= k < 73 + 216 for k in got)


def test_kernel_sources_and_ops():
    src = CU.read_text()
    assert "void plan_warp(int* __restrict__ plan," in src
    assert (
        "if (blockIdx.x == 0 && threadIdx.x < 32) plan_warp(plan, ids, routes);" in src
    )
    assert "q8_aligned_row(qs, ds, x, K, blockIdx.y - 1);" in src
    assert "q8_aligned_row(qs, ds, x, K, blockIdx.y);" in src
    # gate_up7 keeps its own moe_plan launch; gate_up7p skips it.
    assert (
        "gate_up7_impl(xq, xd, hq, hd, ids32, prob, gate, up, gtype, gbook, plan,\n"
        "                true);" in src
    )
    assert (
        "gate_up7_impl(xq, xd, hq, hd, ids32, prob, gate, up, gtype, gbook, plan,\n"
        "                false);" in src
    )
    assert '"q8_plan(Tensor(a!) qs, Tensor(b!) ds, Tensor x, Tensor ids32, "' in src
    assert (
        '"gate_up7p(Tensor xq, Tensor xd, Tensor(a!) hq, Tensor(b!) hd, Tensor "' in src
    )
    envs = (ROOT / "vllm/envs.py").read_text()
    assert '    VLLM_V41_PLAN_FUSE: str = "0"\n' in envs
    assert 'lambda: os.getenv("VLLM_V41_PLAN_FUSE", "0")' in envs
    assert '_PLAN_FUSE = envs.VLLM_V41_PLAN_FUSE == "1"' in MOE.read_text()
    ast.parse((ROOT / "benchmarks/deepseek_v41/plan_fuse_check.py").read_text())


def _dispatch(fuse, ops):
    tree = ast.parse(MOE.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_q8_and_plan"
    )
    calls = []

    def op(name):
        def call(*a):
            calls.append(name)

        call.name = name
        return call

    v41k = types.SimpleNamespace(**{n: op(n) for n in ops})
    ns = {
        "torch": types.SimpleNamespace(ops=types.SimpleNamespace(v41k=v41k)),
        "_PLAN_FUSE": fuse,
        "default_v41k_library": lambda: "/x/v41k.so",
        "_PLAN_FUSE_LOGGED": False,
        "logger": types.SimpleNamespace(
            info=lambda msg, *a: calls.append(("log", msg))
        ),
    }
    exec(compile(ast.Module([fn], []), str(MOE), "exec"), ns)
    return ns["_q8_and_plan"], calls


def test_dispatch_off_is_todays_ops():
    fn, calls = _dispatch(False, ["q8_aligned", "gate_up7"])
    nxt = fn(1, 2, 3, 4, 5)
    assert calls == ["q8_aligned"] and nxt.name == "gate_up7"


def test_dispatch_on_uses_fused_ops():
    fn, calls = _dispatch(True, ["q8_aligned", "gate_up7", "q8_plan", "gate_up7p"])
    nxt = fn(1, 2, 3, 4, 5)
    fn(1, 2, 3, 4, 5)
    log = ("log", "VLLM_V41_PLAN_FUSE=1: expert plan built in q8_plan, gate_up7p")
    # the fused ops on every call; the activation logged once
    assert calls == [log, "q8_plan", "q8_plan"] and nxt.name == "gate_up7p"


def test_dispatch_on_with_an_old_library_raises():
    fn, _ = _dispatch(True, ["q8_aligned", "gate_up7"])
    with pytest.raises(RuntimeError, match="VLLM_V41_PLAN_FUSE=1 needs"):
        fn(1, 2, 3, 4, 5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
