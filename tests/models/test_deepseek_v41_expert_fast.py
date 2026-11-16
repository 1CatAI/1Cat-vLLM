# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_EXPERT_FAST: registration and dispatch of the fast expert ops (CPU).

The functions _v41_experts_rows, _v41_experts_rows_fast and _expert_fast_book
are compiled from gguf_v41_moe.py and run against recorded fake ops: with the
knob off gate/up and down are today's gate_up7 (gate_up7p after q8_plan with
VLLM_V41_PLAN_FUSE) and down_rows calls; with it on gate/up is gate_up_fast
with the same arguments, fast_table(source type) and build_plan exactly when
gate_up7 would build the plan, and down is down_rows_fast with the same
gathered buffer, stride, routes, row base and plan, classes 8 for dn_warps 8
and 4 otherwise; rows != 4 and a library without the ops raise.

Scope: Python dispatch only. Kernel equality is checked on the GPU by
benchmarks/deepseek_v41/expert_fast_check.py (and on the host by the C++
emulator in benchmarks/deepseek_v41/v41k/emu).

    python tests/models/test_deepseek_v41_expert_fast.py   # or pytest
"""

import ast
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CU = ROOT / "benchmarks/deepseek_v41/v41k/v41k.cu"
MOE = ROOT / "vllm/model_executor/layers/quantization/gguf_v41_moe.py"
FUNCS = ("_v41_experts_rows", "_v41_experts_rows_fast", "_expert_fast_book")


class FakeTensor:
    """Named stand-in; slices compose, so a name is the base buffer and the
    absolute byte range (own[a:b][c:d] is own[a + c:a + d])."""

    def __init__(self, name, shape=(1,), base=None, start=0, stop=None):
        self.name, self.shape, self.device = name, shape, "dev"
        self.base, self.start, self.stop = base or name, start, stop

    def __getitem__(self, key):
        lo = self.start + (key.start or 0)
        hi = self.stop if key.stop is None else self.start + key.stop
        return FakeTensor(f"{self.base}[{lo}:{hi}]", None, self.base, lo, hi)

    def view(self, *args):
        return self

    def contiguous(self):
        return self


def _module(fast, ops, peer, plan_fuse=False):
    tree = ast.parse(MOE.read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS]
    assert sorted(n.name for n in fns) == sorted(FUNCS)
    calls, logs = [], []

    def op(name):
        def call(*args):
            calls.append((name, args))
            if name == "fast_table":
                return FakeTensor(f"fbook{args[0]}")

        return call

    def gate_up7(*args):
        calls.append(("gate_up7", args))

    def gate_up7p(*args):
        calls.append(("gate_up7p", args))

    def q8_and_plan(xq, xd, x, ids, plan):  # the contract of _q8_and_plan
        if plan_fuse:
            calls.append(("q8_plan", (xq, xd, x, ids, plan)))
            return gate_up7p
        calls.append(("q8_aligned", (xq, xd, x)))
        return gate_up7

    nccl = types.SimpleNamespace(all_gather=lambda dst, src: calls.append(("ag",)))
    torch = types.SimpleNamespace(
        ops=types.SimpleNamespace(
            v41k=types.SimpleNamespace(**{n: op(n) for n in ops})
        ),
        empty=lambda shape, **k: FakeTensor("empty", shape),
        empty_like=lambda x: FakeTensor("result"),
        int8="int8",
        float32="float32",
        uint8="uint8",
    )
    own = FakeTensor("own")
    ns = {
        "torch": torch,
        "_EXPERT_FAST": fast,
        "_PLAN_FUSE": plan_fuse,
        "_EM": {"books": {}},
        "default_v41k_library": lambda: "/x/v41k.so",
        "logger": types.SimpleNamespace(
            info_once=lambda msg, *args, scope="local": logs.append((msg, scope))
        ),
        "_plan": lambda device: FakeTensor("plan"),
        "_q8_and_plan": q8_and_plan,
        "_PEER": {"dev": (2, (11, 12, 13, 14), own, 7)} if peer else {},
        "_rows_context": lambda device: (1, nccl),
        "_book": lambda kind, device: FakeTensor(f"book{kind}"),
        "ROWS_FLAGS": 99,
    }
    exec(compile(ast.Module(fns, []), str(MOE), "exec"), ns)
    return ns["_v41_experts_rows"], calls, logs


def _run(fn, m=3, rows=4, warps=8, down_type=29):
    x = FakeTensor("x", (m, 5120))
    down = FakeTensor("down", (384, 1280, 504))
    ids = FakeTensor("ids")
    return fn(x, ids, FakeTensor("p"), "gate", "up", down, 16, down_type, rows, warps)


OLD_OPS = ["down_rows", "peer_gather", "table"]
NEW_OPS = OLD_OPS + ["fast_table", "gate_up_fast", "down_rows_fast"]
# the v41k.so of 05dd566: down_rows_fast but no gate_up_fast
DOWN_ONLY_OPS = OLD_OPS + ["fast_table", "down_rows_fast"]
ROLE = {
    "gate_up7": "gate/up",
    "gate_up7p": "gate/up",
    "gate_up_fast": "gate/up",
    "down_rows": "down",
    "down_rows_fast": "down",
}


def _down(calls):
    return [c for c in calls if c[0] in ("down_rows", "down_rows_fast")]


def _gate_up(calls):
    return [c for c in calls if c[0] in ("gate_up7", "gate_up7p", "gate_up_fast")]


def _names(args):
    return [getattr(a, "name", a) for a in args]


def _sequence(calls):
    return [ROLE.get(c[0], c[0]) for c in calls if c[0] != "fast_table"]


@pytest.mark.parametrize("peer", [True, False])
@pytest.mark.parametrize("plan_fuse", [True, False])
def test_knob_off_is_todays_gate_up_and_down_rows(peer, plan_fuse):
    fn, calls, logs = _module(False, NEW_OPS, peer, plan_fuse)
    _run(fn)
    (gu,) = _gate_up(calls)
    assert gu[0] == ("gate_up7p" if plan_fuse else "gate_up7")
    assert gu[1][9].name == "book16" and gu[1][10].name == "plan"
    (down,) = _down(calls)
    assert down[0] == "down_rows" and down[1][8:10] == (4, 8)
    assert down[1][7].name == "book29" and not logs
    assert not any(
        c[0] in ("fast_table", "gate_up_fast", "down_rows_fast") for c in calls
    )


@pytest.mark.parametrize("peer", [True, False])
@pytest.mark.parametrize("plan_fuse", [True, False])
def test_knob_on_gate_up_matches_gate_up7_arguments(peer, plan_fuse):
    m = 4
    fn_old, calls_old, _ = _module(False, NEW_OPS, peer, plan_fuse)
    fn_new, calls_new, _ = _module(True, NEW_OPS, peer, plan_fuse)
    _run(fn_old, m=m)
    _run(fn_new, m=m)
    (old,) = _gate_up(calls_old)
    (new,) = _gate_up(calls_new)
    assert new[0] == "gate_up_fast" and len(new[1]) == 12
    # xq, xd, hq, hd, ids32, prob, gate, up, gtype: as gate_up7 / gate_up7p
    assert _names(new[1][:9]) == _names(old[1][:9])
    assert new[1][8] == 16 and new[1][6:8] == ("gate", "up")
    assert new[1][9].name == "fbook16" and old[1][9].name == "book16"
    assert new[1][10].name == old[1][10].name == "plan"
    # build_plan: gate_up7 builds the plan (moe_plan), gate_up7p does not
    assert new[1][11] is (not plan_fuse)
    q8 = "q8_plan" if plan_fuse else "q8_aligned"
    gather = "peer_gather" if peer else "ag"
    assert (
        _sequence(calls_new) == _sequence(calls_old) == [q8, "gate/up", gather, "down"]
    )


@pytest.mark.parametrize("peer", [True, False])
@pytest.mark.parametrize("warps,classes", [(8, 8), (4, 4), (16, 4)])
def test_knob_on_matches_down_rows_arguments(peer, warps, classes):
    m = 5
    fn_old, calls_old, _ = _module(False, NEW_OPS, peer)
    fn_new, calls_new, logs = _module(True, NEW_OPS, peer)
    _run(fn_old, m=m, warps=warps)
    _run(fn_new, m=m, warps=warps)
    (old,) = _down(calls_old)
    (new,) = _down(calls_new)
    assert new[0] == "down_rows_fast"
    stride = (m * 6 * 648 + 255) // 256 * 256
    rank = 2 if peer else 1
    # out, gathered, stride, routes, down, dtype, row_base: as down_rows
    assert [getattr(a, "name", a) for a in new[1][:7]] == [
        getattr(a, "name", a) for a in old[1][:7]
    ]
    assert new[1][2:4] == (stride, m * 6) and new[1][6] == rank * 1280
    assert new[1][1].name == ("own" if peer else "empty")
    assert new[1][7].name == "fbook29" and new[1][8] == classes
    assert new[1][9].name == old[1][10].name == "plan"
    # the same launches in the same order
    assert _sequence(calls_new) == _sequence(calls_old)
    assert len(logs) == 2 and "VLLM_V41_EXPERT_FAST=1" in logs[0][0]
    # every PP / TP worker logs its activation line, not only rank 0
    assert all(scope == "process" for _, scope in logs)


def test_fast_table_built_once_per_type():
    fn, calls, logs = _module(True, NEW_OPS, True)
    for down_type in (29, 16, 29, 16, 17):
        _run(fn, down_type=down_type)
    built = [c[1][0] for c in calls if c[0] == "fast_table"]
    # gate/up source type 16 first, then the down types as they appear
    assert built == [16, 29, 17] and len(logs) == 3  # info_once dedups in vLLM


def test_knob_on_other_row_tile_raises():
    fn, calls, _ = _module(True, NEW_OPS, True)
    with pytest.raises(ValueError, match="rows=4"):
        _run(fn, rows=8)
    assert not calls


@pytest.mark.parametrize("ops", [OLD_OPS, DOWN_ONLY_OPS])
def test_knob_on_with_an_old_library_raises(ops):
    fn, calls, _ = _module(True, ops, True)
    with pytest.raises(RuntimeError, match="VLLM_V41_EXPERT_FAST=1 needs") as err:
        _run(fn)
    assert "gate_up_fast" in str(err.value) and "/x/v41k.so" in str(err.value)
    assert not calls


def test_registration():
    src = CU.read_text()
    assert '#include "v41k_fast.cuh"' in src
    assert '#include "v41k_fast_gu.cuh"' in src
    assert '"gate_up_fast(Tensor xq, Tensor xd, Tensor(a!) hq, Tensor(b!) hd, "' in src
    assert '"Tensor fbook, Tensor(c!) plan, bool build_plan) -> ()",' in src
    assert '"fast_table(int type, Device device) -> Tensor"' in src
    assert '"down_rows_fast(Tensor(a!) out, Tensor gathered, int rank_stride, "' in src
    envs = (ROOT / "vllm/envs.py").read_text()
    assert '    VLLM_V41_EXPERT_FAST: str = "0"\n' in envs
    assert 'lambda: os.getenv("VLLM_V41_EXPERT_FAST", "0")' in envs
    assert '_EXPERT_FAST = envs.VLLM_V41_EXPERT_FAST == "1"' in MOE.read_text()
    ast.parse((ROOT / "benchmarks/deepseek_v41/expert_fast_check.py").read_text())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
