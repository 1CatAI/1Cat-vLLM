# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMV_X: dense small-M runners on v41k dmv13_x (CPU wiring, fake ops)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41 import dense_tm

Q5K, Q6K, Q8 = 1, 2, 4


def _planes(fmt, width, k):
    tiles, groups = width // 32, k // 128
    cw = 2 if fmt == Q8 else 1
    return (
        torch.zeros(tiles * groups * 2048 * cw, dtype=torch.uint8),
        torch.zeros(tiles * groups * 512 if fmt == Q5K else 0, dtype=torch.uint8),
        torch.zeros(tiles * groups * 512, dtype=torch.uint8),
    )


def _segments(fmt, widths, k):
    return [(_planes(fmt, w, k), fmt, w) for w in widths]


class _Ops:
    def __init__(self):
        self.calls = []

    def old(self, x, codes, high, scale, out, fmts, widths, k, split, warps, *rest):
        self.calls.append(("old", dict(k=k, split=split, warps=warps, tp=rest[2])))
        for i, view in enumerate(out):
            view.fill_(-(i + 1))

    def new(self, x, codes, high, scale, out, x_off, fmt, k, split, warps, acc, *rest):
        ws, cnt, probe = rest
        self.calls.append(
            ("x", dict(x=x, x_off=list(x_off), fmt=fmt, k=k, split=split))
        )
        self.calls[-1][1].update(warps=warps, acc=acc, probe=probe, ws=ws.numel())
        # Stand-in math: segment i's output row r = sum of its K input columns.
        for i, view in enumerate(out):
            cols = x[:, x_off[i] : x_off[i] + k].float().sum(-1, keepdim=True)
            view.copy_((cols + i).expand_as(view))


@pytest.fixture
def ops(monkeypatch):
    fake = _Ops()
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace(sm70_dmv13_out=fake.old))
    monkeypatch.setattr(torch.ops, "v41k", SimpleNamespace(dmv13_x=fake.new))
    monkeypatch.setattr(dense_tm, "_TARGET_CTAS", 80)
    monkeypatch.setattr(dense_tm, "_SMALL_M_CONFIG", {})
    monkeypatch.setattr(dense_tm, "_X_CONFIG", {})
    monkeypatch.setitem(dense_tm._X_READY, "ok", True)
    return fake


@pytest.fixture
def knob(monkeypatch):
    monkeypatch.setattr(dense_tm, "_DMV_X", True)


# (k, widths) -> dmv13 grid observed in a decode profile (M=6, per rank).
SHAPES = [
    ("qkv", Q8, 5120, [1280, 512], (4, 1, 1)),
    ("q_b", Q5K, 1280, [8192], (4, 1, 1)),
    ("wo_a", Q5K, 4096, [1024], (8, 1, 2)),
    ("wo_b", Q5K, 2048, [5120], (4, 1, 1)),
    ("shared", Q5K, 5120, [576, 576], (8, 1, 2)),
    ("head", Q5K, 5120, [32320], (4, 1, 1)),
]


@pytest.mark.parametrize("role,fmt,k,widths,grid", SHAPES)
def test_default_config_matches_profiled_grids(ops, role, fmt, k, widths, grid):
    assert dense_tm.default_config(k, widths) == grid


@pytest.mark.parametrize("role,fmt,k,widths,grid", SHAPES)
def test_knob_off_keeps_dmv13(ops, role, fmt, k, widths, grid):
    runner = dense_tm.SmallM(_segments(fmt, widths, k), k, role=role)
    assert not runner.use_x
    outs = runner(torch.ones(6, k, dtype=torch.float16))
    assert [o.shape for o in outs] == [(6, w) for w in widths]
    assert ops.calls == [("old", dict(k=k, split=grid[2], warps=grid[0], tp=1))]


@pytest.mark.parametrize("role,fmt,k,widths,grid", SHAPES)
def test_knob_on_same_launch_config(ops, knob, role, fmt, k, widths, grid):
    runner = dense_tm.SmallM(_segments(fmt, widths, k), k, role=role)
    assert runner.use_x
    x = torch.ones(6, k, dtype=torch.float16)
    runner(x)
    ((kind, args),) = ops.calls
    assert kind == "x"
    assert (args["warps"], 1, args["split"]) == grid
    assert args["acc"] == 1 and args["probe"] == 0
    assert args["fmt"] == fmt and args["k"] == k
    assert args["x_off"] == [0] * len(widths)
    assert args["ws"] == sum(w // 32 for w in widths) * grid[2] * 256


def test_role_override_sets_warps_split_acc(ops, knob, monkeypatch):
    monkeypatch.setattr(dense_tm, "_X_CONFIG", {"q_b": (8, 4, 2)})
    runner = dense_tm.SmallM(_segments(Q5K, [8192], 1280), 1280, role="q_b")
    assert (runner.warps, runner.tp, runner.split, runner.acc) == (8, 1, 4, 2)
    assert runner.workspace.numel() == 256 * 4 * 256
    assert runner.counters.numel() == 256
    other = dense_tm.SmallM(_segments(Q5K, [5120], 2048), 2048, role="wo_b")
    assert (other.warps, other.split, other.acc) == (4, 1, 1)


@pytest.mark.parametrize("fmts", [[Q6K], [Q5K, Q8]], ids=["q6k", "mixed"])
def test_unsupported_formats_stay_on_dmv13(ops, knob, fmts):
    segments = [(_planes(f, 64, 1024), f, 64) for f in fmts]
    assert not dense_tm.SmallM(segments, 1024, role="extra").use_x


def test_missing_v41k_op_falls_back(ops, knob, monkeypatch):
    monkeypatch.setitem(dense_tm._X_READY, "ok", False)
    runner = dense_tm.SmallM(_segments(Q5K, [8192], 1280), 1280, role="q_b")
    assert not runner.use_x


def test_parse_x_config():
    text = " q_b=8,4,2 ; head=4,1,1;; shared=2,8,1 "
    assert dense_tm.parse_x_config(text) == {
        "q_b": (8, 4, 2),
        "head": (4, 1, 1),
        "shared": (2, 8, 1),
    }
    assert dense_tm.parse_x_config("") == {}


@pytest.mark.parametrize(
    "text",
    ["lm=4,1,1", "q_b=3,1,1", "q_b=4,9,1", "q_b=4,0,1", "q_b=4,1,3", "q_b=4,1", "q_b"],
)
def test_parse_x_config_rejects(text):
    with pytest.raises(ValueError, match="VLLM_V41_DMV_X_CONFIG"):
        dense_tm.parse_x_config(text)


def _wo_a(groups, k, merged):
    runners = [
        dense_tm.SmallM(_segments(Q5K, [1024], k), k, role="wo_a")
        for _ in range(groups)
    ]
    wo_a = SimpleNamespace(v41_group_runners=runners)
    if merged:
        wo_a.v41_group_merged = dense_tm.SmallM(
            _segments(Q5K, [1024] * groups, k),
            k,
            role="wo_a",
            config_widths=[1024],
            x_offsets=[g * k for g in range(groups)],
        )
    return wo_a


def test_merged_wo_a_is_one_block_diagonal_launch(ops, knob):
    groups, k = 2, 4096
    wo_a = _wo_a(groups, k, merged=True)
    merged = wo_a.v41_group_merged
    # The per-group config (8 warps, split 2) is kept, so each tile's FP32
    # order is the per-group launch's.
    assert (merged.warps, merged.tp, merged.split, merged.acc) == (8, 1, 2, 1)
    assert merged.workspace.numel() == 64 * 2 * 256
    output = torch.randn(6, groups, k).half()
    result = dense_tm.grouped_wo_a(wo_a, output)
    ((kind, args),) = ops.calls
    assert kind == "x"
    assert args["x_off"] == [0, k]
    assert args["x"].shape == (6, groups * k)
    assert args["x"].data_ptr() == output.data_ptr()
    expected = torch.cat(
        [
            (output[:, g].float().sum(-1, keepdim=True) + g).expand(6, 1024)
            for g in range(groups)
        ],
        dim=-1,
    ).half()
    assert torch.equal(result, expected)


def test_grouped_wo_a_without_merge_keeps_per_group_launches(ops, monkeypatch):
    monkeypatch.setattr(dense_tm, "_GROUP_VIEWS", True)
    wo_a = _wo_a(2, 4096, merged=False)
    result = dense_tm.grouped_wo_a(wo_a, torch.randn(6, 2, 4096).half())
    assert result.shape == (6, 2048)
    assert [c[0] for c in ops.calls] == ["old", "old"]
    assert all(c[1]["split"] == 2 and c[1]["warps"] == 8 for c in ops.calls)


def test_bench_shapes_and_config_match_dense_tm(ops):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "benchmarks/deepseek_v41/v41k"
    spec = importlib.util.spec_from_file_location(
        "bench_dmv_x", path / "bench_dmv_x.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    for role, fmt, k, widths, grid in SHAPES:
        b_fmt, b_k, b_widths, _ = bench.SHAPES[role]
        assert (b_fmt, b_k, b_widths) == (fmt, k, widths)
        warps, split = bench.default_config(k, widths)
        assert dense_tm.default_config(k, widths) == (warps, 1, split)
