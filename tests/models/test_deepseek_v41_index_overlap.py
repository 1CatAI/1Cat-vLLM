# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 indexer: L4 side-stream overlap of the per-tile TP reduction + top-k.

The functions come from ``attention_ops.py`` source (the module itself needs
a CUDA build to import). CPU checks:

- the knob is off by default; off, select_compressed_positions never calls
  _select_compressed_positions_overlap (its own body runs), on it hands
  every argument over;
- the gate refuses CPU tensors, a single tile and stream capture (without
  touching CUDA for a CPU tensor);
- fork and join: the side stream waits for the main stream, the tile's head
  sums are recorded on the side stream before it uses them, the reduction
  and top-k run inside the side stream, and the main stream waits for the
  side stream once, after the last tile;
- with the overlap on (fake streams) every tile's reduction and top-k run
  on the side stream and the result equals the overlap off, and both equal
  the code before L3/L4 (kept below), for the torch top-k and the fused
  top-k call; with the knob on and CPU tensors (real gate) too;
- the overlap copy rejects the inputs the original rejects and tiles by
  the same score budget.

    python tests/models/test_deepseek_v41_index_overlap.py   # or pytest
"""

import ast
import contextlib
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm" / "models" / "deepseek_v41" / "attention_ops.py"
FUNCTIONS = (
    "_index_overlap",
    "_index_side_stream",
    "_on_side_stream",
    "_join_side_stream",
    "_index_tile_topk",
    "select_compressed_positions",
    "_select_compressed_positions_overlap",
)
HEADS = 16


def _load(**names):
    tree = ast.parse(SOURCE.read_text())
    nodes = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS
    ]
    assert len(nodes) == len(FUNCTIONS)
    namespace = dict(
        torch=torch,
        contextlib=contextlib,
        _INDEX_TILE_BYTES=0,  # 32-row tiles
        _FUSED_TOPK=False,
        _INDEX_ONE_PASS=False,
        _INDEX_OVERLAP=False,
        _INDEX_SIDE={},
        relu_mul_head_sum=None,
    )
    namespace.update(names)
    exec(compile(ast.Module(nodes, []), str(SOURCE), "exec"), namespace)
    return namespace


def reference_select(q, weights, keys, positions, valid, ratio, all_reduce, top_k):
    """select_compressed_positions as it was before L3 and L4 (899e9ed; the
    same at pp2-int 251fcc4), torch top-k branch, 32-row tiles."""
    width = min(top_k, keys.shape[0])
    result = torch.empty(q.shape[0], width, device=q.device, dtype=torch.int32)
    column = torch.arange(keys.shape[0], device=q.device)[None]
    tile = 32
    for first in range(0, q.shape[0], tile):
        last = min(first + tile, q.shape[0])
        heads = q.shape[1]
        scores = torch.mm(q[first:last].reshape(-1, 128), keys.T).view(
            last - first, heads, -1
        )
        scores.relu_().mul_(weights[first:last, :, None])
        scores = all_reduce(scores.sum(1))
        length = ((positions[first:last] + 1) // ratio)[:, None]
        scores.masked_fill_(
            (column >= length) | ~valid[first:last, None], float("-inf")
        )
        indices = scores.topk(width, dim=-1, sorted=False).indices.sort(dim=-1).values
        indices = torch.where(indices < length, indices, -1)
        indices = torch.where(valid[first:last, None], indices, -1)
        result[first:last].copy_(indices)
    return result


def _inputs(tokens=70, keys_n=96, seed=5):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(tokens, HEADS, 128, generator=g) / 8).half()
    weights = (torch.randn(tokens, HEADS, generator=g) / 4).half()
    keys = (torch.randn(keys_n, 128, generator=g) / 8).half()
    positions = torch.arange(150, 150 + tokens)
    valid = torch.ones(tokens, dtype=torch.bool)
    valid[[3, 40]] = False
    return q, weights, keys, positions, valid


def test_knob_default_off():
    values = [
        ast.unparse(n.value)
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        and ast.unparse(n.targets[0] if isinstance(n, ast.Assign) else n.target)
        == "_INDEX_OVERLAP"
    ]
    # The port maps the knob to VLLM_V41_INDEX_OVERLAP, whose default is "0".
    assert values in (["False"], ["envs.VLLM_V41_INDEX_OVERLAP == '1'"]), values
    if values != ["False"]:
        env = (ROOT / "vllm" / "envs.py").read_text()
        assert 'VLLM_V41_INDEX_OVERLAP: str = "0"' in env
        assert 'os.getenv("VLLM_V41_INDEX_OVERLAP", "0")' in env


def test_dispatch():
    """Knob off: select_compressed_positions runs its own body and never
    calls the overlap copy. On: every argument goes to the overlap copy and
    its result comes back."""
    q, weights, keys, positions, valid = _inputs()
    args = (q, weights, keys, positions, valid, 2, lambda x: x)
    ns = _load()
    calls = []

    def overlap(*a, **k):
        calls.append((a, k))
        return "overlap result"

    ns["_select_compressed_positions_overlap"] = overlap
    off = ns["select_compressed_positions"](*args, top_k=16)
    assert not calls and torch.equal(off, reference_select(*args, 16))
    ns["_INDEX_OVERLAP"] = True
    assert ns["select_compressed_positions"](*args, top_k=16) == "overlap result"
    ((a, k),) = calls
    assert all(x is y for x, y in zip(a, args, strict=True)) and k == {"top_k": 16}


class _FakeCuda:
    def __init__(self, capturing=False):
        self.capturing, self.asked = capturing, 0

    def is_current_stream_capturing(self):
        self.asked += 1
        return self.capturing


def test_gate():
    def gate(knob, cuda, tokens, capturing=False):
        fake = _FakeCuda(capturing)
        ns = _load(_INDEX_OVERLAP=knob, torch=types.SimpleNamespace(cuda=fake))
        q = types.SimpleNamespace(is_cuda=cuda, shape=(tokens, HEADS, 128))
        return ns["_index_overlap"](q, 32), fake.asked

    assert gate(True, True, 70) == (True, 1)
    assert gate(False, True, 70) == (False, 0)
    assert gate(True, False, 70) == (False, 0)  # never asks CUDA for a CPU q
    assert gate(True, True, 32) == (False, 0)  # one tile: nothing to overlap
    assert gate(True, True, 70, capturing=True) == (False, 1)


class _Log(list):
    pass


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))


class _StreamTorch:
    """torch with fake CUDA streams that log fork/join and the current one."""

    def __init__(self, log):
        self.log, self.made = log, 0
        self.main = _Stream("main", log)
        self.now = self.main
        outer = self

        class cuda:
            @staticmethod
            def Stream(device=None):
                outer.made += 1
                return _Stream("side", log)

            @staticmethod
            def current_stream():
                return outer.now

            @staticmethod
            @contextlib.contextmanager
            def stream(s):
                prev, outer.now = outer.now, s
                log.append(("enter", s.name))
                try:
                    yield
                finally:
                    outer.now = prev
                    log.append(("exit", s.name))

        self.cuda = cuda

    def __getattr__(self, name):
        return getattr(torch, name)


class _Head:
    device = "cuda:0"

    def __init__(self, log):
        self.log = log

    def record_stream(self, s):
        self.log.append(("record", s.name))


def test_fork_and_join():
    log = _Log()
    fake = _StreamTorch(log)
    ns = _load(torch=fake)
    for _ in range(2):
        with ns["_on_side_stream"](_Head(log)):
            log.append(("body", fake.now.name))
    ns["_join_side_stream"]("cuda:0")
    fork = [
        ("wait", "side", "main"),
        ("record", "side"),
        ("enter", "side"),
        ("body", "side"),
        ("exit", "side"),
    ]
    assert log == fork + fork + [("wait", "main", "side")], log
    assert fake.made == 1  # one side stream per device, reused


def _overlapped(log, fused=None):
    """select_compressed_positions with the overlap forced on: fake streams
    (logged) instead of CUDA ones, real CPU tensors."""
    state = {"side": False}

    @contextlib.contextmanager
    def on_side(head):
        log.append(("fork", tuple(head.shape)))
        state["side"] = True
        try:
            yield
        finally:
            state["side"] = False

    def join(device):
        log.append(("join",))

    names = dict(
        _INDEX_OVERLAP=True,
        _index_overlap=lambda q, tile: q.shape[0] > tile,
        _on_side_stream=on_side,
        _join_side_stream=join,
    )
    if fused is not None:
        names["_FUSED_TOPK"] = True
    ns = _load(**names)
    # The helper functions were compiled before the overrides: point their
    # globals (one shared dict) at the fakes again.
    ns.update(names)
    return ns["select_compressed_positions"], state


def _all_reduce(log, state):
    def all_reduce(x):
        log.append(("reduce", tuple(x.shape), state["side"]))
        return x

    return all_reduce


def test_overlap_wiring_torch_topk():
    q, weights, keys, positions, valid = _inputs()
    log = _Log()
    select, state = _overlapped(log)
    args = (q, weights, keys, positions, valid, 2, _all_reduce(log, state))
    on = select(*args, top_k=16)
    tiles = [(32, 96), (32, 96), (6, 96)]
    expected = []
    for shape in tiles:
        expected += [("fork", shape), ("reduce", shape, True)]
    assert log == expected + [("join",)], log
    off = _load()["select_compressed_positions"](*args, top_k=16)
    ref = reference_select(*args[:6], lambda x: x, 16)
    assert torch.equal(on, off) and torch.equal(off, ref)
    # Knob on with the real gate: CPU tensors stay on the current stream.
    log.clear()
    cpu = _load(_INDEX_OVERLAP=True)["select_compressed_positions"](*args, top_k=16)
    assert torch.equal(cpu, ref)
    assert log == [("reduce", shape, False) for shape in tiles], log


def test_overlap_rejects_bad_inputs():
    q, weights, keys, positions, valid = _inputs()
    bad = {
        "fp32 q": (q.float(), weights, keys),
        "head dim 64": (q[..., :64], weights, keys),
        "weights shape": (q, weights[:, :-1], keys),
        "fp32 weights": (q, weights.float(), keys),
        "key dim 64": (q, weights, keys[:, :64]),
    }
    for knob in (False, True):
        select = _load(_INDEX_OVERLAP=knob)["select_compressed_positions"]
        for name, (q_, w_, k_) in bad.items():
            try:
                select(q_, w_, k_, positions, valid, 2, lambda x: x, top_k=16)
            except ValueError:
                continue
            raise AssertionError(f"overlap={knob} accepted {name}")


def test_overlap_tiles_follow_the_budget():
    """A score budget of 64 rows: two tiles (64 + 6 rows) both ways."""
    q, weights, keys, positions, valid = _inputs()
    budget = 64 * HEADS * keys.shape[0] * 2
    shapes = {}
    for knob in (False, True):
        seen = shapes[knob] = []

        def all_reduce(x, seen=seen):
            seen.append(tuple(x.shape))
            return x

        select = _load(_INDEX_OVERLAP=knob, _INDEX_TILE_BYTES=budget)[
            "select_compressed_positions"
        ]
        out = select(q, weights, keys, positions, valid, 2, all_reduce, top_k=16)
        shapes[knob, "out"] = out
    assert shapes[False] == shapes[True] == [(64, 96), (6, 96)], shapes
    assert torch.equal(shapes[False, "out"], shapes[True, "out"])


def test_overlap_wiring_fused_topk():
    q, weights, keys, positions, valid = _inputs(keys_n=40)
    log = _Log()
    calls = []

    def index_topk(result, scores, positions_, valid_, ratio, top_k):
        calls.append(
            (result.data_ptr(), tuple(scores.shape), positions_[0].item(), ratio, top_k)
        )
        assert scores.is_contiguous() and valid_.shape[0] == result.shape[0]
        result.copy_(scores.topk(top_k, dim=-1).indices.sort(dim=-1).values)

    stub = types.ModuleType("vllm.models.deepseek_v41.index_topk")
    stub.index_topk = index_topk
    saved = sys.modules.get(stub.__name__)
    sys.modules[stub.__name__] = stub
    try:
        select, state = _overlapped(log, fused=True)
        args = (q, weights, keys, positions, valid, 4, _all_reduce(log, state))
        on = select(*args, top_k=16)
        calls_on, calls[:], log_on = list(calls), [], list(log)
        off = _load(_FUSED_TOPK=True)["select_compressed_positions"](*args, top_k=16)
        calls_off = len(calls)
        # Fewer keys than top_k: the fused top-k is skipped (as before L4).
        few = _inputs(keys_n=10)
        narrow = (*few, 4, lambda x: x)
        narrow_on = select(*narrow, top_k=16)
        assert len(calls) == calls_off
        assert torch.equal(narrow_on, reference_select(*narrow, 16))
    finally:
        if saved is None:
            sys.modules.pop(stub.__name__)
        else:
            sys.modules[stub.__name__] = saved
    assert torch.equal(on, off)
    rows = on.shape[1] * 4  # int32 row bytes
    base = on.data_ptr()
    assert [(c[0] - base) // rows for c in calls_on] == [0, 32, 64]
    assert [c[1:] for c in calls_on] == [
        ((32, 40), 150, 4, 16),
        ((32, 40), 182, 4, 16),
        ((6, 40), 214, 4, 16),
    ]
    tiles = [(32, 40), (32, 40), (6, 40)]
    expected = []
    for shape in tiles:
        expected += [("fork", shape), ("reduce", shape, True)]
    assert log_on == expected + [("join",)], log_on
    assert calls_off == 3  # the overlap-off run calls the fused top-k too


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"{len(tests)} passed (cpu)")
