# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 indexer INDEX_REDUCE: live-prefix decode index-score reduce.

The functions come from ``attention_ops.py`` / ``kv_store.py`` source (the
modules need a CUDA build to import). CPU checks, with fake v41k ops:

- bank=None (VLLM_V41_INDEX_REDUCE unset or "0"): scores are a fresh
  [tokens, rows] tensor, all_reduce runs once, index_reduce_live never runs;
- with a bank, every rank scores into its own bank slot, index_reduce_live
  runs once per call with the bank's pointers / rank / epochs and the store
  width, all_reduce never runs, and every rank's top-k equals the knob-off
  top-k and a torch reference (the fake reduce, like the kernel, adds the
  four partials in rank order in FP32 over the live columns only);
- more tokens than bank rows falls back to all_reduce;
- the live path requires the CUDA scorer (never the Triton fallback);
- _index_bank_ready: off by default, refuses non-TP4 / no custom AR / a v41k
  build without index_reduce_live;
- the kernel's int4 walk covers exactly the live columns of every row and
  stays inside the padded row; the CUDA schema, the binding's argument order
  and the bank layout constants match the Python side;
- model.py hands the store's bank to select_packed_positions.

    python tests/models/test_deepseek_v41_index_reduce.py   # or pytest
"""

import ast
import re
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "vllm" / "models" / "deepseek_v41" / "attention_ops.py"
STORE = ROOT / "vllm" / "models" / "deepseek_v41" / "kv_store.py"
MODEL = ROOT / "vllm" / "models" / "deepseek_v41" / "model.py"
KERNEL = ROOT / "benchmarks" / "deepseek_v41" / "v41k" / "v41k.cu"
ENVS = ROOT / "vllm" / "envs.py"

HEADS = 8  # index_n_heads 32 / TP4
RANKS = 4
TOP_K = 64


_SAVED: dict = {}


def _fake_module(module):
    """Install a fake module; teardown_module puts the originals back."""
    _SAVED.setdefault(module.__name__, sys.modules.get(module.__name__))
    sys.modules[module.__name__] = module


def teardown_module(module=None):
    for name, original in _SAVED.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    _SAVED.clear()


def _functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names), [n.name for n in nodes]
    exec(compile(ast.Module(nodes, []), str(path), "exec"), namespace)
    return namespace


def _constant(path, name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
    raise AssertionError(name)


# -- fakes ------------------------------------------------------------------
def fake_scores(out, q, w, keys, positions, heads, ratio):
    """index_scores_decode on fp16 [rows, 128] keys: live columns only."""
    tokens = out.shape[0]
    end = int(positions[tokens - 1] + 1) // ratio
    dots = torch.einsum(
        "thd,nd->thn", q.view(tokens, heads, 128).float(), keys[:end].float()
    )
    s = dots.half().float().clamp_min(0)
    s = (s * w.view(tokens, heads, 1).float()).half().float()
    out[:, :end] = s.sum(1).half()


def reference_topk(out, scores, positions, valid, ratio, k):
    width = scores.shape[1]
    out.fill_(-1)
    for r in range(scores.shape[0]):
        n = min(int(positions[r] + 1) // ratio, width) if int(valid[r]) else 0
        take = min(k, n)
        if take:
            order = torch.sort(scores[r, :n].float(), descending=True, stable=True)
            top = order.indices[:take]
            out[r, :take] = torch.sort(top).values.to(out.dtype)
    return out


class FakeV41k:
    """index_scores_decode plus index_reduce_live over a pointer registry."""

    def __init__(self):
        self.registry = {}  # pointer -> bank
        self.reduce_calls = []

    def index_scores_decode(self, *args):
        fake_scores(*args)

    def index_reduce_live(
        self, out, p0, p1, p2, p3, rank, epochs, positions, ratio, width
    ):
        self.reduce_calls.append((out.shape, (p0, p1, p2, p3), rank, ratio, width))
        tokens = out.shape[0]
        end = max(0, min(int(positions[tokens - 1] + 1) // ratio, width))
        acc = None
        for p in (p0, p1, p2, p3):
            part = self.registry[p].partial[:tokens, :end].float()
            acc = part if acc is None else acc + part
        out[:, :end] = acc.half()
        epochs.add_(1)


class TorchProxy(types.ModuleType):
    def __init__(self, v41k):
        super().__init__("torch")
        self.ops = types.SimpleNamespace(v41k=v41k)

    def __getattr__(self, name):
        return getattr(torch, name)


class Bank:
    def __init__(self, rank, rows, pointers, decode_rows=8):
        self.rank = rank
        self.width = (rows + 7) // 8 * 8
        self.pointers = pointers
        self.partial = torch.full(
            (decode_rows, self.width), float("nan"), dtype=torch.float16
        )
        self.epochs = torch.zeros(64, dtype=torch.int32)


def _ops(cuda=True):
    v41k = FakeV41k()
    topk = types.ModuleType("vllm.models.deepseek_v41.index_topk")
    topk.index_topk = lambda out, s, p, v, ratio, k: reference_topk(
        out, s, p, v, ratio, k
    )
    topk.index_topk_parallel = topk.index_topk
    _fake_module(topk)
    namespace = dict(torch=TorchProxy(v41k), _cuda_index_scores=lambda: cuda)
    _functions(OPS, ("_index_reduce_live", "select_packed_positions"), namespace)
    return namespace, v41k


def _case(seed=0, rows=1001, tokens=3, first=500, ratio=1):
    g = torch.Generator().manual_seed(seed)
    keys = (torch.randn(rows, 128, generator=g) * 0.5).half()
    qs = [
        (torch.randn(tokens, HEADS, 128, generator=g) * 0.3).half()
        for _ in range(RANKS)
    ]
    ws = [torch.rand(tokens, HEADS, generator=g).half() for _ in range(RANKS)]
    positions = torch.arange(first, first + tokens, dtype=torch.int64)
    valid = torch.ones(tokens, dtype=torch.int32)
    return keys, qs, ws, positions, valid, ratio


def _partials(keys, qs, ws, positions, ratio):
    parts = []
    for q, w in zip(qs, ws):
        out = torch.zeros(q.shape[0], keys.shape[0], dtype=torch.float16)
        fake_scores(
            out, q.reshape(-1, 128), w.reshape(-1), keys, positions, HEADS, ratio
        )
        parts.append(out)
    return parts


def _reference(keys, qs, ws, positions, valid, ratio):
    parts = _partials(keys, qs, ws, positions, ratio)
    total = sum(p.float() for p in parts).half()
    out = torch.empty(len(positions), TOP_K, dtype=torch.int32)
    return reference_topk(out, total, positions, valid, ratio, TOP_K), total


def _knob_off(keys, qs, ws, positions, valid, ratio):
    ns, v41k = _ops()
    total = sum(p.float() for p in _partials(keys, qs, ws, positions, ratio)).half()
    calls = []

    def all_reduce(x):
        calls.append(tuple(x.shape))
        return total.clone()

    results = [
        ns["select_packed_positions"](
            qs[r], ws[r], keys, positions, valid, ratio, all_reduce, top_k=TOP_K
        )
        for r in range(RANKS)
    ]
    return results, calls, v41k


def _knob_on(keys, qs, ws, positions, valid, ratio, decode_rows=8, cuda=True):
    ns, v41k = _ops(cuda)
    pointers = tuple(1000 + r for r in range(RANKS))
    banks = [Bank(r, keys.shape[0], pointers, decode_rows) for r in range(RANKS)]
    for r, bank in enumerate(banks):
        v41k.registry[pointers[r]] = bank
    # Concurrent ranks: every partial is in place before anyone reduces.
    for r, bank in enumerate(banks):
        if len(positions) <= decode_rows:
            fake_scores(
                bank.partial[: len(positions)],
                qs[r].reshape(-1, 128),
                ws[r].reshape(-1),
                keys,
                positions,
                HEADS,
                ratio,
            )
    calls = []
    total = sum(p.float() for p in _partials(keys, qs, ws, positions, ratio)).half()

    def all_reduce(x):
        calls.append(tuple(x.shape))
        return total.clone()

    results = [
        ns["select_packed_positions"](
            qs[r],
            ws[r],
            keys,
            positions,
            valid,
            ratio,
            all_reduce,
            top_k=TOP_K,
            bank=banks[r],
        )
        for r in range(RANKS)
    ]
    return results, calls, v41k, banks


# -- tests ------------------------------------------------------------------
def test_knob_off_keeps_all_reduce():
    case = _case()
    results, calls, v41k = _knob_off(*case)
    keys, *_ = case
    assert calls == [(3, keys.shape[0])] * RANKS
    assert v41k.reduce_calls == []
    expected, _ = _reference(*case)
    for result in results:
        assert torch.equal(result, expected)


def test_bank_reduces_live_prefix_without_all_reduce():
    for seed, (rows, tokens, first, ratio) in enumerate(
        [(1001, 3, 500, 1), (1001, 1, 77, 1), (2049, 8, 1900, 1), (513, 4, 900, 2)]
    ):
        case = _case(seed, rows, tokens, first, ratio)
        keys, qs, ws, positions, valid, ratio = case
        on, calls, v41k, banks = _knob_on(*case)
        off, _, _ = _knob_off(*case)
        expected, total = _reference(*case)
        assert calls == []
        assert len(v41k.reduce_calls) == RANKS
        width = (rows + 7) // 8 * 8
        for r, (shape, pointers, rank, got_ratio, got_width) in enumerate(
            v41k.reduce_calls
        ):
            assert shape == (tokens, width)
            assert pointers == banks[r].pointers and rank == r
            assert got_ratio == ratio and got_width == rows
            assert int(banks[r].epochs[0]) == 1
        for r in range(RANKS):
            assert torch.equal(on[r], expected), (seed, r)
            assert torch.equal(on[r], off[r])


def test_live_scores_equal_custom_ar_sum():
    ns, v41k = _ops()
    keys, qs, ws, positions, valid, ratio = _case(7, 1001, 5, 600, 1)
    pointers = (11, 12, 13, 14)
    banks = [Bank(r, 1001, pointers) for r in range(RANKS)]
    for r, bank in enumerate(banks):
        v41k.registry[pointers[r]] = bank
        fake_scores(
            bank.partial[:5],
            qs[r].reshape(-1, 128),
            ws[r].reshape(-1),
            keys,
            positions,
            HEADS,
            ratio,
        )
    reduced = ns["_index_reduce_live"](
        banks[2], banks[2].partial[:5], positions, ratio, 1001
    )
    assert reduced.shape == (5, 1001) and reduced.stride(1) == 1
    end = (600 + 4 + 1) // ratio
    _, total = _reference(keys, qs, ws, positions, valid, ratio)
    assert torch.equal(reduced[:, :end], total[:, :end])


def test_too_many_tokens_fall_back():
    case = _case(3, 1001, 3, 500, 1)
    on, calls, v41k, _ = _knob_on(*case, decode_rows=2)
    assert v41k.reduce_calls == []
    assert calls == [(3, 1001)] * RANKS
    expected, _ = _reference(*case)
    for result in on:
        assert torch.equal(result, expected)


def test_live_path_requires_cuda_scorer():
    tree = ast.parse(OPS.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "select_packed_positions"
    )
    live = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "live"
    )
    text = ast.unparse(live.value)
    for part in (
        "bank is not None",
        "bank.partial.shape[0]",
        "bank.width",
        "_cuda_index_scores()",
    ):
        assert part in text, text


def _ready(knob, world=4, ca=True, disabled=False, connected=True, op=True):
    warnings = []
    logger = types.SimpleNamespace(warning=lambda *a: warnings.append(a))
    moe = types.ModuleType("vllm.model_executor.layers.quantization.gguf_v41_moe")
    moe._expert_major_ready = lambda: True
    _fake_module(moe)
    v41k = types.SimpleNamespace(**({"index_reduce_live": object()} if op else {}))
    namespace = dict(
        envs=types.SimpleNamespace(VLLM_V41_INDEX_REDUCE=knob),
        logger=logger,
        torch=types.SimpleNamespace(ops=types.SimpleNamespace(v41k=v41k)),
    )
    _functions(STORE, ("_index_bank_ready",), namespace)
    comm = types.SimpleNamespace(
        ca_comm=types.SimpleNamespace(disabled=disabled, fully_connected=connected)
        if ca
        else None
    )
    group = types.SimpleNamespace(world_size=world, device_communicator=comm)
    return namespace["_index_bank_ready"](group), warnings


def test_bank_gate():
    assert _ready("0") == (False, [])
    assert _ready("1")[0] is True
    for kwargs in (
        dict(world=2),
        dict(ca=False),
        dict(disabled=True),
        dict(connected=False),
        dict(op=False),
    ):
        ok, warnings = _ready("1", **kwargs)
        assert ok is False and len(warnings) == 1, kwargs


def test_knob_default_off():
    env = ENVS.read_text()
    assert 'VLLM_V41_INDEX_REDUCE: str = "0"' in env
    assert 'os.getenv("VLLM_V41_INDEX_REDUCE", "0")' in env
    ignored = env[env.index('"VLLM_V41_INDEX_DECODE_BLOCKS",\n') :][:4000]
    assert '"VLLM_V41_INDEX_REDUCE"' not in ignored  # a compile factor


def _kernel_walk(tokens, end, n_pad, ctas=64, threads=256):
    """The kernel's grid-stride walk: int4 slots written per row."""
    chunks = (end + 7) // 8
    row4 = n_pad // 8
    written = set()
    total = tokens * chunks
    for start in range(ctas * threads):
        for i in range(start, total, ctas * threads):
            at = (i // chunks) * row4 + i % chunks
            assert at not in written
            written.add(at)
    return written


def test_kernel_walk_covers_live_columns():
    for tokens, end, rows in (
        (1, 0, 1001),
        (1, 1, 1001),
        (3, 503, 1001),
        (8, 1001, 1001),
        (6, 17, 33),
    ):
        n_pad = (rows + 7) // 8 * 8
        written = _kernel_walk(tokens, end, n_pad)
        for t in range(tokens):
            cols = set()
            for at in written:
                if at // (n_pad // 8) == t:
                    cols.update(
                        range((at % (n_pad // 8)) * 8, (at % (n_pad // 8)) * 8 + 8)
                    )
            assert set(range(end)) <= cols and max(cols, default=-1) < n_pad


def test_kernel_contract_matches_python():
    cu = KERNEL.read_text()
    assert "constexpr int kRedCtas = 64;" in cu
    assert "constexpr int kRedFlags = kRedCtas * 4 * 4;" in cu
    assert ast.literal_eval(_constant(STORE, "REDUCE_CTAS")) == 64
    assert ast.unparse(_constant(STORE, "REDUCE_FLAGS")) == "REDUCE_CTAS * 4 * 4"
    schema = re.search(r'"index_reduce_live\((.*?)\) "\s*"-> \(\)"', cu, re.S)
    assert schema, "schema"
    args = "".join(re.findall(r'"([^"]*)"', cu[schema.start() : schema.end()]))
    assert args.startswith(
        "index_reduce_live(Tensor(a!) out, int p0, int p1, int p2, int p3, int rank, "
        "Tensor(b!) epochs, Tensor positions, int ratio, int width)"
    ), args
    call = re.search(
        r"torch\.ops\.v41k\.index_reduce_live\((.*?)\)\n", OPS.read_text(), re.S
    )
    assert call and " ".join(call.group(1).split()) == (
        "out, *bank.pointers, bank.rank, bank.epochs, positions.long(), ratio, width"
    )
    # The flag and partial layout the bank allocates is the kernel's.
    store = STORE.read_text()
    assert (
        "own[REDUCE_FLAGS:].view(torch.float16).view(DECODE_ROWS, self.width)" in store
    )
    assert "peers[r] + kRedFlags" in cu and "b * 4 + rank" in cu


def test_model_passes_bank():
    text = MODEL.read_text()
    assert 'bank=getattr(store, "index_bank", None),' in text


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    teardown_module()
    print(f"{len(tests)} passed")
