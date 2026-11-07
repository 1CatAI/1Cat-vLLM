# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 DSpark greedy Markov steps on vocabulary shards (CPU, Triton interpreter).

``markov_shard.MarkovShardGreedy`` lets each TP rank score only its own
vocabulary rows with upstream's ``_markov_block`` and exchange its best row in
one FP32 all-reduce per step. Its tokens must be ``MarkovGreedy``'s (1Cat
#1277's full fused path) bit for bit, NaN skipped as upstream's ``tl.max``
does, ties to the lowest id.

The kernels run in the Triton interpreter (TRITON_INTERPRET=1) with 16 rows
per block on four emulated TP ranks (one thread per rank in lockstep; the
all-reduce sums and the all-gather concatenates in rank order). The vocabulary
is 160 = 10 blocks of 16 (aligned, as 129280 = 1010 x 128 in production) and a
shard is 40 rows: unaligned starts and a partial last block, as 32320 rows are
252.5 blocks of 128. The model glue (``prepare_markov_shard`` and its
self-check, ``local_logits``, ``_greedy_draft``) and the proposer hook are taken
from source by AST: those modules need the compiled extension.

    python tests/models/test_deepseek_v41_markov_shard.py   # or pytest
"""

import os
import sys

import pytest

if "triton" in sys.modules and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip(
        "needs TRITON_INTERPRET=1 in the environment: triton was imported "
        "before this file could turn the interpreter on",
        allow_module_level=True,
    )
os.environ["TRITON_INTERPRET"] = "1"  # before triton is imported

import ast  # noqa: E402
import importlib.util  # noqa: E402
import logging  # noqa: E402
import threading  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PKG = "vllm.models.deepseek_v41"
V41 = ROOT / "vllm" / "models" / "deepseek_v41"
DRAFT = V41 / "dspark.py"
PROPOSER = ROOT / "vllm" / "v1" / "spec_decode" / "dspark.py"
TP, N, K, ROWS = 4, 40, 16, 16
V = TP * N
BIG = 2147483647
INF, NAN = float("inf"), float("nan")
LOCK = threading.Lock()  # the interpreter is not thread-safe


def _load_kernels():
    """markov_fused and markov_shard under the interpreter, triton_utils shimmed."""
    shim = types.ModuleType("vllm.triton_utils")
    shim.tl, shim.triton = tl, triton
    names = ["vllm.triton_utils", f"{PKG}.markov_fused", f"{PKG}.markov_shard"]
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["vllm.triton_utils"] = shim
    modules = []
    try:
        for name in names[1:]:
            path = V41 / (name.rsplit(".", 1)[1] + ".py")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module  # markov_shard imports markov_fused
            spec.loader.exec_module(module)
            modules.append(module)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return modules


class _Locked:
    """A kernel whose launches hold LOCK (rank threads share the interpreter)."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            with LOCK:
                return launch(*args, **kwargs)

        return run


FUSED, SHARD = _load_kernels()
FUSED.ROWS = SHARD.ROWS = ROWS
for _module, _name in (
    (FUSED, "_markov_block"),
    (FUSED, "_markov_reduce"),
    (SHARD, "_markov_block"),
    (SHARD, "_markov_shard_slot"),
    (SHARD, "_markov_shard_pick"),
):
    setattr(_module, _name, _Locked(getattr(_module, _name)))


class _Barrier:
    """threading.Barrier, except that abort() spares released waiters.

    threading.Barrier breaks a waiter that was released but has not woken yet
    when another thread aborts right after the release; here a rank that
    raises after a completed collective cannot break its peers' return.
    """

    def __init__(self, parties, timeout):
        self.cond = threading.Condition()
        self.parties, self.timeout = parties, timeout
        self.count = self.generation = 0
        self.broken = False

    def wait(self):
        with self.cond:
            if self.broken:
                raise threading.BrokenBarrierError
            generation = self.generation
            self.count += 1
            if self.count == self.parties:
                self.count, self.generation = 0, generation + 1
                self.cond.notify_all()
                return
            self.cond.wait_for(
                lambda: self.generation != generation or self.broken, self.timeout
            )
            if self.generation == generation:
                self.broken = True
                self.cond.notify_all()
                raise threading.BrokenBarrierError

    def abort(self):
        with self.cond:
            self.broken = True
            self.cond.notify_all()


class Group:
    """TP ranks as threads; collectives in lockstep, reduced in rank order."""

    def __init__(self, tp=TP):
        self.tp = tp
        # Generous: ranks queue for the interpreter lock between collectives.
        self.barrier = _Barrier(tp, timeout=600)
        self.box = [None] * tp
        self.ops = [[] for _ in range(tp)]
        self.local = threading.local()

    def rank(self):
        return self.local.rank

    def _collect(self, op, tensor):
        rank = self.rank()
        self.ops[rank].append((op, tuple(tensor.shape)))
        self.box[rank] = tensor.clone()
        self.barrier.wait()
        parts = list(self.box)
        self.barrier.wait()
        return parts

    def all_reduce(self, tensor):
        parts = self._collect("all_reduce", tensor)
        out = parts[0]
        for part in parts[1:]:
            out = out + part
        return out

    def all_gather(self, tensor):
        return torch.cat(self._collect("all_gather", tensor), dim=-1)

    def run(self, fn):
        """fn(rank) on every rank; returns the results or the exceptions."""
        results = [None] * self.tp

        def work(rank):
            self.local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as error:  # noqa: BLE001 - reported per rank
                results[rank] = error
                self.barrier.abort()

        threads = [threading.Thread(target=work, args=(r,)) for r in range(self.tp)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return results


def _weights(seed, zero_head=False):
    g = torch.Generator().manual_seed(seed)
    head = torch.randn((V, K), generator=g)
    if zero_head:
        head.zero_()
    embed = (torch.randn((V, K), generator=g) * 0.5).half()
    return head.to(torch.bfloat16), embed


def _shards(head, embed, group):
    """Every rank's shard step, each over its own MarkovGreedy buffers."""
    return [
        SHARD.MarkovShardGreedy(
            FUSED.MarkovGreedy(head, embed), r, TP, group.all_reduce
        )
        for r in range(TP)
    ]


def _shard_run(head, embed, logits, anchor):
    """Upstream's full-path tokens and every rank's shard-path tokens."""
    anchor = torch.tensor([anchor], dtype=torch.int64)
    want = FUSED.MarkovGreedy(head, embed)(logits, anchor)
    group = Group()
    shards = _shards(head, embed, group)
    got = group.run(lambda r: shards[r](logits[:, r * N : (r + 1) * N], anchor))
    return want, got, shards, group


def _assert_tokens(name, want, got):
    for rank, tokens in enumerate(got):
        assert not isinstance(tokens, BaseException), (name, rank, tokens)
        assert torch.equal(tokens, want), (name, rank, tokens.tolist(), want.tolist())


def _rows(g, scale=8.0):
    return (torch.randn((5, V), generator=g) * scale).half()


def _fixtures(g):
    """(name, [5, V] FP16 logits, expected tokens or None) for forced cases."""
    out = []

    def add(name, rows, want=None):
        out.append((name, rows, want))

    r = _rows(g)
    r[:, N - 1] = r[:, N] = INF
    add("inf tie across ranks 0|1", r, [N - 1] * 5)
    r = _rows(g)
    r[:, N + N // 2] = r[:, V - 2] = INF
    add("inf in rank 1 and rank 3", r, [N + N // 2] * 5)
    r = _rows(g)
    r[:, 3 * N + 10] = r[:, 3 * N + 20] = INF
    add("inf twice inside one shard", r, [3 * N + 10] * 5)
    r = _rows(g)
    for step in range(5):  # boundaries 0|1, 1|2, 2|3 and block edges
        edge = (step % 3 + 1) * N
        r[step, edge - 1] = r[step, edge] = INF
    edges = [N - 1, 2 * N - 1, 3 * N - 1, N - 1, 2 * N - 1]
    add("inf tie at each shard boundary", r, edges)
    r = _rows(g)
    r[:, N + 15] = r[:, N + 16] = INF  # a block edge inside rank 1
    add("inf tie at a block edge", r, [N + 15] * 5)
    add("all -inf", torch.full((5, V), -INF).half(), [0] * 5)
    r = _rows(g)
    r[:, ::7] = NAN
    add("NaN every 7th row", r)
    r = _rows(g)
    r[:, N : 2 * N] = NAN
    add("rank 1 all NaN", r)
    r = _rows(g)
    r[:, N + 32 : 2 * N] = NAN  # rank 1's partial last block only
    add("NaN in a partial last block", r)
    r = torch.full((5, V), -INF).half()
    r[:, : 3 * N] = NAN
    add("NaN, then -inf in rank 3", r, [3 * N] * 5)
    r = torch.full((5, V), -INF).half()
    r[:, :N] = NAN
    add("rank 0 NaN, the rest -inf", r, [N] * 5)
    r = torch.full((5, V), -INF).half()
    r[:, 3 * N :] = NAN
    add("rank 3 NaN, the rest -inf", r, [0] * 5)
    r = torch.full((5, V), NAN).half()
    r[:, V - 1] = -INF
    add("NaN but the last row -inf", r, [V - 1] * 5)
    r = torch.full((5, V), NAN).half()
    r[:, N + 5] = -INF  # rank 0's padding rows must not claim id N
    add("NaN but one -inf in rank 1", r, [N + 5] * 5)
    r = _rows(g)
    r[:, 5] = NAN
    r[:, V - 3] = INF
    add("NaN against +inf", r, [V - 3] * 5)
    return out


# ---------- the kernels against upstream ----------


def test_random_tokens_equal_upstream():
    g = torch.Generator().manual_seed(7)
    for seed in range(6):
        head, embed = _weights(seed)
        logits = _rows(g, 4.0)
        anchor = int(torch.randint(-3, V + 3, (1,), generator=g))  # clamps too
        want, got, _, _ = _shard_run(head, embed, logits, anchor)
        _assert_tokens(f"seed {seed}", want, got)


def test_forced_ties_inf_and_nan_equal_upstream():
    g = torch.Generator().manual_seed(11)
    head, embed = _weights(3)
    for name, rows, expected in _fixtures(g):
        want, got, _, _ = _shard_run(head, embed, rows, 9)
        if expected is not None:
            assert want.tolist() == expected, (name, want.tolist())
        _assert_tokens(name, want, got)


def test_exact_finite_ties_equal_upstream():
    # A zero head makes the bias exactly 0, so integer logits tie everywhere,
    # inside blocks, across blocks and across shards; -0.0 ties +0.0.
    g = torch.Generator().manual_seed(13)
    head, embed = _weights(5, zero_head=True)
    cases = [torch.randint(-2, 3, (5, V), generator=g).half() for _ in range(3)]
    zeros = torch.zeros((5, V)).half()
    zeros[:, ::3] = -0.0
    cases.append(zeros)
    top = torch.zeros((5, V)).half()
    top[:, [N - 1, N, 2 * N + 39, 3 * N]] = 3.0
    cases.append(top)
    for i, rows in enumerate(cases):
        want, got, _, _ = _shard_run(head, embed, rows, 1)
        assert want.tolist() == rows.float().argmax(-1).tolist(), i
        _assert_tokens(f"ties {i}", want, got)


def test_all_nan_step_gives_the_upstream_sentinel_and_reads_the_last_row():
    head, embed = _weights(2)
    group = Group()
    shards = _shards(head, embed, group)
    rows = torch.full((5, V), NAN).half()
    anchor = torch.tensor([3])
    got = group.run(lambda r: shards[r](rows[:, r * N : (r + 1) * N], anchor))
    for rank, tokens in enumerate(got):
        assert tokens.tolist() == [BIG] * 5, (rank, tokens)
        assert torch.equal(shards[rank].row, embed[V - 1])


def test_one_exchange_per_step_with_only_the_own_slot_filled():
    g = torch.Generator().manual_seed(17)
    head, embed = _weights(4)
    want, got, shards, group = _shard_run(head, embed, _rows(g), 21)
    _assert_tokens("exchange", want, got)
    for rank, shard in enumerate(shards):
        assert group.ops[rank] == [("all_reduce", (TP, 2))] * 5
        for step in range(5):
            slots = shard.slots[step]
            others = torch.cat([slots[:rank], slots[rank + 1 :]])
            assert torch.equal(others.view(torch.int32), torch.zeros_like(others).int())
            gid = float(slots[rank, 1])
            assert gid == int(gid) and rank * N <= gid < (rank + 1) * N


def test_head_and_embedding_are_views():
    head, embed = _weights(1)
    greedy = FUSED.MarkovGreedy(head, embed)
    for rank in range(TP):
        shard = SHARD.MarkovShardGreedy(greedy, rank, TP, None)
        assert shard.weight.shape == (N, K) and shard.weight.is_contiguous()
        offset = rank * N * K * shard.weight.element_size()
        assert shard.weight.data_ptr() == greedy.weight.data_ptr() + offset
        assert shard.embed is greedy.embed
        assert shard.blocks == 3 and (shard.start, shard.n) == (rank * N, N)


def test_bad_layouts_raise():
    head, embed = _weights(1)
    greedy = FUSED.MarkovGreedy(head, embed)
    for rank, tp in ((4, 4), (-1, 4), (0, 3), (0, 0)):
        try:
            SHARD.MarkovShardGreedy(greedy, rank, tp, None)
        except ValueError:
            continue
        raise AssertionError(f"rank {rank} of {tp} was accepted")
    huge = SimpleNamespace(vocab=1 << 24, rank=K, steps=5)
    try:
        SHARD.MarkovShardGreedy(huge, 0, 4, None)
    except ValueError as error:
        assert "2**24" in str(error)
    else:
        raise AssertionError("ids past 2**24 were accepted")


def test_warm_runs_no_collective_and_leaves_drafts_exact():
    g = torch.Generator().manual_seed(19)
    head, embed = _weights(6)
    logits = _rows(g)
    anchor = torch.tensor([33])
    want = FUSED.MarkovGreedy(head, embed)(logits, anchor)
    group = Group()

    def rank_fn(r):
        shard = SHARD.MarkovShardGreedy(
            FUSED.MarkovGreedy(head, embed), r, TP, group.all_reduce
        )
        shard.warm()
        assert group.ops[r] == []
        return shard(logits[:, r * N : (r + 1) * N], anchor)

    _assert_tokens("after warm", want, group.run(rank_fn))


# ---------- the model glue (taken from dspark.py by AST) ----------


def _method(path, cls, name):
    tree = ast.parse(path.read_text())
    body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(
        n for n in body.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _class(name, bases, methods, namespace, path):
    cls = ast.ClassDef(
        name=name,
        bases=[ast.Name(id=b, ctx=ast.Load()) for b in bases],
        keywords=[],
        body=methods,
        decorator_list=[],
        type_params=[],
    )
    module = ast.Module(body=[cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


DRAFT_METHODS = (
    "compute_logits",
    "local_logits",
    "_greedy_draft",
    "prepare_markov_shard",
    "_markov_shard_layout",
    "_markov_shard_agree",
    "_markov_shard_check",
)
HIDDEN = 8


def _draft_class(group, knob):
    namespace = {
        "torch": torch,
        "MarkovShardGreedy": SHARD.MarkovShardGreedy,
        "tensor_model_parallel_all_reduce": group.all_reduce,
        "get_tensor_model_parallel_rank": group.rank,
        "get_tensor_model_parallel_world_size": lambda: group.tp,
        "_MARKOV_SHARD": knob,
        "logger": logging.getLogger("test_markov_shard"),
    }
    methods = [_method(DRAFT, "DSparkDeepseekV41ForCausalLM", n) for n in DRAFT_METHODS]
    return _class("Draft", [], methods, namespace, DRAFT)


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HIDDEN), requires_grad=False)

    def forward(self, x):
        return x * 0.5  # exact; the norm runs before both logits paths


def _apply(layer, x, bias=None):
    assert bias is None
    return (x.float() @ layer.weight.t()).half()


class _LogitsProcessor:
    logits_as_input, soft_cap, scale, use_all_gather = False, None, 1.0, True

    def __init__(self, group):
        self.group, self.org_vocab_size = group, V

    def __call__(self, lm_head, hidden):
        return self.group.all_gather(lm_head.quant_method.apply(lm_head, hidden))


def _draft(group, rank, knob=True, seed=8):
    """One rank's V4.1 draft: a vocab-parallel head, the fused Markov path."""
    g = torch.Generator().manual_seed(seed)
    vocab_head = torch.randint(-2, 3, (V, HIDDEN), generator=g).float()
    markov_head, embed = _weights(seed)
    draft = _draft_class(group, knob)()
    draft.config = SimpleNamespace(hidden_size=HIDDEN)
    draft.model = SimpleNamespace(norm=_Norm())
    draft.lm_head = SimpleNamespace(
        weight=vocab_head[rank * N : (rank + 1) * N],
        quant_method=SimpleNamespace(apply=_apply),
        org_vocab_size=V,
        num_embeddings_padded=V,
        num_embeddings_per_partition=N,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=rank * N,
            org_vocab_end_index=(rank + 1) * N,
            num_org_vocab_padding=0,
        ),
    )
    draft.logits_processor = _LogitsProcessor(group)
    draft._markov = FUSED.MarkovGreedy(markov_head, embed)
    draft._markov_shard = None
    return draft


def _raised_alike(results, text):
    for rank, result in enumerate(results):
        assert isinstance(result, RuntimeError), (rank, result)
        assert text in str(result), (rank, result)


def test_knob_off_changes_nothing_and_runs_no_collective():
    group = Group()
    drafts = [_draft(group, r, knob=False) for r in range(TP)]
    assert group.run(lambda r: drafts[r].prepare_markov_shard()) == [False] * TP
    assert all(d._markov_shard is None for d in drafts)
    assert group.ops == [[] for _ in range(TP)]


def test_knob_on_checks_then_drafts_from_shards_without_the_gather():
    group = Group()
    drafts = [_draft(group, r) for r in range(TP)]
    assert group.run(lambda r: drafts[r].prepare_markov_shard()) == [True] * TP
    g = torch.Generator().manual_seed(23)
    hidden = (torch.randn((5, HIDDEN), generator=g) * 4).half()
    anchor = torch.tensor([77], dtype=torch.int32)

    def rank_fn(r):
        draft = drafts[r]
        shard, draft._markov_shard = draft._markov_shard, None
        want = draft._greedy_draft(hidden, anchor)  # the full fused path
        draft._markov_shard = shard
        before = len(group.ops[r])
        got = draft._greedy_draft(hidden, anchor)
        return want, got, group.ops[r][before:]

    for want, got, ops in group.run(rank_fn):
        assert torch.equal(want, got), (want.tolist(), got.tolist())
        assert ops == [("all_reduce", (TP, 2))] * 5  # no all-gather


def test_a_layout_problem_on_one_rank_raises_on_every_rank():
    group = Group()
    drafts = [_draft(group, r) for r in range(TP)]
    drafts[2].lm_head.shard_indices.org_vocab_start_index += 1
    results = group.run(lambda r: drafts[r].prepare_markov_shard())
    _raised_alike(results, "cannot run")
    assert all(d._markov_shard is None for d in drafts)
    group = Group()
    drafts = [_draft(group, r) for r in range(TP)]
    for draft in drafts:
        draft.logits_processor.soft_cap = 30.0
    _raised_alike(group.run(lambda r: drafts[r].prepare_markov_shard()), "cannot")
    group = Group()
    drafts = [_draft(group, r) for r in range(TP)]
    drafts[1]._markov = None  # VLLM_V41_MARKOV_FUSED=0
    results = group.run(lambda r: drafts[r].prepare_markov_shard())
    _raised_alike(results, "cannot")
    assert "VLLM_V41_MARKOV_FUSED" in str(results[1])


def _use_on_rank(draft, shard_cls):
    """Give one rank's draft another shard class (each draft has its globals)."""
    type(draft).prepare_markov_shard.__globals__["MarkovShardGreedy"] = shard_cls


def test_a_kernel_failure_on_one_rank_raises_on_every_rank():
    class Failing(SHARD.MarkovShardGreedy):
        def warm(self):
            raise RuntimeError("launch failed")

    group = Group()
    drafts = [_draft(group, r) for r in range(TP)]
    _use_on_rank(drafts[3], Failing)
    results = group.run(lambda r: drafts[r].prepare_markov_shard())
    _raised_alike(results, "cannot run")
    assert "launch failed" in str(results[3])


def test_a_numeric_fault_on_one_rank_fails_the_self_check_everywhere():
    group = Group()
    drafts = [_draft(group, r) for r in range(TP)]
    honest = drafts[1].local_logits

    def one_ulp_off(hidden):
        out = honest(hidden).clone()
        out.view(torch.int16)[0, 7] += 1
        return out

    drafts[1].local_logits = one_ulp_off
    results = group.run(lambda r: drafts[r].prepare_markov_shard())
    _raised_alike(results, "self-check failed")
    assert all(d._markov_shard is None for d in drafts)


def test_a_cross_rank_tie_error_on_one_rank_fails_the_self_check_everywhere():
    class LaterTie(SHARD.MarkovShardGreedy):
        def _pick(self, slots, token_out):  # cross-rank ties go to a later rank
            slots = slots.clone()
            values = slots[:, 0]
            real = values[~values.isnan()]
            if real.numel():
                ties = (values == real.max()).nonzero().flatten()
                if ties.numel() > 1:
                    slots[ties[0], 0] = NAN
            super()._pick(slots, token_out)

    group = Group()
    drafts = [_draft(group, r) for r in range(TP)]
    _use_on_rank(drafts[0], LaterTie)
    _raised_alike(group.run(lambda r: drafts[r].prepare_markov_shard()), "self-check")


# ---------- the proposer hook ----------


def test_proposer_prepares_the_shard_after_loading():
    calls = []

    class Base:
        def load_model(self, target_model):
            calls.append(("load", target_model))

    namespace = {"override": lambda fn: fn, "torch": torch, "Base": Base}
    method = _method(PROPOSER, "DSparkProposer", "load_model")
    proposer_cls = _class("DSparkProposer", ["Base"], [method], namespace, PROPOSER)
    proposer = proposer_cls()
    proposer.model = SimpleNamespace(prepare_markov_shard=lambda: calls.append("prep"))
    proposer.load_model("target")
    assert calls == [("load", "target"), "prep"]
    calls.clear()
    proposer.model = SimpleNamespace()  # other DSpark models have no hook
    proposer.load_model("target")
    assert calls == [("load", "target")]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"{len(tests)} passed")
