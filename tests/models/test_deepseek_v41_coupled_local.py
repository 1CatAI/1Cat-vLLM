# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_COUPLED_LOCAL: target keyed verification from vocabulary shards.

1. Fallback conditions. ``coupled_draw.local_fallback`` clears only
   plain-temperature steps with every row random; greedy rows, top-k, top-p,
   active min-p or min-tokens, logprobs, penalties, allowed or bad words, a
   thinking budget or any other logits processor take today's path. The
   runner's ``_can_use_v41_keyed_local`` (run from the source with a fake
   runner) adds grammar, DDTree, tracing, probabilistic MTP, synthetic
   acceptance, dynamic draft vocabulary, PP broadcast and non-standard rows,
   and stays off unless both knobs are set.
2. Exactness. ``coupled_draw.KeyedShard`` on 4 emulated TP ranks (threads,
   sum all-reduce in rank order, Triton interpreter) gives every rank
   ``keyed_argmax``'s tokens on the gathered, temperature-divided rows:
   random rows at several temperatures, uneven shards that are not a
   multiple of the block, a greedy row, and infinite ties across shards.

    python tests/models/test_deepseek_v41_coupled_local.py   # or pytest
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
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "vllm/v1/worker/gpu_model_runner.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _coupled_draw():
    for name in ("vllm", "vllm.v1", "vllm.v1.spec_decode"):
        sys.modules.setdefault(name, types.ModuleType(name))
    if "vllm.triton_utils" not in sys.modules:
        tu = types.ModuleType("vllm.triton_utils")
        tu.tl, tu.triton = tl, triton
        sys.modules["vllm.triton_utils"] = tu
    cd = _load(
        "vllm.v1.spec_decode.coupled_draw",
        ROOT / "vllm" / "v1" / "spec_decode" / "coupled_draw.py",
    )
    sys.modules["vllm.v1.spec_decode"].coupled_draw = cd
    return cd


CD = _coupled_draw()
MS = _load(
    "v41_coupled_local_emulation",
    ROOT / "tests/models/test_deepseek_v41_markov_shard.py",
)
for _name in ("_keyed_shard_block", "_keyed_shard_slot", "_keyed_shard_pick"):
    setattr(CD, _name, MS._Locked(getattr(CD, _name)))

# ---------------------------------------------------------------- fallback


class MinP:
    def __init__(self, count=0):
        self.min_p_count = count


class MinTokens:
    def __init__(self, active=False):
        self.min_toks = {0: (3, set())} if active else {}


class Budget:
    def __init__(self, tracked):
        self.tracked = tracked

    def has_tracked_requests(self):
        return self.tracked


def _metadata(**changes):
    fields = dict(
        all_greedy=False,
        all_random=True,
        temperature=torch.ones(2),
        top_k=None,
        top_p=None,
        max_num_logprobs=None,
        logprob_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        thinking_budget_state_holder=None,
        logitsprocs=types.SimpleNamespace(all=[MinTokens(), MinP()]),
    )
    fields.update(changes)
    return types.SimpleNamespace(**fields)


def _procs(*procs):
    return types.SimpleNamespace(all=list(procs))


FALLBACKS = {
    "greedy_rows": [
        dict(all_random=False),  # a greedy row in a random batch
        dict(all_greedy=True, all_random=False),
    ],
    "no_temperature": [dict(temperature=None)],
    "top_k": [dict(top_k=torch.tensor([50, 0]))],
    "top_p": [dict(top_p=torch.tensor([0.9, 1.0]))],
    "logprobs": [dict(max_num_logprobs=0), dict(logprob_token_ids={0: [5]})],
    "penalties": [dict(no_penalties=False)],
    "allowed_token_ids": [dict(allowed_token_ids_mask=torch.zeros(2, 8))],
    "bad_words": [dict(bad_words_token_ids={0: [[7]]})],
    "thinking_budget": [dict(thinking_budget_state_holder=Budget(True))],
    "min_p": [dict(logitsprocs=_procs(MinTokens(), MinP(1)))],
    "min_tokens": [dict(logitsprocs=_procs(MinTokens(True), MinP()))],
    "logits_processor": [dict(logitsprocs=_procs(MinP(), object()))],
}


def test_plain_temperature_steps_take_the_shard_path():
    assert CD.local_fallback(_metadata()) is None
    assert CD.local_fallback(_metadata(temperature=torch.tensor([0.6, 1.3]))) is None
    assert CD.local_fallback(_metadata(logitsprocs=_procs())) is None
    holder = Budget(False)
    assert CD.local_fallback(_metadata(thinking_budget_state_holder=holder)) is None


@pytest.mark.parametrize(
    "reason,change",
    [(r, c) for r, cases in FALLBACKS.items() for c in cases],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_other_sampling_takes_todays_path(reason, change):
    assert CD.local_fallback(_metadata(**change)) == reason


def _runner_gate(drafts=True, local=True):
    """_can_use_v41_keyed_local from the runner source, its globals faked."""
    tree = ast.parse(RUNNER.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_can_use_v41_keyed_local"
    )
    ns = {
        "_COUPLED_DRAFTS": drafts,
        "_COUPLED_LOCAL": local,
        "SpecDecodeMetadata": type("SpecDecodeMetadata", (), {}),
    }
    exec(compile(ast.Module([fn], []), str(RUNNER), "exec"), ns)
    return ns["_can_use_v41_keyed_local"]


def _runner(metadata=None, shard=True, **changes):
    spec = types.SimpleNamespace(
        method="dspark", draft_sample_method="greedy", use_dflash_ddtree=lambda: False
    )
    fields = dict(
        broadcast_pp_output=False,
        is_pooling_model=False,
        rejection_sampler=types.SimpleNamespace(synthetic_mode=False),
        dynamic_draft_vocab_prefill_topk=0,
        speculative_config=spec,
        input_batch=types.SimpleNamespace(sampling_metadata=metadata or _metadata()),
        _v41_keyed_shard=lambda: object() if shard else None,
    )
    fields.update(changes)
    return types.SimpleNamespace(**fields)


def _spec(drafts=(5, 5)):
    return types.SimpleNamespace(
        num_draft_tokens=list(drafts),
        bonus_logits_indices=torch.zeros(len(drafts)),
    )


def _sched(grammar=False):
    return types.SimpleNamespace(has_structured_output_requests=grammar)


def test_runner_gate_on_for_plain_temperature():
    assert _runner_gate()(_runner(), _sched(), _spec(), 12) is True
    assert _runner_gate()(_runner(), _sched(), _spec((5,)), 6) is True


def _ddtree():
    return types.SimpleNamespace(
        method="dspark", draft_sample_method="greedy", use_dflash_ddtree=lambda: True
    )


def _mtp_probabilistic():
    return types.SimpleNamespace(
        method="mtp",
        draft_sample_method="probabilistic",
        use_dflash_ddtree=lambda: False,
    )


@pytest.mark.parametrize(
    "case",
    [
        "knob_off",
        "drafts_off",
        "no_spec",
        "grammar",
        "pp_broadcast",
        "pooling",
        "trace",
        "synthetic",
        "dynamic_vocab",
        "ddtree",
        "mtp_probabilistic",
        "no_spec_config",
        "rows",
        "bonus_rows",
        "no_shard_layout",
        "top_p",
        "greedy_row",
    ],
)
def test_runner_gate_falls_back(case, monkeypatch):
    gate, runner, sched, spec, rows = _runner_gate(), _runner(), _sched(), _spec(), 12
    if case == "knob_off":
        gate = _runner_gate(local=False)
    elif case == "drafts_off":
        gate = _runner_gate(drafts=False)
    elif case == "no_spec":
        spec = None
    elif case == "grammar":
        sched = _sched(grammar=True)
    elif case == "pp_broadcast":
        runner = _runner(broadcast_pp_output=True)
    elif case == "pooling":
        runner = _runner(is_pooling_model=True)
    elif case == "trace":
        monkeypatch.setattr(CD, "TRACE", "/tmp/never-written")
    elif case == "synthetic":
        runner = _runner(rejection_sampler=types.SimpleNamespace(synthetic_mode=True))
    elif case == "dynamic_vocab":
        runner = _runner(dynamic_draft_vocab_prefill_topk=64)
    elif case == "ddtree":
        runner = _runner(speculative_config=_ddtree())
    elif case == "mtp_probabilistic":
        runner = _runner(speculative_config=_mtp_probabilistic())
    elif case == "no_spec_config":
        runner = _runner(speculative_config=None)
    elif case == "rows":
        rows = 11
    elif case == "bonus_rows":
        spec.bonus_logits_indices = torch.zeros(3)
    elif case == "no_shard_layout":
        runner = _runner(shard=False)
    elif case == "top_p":
        runner = _runner(_metadata(top_p=torch.tensor([0.9, 1.0])))
    elif case == "greedy_row":
        runner = _runner(_metadata(all_random=False))
    assert gate(runner, sched, spec, rows) is False


def test_runner_wiring():
    src = RUNNER.read_text()
    assert '_COUPLED_LOCAL = envs.VLLM_V41_COUPLED_LOCAL == "1"' in src
    # Decided after every existing fast path, before the logits are computed.
    assert (
        "                    logits = None\n"
        "                elif self._can_use_v41_keyed_local(\n"
    ) in src
    assert "self._v41_keyed_local_step = True\n                    logits = None" in src
    # _sample: only a step the gate cleared, and draft probabilities (or a
    # grammar that computed the logits) send it back to today's path.
    assert (
        'keyed_local = logits is None and getattr(self, "_v41_keyed_local_step", False)'
    ) in src
    assert "if keyed_local and draft_probs is not None:\n" in src
    assert "if logits is None and not keyed_local:\n" in src
    assert "self._set_coupled_rows(spec_decode_metadata, logits)" in src
    sampler = (ROOT / "vllm/v1/sample/rejection_sampler.py").read_text()
    assert "    def sample_coupled_local(\n" in sampler
    assert "processed.div_(divisor.unsqueeze(-1))" in sampler
    model = (ROOT / "vllm/models/deepseek_v41/model.py").read_text()
    assert "    def local_vocab(self) -> tuple[int, int] | None:\n" in model
    envs = (ROOT / "vllm/envs.py").read_text()
    for name in ("VLLM_V41_COUPLED_LOCAL", "VLLM_V41_COUPLED_LOCAL_CHECK"):
        assert f'    {name}: str = "0"\n' in envs
        assert f'lambda: os.getenv("{name}", "0")' in envs


def _local_vocab(vocab, tp, rank, padding=64, added=0, scale=1.0, soft_cap=None):
    """DeepseekV41ForCausalLM.local_vocab from the model source on a fake
    head laid out as VocabParallelEmbedding lays it out."""
    model = ROOT / "vllm/models/deepseek_v41/model.py"
    tree = ast.parse(model.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and any(
            isinstance(f, ast.FunctionDef) and f.name == "local_vocab" for f in n.body
        )
    )
    fn = next(f for f in cls.body if getattr(f, "name", "") == "local_vocab")
    missing = type("PPMissingLayer", (), {})
    ns = {
        "PPMissingLayer": missing,
        "get_tensor_model_parallel_rank": lambda: rank,
        "get_tensor_model_parallel_world_size": lambda: tp,
    }
    exec(compile(ast.Module([fn], []), str(model), "exec"), ns)
    padded = -(-vocab // padding) * padding
    per_rank = -(-(padded + added) // (padding * tp)) * padding
    org_start = min(rank * per_rank, vocab)
    head = types.SimpleNamespace(
        num_embeddings_per_partition=per_rank,
        num_added_embeddings=added,
        org_vocab_size=vocab,
        shard_indices=types.SimpleNamespace(
            org_vocab_start_index=org_start,
            padded_org_vocab_start_index=min(rank * per_rank, padded),
            num_org_elements=min((rank + 1) * per_rank, vocab) - org_start,
        ),
    )
    proc = types.SimpleNamespace(
        logits_as_input=False, scale=scale, soft_cap=soft_cap, org_vocab_size=vocab
    )
    return ns["local_vocab"](types.SimpleNamespace(lm_head=head, logits_processor=proc))


def test_local_vocab_layouts():
    # 8-GPU build: 129280 = 4 x 32320, no padding
    assert [_local_vocab(129280, 4, r) for r in range(4)] == [
        (r * 32320, 32320) for r in range(4)
    ]
    # padding only in the last shard's tail
    assert [_local_vocab(1000, 4, r) for r in range(4)] == [
        (0, 256),
        (256, 256),
        (512, 256),
        (768, 232),
    ]
    assert _local_vocab(100, 4, 0) is None  # rank 3 would be empty: all ranks off
    assert _local_vocab(129280, 4, 1, scale=0.5) is None
    assert _local_vocab(129280, 4, 1, soft_cap=30.0) is None
    assert _local_vocab(129280, 4, 1, added=64) is None


# --------------------------------------------------------------- exactness

SIZES = [1500, 1500, 1500, 1300]  # not block multiples; the last one shorter
STARTS = [sum(SIZES[:r]) for r in range(len(SIZES))]
VOCAB = sum(SIZES)


def _keys(n, seed):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randint(0, 1 << 62, (n,), generator=g, dtype=torch.int64)
    pos = torch.randint(0, 1 << 20, (n,), generator=g, dtype=torch.int64)
    return keys, pos


def _shard_tokens(logits, temperature, keys, positions):
    """Every rank's picks from its own columns (temperature divided per rank,
    as sample_coupled_local does) and the gathered keyed_argmax picks."""
    divisor = torch.where(temperature > 0, temperature, torch.ones_like(temperature))
    want = CD.keyed_argmax(logits / divisor.unsqueeze(1), temperature, keys, positions)
    group = MS.Group(tp=len(SIZES))
    shards = [
        CD.KeyedShard(STARTS[r], SIZES[r], r, len(SIZES), group.all_reduce)
        for r in range(len(SIZES))
    ]

    def rank(r):
        local = logits[:, STARTS[r] : STARTS[r] + SIZES[r]].clone()
        local.div_(divisor.unsqueeze(1))
        return shards[r](local, temperature, keys, positions)

    got = group.run(rank)
    for r, tokens in enumerate(got):
        if isinstance(tokens, BaseException):
            raise tokens
        assert torch.equal(tokens, want), (r, tokens.tolist(), want.tolist())
    assert [op for op, _ in group.ops[0]] == ["all_reduce"]
    return want


@pytest.mark.parametrize("temperature", [1.0, 0.6, 2.0])
def test_shard_picks_equal_gathered_keyed_argmax(temperature):
    rows = 6
    g = torch.Generator().manual_seed(int(temperature * 10))
    logits = torch.randn(rows, VOCAB, generator=g) * 3.0
    keys, pos = _keys(rows, 1)
    _shard_tokens(logits, torch.full((rows,), temperature), keys, pos)


def test_mixed_rows_and_a_greedy_row():
    rows = 5
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(rows, VOCAB, generator=g) * 2.0
    keys, pos = _keys(rows, 2)
    temperature = torch.tensor([1.0, 0.0, 0.7, 1.3, 1.0])
    want = _shard_tokens(logits, temperature, keys, pos)
    assert want[1] == logits[1].argmax()


def test_infinite_ties_across_shards_take_the_lowest_global_id():
    rows = 3
    g = torch.Generator().manual_seed(9)
    logits = torch.randn(rows, VOCAB, generator=g)
    logits[:, STARTS[1] + SIZES[1] - 1] = float("inf")
    logits[:, STARTS[3]] = float("inf")
    keys, pos = _keys(rows, 3)
    want = _shard_tokens(logits, torch.ones(rows), keys, pos)
    assert want.tolist() == [STARTS[1] + SIZES[1] - 1] * rows


def test_one_rank_is_keyed_argmax():
    g = torch.Generator().manual_seed(11)
    logits = torch.randn(4, 2500, generator=g)
    keys, pos = _keys(4, 4)
    t = torch.tensor([1.0, 0.5, 1.0, 0.0])
    shard = CD.KeyedShard(0, 2500, 0, 1, lambda slots: slots)
    divisor = torch.where(t > 0, t, torch.ones_like(t)).unsqueeze(1)
    got = shard(logits / divisor, t, keys, pos)
    assert torch.equal(got, CD.keyed_argmax(logits / divisor, t, keys, pos))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
