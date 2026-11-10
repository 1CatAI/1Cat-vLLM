# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 coupled drafts: fused keyed kernels against the torch definitions (CPU).

``coupled_draw.keyed_argmax`` (sampler and drafter) and
``markov_coupled.coupled_markov_tokens`` (drafter, Markov bias in registers)
compute the keyed Gumbel draws inline. They must give the torch reference
picks: argmax(logits + keyed_gumbel) for the sampler, coupled_pick on
logits + Markov bias for the drafter, greedy rows plain argmax, ties to the
lowest id. Kernels run in the Triton interpreter (TRITON_INTERPRET=1), so
this proves the hash, indexing and reduction logic, not GPU float rounding.

    python tests/models/test_deepseek_v41_coupled_fused.py   # or pytest
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

import importlib.util  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "vllm" / "models" / "deepseek_v41"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _modules():
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    if "vllm.triton_utils" not in sys.modules:
        tu = types.ModuleType("vllm.triton_utils")
        tu.tl, tu.triton = tl, triton
        sys.modules["vllm.triton_utils"] = tu
    cd = _load(
        "vllm.v1.spec_decode.coupled_draw",
        ROOT / "vllm" / "v1" / "spec_decode" / "coupled_draw.py",
    )
    pkg = types.ModuleType("v41fused")
    pkg.__path__ = [str(PKG)]
    sys.modules["v41fused"] = pkg
    _load("v41fused.markov_fused", PKG / "markov_fused.py")
    mc = _load("v41fused.markov_coupled", PKG / "markov_coupled.py")
    mf = sys.modules["v41fused.markov_fused"]
    return cd, mf, mc


CD, MF, MC = _modules()


def _keys(n, seed):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randint(0, 1 << 62, (n,), generator=g, dtype=torch.int64)
    pos = torch.randint(0, 1 << 20, (n,), generator=g, dtype=torch.int64)
    return keys, pos


def test_inline_draws_match_keyed_gumbel():
    @triton.jit
    def draws(keys, pos, out, V: tl.constexpr):
        r = tl.program_id(0)
        v = tl.arange(0, V)
        seed = CD.row_seed(tl.load(keys + r), tl.load(pos + r))
        tl.store(out + r * V + v, CD.gumbel(seed, v.to(tl.int64)))

    keys, pos = _keys(5, 0)
    keys[0], pos[0] = (1 << 63) - 1, (1 << 32) + 7  # top bits and wide positions
    out = torch.empty(5, 256)
    draws[(5,)](keys, pos, out, V=256)
    torch.testing.assert_close(out, CD.keyed_gumbel(keys, pos, 256))


@pytest.mark.parametrize("vocab", [1000, 3000])
def test_sampler_keyed_argmax_matches_torch(vocab):
    g = torch.Generator().manual_seed(vocab)
    rows = 7
    logits = torch.randn(rows, vocab, generator=g) * 3.0
    logits[2, 100:] = float("-inf")  # top-k masked row
    keys, pos = _keys(rows, 1)
    temperature = torch.tensor([1.0, 0.7, 1.3, 0.0, 1.0, 0.0, 2.0])
    random_rows = temperature > 0
    got = CD.keyed_argmax(logits, temperature, keys, pos)
    want = CD.keyed_sample(logits, keys, pos, random_rows)
    assert got.dtype == torch.int64 and torch.equal(got, want)
    assert torch.equal(got[~random_rows], logits[~random_rows].argmax(-1))


def test_drafter_keyed_argmax_matches_coupled_pick_with_strided_positions():
    g = torch.Generator().manual_seed(5)
    batch, steps, vocab = 3, 4, 2048
    logits = torch.randn(batch, vocab, generator=g) * 2.0
    keys, _ = _keys(batch, 2)
    positions = torch.arange(batch * steps, dtype=torch.int64).view(batch, steps) + 90
    temperature = torch.tensor([1.0, 0.0, 0.6])
    for step in range(steps):
        got = CD.keyed_argmax(logits, temperature, keys, positions[:, step], 0.5)
        draws = CD.keyed_gumbel(keys, positions[:, step].contiguous(), vocab)
        want = CD.coupled_pick(logits, temperature, draws, scale=0.5)
        assert torch.equal(got, want)


def test_keyed_argmax_ties_go_to_the_lowest_id():
    logits = torch.zeros(2, 3000)
    logits[:, [5, 1500, 2999]] = 10.0
    out = CD.keyed_argmax(logits, torch.zeros(2), *_keys(2, 3))
    assert out.tolist() == [5, 5]


def _greedy(vocab, rank, seed):
    g = torch.Generator().manual_seed(seed)
    weight = (torch.randn(vocab, rank, generator=g) * 0.05).to(torch.bfloat16)
    embed = (torch.randn(vocab, rank, generator=g) * 0.5).to(torch.float16)
    return MF.MarkovGreedy(weight, embed, steps=4), weight, embed


def test_coupled_markov_tokens_match_eager_coupled_steps():
    vocab, rank, batch, steps = 1000, 32, 3, 4
    greedy, weight, embed = _greedy(vocab, rank, 0)
    g = torch.Generator().manual_seed(1)
    base = torch.randn(batch, steps, vocab, generator=g) * 2.0
    prev = torch.tensor([3, vocab + 5, 77])  # out-of-range anchor clamps
    temperature = torch.tensor([1.0, 0.8, 0.0])
    keys, _ = _keys(batch, 4)
    positions = torch.arange(batch * steps, dtype=torch.int64).view(batch, steps) + 11
    got = MC.coupled_markov_tokens(
        greedy, base, prev, temperature, keys, positions, 0.5
    )
    # eager reference: the drafter's unfused coupled loop
    cur = prev.clamp(0, vocab - 1)
    for step in range(steps):
        rows = embed[cur]
        bias = MF.markov_bias(greedy, rows)
        step_logits = base[:, step] + bias
        draws = CD.keyed_gumbel(keys, positions[:, step].contiguous(), vocab)
        cur = CD.coupled_pick(step_logits, temperature, draws, scale=0.5)
        assert torch.equal(got[:, step], cur), step
    assert weight.dtype == torch.bfloat16


def test_coupled_markov_greedy_row_matches_markov_greedy():
    vocab, rank, steps = 1000, 32, 4
    greedy, _, _ = _greedy(vocab, rank, 2)
    g = torch.Generator().manual_seed(3)
    base = torch.randn(1, steps, vocab, generator=g)
    keys, _ = _keys(1, 5)
    positions = torch.arange(steps, dtype=torch.int64).view(1, steps)
    got = MC.coupled_markov_tokens(
        greedy, base, torch.tensor([9]), torch.zeros(1), keys, positions, 0.5
    )
    want = greedy(base[0], torch.tensor([9]))
    assert torch.equal(got[0], want)


def test_trace_rows_writes_one_line_per_row(tmp_path):
    g = torch.Generator().manual_seed(8)
    logits = torch.randn(4, 500, generator=g)
    keys, pos = _keys(4, 6)
    temperature = torch.tensor([1.0, 0.0, 1.0, 1.0])
    sampled = CD.keyed_argmax(logits, temperature, keys, pos)
    prefix = str(tmp_path / "trace")
    CD.trace_rows(logits, temperature, keys, pos, sampled, path=prefix)
    (out,) = tmp_path.iterdir()
    lines = [ln.split("\t") for ln in out.read_text().splitlines()]
    assert len(lines) == 4 and all(len(c) == 11 for c in lines)
    assert [c[3] for c in lines] == [str(p) for p in pos.tolist()]
    assert all(c[10] == "1" and float(c[7]) >= 0 for c in lines)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
