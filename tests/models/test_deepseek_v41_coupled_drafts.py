# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 coupled sampled drafts with keyed-draw verification (CPU).

``coupled_draw`` gives every (request key, output position, token) one Gumbel
draw from a 32-bit hash. The target samples argmax(logits + G); the drafter
picks argmax(draft_logits / (T x 0.5) + G) with the same G, and verification
accepts the longest draft prefix equal to the target's keyed samples (token
matching). The checks: the hash equals a pure-Python lowbias32; the draws are
Gumbel and independent across positions; Gumbel-max follows the target; a
simulated speculative loop emits exactly the non-speculative keyed sequence
whatever the drafts are (exactness), and coupling raises acceptance over an
independent draft. The module is loaded from its file (torch only); the
sampler, runner and drafter glue and the env registration are checked from
source.

    python tests/models/test_deepseek_v41_coupled_drafts.py   # or pytest
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
import math  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "vllm" / "v1" / "spec_decode" / "coupled_draw.py"


def _load():
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    if "vllm.triton_utils" not in sys.modules:
        tu = types.ModuleType("vllm.triton_utils")
        tu.tl, tu.triton = tl, triton
        sys.modules["vllm.triton_utils"] = tu
    name = "v41_coupled_draw_under_test"
    spec = importlib.util.spec_from_file_location(name, SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CD = _load()


def _py_lowbias32(x):
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def test_mix32_matches_python_lowbias32():
    g = torch.Generator().manual_seed(0)
    xs = torch.randint(0, 1 << 32, (4096,), generator=g, dtype=torch.int64)
    xs[:4] = torch.tensor([0, 1, 0xFFFFFFFF, 0x80000000])
    got = CD.mix32(xs).tolist()
    assert got == [_py_lowbias32(x) for x in xs.tolist()]


def test_draws_are_deterministic_finite_and_keyed():
    keys = torch.tensor([1, 1, 2, (1 << 62) + 5])
    pos = torch.tensor([10, 11, 10, 10])
    g1 = CD.keyed_gumbel(keys, pos, 1000)
    g2 = CD.keyed_gumbel(keys.clone(), pos.clone(), 1000)
    assert torch.equal(g1, g2)
    assert torch.isfinite(g1).all()
    for a in range(4):
        for b in range(a + 1, 4):
            assert not torch.equal(g1[a], g1[b])
    # A row's draws do not depend on the other rows or on the vocab width.
    solo = CD.keyed_gumbel(keys[2:3], pos[2:3], 600)
    assert torch.equal(solo[0], g1[2, :600])


def test_draws_are_standard_gumbel():
    keys = torch.arange(64, dtype=torch.int64) * 7919
    pos = torch.arange(64, dtype=torch.int64) + 100
    g = CD.keyed_gumbel(keys, pos, 4096).double().flatten()
    # Gumbel(0, 1): mean = Euler-Mascheroni, variance = pi^2 / 6.
    assert abs(g.mean().item() - 0.5772156649) < 0.01
    assert abs(g.var().item() - math.pi**2 / 6) < 0.02
    # The CDF exp(-exp(-g)) is uniform: deciles within 1 % of 0.1 each.
    u = torch.exp(-torch.exp(-g))
    counts = torch.histc(u, bins=10, min=0.0, max=1.0) / u.numel()
    assert (counts - 0.1).abs().max().item() < 0.01


def _chi2_ok(observed, expected, dof):
    chi2 = (((observed - expected) ** 2) / expected).sum().item()
    # Generous bound: mean dof, sd sqrt(2 dof); 6 sd.
    return chi2 < dof + 6 * math.sqrt(2 * dof)


def test_gumbel_max_follows_the_target_distribution():
    logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -3.0, 0.2])
    n = 40000
    keys = torch.full((n,), 12345, dtype=torch.int64)
    pos = torch.arange(n, dtype=torch.int64)
    rows = logits.expand(n, -1)
    sampled = CD.keyed_sample(rows, keys, pos, torch.ones(n, dtype=torch.bool))
    observed = torch.bincount(sampled, minlength=8).double()
    expected = logits.double().softmax(0) * n
    assert _chi2_ok(observed, expected, 7)


def test_draws_are_independent_across_positions_and_keys():
    logits = torch.zeros(4)
    n = 40000
    keys = torch.arange(n, dtype=torch.int64) // 2  # pairs share a key
    pos = torch.arange(n, dtype=torch.int64) % 2 + 7  # adjacent positions
    s = CD.keyed_sample(logits.expand(n, -1), keys, pos, torch.ones(n, dtype=bool))
    joint = torch.bincount(s[0::2] * 4 + s[1::2], minlength=16).double()
    assert _chi2_ok(joint, torch.full((16,), n / 2 / 16.0), 15)
    keys2 = torch.arange(n, dtype=torch.int64)
    pos2 = torch.full((n,), 3, dtype=torch.int64)
    s2 = CD.keyed_sample(logits.expand(n, -1), keys2, pos2, torch.ones(n, dtype=bool))
    joint2 = torch.bincount(s2[0::2] * 4 + s2[1::2], minlength=16).double()
    assert _chi2_ok(joint2, torch.full((16,), n / 2 / 16.0), 15)


def test_greedy_rows_take_argmax():
    logits = torch.randn(6, 50, generator=torch.Generator().manual_seed(1))
    keys = torch.arange(6, dtype=torch.int64)
    pos = torch.arange(6, dtype=torch.int64)
    random_rows = torch.tensor([True, False, True, False, False, True])
    s = CD.keyed_sample(logits, keys, pos, random_rows)
    assert torch.equal(s[~random_rows], logits[~random_rows].argmax(-1))
    g = CD.keyed_gumbel(keys, pos, 50)
    temperature = torch.tensor([1.0, 0.0, 0.7, 0.0, 0.0, 2.0])
    d = CD.coupled_pick(logits, temperature, g, scale=0.5)
    assert torch.equal(d[temperature == 0], logits[temperature == 0].argmax(-1))
    want = (logits[0] / 0.5 + g[0]).argmax()
    assert d[0] == want and d[2] == (logits[2] / 0.35 + g[2]).argmax()


def test_request_keys():
    a = CD.request_key(7, "req-a", 0)
    assert a == CD.request_key(7, "req-b", 99)  # seeded: the seed alone
    assert CD.request_key(None, "req-a", 0) != CD.request_key(None, "req-b", 0)
    assert CD.request_key(None, "req-a", 0) != CD.request_key(None, "req-a", 1)
    assert CD.request_key(None, "req-a", 0) == CD.request_key(None, "req-a", 0)
    assert 0 <= a < (1 << 63)


def test_keys_buffer_is_persistent():
    b1 = CD.keys_buffer(torch.device("cpu"), 8)
    b2 = CD.keys_buffer(torch.device("cpu"), 4)
    assert b1.data_ptr() == b2.data_ptr() and b2.numel() == 8


# Simulated speculative decoding on a Markov "model" over a small vocab.

V = 24
K = 5


def _tables(seed, noise):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(V, V, generator=g) * 2.0
    draft = target + torch.randn(V, V, generator=g) * noise
    return target, draft


def _keyed(target, prev, key, pos, temperature):
    logits = (target[prev] / (temperature or 1.0)).unsqueeze(0)
    return int(
        CD.keyed_sample(
            logits,
            torch.tensor([key]),
            torch.tensor([pos]),
            torch.tensor([temperature > 0]),
        )
    )


def _plain(target, key, temperature, n, start=0):
    seq = [start]
    while len(seq) < n:
        seq.append(_keyed(target, seq[-1], key, len(seq), temperature))
    return seq


def _speculative(target, draft, key, temperature, n, draft_key=None, start=0):
    """Rounds of K coupled drafts + keyed token-matching verification."""
    seq, rounds, accepted = [start], 0, 0
    dkey = key if draft_key is None else draft_key
    while len(seq) < n:
        base = len(seq)  # output position of draft 0
        prev, drafts = seq[-1], []
        for s in range(K):
            g = CD.keyed_gumbel(torch.tensor([dkey]), torch.tensor([base + s]), V)
            d = CD.coupled_pick(
                draft[prev].unsqueeze(0), torch.tensor([temperature]), g, 0.5
            )
            prev = int(d)
            drafts.append(prev)
        # Verify: row j sees context seq + drafts[:j] and predicts base + j.
        ctx = seq[-1]
        out = []
        for j in range(K + 1):
            t = _keyed(target, ctx, key, base + j, temperature)
            out.append(t)
            if j == K or t != drafts[j]:
                break
            ctx = drafts[j]
        rounds += 1
        accepted += len(out) - 1
        seq.extend(out)
    return seq[:n], rounds, accepted


@pytest.mark.parametrize("temperature", [1.0, 0.6])
@pytest.mark.parametrize("noise", [0.0, 0.5, 3.0, 50.0])
def test_speculative_output_equals_plain_keyed_sampling(temperature, noise):
    target, draft = _tables(3, noise)
    for key in (11, 12345, (1 << 62) + 3):
        want = _plain(target, key, temperature, 120)
        got, _, _ = _speculative(target, draft, key, temperature, 120)
        assert got == want
        # Drafts with an unrelated key change nothing in the output either.
        got2, _, _ = _speculative(target, draft, key, temperature, 120, key + 1)
        assert got2 == want


def test_greedy_requests_are_plain_argmax():
    target, draft = _tables(4, 0.5)
    seq = _plain(target, 5, 0.0, 40)
    for a, b in zip(seq, seq[1:]):
        assert b == int(target[a].argmax())
    got, _, _ = _speculative(target, draft, 5, 0.0, 40)
    assert got == seq


def test_coupling_raises_acceptance_over_an_independent_draft():
    target, draft = _tables(5, 0.5)
    coupled = independent = rounds_c = rounds_i = 0
    for key in range(20):
        _, r, a = _speculative(target, draft, 1000 + key, 1.0, 200)
        coupled += a
        rounds_c += r
        _, r, a = _speculative(target, draft, 1000 + key, 1.0, 200, 777 + key)
        independent += a
        rounds_i += r
    per_round_c = coupled / rounds_c
    per_round_i = independent / rounds_i
    assert per_round_c > 1.3 * per_round_i, (per_round_c, per_round_i)


def _src(rel):
    return (ROOT / rel).read_text()


def test_sampler_runner_and_drafter_glue():
    rs = _src("vllm/v1/sample/rejection_sampler.py")
    assert "coupled_rows, self._coupled_rows = self._coupled_rows, None" in rs
    assert "def _sample_coupled(" in rs
    assert "token_match_sample(metadata, sampled)" in rs
    tree = ast.parse(rs)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_coupled_sampling_enabled"
    )
    body = ast.unparse(fn)
    for needle in (
        "max_num_logprobs",
        "logprob_token_ids",
        "no_penalties",
        "allowed_token_ids_mask",
        "bad_words_token_ids",
        "thinking_budget_state_holder",
    ):
        assert needle in body
    runner = _src("vllm/v1/worker/gpu_model_runner.py")
    flag = '_COUPLED_DRAFTS = os.getenv("VLLM_V41_COUPLED_DRAFTS", "0") == "1"'
    assert flag in runner
    assert "self._set_coupled_rows(spec_decode_metadata, logits)" in runner
    rows = "positions = self.positions[metadata.logits_indices.long()] + 1"
    assert rows in runner
    # greedy batches skip every coupled op; keys copy pinned, only on change
    assert "if _COUPLED_DRAFTS and not sampling_metadata.all_greedy:" in runner
    assert "buffer[:num_reqs].copy_(staged, non_blocking=True)" in runner
    assert "and not sampling_metadata.all_greedy\n        ):\n" in runner
    fused = (
        "sampled = coupled_draw.keyed_argmax(processed, temperature, keys, positions)"
    )
    assert fused in rs
    dspark = _src("vllm/v1/spec_decode/dspark.py")
    assert "coupled_draw.keyed_argmax(" in dspark
    assert "coupled_draw.keyed_gumbel" not in dspark
    pos = "positions = (self.positions[: batch_size * k] + 1).view(batch_size, k)"
    assert pos in dspark


def test_env_knobs_registered_and_not_compile_factors():
    src = _src("vllm/envs.py")
    for name, d in (
        ("VLLM_V41_COUPLED_DRAFTS", "0"),
        ("VLLM_V41_COUPLED_DRAFT_TEMP", "0.5"),
        ("VLLM_V41_COUPLED_TRACE", ""),
    ):
        assert f'    {name}: str = "{d}"\n' in src
        assert f'lambda: os.getenv("{name}", "{d}")' in src
        start = src.index("ignored_factors: set[str] = {")
        assert f'        "{name}",\n' in src[start : src.index("}", start)]
    assert CD.ENABLED is False and CD.DRAFT_TEMP_SCALE == 0.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
