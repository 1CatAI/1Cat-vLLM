# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""4-card (TP4, PP=1) fork: coupled drafts under cold-expert verify truncation (CPU).

On one 4x V100-32GB board the routed experts do not fit, so the 4-card profile
runs #1277's shared_original cache with a nonzero cold set. With a cold set,
VLLM_V41_TRUNC (default 1) masks every draft at a verify position the step did
not fully compute: draft i (verify position i + 1) becomes -1 when
i + 1 >= trunc. The 8-GPU PP2 build keeps every expert resident, so it never
truncates, and the coupled-draft levers (VLLM_V41_COUPLED_DRAFTS, _SHARD,
_LOCAL) were measured only without truncation. These checks cover the one
interaction that is new at PP1:

- the runner masks the drafts before either coupled verification path (the
  gathered keyed path and the vocabulary-shard COUPLED_LOCAL path), and both
  paths verify the masked metadata through token_match_sample;
- token_match_sample (the real Triton kernel, run in the interpreter) stops at
  a masked draft and emits the keyed target sample of a fully computed row;
- a simulated coupled speculative loop with random truncation emits exactly
  the plain keyed sequence, and never a token from a row at or past trunc.

    python tests/models/test_deepseek_v41_fork4card_pp1.py   # or pytest
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
import random  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RS = ROOT / "vllm" / "v1" / "sample" / "rejection_sampler.py"
RUNNER = ROOT / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
CD_SRC = ROOT / "vllm" / "v1" / "spec_decode" / "coupled_draw.py"
PLACEHOLDER = -1  # vllm.v1.sample.rejection_sampler PLACEHOLDER_TOKEN_ID
POISON = 1 << 20  # what an uncomputed verify row would "sample"


def _load_coupled_draw():
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    if "vllm.triton_utils" not in sys.modules:
        tu = types.ModuleType("vllm.triton_utils")
        tu.tl, tu.triton = tl, triton
        sys.modules["vllm.triton_utils"] = tu
    name = "v41_coupled_draw_fork4card"
    spec = importlib.util.spec_from_file_location(name, CD_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_token_match_kernel():
    """token_match_sample_kernel compiled from rejection_sampler.py alone."""
    tree = ast.parse(RS.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "token_match_sample_kernel"
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"triton": triton, "tl": tl}
    exec(compile(module, str(RS), "exec"), namespace)  # noqa: S102
    return namespace["token_match_sample_kernel"]


CD = _load_coupled_draw()
KERNEL = _load_token_match_kernel()


def _token_match(drafts, sampled):
    """One request: token_match_sample on [K] drafts and [K + 1] row samples
    (row j predicts draft j; row K is the bonus row)."""
    k = len(drafts)
    out = torch.full((1, k + 1), PLACEHOLDER, dtype=torch.int32)
    KERNEL[(1,)](
        out,
        torch.tensor([k], dtype=torch.int32),
        torch.tensor(drafts, dtype=torch.int32),
        torch.arange(k, dtype=torch.int32),
        torch.tensor([k], dtype=torch.int32),
        torch.tensor(sampled, dtype=torch.int32),
        k,
    )
    return [t for t in out[0].tolist() if t != PLACEHOLDER]


def _truncate(drafts, trunc):
    """The runner's mask (gpu_model_runner, VLLM_V41_TRUNC): draft i is verify
    position i + 1; positions >= trunc can never be accepted."""
    d = torch.tensor(drafts)
    index = torch.arange(d.numel())
    return torch.where(index + 1 >= trunc, torch.full_like(d, -1), d).tolist()


def test_runner_masks_drafts_before_both_coupled_paths():
    tree = ast.parse(RUNNER.read_text())
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and "current_trunc(self.device)" in (
            ast.unparse(node)
        ):
            fn = node
            break
    assert fn is not None, "the VLLM_V41_TRUNC mask is gone from the runner"
    body = ast.unparse(fn)
    mask = body.index("gguf_shared_runtime.current_trunc(self.device)")
    local = body.index("self._sample_v41_keyed_local(")
    gathered = body.index("self._set_coupled_rows(spec_decode_metadata, logits)")
    sampler = body.index("self.rejection_sampler(spec_decode_metadata")
    assert mask < local and mask < gathered and mask < sampler
    assert "spec_decode_metadata.draft_token_ids = torch.where(" in body


def test_both_coupled_paths_verify_the_masked_metadata():
    rs = RS.read_text()
    tree = ast.parse(rs)
    for name in ("_sample_coupled", "sample_coupled_local"):
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        assert "token_match_sample(metadata, sampled)" in ast.unparse(fn), name


@pytest.mark.parametrize("trunc", [1, 2, 3, 4, 5, 6])
def test_token_match_stops_at_the_first_masked_draft(trunc):
    k = 5
    target = [10, 11, 12, 13, 14]  # every draft matches its keyed sample
    sampled = target + [99]  # row k: bonus
    # Rows at verify positions >= trunc were not fully computed.
    sampled = [POISON if j >= trunc else t for j, t in enumerate(sampled)]
    got = _token_match(_truncate(target, trunc), sampled)
    accepted = min(trunc - 1, k)
    want = target[:accepted] + ([sampled[accepted]] if accepted < k else [99])
    assert got == want
    assert POISON not in got


def test_token_match_without_truncation_is_unchanged():
    assert _token_match([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]) == [1, 2, 3, 4, 5, 6]
    assert _token_match([1, 2, 9, 4, 5], [1, 2, 3, 4, 5, 6]) == [1, 2, 3]


V, K = 24, 5


def _tables(seed, noise):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(V, V, generator=g) * 2.0
    return target, target + torch.randn(V, V, generator=g) * noise


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


def _plain(target, key, temperature, n):
    seq = [0]
    while len(seq) < n:
        seq.append(_keyed(target, seq[-1], key, len(seq), temperature))
    return seq


def _speculative_truncated(target, draft, key, temperature, n, rng):
    """Coupled drafts, then the runner's truncation at a random trunc in
    1..K+1 (K+1: nothing cold), then token_match_sample (the real kernel)."""
    seq = [0]
    while len(seq) < n:
        base, prev, drafts = len(seq), seq[-1], []
        for s in range(K):
            g = CD.keyed_gumbel(torch.tensor([key]), torch.tensor([base + s]), V)
            prev = int(
                CD.coupled_pick(
                    draft[prev].unsqueeze(0), torch.tensor([temperature]), g, 0.5
                )
            )
            drafts.append(prev)
        trunc = rng.randint(1, K + 1)
        # Row j sees seq + drafts[:j]; rows at positions >= trunc are garbage.
        ctx, sampled = seq[-1], []
        for j in range(K + 1):
            if j >= trunc:
                sampled.append(POISON)
            else:
                sampled.append(_keyed(target, ctx, key, base + j, temperature))
            if j < K:
                ctx = drafts[j]
        out = _token_match(_truncate(drafts, trunc), sampled)
        assert out and POISON not in out
        seq.extend(out)
    return seq[:n]


@pytest.mark.parametrize("temperature", [1.0, 0.0])
@pytest.mark.parametrize("noise", [0.0, 0.5, 50.0])
def test_truncated_coupled_speculation_equals_plain_keyed_sampling(temperature, noise):
    target, draft = _tables(7, noise)
    rng = random.Random(41)
    for key in (3, 98765):
        want = _plain(target, key, temperature, 60)
        got = _speculative_truncated(target, draft, key, temperature, 60, rng)
        assert got == want


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
