# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 coupled drafts from vocabulary shards (CPU, interpreter, 4 TP ranks).

``markov_coupled.coupled_shard_tokens`` (each rank scores its own rows with
keyed draws at the global ids, one slot all-reduce per step) must give every
rank exactly ``coupled_markov_tokens``' drafts on the gathered logits: random
rows, mixed temperatures, a greedy row (then also ``MarkovGreedy``), and
infinite ties across shard boundaries. The TP ranks are threads in lockstep,
reusing test_deepseek_v41_markov_shard's emulation (ROWS 16, N 40 per rank).

    python tests/models/test_deepseek_v41_coupled_shard.py   # or pytest
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
V41 = ROOT / "vllm" / "models" / "deepseek_v41"
PKG = "vllm.models.deepseek_v41"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MS = _load(
    "v41_coupled_shard_emulation",
    ROOT / "tests/models/test_deepseek_v41_markov_shard.py",
)


def _coupled():
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    if "vllm.triton_utils" not in sys.modules:
        tu = types.ModuleType("vllm.triton_utils")
        tu.tl, tu.triton = tl, triton
        sys.modules["vllm.triton_utils"] = tu
    cd = _load(
        "vllm.v1.spec_decode.coupled_draw",
        ROOT / "vllm" / "v1" / "spec_decode" / "coupled_draw.py",
    )
    names = [f"{PKG}.markov_fused", f"{PKG}.markov_shard"]
    saved = {n: sys.modules.get(n) for n in names}
    sys.modules[names[0]], sys.modules[names[1]] = MS.FUSED, MS.SHARD
    try:
        mc = _load(f"{PKG}.markov_coupled", V41 / "markov_coupled.py")
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    for name in ("_markov_keyed_block", "_markov_keyed_reduce"):
        setattr(mc, name, MS._Locked(getattr(mc, name)))
    return cd, mc


CD, MC = _coupled()
TP, N, V = MS.TP, MS.N, MS.V


def _run(logits, anchor, temperature, key, positions, seed):
    head, embed = MS._weights(seed)
    anchor = torch.tensor([anchor], dtype=torch.int64)
    temperature = torch.tensor([temperature], dtype=torch.float32)
    keys = torch.tensor([key], dtype=torch.int64)
    want = MC.coupled_markov_tokens(
        MS.FUSED.MarkovGreedy(head, embed),
        logits.view(1, 5, V),
        anchor,
        temperature,
        keys,
        positions.view(1, 5),
        0.5,
    )[0]
    group = MS.Group()
    shards = MS._shards(head, embed, group)
    got = group.run(
        lambda r: MC.coupled_shard_tokens(
            shards[r],
            logits[:, r * N : (r + 1) * N],
            anchor,
            temperature,
            keys,
            positions,
            0.5,
        )
    )
    return want, got, (head, embed, anchor)


@pytest.mark.parametrize("temperature", [1.0, 0.6, 2.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_shard_drafts_equal_full_coupled_drafts(temperature, seed):
    g = torch.Generator().manual_seed(100 + seed)
    logits = MS._rows(g, scale=3.0)
    positions = torch.arange(5, dtype=torch.int64) + 1000 * seed + 7
    want, got, _ = _run(
        logits, 5 + seed, temperature, (1 << 61) + seed, positions, seed
    )
    MS._assert_tokens(f"T{temperature}", want, got)


def test_greedy_row_equals_markov_greedy():
    g = torch.Generator().manual_seed(3)
    logits = MS._rows(g)
    positions = torch.arange(5, dtype=torch.int64)
    want, got, (head, embed, anchor) = _run(logits, 9, 0.0, 12345, positions, 2)
    MS._assert_tokens("greedy", want, got)
    assert torch.equal(want, MS.FUSED.MarkovGreedy(head, embed)(logits, anchor))


def test_infinite_ties_across_shards_take_the_lowest_global_id():
    g = torch.Generator().manual_seed(4)
    logits = MS._rows(g)
    logits[:, N - 1] = logits[:, 2 * N] = float("inf")
    positions = torch.arange(5, dtype=torch.int64) + 50
    want, got, _ = _run(logits, 1, 1.0, 777, positions, 3)
    MS._assert_tokens("inf tie", want, got)
    assert want.tolist() == [N - 1] * 5


def test_wiring_and_knobs():
    model = (V41 / "dspark.py").read_text()
    assert '"VLLM_V41_COUPLED_SHARD", "0") == "1"' in model
    assert "if not _COUPLED_SHARD or self._markov_shard is None:" in model
    assert "V41CoupledDraftGraph(self._coupled_shard_draft)" in model
    graph = (V41 / "query_graph.py").read_text()
    assert "class V41CoupledDraftGraph:" in graph
    proposer = (ROOT / "vllm/v1/spec_decode/dspark.py").read_text()
    assert 'getattr(self.model, "coupled_shard_draft_tokens", None)' in proposer
    assert "hidden_states, self.input_ids[:1], temperature, keys, positions[0]" in (
        proposer
    )
    envs = (ROOT / "vllm/envs.py").read_text()
    for name in ("VLLM_V41_COUPLED_SHARD", "VLLM_V41_COUPLED_SHARD_CHECK"):
        assert f'    {name}: str = "0"\n' in envs
        assert f'lambda: os.getenv("{name}", "0")' in envs


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
