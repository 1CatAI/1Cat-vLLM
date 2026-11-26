# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_INDEX_TC through the packed caller, on the CPU.

The two scorers are the real ones, loaded from the CPU build of their own
kernels (benchmarks/deepseek_v41/v41idx3/build_cpu_ops.sh compiles the
``__host__ __device__`` bodies of v41idx.cu and v41idx3.cu, with the card's
recorded mma layout and adder, into one library). Expected scores are the tiled
scorer's (#1372), bit for bit; expected counters come from the guard's rule
stated a second time in numpy (guard_rule.py) and from arithmetic. Only the
SM70 capability gate and the GPU-only top-k are replaced; the dispatch runs as
in serving. Without V41IDX3_CPU_OPS_DIR the library tests skip; the refusals
always run.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

from vllm.models.deepseek_v41 import attention_ops, index_topk
from vllm.models.deepseek_v41.index_tiled import (
    IndexTensorCoreScorer,
    IndexTiledScorer,
)

BENCH = Path(__file__).resolve().parents[2] / "benchmarks/deepseek_v41/v41idx3"
CPU_OPS = os.environ.get("V41IDX3_CPU_OPS_DIR")
needs_library = pytest.mark.skipif(
    CPU_OPS is None, reason="V41IDX3_CPU_OPS_DIR names no CPU build of the scorers"
)


@pytest.mark.parametrize("heads", [0, 4, 16])
def test_tensor_core_scorer_refuses_other_head_counts_before_loading(heads):
    with pytest.raises(ValueError, match="8 heads"):
        IndexTensorCoreScorer(None, heads, torch.device("cpu"))


def test_tensor_core_scorer_refuses_an_unresolved_library():
    with pytest.raises(RuntimeError, match="not initialized"):
        IndexTensorCoreScorer(None, 8, torch.device("cpu"))


def test_prefill_reads_the_scorer_counters_first():
    """The prefill branch is where Python runs: report() precedes its GPU work."""

    class Reported(Exception):
        pass

    class Scorer:
        def report(self):
            raise Reported

    tokens = 9  # 9 tokens x 8 heads > 64: the prefill branch
    with pytest.raises(Reported):
        attention_ops.select_packed_positions(
            torch.zeros(tokens, 8, 128, dtype=torch.float16),
            torch.zeros(tokens, 8, dtype=torch.float16),
            torch.zeros(64, 68, dtype=torch.uint8),
            torch.arange(tokens, dtype=torch.int64),
            torch.ones(tokens, dtype=torch.bool),
            1,
            lambda value: value,
            top_k=4,
            scorer=Scorer(),
        )


@pytest.fixture
def scorers(monkeypatch):
    sys.path.insert(0, str(BENCH))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 0))
    library = os.path.join(CPU_OPS, "v41idx3_cpu_ops.so")
    tc = IndexTensorCoreScorer(library, 8, torch.device("cpu"), report_seconds=0.0)
    tiled = IndexTiledScorer(library, 8)
    yield tc, tiled
    sys.path.remove(str(BENCH))


def decode(scorer, q, w, keys, end, monkeypatch):
    """select_packed_positions' decode branch; returns the scores top-k saw."""
    seen = []

    # GPU-only top-k: record the reduced scores, then a CPU top-k.
    def topk(out, scores, positions, valid, ratio, count):
        seen.append(scores.clone())
        out.copy_(torch.topk(scores.float(), count, dim=1).indices.to(torch.int32))

    monkeypatch.setattr(index_topk, "index_topk", topk)
    tokens = q.shape[0] // 8
    attention_ops.select_packed_positions(
        q.view(tokens, 8, 128),
        w.view(tokens, 8),
        keys,
        torch.full((tokens,), end - 1, dtype=torch.int64),
        torch.ones(tokens, dtype=torch.bool),
        1,
        lambda value: value,
        top_k=4,
        scorer=scorer,
    )
    (scores,) = seen
    return scores


@needs_library
@pytest.mark.parametrize("kind", ["close", "spread", "plain", "anybits"])
@pytest.mark.parametrize("end", [1, 150, 300, 420])
def test_decode_scores_are_the_tiled_scorer_s_bit_for_bit(
    scorers, monkeypatch, kind, end
):
    import guard_rule
    import v41idx3_cases as cases

    tc, tiled = scorers
    tokens, capacity = 6, 300  # end 420: a stale position past the capacity
    q, w, keys = cases.operands(
        kind, tokens, capacity, torch.Generator().manual_seed(7)
    )
    want = decode(tiled, q, w, keys, end, monkeypatch)
    got = decode(tc, q, w, keys, end, monkeypatch)
    # The caller's buffer is uninitialized past the live keys; top-k reads only these.
    live = min(end, capacity)
    assert torch.equal(
        got[:, :live].view(torch.int16), want[:, :live].view(torch.int16)
    )
    cells, heads = guard_rule.expected_counts(q, keys, tokens, live)
    if kind == "close":
        assert (cells, heads) == (0, 0)  # every cell within the limit by construction
    tc.report()
    assert tc.last == [cells, heads, 1, tokens * live]


@needs_library
def test_counters_accumulate_over_decode_calls(scorers, monkeypatch):
    import v41idx3_cases as cases

    tc, _ = scorers
    q, w, keys = cases.operands("close", 2, 128, torch.Generator().manual_seed(3))
    for end in (10, 64, 128):
        decode(tc, q, w, keys, end, monkeypatch)
    tc.report()
    assert tc.last == [0, 0, 3, 2 * (10 + 64 + 128)]


@needs_library
@pytest.mark.parametrize("byte", [113, 141])
def test_keys_outside_the_scale_range_are_recomputed(scorers, monkeypatch, byte):
    """Scale bytes 114..140 keep every E2M1 x 2^(byte - 127) a normal fp16;
    one step outside, the guard passes no cell and every head takes the
    tiled scorer's chain: tokens x keys cells, eight heads each."""
    import v41idx3_cases as cases

    tc, tiled = scorers
    gen = torch.Generator().manual_seed(byte)
    tokens, capacity = 2, 64
    q, w, _ = cases.operands("close", tokens, capacity, gen)
    keys = cases.make_keys(capacity, byte, byte, gen)
    want = decode(tiled, q, w, keys, capacity, monkeypatch)
    got = decode(tc, q, w, keys, capacity, monkeypatch)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    tc.report()
    cells = tokens * capacity
    assert tc.last == [cells, 8 * cells, 1, cells]


def test_indexer_refuses_tensor_core_without_the_tiled_scorer():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.attention import AttentionConfig, V41IndexTiledPolicy
    from vllm.models.deepseek_v41.model import V41Indexer

    policy = V41IndexTiledPolicy(
        enabled=False, library="", tensor_core=True, tensor_core_library=""
    )
    config = VllmConfig(attention_config=AttentionConfig(v41_index_tiled=policy))
    with (
        set_current_vllm_config(config),
        pytest.raises(ValueError, match="VLLM_V41_INDEX_TILED=1"),
    ):
        V41Indexer(None, None, None, 1, "")
