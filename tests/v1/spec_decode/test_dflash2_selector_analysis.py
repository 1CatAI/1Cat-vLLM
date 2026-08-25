# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

import torch

_ANALYZER_PATH = (
    Path(__file__).parents[3] / "benchmarks/analyze_dflash2_selector_alignment.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "analyze_dflash2_selector_alignment", _ANALYZER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_ANALYZER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ANALYZER
_SPEC.loader.exec_module(_ANALYZER)


def _record():
    candidate_ids = torch.tensor([[10, 11], [20, 21]])
    lattice = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 4.0], [0.0, -4.0]],
        ],
        dtype=torch.float64,
    )
    return _ANALYZER.AlignmentRecord(
        path=Path("synthetic.pt"),
        step=1,
        temperature=1.0,
        top_p=1.0,
        target_topk_ids=candidate_ids,
        target_topk_logits=torch.zeros((2, 2), dtype=torch.float64),
        candidate_ids=candidate_ids,
        realized_logits=torch.stack((lattice[0, 0], lattice[1, 0])),
        unary_logits=torch.zeros((2, 2), dtype=torch.float64),
        lattice_scores=lattice,
        draft_sampled=torch.tensor([1, 10, 20]),
        num_sampled=2,
    )


def test_compact_top_p_keeps_the_crossing_token():
    probs = _ANALYZER._compact_probs(torch.log(torch.tensor([0.6, 0.3, 0.1])), 0.7)
    torch.testing.assert_close(
        probs, torch.tensor([2 / 3, 1 / 3, 0.0], dtype=torch.float64)
    )


def test_beta_zero_reconstructs_realized_selector_rows():
    record = _record()
    current = _ANALYZER._proposal_probs(
        record,
        _ANALYZER.ProposalConfig(name="current", use_cached_logits=True),
    )
    reconstructed = _ANALYZER._proposal_probs(
        record,
        _ANALYZER.ProposalConfig(name="beta-zero"),
    )

    for actual, expected in zip(reconstructed, current):
        torch.testing.assert_close(actual, expected)


def test_future_message_can_prefer_a_better_supported_branch():
    record = _record()
    local = _ANALYZER._proposal_probs(
        record,
        _ANALYZER.ProposalConfig(name="local"),
    )[0]
    global_chain = _ANALYZER._proposal_probs(
        record,
        _ANALYZER.ProposalConfig(name="global", future_beta=1.0),
    )[0]

    assert local[0] == local[1]
    assert global_chain[0] > global_chain[1]


def test_greedy_mixture_is_normalized_and_moves_exact_mass():
    record = _record()
    baseline = _ANALYZER._proposal_probs(
        record,
        _ANALYZER.ProposalConfig(name="baseline", use_cached_logits=True),
    )[1]
    mixed = _ANALYZER._proposal_probs(
        record,
        _ANALYZER.ProposalConfig(
            name="mixture",
            greedy_mix=0.25,
            use_cached_logits=True,
        ),
    )[1]

    expected = baseline * 0.75
    expected[torch.argmax(baseline)] += 0.25
    torch.testing.assert_close(mixed, expected)
    torch.testing.assert_close(mixed.sum(), torch.tensor(1.0, dtype=torch.float64))
