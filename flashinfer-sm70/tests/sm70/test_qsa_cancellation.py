# SPDX-License-Identifier: Apache-2.0
"""Fixed cancellation cases from the 128K paged-engine comparison."""

import json
import math
from pathlib import Path

import pytest
import torch

from flashinfer.sm70 import QsaScorePlan

CASES = json.loads((Path(__file__).parent / "data/qsa_cancellation.json").read_text())[
    "cases"
]


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires physical SM70",
)
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"r{c['row']}-c{c['column']}")
def test_grouped_cancellation_matches_oracle_and_control(case):
    row, column = case["row"], case["column"] % 32
    q = torch.zeros(20, 4, 128, device="cuda", dtype=torch.float16)
    cache = torch.zeros(1, 32, 1, 128, device="cuda", dtype=q.dtype)
    q[row] = torch.tensor(case["query_bits"], device="cuda", dtype=torch.int16).view(
        torch.float16
    )
    cache[0, column, 0] = torch.tensor(
        case["key_bits"], device="cuda", dtype=torch.int16
    ).view(torch.float16)
    table = torch.zeros(4, 1, device="cuda", dtype=torch.int32)
    requests = torch.arange(20, device="cuda", dtype=torch.int32) // 5
    positions = torch.full((20,), 127, device="cuda", dtype=torch.int64)
    lengths = torch.full((4,), 128, device="cuda", dtype=torch.int32)
    plan = QsaScorePlan(32, 4, 32, "cuda")
    scores, _ = plan.run(q, cache, table, requests, positions, lengths)
    dots = (q[row].double() * cache[0, column, 0].double()).sum(-1)
    oracle = (torch.fmax(dots, torch.zeros_like(dots)).sum() / math.sqrt(128)).float()
    control = torch.tensor(case["control_score"], device="cuda", dtype=torch.float32)
    for expected in (oracle, control):
        torch.testing.assert_close(scores[row, column], expected, rtol=2e-6, atol=2e-6)
