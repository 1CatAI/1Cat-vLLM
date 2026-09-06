# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU sanity gates for the numerical counterfactual, not native CUDA gates."""

import pytest
import torch

from benchmarks.kernels.benchmark_sm70_qsa_rounding_isolation import isolate


@pytest.mark.parametrize("rounded", [False, True])
def test_order_duplicates_padding_and_masked_nan(rounded):
    # Zero logits make every valid probability exactly one before normalization.
    k = torch.zeros(2, 16, 1, 16, dtype=torch.float16)
    v = torch.arange(32).half().reshape(2, 16, 1, 1).expand_as(k).clone()
    k[0, 0] = torch.nan
    v[0, 0] = torch.nan
    capture = dict(
        q=torch.zeros(2, 2, 16, dtype=torch.float16),
        k=k,
        v=v,
        indices=torch.tensor([[18, 2, 18, -1] + [-1] * 13] * 2),
        table=torch.tensor([[0, 1], [0, 1]]),
        requests=torch.tensor([0, -1]),
        gate=None,
    )
    result = isolate(capture, rounded)
    expected = torch.zeros_like(result)
    expected[0].fill_(38 / 3)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.isfinite(result).all()


def test_probability_cast_does_not_round_the_denominator():
    q = torch.ones(1, 1, 16, dtype=torch.float16)
    k = torch.zeros(1, 16, 1, 16, dtype=torch.float16)
    k[0, 1] = 0.31
    v = torch.zeros_like(k)
    v[0, 0] = 1
    v[0, 1] = -0.8
    capture = dict(
        q=q,
        k=k,
        v=v,
        indices=torch.tensor([[0, 1]]),
        table=torch.tensor([[0]]),
        requests=torch.tensor([0]),
        gate=None,
    )
    scores = (q.double()[:, 0] @ k[0, :2, 0].double().t())[0] / 4
    p = (scores - scores.max()).exp()
    expected = (p.half().double() @ v[0, :2, 0].double()) / p.sum()
    torch.testing.assert_close(
        isolate(capture, True)[0, 0], expected.half(), atol=0, rtol=0
    )
