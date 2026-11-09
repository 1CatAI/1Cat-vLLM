# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host checks for the explicit, non-automatic SM70 collective interface."""

import numpy as np
import pytest

from vllm.model_executor.layers.sm70_collective_calibration import PROBES
from vllm.model_executor.layers.sm70_collectives import _layout


@pytest.mark.parametrize(
    "shape", [(0, 8), (3, 8), (4, 0), (-4, 8), (4,), (True, 8), (4, 2**63)]
)
def test_reject_invalid_layout(shape):
    with pytest.raises((TypeError, ValueError)):
        _layout(shape)


def test_budget_covers_ipc_output_codes_and_calibration():
    shape, raw, resident, calibration = _layout((34560, 5376))
    assert shape == (34560, 5376)
    assert raw == 743_180_800
    assert resident > raw
    assert calibration > resident + 2 * 34560 * 5376 * 4


def _trees(indices):
    if len(indices) == 1:
        return [indices[0]]
    result: list[tuple] = []
    # Anchor the first leaf on the left to remove commutative duplicates.
    for mask in range(1, (1 << len(indices)) - 1, 2):
        left = tuple(x for i, x in enumerate(indices) if mask & (1 << i))
        right = tuple(x for i, x in enumerate(indices) if not mask & (1 << i))
        result.extend((a, b) for a in _trees(left) for b in _trees(right))
    return result


def _evaluate(tree, values):
    if isinstance(tree, int):
        return np.float32(values[tree])
    return np.float32(_evaluate(tree[0], values) + _evaluate(tree[1], values))


def test_fixed_probes_cover_all_fp32_addition_trees():
    trees = _trees((0, 1, 2, 3))
    assert len(trees) == 15
    actual = {
        tuple(int(_evaluate(tree, values).view(np.uint32)) for values, _ in PROBES)
        for tree in trees
    }
    stored = {tuple(bits[i] for _, bits in PROBES) for i in range(15)}
    assert len(stored) == 15
    assert actual == stored
