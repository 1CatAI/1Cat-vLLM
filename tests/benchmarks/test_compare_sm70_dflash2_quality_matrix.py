# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from benchmarks.compare_sm70_dflash2_quality_matrix import (
    _mcnemar_exact,
    _ordered_arm_names,
    _paired_bootstrap_delta,
)


def test_mcnemar_exact_counts_discordant_pairs():
    result = _mcnemar_exact(
        [True, True, False, False],
        [False, True, True, False],
    )

    assert result == {
        "left_only": 1,
        "right_only": 1,
        "discordant": 2,
        "p_value": 1.0,
    }


def test_paired_bootstrap_delta_is_reproducible():
    pairs = [(1.0, 2.0), (3.0, 3.0), (4.0, 3.0)]

    left = _paired_bootstrap_delta(pairs, samples=1000, seed=7)
    right = _paired_bootstrap_delta(pairs, samples=1000, seed=7)

    assert left == right
    assert left["count"] == 3
    assert left["mean_delta"] == 0.0


def test_matrix_arm_order_tracks_acceleration_layers():
    arms: dict[str, object] = {
        "d3": {},
        "d1": {},
        "t0": {},
        "d0": {},
        "d2": {},
    }

    assert _ordered_arm_names(arms) == ["t0", "d0", "d1", "d2", "d3"]
