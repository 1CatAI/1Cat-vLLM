# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from benchmarks.kernels.sm70_paired_stats import paired_latency_interval


def test_identical_ratio_has_exact_interval():
    result = paired_latency_interval([10, 20, 30, 40, 50], [5, 10, 15, 20, 25])
    assert result["reduction_pct_ci95"] == pytest.approx([50.0, 50.0])
    assert result["positive_lower_bound"]


def test_noise_is_not_a_pass():
    assert not paired_latency_interval([10] * 5, [8, 12, 8, 12, 10])[
        "positive_lower_bound"
    ]
    assert not paired_latency_interval([10] * 5, [10] * 5)["positive_lower_bound"]


@pytest.mark.parametrize(
    "samples", [[0] * 5, [-1] * 5, [float("nan")] * 5, [float("inf")] * 5, [1] * 4]
)
def test_invalid_samples_fail_closed(samples):
    with pytest.raises(ValueError):
        paired_latency_interval([10] * 5, samples)
