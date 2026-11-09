# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired five-block microbenchmark interval, not cross-run model evidence."""

import math
import statistics


def paired_latency_interval(control, candidate):
    """Student-t interval on five interleaved log speed ratios (four DOF)."""
    if len(control) != 5 or len(candidate) != 5:
        raise ValueError("Exactly five paired timing blocks are required")
    if any(not math.isfinite(v) or v <= 0 for v in (*control, *candidate)):
        raise ValueError("Timing samples must be finite and positive")
    gains = [math.log(a / b) for a, b in zip(control, candidate, strict=True)]
    center = statistics.mean(gains)
    radius = 2.7764451051977987 * statistics.stdev(gains) / math.sqrt(5)
    return {
        "method": "paired log ratio, Student-t df=4, five within-run blocks",
        "geomean_reduction_pct": 100 * (1 - math.exp(-center)),
        "reduction_pct_ci95": [
            100 * (1 - math.exp(-(center - radius))),
            100 * (1 - math.exp(-(center + radius))),
        ],
        "positive_lower_bound": center - radius > 0,
    }
