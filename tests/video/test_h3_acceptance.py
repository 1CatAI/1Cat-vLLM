# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy
from dataclasses import asdict

import pytest

from vllm.model_executor.models.minimax_h3.config import H3Config, H3Request
from vllm.video.metrics import evaluate_performance


def measurements():
    run = {
        "config": asdict(H3Config()),
        "request": asdict(H3Request()),
        "gpus": [0, 1, 2, 3],
        "measurement": {"profiled": False, "warmup_runs": 1},
        "ranks": [
            {
                "rank": rank,
                "dit_calls": 49,
                "useful_denoise_flops": 6_000_000_000_000_000,
                "peak_allocated_bytes": 29 * 1024**3,
                "stage_seconds": {"denoise": 60.0 if rank == 3 else 50.0},
            }
            for rank in range(4)
        ],
    }
    return [deepcopy(run) for _ in range(3)]


def test_all_ranks_use_slowest_rank_wall_time_and_strict_threshold():
    runs = measurements()
    report = evaluate_performance(runs)
    assert report["rank_median_tflops"] == [100.0] * 4
    assert report["performance_passed"]
    for run in runs:
        run["ranks"][2]["useful_denoise_flops"] = 4_800_000_000_000_000
    assert not evaluate_performance(runs)["performance_passed"]


@pytest.mark.parametrize(
    "invalid", ["profiled", "seed", "rank", "calls", "excluded", "residual"]
)
def test_rejects_incomparable_measurements(invalid):
    runs = measurements()
    if invalid == "profiled":
        runs[1]["measurement"]["profiled"] = True
    elif invalid == "seed":
        runs[1]["request"]["sampling"]["seed"] = 2026
    elif invalid == "rank":
        runs[1]["ranks"][3]["rank"] = 2
    elif invalid == "excluded":
        runs[1]["timing_valid"] = False
    elif invalid == "residual":
        runs[1]["config"]["residual_sequence_parallel"] = True
    else:
        runs[1]["ranks"][0]["dit_calls"] = 48
    with pytest.raises(ValueError):
        evaluate_performance(runs)


def test_memory_gate_and_variability_are_reported_separately():
    runs = measurements()
    runs[0]["ranks"][1]["peak_allocated_bytes"] = 31 * 1024**3
    runs[0]["ranks"][3]["stage_seconds"]["denoise"] = 80.0
    report = evaluate_performance(runs)
    assert not report["memory_passed"]
    assert report["denoise_cv"] > 0.05
    assert not report["performance_passed"]
