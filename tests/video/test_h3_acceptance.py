# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy
from dataclasses import asdict

import pytest

from vllm.model_executor.models.minimax_h3.config import H3Config, H3Request
from vllm.video.metrics import evaluate_performance


def measurements(calls=49, api_steps=50, tp=4):
    workload = {
        "work_accounting": "dense_tp_lora_v2",
        "partition": "fl2va",
        "task": "t2va",
        "adapter": None,
        "video_sigmas": [1 - i / calls for i in range(calls + 1)],
        "audio_sigmas": [1 - i / calls for i in range(calls + 1)],
        "blocks_per_call": 52,
        "used_length": 34551,
        "attention_algorithm": "dense",
        "cache_algorithm": None,
        "actual_backends": ["FLASH_ATTN_V100"],
    }
    total = 6_000_000_000_000_000
    quot, rem = divmod(total, calls)
    run: dict = {
        "config": asdict(H3Config(tensor_parallel_size=tp)),
        "request": asdict(H3Request()),
        "gpus": list(range(tp)),
        "engine_session_id": "test-session",
        "request_index": 0,
        "measurement": {"profiled": False, "warmup": False, "capture": False},
        "end_to_end_seconds": 90.0,
        "ranks": [
            {
                "rank": rank,
                "dit_calls": calls,
                "useful_denoise_flops": total,
                "denoise_flops_by_layer": {"example": total},
                "redundant_denoise_flops": 0,
                "redundant_flops_by_layer": {},
                "denoise_workload": deepcopy(workload),
                "denoise_executed_blocks": {str(i): calls for i in range(52)},
                "denoise_steps": [
                    {
                        "index": i,
                        "dit_calls": 1,
                        "executed_blocks": 52,
                        "useful_flops": quot + (i < rem),
                        "redundant_flops": 0,
                        "gpu_seconds": 1.0,
                        "cpu_enqueue_seconds": 0.1,
                        "sparse_blocks": 0,
                        "cache_hits": 0,
                    }
                    for i in range(calls)
                ],
                "peak_allocated_bytes": 29 * 1024**3,
                "stage_seconds": {"denoise": 60.0 if rank == tp - 1 else 50.0},
            }
            for rank in range(tp)
        ],
    }
    run["request"]["sampling"]["num_inference_steps"] = api_steps
    runs = [deepcopy(run) for _ in range(3)]
    for index, measurement in enumerate(runs, 1):
        measurement["request_index"] = index
    warmup = deepcopy(run)
    warmup["measurement"]["warmup"] = True
    return warmup, runs


def test_all_ranks_use_slowest_rank_wall_time_and_strict_threshold():
    warmup, runs = measurements()
    report = evaluate_performance(runs, warmup=warmup)
    assert report["rank_median_tflops"] == [100.0] * 4
    assert report["performance_passed"]
    for run in runs:
        run["ranks"][-1]["stage_seconds"]["denoise"] = 75.0
    assert not evaluate_performance(runs, warmup=warmup)["performance_passed"]


@pytest.mark.parametrize("calls,api_steps", [(49, 50), (4, 5), (8, 9), (4, 4)])
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_uses_actual_intervals_for_all_dense_schedules(calls, api_steps, tp):
    warmup, runs = measurements(calls, api_steps, tp)
    for run in (warmup, *runs):
        run["request"]["sampling"].update(width=1280, height=736, num_frames=120)
    report = evaluate_performance(runs, warmup=warmup)
    assert len(report["rank_median_tflops"]) == tp
    assert len(report["workflow"]["video_sigmas"]) == calls + 1
    assert report["performance_passed"]
    assert report["quality_status"] == "requires_separate_numerical_and_human_review"


@pytest.mark.parametrize(
    "invalid",
    [
        "profiled",
        "seed",
        "rank",
        "calls",
        "excluded",
        "residual",
        "capture",
        "warmup",
        "session",
        "index",
        "missing_step",
        "work",
        "block",
        "layer_work",
        "negative_rank_time",
        "nan_sigma",
        "short_audio",
        "sparse",
        "cache",
        "missing_memory",
        "warmup_incomplete",
        "missing_warmup",
        "nan_step_time",
        "legacy_work",
        "redundant_work",
    ],
)
def test_rejects_incomplete_or_incomparable_measurements(invalid):
    warmup, runs = measurements(4, 5)
    run, rank = runs[1], runs[1]["ranks"][0]
    if invalid in ("profiled", "capture", "warmup"):
        run["measurement"][invalid] = True
    elif invalid == "seed":
        run["request"]["sampling"]["seed"] = 2026
    elif invalid == "rank":
        run["ranks"][3]["rank"] = 2
    elif invalid == "excluded":
        run["timing_valid"] = False
    elif invalid == "residual":
        run["config"]["residual_sequence_parallel"] = True
    elif invalid == "session":
        run["engine_session_id"] = "restarted"
    elif invalid == "index":
        run["request_index"] = 0
    elif invalid == "missing_step":
        rank["denoise_steps"].pop()
    elif invalid == "work":
        rank["useful_denoise_flops"] += 1
    elif invalid == "block":
        rank["denoise_executed_blocks"]["0"] -= 1
    elif invalid == "layer_work":
        rank["denoise_flops_by_layer"]["example"] += 1
    elif invalid == "negative_rank_time":
        rank["stage_seconds"]["denoise"] = -1.0
    elif invalid == "nan_sigma":
        rank["denoise_workload"]["video_sigmas"][1] = float("nan")
    elif invalid == "short_audio":
        rank["denoise_workload"]["audio_sigmas"].pop()
    elif invalid == "sparse":
        rank["denoise_workload"]["attention_algorithm"] = "vsa"
    elif invalid == "cache":
        rank["denoise_workload"]["cache_algorithm"] = "teacache"
    elif invalid == "missing_memory":
        rank["peak_allocated_bytes"] = 0
    elif invalid == "warmup_incomplete":
        warmup["ranks"][0]["dit_calls"] = 3
    elif invalid == "missing_warmup":
        warmup["measurement"]["warmup"] = False
    elif invalid == "nan_step_time":
        rank["denoise_steps"][0]["gpu_seconds"] = float("nan")
    elif invalid == "legacy_work":
        rank["denoise_workload"].pop("work_accounting")
    elif invalid == "redundant_work":
        rank["redundant_denoise_flops"] = 1
    else:
        rank["dit_calls"] = 48
    with pytest.raises(ValueError):
        evaluate_performance(runs, warmup=warmup)


def test_memory_gate_and_variability_are_reported_separately():
    warmup, runs = measurements()
    runs[0]["ranks"][1]["peak_allocated_bytes"] = 31 * 1024**3
    runs[0]["ranks"][3]["stage_seconds"]["denoise"] = 80.0
    report = evaluate_performance(runs, warmup=warmup)
    assert not report["memory_passed"]
    assert report["denoise_cv"] > 0.05
    assert not report["performance_passed"]
