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


def sparse_measurements():
    warmup, runs = measurements(calls=4, api_steps=4)
    heads, gated, calls = 14, 50, 4
    selected_pairs = heads * (33 + 32 * 17)
    selected_blocks = heads * (3 + 2 * 2)
    compression = 4 * heads * 3**2 * 128
    dense_pairs = heads * 33**2
    for run in (warmup, *runs):
        run["config"].update(attention_backend="FASTVIDEO_VSA", vsa_topk=1)
        for rank in run["ranks"]:
            rank["denoise_workload"].update(
                work_accounting="sparse_tp_v1",
                attention_algorithm="vsa",
                used_length=33,
                actual_backends=["FASTVIDEO_VSA", "FLASH_ATTN_V100"],
                sparse_config=dict(
                    topk=1,
                    gated_blocks=gated,
                    heads=heads,
                    head_size=128,
                    prefix_segments=[1],
                    video_shape=[1, 4, 8],
                ),
            )
            rank["denoise_sparse_work_by_layer"] = {}
            for i in range(gated):
                name = f"blocks.{i}.attn.attention"
                rank["denoise_sparse_work_by_layer"][name] = dict(
                    head_size=128,
                    heads=heads,
                    selected_blocks=calls * selected_blocks,
                    selected_token_pairs=calls * selected_pairs,
                    compression_flops=calls * compression,
                    dense_token_pairs=calls * dense_pairs,
                )
                flops = 4 * calls * selected_pairs * 128
                rank["denoise_flops_by_layer"][name] = flops
                rank["denoise_flops_by_layer"][name + ".compression"] = (
                    calls * compression
                )
                rank["denoise_flops_by_layer"]["example"] -= flops + calls * compression
            for step in rank["denoise_steps"]:
                step.update(
                    sparse_blocks=gated * selected_blocks,
                    sparse_token_pairs=gated * selected_pairs,
                    sparse_compression_flops=gated * compression,
                    attention_avoided_flops=4
                    * 128
                    * gated
                    * (dense_pairs - selected_pairs),
                )
    return warmup, runs


def test_sparse_acceptance_keeps_algorithm_savings_out_of_numerator():
    warmup, runs = sparse_measurements()
    report = evaluate_performance(runs, warmup=warmup)
    assert report["rank_median_tflops"] == [100.0] * 4
    assert report["performance_passed"]
    assert all(n > 0 for n in report["attention_avoided_flops_by_run_and_rank"][0])


@pytest.mark.parametrize(
    "invalid",
    [
        "padded_pairs",
        "block_count",
        "missing_gate",
        "compression",
        "savings",
        "backend",
    ],
)
def test_sparse_acceptance_rejects_invented_or_incomplete_work(invalid):
    warmup, runs = sparse_measurements()
    rank = runs[0]["ranks"][0]
    layers = rank["denoise_sparse_work_by_layer"]
    name = next(iter(layers))
    if invalid == "padded_pairs":
        layers[name]["selected_token_pairs"] = layers[name]["selected_blocks"] * 64**2
    elif invalid == "block_count":
        rank["denoise_steps"][0]["sparse_blocks"] += 1
    elif invalid == "missing_gate":
        layers.pop(name)
    elif invalid == "compression":
        layers[name]["compression_flops"] += 1
    elif invalid == "savings":
        rank["denoise_steps"][0]["attention_avoided_flops"] += 1
    else:
        rank["denoise_workload"]["actual_backends"] = ["FLASH_ATTN_V100"]
    with pytest.raises(ValueError):
        evaluate_performance(runs, warmup=warmup)


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


def cached_measurements(algorithm="tea_cache"):
    from collections import Counter

    warmup, runs = measurements(calls=4, api_steps=5)
    refiners = ["token_refiner.blocks.0", "token_refiner.blocks.1"]
    all_blocks = refiners + [f"blocks.{i}" for i in range(50)]
    reused = refiners if algorithm == "tea_cache" else refiners + ["blocks.0"]
    for run in (warmup, *runs):
        for rank in run["ranks"]:
            rank["denoise_workload"].update(
                work_accounting="cached_tp_v1",
                cache_algorithm=algorithm,
                cache_config={"rel_l1_thresh": 0.17}
                if algorithm == "tea_cache"
                else {
                    "Fn_compute_blocks": 1,
                    "Bn_compute_blocks": 0,
                },
            )
            observed: Counter[str] = Counter()
            for index, step in enumerate(rank["denoise_steps"]):
                names = all_blocks if index in (0, 3) else reused
                observed.update(names)
                step.update(
                    executed_blocks=len(names),
                    executed_block_names=list(names),
                    cache_hits=int(len(names) < 52),
                    cache_skipped_blocks=52 - len(names),
                    cache_decision_flops=123456789,
                    useful_flops=len(names) * 10**12,
                )
            rank["denoise_executed_blocks"] = dict(observed)
            rank["denoise_flops_by_layer"] = {
                name + ".matrix": value * 10**12 for name, value in observed.items()
            }
            rank["useful_denoise_flops"] = sum(rank["denoise_flops_by_layer"].values())
    return warmup, runs


@pytest.mark.parametrize("algorithm", ["tea_cache", "cache_dit"])
def test_cache_counts_only_executed_blocks(algorithm):
    warmup, runs = cached_measurements(algorithm)
    result = evaluate_performance(runs, warmup=warmup)
    actual = runs[0]["ranks"][0]["useful_denoise_flops"]
    assert result["rank_median_tflops"] == [actual / 60 / 1e12] * 4
    assert actual < 4 * 52 * 10**12


@pytest.mark.parametrize(
    "invalid",
    [
        "first_hit",
        "invented_hit",
        "wrong_middle",
        "duplicate",
        "layer_total",
        "overhead",
    ],
)
def test_cache_validator_rejects_wrong_reuse_and_counts(invalid):
    warmup, runs = cached_measurements()
    rank = runs[0]["ranks"][0]
    step = rank["denoise_steps"][1]
    if invalid == "first_hit":
        rank["denoise_steps"][0]["cache_hits"] = 1
    elif invalid == "invented_hit":
        step["cache_hits"] = 0
    elif invalid == "wrong_middle":
        step["executed_block_names"][-1] = "blocks.0"
    elif invalid == "duplicate":
        step["executed_block_names"][-1] = step["executed_block_names"][0]
    elif invalid == "layer_total":
        rank["denoise_executed_blocks"]["blocks.0"] += 1
    elif invalid == "overhead":
        step["cache_decision_flops"] = -1
    with pytest.raises(ValueError):
        evaluate_performance(runs, warmup=warmup)
