# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage acceptance cannot substitute shorter, captured or mismatched runs."""

import pytest

from tests.video.test_h3_acceptance import measurements
from vllm.model_executor.models.minimax_h3.fasth3 import FastH3Spec
from vllm.model_executor.models.minimax_h3.sigma_schedule import DMD2SigmaSchedule
from vllm.video.vsa_acceptance import evaluate_vsa_stage


def stage_measurements():
    result = {}
    schedule = DMD2SigmaSchedule(FastH3Spec().base_schedule)
    for algorithm in ("vsa", "dense"):
        warmup, runs = measurements(calls=4, api_steps=4)
        result[algorithm] = dict(warmup=warmup, runs=runs)
        for run in (warmup, *runs):
            run["config"].update(
                attention_backend="FASTVIDEO_VSA"
                if algorithm == "vsa"
                else "FLASH_ATTN_V100",
                vsa_topk=64 if algorithm == "vsa" else None,
                lora_path=algorithm + "-datafree",
            )
            run["request"]["sampling"].update(
                width=1280, height=736, num_frames=120, fps=24
            )
            run["end_to_end_seconds"] = 50 if algorithm == "vsa" else 70
            for rank in run["ranks"]:
                rank["stage_seconds"]["denoise"] = 31.3 if rank["rank"] == 3 else 30
                rank["useful_denoise_flops"] = 1_700_000_000_000_000
                rank["denoise_flops_by_layer"] = dict(
                    example=rank["useful_denoise_flops"]
                )
                for step in rank["denoise_steps"]:
                    step["useful_flops"] = rank["useful_denoise_flops"] // 4
                rank["denoise_workload"].update(
                    adapter="FastH3Spec",
                    video_sigmas=schedule.shifted_sigmas(12),
                    audio_sigmas=schedule.shifted_sigmas(3),
                )
                if algorithm == "dense":
                    continue
                rank["denoise_workload"].update(
                    work_accounting="sparse_tp_v1",
                    attention_algorithm="vsa",
                    actual_backends=["FASTVIDEO_VSA", "FLASH_ATTN_V100"],
                    sparse_config=dict(
                        topk=64,
                        prefix_segments=[97, 414],
                        video_shape=[37, 23, 40],
                        gated_blocks=50,
                        heads=14,
                        head_size=128,
                    ),
                )
                selected_pairs = 14 * (511 * 34551 + 34040 * (511 + 64 * 64))
                selected_blocks = 14 * (9 * 609 + 600 * 73)
                dense_pairs = 14 * 34551**2
                compression = 4 * 14 * 609**2 * 128
                rank["denoise_sparse_work_by_layer"] = {}
                for layer in range(50):
                    name = f"blocks.{layer}.attn.attention"
                    rank["denoise_sparse_work_by_layer"][name] = dict(
                        head_size=128,
                        heads=14,
                        selected_blocks=4 * selected_blocks,
                        selected_token_pairs=4 * selected_pairs,
                        compression_flops=4 * compression,
                        dense_token_pairs=4 * dense_pairs,
                    )
                    flops = 4 * 4 * selected_pairs * 128
                    rank["denoise_flops_by_layer"][name] = flops
                    rank["denoise_flops_by_layer"][name + ".compression"] = (
                        4 * compression
                    )
                    rank["denoise_flops_by_layer"]["example"] -= flops + 4 * compression
                for step in rank["denoise_steps"]:
                    step.update(
                        sparse_blocks=50 * selected_blocks,
                        sparse_token_pairs=50 * selected_pairs,
                        sparse_compression_flops=50 * compression,
                        attention_avoided_flops=4
                        * 128
                        * 50
                        * (dense_pairs - selected_pairs),
                    )
    return result


def test_stage_uses_slowest_rank_and_keeps_80_tf_goal_separate():
    data = stage_measurements()
    report = evaluate_vsa_stage(data["vsa"], data["dense"])
    assert report["stage_performance_passed"]
    assert report["denoise_median_seconds"] == 31.3
    assert not report["measurements"]["vsa"]["future_80_tflops_passed"]
    assert report["quality_status"] == "requires_independent_numerical_and_human_review"


@pytest.mark.parametrize("failure", ["denoise", "request", "variance"])
def test_stage_rejects_failed_timing_even_with_three_complete_runs(failure):
    data = stage_measurements()
    for i, run in enumerate(data["vsa"]["runs"]):
        if failure == "denoise":
            run["ranks"][3]["stage_seconds"]["denoise"] = 31.300001
        elif failure == "request":
            run["end_to_end_seconds"] = 70
        else:
            for rank in run["ranks"]:
                rank["stage_seconds"]["denoise"] = (25, 30, 35)[i]
    assert not evaluate_vsa_stage(data["vsa"], data["dense"])[
        "stage_performance_passed"
    ]


@pytest.mark.parametrize("failure", ["capture", "host", "short", "schedule", "topk"])
def test_stage_rejects_noncomparable_or_changed_workloads(failure):
    data = stage_measurements()
    if failure == "capture":
        data["vsa"]["runs"][0]["measurement"]["capture"] = True
    for name, item in data.items():
        for run in (item["warmup"], *item["runs"]):
            if failure == "host" and name == "dense":
                run["config"]["host_weight_pin_memory"] = not run["config"][
                    "host_weight_pin_memory"
                ]
            if failure == "short":
                run["request"]["sampling"]["num_frames"] = 60
            if failure == "schedule":
                for rank in run["ranks"]:
                    rank["denoise_workload"]["video_sigmas"][1] = 0.97
            if failure == "topk" and name == "vsa":
                run["config"]["vsa_topk"] = 32
    with pytest.raises(ValueError):
        evaluate_vsa_stage(data["vsa"], data["dense"])
