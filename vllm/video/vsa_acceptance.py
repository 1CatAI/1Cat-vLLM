# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate the primary VSA timing stage independently of the future 80-TF goal.

This checks measured timing, work and matched configuration. Weight identity,
numerical quality, human review and other shapes remain separate evidence.
Inputs are native benchmark directories or JSON with ``warmup`` and three
``runs``. No profiler or quality-capture run is eligible for formal timing.
"""

import argparse
import json
import statistics
from pathlib import Path

from vllm.model_executor.models.minimax_h3.fasth3 import FastH3Spec
from vllm.model_executor.models.minimax_h3.sigma_schedule import DMD2SigmaSchedule
from vllm.video.metrics import evaluate_performance


def evaluate_vsa_stage(vsa, dense):
    reports = {
        name: evaluate_performance(data["runs"], warmup=data["warmup"])
        for name, data in (("vsa", vsa), ("dense", dense))
    }
    for report in reports.values():
        report["future_80_tflops_passed"] = report.pop("performance_passed")
    candidate = vsa["runs"][0]
    control = dense["runs"][0]
    allowed_differences = {
        "attention_backend",
        "attention_query_tile",
        "vsa_topk",
        "lora_path",
    }
    configs = [
        {
            key: value
            for key, value in run["config"].items()
            if key not in allowed_differences
        }
        for run in (candidate, control)
    ]
    if configs[0] != configs[1] or any(
        candidate[key] != control[key] for key in ("request", "gpus")
    ):
        raise ValueError(
            "VSA and Dense require matched host, VAE, communication, "
            "export and request settings"
        )
    sampling = candidate["request"]["sampling"]
    if (
        candidate["config"]["tensor_parallel_size"] != 4
        or tuple(sampling[key] for key in ("width", "height", "num_frames", "fps"))
        != (1280, 736, 120, 24)
        or sampling["num_inference_steps"] != 4
        or sampling["num_outputs_per_prompt"] != 1
    ):
        raise ValueError(
            "31.3-second acceptance applies to the TP4 primary four-step request only"
        )
    sparse_work = reports["vsa"]["workflow"]
    dense_work = reports["dense"]["workflow"]
    spec = FastH3Spec()
    schedule = DMD2SigmaSchedule(spec.base_schedule)
    if (
        sparse_work["attention_algorithm"] != "vsa"
        or dense_work["attention_algorithm"] != "dense"
        or sparse_work["sparse_config"]["topk"] != 64
        or sparse_work["sparse_config"]["video_shape"] != [37, 23, 40]
        or sparse_work["sparse_config"]["prefix_segments"] != [97, 414]
        or sparse_work["sparse_config"]["gated_blocks"] != 50
        or candidate["config"]["attention_backend"] != "FASTVIDEO_VSA"
        or candidate["config"]["vsa_topk"] != 64
        or control["config"]["attention_backend"] != "FLASH_ATTN_V100"
        or sparse_work["adapter"] != "FastH3Spec"
        or dense_work["adapter"] != "FastH3Spec"
        or sparse_work["video_sigmas"] != schedule.shifted_sigmas(spec.video_shift)
        or sparse_work["audio_sigmas"] != schedule.shifted_sigmas(spec.audio_shift)
        or sparse_work["task"] != "t2va"
        or any(
            sparse_work[key] != dense_work[key]
            for key in (
                "partition",
                "task",
                "video_sigmas",
                "audio_sigmas",
                "used_length",
                "blocks_per_call",
                "cache_algorithm",
            )
        )
        or len(sparse_work["video_sigmas"]) != 5
    ):
        raise ValueError(
            "primary acceptance requires matching FastH3 four-step Dense "
            "and top-k64 VSA algorithms"
        )
    denoise = statistics.median(reports["vsa"]["denoise_seconds"])
    requests = {
        name: statistics.median(report["end_to_end_seconds"])
        for name, report in reports.items()
    }
    checks = {
        "denoise_median": denoise <= 31.3,
        "denoise_cv": reports["vsa"]["denoise_cv"] <= 0.05,
        "whole_request_beats_dense": requests["vsa"] < requests["dense"],
    }
    return {
        "stage": "primary_vsa_31_3_seconds",
        "checks": checks,
        "stage_performance_passed": all(checks.values()),
        "denoise_median_seconds": denoise,
        "request_median_seconds": requests,
        "measurements": reports,
        "quality_status": "requires_independent_numerical_and_human_review",
        "weight_identity_status": "requires_separate_frozen_base_and_adapter_identity",
        "official_hardware_status": "deferred",
        "other_shapes_status": "not_established_by_primary_timing",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vsa", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        if path.is_file():
            return json.loads(path.read_text())
        return {
            "warmup": json.loads((path / "warmup/run.json").read_text()),
            "runs": [
                json.loads((path / f"run-{i}/run.json").read_text())
                for i in range(1, 4)
            ],
        }

    result = evaluate_vsa_stage(read(args.vsa), read(args.dense))
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
