# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up once, then measure the approved H3 workload three times."""

import argparse
import json
from pathlib import Path

from vllm.model_executor.models.minimax_h3.comfy_checkpoint import (
    inspect_comfy_checkpoint,
)
from vllm.model_executor.models.minimax_h3.config import H3Config, H3Request
from vllm.video.engine import H3Engine
from vllm.video.metrics import evaluate_performance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--transformer-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fp16-weight-cache-gib", type=float, default=0)
    parser.add_argument("--fp16-cache-layer", action="append", default=[])
    parser.add_argument("--residual-sequence-parallel", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=("FLASH_ATTN_V100", "FLASHINFER_SM70"),
        default="FLASH_ATTN_V100",
    )
    args = parser.parse_args()
    inspect_comfy_checkpoint(args.transformer_path, expected_partition="fl2va")
    config = H3Config(
        model=args.model,
        transformer_path=args.transformer_path,
        attention_backend=args.attention_backend,
        fp16_weight_cache_gib=args.fp16_weight_cache_gib,
        fp16_cache_layers=tuple(args.fp16_cache_layer),
        residual_sequence_parallel=args.residual_sequence_parallel,
    )
    results = []
    with H3Engine(config) as engine:
        warmup = engine.generate(H3Request(), args.output_dir / "warmup")
        if not warmup["ranks"][0]["quality"]["automatic_passed"]:
            raise RuntimeError("warmup media checks failed; fix quality before timing")
        for index in range(3):
            result = engine.generate(H3Request(), args.output_dir / f"run-{index + 1}")
            result["measurement"] = {"profiled": False, "warmup_runs": 1}
            (args.output_dir / f"run-{index + 1}" / "run.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False)
            )
            results.append(result)
    report = evaluate_performance(results)
    report["quality"] = [run["ranks"][0]["quality"] for run in results]
    report["accepted"] = False  # Human quality review is always required.
    (args.output_dir / "acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
