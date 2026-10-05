# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired prompt-cluster bootstrap for matched MTP acceptance reports."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(gguf, nvfp4, repetitions=20000):
    for key in ("prompt_tokens_sha256", "sampling", "core_sha256", "torch", "cuda"):
        if gguf[key] != nvfp4[key]:
            raise ValueError(f"Unmatched {key}")
    for key in (
        "tensor_parallel_size",
        "dtype",
        "kv_cache_dtype",
        "mamba_ssm_cache_dtype",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "gpu_memory_utilization",
        "enable_prefix_caching",
        "disable_log_stats",
        "speculative_config",
        "compilation_config",
    ):
        if gguf["config"][key] != nvfp4["config"][key]:
            raise ValueError(f"Unmatched runtime setting: {key}")
    if not gguf["complete"] or not nvfp4["complete"]:
        raise ValueError("Both runs must be complete")
    left, right = gguf["rows"], nvfp4["rows"]
    if len(left) != 8 or [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("Eight paired prompt IDs are required")
    index = np.random.default_rng(20261005).integers(0, 8, (repetitions, 8))
    result = {
        "resampling_unit": "paired_prompt",
        "prompts": 8,
        "bootstrap_repetitions": repetitions,
        "metrics": {},
        "rows": [],
    }
    for metric in ("draft_acceptance_rate", "mean_acceptance_length"):
        a = np.asarray([r["acceptance"][metric] for r in left], dtype=float)
        b = np.asarray([r["acceptance"][metric] for r in right], dtype=float)
        result["metrics"][metric] = {
            "gguf_prompt_mean": float(a.mean()),
            "gguf_mean_95ci": np.quantile(a[index].mean(1), [0.025, 0.975]).tolist(),
            "nvfp4_prompt_mean": float(b.mean()),
            "nvfp4_mean_95ci": np.quantile(b[index].mean(1), [0.025, 0.975]).tolist(),
            "paired_difference": float((a - b).mean()),
            "paired_difference_95ci": np.quantile(
                (a - b)[index].mean(1), [0.025, 0.975]
            ).tolist(),
        }
    for a, b in zip(left, right, strict=True):
        result["rows"].append(
            {
                "id": a["id"],
                "gguf_output_tokens": a["output_tokens"],
                "nvfp4_output_tokens": b["output_tokens"],
                "gguf_finish_reason": a["finish_reason"],
                "nvfp4_finish_reason": b["finish_reason"],
                "gguf_acceptance": a["acceptance"],
                "nvfp4_acceptance": b["acceptance"],
            }
        )
    result["limitation"] = (
        "Eight prompt clusters measure this fixed greedy workload. A confidence "
        "interval spanning zero does not establish acceptance equivalence."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("nvfp4", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.gguf.read_text()), json.loads(args.nvfp4.read_text())
    )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
