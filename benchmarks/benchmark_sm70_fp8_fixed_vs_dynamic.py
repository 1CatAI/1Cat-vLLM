# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare SM70 FP8 fixed dispatch with dynamic small-shape dispatch.

This diagnostic isolates the TurboMind FP8 dense selector. It uses identical
inputs and packed real model weights, first forcing fixed dispatch and then
forcing dynamic/reuse dispatch in the same process. It intentionally does not
compare against torch matmul, because that does not isolate scheduler changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from vllm import _sm70_ops as sm70_ops


def _digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _load_fp8_layer(
    model_path: Path,
    weight_map: dict[str, str],
    prefix: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    filename = weight_map[f"{prefix}.weight"]
    with safe_open(model_path / filename, framework="pt", device="cpu") as f:
        qweight = f.get_tensor(f"{prefix}.weight").to(device).contiguous()
        scales = (
            f.get_tensor(f"{prefix}.weight_scale_inv")
            .to(device)
            .to(torch.float32)
            .contiguous()
        )
    return qweight, scales


def _candidate_layers(model_path: Path) -> tuple[dict[str, str], list[str]]:
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    layers: list[str] = []
    for key in sorted(weight_map):
        if not key.endswith(".weight"):
            continue
        prefix = key[: -len(".weight")]
        if not prefix.startswith("model.language_model.layers."):
            continue
        if ".experts." in prefix:
            continue
        if f"{prefix}.weight_scale_inv" not in weight_map:
            continue
        layers.append(prefix)
    return weight_map, layers


def _select_layers(layers: list[str], layer_ids: set[int],
                   max_layers: int) -> list[str]:
    selected: list[str] = []
    seen_kind: set[str] = set()
    for prefix in layers:
        parts = prefix.split(".")
        try:
            layer_id = int(parts[3])
        except (IndexError, ValueError):
            layer_id = -1
        kind = ".".join(parts[4:])
        if layer_id in layer_ids or kind not in seen_kind:
            selected.append(prefix)
            seen_kind.add(kind)
        if len(selected) >= max_layers:
            break
    return selected


def _run_case(
    prefix: str,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    m: int,
) -> dict[str, Any]:
    group_size = 128
    n, k = qweight.shape
    if k % group_size != 0 or n % group_size != 0:
        return {
            "layer": prefix,
            "m": m,
            "shape": [int(n), int(k)],
            "skip": "shape_not_multiple_128",
        }

    tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(
        qweight, scales, group_size)
    x = torch.randn((m, k), device=qweight.device, dtype=torch.float16)

    fixed = torch.empty((m, n), device=qweight.device, dtype=torch.float16)
    os.environ["VLLM_SM70_FP8_TUNE_SMALL_SHAPES"] = "0"
    sm70_ops.fp8_gemm_sm70_out_meta(fixed, x, tm_weight, tm_scales, meta)
    torch.cuda.synchronize(qweight.device)

    os.environ["VLLM_SM70_FP8_TUNE_SMALL_SHAPES"] = "1"
    warm = torch.empty((m, n), device=qweight.device, dtype=torch.float16)
    sm70_ops.fp8_gemm_sm70_out_meta(warm, x, tm_weight, tm_scales, meta)
    torch.cuda.synchronize(qweight.device)

    dynamic = torch.empty((m, n), device=qweight.device, dtype=torch.float16)
    sm70_ops.fp8_gemm_sm70_out_meta(dynamic, x, tm_weight, tm_scales, meta)
    torch.cuda.synchronize(qweight.device)

    os.environ["VLLM_SM70_FP8_TUNE_SMALL_SHAPES"] = "0"
    fixed_after = torch.empty((m, n), device=qweight.device, dtype=torch.float16)
    sm70_ops.fp8_gemm_sm70_out_meta(
        fixed_after, x, tm_weight, tm_scales, meta)
    torch.cuda.synchronize(qweight.device)

    diff_dynamic = (fixed.float() - dynamic.float()).abs()
    diff_warm = (fixed.float() - warm.float()).abs()
    diff_fixed_after = (fixed.float() - fixed_after.float()).abs()
    return {
        "layer": prefix,
        "m": m,
        "shape": [int(n), int(k)],
        "fixed_vs_dynamic_max_diff": float(diff_dynamic.max().item()),
        "fixed_vs_dynamic_mean_diff": float(diff_dynamic.mean().item()),
        "fixed_vs_dynamic_equal": bool(torch.equal(fixed, dynamic)),
        "fixed_vs_warm_max_diff": float(diff_warm.max().item()),
        "fixed_vs_warm_equal": bool(torch.equal(fixed, warm)),
        "fixed_repeat_max_diff": float(diff_fixed_after.max().item()),
        "fixed_repeat_equal": bool(torch.equal(fixed, fixed_after)),
        "fixed_hash": _digest(fixed),
        "dynamic_hash": _digest(dynamic),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--m", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-layers", type=int, default=160)
    parser.add_argument(
        "--layer-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 10, 30, 58, 62],
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires CUDA.")
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("This diagnostic is intended for SM70/V100.")
    if not hasattr(torch.ops._C, "fp8_sm70_prepare"):
        raise RuntimeError("Missing _C::fp8_sm70_prepare.")
    if not hasattr(torch.ops._C, "fp8_gemm_sm70_out_meta"):
        raise RuntimeError("Missing _C::fp8_gemm_sm70_out_meta.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    weight_map, layers = _candidate_layers(args.model)
    selected = _select_layers(layers, set(args.layer_ids), args.max_layers)

    original_env = os.environ.get("VLLM_SM70_FP8_TUNE_SMALL_SHAPES")
    results: list[dict[str, Any]] = []
    try:
        for prefix in selected:
            qweight, scales = _load_fp8_layer(
                args.model, weight_map, prefix, device)
            for m in args.m:
                results.append(_run_case(prefix, qweight, scales, m))
    finally:
        if original_env is None:
            os.environ.pop("VLLM_SM70_FP8_TUNE_SMALL_SHAPES", None)
        else:
            os.environ["VLLM_SM70_FP8_TUNE_SMALL_SHAPES"] = original_env

    failures = [
        item for item in results
        if item.get("fixed_vs_dynamic_max_diff", 0.0) != 0.0
        or not item.get("fixed_vs_dynamic_equal", True)
    ]
    summary = {
        "model": str(args.model),
        "selected_layers": len(selected),
        "cases": len([item for item in results if "fixed_vs_dynamic_equal" in item]),
        "failures": len(failures),
        "max_fixed_vs_dynamic_diff": max(
            (item.get("fixed_vs_dynamic_max_diff", 0.0) for item in results),
            default=0.0,
        ),
        "max_fixed_repeat_diff": max(
            (item.get("fixed_repeat_max_diff", 0.0) for item in results),
            default=0.0,
        ),
        "failed_examples": failures[:20],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
