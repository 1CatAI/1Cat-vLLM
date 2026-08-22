# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired NCU probe for current and QPN8 fused gate/up kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import vllm._C  # noqa: F401
from benchmark_sm70_fp8_qpn8 import (
    _dequantized_weight,
    _error_stats,
    _load_case,
    _qpn8_group_scales,
    _qpn8_prepack,
)

from vllm import _sm70_ops as sm70_ops


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/ymzx/models/Qwen3.8-27B-FP8"),
    )
    parser.add_argument("--qpn8-library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def _time(launch, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / repeats)


def main() -> int:
    args = _parse_args()
    torch.ops.load_library(str(args.qpn8_library.resolve()))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("This profiler requires SM70/V100")

    qweight, scales, _ = _load_case(args.model, "gate_up_fused", 4, 0, device)
    n, k = (int(dim) for dim in qweight.shape)
    hidden = n // 2
    codes = _qpn8_prepack(qweight.view(torch.uint8))
    group_scales = _qpn8_group_scales(scales, n, k)
    tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(qweight, scales, 128, True)
    k_ld, q_ld = (int(value) for value in meta.tolist())
    inputs = torch.randn((args.m, k), device=device, dtype=torch.float16).mul_(0.1)
    current_out = torch.empty((args.m, hidden), device=device, dtype=torch.float16)
    qpn_out = torch.empty_like(current_out)

    def launch_current() -> None:
        sm70_ops.fp8_gemm_sm70_out(
            current_out,
            inputs,
            tm_weight,
            tm_scales,
            128,
            k_ld,
            q_ld,
            True,
        )

    def launch_qpn() -> None:
        sm70_ops.fp8_qpn8_gated_pair_sm70_out(
            qpn_out,
            inputs,
            codes,
            group_scales,
            8,
            2,
            True,
            True,
        )

    current_us = _time(launch_current, args.warmup, args.repeats)
    qpn_us = _time(launch_qpn, args.warmup, args.repeats)
    reference_raw = inputs.float().matmul(_dequantized_weight(qweight, scales).t())
    gate, up = reference_raw.chunk(2, dim=1)
    reference = (torch.nn.functional.silu(gate) * up).half()
    quality = {
        "current": _error_stats(current_out, reference),
        "qpn8_pair": _error_stats(qpn_out, reference),
    }

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("current_gate_up_fused")
    launch_current()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("qpn8_gate_up_fused_pair")
    launch_qpn()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    torch.cuda.cudart().cudaProfilerStop()

    result = {
        "device": torch.cuda.get_device_name(device),
        "m": args.m,
        "n": n,
        "k": k,
        "hidden": hidden,
        "current_unprofiled_us": current_us,
        "qpn8_pair_unprofiled_us": qpn_us,
        "speedup": current_us / qpn_us,
        "qpn8_config": {
            "split_k": 8,
            "nacc": 2,
            "decoder": "fast",
            "prefetch": True,
        },
        "quality": quality,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
