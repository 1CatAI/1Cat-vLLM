# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile only the real-shape QPN8 winners with Nsight Compute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import vllm._C  # noqa: F401
from benchmark_sm70_fp8_qpn8 import (
    _load_case,
    _qpn8_group_scales,
    _qpn8_prepack,
)

from vllm import _sm70_ops as sm70_ops

_CONFIGS = {
    "down": (16, 1, True),
    "gdn_in": (32, 1, True),
    "output": (16, 1, True),
    "full_qkv": (16, 1, True),
    "gate_up_raw": (16, 1, True),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/ymzx/models/Qwen3.8-27B-FP8"),
    )
    parser.add_argument("--qpn8-library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--paired-current", action="store_true")
    return parser.parse_args()


def _time(launch: Any, warmup: int, repeats: int) -> float:
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
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("This profiler requires SM70/V100")
    if args.m < 1 or args.m > 8:
        raise ValueError("M must be in [1, 8]")

    prepared: list[dict[str, Any]] = []
    for case, (split_k, nacc, fast_decoder) in _CONFIGS.items():
        qweight, scales, note = _load_case(args.model, case, 4, 0, device)
        n, k = (int(dim) for dim in qweight.shape)
        codes = _qpn8_prepack(qweight.view(torch.uint8))
        group_scales = _qpn8_group_scales(scales, n, k)
        tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(
            qweight, scales, 128, False
        )
        k_ld, q_ld = (int(value) for value in meta.tolist())
        inputs = torch.randn((args.m, k), device=device, dtype=torch.float16).mul_(0.1)
        qpn_output = torch.empty((args.m, n), device=device, dtype=torch.float16)
        current_output = torch.empty_like(qpn_output)

        def launch_qpn(
            output: torch.Tensor = qpn_output,
            inputs: torch.Tensor = inputs,
            codes: torch.Tensor = codes,
            group_scales: torch.Tensor = group_scales,
            split_k: int = split_k,
            nacc: int = nacc,
            fast_decoder: bool = fast_decoder,
        ) -> None:
            sm70_ops.fp8_qpn8_gemm_sm70_out(
                output,
                inputs,
                codes,
                group_scales,
                split_k,
                nacc,
                fast_decoder,
                False,
            )

        def launch_current(
            output: torch.Tensor = current_output,
            inputs: torch.Tensor = inputs,
            tm_weight: torch.Tensor = tm_weight,
            tm_scales: torch.Tensor = tm_scales,
            k_ld: int = k_ld,
            q_ld: int = q_ld,
        ) -> None:
            sm70_ops.fp8_gemm_sm70_out(
                output,
                inputs,
                tm_weight,
                tm_scales,
                128,
                k_ld,
                q_ld,
                False,
            )

        prepared.append(
            {
                "case": case,
                "note": note,
                "m": args.m,
                "n": n,
                "k": k,
                "split_k": split_k,
                "nacc": nacc,
                "decoder": "fast" if fast_decoder else "scalar",
                "launch_qpn": launch_qpn,
                "launch_current": launch_current,
                "tensors": [
                    qweight,
                    scales,
                    codes,
                    group_scales,
                    inputs,
                    qpn_output,
                    current_output,
                    tm_weight,
                    tm_scales,
                ],
            }
        )

    for item in prepared:
        item["qpn_unprofiled_us"] = _time(item["launch_qpn"], args.warmup, args.repeats)
        if args.paired_current:
            item["current_unprofiled_us"] = _time(
                item["launch_current"], args.warmup, args.repeats
            )

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("qpn8_real_shape_winners")
    for item in prepared:
        if args.paired_current:
            torch.cuda.nvtx.range_push(f"current_{item['case']}")
            item["launch_current"]()
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"qpn8_{item['case']}")
        item["launch_qpn"]()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    public = [
        {
            key: value
            for key, value in item.items()
            if key not in ("launch_qpn", "launch_current", "tensors")
        }
        for item in prepared
    ]
    result = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "profiles": public,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
