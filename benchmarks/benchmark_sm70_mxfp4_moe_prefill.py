# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check SM70 MXFP4 MoE prefill latency and repeatability.

The default shape models one DeepSeek-V4-Flash TP8 MoE stage after routing:
1024 prompt tokens, top-k=6, and 256 evenly populated experts. The benchmark
keeps all inputs and routing metadata fixed so any output drift is an operator
correctness failure rather than a routing or sampling effect.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import torch

from vllm import _sm70_ops as sm70_ops


@dataclass(frozen=True)
class StageShape:
    name: str
    k: int
    n: int


STAGES = {
    "w13": StageShape("w13", 4096, 512),
    "w2": StageShape("w2", 256, 4096),
}


def _expert_pattern(num_experts: int, device: torch.device) -> torch.Tensor:
    nibble = torch.arange(num_experts, dtype=torch.int32, device=device) & 0xF
    pattern = nibble.clone()
    for shift in range(4, 32, 4):
        pattern |= nibble << shift
    return pattern


def _require_sm70() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (7, 0):
        raise RuntimeError(f"This benchmark requires SM70, got SM{major}{minor}")
    for op_name in (
        "mxfp4_sm70_prepare",
        "mxfp4_moe_dense_stage_sm70_out",
        "awq_moe_build_strided_ptrs",
    ):
        if not hasattr(torch.ops._C, op_name):
            raise RuntimeError(f"Required operator is missing: _C::{op_name}")


def _prepare_experts(
    shape: StageShape, num_experts: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qweight = torch.randint(
        0,
        16,
        (shape.k, shape.n),
        dtype=torch.uint8,
        device=device,
    )
    scales = torch.full(
        (shape.k // 32, shape.n),
        127,
        dtype=torch.uint8,
        device=device,
    )
    weight, prepared_scales, meta = sm70_ops.mxfp4_sm70_prepare(qweight, scales, 32)
    weights = weight.unsqueeze(0).repeat(num_experts, 1, 1)
    weights.bitwise_xor_(_expert_pattern(num_experts, device)[:, None, None])
    expert_scales = prepared_scales.unsqueeze(0).repeat(num_experts, 1, 1)
    ptrs_w, ptrs_s = sm70_ops.awq_moe_build_strided_ptrs(
        weights,
        expert_scales,
        int(meta[0].item()),
        int(meta[1].item()),
        num_experts,
    )
    return weights, expert_scales, ptrs_w, ptrs_s


def _run_stage(
    shape: StageShape,
    *,
    prompt_tokens: int,
    top_k: int,
    num_experts: int,
    repeats: int,
    zero_output: bool,
    route: str,
    seed: int,
) -> dict[str, object]:
    routed_rows = prompt_tokens * top_k
    if routed_rows % num_experts != 0:
        raise ValueError("prompt_tokens * top_k must be divisible by num_experts")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    weights, scales, ptrs_w, ptrs_s = _prepare_experts(shape, num_experts, device)
    input_tensor = (
        torch.randn(routed_rows, shape.k, dtype=torch.float16, device=device) * 0.01
    )
    output = torch.empty(routed_rows, shape.n, dtype=torch.float16, device=device)
    rows_per_expert = routed_rows // num_experts
    expert_offsets = torch.arange(
        0,
        routed_rows + 1,
        rows_per_expert,
        dtype=torch.int32,
        device=device,
    )
    dense_ids = torch.arange(num_experts, dtype=torch.int32, device=device)

    def call() -> None:
        if zero_output:
            output.zero_()
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            output,
            input_tensor,
            expert_offsets,
            dense_ids,
            ptrs_w,
            ptrs_s,
            num_experts,
            shape.k,
            shape.n,
            32,
        )

    def measure(grouped: bool) -> tuple[dict[str, object], torch.Tensor]:
        os.environ["VLLM_SM70_MXFP4_MOE_GROUPED_PREFILL"] = "1" if grouped else "0"
        reference: torch.Tensor | None = None
        times_ms: list[float] = []
        bitwise_by_repeat: list[bool] = []
        max_abs_by_repeat: list[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))
            if reference is None:
                reference = output.clone()
            else:
                bitwise_by_repeat.append(torch.equal(output, reference))
                max_abs_by_repeat.append(float((output - reference).abs().max().item()))
        assert reference is not None
        return (
            {
                "route": "grouped" if grouped else "legacy_loop",
                "times_ms": times_ms,
                "repeat_bitwise": bitwise_by_repeat,
                "repeat_max_abs": max_abs_by_repeat,
                "all_repeats_bitwise": all(bitwise_by_repeat),
            },
            reference,
        )

    route_results: list[dict[str, object]] = []
    references: list[torch.Tensor] = []
    if route in ("legacy", "compare"):
        legacy_result, legacy_reference = measure(False)
        route_results.append(legacy_result)
        references.append(legacy_reference)
    if route in ("grouped", "compare"):
        grouped_result, grouped_reference = measure(True)
        route_results.append(grouped_result)
        references.append(grouped_reference)

    cross_route_bitwise = None
    cross_route_max_abs = None
    if route == "compare":
        cross_route_bitwise = torch.equal(references[0], references[1])
        cross_route_max_abs = float((references[0] - references[1]).abs().max().item())

    # Keep owning tensors alive until all asynchronous pointer-table uses finish.
    torch.cuda.synchronize()
    del weights, scales
    return {
        "stage": shape.name,
        "prompt_tokens": prompt_tokens,
        "top_k": top_k,
        "routed_rows": routed_rows,
        "num_experts": num_experts,
        "rows_per_expert": rows_per_expert,
        "zero_output": zero_output,
        "route_results": route_results,
        "cross_route_bitwise": cross_route_bitwise,
        "cross_route_max_abs": cross_route_max_abs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("w13", "w2", "both"), default="both")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--zero-output", action="store_true")
    parser.add_argument(
        "--route",
        choices=("legacy", "grouped", "compare"),
        default="legacy",
    )
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3")
    _require_sm70()
    selected = STAGES.values() if args.stage == "both" else (STAGES[args.stage],)
    results = [
        _run_stage(
            shape,
            prompt_tokens=args.prompt_tokens,
            top_k=args.top_k,
            num_experts=args.num_experts,
            repeats=args.repeats,
            zero_output=args.zero_output,
            route=args.route,
            seed=args.seed + index,
        )
        for index, shape in enumerate(selected)
    ]
    print(
        json.dumps(
            {
                "benchmark": "sm70_mxfp4_moe_prefill_repeatability",
                "device": torch.cuda.get_device_name(),
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
