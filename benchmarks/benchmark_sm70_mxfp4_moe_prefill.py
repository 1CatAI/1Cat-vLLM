# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare legacy and grouped SM70 MXFP4 MoE prefill dispatch.

The default shape is one DeepSeek-V4-Flash TP8 MoE stage after routing:
1024 prompt tokens, top-k=6, and 256 local experts. Inputs and routing remain
fixed between routes so any drift is an operator correctness failure.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
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


def _cuda_device_module():
    if not torch.accelerator.is_available():
        raise RuntimeError("CUDA is required")
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None or accelerator.type != "cuda":
        raise RuntimeError(f"CUDA is required, got {accelerator}")
    return torch.get_device_module(accelerator)


def _require_sm70() -> None:
    capability = _cuda_device_module().get_device_capability()
    if capability != (7, 0):
        raise RuntimeError(f"This benchmark requires SM70, got SM{capability}")
    for op_name in (
        "mxfp4_sm70_prepare",
        "mxfp4_moe_dense_stage_sm70_out",
        "awq_moe_build_strided_ptrs",
    ):
        if not hasattr(torch.ops._C, op_name):
            raise RuntimeError(f"Required operator is missing: _C::{op_name}")


def _expert_pattern(num_experts: int, device: torch.device) -> torch.Tensor:
    nibble = torch.arange(num_experts, dtype=torch.int32, device=device) & 0xF
    pattern = nibble.clone()
    for shift in range(4, 32, 4):
        pattern |= nibble << shift
    return pattern


def _prepare_experts(
    shape: StageShape, num_experts: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qweight = torch.randint(0, 16, (shape.k, shape.n), dtype=torch.uint8, device=device)
    scales = torch.full((shape.k // 32, shape.n), 127, dtype=torch.uint8, device=device)
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


def _routing_counts(
    pattern: str,
    routed_rows: int,
    num_experts: int,
    seed: int,
) -> list[int]:
    if pattern == "balanced":
        if routed_rows % num_experts:
            raise ValueError("balanced routing requires divisible routed rows")
        return [routed_rows // num_experts] * num_experts
    if pattern == "half_active":
        active_experts = num_experts // 2
        if routed_rows % active_experts:
            raise ValueError("half-active routing requires divisible routed rows")
        return [routed_rows // active_experts] * active_experts + [0] * (
            num_experts - active_experts
        )
    if pattern == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        assignments = torch.randint(
            num_experts,
            (routed_rows,),
            generator=generator,
            dtype=torch.int64,
        )
        return torch.bincount(assignments, minlength=num_experts).tolist()
    raise ValueError(f"Unknown routing pattern: {pattern}")


def _offsets_from_counts(counts: list[int], device: torch.device) -> torch.Tensor:
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def _measure(call, repeats: int) -> dict[str, object]:
    for _ in range(3):
        call()
    torch.accelerator.synchronize()

    gpu_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(repeats):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        torch.accelerator.synchronize()
        wall_start = time.perf_counter()
        start.record()
        call()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000)
        gpu_ms.append(start.elapsed_time(end))
    return {
        "gpu_ms": gpu_ms,
        "gpu_median_ms": statistics.median(gpu_ms),
        "wall_ms": wall_ms,
        "wall_median_ms": statistics.median(wall_ms),
    }


def _diff_summary(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    counts: list[int],
) -> dict[str, object]:
    mismatch = reference != candidate
    row_mismatch = mismatch.any(dim=1)
    routed_rows = sum(counts)
    mismatch_rows = row_mismatch.nonzero().flatten()
    first_element = mismatch.nonzero()

    expert_ids = []
    row_start = 0
    for expert_id, count in enumerate(counts):
        row_end = row_start + count
        if count and bool(row_mismatch[row_start:row_end].any().item()):
            expert_ids.append(expert_id)
        row_start = row_end

    return {
        "mismatch_elements": int(mismatch.sum().item()),
        "mismatch_rows": int(row_mismatch.sum().item()),
        "active_mismatch_elements": int(mismatch[:routed_rows].sum().item()),
        "active_mismatch_rows": int(row_mismatch[:routed_rows].sum().item()),
        "tail_mismatch_elements": int(mismatch[routed_rows:].sum().item()),
        "tail_mismatch_rows": int(row_mismatch[routed_rows:].sum().item()),
        "mismatched_experts": expert_ids,
        "first_mismatch_element": (
            first_element[0].cpu().tolist() if first_element.numel() else None
        ),
        "first_mismatch_row": (
            int(mismatch_rows[0].item()) if mismatch_rows.numel() else None
        ),
        "sign_flip_elements": int(((reference < 0) != (candidate < 0)).sum().item()),
        "argmax_change_rows": int(
            (reference.argmax(dim=1) != candidate.argmax(dim=1)).sum().item()
        ),
    }


def _run_stage(
    shape: StageShape,
    *,
    prompt_tokens: int,
    capacity_prompt_tokens: int,
    top_k: int,
    num_experts: int,
    pattern: str,
    grouped_experts_per_launch: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    routed_rows = prompt_tokens * top_k
    capacity_rows = capacity_prompt_tokens * top_k
    if capacity_rows < routed_rows:
        raise ValueError("capacity rows must cover all routed rows")
    torch.manual_seed(seed)
    device = torch.device("cuda")
    weights, scales, ptrs_w, ptrs_s = _prepare_experts(shape, num_experts, device)
    input_tensor = (
        torch.randn(capacity_rows, shape.k, dtype=torch.float16, device=device) * 0.01
    )
    output = torch.empty(capacity_rows, shape.n, dtype=torch.float16, device=device)
    counts = _routing_counts(pattern, routed_rows, num_experts, seed)
    expert_offsets = _offsets_from_counts(counts, device)
    dense_ids = torch.arange(num_experts, dtype=torch.int32, device=device)

    def call() -> None:
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

    original = os.environ.get("VLLM_SM70_MXFP4_MOE_GROUPED_PREFILL")
    batch_env = "VLLM_SM70_MXFP4_MOE_GROUPED_PREFILL_EXPERTS_PER_LAUNCH"
    original_batch = os.environ.get(batch_env)
    try:
        os.environ[batch_env] = str(grouped_experts_per_launch)
        os.environ["VLLM_SM70_MXFP4_MOE_GROUPED_PREFILL"] = "0"
        output.fill_(7.0)
        call()
        torch.accelerator.synchronize()
        legacy_output = output.clone()

        os.environ["VLLM_SM70_MXFP4_MOE_GROUPED_PREFILL"] = "1"
        output.fill_(7.0)
        call()
        torch.accelerator.synchronize()
        grouped_output = output.clone()
        cross_route_equal = torch.equal(legacy_output, grouped_output)
        cross_route_max_abs = float((legacy_output - grouped_output).abs().max().item())

        output.fill_(7.0)
        call()
        torch.accelerator.synchronize()
        repeated_output = output
        repeat_equal = torch.equal(grouped_output, repeated_output)
        repeat_max_abs = float((grouped_output - repeated_output).abs().max().item())

        os.environ["VLLM_SM70_MXFP4_MOE_GROUPED_PREFILL"] = "0"
        legacy_timing = _measure(call, repeats)
        os.environ["VLLM_SM70_MXFP4_MOE_GROUPED_PREFILL"] = "1"
        grouped_timing = _measure(call, repeats)
    finally:
        if original is None:
            os.environ.pop("VLLM_SM70_MXFP4_MOE_GROUPED_PREFILL", None)
        else:
            os.environ["VLLM_SM70_MXFP4_MOE_GROUPED_PREFILL"] = original
        if original_batch is None:
            os.environ.pop(batch_env, None)
        else:
            os.environ[batch_env] = original_batch

    result = {
        "stage": shape.name,
        "k": shape.k,
        "n": shape.n,
        "prompt_tokens": prompt_tokens,
        "capacity_prompt_tokens": capacity_prompt_tokens,
        "top_k": top_k,
        "routed_rows": routed_rows,
        "capacity_rows": capacity_rows,
        "num_experts": num_experts,
        "routing_pattern": pattern,
        "active_experts": sum(count > 0 for count in counts),
        "grouped_experts_per_launch": grouped_experts_per_launch,
        "legacy": legacy_timing,
        "grouped": grouped_timing,
        "gpu_speedup": (
            legacy_timing["gpu_median_ms"] / grouped_timing["gpu_median_ms"]
        ),
        "wall_speedup": (
            legacy_timing["wall_median_ms"] / grouped_timing["wall_median_ms"]
        ),
        "cross_route_bitwise": cross_route_equal,
        "cross_route_max_abs": cross_route_max_abs,
        "cross_route_diff": _diff_summary(legacy_output, grouped_output, counts),
        "grouped_repeat_bitwise": repeat_equal,
        "grouped_repeat_max_abs": repeat_max_abs,
    }
    if not cross_route_equal or not repeat_equal:
        raise RuntimeError(f"Grouped MXFP4 prefill correctness gate failed: {result}")

    torch.accelerator.synchronize()
    del weights, scales
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("w13", "w2", "both"), default="both")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument(
        "--capacity-prompt-tokens",
        type=int,
        help=(
            "Graph-safe token capacity. Defaults to --prompt-tokens; use 2048 "
            "to reproduce a 1024-token request in a 2048-token staging buffer."
        ),
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--grouped-experts-per-launch", type=int, default=64)
    parser.add_argument(
        "--routing-pattern",
        choices=("balanced", "random", "half_active", "both", "all"),
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3")
    if args.grouped_experts_per_launch < 1:
        raise ValueError("--grouped-experts-per-launch must be positive")
    _require_sm70()
    capacity_prompt_tokens = args.capacity_prompt_tokens or args.prompt_tokens
    stages = STAGES.values() if args.stage == "both" else (STAGES[args.stage],)
    if args.routing_pattern == "both":
        patterns = ("balanced", "half_active")
    elif args.routing_pattern == "all":
        patterns = ("balanced", "random", "half_active")
    else:
        patterns = (args.routing_pattern,)
    results = []
    for stage_index, shape in enumerate(stages):
        for pattern_index, pattern in enumerate(patterns):
            results.append(
                _run_stage(
                    shape,
                    prompt_tokens=args.prompt_tokens,
                    capacity_prompt_tokens=capacity_prompt_tokens,
                    top_k=args.top_k,
                    num_experts=args.num_experts,
                    pattern=pattern,
                    grouped_experts_per_launch=args.grouped_experts_per_launch,
                    repeats=args.repeats,
                    seed=args.seed + stage_index * 10 + pattern_index,
                )
            )
    print(
        json.dumps(
            {
                "benchmark": "sm70_mxfp4_moe_grouped_prefill",
                "device": _cuda_device_module().get_device_name(),
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
