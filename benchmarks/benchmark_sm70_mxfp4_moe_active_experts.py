# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark dense-expert versus active-expert SM70 MXFP4 MoE stages.

This benchmark uses the exact DeepSeek-V4-Flash TP8 W13/W2 shapes. Both paths
call the same TurboMind per-expert GEMM. They differ only in whether the fixed
CUDA Graph contains all 256 experts or the six routed slots.
"""

from __future__ import annotations

import argparse
import json
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


def _expert_pattern(num_experts: int, device: torch.device) -> torch.Tensor:
    nibble = torch.arange(num_experts, dtype=torch.int32, device=device) & 0xF
    pattern = nibble.clone()
    for shift in range(4, 32, 4):
        pattern |= nibble << shift
    return pattern


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


def _full_offsets(route: list[int], num_experts: int) -> list[int]:
    counts = [0] * num_experts
    for expert_id in route:
        counts[expert_id] += 1
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return offsets


def _capture(fn) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, repeats: int) -> float:
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _validate_permute_contract(
    shape: StageShape,
    *,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_experts: int,
    route: list[int],
) -> dict[str, object]:
    device = ptrs_w.device
    top_k = len(route)
    x = torch.randn(1, shape.k, dtype=torch.float16, device=device) * 0.01
    topk_ids = torch.tensor([route], dtype=torch.int32, device=device)
    token_expert_indices = torch.arange(top_k, dtype=torch.int32, device=device).view(
        1, top_k
    )
    permuted_input = torch.empty(top_k, shape.k, dtype=torch.float16, device=device)
    expert_offsets64 = torch.empty(num_experts + 1, dtype=torch.int64, device=device)
    inv_permuted_idx = torch.empty(1, top_k, dtype=torch.int32, device=device)
    permuted_idx = torch.full((top_k,), top_k, dtype=torch.int32, device=device)
    permuted_experts_id = torch.empty(top_k, dtype=torch.int32, device=device)
    sorted_row_idx = torch.empty(top_k, dtype=torch.int32, device=device)
    topk_ids_for_sort = torch.empty(top_k, dtype=torch.int32, device=device)
    workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
        top_k, num_experts
    )
    workspace = torch.empty(workspace_size, dtype=torch.int8, device=device)
    expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    dense_ids = torch.arange(num_experts, dtype=torch.int32, device=device)
    compact_offsets = torch.arange(top_k + 1, dtype=torch.int32, device=device)
    dense_out = torch.empty(top_k, shape.n, dtype=torch.float16, device=device)
    active_out = torch.empty_like(dense_out)

    def permute_call() -> None:
        permuted_idx.fill_(top_k)
        torch.ops._moe_C.moe_permute_with_scratch(
            x,
            topk_ids,
            token_expert_indices,
            None,
            num_experts,
            num_experts,
            top_k,
            permuted_input,
            expert_offsets64,
            inv_permuted_idx,
            permuted_idx,
            workspace,
            permuted_experts_id,
            sorted_row_idx,
            topk_ids_for_sort,
        )
        expert_offsets.copy_(expert_offsets64, non_blocking=True)

    def active_graph_call() -> None:
        permute_call()
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            active_out,
            permuted_input,
            compact_offsets,
            permuted_experts_id,
            ptrs_w,
            ptrs_s,
            top_k,
            shape.k,
            shape.n,
            32,
        )

    permute_call()
    sm70_ops.mxfp4_moe_dense_stage_sm70_out(
        dense_out,
        permuted_input,
        expert_offsets,
        dense_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        shape.k,
        shape.n,
        32,
    )
    active_graph_call()
    torch.cuda.synchronize()
    expected_sorted = sorted(route)
    actual_sorted = permuted_experts_id.cpu().tolist()
    initial_equal = torch.equal(dense_out, active_out)
    initial_max_abs = float((dense_out - active_out).abs().max().item())

    graph = _capture(active_graph_call)
    route_a_graph_out = active_out.clone()
    route_b = [1, 9, 63, 111, 177, 240]
    topk_ids.copy_(torch.tensor([route_b], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    route_b_graph_out = active_out.clone()
    graph_sorted = permuted_experts_id.cpu().tolist()

    permute_call()
    sm70_ops.mxfp4_moe_dense_stage_sm70_out(
        dense_out,
        permuted_input,
        expert_offsets,
        dense_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        shape.k,
        shape.n,
        32,
    )
    torch.cuda.synchronize()
    graph_equal = torch.equal(dense_out, route_b_graph_out)
    graph_max_abs = float((dense_out - route_b_graph_out).abs().max().item())
    graph_route_changes_output = not torch.equal(route_a_graph_out, route_b_graph_out)
    result = {
        "bitwise_equal": initial_equal,
        "max_abs": initial_max_abs,
        "expected_sorted_expert_ids": expected_sorted,
        "actual_sorted_expert_ids": actual_sorted,
        "sorted_expert_ids_match": actual_sorted == expected_sorted,
        "full_graph_dynamic_replay_bitwise_equal": graph_equal,
        "full_graph_dynamic_replay_max_abs": graph_max_abs,
        "full_graph_expected_sorted_expert_ids": sorted(route_b),
        "full_graph_actual_sorted_expert_ids": graph_sorted,
        "full_graph_sorted_expert_ids_match": graph_sorted == sorted(route_b),
        "full_graph_dynamic_route_changes_output": graph_route_changes_output,
    }
    if not all(
        (
            result["bitwise_equal"],
            result["sorted_expert_ids_match"],
            result["full_graph_dynamic_replay_bitwise_equal"],
            result["full_graph_sorted_expert_ids_match"],
            result["full_graph_dynamic_route_changes_output"],
        )
    ):
        raise RuntimeError(f"MXFP4 permute contract gate failed: {result}")
    return result


def benchmark_stage(
    shape: StageShape,
    *,
    num_experts: int,
    top_k: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    _weights, _scales, ptrs_w, ptrs_s = _prepare_experts(shape, num_experts, device)

    route_a = [3, 17, 42, 99, 128, 255]
    route_b = [1, 9, 63, 111, 177, 240]
    if top_k != len(route_a) or max(route_a + route_b) >= num_experts:
        raise ValueError("The exact benchmark requires 256 experts and top-k=6")

    x = torch.randn(top_k, shape.k, dtype=torch.float16, device=device) * 0.01
    dense_out = torch.empty(top_k, shape.n, dtype=torch.float16, device=device)
    active_out = torch.empty_like(dense_out)
    dense_ids = torch.arange(num_experts, dtype=torch.int32, device=device)
    active_ids = torch.tensor(route_a, dtype=torch.int32, device=device)
    dense_offsets = torch.tensor(
        _full_offsets(route_a, num_experts), dtype=torch.int32, device=device
    )
    active_offsets = torch.arange(top_k + 1, dtype=torch.int32, device=device)

    def dense_call() -> None:
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            dense_out,
            x,
            dense_offsets,
            dense_ids,
            ptrs_w,
            ptrs_s,
            num_experts,
            shape.k,
            shape.n,
            32,
        )

    def active_call() -> None:
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            active_out,
            x,
            active_offsets,
            active_ids,
            ptrs_w,
            ptrs_s,
            top_k,
            shape.k,
            shape.n,
            32,
        )

    dense_call()
    active_call()
    torch.cuda.synchronize()
    initial_equal = torch.equal(dense_out, active_out)
    initial_max_abs = float((dense_out - active_out).abs().max().item())

    dense_graph = _capture(dense_call)
    active_graph = _capture(active_call)
    dense_ms = _time_graph(dense_graph, repeats)
    active_ms = _time_graph(active_graph, repeats)

    active_ids.copy_(torch.tensor(route_b, dtype=torch.int32, device=device))
    active_graph.replay()
    torch.cuda.synchronize()
    route_b_graph_out = active_out.clone()

    dense_offsets.copy_(
        torch.tensor(
            _full_offsets(route_b, num_experts), dtype=torch.int32, device=device
        )
    )
    dense_call()
    torch.cuda.synchronize()
    replay_equal = torch.equal(dense_out, route_b_graph_out)
    replay_max_abs = float((dense_out - route_b_graph_out).abs().max().item())

    active_ids.copy_(torch.tensor(route_a, dtype=torch.int32, device=device))
    active_graph.replay()
    torch.cuda.synchronize()
    route_a_graph_out = active_out.clone()
    route_changes_output = not torch.equal(route_a_graph_out, route_b_graph_out)

    result = {
        "stage": shape.name,
        "k": shape.k,
        "n": shape.n,
        "num_experts": num_experts,
        "top_k": top_k,
        "dense_graph_ms": dense_ms,
        "active_graph_ms": active_ms,
        "speedup": dense_ms / active_ms,
        "saved_expert_launches_per_stage": num_experts - top_k,
        "initial_bitwise_equal": initial_equal,
        "initial_max_abs": initial_max_abs,
        "dynamic_replay_bitwise_equal": replay_equal,
        "dynamic_replay_max_abs": replay_max_abs,
        "dynamic_route_changes_output": route_changes_output,
        "moe_permute_contract": _validate_permute_contract(
            shape,
            ptrs_w=ptrs_w,
            ptrs_s=ptrs_s,
            num_experts=num_experts,
            route=list(reversed(route_a)),
        ),
    }
    if not initial_equal or not replay_equal or not route_changes_output:
        raise RuntimeError(f"MXFP4 active-expert correctness gate failed: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("w13", "w2", "both"), default="both")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_sm70()
    selected = STAGES.values() if args.stage == "both" else (STAGES[args.stage],)
    results = [
        benchmark_stage(
            shape,
            num_experts=args.num_experts,
            top_k=args.top_k,
            repeats=args.repeats,
            seed=args.seed + index,
        )
        for index, shape in enumerate(selected)
    ]
    print(
        json.dumps(
            {
                "benchmark": "sm70_mxfp4_moe_active_experts",
                "device": torch.cuda.get_device_name(),
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
