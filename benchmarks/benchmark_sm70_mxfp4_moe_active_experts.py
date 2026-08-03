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
    expert_offsets = expert_offsets64.to(torch.int32)
    dense_ids = torch.arange(num_experts, dtype=torch.int32, device=device)
    compact_offsets = torch.arange(top_k + 1, dtype=torch.int32, device=device)
    dense_out = torch.empty(top_k, shape.n, dtype=torch.float16, device=device)
    active_out = torch.empty_like(dense_out)
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
    torch.cuda.synchronize()
    expected_sorted = sorted(route)
    actual_sorted = permuted_experts_id.cpu().tolist()
    result = {
        "bitwise_equal": torch.equal(dense_out, active_out),
        "max_abs": float((dense_out - active_out).abs().max().item()),
        "expected_sorted_expert_ids": expected_sorted,
        "actual_sorted_expert_ids": actual_sorted,
        "sorted_expert_ids_match": actual_sorted == expected_sorted,
    }
    if not result["bitwise_equal"] or not result["sorted_expert_ids_match"]:
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


def benchmark_full_pipeline(
    *,
    num_experts: int,
    top_k: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """Compare the generic routed pipeline with direct top-6 decode."""
    if num_experts != 256 or top_k != 6:
        raise ValueError("The full-pipeline benchmark requires 256 experts/top-k=6")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    w13_weights, w13_scales, w13_ptrs_w, w13_ptrs_s = _prepare_experts(
        STAGES["w13"], num_experts, device
    )
    w2_weights, w2_scales, w2_ptrs_w, w2_ptrs_s = _prepare_experts(
        STAGES["w2"], num_experts, device
    )
    # Keep prepared storage alive for the pointer tables.
    _storage = (w13_weights, w13_scales, w2_weights, w2_scales)

    route_a = [255, 3, 128, 17, 99, 42]
    route_b = [240, 1, 177, 9, 111, 63]
    x = torch.randn(1, STAGES["w13"].k, dtype=torch.float16, device=device) * 0.01
    topk_ids = torch.tensor([route_a], dtype=torch.int32, device=device)
    topk_weights = torch.rand(1, top_k, dtype=torch.float32, device=device)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    def make_buffers() -> dict[str, torch.Tensor]:
        workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            top_k, num_experts
        )
        return {
            "output": torch.empty(1, 4096, dtype=torch.float16, device=device),
            "permuted_input": torch.empty(
                top_k, 4096, dtype=torch.float16, device=device
            ),
            "gate_up": torch.empty(top_k, 512, dtype=torch.float16, device=device),
            "intermediate": torch.empty(top_k, 256, dtype=torch.float16, device=device),
            "sorted_output": torch.empty(
                top_k, 4096, dtype=torch.float16, device=device
            ),
            "expert_offsets": torch.empty(
                num_experts + 1, dtype=torch.int32, device=device
            ),
            "expert_offsets64": torch.empty(
                num_experts + 1, dtype=torch.int64, device=device
            ),
            "inv_permuted_idx": torch.empty(1, top_k, dtype=torch.int32, device=device),
            "topk_ids_i32": torch.empty(1, top_k, dtype=torch.int32, device=device),
            "token_expert_indices": torch.arange(
                top_k, dtype=torch.int32, device=device
            ).view(1, top_k),
            "permuted_idx": torch.empty(top_k, dtype=torch.int32, device=device),
            "workspace": torch.empty(workspace_size, dtype=torch.int8, device=device),
            "permuted_experts_id": torch.empty(top_k, dtype=torch.int32, device=device),
            "sorted_row_idx": torch.empty(top_k, dtype=torch.int32, device=device),
            "topk_ids_for_sort": torch.empty(top_k, dtype=torch.int32, device=device),
            "compact_offsets": torch.arange(
                top_k + 1, dtype=torch.int32, device=device
            ),
            "compact_offsets64": torch.arange(
                top_k + 1, dtype=torch.int64, device=device
            ),
        }

    generic = make_buffers()
    direct = make_buffers()

    def generic_call() -> None:
        generic["output"].zero_()
        generic["topk_ids_i32"].copy_(topk_ids)
        generic["permuted_idx"].fill_(top_k)
        torch.ops._moe_C.moe_permute_with_scratch(
            x,
            generic["topk_ids_i32"],
            generic["token_expert_indices"],
            None,
            num_experts,
            num_experts,
            top_k,
            generic["permuted_input"],
            generic["expert_offsets64"],
            generic["inv_permuted_idx"],
            generic["permuted_idx"],
            generic["workspace"],
            generic["permuted_experts_id"],
            generic["sorted_row_idx"],
            generic["topk_ids_for_sort"],
        )
        generic["expert_offsets"].copy_(generic["expert_offsets64"])
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            generic["gate_up"],
            generic["permuted_input"],
            generic["compact_offsets"],
            generic["permuted_experts_id"],
            w13_ptrs_w,
            w13_ptrs_s,
            top_k,
            4096,
            512,
            32,
        )
        torch.ops._C.silu_and_mul_with_clamp(
            generic["intermediate"], generic["gate_up"], 10.0
        )
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            generic["sorted_output"],
            generic["intermediate"],
            generic["compact_offsets"],
            generic["permuted_experts_id"],
            w2_ptrs_w,
            w2_ptrs_s,
            top_k,
            256,
            4096,
            32,
        )
        torch.ops._moe_C.moe_unpermute(
            generic["sorted_output"],
            topk_weights,
            generic["inv_permuted_idx"],
            generic["expert_offsets64"],
            top_k,
            generic["output"],
        )

    def direct_call() -> None:
        sm70_ops.mxfp4_moe_single_token_prepare_w13_sm70_out(
            direct["gate_up"],
            direct["permuted_input"],
            x,
            topk_ids,
            w13_ptrs_w,
            w13_ptrs_s,
            direct["compact_offsets"],
            direct["inv_permuted_idx"],
            direct["permuted_experts_id"],
            4096,
            512,
            32,
            4096,
        )
        torch.ops._C.silu_and_mul_with_clamp(
            direct["intermediate"], direct["gate_up"], 10.0
        )
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            direct["sorted_output"],
            direct["intermediate"],
            direct["compact_offsets"],
            direct["permuted_experts_id"],
            w2_ptrs_w,
            w2_ptrs_s,
            top_k,
            256,
            4096,
            32,
        )
        torch.ops._moe_C.moe_unpermute(
            direct["sorted_output"],
            topk_weights,
            direct["inv_permuted_idx"],
            direct["compact_offsets64"],
            top_k,
            direct["output"],
        )

    generic_call()
    direct_call()
    torch.cuda.synchronize()
    initial_equal = torch.equal(generic["output"], direct["output"])
    initial_max_abs = float((generic["output"] - direct["output"]).abs().max().item())
    stage_parity = {
        "gate_up_equal": torch.equal(generic["gate_up"], direct["gate_up"]),
        "gate_up_max_abs": float(
            (generic["gate_up"] - direct["gate_up"]).abs().max().item()
        ),
        "intermediate_equal": torch.equal(
            generic["intermediate"], direct["intermediate"]
        ),
        "intermediate_max_abs": float(
            (generic["intermediate"] - direct["intermediate"]).abs().max().item()
        ),
        "sorted_output_equal": torch.equal(
            generic["sorted_output"], direct["sorted_output"]
        ),
        "sorted_output_max_abs": float(
            (generic["sorted_output"] - direct["sorted_output"]).abs().max().item()
        ),
        "inv_permuted_idx_equal": torch.equal(
            generic["inv_permuted_idx"], direct["inv_permuted_idx"]
        ),
    }

    generic_graph = _capture(generic_call)
    direct_graph = _capture(direct_call)
    generic_ms = _time_graph(generic_graph, repeats)
    direct_ms = _time_graph(direct_graph, repeats)

    topk_ids.copy_(torch.tensor([route_b], dtype=torch.int32, device=device))
    generic_graph.replay()
    direct_graph.replay()
    torch.cuda.synchronize()
    replay_equal = torch.equal(generic["output"], direct["output"])
    replay_max_abs = float((generic["output"] - direct["output"]).abs().max().item())

    result = {
        "generic_graph_ms": generic_ms,
        "direct_graph_ms": direct_ms,
        "speedup": generic_ms / direct_ms,
        "projected_savings_ms_per_token": (generic_ms - direct_ms) * 43,
        "initial_bitwise_equal": initial_equal,
        "initial_max_abs": initial_max_abs,
        "dynamic_replay_bitwise_equal": replay_equal,
        "dynamic_replay_max_abs": replay_max_abs,
        "initial_stage_parity": stage_parity,
    }
    if not initial_equal or not replay_equal:
        raise RuntimeError(f"MXFP4 direct top-6 correctness gate failed: {result}")
    return result


def profile_active_stage_once(
    shape: StageShape,
    *,
    num_experts: int,
    top_k: int,
    seed: int,
) -> dict[str, object]:
    """Capture one warmed active-expert stage without dense-control kernels."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    _weights, _scales, ptrs_w, ptrs_s = _prepare_experts(shape, num_experts, device)
    route = [3, 17, 42, 99, 128, 255]
    if top_k != len(route) or max(route) >= num_experts:
        raise ValueError("The exact profile requires 256 experts and top-k=6")

    x = torch.randn(top_k, shape.k, dtype=torch.float16, device=device) * 0.01
    active_out = torch.empty(top_k, shape.n, dtype=torch.float16, device=device)
    active_ids = torch.tensor(route, dtype=torch.int32, device=device)
    active_offsets = torch.arange(top_k + 1, dtype=torch.int32, device=device)

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

    for _ in range(5):
        active_call()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    active_call()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    return {
        "stage": shape.name,
        "k": shape.k,
        "n": shape.n,
        "num_experts": num_experts,
        "top_k": top_k,
        "captured_active_expert_launches": top_k,
        "output_finite": bool(torch.isfinite(active_out).all().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("w13", "w2", "both"), default="both")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--full-pipeline", action="store_true")
    parser.add_argument(
        "--profile-active-once",
        action="store_true",
        help="Warm up, then CUDA-profiler capture one six-expert stage.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_sm70()
    if args.full_pipeline:
        result = benchmark_full_pipeline(
            num_experts=args.num_experts,
            top_k=args.top_k,
            repeats=args.repeats,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "benchmark": "sm70_mxfp4_moe_direct_top6_pipeline",
                    "device": torch.cuda.get_device_name(),
                    "result": result,
                },
                indent=2,
            )
        )
        return 0
    if args.profile_active_once:
        if args.stage == "both":
            raise ValueError("--profile-active-once requires --stage w13 or w2")
        result = profile_active_stage_once(
            STAGES[args.stage],
            num_experts=args.num_experts,
            top_k=args.top_k,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "benchmark": "sm70_mxfp4_moe_active_experts_profile",
                    "device": torch.cuda.get_device_name(),
                    "result": result,
                },
                indent=2,
            )
        )
        return 0
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
