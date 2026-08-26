# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark dense-expert versus active-expert SM70 MXFP4 MoE stages.

This benchmark uses the exact DeepSeek-V4-Flash TP8 W13/W2 shapes. Both paths
call the same TurboMind per-expert GEMM. They differ only in whether the fixed
CUDA Graph contains all 256 experts, six B=1 slots, or 48 verifier M=8 slots.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import torch

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.layers.quantization.mxfp4_sm70_moe import (
    _compact_mxfp4_active_experts,
)


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
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.accelerator.synchronize()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, repeats: int) -> float:
    for _ in range(5):
        graph.replay()
    torch.accelerator.synchronize()
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
    torch.accelerator.synchronize()
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
    torch.accelerator.synchronize()
    initial_equal = torch.equal(dense_out, active_out)
    initial_max_abs = float((dense_out - active_out).abs().max().item())

    dense_graph = _capture(dense_call)
    active_graph = _capture(active_call)
    dense_ms = _time_graph(dense_graph, repeats)
    active_ms = _time_graph(active_graph, repeats)

    active_ids.copy_(torch.tensor(route_b, dtype=torch.int32, device=device))
    active_graph.replay()
    torch.accelerator.synchronize()
    route_b_graph_out = active_out.clone()

    dense_offsets.copy_(
        torch.tensor(
            _full_offsets(route_b, num_experts), dtype=torch.int32, device=device
        )
    )
    dense_call()
    torch.accelerator.synchronize()
    replay_equal = torch.equal(dense_out, route_b_graph_out)
    replay_max_abs = float((dense_out - route_b_graph_out).abs().max().item())

    active_ids.copy_(torch.tensor(route_a, dtype=torch.int32, device=device))
    active_graph.replay()
    torch.accelerator.synchronize()
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
    torch.accelerator.synchronize()
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
    torch.accelerator.synchronize()
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


def benchmark_verifier_m8_pipeline(
    *,
    num_experts: int,
    top_k: int,
    repeats: int,
    seed: int,
    route_case: str = "mixed",
    input_scale: float = 0.01,
    require_grouped_bitwise: bool = True,
) -> dict[str, object]:
    """Compare dense-256, active-48 loop, and grouped verifier pipelines."""
    num_tokens = 8
    if num_experts != 256 or top_k != 6:
        raise ValueError("The verifier benchmark requires 256 experts/top-k=6")

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

    route_cases = {
        "mixed": (
            [
                [3, 17 + row, 42, 99 + row, 128 + row, 255 - row]
                for row in range(num_tokens)
            ],
            [
                [1, 9 + row, 63, 111 + row, 177 + row, 240 - row]
                for row in range(num_tokens)
            ],
        ),
        "unique48": (
            [list(range(row * top_k, (row + 1) * top_k)) for row in range(num_tokens)],
            [
                list(range(49 + row * top_k, 49 + (row + 1) * top_k))
                for row in range(num_tokens)
            ],
        ),
        "hot6": (
            [[3, 17, 42, 99, 128, 255] for _ in range(num_tokens)],
            [[1, 9, 63, 111, 177, 240] for _ in range(num_tokens)],
        ),
    }
    route_a, route_b = route_cases[route_case]
    topk_ids = torch.tensor(route_a, dtype=torch.int32, device=device)
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=device)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    x = (
        torch.randn(num_tokens, STAGES["w13"].k, dtype=torch.float16, device=device)
        * input_scale
    )
    total_slots = num_tokens * top_k

    def make_buffers() -> dict[str, torch.Tensor]:
        workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            total_slots, num_experts
        )
        return {
            "output": torch.empty(num_tokens, 4096, dtype=torch.float16, device=device),
            "permuted_input": torch.empty(
                total_slots, 4096, dtype=torch.float16, device=device
            ),
            "gate_up": torch.empty(
                total_slots, 512, dtype=torch.float16, device=device
            ),
            "intermediate": torch.empty(
                total_slots, 256, dtype=torch.float16, device=device
            ),
            "sorted_output": torch.empty(
                total_slots, 4096, dtype=torch.float16, device=device
            ),
            "expert_offsets": torch.empty(
                num_experts + 1, dtype=torch.int32, device=device
            ),
            "expert_offsets64": torch.empty(
                num_experts + 1, dtype=torch.int64, device=device
            ),
            "inv_permuted_idx": torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=device
            ),
            "topk_ids_i32": torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=device
            ),
            "token_expert_indices": torch.arange(
                total_slots, dtype=torch.int32, device=device
            ).view(num_tokens, top_k),
            "permuted_idx": torch.empty(total_slots, dtype=torch.int32, device=device),
            "workspace": torch.empty(workspace_size, dtype=torch.int8, device=device),
            "permuted_experts_id": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "sorted_row_idx": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "topk_ids_for_sort": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
            "dense_expert_ids": torch.arange(
                num_experts, dtype=torch.int32, device=device
            ),
            "compact_offsets": torch.arange(
                total_slots + 1, dtype=torch.int32, device=device
            ),
            "active_expert_ids": torch.empty(
                total_slots, dtype=torch.int32, device=device
            ),
        }

    dense = make_buffers()
    active = make_buffers()
    grouped = make_buffers()

    def pipeline_call(
        buffers: dict[str, torch.Tensor], *, active_only: bool, grouped_m8: bool
    ) -> None:
        os.environ["VLLM_SM70_MXFP4_MOE_GROUPED_M8"] = "1" if grouped_m8 else "0"
        buffers["output"].zero_()
        buffers["topk_ids_i32"].copy_(topk_ids)
        buffers["permuted_idx"].fill_(total_slots)
        torch.ops._moe_C.moe_permute_with_scratch(
            x,
            buffers["topk_ids_i32"],
            buffers["token_expert_indices"],
            None,
            num_experts,
            num_experts,
            top_k,
            buffers["permuted_input"],
            buffers["expert_offsets64"],
            buffers["inv_permuted_idx"],
            buffers["permuted_idx"],
            buffers["workspace"],
            buffers["permuted_experts_id"],
            buffers["sorted_row_idx"],
            buffers["topk_ids_for_sort"],
        )
        buffers["expert_offsets"].copy_(buffers["expert_offsets64"])
        if active_only:
            stage_offsets = buffers["compact_offsets"]
            if grouped_m8:
                stage_expert_ids = buffers["permuted_experts_id"]
            else:
                _compact_mxfp4_active_experts(
                    buffers["permuted_experts_id"],
                    buffers["compact_offsets"],
                    buffers["active_expert_ids"],
                )
                stage_expert_ids = buffers["active_expert_ids"]
            stage_expert_count = total_slots
        else:
            stage_offsets = buffers["expert_offsets"]
            stage_expert_ids = buffers["dense_expert_ids"]
            stage_expert_count = num_experts
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            buffers["gate_up"],
            buffers["permuted_input"],
            stage_offsets,
            stage_expert_ids,
            w13_ptrs_w,
            w13_ptrs_s,
            stage_expert_count,
            4096,
            512,
            32,
        )
        torch.ops._C.silu_and_mul_with_clamp(
            buffers["intermediate"], buffers["gate_up"], 10.0
        )
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            buffers["sorted_output"],
            buffers["intermediate"],
            stage_offsets,
            stage_expert_ids,
            w2_ptrs_w,
            w2_ptrs_s,
            stage_expert_count,
            256,
            4096,
            32,
        )
        torch.ops._moe_C.moe_unpermute(
            buffers["sorted_output"],
            topk_weights,
            buffers["inv_permuted_idx"],
            buffers["expert_offsets64"],
            top_k,
            buffers["output"],
        )

    def dense_call() -> None:
        pipeline_call(dense, active_only=False, grouped_m8=False)

    def active_call() -> None:
        pipeline_call(active, active_only=True, grouped_m8=False)

    def grouped_call() -> None:
        pipeline_call(grouped, active_only=True, grouped_m8=True)

    dense_call()
    active_call()
    grouped_call()
    torch.accelerator.synchronize()
    initial_equal = torch.equal(dense["output"], active["output"])
    grouped_initial_equal = torch.equal(dense["output"], grouped["output"])
    stage_parity = {
        name: torch.equal(dense[name], active[name])
        for name in ("gate_up", "intermediate", "sorted_output", "output")
    }
    grouped_stage_parity = {
        name: torch.equal(dense[name], grouped[name])
        for name in ("gate_up", "intermediate", "sorted_output", "output")
    }
    grouped_stage_max_abs = {
        name: float((dense[name] - grouped[name]).abs().max().item())
        for name in ("gate_up", "intermediate", "sorted_output", "output")
    }
    grouped_stages_finite = all(
        bool(torch.isfinite(grouped[name]).all().item())
        for name in ("gate_up", "intermediate", "sorted_output", "output")
    )
    expected_active_ids = sorted({expert for row in route_a for expert in row})
    actual_active_ids = active["active_expert_ids"][: len(expected_active_ids)]
    compact_ids_match = actual_active_ids.cpu().tolist() == expected_active_ids

    dense_graph = _capture(dense_call)
    active_graph = _capture(active_call)
    grouped_graph = _capture(grouped_call)
    dense_ms = _time_graph(dense_graph, repeats)
    active_ms = _time_graph(active_graph, repeats)
    grouped_ms = _time_graph(grouped_graph, repeats)

    topk_ids.copy_(torch.tensor(route_b, dtype=torch.int32, device=device))
    dense_graph.replay()
    active_graph.replay()
    grouped_graph.replay()
    torch.accelerator.synchronize()
    replay_equal = torch.equal(dense["output"], active["output"])
    replay_max_abs = float((dense["output"] - active["output"]).abs().max().item())
    grouped_replay_equal = torch.equal(dense["output"], grouped["output"])
    grouped_replay_max_abs = float(
        (dense["output"] - grouped["output"]).abs().max().item()
    )
    route_b_output = grouped["output"].clone()

    topk_ids.copy_(torch.tensor(route_a, dtype=torch.int32, device=device))
    grouped_graph.replay()
    torch.accelerator.synchronize()
    route_changes_output = not torch.equal(grouped["output"], route_b_output)

    result = {
        "route_case": route_case,
        "input_scale": input_scale,
        "grouped_bitwise_required": require_grouped_bitwise,
        "num_tokens": num_tokens,
        "total_routed_slots": total_slots,
        "unique_active_experts": len(expected_active_ids),
        "fixed_graph_expert_slots": total_slots,
        "dense_graph_ms": dense_ms,
        "active_graph_ms": active_ms,
        "grouped_graph_ms": grouped_ms,
        "speedup": dense_ms / active_ms,
        "grouped_vs_active_speedup": active_ms / grouped_ms,
        "grouped_projected_savings_ms_per_43_layers": (active_ms - grouped_ms) * 43,
        "projected_savings_ms_per_43_layers": (dense_ms - active_ms) * 43,
        "initial_bitwise_equal": initial_equal,
        "grouped_initial_bitwise_equal": grouped_initial_equal,
        "initial_stage_bitwise_equal": stage_parity,
        "grouped_initial_stage_bitwise_equal": grouped_stage_parity,
        "grouped_initial_stage_max_abs": grouped_stage_max_abs,
        "grouped_stages_finite": grouped_stages_finite,
        "compact_ids_match": compact_ids_match,
        "dynamic_replay_bitwise_equal": replay_equal,
        "dynamic_replay_max_abs": replay_max_abs,
        "grouped_dynamic_replay_bitwise_equal": grouped_replay_equal,
        "grouped_dynamic_replay_max_abs": grouped_replay_max_abs,
        "dynamic_route_changes_output": route_changes_output,
    }
    base_gate_passed = (
        initial_equal
        and all(stage_parity.values())
        and compact_ids_match
        and replay_equal
        and route_changes_output
        and grouped_stages_finite
    )
    grouped_bitwise_gate_passed = (
        grouped_initial_equal
        and all(grouped_stage_parity.values())
        and grouped_replay_equal
    )
    result["base_gate_passed"] = base_gate_passed
    result["grouped_bitwise_gate_passed"] = grouped_bitwise_gate_passed
    if not base_gate_passed or (
        require_grouped_bitwise and not grouped_bitwise_gate_passed
    ):
        raise RuntimeError(f"MXFP4 verifier M8 correctness gate failed: {result}")
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
    torch.accelerator.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    active_call()
    torch.accelerator.synchronize()
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


def profile_grouped_m8_stage_once(
    shape: StageShape,
    *,
    num_experts: int,
    top_k: int,
    seed: int,
    route_case: str,
) -> dict[str, object]:
    """Capture one warmed exact-M8 grouped stage for Nsight Compute."""
    num_tokens = 8
    total_slots = num_tokens * top_k
    if num_experts != 256 or top_k != 6:
        raise ValueError("The exact M8 profile requires 256 experts and top-k=6")

    route_cases = {
        "mixed": [
            [3, 17 + row, 42, 99 + row, 128 + row, 255 - row]
            for row in range(num_tokens)
        ],
        "unique48": [
            list(range(row * top_k, (row + 1) * top_k)) for row in range(num_tokens)
        ],
        "hot6": [[3, 17, 42, 99, 128, 255] for _ in range(num_tokens)],
    }
    routed_experts = sorted(expert for row in route_cases[route_case] for expert in row)

    torch.manual_seed(seed)
    device = torch.device("cuda")
    _weights, _scales, ptrs_w, ptrs_s = _prepare_experts(shape, num_experts, device)
    x = torch.randn(total_slots, shape.k, dtype=torch.float16, device=device) * 0.01
    out = torch.empty(total_slots, shape.n, dtype=torch.float16, device=device)
    slot_offsets = torch.arange(total_slots + 1, dtype=torch.int32, device=device)
    slot_expert_ids = torch.tensor(routed_experts, dtype=torch.int32, device=device)

    os.environ["VLLM_SM70_MXFP4_MOE_COMPACT_GROUPED_DECODE"] = "1"
    os.environ["VLLM_SM70_MXFP4_MOE_GROUPED_M8"] = "1"

    def grouped_call() -> None:
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            out,
            x,
            slot_offsets,
            slot_expert_ids,
            ptrs_w,
            ptrs_s,
            total_slots,
            shape.k,
            shape.n,
            32,
        )

    for _ in range(5):
        grouped_call()
    torch.accelerator.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    grouped_call()
    torch.accelerator.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    return {
        "stage": shape.name,
        "k": shape.k,
        "n": shape.n,
        "num_tokens": num_tokens,
        "total_routed_slots": total_slots,
        "route_case": route_case,
        "unique_active_experts": len(set(routed_experts)),
        "captured_grouped_launches": 1,
        "output_finite": bool(torch.isfinite(out).all().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("w13", "w2", "both"), default="both")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--input-scale", type=float, default=0.01)
    parser.add_argument("--full-pipeline", action="store_true")
    parser.add_argument("--verifier-m8-pipeline", action="store_true")
    parser.add_argument(
        "--allow-grouped-numeric-drift",
        action="store_true",
        help=(
            "Report grouped tactic drift while retaining finite/replay baseline gates."
        ),
    )
    parser.add_argument(
        "--route-case",
        choices=("mixed", "unique48", "hot6"),
        default="mixed",
        help="Verifier M8 expert-overlap distribution.",
    )
    parser.add_argument(
        "--profile-active-once",
        action="store_true",
        help="Warm up, then CUDA-profiler capture one six-expert stage.",
    )
    parser.add_argument(
        "--profile-grouped-m8-once",
        action="store_true",
        help="Warm up, then capture one exact-M8 grouped stage for NCU.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_sm70()
    if args.verifier_m8_pipeline:
        result = benchmark_verifier_m8_pipeline(
            num_experts=args.num_experts,
            top_k=args.top_k,
            repeats=args.repeats,
            seed=args.seed,
            route_case=args.route_case,
            input_scale=args.input_scale,
            require_grouped_bitwise=not args.allow_grouped_numeric_drift,
        )
        print(
            json.dumps(
                {
                    "benchmark": "sm70_mxfp4_moe_verifier_m8_pipeline",
                    "device": torch.cuda.get_device_name(),
                    "result": result,
                },
                indent=2,
            )
        )
        return 0
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
    if args.profile_grouped_m8_once:
        if args.stage == "both":
            raise ValueError("--profile-grouped-m8-once requires --stage w13 or w2")
        result = profile_grouped_m8_stage_once(
            STAGES[args.stage],
            num_experts=args.num_experts,
            top_k=args.top_k,
            seed=args.seed,
            route_case=args.route_case,
        )
        print(
            json.dumps(
                {
                    "benchmark": "sm70_mxfp4_moe_grouped_m8_profile",
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
