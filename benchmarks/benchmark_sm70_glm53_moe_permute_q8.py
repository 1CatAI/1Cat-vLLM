# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the fixed-shape GLM-5.3 q8 MoE permutation on SM70."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    _prepare_compact_expert_groups,
)

TOKENS = 8
TOP_K = 8
SLOTS = TOKENS * TOP_K
HIDDEN = 4096
EXPERTS = 288


def _capture(call: Callable[[], None]) -> torch.cuda.CUDAGraph:
    for _ in range(8):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, replays: int, repeats: int) -> list[float]:
    for _ in range(32):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / replays)
    return samples


def _buffers(device: torch.device) -> dict[str, torch.Tensor]:
    workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(SLOTS, EXPERTS)
    return {
        "permuted_input": torch.empty(
            (SLOTS, HIDDEN), device=device, dtype=torch.float16
        ),
        "expert_offsets64": torch.empty(
            (EXPERTS + 1,), device=device, dtype=torch.int64
        ),
        "expert_offsets32": torch.empty(
            (EXPERTS + 1,), device=device, dtype=torch.int32
        ),
        "inv_permuted_idx": torch.empty(
            (TOKENS, TOP_K), device=device, dtype=torch.int32
        ),
        "permuted_idx": torch.empty((SLOTS,), device=device, dtype=torch.int32),
        "sort_workspace": torch.empty(
            (workspace_size,), device=device, dtype=torch.int8
        ),
        "permuted_experts_id": torch.empty((SLOTS,), device=device, dtype=torch.int32),
        "sorted_row_idx": torch.empty((SLOTS,), device=device, dtype=torch.int32),
        "topk_ids_for_sort": torch.empty((SLOTS,), device=device, dtype=torch.int32),
        "topk_ids": torch.empty((TOKENS, TOP_K), device=device, dtype=torch.int32),
        "compact_offsets": torch.empty((SLOTS + 1,), device=device, dtype=torch.int32),
        "active_expert_ids": torch.empty((SLOTS,), device=device, dtype=torch.int32),
        "output": torch.empty((TOKENS, HIDDEN), device=device, dtype=torch.float16),
    }


def _generic(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    *,
    zero_output: bool,
) -> None:
    if zero_output:
        buffers["output"].zero_()
    buffers["topk_ids"].copy_(topk_ids, non_blocking=True)
    buffers["permuted_idx"].fill_(SLOTS)
    torch.ops._moe_C.moe_permute_with_scratch(
        x,
        buffers["topk_ids"],
        token_expert_indices,
        None,
        EXPERTS,
        EXPERTS,
        TOP_K,
        buffers["permuted_input"],
        buffers["expert_offsets64"],
        buffers["inv_permuted_idx"],
        buffers["permuted_idx"],
        buffers["sort_workspace"],
        buffers["permuted_experts_id"],
        buffers["sorted_row_idx"],
        buffers["topk_ids_for_sort"],
    )
    buffers["expert_offsets32"].copy_(buffers["expert_offsets64"], non_blocking=True)
    _prepare_compact_expert_groups(
        buffers["permuted_experts_id"],
        buffers["compact_offsets"],
        buffers["active_expert_ids"],
    )


def _candidate(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    *,
    shuffle_sort: bool = True,
) -> None:
    os.environ["VLLM_SM70_GLM53_MOE_SHUFFLE_SORT_Q8"] = "1" if shuffle_sort else "0"
    sm70_ops.sm70_glm53_moe_permute_q8_out(
        x,
        topk_ids,
        buffers["permuted_input"],
        buffers["sorted_row_idx"],
        buffers["inv_permuted_idx"],
        buffers["compact_offsets"],
        buffers["active_expert_ids"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--oracle-cases", type=int, default=64)
    parser.add_argument("--replays", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires NVIDIA V100/SM70.")
    if not hasattr(torch.ops._C, "sm70_glm53_moe_permute_q8_out"):
        raise RuntimeError("Rebuild vLLM with sm70_glm53_moe_permute_q8_out.")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(
        (TOKENS, HIDDEN), device=device, dtype=torch.float16, generator=generator
    )
    token_expert_indices = torch.arange(SLOTS, device=device, dtype=torch.int32).view(
        TOKENS, TOP_K
    )
    reference_buffers = _buffers(device)
    candidate_buffers = _buffers(device)
    all_exact = True
    failures = []
    timed_topk_ids = None

    for case in range(args.oracle_cases):
        if case == 0:
            # Real layer-3 route pattern retained by the GLM q8 MoE benchmark.
            topk_ids = torch.tensor(
                [
                    [218, 179, 286, 10, 268, 171, 220, 65],
                    [218, 268, 179, 286, 10, 254, 220, 85],
                    [218, 268, 179, 286, 10, 185, 85, 250],
                    [218, 268, 179, 85, 185, 254, 286, 10],
                    [218, 268, 85, 254, 185, 179, 286, 10],
                    [218, 268, 254, 85, 185, 179, 27, 286],
                    [218, 268, 254, 85, 185, 179, 219, 27],
                    [218, 268, 254, 185, 85, 219, 179, 27],
                ],
                device=device,
                dtype=torch.int32,
            )
            timed_topk_ids = topk_ids
        elif case == 1:
            topk_ids = torch.zeros((TOKENS, TOP_K), device=device, dtype=torch.int32)
        else:
            topk_ids = torch.randint(
                0,
                EXPERTS,
                (TOKENS, TOP_K),
                device=device,
                dtype=torch.int32,
                generator=generator,
            )

        _generic(
            x,
            topk_ids,
            token_expert_indices,
            reference_buffers,
            zero_output=False,
        )
        _candidate(x, topk_ids, candidate_buffers)
        torch.cuda.synchronize()
        expected_sorted_experts = topk_ids.view(-1).index_select(
            0, reference_buffers["sorted_row_idx"]
        )
        candidate_sorted_experts = topk_ids.view(-1).index_select(
            0, candidate_buffers["sorted_row_idx"]
        )
        comparisons = {
            "permuted_input": torch.equal(
                candidate_buffers["permuted_input"],
                reference_buffers["permuted_input"],
            ),
            "sorted_row_idx": torch.equal(
                candidate_buffers["sorted_row_idx"],
                reference_buffers["sorted_row_idx"],
            ),
            "sorted_experts": torch.equal(
                candidate_sorted_experts, expected_sorted_experts
            ),
            "inv_permuted_idx": torch.equal(
                candidate_buffers["inv_permuted_idx"],
                reference_buffers["inv_permuted_idx"],
            ),
            "compact_offsets": torch.equal(
                candidate_buffers["compact_offsets"],
                reference_buffers["compact_offsets"],
            ),
            "active_expert_ids": torch.equal(
                candidate_buffers["active_expert_ids"],
                reference_buffers["active_expert_ids"],
            ),
        }
        if not all(comparisons.values()):
            all_exact = False
            failures.append({"case": case, "comparisons": comparisons})
            break

    assert timed_topk_ids is not None
    generic_graph = _capture(
        lambda: _generic(
            x,
            timed_topk_ids,
            token_expert_indices,
            reference_buffers,
            zero_output=False,
        )
    )
    generic_zero_graph = _capture(
        lambda: _generic(
            x,
            timed_topk_ids,
            token_expert_indices,
            reference_buffers,
            zero_output=True,
        )
    )
    shared_sort_graph = _capture(
        lambda: _candidate(x, timed_topk_ids, candidate_buffers, shuffle_sort=False)
    )
    shuffle_sort_graph = _capture(
        lambda: _candidate(x, timed_topk_ids, candidate_buffers, shuffle_sort=True)
    )
    timings = {}
    for name, graph in (
        ("generic", generic_graph),
        ("generic_with_output_zero", generic_zero_graph),
        ("shared_sort", shared_sort_graph),
        ("shuffle_sort", shuffle_sort_graph),
    ):
        samples = _time_graph(graph, args.replays, args.repeats)
        timings[name] = {
            "samples_us": samples,
            "median_us": statistics.median(samples),
            "mean_us": statistics.fmean(samples),
        }
    timings["shuffle_sort"]["speedup_vs_generic"] = (
        timings["generic"]["median_us"] / timings["shuffle_sort"]["median_us"]
    )
    timings["shuffle_sort"]["speedup_vs_current"] = (
        timings["generic_with_output_zero"]["median_us"]
        / timings["shuffle_sort"]["median_us"]
    )
    timings["shuffle_sort"]["speedup_vs_shared_sort"] = (
        timings["shared_sort"]["median_us"] / timings["shuffle_sort"]["median_us"]
    )

    payload = {
        "contract": {
            "tokens": TOKENS,
            "top_k": TOP_K,
            "slots": SLOTS,
            "hidden": HIDDEN,
            "experts": EXPERTS,
            "dtype": "float16",
            "cuda_graph": True,
        },
        "oracle_cases": args.oracle_cases,
        "all_outputs_bitwise_equal": all_exact,
        "failures": failures,
        "timings": timings,
    }
    encoded = json.dumps(payload, indent=2)
    print(encoded)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
