# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B the full and token-chunked AWQ W2 output paths on SM70.

The benchmark loads one real Qwen3.8 AWQ MoE layer, applies the same TP shard
and SM70 preprocessing as the service, and checks the final weighted output
bitwise before reporting latency and scratch bytes.
"""

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from benchmarks.benchmark_sm70_turbomind_exactness import (
    _expert_offsets,
    _load_awq_moe_layer,
    _make_input,
    _make_logical_expert_ids,
    _prepare_awq_moe_checkpoint_for_tp,
    _prepare_awq_moe_weights,
)
from vllm import _sm70_ops as sm70_ops


def _time_ms(fn: Callable[[], None], warmups: int, iterations: int) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _make_expert_ids(
    m: int,
    top_k: int,
    num_experts: int,
    device: torch.device,
    pattern: str,
) -> torch.Tensor:
    if not pattern.startswith("hot_route0:"):
        return _make_logical_expert_ids(m * top_k, num_experts, device, pattern)

    hot_expert = int(pattern.split(":", 1)[1])
    if not 0 <= hot_expert < num_experts:
        raise ValueError("hot expert is outside the configured expert range.")
    tokens = torch.arange(m, dtype=torch.int64, device=device).view(-1, 1)
    routes = torch.arange(top_k, dtype=torch.int64, device=device).view(1, -1)
    expert_ids = (tokens * 17 + routes * 53 + 1) % (num_experts - 1)
    expert_ids += expert_ids >= hot_expert
    expert_ids[:, 0] = hot_expert
    return expert_ids.flatten()


def _run_shape(
    m: int,
    top_k: int,
    num_experts: int,
    group_size: int,
    expert_pattern: str,
    device: torch.device,
    w2_ptrs_w: torch.Tensor,
    w2_ptrs_s: torch.Tensor,
    w2_k: int,
    w2_n: int,
    chunk_tokens: int,
    warmups: int,
    iterations: int,
    check_cuda_graph: bool,
) -> dict[str, Any]:
    if chunk_tokens not in (4096, 6144):
        raise ValueError("chunk_tokens must be 4096 or 6144.")
    tail_tokens = m % chunk_tokens
    if m < 8192 or chunk_tokens >= m or 0 < tail_tokens < 2048:
        raise ValueError(
            "chunked W2 requires m >= 8192, chunk_tokens in {4096, 6144}, "
            "chunk_tokens < m, and an empty or >= 2048-token tail."
        )
    total_slots = m * top_k
    logical_expert_ids = _make_expert_ids(m, top_k, num_experts, device, expert_pattern)
    sorted_expert_ids, order = torch.sort(logical_expert_ids, stable=True)
    expert_offsets, _ = _expert_offsets(sorted_expert_ids, num_experts)

    route_input = _make_input(total_slots, w2_k, device)
    sorted_input = route_input[order].contiguous()
    inv_permuted_idx = torch.empty(total_slots, dtype=torch.int32, device=device)
    inv_permuted_idx[order] = torch.arange(
        total_slots, dtype=torch.int32, device=device
    )
    inv_permuted_idx = inv_permuted_idx.view(m, top_k)
    permuted_idx = order.to(torch.int32).contiguous()
    weights = torch.arange(1, top_k + 1, dtype=torch.float32, device=device)
    topk_weights = (weights / weights.sum()).view(1, top_k).expand(m, -1)
    topk_weights = topk_weights.contiguous()

    sorted_output = torch.empty(total_slots, w2_n, dtype=torch.float16, device=device)
    baseline_output = torch.empty(m, w2_n, dtype=torch.float16, device=device)

    chunk_slots = chunk_tokens * top_k
    chunk_output = torch.empty(chunk_slots, w2_n, dtype=torch.float16, device=device)
    chunk_expert_offsets = torch.empty(
        num_experts + 1, dtype=torch.int32, device=device
    )
    chunk_range_begin = torch.empty(num_experts, dtype=torch.int32, device=device)
    chunk_range_end = torch.empty_like(chunk_range_begin)
    chunk_a_indices = torch.empty(chunk_slots, dtype=torch.int32, device=device)
    chunk_inv_permuted_idx = torch.empty_like(chunk_a_indices)
    chunked_output = torch.empty_like(baseline_output)

    def baseline() -> None:
        sm70_ops.awq_moe_gemm_sm70_per_expert_dispatch_out(
            sorted_output,
            sorted_input,
            expert_offsets,
            w2_ptrs_w,
            w2_ptrs_s,
            num_experts,
            w2_k,
            w2_n,
            group_size,
            False,
        )
        torch.ops._moe_C.moe_unpermute(
            sorted_output,
            topk_weights,
            inv_permuted_idx,
            None,
            top_k,
            baseline_output,
        )

    def chunked() -> None:
        sm70_ops.awq_moe_chunked_w2_sm70_out(
            chunked_output,
            chunk_output,
            sorted_input,
            expert_offsets,
            permuted_idx,
            topk_weights,
            chunk_expert_offsets,
            chunk_range_begin,
            chunk_range_end,
            chunk_a_indices,
            chunk_inv_permuted_idx,
            w2_ptrs_w,
            w2_ptrs_s,
            m,
            top_k,
            num_experts,
            w2_k,
            w2_n,
            w2_n,
            group_size,
            chunk_tokens,
        )

    baseline()
    chunked()
    torch.cuda.synchronize()
    first_chunked_output = chunked_output.clone()
    chunked()
    torch.cuda.synchronize()
    mismatches = int((chunked_output != baseline_output).sum().item())
    repeat_mismatches = int((chunked_output != first_chunked_output).sum().item())
    bad_token_indices = torch.nonzero(
        (chunked_output != baseline_output).any(dim=1), as_tuple=False
    ).flatten()
    max_abs_error = float(
        (chunked_output.float() - baseline_output.float()).abs().max().item()
    )
    del first_chunked_output

    graph_bitwise_equal = None
    graph_repeat_bitwise_equal = None
    if check_cuda_graph:
        # Warm the dispatch cache before capture; graph replay must preserve the
        # same route-order reduction result without any address changes.
        chunked()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chunked()
        graph.replay()
        torch.cuda.synchronize()
        graph_first_output = chunked_output.clone()
        graph_bitwise_equal = bool(torch.equal(chunked_output, baseline_output))
        graph.replay()
        torch.cuda.synchronize()
        graph_repeat_bitwise_equal = bool(
            torch.equal(chunked_output, graph_first_output)
        )
        del graph_first_output, graph

    baseline_ms = _time_ms(baseline, warmups, iterations)
    chunked_ms = _time_ms(chunked, warmups, iterations)
    baseline_scratch_bytes = _tensor_bytes(sorted_output)
    chunked_scratch_bytes = _tensor_bytes(
        chunk_output,
        chunk_expert_offsets,
        chunk_range_begin,
        chunk_range_end,
        chunk_a_indices,
        chunk_inv_permuted_idx,
    )
    return {
        "m": m,
        "top_k": top_k,
        "total_slots": total_slots,
        "expert_pattern": expert_pattern,
        "bitwise_equal": mismatches == 0,
        "mismatches": mismatches,
        "bad_token_count": int(bad_token_indices.numel()),
        "first_bad_token": (
            int(bad_token_indices[0].item()) if bad_token_indices.numel() else None
        ),
        "last_bad_token": (
            int(bad_token_indices[-1].item()) if bad_token_indices.numel() else None
        ),
        "nan_count": int(torch.isnan(chunked_output).sum().item()),
        "repeat_bitwise_equal": repeat_mismatches == 0,
        "repeat_mismatches": repeat_mismatches,
        "cuda_graph_checked": check_cuda_graph,
        "cuda_graph_bitwise_equal": graph_bitwise_equal,
        "cuda_graph_repeat_bitwise_equal": graph_repeat_bitwise_equal,
        "max_abs_error": max_abs_error,
        "baseline_ms": baseline_ms,
        "chunk_tokens": chunk_tokens,
        "chunked_ms": chunked_ms,
        "latency_ratio": chunked_ms / baseline_ms,
        "baseline_scratch_bytes": baseline_scratch_bytes,
        "chunked_scratch_bytes": chunked_scratch_bytes,
        "scratch_bytes_saved": baseline_scratch_bytes - chunked_scratch_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", default="model.language_model.layers.0.mlp.experts")
    parser.add_argument("--m", type=int, nargs="+", default=[8192])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-experts", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--chunk-tokens", type=int, nargs="+", default=[4096])
    parser.add_argument("--expert-pattern", default="random_unique:17")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--check-cuda-graph", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("This benchmark requires an SM70 GPU.")
    if not hasattr(torch.ops._C, "awq_moe_chunked_w2_sm70_out"):
        raise RuntimeError("The chunked W2 CUDA op is unavailable.")

    loaded = _load_awq_moe_layer(args.model, args.layer, args.num_experts, device)
    prepared = _prepare_awq_moe_checkpoint_for_tp(
        loaded, args.tp_size, args.tp_rank, args.group_size
    )
    _, _, _, w2_qweight, w2_scales, w2_qzeros, group_size = prepared
    w2_tm_weight, _, w2_ptrs_w, w2_ptrs_s, _, _ = _prepare_awq_moe_weights(
        w2_qweight, w2_scales, w2_qzeros, group_size
    )
    w2_k = int(w2_tm_weight.shape[1])
    w2_n = int(w2_qweight.shape[2]) * 8

    results = [
        _run_shape(
            m,
            args.top_k,
            args.num_experts,
            group_size,
            args.expert_pattern,
            device,
            w2_ptrs_w,
            w2_ptrs_s,
            w2_k,
            w2_n,
            chunk_tokens,
            args.warmups,
            args.iterations,
            args.check_cuda_graph,
        )
        for m in args.m
        for chunk_tokens in args.chunk_tokens
    ]
    report = {
        "device": torch.cuda.get_device_name(device),
        "device_capability": torch.cuda.get_device_capability(device),
        "model": str(args.model),
        "layer": args.layer,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.json_out is not None:
        args.json_out.write_text(encoded + "\n")
    if not all(
        result["bitwise_equal"]
        and result["repeat_bitwise_equal"]
        and (
            not result["cuda_graph_checked"]
            or (
                result["cuda_graph_bitwise_equal"]
                and result["cuda_graph_repeat_bitwise_equal"]
            )
        )
        for result in results
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
