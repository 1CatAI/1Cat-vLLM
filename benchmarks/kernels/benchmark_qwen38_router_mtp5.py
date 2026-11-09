# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the exact SM70 Qwen3.8 E512/K10 small-batch router."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    _sm70_qwen38_router_topk,
)

EXPERTS = 512
TOP_K = 10
LAYERS = 48


def _capture_round(launch: Callable[[], None]) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(LAYERS):
            launch()
    return graph


def _time_graph(
    graph: torch.cuda.CUDAGraph, *, warmup: int, iterations: int
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()
    samples: list[float] = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples), samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tokens", type=int, choices=range(1, 17), default=5)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("benchmark requires exact SM70")

    torch.manual_seed(args.seed)
    tokens = args.tokens
    logits = torch.randn(tokens, EXPERTS, device="cuda", dtype=torch.float16)
    baseline_weights = torch.empty(tokens, TOP_K, device="cuda", dtype=torch.float32)
    baseline_ids = torch.empty(tokens, TOP_K, device="cuda", dtype=torch.int32)
    baseline_rows = torch.empty_like(baseline_ids)
    candidate_weights = torch.empty_like(baseline_weights)
    candidate_ids = torch.empty_like(baseline_ids)
    candidate_rows = torch.empty_like(baseline_rows)

    def baseline_launch() -> None:
        ops.topk_softmax(
            baseline_weights,
            baseline_ids,
            baseline_rows,
            logits,
            True,
        )

    def candidate_launch() -> None:
        _sm70_qwen38_router_topk(
            candidate_weights,
            candidate_ids,
            candidate_rows,
            logits,
        )

    for _ in range(8):
        baseline_launch()
        candidate_launch()
    torch.accelerator.synchronize()
    baseline_graph = _capture_round(baseline_launch)
    candidate_graph = _capture_round(candidate_launch)

    logits.mul_(0.75).add_(0.03125)
    baseline_graph.replay()
    candidate_graph.replay()
    torch.accelerator.synchronize()
    comparison = {
        "ids_equal": bool(torch.equal(candidate_ids, baseline_ids)),
        "rows_equal": bool(torch.equal(candidate_rows, baseline_rows)),
        "weights_max_abs": float((candidate_weights - baseline_weights).abs().max()),
        "weights_allclose_1e7": bool(
            torch.allclose(
                candidate_weights,
                baseline_weights,
                atol=1e-7,
                rtol=1e-7,
            )
        ),
    }

    baseline_ms, baseline_samples = _time_graph(
        baseline_graph, warmup=args.warmup, iterations=args.iterations
    )
    candidate_ms, candidate_samples = _time_graph(
        candidate_graph, warmup=args.warmup, iterations=args.iterations
    )
    result = {
        "device": torch.cuda.get_device_name(),
        "shape": [tokens, EXPERTS],
        "top_k": TOP_K,
        "layers_per_round": LAYERS,
        "dynamic_graph_comparison": comparison,
        "baseline_ms_per_48_layers": baseline_ms,
        "candidate_ms_per_48_layers": candidate_ms,
        "saved_ms_per_verifier": baseline_ms - candidate_ms,
        "speedup": baseline_ms / candidate_ms,
        "baseline_us_per_layer": baseline_ms * 1000.0 / LAYERS,
        "candidate_us_per_layer": candidate_ms * 1000.0 / LAYERS,
        "baseline_samples_ms": baseline_samples,
        "candidate_samples_ms": candidate_samples,
        "warmup_replays": args.warmup,
        "timed_replays": args.iterations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not (
        comparison["ids_equal"]
        and comparison["rows_equal"]
        and comparison["weights_allclose_1e7"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
