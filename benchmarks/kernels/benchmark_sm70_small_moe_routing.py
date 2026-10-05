# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare alignment and output restoration at Flash-Next expert geometry."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.sm70_small_routing import (
    SM70_SMALL_ROUTING,
)


def timing(fn):
    for _ in range(10):
        fn()
    graph = torch.cuda.CUDAGraph()
    inner = 8
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        # Several device invocations per replay prevent host replay gaps
        # from dominating a pair of short kernels.
        samples.append(start.elapsed_time(end) * 10 / inner)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", nargs="+", type=int, default=[1, 5, 10, 20, 32])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    results = []
    for m in args.m:
        torch.manual_seed(20261004 + m)
        x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
        ids = torch.randn(m, 512, device="cuda").topk(10, dim=1).indices
        weights = torch.rand(m, 10, device="cuda")
        weights /= weights.sum(1, keepdim=True)
        assert SM70_SMALL_ROUTING.reason(x, ids, 512) is None
        reference_down = torch.randn(m * 10, 2560, device="cuda").half()
        _, old_order = ids.flatten().sort()
        _, new_order = ids.flatten().sort(stable=True)
        old_down = reference_down[old_order].contiguous()
        new_down = reference_down[new_order].contiguous()
        boundaries = torch.arange(513, device="cuda")

        def old_path(
            x=x, ids=ids, boundaries=boundaries, down=old_down, weights=weights, m=m
        ):
            sorted_ids, order = ids.flatten().sort()
            offsets = torch.searchsorted(sorted_ids, boundaries).int()
            routed = x[torch.div(order, 10, rounding_mode="floor")].contiguous()
            restored = down[order.argsort()].view(m, 10, 2560)
            out = (restored.float() * weights[:, :, None].float()).sum(1).half()
            return routed, offsets, out

        def new_path(x=x, ids=ids, down=new_down, weights=weights):
            routed, offsets, _, inverse = torch.ops.vllm.sm70_small_expert_route(
                x, ids, 512
            )
            out = torch.ops.vllm.sm70_small_expert_unroute(down, inverse, weights)
            return routed, offsets, out

        old, new = old_path(), new_path()
        torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)
        torch.testing.assert_close(old[2], new[2], rtol=0.003, atol=0.0001)
        old_us, new_us = timing(old_path), timing(new_path)
        delta = new[2].float() - old[2].float()
        row = {
            "m": m,
            "hidden_size": 2560,
            "experts": 512,
            "top_k": 10,
            "active_experts": ids.unique().numel(),
            "old_chain_us": old_us,
            "new_chain_us": new_us,
            "estimated_48_layers_saved_ms": (old_us - new_us) * 48 / 1000,
            "max_abs_error": delta.abs().max().item(),
            "relative_l2": (delta.norm() / old[2].float().norm()).item(),
            "scope": (
                "Synthetic routes at actual Flash-Next geometry; "
                "no expert GEMM or model round"
            ),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
