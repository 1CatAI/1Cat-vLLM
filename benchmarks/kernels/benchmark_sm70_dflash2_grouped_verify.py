# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark the SM70 q8/q16/q32 grouped DFlash2 verifier."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from functools import partial

import torch


def _make_case(
    query_len: int, prefix_len: int, page_size: int
) -> tuple[torch.Tensor, ...]:
    total_len = prefix_len + query_len
    logical_pages = math.ceil(total_len / page_size)
    physical_pages = logical_pages + 3
    source = torch.randn(
        (physical_pages, 2, page_size, 1, 256),
        dtype=torch.float16,
        device="cuda",
    ).mul_(0.25)
    cache = source.to(torch.float8_e5m2).view(torch.uint8)
    key_cache, value_cache = cache.unbind(1)
    block_table = torch.randperm(physical_pages, dtype=torch.int32, device="cuda")[
        :logical_pages
    ].view(1, -1)
    query = torch.randn((query_len, 6, 256), dtype=torch.float16, device="cuda").mul_(
        0.25
    )
    seq_lens = torch.tensor([total_len], dtype=torch.int32, device="cuda")
    return query, key_cache, value_cache, block_table, seq_lens


def _time_cuda(fn: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _capture(fn: Callable[[], object]) -> torch.cuda.CUDAGraph:
    fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-lens", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument(
        "--prefix-lens", type=int, nargs="+", default=[1024, 32768, 128000]
    )
    parser.add_argument("--page-size", type=int, default=3296)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires an SM70 GPU")

    import flash_attn_v100

    torch.manual_seed(20260827)
    results = []
    for query_len in args.query_lens:
        for prefix_len in args.prefix_lens:
            query, key_cache, value_cache, block_table, seq_lens = _make_case(
                query_len, prefix_len, args.page_size
            )
            grouped_out = torch.empty_like(query)
            independent_out = torch.empty_like(query)
            decode_block_table = block_table.repeat(query_len, 1).contiguous()
            decode_seq_lens = torch.arange(
                prefix_len + 1,
                prefix_len + query_len + 1,
                dtype=torch.int32,
                device=query.device,
            )

            grouped = partial(
                flash_attn_v100.flash_attn_grouped_verify_paged,
                query,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                out=grouped_out,
                one_pass=True,
            )
            independent = partial(
                flash_attn_v100.flash_attn_decode_paged_xqa,
                query,
                key_cache,
                value_cache,
                decode_block_table,
                decode_seq_lens,
                out=independent_out,
                kv_cache_dtype="fp8_e5m2",
                max_seq_len_hint=prefix_len + query_len,
                workspace_seq_capacity_hint=prefix_len + query_len,
            )

            grouped()
            independent()
            torch.accelerator.synchronize()
            difference = grouped_out.float().sub(independent_out.float()).abs()
            grouped_eager_ms = _time_cuda(grouped, args.warmup, args.repeats)
            independent_eager_ms = _time_cuda(independent, args.warmup, args.repeats)
            grouped_graph = _capture(grouped)
            independent_graph = _capture(independent)
            grouped_graph_ms = _time_cuda(
                grouped_graph.replay, args.warmup, args.repeats
            )
            independent_graph_ms = _time_cuda(
                independent_graph.replay, args.warmup, args.repeats
            )
            results.append(
                {
                    "query_len": query_len,
                    "prefix_len": prefix_len,
                    "page_size": args.page_size,
                    "grouped_eager_ms": grouped_eager_ms,
                    "independent_eager_ms": independent_eager_ms,
                    "eager_speedup": independent_eager_ms / grouped_eager_ms,
                    "grouped_graph_ms": grouped_graph_ms,
                    "independent_graph_ms": independent_graph_ms,
                    "graph_speedup": independent_graph_ms / grouped_graph_ms,
                    "max_abs_vs_independent": difference.max().item(),
                    "mean_abs_vs_independent": difference.mean().item(),
                }
            )

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
