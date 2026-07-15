# SPDX-License-Identifier: Apache-2.0
"""Benchmark the FlashInfer paged-decode shape used by Qwen3.6-27B.

This isolates one full-attention layer with FP16 queries and FP8 KV cache.
It is intended for Nsight Systems/Compute comparison, not model throughput.
"""

import argparse
import json
import math

import flashinfer
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-qo-heads", type=int, default=24)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.context_len <= 0:
        raise ValueError("--context-len must be positive")

    device = torch.device("cuda")
    page_size = args.page_size
    num_pages = math.ceil(args.context_len / page_size)
    last_page_len = args.context_len - (num_pages - 1) * page_size

    workspace = torch.empty(
        args.workspace_mib * 1024 * 1024, dtype=torch.uint8, device=device
    )
    indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    last_page = torch.tensor([last_page_len], dtype=torch.int32, device=device)

    use_cuda_graph = not args.eager
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        use_cuda_graph=use_cuda_graph,
        use_tensor_cores=True,
        paged_kv_indptr_buffer=indptr if use_cuda_graph else None,
        paged_kv_indices_buffer=indices if use_cuda_graph else None,
        paged_kv_last_page_len_buffer=last_page if use_cuda_graph else None,
        backend="auto",
    )
    wrapper.plan(
        indptr,
        indices,
        last_page,
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        page_size=page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.float16,
        sm_scale=1.0 / math.sqrt(args.head_dim),
        fixed_split_size=-1,
        disable_split_kv=False,
        non_blocking=False,
    )

    q = torch.randn(
        1, args.num_qo_heads, args.head_dim, dtype=torch.float16, device=device
    )
    cache_shape = (
        num_pages,
        page_size,
        args.num_kv_heads,
        args.head_dim,
    )
    k_cache = torch.zeros(cache_shape, dtype=torch.float8_e4m3fn, device=device)
    v_cache = torch.zeros_like(k_cache)
    out = torch.empty_like(q)

    def run() -> None:
        wrapper.run(
            q,
            (k_cache, v_cache),
            q_scale=1.0,
            k_scale=1.0,
            v_scale=1.0,
            out=out,
        )

    graph = None
    if use_cuda_graph:
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()

        def replay() -> None:
            graph.replay()

    else:
        replay = run

    for _ in range(args.warmup):
        replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    range_name = f"flashinfer_decode_ctx_{args.context_len}"
    torch.cuda.nvtx.range_push(range_name)
    start.record()
    for _ in range(args.iters):
        replay()
    end.record()
    end.synchronize()
    torch.cuda.nvtx.range_pop()

    total_ms = start.elapsed_time(end)
    print(
        json.dumps(
            {
                "context_len": args.context_len,
                "page_size": page_size,
                "num_pages": num_pages,
                "num_qo_heads": args.num_qo_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_dim": args.head_dim,
                "kv_dtype": "float8_e4m3fn",
                "cuda_graph": use_cuda_graph,
                "iterations": args.iters,
                "total_ms": total_ms,
                "mean_us": total_ms * 1000.0 / args.iters,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
