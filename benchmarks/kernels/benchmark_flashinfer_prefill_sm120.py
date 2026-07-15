# SPDX-License-Identifier: Apache-2.0
"""Benchmark one FlashInfer prefill layer for Qwen3.6-27B on SM120."""

import argparse
import json
import math

import flashinfer
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-len", type=int, required=True)
    parser.add_argument("--kv-len", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-qo-heads", type=int, default=24)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.query_len <= args.kv_len:
        raise ValueError("expected 0 < query_len <= kv_len")

    device = torch.device("cuda")
    num_pages = math.ceil(args.kv_len / args.page_size)
    last_page_len = args.kv_len - (num_pages - 1) * args.page_size
    workspace = torch.empty(
        args.workspace_mib * 1024 * 1024, dtype=torch.uint8, device=device
    )
    qo_indptr = torch.tensor([0, args.query_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    last_page = torch.tensor([last_page_len], dtype=torch.int32, device=device)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="auto"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page,
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim_qk=args.head_dim,
        head_dim_vo=args.head_dim,
        page_size=args.page_size,
        causal=True,
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
        args.query_len,
        args.num_qo_heads,
        args.head_dim,
        dtype=torch.float16,
        device=device,
    )
    cache_shape = (
        num_pages,
        args.page_size,
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

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    range_name = f"flashinfer_prefill_q_{args.query_len}_kv_{args.kv_len}"
    torch.cuda.nvtx.range_push(range_name)
    start.record()
    for _ in range(args.iters):
        run()
    end.record()
    end.synchronize()
    torch.cuda.nvtx.range_pop()

    total_ms = start.elapsed_time(end)
    print(
        json.dumps(
            {
                "query_len": args.query_len,
                "kv_len": args.kv_len,
                "page_size": args.page_size,
                "num_pages": num_pages,
                "num_qo_heads": args.num_qo_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_dim": args.head_dim,
                "kv_dtype": "float8_e4m3fn",
                "iterations": args.iters,
                "total_ms": total_ms,
                "mean_ms": total_ms / args.iters,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
