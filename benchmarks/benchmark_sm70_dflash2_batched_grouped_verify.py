# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Race request-major grouped DFlash2 verification against independent XQA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def _make_case(
    *, batch_size: int, seq_len: int, page_size: int
) -> tuple[torch.Tensor, ...]:
    query_len = 8
    seq_lens = torch.tensor(
        [seq_len - (req_idx % 3) * 17 for req_idx in range(batch_size)],
        dtype=torch.int32,
        device="cuda",
    )
    max_pages = math.ceil(seq_len / page_size)
    physical_pages = batch_size * max_pages + 3
    source = torch.randn(
        (physical_pages, 2, page_size, 1, 256),
        dtype=torch.float16,
        device="cuda",
    ).mul_(0.25)
    cache = source.to(torch.float8_e5m2).view(torch.uint8)
    del source
    key_cache, value_cache = cache.unbind(1)
    block_table = torch.randperm(physical_pages, dtype=torch.int32, device="cuda")[
        : batch_size * max_pages
    ].view(batch_size, max_pages)
    query = torch.randn(
        (batch_size * query_len, 6, 256),
        dtype=torch.float16,
        device="cuda",
    ).mul_(0.25)
    return query, key_cache, value_cache, block_table, seq_lens


def _measure_ms(fn, *, warmups: int, repeats: int) -> float:
    for _ in range(warmups):
        fn()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / repeats


def _capture(fn) -> torch.cuda.CUDAGraph:
    fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=3296)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.seq_len < 42:
        parser.error("--seq-len must be at least 42 for the varied batch case")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires one SM70 GPU")

    import flash_attn_v100

    torch.manual_seed(20260903 + args.batch_size + args.seq_len)
    query, key_cache, value_cache, block_table, seq_lens = _make_case(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        page_size=args.page_size,
    )
    grouped_out = torch.empty_like(query)
    xqa_out = torch.empty_like(query)
    query_len = 8
    decode_block_table = block_table.repeat_interleave(query_len, dim=0).contiguous()
    decode_seq_lens = (
        seq_lens[:, None]
        - query_len
        + torch.arange(1, query_len + 1, dtype=torch.int32, device="cuda")
    ).flatten()

    def grouped() -> None:
        flash_attn_v100.flash_attn_grouped_verify_paged(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            out=grouped_out,
            one_pass=True,
        )

    def xqa() -> None:
        flash_attn_v100.flash_attn_decode_paged_xqa(
            query,
            key_cache,
            value_cache,
            decode_block_table,
            decode_seq_lens,
            out=xqa_out,
            kv_cache_dtype="fp8_e5m2",
            max_seq_len_hint=args.seq_len,
            workspace_seq_capacity_hint=args.seq_len,
        )

    grouped()
    xqa()
    per_request = torch.cat(
        [
            flash_attn_v100.flash_attn_grouped_verify_paged(
                query[req_idx * query_len : (req_idx + 1) * query_len],
                key_cache,
                value_cache,
                block_table[req_idx : req_idx + 1],
                seq_lens[req_idx : req_idx + 1],
                one_pass=True,
            ).clone()
            for req_idx in range(args.batch_size)
        ]
    )
    torch.accelerator.synchronize()
    grouped_vs_xqa = grouped_out.float().sub(xqa_out.float()).abs()

    grouped_eager_ms = _measure_ms(grouped, warmups=args.warmups, repeats=args.repeats)
    xqa_eager_ms = _measure_ms(xqa, warmups=args.warmups, repeats=args.repeats)
    grouped_graph = _capture(grouped)
    xqa_graph = _capture(xqa)
    grouped_graph_ms = _measure_ms(
        grouped_graph.replay, warmups=args.warmups, repeats=args.repeats
    )
    xqa_graph_ms = _measure_ms(
        xqa_graph.replay, warmups=args.warmups, repeats=args.repeats
    )

    result = {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "page_size": args.page_size,
        "batched_is_bitwise_per_request": bool(torch.equal(grouped_out, per_request)),
        "grouped_vs_xqa_max_abs": float(grouped_vs_xqa.max().item()),
        "grouped_vs_xqa_mean_abs": float(grouped_vs_xqa.mean().item()),
        "grouped_eager_ms": grouped_eager_ms,
        "xqa_eager_ms": xqa_eager_ms,
        "grouped_graph_ms": grouped_graph_ms,
        "xqa_graph_ms": xqa_graph_ms,
        "graph_speedup": xqa_graph_ms / grouped_graph_ms,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
