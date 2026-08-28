# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark DFlash2 lookup drafting against the production UVA history."""

import argparse
import json
from collections.abc import Callable

import torch

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.buffer_utils import UvaBuffer
from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import (
    _point_mass_draft_logits_kernel,
    fuse_draft,
    suffix_lookup,
)


def _time_ms(fn: Callable[[], None], repeats: int, warmup: int) -> float:
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


def _make_history(batch_size: int, context_len: int) -> UvaBuffer:
    history = UvaBuffer((batch_size, context_len + 32), torch.int32)
    history.cpu.random_(1000, 200000)
    suffix = torch.arange(220001, 220013, dtype=torch.int32)
    continuation = torch.arange(230001, 230016, dtype=torch.int32)
    match_start = max(0, context_len // 2 - suffix.numel())
    for row in range(batch_size):
        history.cpu[row, match_start : match_start + suffix.numel()].copy_(suffix)
        history.cpu[
            row,
            match_start + suffix.numel() : match_start
            + suffix.numel()
            + continuation.numel(),
        ].copy_(continuation)
        history.cpu[row, context_len - suffix.numel() : context_len].copy_(suffix)
    return history


def _benchmark_case(
    batch_size: int,
    context_len: int,
    repeats: int,
    warmup: int,
) -> dict[str, float | int]:
    device = torch.device("cuda")
    k, draft_block, top_k, vocab_size = 15, 7, 16, 248320
    history = _make_history(batch_size, context_len)
    total_len = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)
    idx_mapping = torch.arange(
        batch_size, dtype=torch.int32, device=device
    ).repeat_interleave(
        draft_block,
    )
    eligible = torch.ones(batch_size, dtype=torch.int32, device=device)
    lookup_tokens = torch.zeros((batch_size, k), dtype=torch.int32, device=device)
    match_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
    valid = torch.zeros_like(match_len)
    drafted = torch.zeros((batch_size, k), dtype=torch.int64, device=device)
    use = torch.zeros((batch_size, k), dtype=torch.int32, device=device)
    hits = torch.zeros((), dtype=torch.int64, device=device)
    take_flags = torch.zeros(batch_size, dtype=torch.int32, device=device)
    cached_ids = torch.arange(top_k, dtype=torch.int64, device=device).view(1, 1, top_k)
    cached_ids = cached_ids.expand(batch_size, k, top_k).clone()
    cached_scores = torch.zeros(
        (batch_size, k, top_k), dtype=torch.float32, device=device
    )
    draft_logits = torch.full(
        (batch_size, k, vocab_size),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    draft_logits.scatter_(2, cached_ids, cached_scores)

    def lookup_only() -> None:
        suffix_lookup(
            history.uva,
            total_len,
            idx_mapping,
            eligible,
            batch_size,
            k,
            idx_mapping_stride=draft_block,
            nmin=6,
            nmax=12,
            out_tokens=lookup_tokens,
            out_len=match_len,
            out_valid=valid,
        )

    def lookup_and_fuse() -> None:
        lookup_only()
        fuse_draft(
            drafted,
            lookup_tokens,
            match_len,
            valid,
            use,
            idx_mapping,
            hits,
            batch_size,
            k,
            draft_block=draft_block,
            idx_mapping_stride=draft_block,
            nmin=6,
            nstrong=6,
            agree_min=0,
            nmin_tail=4,
            long_min=6,
            take_flags=take_flags,
        )

    def draftless_chain() -> None:
        lookup_tokens.zero_()
        lookup_only()
        drafted.copy_(lookup_tokens)
        use.fill_(1)
        _point_mass_draft_logits_kernel[(batch_size * k,)](
            draft_logits,
            cached_ids,
            cached_scores,
            drafted,
            drafted.stride(0),
            use,
            idx_mapping,
            draft_block,
            cached_ids.stride(0),
            cached_ids.stride(1),
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=k,
            top_k=top_k,
            BLOCK_K=triton.next_power_of_2(top_k),
            CACHE_SCORES=True,
            num_warps=1,
        )

    eager_ms = _time_ms(lookup_and_fuse, repeats, warmup)
    chain_eager_ms = _time_ms(draftless_chain, repeats, warmup)
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            lookup_and_fuse()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.accelerator.synchronize()
    with torch.cuda.graph(graph):
        lookup_and_fuse()
    graph_ms = _time_ms(graph.replay, repeats, warmup)

    assert torch.all(match_len == 12)
    assert torch.all(valid == k)
    assert torch.all(take_flags == 1)
    return {
        "batch_size": batch_size,
        "context_len": context_len,
        "uva_lookup_fuse_eager_ms": eager_ms,
        "uva_lookup_fuse_graph_ms": graph_ms,
        "uva_draftless_chain_eager_ms": chain_eager_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="1024,32768,65536,131072")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    results = [
        _benchmark_case(batch, context, args.repeats, args.warmup)
        for context in (int(value) for value in args.contexts.split(","))
        for batch in (int(value) for value in args.batches.split(","))
    ]
    payload = {
        "device": torch.cuda.get_device_name(),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
