# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare serial and batched DFlash attention with the same CUDA graph inputs.

Example: CUDA_VISIBLE_DEVICES=0 uv run --no-project \
    --python .venv/bin/python .venv/bin/python \
    benchmarks/kernels/benchmark_sm70_dflash_batched_prefill.py
These are attention operator timings, not complete draft or endpoint latency.
"""

import argparse
import json
import statistics

import torch


def graph_us(fn):
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(5):
        graph.replay()
    samples = []
    for _ in range(5):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(30):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 30)
    return statistics.median(samples)


@torch.inference_mode()
def screen(context, batch):
    from flash_attn_v100 import flash_attn_prefill_paged

    pages = (context + 15) // 16
    query = torch.randn(batch, 8, 8, 128, device="cuda", dtype=torch.float16)
    key = torch.randn(batch * pages, 16, 2, 128, device="cuda").half()
    value = torch.randn_like(key)
    table = torch.randperm(batch * pages, device="cuda", dtype=torch.int32).view(
        batch, pages
    )
    lengths = torch.tensor(
        [context - i % 7 for i in range(batch)],
        device="cuda",
        dtype=torch.int32,
    )
    serial, combined = torch.empty_like(query), torch.empty_like(query)

    def old():
        for i in range(batch):
            flash_attn_prefill_paged(
                query[i : i + 1],
                key,
                value,
                table[i : i + 1],
                lengths[i : i + 1],
                out=serial[i : i + 1],
                causal=False,
                window_size=(2047, 2047),
            )

    def new():
        flash_attn_prefill_paged(
            query,
            key,
            value,
            table,
            lengths,
            out=combined,
            causal=False,
            window_size=(2047, 2047),
        )

    old()
    new()
    torch.testing.assert_close(combined, serial, rtol=0, atol=0)
    print(
        json.dumps(
            {
                "batch": batch,
                "context": context,
                "serial_us": graph_us(old),
                "batched_us": graph_us(new),
                "bitwise_equal": torch.equal(combined, serial),
            }
        ),
        flush=True,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", nargs="+", type=int, default=[2048, 32768])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark targets SM70")
    torch.manual_seed(20260925)
    for context in args.contexts:
        for batch in args.batches:
            screen(context, batch)


if __name__ == "__main__":
    main()
