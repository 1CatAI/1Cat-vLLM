# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired native MQA plan+score versus actual paged Triton serving arithmetic.

Frozen 256K capacity, independent request pages and live-length changes.
This measures indexer scoring only, NOT complete QSA or endpoint performance.
"""

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_sm70_flashinfer_gdn_conv import check_exclusive
from benchmarks.kernels.flashinfer_sm70_mqa import FlashInferMQA, build
from benchmarks.kernels.sm70_paired_stats import paired_latency_interval
from vllm import envs
from vllm.models.qwen4_exp.nvidia.ops.qsa import _qsa_mqa_paged_kernel
from vllm.triton_utils import triton


def make_inputs(rows, length, dim=128, heads=4, table_width=335):
    device = "cuda"
    page_size = 196
    torch.manual_seed(20260906)
    q = torch.randn(rows, heads, dim, device=device, dtype=torch.float16)
    k = torch.randn(
        rows * table_width, page_size, 1, dim, device=device, dtype=torch.float16
    )
    table = torch.randperm(rows * table_width, device=device).to(torch.int32)
    table = table.reshape(rows, table_width)
    requests = torch.arange(rows, device=device, dtype=torch.int32)
    # QSA metadata uses int64 logical_positions in the actual model runner.
    positions = torch.full_like(requests, length - 1, dtype=torch.int64)
    lengths = torch.full_like(requests, length)
    return q, k, table, requests, positions, lengths


def reference(inputs, logits, visible):
    q, k, table, requests, positions, lengths = inputs
    rows, heads, dim = q.shape
    columns = logits.shape[1]
    block_n = 32 if rows == 1 else 64
    grouping = 1 if rows <= 32 else 8
    _qsa_mqa_paged_kernel[(rows, triton.cdiv(columns, block_n * grouping))](
        q,
        k,
        table,
        requests,
        positions,
        lengths,
        visible,
        logits,
        *q.stride(),
        k.stride(0),
        k.stride(1),
        k.stride(3),
        *table.stride(),
        logits.stride(0),
        rows,
        columns,
        k.shape[0],
        table.shape[0],
        math.sqrt(dim),
        PAGE_SIZE=k.shape[1],
        PAGE_TABLE_WIDTH=table.shape[1],
        NUM_HEADS=heads,
        HEAD_DIM=dim,
        BLOCK_N=block_n,
        BLOCK_D=max(16, triton.next_power_of_2(dim)),
        TILES_PER_PROG=grouping,
        STAGES=2,
        MAX_N=max(16, triton.next_power_of_2(heads)),
        COMPRESS_RATIO=4,
        num_warps=2,
    )


def check_schedule(op):
    visible = op.visible.cpu().tolist()
    counts = [(v + 63) // 64 for v in visible]
    total = sum(counts)
    quotient, remainder = divmod(total, op.workers)
    expected = []
    for worker in range(op.workers + 1):
        offset = worker * quotient + min(worker, remainder)
        row = 0
        while row < len(counts) and offset >= counts[row]:
            offset -= counts[row]
            row += 1
        expected.append([row, offset])
    assert op.schedule.cpu().tolist() == expected


def check_scores(inputs, op, logits, visible):
    assert torch.equal(op.visible, visible)
    live = torch.arange(logits.shape[1], device=logits.device)[None] < visible[:, None]
    a, b = op.logits[live], logits[live]
    assert torch.equal(torch.isneginf(a), torch.isneginf(b))
    finite = torch.isfinite(b)
    assert torch.isfinite(a[finite]).all()
    error = a[finite] - b[finite]
    max_abs = error.abs().max().item() if error.numel() else 0.0
    relative = (error.norm() / b[finite].norm().clamp_min(1e-12)).item()
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    check_schedule(op)
    return dict(max_abs=max_abs, relative_l2=relative)


def measure(call, repeats):
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def run_case(rows, length, workers, repeats):
    inputs = make_inputs(rows, length)
    q, k, table, requests, positions, lengths = inputs
    columns = table.shape[1] * k.shape[1]
    ops = {str(w): FlashInferMQA(q, columns, w) for w in workers}
    logits = torch.empty((rows, columns), device=q.device)
    visible = torch.empty_like(requests)
    calls = {"reference": lambda: reference(inputs, logits, visible)}
    calls.update({name: lambda op=op: op(*inputs) for name, op in ops.items()})
    graphs = {}
    for name, call in calls.items():
        for _ in range(3):
            call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs[name] = graph

    errors = {name: [] for name in ops}
    for iteration in range(6):
        values = [max(0, length - row * 257 - iteration * 197) for row in range(rows)]
        if iteration == 1:
            values = [0 if row % 2 else 256 for row in range(rows)]
        if iteration == 2:
            values = [262144 - row * 3 for row in range(rows)]
        if iteration == 3:
            values = [255 + row % 3 for row in range(rows)]
        if iteration == 4:
            values = [0] * rows
        lengths.copy_(torch.tensor(values, device=q.device, dtype=torch.int32))
        positions.copy_(lengths - 1)
        q.mul_(-0.875)
        logits.fill_(float("nan"))
        for op in ops.values():
            op.logits.fill_(float("nan"))
            op.schedule.fill_(-991)
        for graph in graphs.values():
            graph.replay()
        for name, op in ops.items():
            errors[name].append(check_scores(inputs, op, logits, visible))

    lengths.fill_(length)
    positions.fill_(length - 1)
    times = {name: [] for name in graphs}
    for cycle in range(5):
        names = list(graphs) if cycle % 2 == 0 else list(reversed(graphs))
        for name in names:
            graphs[name].replay()
            times[name].append(measure(graphs[name].replay, repeats))
    for graph in graphs.values():
        graph.replay()
    for op in ops.values():
        check_scores(inputs, op, logits, visible)
    ref = statistics.median(times["reference"])
    return dict(
        rows=rows,
        live_length=length,
        capacity_columns=columns,
        scope="MQA plan+score only; not full QSA or model quality",
        changed_replays=6,
        timings={
            name: dict(
                median_us=statistics.median(t),
                samples_us=t,
                reduction_pct=100 * (1 - statistics.median(t) / ref),
            )
            for name, t in times.items()
        },
        errors=errors,
        paired_intervals={
            name: paired_latency_interval(times["reference"], t)
            for name, t in times.items()
            if name != "reference"
        },
    )


def screen_selector(rows, length, repeats):
    from vllm.model_executor.layers import sm70_flashinfer_batch as fi
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_select_paged_tokens

    inputs = make_inputs(rows, length)
    q = inputs[0]
    fi._MQA_SMS[q.device] = torch.cuda.get_device_properties(
        q.device
    ).multi_processor_count
    saved = os.environ.get("VLLM_SM70_FLASHINFER_BATCH")
    outputs = [
        torch.empty((rows, 2051), device=q.device, dtype=torch.int32) for _ in range(2)
    ]
    graphs = []
    try:
        for mode in range(2):
            os.environ["VLLM_SM70_FLASHINFER_BATCH"] = str(mode)
            envs.disable_envs_cache()
            for _ in range(3):
                qsa_select_paged_tokens(*inputs, 2048, 4, outputs[mode])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                qsa_select_paged_tokens(*inputs, 2048, 4, outputs[mode])
            graphs.append(graph)
        for cycle in range(8):
            inputs[4].copy_(inputs[5] - 1 - cycle)
            q.mul_(-0.875)
            for output, graph in zip(outputs, graphs):
                output.fill_(-919)
                graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(outputs[0], outputs[1]), (rows, length, cycle)
        inputs[4].copy_(inputs[5] - 1)
        samples = [[], []]
        for cycle in range(5):
            for mode in [0, 1] if cycle % 2 == 0 else [1, 0]:
                graphs[mode].replay()
                samples[mode].append(measure(graphs[mode].replay, repeats))
        assert torch.equal(outputs[0], outputs[1])
        medians = list(map(statistics.median, samples))
        return dict(
            rows=rows,
            length=length,
            scope="MQA+topk+index expansion; excludes sparse attention and model",
            changed_replays_exact=8,
            samples_us=samples,
            median_us=medians,
            reduction_pct=100 * (1 - medians[1] / medians[0]),
            paired_interval=paired_latency_interval(*samples),
        )
    finally:
        if saved is None:
            os.environ.pop("VLLM_SM70_FLASHINFER_BATCH", None)
        else:
            os.environ["VLLM_SM70_FLASHINFER_BATCH"] = saved
        envs.disable_envs_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 65536, 262144])
    parser.add_argument("--workers", nargs="+", type=int, default=[80, 160, 320])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--selector",
        action="store_true",
        help="Include production topk/index expansion after the scorer",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite evidence")
    if not all(1 <= b <= 64 for b in args.rows) or args.repeats < 1:
        parser.error("Require positive repeats and rows in 1..64")
    if not all(1 <= n <= 262144 for n in args.lengths):
        parser.error("Lengths must fit the frozen capacity")
    check_exclusive()
    build()
    results = []
    for rows in args.rows:
        for length in args.lengths:
            result = (
                screen_selector(rows, length, args.repeats)
                if args.selector
                else run_case(rows, length, args.workers, args.repeats)
            )
            results.append(result)
            print(json.dumps(result), flush=True)
            check_exclusive()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
