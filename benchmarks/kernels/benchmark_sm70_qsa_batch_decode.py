# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen C1 score tiles/Top-K at small batches; no runtime admission.

Uses the installed native selector, not a JIT sidecar. Compare score bits and
selected indices before timing. A graph contains twelve calls sharing data;
this is a component microbenchmark, not twelve model layers or model TPOT.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import torch

from vllm import _custom_ops as _ops  # noqa: F401
from vllm.models.qwen4_exp.nvidia.ops import qsa


def capture(fn, calls=12):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    return graph


def paired(graphs):
    samples = [[] for _ in graphs]
    for repeat in range(7):
        order = range(len(graphs)) if repeat % 2 == 0 else reversed(range(len(graphs)))
        for i in order:
            graph = graphs[i]
            for _ in range(20):
                graph.replay()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(50):
                graph.replay()
            end.record()
            end.synchronize()
            samples[i].append(start.elapsed_time(end) / 50)
    return dict(median_ms=[median(v) for v in samples], samples_ms=samples)


def screen(rows, contexts, ordered_head_sum):
    pages = math.ceil(262144 / 4 / 100)
    columns = pages * 100
    cache = torch.randn(rows * pages, 100, 1, 128, dtype=torch.float16, device="cuda")
    query = torch.randn(rows, 4, 128, dtype=torch.float16, device="cuda")
    table = torch.randperm(rows * pages, device="cuda").int().view(rows, pages)
    requests = torch.arange(rows, dtype=torch.int32, device="cuda")
    positions = torch.full_like(requests, 8191)
    lengths = torch.full_like(requests, 8192)
    scores = [torch.empty(rows, columns, device="cuda") for _ in range(2)]
    visible_buffers = [torch.empty_like(requests) for _ in range(2)]
    selected = [
        torch.empty(rows, 512, dtype=torch.int32, device="cuda") for _ in range(2)
    ]
    compiled = []

    def launch_score(arm):
        block = 32 if rows == 1 or arm else 64
        kernel = qsa._qsa_mqa_paged_kernel[(rows, math.ceil(columns / block))](
            query,
            cache,
            table,
            requests,
            positions,
            lengths,
            visible_buffers[arm],
            scores[arm],
            *query.stride(),
            cache.stride(0),
            cache.stride(1),
            cache.stride(3),
            *table.stride(),
            scores[arm].stride(0),
            rows,
            columns,
            cache.shape[0],
            rows,
            math.sqrt(128),
            PAGE_SIZE=100,
            PAGE_TABLE_WIDTH=pages,
            NUM_HEADS=4,
            HEAD_DIM=128,
            BLOCK_N=block,
            BLOCK_D=128,
            TILES_PER_PROG=1,
            STAGES=2,
            MAX_N=16,
            COMPRESS_RATIO=4,
            ORDERED_HEAD_SUM=bool(arm and ordered_head_sum and rows > 1),
            num_warps=2,
        )
        return kernel

    for arm in range(2):
        kernel = launch_score(arm)
        compiled.append(
            dict(registers=kernel.n_regs, shared_bytes=kernel.metadata.shared)
        )
    score_graphs = [capture(lambda arm=arm: launch_score(arm)) for arm in range(2)]
    op = torch.ops._C.qsa_lexicographic_topk
    # Both selectors read the control score buffer: isolate selector arithmetic.
    topk_graphs = [
        capture(
            lambda arm=arm: op(
                scores[0], visible_buffers[0], selected[arm], 512, bool(arm)
            )
        )
        for arm in range(2)
    ]
    checks = []
    for step, context in enumerate((1, 4, 2049, 8192, 9216, 9220, 131077, 262144)):
        query.normal_(0, (0.001, 0.1, 1.0, 3.0)[step % 4])
        cache.normal_(0, 0.1)
        lengths.fill_(context)
        positions.fill_(context - 1)
        if rows > 1:
            positions[0] = max(0, context // 2 - 1)
        if step % 2:
            table[0, 0] = -1
            requests[-1] = -1
        else:
            table.copy_(
                torch.randperm(rows * pages, device="cuda").int().view(rows, pages)
            )
            requests.copy_(torch.arange(rows, dtype=torch.int32, device="cuda"))
        for out, graph in zip(scores, score_graphs):
            out.fill_(-float("inf"))
            graph.replay()
        score_mismatches = int(
            (scores[0].view(torch.int32) != scores[1].view(torch.int32)).sum()
        )
        visible_mismatches = int((visible_buffers[0] != visible_buffers[1]).sum())
        for out, graph in zip(selected, topk_graphs):
            out.fill_(-77)
            graph.replay()
        selected_mismatches = int((selected[0] != selected[1]).sum())
        checks.append(
            dict(
                context=context,
                scores=score_mismatches,
                visible=visible_mismatches,
                selected=selected_mismatches,
            )
        )

    score_exact = all(not v["scores"] and not v["visible"] for v in checks)
    topk_exact = all(not v["selected"] for v in checks)
    results = dict(
        rows=rows,
        checks=checks,
        score_exact=score_exact,
        topk_exact=topk_exact,
        compiled=compiled,
        timings=[],
    )
    requests.copy_(torch.arange(rows, dtype=torch.int32, device="cuda"))
    table.copy_(torch.randperm(rows * pages, device="cuda").int().view(rows, pages))
    query.normal_(0, 0.1)
    for context in contexts:
        lengths.fill_(context)
        positions.fill_(context - 1)
        for graph in score_graphs:
            graph.replay()
        results["timings"].append(
            dict(
                context=context,
                score=paired(score_graphs) if score_exact else None,
                topk=paired(topk_graphs) if topk_exact else None,
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,2,4,8,16")
    parser.add_argument("--contexts", default="8192,65536,262144")
    parser.add_argument("--ordered-head-sum", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20260927)
    for name in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_accumulation",
    ):
        setattr(torch.backends.cuda.matmul, name, False)
    assert torch.cuda.get_device_capability() == (7, 0)
    root = Path(__file__).resolve().parents[2]
    result = dict(
        complete=False,
        calls_per_graph=12,
        shared_inputs=True,
        ordered_head_sum=args.ordered_head_sum,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        native_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "vllm").glob("_C*.so")
        },
        results=[],
    )
    contexts = [int(v) for v in args.contexts.split(",")]
    for rows in map(int, args.rows.split(",")):
        entry = screen(rows, contexts, args.ordered_head_sum)
        result["results"].append(entry)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
    result["complete"] = True
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    assert all(r["score_exact"] and r["topk_exact"] for r in result["results"])


if __name__ == "__main__":
    main()
