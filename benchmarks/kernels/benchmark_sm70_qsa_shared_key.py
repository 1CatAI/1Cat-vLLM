# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-process ABBA for the packaged single-request QSA indexer."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_mqa_paged, qsa_select_paged_tokens


def compare_time(a, b, count):
    for _ in range(3):
        a()
        b()
    torch.cuda.synchronize()
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    for graph, call in zip(graphs, (a, b)):
        with torch.cuda.graph(graph):
            for _ in range(count):
                call()
    samples = [[], []]
    for _ in range(5):
        for arm in (0, 1, 1, 0):
            for _ in range(5):
                graphs[arm].replay()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(30):
                graphs[arm].replay()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end) / 30)
    medians = [statistics.median(s) for s in samples]
    return dict(
        calls=count,
        control_ms=medians[0],
        candidate_ms=medians[1],
        saved_ms=medians[0] - medians[1],
        samples_ms=samples,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--position-dtype", choices=("int32", "int64"), default="int64")
    parser.add_argument("--page-size", type=int, default=204)
    parser.add_argument("--pages-per-request", type=int, default=12)
    args = parser.parse_args()
    if args.page_size <= 0 or args.pages_per_request <= 0:
        parser.error("Page dimensions must be positive")
    position_dtype = getattr(torch, args.position_dtype)
    torch.manual_seed(902)
    results = []
    for m in (5, 20):
        layers = []
        for _ in range(12):
            requests = m // 5
            columns = args.page_size * args.pages_per_request
            cache = torch.randn(
                requests * args.pages_per_request,
                args.page_size,
                1,
                128,
                device="cuda",
                dtype=torch.float16,
            )
            table = torch.randperm(
                requests * args.pages_per_request, device="cuda", dtype=torch.int32
            ).view(requests, -1)
            q = torch.randn(m, 4, 128, device="cuda", dtype=torch.float16)
            req = torch.arange(m, device="cuda", dtype=torch.int32) // 5
            pos = 8192 + torch.arange(m, device="cuda", dtype=position_dtype) % 5
            lengths = torch.full((requests,), 8197, device="cuda", dtype=torch.int32)
            layers.append((q, cache, table, req, pos, lengths))
        for full_selection in (False, True):

            def run(enabled, full_selection=full_selection, layers=layers):
                if full_selection:
                    return [
                        qsa_select_paged_tokens(*v, 2048, 4, shared_key_scoring=enabled)
                        for v in layers
                    ]
                return [
                    qsa_mqa_paged(*v, 4, shared_key_scoring=enabled) for v in layers
                ]

            ref, cand = run(False), run(True)
            if full_selection:
                assert all(torch.equal(a, b) for a, b in zip(ref, cand))
            timing = compare_time(lambda: run(False), lambda: run(True), 1)
            row = dict(
                m=m,
                full_selection=full_selection,
                layers=12,
                position_dtype=args.position_dtype,
                page_size=args.page_size,
                columns=columns,
                timing=timing,
            )
            results.append(row)
            print(row, flush=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
