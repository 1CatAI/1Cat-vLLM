# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-process ABBA for the packaged single-request QSA indexer."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_mqa_paged, qsa_select_paged_tokens
from vllm.v1.attention.ops.sm70_qsa_shared_key import shared_key_reason


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
    parser.add_argument("--query-rows", type=int, nargs="+", default=[5])
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--live-length", type=int, default=8197)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument(
        "--component", choices=("scorer", "selection", "both"), default="both"
    )
    args = parser.parse_args()
    if (
        min(
            args.page_size,
            args.pages_per_request,
            args.layers,
            *args.query_rows,
            *args.requests,
        )
        <= 0
    ):
        parser.error(
            "Page dimensions, layers, query rows and requests must be positive"
        )
    columns = args.page_size * args.pages_per_request
    if not max(args.query_rows) <= args.live_length <= columns * 4:
        parser.error("Live length must cover query rows and fit the compressed cache")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        parser.error("An SM70 GPU is required")
    position_dtype = getattr(torch, args.position_dtype)
    torch.manual_seed(902)
    results = []
    workloads = [
        (rows, requests) for rows in args.query_rows for requests in args.requests
    ]
    for query_rows, requests in workloads:
        m = query_rows * requests
        layers = []
        for _ in range(args.layers):
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
            req = torch.arange(m, device="cuda", dtype=torch.int32) // query_rows
            pos = (
                args.live_length
                - query_rows
                + torch.arange(m, device="cuda", dtype=position_dtype) % query_rows
            )
            lengths = torch.full(
                (requests,), args.live_length, device="cuda", dtype=torch.int32
            )
            layers.append((q, cache, table, req, pos, lengths))
        admission_reason = shared_key_reason(*layers[0])
        for full_selection in (False, True):
            if args.component == "scorer" and full_selection:
                continue
            if args.component == "selection" and not full_selection:
                continue

            def run(enabled, full_selection=full_selection, layers=layers):
                if full_selection:
                    return [
                        qsa_select_paged_tokens(*v, 2048, 4, shared_key_scoring=enabled)
                        for v in layers
                    ]
                return [
                    qsa_mqa_paged(*v, 4, shared_key_scoring=enabled) for v in layers
                ]

            ref = run(False)
            # Observe dispatch outside timed capture. A requested switch or an
            # admission decision alone is not proof of native execution.
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profile:
                cand = run(True)
            native_calls = sum(
                event.count
                for event in profile.key_averages()
                if event.key == "vllm_sm70_qsa_indexer::run"
            )
            compared_scores = 0
            max_abs_score_error = None
            if full_selection:
                assert all(torch.equal(a, b) for a, b in zip(ref, cand))
            else:
                max_abs_score_error = 0.0
                assert native_calls == (args.layers if admission_reason is None else 0)
                for (expected, ev), (actual, av) in zip(ref, cand):
                    assert torch.equal(av, ev)
                    mask = torch.arange(columns, device="cuda")[None, :] < ev[:, None]
                    actual_scores, expected_scores = actual[mask], expected[mask]
                    torch.testing.assert_close(
                        actual_scores, expected_scores, rtol=2e-6, atol=2e-6
                    )
                    compared_scores += actual_scores.numel()
                    if actual_scores.numel():
                        max_abs_score_error = max(
                            max_abs_score_error,
                            (actual_scores - expected_scores).abs().max().item(),
                        )
            timing = compare_time(lambda: run(False), lambda: run(True), 1)
            row = dict(
                m=m,
                query_rows_per_request=query_rows,
                requests=requests,
                live_length_tokens=args.live_length,
                capacity_tokens=columns * 4,
                compression_ratio=4,
                full_selection=full_selection,
                layers=args.layers,
                position_dtype=args.position_dtype,
                page_size=args.page_size,
                columns=columns,
                admission_reason=admission_reason,
                observed_native_calls=native_calls,
                compared_scores=compared_scores,
                max_abs_score_error=max_abs_score_error,
                timing=timing,
            )
            results.append(row)
            print(row, flush=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
