# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired QSA score/selection screen; excludes projection, attention and the model."""

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


def check_outputs(expected, actual, selection, tolerance):
    if selection:
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        return
    for (a, av), (b, bv) in zip(expected, actual):
        torch.testing.assert_close(av, bv, rtol=0, atol=0)
        mask = torch.arange(a.shape[1], device=a.device)[None, :] < av[:, None]
        torch.testing.assert_close(a[mask], b[mask], rtol=tolerance, atol=tolerance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--position-dtype", choices=("int32", "int64"), default="int64")
    parser.add_argument(
        "--flashinfer",
        action="store_true",
        help="Screen the forced SM70 candidate against the existing best route",
    )
    parser.add_argument(
        "--context", type=int, choices=(8192, 32768, 131072), default=8192
    )
    parser.add_argument(
        "--rows", type=int, nargs="+", choices=(1, 4, 5, 20), default=(5, 20)
    )
    parser.add_argument("--page-size", type=int, default=204)
    parser.add_argument("--pages-per-request", type=int, default=None)
    args = parser.parse_args()
    if args.pages_per_request is None:
        args.pages_per_request = (
            ((args.context + 5 + 3) // 4 + args.page_size - 1) // args.page_size
            if args.page_size > 0
            else 0
        )
    if args.page_size <= 0 or args.pages_per_request <= 0:
        parser.error("Page dimensions must be positive")
    position_dtype = getattr(torch, args.position_dtype)
    torch.manual_seed(902)
    results = []
    for m in args.rows:
        layers = []
        for _ in range(12):
            rows_per_request = 5 if m in (5, 20) else 1
            requests = m // rows_per_request
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
            req = torch.arange(m, device="cuda", dtype=torch.int32) // rows_per_request
            pos = (
                args.context
                + torch.arange(m, device="cuda", dtype=position_dtype)
                % rows_per_request
            )
            lengths = torch.full(
                (requests,),
                args.context + rows_per_request,
                device="cuda",
                dtype=torch.int32,
            )
            layers.append((q, cache, table, req, pos, lengths))
        scorers = [None] * len(layers)
        if args.flashinfer:
            from vllm.model_executor.kernels.flashinfer_sm70 import FlashInferQsaScorer

            scorers = [FlashInferQsaScorer("force") for _ in layers]
        for full_selection in (False, True):

            def run(
                enabled, full_selection=full_selection, layers=layers, scorers=scorers
            ):
                shared = True if args.flashinfer else enabled
                providers = (
                    scorers if args.flashinfer and enabled else [None] * len(layers)
                )
                if full_selection:
                    return [
                        qsa_select_paged_tokens(
                            *v,
                            2048,
                            4,
                            shared_key_scoring=shared,
                            score_provider=provider,
                        )
                        for v, provider in zip(layers, providers)
                    ]
                return [
                    qsa_mqa_paged(
                        *v, 4, shared_key_scoring=shared, score_provider=provider
                    )
                    for v, provider in zip(layers, providers)
                ]

            ref, repeated_ref = run(False), run(False)
            check_outputs(ref, repeated_ref, full_selection, tolerance=0)
            cand = run(True)
            check_outputs(ref, cand, full_selection, tolerance=2e-6)
            aa = compare_time(lambda: run(False), lambda: run(False), 1)
            aa_relative = abs(aa["candidate_ms"] / aa["control_ms"] - 1)
            if aa_relative > 0.02:
                args.out.write_text(
                    json.dumps(
                        {
                            "status": "unstable_A_A",
                            "m": m,
                            "context": args.context,
                            "full_selection": full_selection,
                            "aa": aa,
                            "completed_results": results,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                raise RuntimeError(
                    "A/A differs by more than 2%; "
                    "do not use this control for a speed claim"
                )
            timing = compare_time(lambda: run(False), lambda: run(True), 1)
            row = dict(
                m=m,
                full_selection=full_selection,
                layers=12,
                position_dtype=args.position_dtype,
                page_size=args.page_size,
                columns=columns,
                timing=timing,
                aa=aa,
                context=args.context,
                runtime={
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(),
                    "arch": torch.cuda.get_device_capability(),
                    "precision": "FP16 Q/K, FP32 scores",
                    "seed": 902,
                    "cuda_graph": True,
                },
                provider="flashinfer" if args.flashinfer else "legacy_shared_key",
                scope="score_select_expand" if full_selection else "score",
                complete_qsa_chain=False,
                performance_admitted=False,
                note="Synthetic TP-local screen, not full QSA or model acceptance",
                native=scorers[0].plan.provenance() if args.flashinfer else None,
            )
            results.append(row)
            # Preserve completed points if a later shape fails or is interrupted.
            args.out.write_text(json.dumps(results, indent=2) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
