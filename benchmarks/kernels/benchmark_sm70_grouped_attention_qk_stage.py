# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check full FP32 tiled QK scores, then time sixteen distinct KV working sets.

The candidate exposes its changed producer and the frozen staged reference in
one DSO. Neither arm includes softmax, PV, merge, or service execution.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_sm70_grouped_attention_long import (
    byte_equal,
    make_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists()
    assert torch.cuda.get_device_capability() == (7, 0)
    manifest = json.loads(args.candidate.read_text())
    library = Path(manifest["library"])
    assert (
        hashlib.sha256(library.read_bytes()).hexdigest() == manifest["library_sha256"]
    )
    spec = importlib.util.spec_from_file_location(manifest["module_name"], library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == library.resolve()
    report = {
        "scope": "QK producer only; not complete attention or service performance",
        "manifest": manifest,
        "device": torch.cuda.get_device_name(),
        "checks": [],
        "performance": [],
        "complete": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def make(page, length, padding=0):
        case = make_case(8, page, length, 0, stride_padding=padding)
        if manifest.get("predecoded_keys"):
            # Diagnostic only: conversion and extra persistent memory are not
            # part of this QK-stage timing and do not authorize a cache change.
            case["half_keys"] = case["k"].view(torch.float8_e4m3fn).half()
        tiles = (case["table"].numel() * page + 31) // 32
        case["score_guards"] = [
            torch.full((tiles + 2, 48, 32), -777.0, device="cuda") for _ in range(2)
        ]
        case["scores"] = [x[1:-1] for x in case["score_guards"]]
        return case

    def run(case, arm):
        keys = (
            case["half_keys"] if arm and manifest.get("predecoded_keys") else case["k"]
        )
        module.qk_stage(
            case["q"],
            keys,
            case["table"],
            case["lengths"],
            case["scores"][arm],
            0.03125,
            bool(arm),
        )

    torch.manual_seed(20260909)
    for page, length, padding in (
        (3296, 8, 0),
        (3296, 65, 0),
        (1648, 1649, 0),
        (3296, 3297, 0),
        (848, 8197, 8),
        (3296, 131072, 0),
        (3296, 261888, 0),
        (1648, 262144, 8),
    ):
        case = make(page, length, padding)
        for state in ("original", "zero", "restored"):
            if state == "zero":
                case["lengths"].zero_()
            else:
                case["lengths"].copy_(case["initial_lengths"])
            for buffer in case["score_guards"]:
                buffer.fill_(-777.0)
            run(case, 0)
            run(case, 1)
            torch.cuda.synchronize()
            exact = byte_equal(*case["score_guards"])
            guards_intact = all(
                bool((buffer[[0, -1]] == -777.0).all())
                for buffer in case["score_guards"]
            )
            row = {
                "page": page,
                "context": length,
                "stride_padding": padding,
                "state": state,
                "all_score_bytes_exact": exact,
                "guards_intact": guards_intact,
            }
            if not exact:
                left, right = case["scores"]
                row["max_abs_error"] = (left - right).abs().max().item()
                row["mismatching_floats"] = (left != right).sum().item()
            report["checks"].append(row)
            save()
            print(json.dumps(row), flush=True)
            assert exact and guards_intact, row
        del case

    for length in () if args.correctness_only else (131072, 261888):
        cases = [make(3296, length) for _ in range(16)]
        graphs = []
        for arm in range(2):
            for case in cases:
                run(case, arm)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for case in cases:
                    run(case, arm)
            graphs.append(graph)
        for _ in range(5):
            for graph in graphs:
                graph.replay()
        torch.cuda.synchronize()
        samples = [[], []]
        for trial in range(5):
            for arm in (0, 1) if trial % 2 == 0 else (1, 0):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(8):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                samples[arm].append(start.elapsed_time(end) / 8)
        row = {
            "context": length,
            "layers": 16,
            "phase": "QK only",
            "samples_ms": samples,
            "median_ms": [statistics.median(x) for x in samples],
        }
        report["performance"].append(row)
        save()
        print(json.dumps(row), flush=True)
        del cases, graphs, graph, case
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
