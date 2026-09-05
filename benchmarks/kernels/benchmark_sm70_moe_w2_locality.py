# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
"""Exact-arithmetic MoE locality screens using real weights and route traces.

Screen W2 warp mapping, paired W13 projections, or tile-major scale layout.
Measure the affected component and complete grouped MoE, including planning,
scatter and reduction. Activations are synthetic; this is not a quality gate.
"""

import argparse
import glob
import hashlib
import json
import statistics
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_sm70_moe_packed_w13 import (
    checkpoint_weights,
    graph,
    latency,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument("--w13-pair", action="store_true")
    experiment.add_argument("--scale-layout", action="store_true")
    parser.add_argument("--routes", required=True, help="Glob of saved [M,10] tensors")
    parser.add_argument("--route-limit", type=int, default=1)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite a previous result")
    if any(m not in (4, 8, 16) for m in args.tokens):
        parser.error("This paired screen supports M4/8/16")
    if min(args.route_limit, args.repeats, args.samples) < 1:
        parser.error("Counts must be positive")
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.ops.load_library(str(args.library.resolve()))
    native = torch.ops._C
    candidate = (
        torch.ops._C_moe_scale_layout
        if args.scale_layout
        else torch.ops._C_moe_pair
        if args.w13_pair
        else torch.ops._C_moe_locality
    )
    paths = sorted(map(Path, glob.glob(args.routes)))[: args.route_limit]
    if not paths:
        parser.error("No route files matched")
    captured = {}
    for path in paths:
        value = torch.load(path, map_location="cpu", weights_only=True)
        value = value["tensor"] if isinstance(value, dict) else value
        assert value.ndim == 2 and value.shape[1] == 10
        assert value.shape[0] >= max(args.tokens)
        captured[path.name] = value.to(torch.int32)
    torch.manual_seed(20260905)
    w13, s13, w2, s2 = checkpoint_weights(args.model, 0, 0, True)
    resources = (
        {}
        if args.scale_layout
        else {
            str(mode): list(candidate.resources(mode))
            for mode in ((4, 5, 8) if args.w13_pair else (1, 2, 4))
        }
    )
    if args.scale_layout:
        tile_s13 = s13.reshape(512, 160, 10, 32).permute(0, 2, 1, 3).contiguous()
        tile_s2 = s2.reshape(512, 10, 80, 32).permute(0, 2, 1, 3).contiguous()
    results = []
    for m in args.tokens:
        n = m * 10
        split = {4: 5, 8: 4, 16: 8}[m]
        x = torch.randn(m, 2560, device="cuda", dtype=torch.float16) * 0.1
        ids = torch.empty(n, device="cuda", dtype=torch.int32)
        topk = torch.softmax(torch.randn(m, 10, device="cuda"), -1)
        mid = torch.empty(n, 160, device="cuda", dtype=torch.float16)
        routed = torch.empty(n, 2560, device="cuda", dtype=torch.float16)
        rows = torch.empty(n, 8, device="cuda", dtype=torch.int32)
        experts = torch.empty(n, device="cuda", dtype=torch.int32)
        sizes = torch.empty_like(experts)
        total = torch.empty(1, device="cuda", dtype=torch.int32)
        modes = (
            (0, 1, 2, 3)
            if args.scale_layout
            else (0, 1)
            if args.w13_pair
            else (0, 1, 2, 4)
        )
        outputs = {mode: torch.empty_like(x) for mode in modes}

        def w13_call(mode=0):
            if args.scale_layout and mode & 1:
                candidate.nvfp4_grouped_w13_sm70_out(
                    mid, x, w13, tile_s13, ids, rows, experts, sizes, total, split, True
                )
                return
            run = (
                candidate.run
                if args.w13_pair and mode
                else native.nvfp4_grouped_w13_sm70_out
            )
            run(mid, x, w13, s13, ids, rows, experts, sizes, total, split, True)

        def w2_call(mode):
            params = (
                outputs[mode],
                routed,
                mid,
                w2,
                tile_s2 if args.scale_layout and mode & 2 else s2,
                topk,
                rows,
                experts,
                sizes,
                total,
            )
            if args.scale_layout and mode & 2:
                candidate.nvfp4_grouped_w2_sm70_out(*params)
            elif mode == 0 or args.w13_pair or args.scale_layout:
                native.nvfp4_grouped_w2_sm70_out(*params)
            else:
                candidate.w2(*params, mode)

        def complete(mode):
            w13_call(mode)
            w2_call(mode)

        cases = {
            "distinct": torch.arange(n).reshape(m, 10),
            "shared10": torch.arange(10).repeat(m, 1),
            **{name: value[:m] for name, value in captured.items()},
        }
        for name, values in cases.items():
            ids.copy_(values.reshape(-1))
            full = {mode: graph(lambda: complete(mode)) for mode in outputs}
            changed_cases = (
                values,
                values.flip(0),
                torch.arange(10).repeat(m, 1),
                torch.zeros(m, 10, dtype=torch.int32),
                torch.arange(n).reshape(m, 10),
                torch.full((m, 10), -1),
                torch.full((m, 10), 512),
            )
            for changed in changed_cases:
                x.normal_(0, 0.1)
                topk.copy_(torch.softmax(torch.randn_like(topk), -1))
                ids.copy_(changed.reshape(-1))
                for mode in outputs:
                    for buf in (rows, experts, sizes, total):
                        buf.fill_(-12345)
                    mid.fill_(float("nan"))
                    routed.fill_(float("nan"))
                    outputs[mode].fill_(float("nan"))
                    full[mode].replay()
                    if mode == 0:
                        reference_mid = mid.clone()
                    assert torch.equal(mid, reference_mid), (m, name, mode, "W13")
                    assert torch.isfinite(outputs[mode]).all()
                    assert torch.equal(outputs[mode], outputs[0]), (m, name, mode)
            ids.copy_(values.reshape(-1))
            w13_call()
            count = total.item()
            parts = (
                (("w13", w13_call), ("w2", w2_call))
                if args.scale_layout
                else (("w13", w13_call),)
                if args.w13_pair
                else (("w2", w2_call),)
            )
            scopes = [
                (name, {mode: graph(lambda: part(mode)) for mode in outputs})
                for name, part in parts
            ]
            scopes.append(("complete_moe", full))
            for scope, graphs in scopes:
                times = {mode: [] for mode in outputs}
                for sample in range(args.samples):
                    order = modes if sample % 2 == 0 else tuple(reversed(modes))
                    for mode in order:
                        times[mode].append(latency(graphs[mode], args.repeats))
                result = {
                    "m": m,
                    "case": name,
                    "scope": scope,
                    "groups": count,
                    "split": split,
                    "exact_changed_replays": len(changed_cases),
                    "median_us": {
                        str(k): statistics.median(v) for k, v in times.items()
                    },
                    "samples_us": times,
                }
                results.append(result)
                print(json.dumps(result), flush=True)
    report = {
        "model": str(args.model),
        "layer": 0,
        "tp4_rank": 0,
        "synthetic_activations": True,
        "w13_pair": args.w13_pair,
        "scale_layout": args.scale_layout,
        "graph_unroll": 16,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "route_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "resource_fields": [
            "registers",
            "shared_bytes",
            "local_bytes",
            "max_ctas_per_sm",
        ],
        "resources": resources,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
