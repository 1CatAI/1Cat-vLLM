# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare original and CTA-shared activations with cold graph replays."""

import argparse
import hashlib
import json
import runpy
import statistics
from functools import partial
from pathlib import Path

import torch

from vllm import _sm70_ops as ops


def measure(functions, eviction):
    stream = torch.cuda.Stream()
    graphs = []
    for name, fn in functions:
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        begin = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph, stream=stream):
            eviction.fill_(1)
            begin.record()
            fn()
            end.record()
        graphs.append((name, graph, begin, end))
    samples = {name: [] for name, _ in functions}
    for repeat in range(45):
        order = list(range(len(graphs)))
        order = order[repeat % len(order) :] + order[: repeat % len(order)]
        for index in order:
            name, graph, begin, end = graphs[index]
            for _ in range(3):
                graph.replay()
            end.synchronize()
            if repeat >= 5:
                samples[name].append(begin.elapsed_time(end) * 1000)
    for _, graph, _, _ in graphs:
        graph.reset()
    return {
        name: {"mean_us": statistics.mean(values), "samples_us": values}
        for name, values in samples.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--libraries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panels", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 16, 32, 55])
    parser.add_argument("--n16", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(20261005)
    loader = runpy.run_path(
        str(args.source_root / "benchmarks/kernels/benchmark_sm70_nvfp4_qpn2.py")
    )["_load_projection_shards"]
    libraries = {}
    candidates = []
    for panel in args.panels:
        path = args.libraries / f"panel{panel}/qpn2_activation_panel{panel}.so"
        libraries[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        torch.ops.load_library(str(path))
        candidates.append(
            (f"panel{panel}", getattr(torch.ops, f"_qpn2_activation_panel{panel}"))
        )
    if args.n16:
        path = args.libraries / "native_n16/qpn2_activation_native_n16.so"
        libraries[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        torch.ops.load_library(str(path))
    eviction = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    records = []
    for layer in args.layers:
        for projection in loader(args.model, layer, 0, 4):
            codes, scales = ops.nvfp4_qpn2_prepare_sm70(
                projection.packed.cuda(), projection.scales.cuda()
            )
            n, k = projection.packed.shape
            k *= 2
            gated = projection.gated_silu
            for rows in args.rows:
                x = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.1
                out = x.new_empty((rows, n // 2 if gated else n))
                original = (
                    ops.nvfp4_qpn2_gated_sm70_out
                    if gated
                    else ops.nvfp4_qpn2_gemm_sm70_out
                )
                arguments = (
                    out,
                    x,
                    codes,
                    scales,
                    projection.inverse_global_scale,
                    8 if gated else 16,
                    1 if gated else 2,
                )
                functions = [("control", partial(original, *arguments))]
                if gated:
                    functions += [
                        (name, partial(namespace.gated, *arguments))
                        for name, namespace in candidates
                    ]
                if gated and args.n16 and rows <= 8:
                    functions.append(
                        (
                            "native_n16",
                            partial(torch.ops._qpn2_native_n16.gated, *arguments[:5]),
                        )
                    )
                checks = []
                for amplitude in [0.0, 0.125, -0.125, 0.25]:
                    x.copy_(torch.randn_like(x) * amplitude)
                    functions[0][1]()
                    expected = out.clone()
                    for name, fn in functions[1:]:
                        fn()
                        exact = torch.equal(
                            out.view(torch.int16), expected.view(torch.int16)
                        )
                        checks.append(
                            {
                                "route": name,
                                "amplitude": amplitude,
                                "bitwise": exact,
                            }
                        )
                        if not exact:
                            raise AssertionError((layer, rows, name, amplitude))
                x.copy_(torch.randn_like(x) * 0.1)
                row = {
                    "layer": layer,
                    "rows": rows,
                    "gated": gated,
                    "shape": [n, k],
                    "checks": checks,
                    "timing": measure(functions, eviction),
                }
                records.append(row)
                print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "research_only": True,
                "cold_l2_bytes": eviction.numel(),
                "libraries": libraries,
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
