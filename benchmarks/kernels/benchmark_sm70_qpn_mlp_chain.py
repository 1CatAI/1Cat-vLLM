# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the complete local MLP, including activation packing costs.

This is a producer/consumer subgraph, not a whole transformer layer or TP
model benchmark. The packed gate/up output is consumed directly by down.
"""

import argparse
import hashlib
import json
import runpy
import statistics
from functools import partial
from pathlib import Path

import torch

from vllm import _sm70_ops as ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 16, 32, 55])
    parser.add_argument("--replays-per-sample", type=int, default=10)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(20261005)
    library = args.library_dir / "qpn_operands700.so"
    torch.ops.load_library(str(library))
    ext = torch.ops._qpn_operands700
    loader = runpy.run_path(
        str(args.source_root / "benchmarks/kernels/benchmark_sm70_nvfp4_qpn2.py")
    )["_load_projection_shards"]
    eviction = torch.ones(33554432, device="cuda", dtype=torch.int32)
    sink = torch.empty(256, device="cuda", dtype=torch.int64)
    records = []
    for layer in args.layers:
        projections = loader(args.model, layer, 0, 4)
        prepared = []
        for projection in projections:
            codes, scales = ops.nvfp4_qpn2_prepare_sm70(
                projection.packed.cuda(), projection.scales.cuda()
            )
            effective = (
                scales.view(torch.float8_e4m3fn).float()
                * projection.inverse_global_scale
            ).half()
            if not torch.isfinite((effective.float() * 16384).half()).all().item():
                raise ValueError("Unsafe range-decode scale")
            prepared.append((codes, scales, projection.inverse_global_scale))
        x = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * 0.1
        packed_x = torch.empty_like(x)
        intermediate = x.new_empty((8, 4352))
        result = torch.empty_like(x)

        def production(
            x=x, intermediate=intermediate, result=result, prepared=prepared
        ):
            ops.nvfp4_qpn2_gated_sm70_out(intermediate, x, *prepared[0], 8, 1)
            ops.nvfp4_qpn2_gemm_sm70_out(result, intermediate, *prepared[1], 16, 2)

        def candidate(
            variant,
            x=x,
            packed_x=packed_x,
            intermediate=intermediate,
            result=result,
            prepared=prepared,
        ):
            ext.pack(x, packed_x)
            ext.run(intermediate, packed_x, *prepared[0], True, variant)
            ext.run(result, intermediate, *prepared[1], False, variant)

        functions = [production, partial(candidate, 6), partial(candidate, 7)]
        names = ["production", "packed_chain", "packed_scaled_chain"]
        checks = []
        for amplitude in [0.0, 0.125, -0.125, 0.25]:
            x.copy_(torch.randn_like(x) * amplitude)
            production()
            expected = result.clone()
            for name, fn in zip(names[1:], functions[1:], strict=True):
                fn()
                exact = torch.equal(
                    result.view(torch.int16), expected.view(torch.int16)
                )
                checks.append(
                    {"variant": name, "amplitude": amplitude, "bitwise": exact}
                )
                if not exact:
                    raise AssertionError((layer, checks[-1]))
        graphs = []
        for fn in functions:
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            begin = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            with torch.cuda.graph(graph):
                ext.evict(eviction, sink)
                begin.record()
                fn()
                end.record()
            graphs.append((graph, begin, end))
        samples = {name: [] for name in names}
        for repeat in range(65):
            for offset in range(3):
                index = (repeat + offset) % 3
                graph, begin, end = graphs[index]
                for _ in range(args.replays_per_sample):
                    graph.replay()
                end.synchronize()
                if repeat >= 5:
                    samples[names[index]].append(begin.elapsed_time(end) * 1000)
        records.append(
            {
                "layer": layer,
                "checks": checks,
                "weight_scale_bytes": sum(
                    w.numel() + s.numel() for w, s, _ in prepared
                ),
                "timing": {
                    name: {"mean_us": statistics.mean(values), "samples_us": values}
                    for name, values in samples.items()
                },
            }
        )
        print(
            json.dumps(
                {
                    "layer": layer,
                    "mean_us": {
                        name: statistics.mean(values)
                        for name, values in samples.items()
                    },
                }
            ),
            flush=True,
        )
        for graph, _, _ in graphs:
            graph.reset()
    args.output.write_text(
        json.dumps(
            {
                "scope": "TP-local M8 MLP subgraph, including pack; no AR or norms",
                "research_only": True,
                "cold_l2_bytes": 134217728,
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
