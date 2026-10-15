# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound address-translation sensitivity separately from cold-L2 payload.

The target operation is unchanged. Before it, compare a 128MiB read eviction
with the same eviction plus sparse reads spanning either 128MiB or 4GiB.
This is a perturbation experiment, not a measured model working set.
"""

import argparse
import json
import runpy
import statistics
from functools import partial
from pathlib import Path

import torch

from vllm import _sm70_ops as ops
from vllm.triton_utils import tl, triton


@triton.jit
def touch_span(storage, sink, PAGES: tl.constexpr):
    block = tl.program_id(0)
    page = (block * 32 + tl.arange(0, 32)) % PAGES
    values = tl.load(storage + page.to(tl.int64) * 2097152, cache_modifier=".cg")
    tl.store(sink + block, tl.sum(values.to(tl.int32), 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(20261005)
    torch.ops.load_library(str(args.library))
    loader = runpy.run_path(
        str(args.source_root / "benchmarks/kernels/benchmark_sm70_nvfp4_qpn2.py")
    )["_load_projection_shards"]
    storage = torch.zeros(4 * 1024**3, device="cuda", dtype=torch.uint8)
    sink = torch.empty(160, device="cuda", dtype=torch.int32)
    eviction = torch.ones(33554432, device="cuda", dtype=torch.int32)
    evicted = torch.empty(256, device="cuda", dtype=torch.int64)
    records = []
    for projection in loader(args.model, 0, 0, 4):
        w, s = ops.nvfp4_qpn2_prepare_sm70(
            projection.packed.cuda(), projection.scales.cuda()
        )
        n, packed_k = projection.packed.shape
        gated = projection.gated_silu
        x = torch.randn(8, packed_k * 2, device="cuda", dtype=torch.float16) * 0.1
        y = x.new_empty((8, n // 2 if gated else n))
        op = ops.nvfp4_qpn2_gated_sm70_out if gated else ops.nvfp4_qpn2_gemm_sm70_out

        run = partial(
            op,
            y,
            x,
            w,
            s,
            projection.inverse_global_scale,
            8 if gated else 16,
            1 if gated else 2,
        )

        run()
        expected = y.clone()
        graphs = []
        names = ["l2_only", "l2_plus_128mib_span", "l2_plus_4gib_span"]
        for pages in [0, 64, 2048]:
            if pages:
                touch_span[(160,)](storage, sink, pages, num_warps=4)
            run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            begin, end = [
                torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)
            ]
            with torch.cuda.graph(graph):
                torch.ops._qpn_operands700.evict(eviction, evicted)
                if pages:
                    touch_span[(160,)](storage, sink, pages, num_warps=4)
                begin.record()
                run()
                end.record()
            graph.replay()
            end.synchronize()
            assert torch.equal(y.view(torch.int16), expected.view(torch.int16))
            graphs.append((graph, begin, end))
        samples = {name: [] for name in names}
        for repeat in range(65):
            for offset in range(3):
                i = (repeat + offset) % 3
                graph, begin, end = graphs[i]
                for _ in range(10):
                    graph.replay()
                end.synchronize()
                if repeat >= 5:
                    samples[names[i]].append(begin.elapsed_time(end) * 1000)
        row = dict(
            gated=gated,
            weight_scale_bytes=w.numel() + s.numel(),
            timing={
                name: dict(mean_us=statistics.mean(a), samples_us=a)
                for name, a in samples.items()
            },
        )
        records.append(row)
        print(
            json.dumps(
                dict(
                    gated=gated,
                    mean_us={name: statistics.mean(a) for name, a in samples.items()},
                )
            ),
            flush=True,
        )
        for graph, _, _ in graphs:
            graph.reset()
    args.output.write_text(
        json.dumps(
            dict(
                research_only=True,
                no_kernel_change=True,
                fixed_shape="M8 TP4",
                l2_eviction_bytes=134217728,
                span_stride_bytes=2097152,
                largest_span_bytes=4 * 1024**3,
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
