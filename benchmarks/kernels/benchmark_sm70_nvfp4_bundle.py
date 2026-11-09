# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native QPN2 layouts using real TP4 MLP weights and cold graphs.

Both layouts are retained here for the experiment. Production retains one.
CUDA events exclude cache eviction and all host/packing work.
"""

import argparse
import json
import random
import runpy
import statistics
from pathlib import Path

import torch

from vllm import _sm70_ops as ops


def paired_graphs(control, candidate, iterations):
    eviction = torch.empty(128 * 2**20, device="cuda", dtype=torch.uint8)
    graphs = []
    for run in (control, candidate):
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        begin = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            eviction.fill_(1)
            begin.record()
            run()
            end.record()
        graphs.append((graph, begin, end))
    randomizer = random.Random(123)
    samples = [[], []]
    for iteration in range(iterations + 20):
        order = [0, 1]
        randomizer.shuffle(order)
        for arm in order:
            graph, begin, end = graphs[arm]
            graph.replay()
            end.synchronize()
            if iteration >= 20:
                samples[arm].append(begin.elapsed_time(end) * 1000)
    savings = [a - b for a, b in zip(*samples)]
    bootstrap = sorted(
        statistics.mean(randomizer.choices(savings, k=len(savings)))
        for _ in range(2000)
    )
    return {
        "control_us": statistics.mean(samples[0]),
        "bundled_us": statistics.mean(samples[1]),
        "saving_us": statistics.mean(savings),
        "saving_95ci_us": [bootstrap[50], bootstrap[1949]],
        "samples_us": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 32])
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    loader = runpy.run_path(
        str(Path(__file__).with_name("benchmark_sm70_nvfp4_qpn2.py"))
    )["_load_projection_shards"]
    weights = []
    for projection in loader(args.model, 0, 0, 4):
        c, s = ops.nvfp4_qpn2_prepare_sm70(
            projection.packed.cuda(), projection.scales.cuda()
        )
        bundled = ops.nvfp4_qpn2_bundle_sm70(c, s)
        weights.append(((c, s), bundled, projection.inverse_global_scale))
    results = []
    for rows in args.rows:
        x = torch.randn(rows, 5120, dtype=torch.float16, device="cuda") * 0.125
        mid = [torch.empty(rows, 4352, device="cuda", dtype=x.dtype) for _ in range(2)]
        out = [torch.empty_like(x) for _ in range(2)]

        def run(arm, x=x, mid=mid, out=out, rows=rows):
            c, s = weights[0][arm]
            ops.nvfp4_qpn2_gated_sm70_out(
                mid[arm], x, c, s, weights[0][2], 8, 1 if rows <= 8 else 2
            )
            c, s = weights[1][arm]
            ops.nvfp4_qpn2_gemm_sm70_out(out[arm], mid[arm], c, s, weights[1][2], 16, 2)

        for amplitude in (0.01, 0.125, 1.0, 4.0):
            x.normal_().mul_(amplitude)
            run(0)
            run(1)
            assert torch.equal(mid[0].view(torch.int16), mid[1].view(torch.int16))
            assert torch.equal(out[0].view(torch.int16), out[1].view(torch.int16))
        result = paired_graphs(lambda: run(0), lambda: run(1), args.iterations)
        result.update(rows=rows, output_bitwise=True, compute_kernels=2)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "samples_us"}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
