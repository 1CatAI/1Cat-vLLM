# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare FP16-source compact top-64 selection under CUDA graph replay."""

import argparse
import json
import statistics
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, required=True)
    args = parser.parse_args()
    logits = torch.load(args.logits, map_location="cuda", weights_only=True)
    if logits.ndim != 2 or logits.dtype != torch.float16:
        raise ValueError("Supply retained two-dimensional FP16 head output.")
    from vllm.model_executor.layers.sm70_compact_topk import compact_half_topk

    flush = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    graphs = []
    for name, select in (
        ("torch", lambda: torch.topk(logits.float(), 64, dim=-1)),
        ("compact", lambda: compact_half_topk(logits)),
    ):
        select()
        torch.accelerator.synchronize()
        begin = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flush.fill_(0)
            begin.record()
            retained = select()
            end.record()
        graphs.append((name, graph, begin, end, retained))
    samples = [[], []]
    for iteration in range(45):
        for index in (iteration % 2, (iteration + 1) % 2):
            _, graph, begin, end, _ = graphs[index]
            for _ in range(10):
                graph.replay()
            end.synchronize()
            if iteration >= 5:
                samples[index].append(begin.elapsed_time(end) * 1000)
    print(
        json.dumps(
            [
                {"route": graph[0], "us": statistics.mean(sample)}
                for graph, sample in zip(graphs, samples)
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
