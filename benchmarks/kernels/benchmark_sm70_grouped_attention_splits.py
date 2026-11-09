# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen split schedules against FP64 and a sixteen-layer KV working set.

This records final FP16-output error only. It cannot admit arithmetic changes:
native pre-cast FP32 error, sanitizers and model quality remain separate gates.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_sm70_grouped_attention_long import (
    call,
    load_operator,
    make_case,
)


def fp64_reference(case, length):
    order = case["table"][0].long()
    key = case["k"].index_select(0, order).reshape(-1, 256)[:length]
    value = case["v"].index_select(0, order).reshape(-1, 256)[:length]
    key = key.view(torch.float8_e4m3fn).double()
    value = value.view(torch.float8_e4m3fn).double()
    query = case["q"].reshape(-1, 256).double()
    scores = (query @ key.T) * (0.0625 * 0.5)
    lengths = case["lengths"].repeat_interleave(6)
    visible = torch.arange(length, device="cuda")[None, :] < lengths[:, None]
    scores.masked_fill_(~visible, -torch.inf)
    probabilities = scores.softmax(-1)
    probabilities.masked_fill_(lengths[:, None] == 0, 0)
    return ((probabilities @ value) * 1.25).reshape_as(case["q"])


def error(output, reference):
    difference = output.double() - reference
    absolute = difference.abs().reshape(-1)
    return {
        "max_abs": absolute.max().item(),
        "p99_abs": torch.quantile(absolute, 0.99).item(),
        "relative_l2": (difference.norm() / reference.norm().clamp_min(1e-30)).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.manual_seed(20260910)
    loaded = [load_operator(p) for p in [args.baseline, *args.candidate]]
    operators, manifests = zip(*loaded)
    assert len({id(op) for op in operators}) == len(operators)
    splits = [m.get("splits", 80) for m in manifests]
    report = {
        "manifests": manifests,
        "splits": splits,
        "reference": "Independent FP64 QK, softmax and PV; decoded E4M3 operands",
        "error_scope": "Final FP16 output; native pre-cast FP32 not yet measured",
        "arithmetic_admitted": False,
        "complete_round_performance": False,
        "checks": [],
        "performance": [],
        "complete": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for length in (129, 3297, 32768, 131072):
        case = make_case(8, 3296, length, len(operators), split_counts=splits)
        case["lengths"][-1] = 0
        reference = fp64_reference(case, length)
        for arm, operator in enumerate(operators):
            call(case, arm, operator)
            output = case["arms"][arm]["out"]
            assert torch.isfinite(output).all()
            assert (output[-1] == 0).all()
            assert (case["arms"][arm]["guard"][[0, -1]] == -777).all()
            report["checks"].append(
                {"length": length, "splits": splits[arm], **error(output, reference)}
            )
        save()
        print(json.dumps({"reference_length": length}), flush=True)
        del case, reference

    for length in (1024, 32768, 65536, 131072):
        cases = [
            make_case(8, 3296, length, len(operators), split_counts=splits)
            for _ in range(16)
        ]
        graphs = []
        for arm, operator in enumerate(operators):
            for case in cases:
                call(case, arm, operator)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for case in cases:
                    call(case, arm, operator)
            graphs.append(graph)
        for _ in range(5):
            for graph in graphs:
                graph.replay()
        samples = [[] for _ in operators]
        for trial in range(5):
            order = list(range(len(operators)))
            if trial % 2:
                order.reverse()
            for arm in order:
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(8):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                samples[arm].append(start.elapsed_time(end) / 8)
        row = {
            "context": length,
            "layers": 16,
            "page": 3296,
            "samples_ms": samples,
            "median_ms": [statistics.median(x) for x in samples],
        }
        report["performance"].append(row)
        save()
        print(json.dumps(row), flush=True)
        del cases, graphs
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
