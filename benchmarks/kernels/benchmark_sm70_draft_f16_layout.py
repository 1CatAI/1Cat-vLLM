# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen cuBLAS operand layouts using retained actual TP4 draft projections.

Numerical checks replay all four ranks' real weights and inputs on GPU 4.
The separate timing uses twenty consecutive-layer weights from rank zero.
No altered layout or padded shape is admitted without bytewise output parity.
"""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch


def errors(value, reference):
    difference = (value.double() - reference).abs().flatten()
    return dict(
        max_abs=difference.max().item(),
        p99_abs=torch.quantile(difference, 0.99).item(),
        relative_l2=(difference.norm() / reference.norm()).item(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4"
    assert torch.cuda.get_device_capability() == (7, 0)
    modes = ["row_weight", "column_weight", "row_weight_m16", "column_weight_m16"]
    numerical, working_set, provenance = [], [], []

    def run(item, mode):
        x = item["padded"] if mode >= 2 else item["x"]
        weight = item["column_weight"] if mode % 2 else item["weight"]
        torch.mm(x, weight.T, out=item["outputs"][mode])

    for rank in range(4):
        root = args.root / f"rank{rank}"
        sample_paths = sorted(root.glob("inputs-step*.pt"))
        weight_paths = sorted(root.glob("*-weight.pt"))
        assert len(sample_paths) == 5 and len(weight_paths) == 20
        samples = [
            torch.load(p, map_location="cpu", weights_only=True) for p in sample_paths
        ]
        for path in sample_paths + weight_paths:
            provenance.append(
                dict(
                    file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                )
            )
        for path in weight_paths:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            name, cpu_weight = saved["name"], saved["weight"]
            weight = cpu_weight.cuda()
            n, k = weight.shape
            item = dict(
                weight=weight,
                column_weight=weight.T.contiguous().T,
                x=torch.empty((8, k), device="cuda", dtype=torch.float16),
                padded=torch.zeros((16, k), device="cuda", dtype=torch.float16),
                outputs=[
                    torch.empty(
                        (8 if m < 2 else 16, n), device="cuda", dtype=torch.float16
                    )
                    for m in range(4)
                ],
            )
            assert torch.equal(weight, item["column_weight"])
            weight64 = weight.double()
            for sample_path, snapshot in zip(sample_paths, samples):
                values = snapshot[name]
                item["x"].copy_(values["input"])
                item["padded"][:8].copy_(item["x"])
                oracle = item["x"].double() @ weight64.T
                for mode in range(4):
                    run(item, mode)
                    output = item["outputs"][mode][:8]
                    if mode == 0:
                        control_errors = errors(output, oracle)
                        saved_equal = torch.equal(
                            output.cpu().view(torch.uint8),
                            values["control"].view(torch.uint8),
                        )
                    metrics = errors(output, oracle)
                    numerical.append(
                        dict(
                            rank=rank,
                            name=name,
                            snapshot=sample_path.name,
                            mode=modes[mode],
                            shape=[8, n, k],
                            saved_control_equal=saved_equal,
                            byte_equal=torch.equal(
                                output.view(torch.uint8),
                                item["outputs"][0].view(torch.uint8),
                            ),
                            finite=bool(torch.isfinite(output).all()),
                            control=control_errors,
                            candidate=metrics,
                            reference_error_not_expanded=all(
                                metrics[key] <= control_errors[key] for key in metrics
                            ),
                        )
                    )
            if rank == 0:
                working_set.append(item)
        print(f"Checked actual rank {rank} inputs", flush=True)
    assert len(working_set) == 20
    graphs = []
    for mode in range(4):
        for _ in range(3):
            for item in working_set:
                run(item, mode)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for item in working_set:
                run(item, mode)
        graphs.append(graph)
    timings = [[] for _ in modes]
    for trial in range(7):
        for mode in range(4) if trial % 2 == 0 else reversed(range(4)):
            for _ in range(20):
                graphs[mode].replay()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(50):
                graphs[mode].replay()
            end.record()
            end.synchronize()
            timings[mode].append(start.elapsed_time(end) / 50)
    summary = {}
    for mode in modes:
        rows = [r for r in numerical if r["mode"] == mode]
        summary[mode] = dict(
            comparisons=len(rows),
            differing=sum(not r["byte_equal"] for r in rows),
            expanded_reference_error=sum(
                not r["reference_error_not_expanded"] for r in rows
            ),
            mismatched_saved_control=sum(not r["saved_control_equal"] for r in rows),
        )
    report = dict(
        modes=modes,
        summary=summary,
        numerical=numerical,
        provenance=provenance,
        samples_ms=timings,
        medians_ms=[statistics.median(t) for t in timings],
        complete_round_performance=False,
        torch_version=torch.__version__,
        allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("numerical", "provenance")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
