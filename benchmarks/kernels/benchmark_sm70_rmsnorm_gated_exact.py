# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-built gated RMSNorm versus its native chain, 36 checkpoint weights.

Inputs are synthetic. This measures a component graph, not model throughput.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import torch
from safetensors import safe_open

from vllm.model_executor.layers.layernorm import RMSNormGated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", default="12,60,120")
    parser.add_argument("--activation", choices=("sigmoid", "silu"), default="sigmoid")
    args = parser.parse_args()
    if not hasattr(torch.ops._C, "sm70_rmsnorm_gated_exact_out"):
        parser.error("Build this source tree's normal native extension first")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    keys = sorted(
        key
        for key in index
        if key.endswith(".linear_attn.norm.weight") and "mtp" not in key
    )
    weights = []
    for key in keys:
        with safe_open(args.model / index[key], framework="pt", device="cpu") as handle:
            weights.append(handle.get_tensor(key).half().to("cuda"))
    if len(weights) != 36:
        parser.error("Expected 36 target GDN norm weights")
    torch.manual_seed(20260927)
    torch.set_num_threads(1)
    native = Path(__file__).resolve().parents[2] / "vllm/_C.abi3.so"
    report = dict(
        synthetic_inputs=True,
        layers=len(weights),
        activation=args.activation,
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        rows=[],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for rows in map(int, args.rows.split(",")):
        xs = [
            torch.randn(rows, 128, device="cuda", dtype=torch.float16) for _ in weights
        ]
        zs = [torch.randn_like(x) for x in xs]
        outputs = [[], []]
        graphs = []

        def run(arm, xs=xs, zs=zs, outputs=outputs):
            outputs[arm].clear()
            for x, z, weight in zip(xs, zs, weights):
                if arm == 0:
                    value = RMSNormGated.forward_static(
                        x,
                        z,
                        weight,
                        1e-6,
                        x.dtype,
                        norm_before_gate=True,
                        activation=args.activation,
                    )
                else:
                    value = torch.ops.vllm.sm70_rmsnorm_gated_exact(
                        x, z, weight, 1e-6, args.activation == "silu"
                    )
                outputs[arm].append(value)

        for arm in range(2):
            for _ in range(3):
                run(arm)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(arm)
            graphs.append(graph)
        mismatches = []
        for scale in (0.0, 0.001, 0.1, 1.0, 3.0, 30.0):
            for x, z in zip(xs, zs):
                x.normal_(0, scale)
                z.normal_(0, scale)
            for output in outputs[1]:
                output.fill_(float("nan"))
            for graph in graphs:
                graph.replay()
            mismatches.append(
                sum(
                    int((a.view(torch.int16) != b.view(torch.int16)).count_nonzero())
                    for a, b in zip(*outputs)
                )
            )
        row = dict(rows=rows, mismatches=mismatches)
        report["rows"].append(row)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        if any(mismatches):
            raise AssertionError(row)
        samples = [[], []]
        for trial in range(7):
            for arm in (0, 1) if trial % 2 == 0 else (1, 0):
                for _ in range(3):
                    graphs[arm].replay()
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                begin.record()
                for _ in range(40):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                samples[arm].append(begin.elapsed_time(end) / 40)
        row.update(
            control_ms=median(samples[0]),
            candidate_ms=median(samples[1]),
            samples_ms=samples,
        )
        print(json.dumps(row), flush=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
