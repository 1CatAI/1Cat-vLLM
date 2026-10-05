# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen QPN readers for already prepared canonical U4/IQ4 weight streams."""

import argparse
import json
from pathlib import Path

import gguf
import torch
from benchmark_gguf_iq3_gated import clocks, cold_graph

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    apply_prepared_gguf_projections,
    prepare_gguf_projections,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    tensors = {t.name: t for t in gguf.GGUFReader(args.model).tensors}
    report = {"model": args.model.name, "cases": [], "complete": False}
    flush = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    cases = []
    seen = set()
    for layer in range(64):
        down = tensors[f"blk.{layer}.ffn_down.weight"]
        kind = int(down.tensor_type)
        if kind in (12, 23) and ("down", kind) not in seen:
            cases.append((layer, "down", [down], [kind]))
            seen.add(("down", kind))
        gate, up = [tensors[f"blk.{layer}.ffn_{r}.weight"] for r in ("gate", "up")]
        kinds = [int(t.tensor_type) for t in (gate, up)]
        if kinds in ([23, 23], [12, 12]) and ("pair", kinds[0]) not in seen:
            cases.append((layer, "pair", [gate, up], kinds))
            seen.add(("pair", kinds[0]))
    for layer, role, tensors_, kinds in cases:
        raw = [
            t.data[:, : t.data.shape[1] // 4].copy()
            if role == "down"
            else t.data[:4352].copy()
            for t in tensors_
        ]
        n, k = (5120, 4352) if role == "down" else (4352, 5120)
        projections = prepare_gguf_projections(
            [(torch.from_numpy(w).cuda(), kind) for w, kind in zip(raw, kinds)],
            torch.float16,
            True,
            8,
        )
        assert len(projections) == 1
        projection = projections[0]
        views = []
        offset = 0
        for width in projection.source_output_sizes:
            count = width * k // 8
            code = projection.codes.reshape(-1).narrow(0, offset * k // 8, count)
            views.append(
                (code.view(k, width // 8), projection.stats[:, offset : offset + width])
            )
            offset += width
        output = torch.empty(8, n, dtype=torch.float16, device="cuda")
        partials = torch.empty(n // 64, 2, 512, dtype=torch.float32, device="cuda")
        counters = torch.zeros(n // 64, dtype=torch.int32, device="cuda")

        def candidate(
            x,
            role=role,
            output=output,
            views=views,
            kinds=kinds,
            partials=partials,
            counters=counters,
        ):
            if role == "pair":
                torch.ops._C.gguf_canonical_pair_sm70_out(
                    output, x, *views[0], *views[1], 1 if kinds[0] == 23 else 0
                )
            elif kinds[0] == 23:
                torch.ops._C.gguf_canonical_iq4_linear_sm70_out(
                    output, x, *views[0], partials, counters
                )
            else:
                torch.ops._C.gguf_canonical_linear_n64_sm70_out(
                    output, x, *views[0], partials, counters, 4, 32
                )
            return output

        def canonical(x, projections=projections, role=role):
            y = apply_prepared_gguf_projections(x, projections)
            if role == "pair":
                g, u = y.chunk(2, -1)
                y = torch.nn.functional.silu(g) * u
            return y

        reference = [
            torch.from_numpy(gguf.quants.dequantize(w, t.tensor_type)).cuda()
            for w, t in zip(raw, tensors_)
        ]
        checks = []
        for seed in (131, 132, 133):
            torch.manual_seed(seed)
            x = torch.randn(8, k, dtype=torch.float16, device="cuda")
            expected = [(x.float() @ w.T).half() for w in reference]
            oracle = (
                torch.nn.functional.silu(expected[0]) * expected[1]
                if role == "pair"
                else expected[0]
            )
            actual = candidate(x).clone()
            difference = actual.float() - oracle.float()
            relative = float(difference.norm() / oracle.float().norm())
            assert relative < 0.002 and bool(torch.isfinite(actual).all()), relative
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replay = candidate(x)
            for _ in range(1000):
                graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(replay, actual, rtol=0, atol=0)
            assert int(counters.abs().max()) == 0
            checks.append(
                {
                    "seed": seed,
                    "relative_l2": relative,
                    "max_abs": float(difference.abs().max()),
                }
            )
        timings = []
        for label, call in (
            ("canonical", canonical),
            ("candidate", candidate),
            ("candidate", candidate),
            ("canonical", canonical),
        ):
            us = cold_graph(lambda call=call, x=x: call(x), flush)
            timings.append({"route": label, "us": us, "clock": clocks()})
        report["cases"].append(
            {
                "layer": layer,
                "role": role,
                "source_types": kinds,
                "n": n,
                "k": k,
                "source_bytes": sum(w.nbytes for w in raw),
                "candidate_weight_stream_bytes": sum(
                    t.numel() * t.element_size()
                    for t in (projection.codes, projection.stats)
                ),
                "checks": checks,
                "timings": timings,
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
