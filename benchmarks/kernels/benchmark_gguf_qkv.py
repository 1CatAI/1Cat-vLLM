# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen joint TP4 attention Q/K/V against coalesced canonical projections."""

import argparse
import json
from pathlib import Path

import gguf
import torch
from benchmark_gguf_iq3_gated import clocks, cold_graph

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_native_pair import _SOURCE_PACKERS
from vllm.model_executor.layers.quantization.gguf_qkvz import prepared_source_views
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    apply_prepared_gguf_projections,
    prepare_gguf_projections,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    assert 0 <= args.rank < 4
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    tensors = {t.name: t for t in gguf.GGUFReader(args.model).tensors}
    report = {"model": args.model.name, "rank": args.rank, "cases": []}
    flush = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    seen = set()
    for layer in range(64):
        names = [f"blk.{layer}.attn_{role}.weight" for role in ("q", "k", "v")]
        if not all(name in tensors for name in names):
            continue
        kinds = tuple(int(tensors[name].tensor_type) for name in names)
        if kinds in seen:
            continue
        seen.add(kinds)
        sources, references, raw_weights = [], [], []
        for name, kind, width in zip(names, kinds, (3072, 256, 256)):
            data = tensors[name].data
            assert data.shape[0] == width * 4, (name, data.shape)
            raw = data[args.rank * width : (args.rank + 1) * width].copy()
            raw_weights.append(raw)
            sources.append((torch.from_numpy(raw).cuda(), kind))
            references.append(
                torch.from_numpy(
                    gguf.quants.dequantize(raw, tensors[name].tensor_type)
                ).cuda()
            )
        prepared = prepare_gguf_projections(sources, torch.float16, True, 8)
        views = prepared_source_views(prepared)
        weights, scales, types = [], [], []
        for kind, raw, view in zip(kinds, raw_weights, views):
            if kind in (10, 12):
                assert view is not None
                codes, stats = view
                weights.append(codes)
                scales.append(stats)
                types.append(102 if kind == 10 else 104)
            else:
                weights.append(torch.from_numpy(_SOURCE_PACKERS[kind](raw)).cuda())
                scales.append(torch.empty(0, dtype=torch.int32, device="cuda"))
                types.append(kind)
        partials = torch.empty(56, 2, 512, dtype=torch.float32, device="cuda")
        counters = torch.zeros(56, dtype=torch.int32, device="cuda")
        output = torch.empty(8, 3584, dtype=torch.float16, device="cuda")

        def candidate(
            x,
            output=output,
            weights=weights,
            scales=scales,
            types=types,
            partials=partials,
            counters=counters,
        ):
            torch.ops._C.gguf_qkv_sm70_out(
                output, x, weights, scales, types, partials, counters
            )
            return output

        def canonical(x, prepared=prepared):
            return apply_prepared_gguf_projections(x, prepared)

        checks = []
        for seed in (131, 132, 133):
            torch.manual_seed(seed)
            x = torch.randn(8, 5120, dtype=torch.float16, device="cuda")
            oracle = torch.cat([(x.float() @ w.T).half() for w in references], -1)
            actual = candidate(x).clone()
            difference = actual.float() - oracle.float()
            errors = []
            for offset, width in ((0, 3072), (3072, 256), (3328, 256)):
                error = difference[:, offset : offset + width].norm()
                norm = oracle[:, offset : offset + width].float().norm()
                relative = float(error / norm)
                assert relative < 0.001, (layer, offset, relative)
                errors.append(relative)
            assert bool(torch.isfinite(actual).all())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replay = candidate(x)
            for _ in range(1000):
                graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(replay, actual, rtol=0, atol=0)
            assert torch.count_nonzero(counters).item() == 0
            checks.append(
                {
                    "seed": seed,
                    "relative_l2_by_source": errors,
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
            us = cold_graph(lambda x=x, call=call: call(x), flush)
            timings.append({"route": label, "us": us, "clock": clocks()})
        report["cases"].append(
            {
                "layer": layer,
                "source_types": kinds,
                "m": 8,
                "n": 3584,
                "k": 5120,
                "source_bytes": sum(raw.nbytes for raw in raw_weights),
                "candidate_weight_stream_bytes": sum(
                    t.numel() * t.element_size() for t in (*weights, *scales)
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
