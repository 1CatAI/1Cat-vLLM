# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen one-launch QKVZ/B/A against coalesced canonical TP4 projections."""

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from benchmark_gguf_iq3_gated import clocks, cold_graph

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_layout import GGUFHeadTilingLayout
from vllm.model_executor.layers.quantization.gguf_native_pair import _SOURCE_PACKERS
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    apply_prepared_gguf_projections,
    prepare_gguf_projections,
)


def load_sources(tensors, layer, rank):
    """Use the adapter's GDN head restoration before the exact TP row shard."""
    layout = GGUFHeadTilingLayout(3, 128)
    qkv = tensors[f"blk.{layer}.attn_qkv.weight"]
    z = tensors[f"blk.{layer}.attn_gate.weight"]
    qkv_data = torch.from_numpy(qkv.data.copy())
    pieces = [qkv_data[:2048], qkv_data[2048:4096]]
    pieces.append(layout.weight_to_vllm(qkv_data[4096:], dim=0))
    pieces.append(layout.weight_to_vllm(torch.from_numpy(z.data.copy()), dim=0))
    kinds = [int(qkv.tensor_type)] * 3 + [int(z.tensor_type)]
    sources, references, weights, stats, types = [], [], [], [], []
    raw_bytes = 0
    for data, kind in zip(pieces, kinds):
        width = data.shape[0] // 4
        raw = data[rank * width : (rank + 1) * width].numpy().copy()
        source = torch.from_numpy(raw).cuda()
        sources.append((source, kind))
        references.append(
            torch.from_numpy(
                gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(kind))
            ).cuda()
        )
        raw_bytes += raw.nbytes
        if kind in (10, 12):
            prepared = prepare_gguf_projections(
                [(source, kind)], torch.float16, True, 8
            )[0]
            weights.append(prepared.codes)
            stats.append(prepared.stats)
            types.append(102 if kind == 10 else 104)
        else:
            weights.append(torch.from_numpy(_SOURCE_PACKERS[kind](raw)).cuda())
            stats.append(torch.empty(0, dtype=torch.int32, device="cuda"))
            types.append(kind)
    floating = []
    for role in ("beta", "alpha"):
        tensor = tensors[f"blk.{layer}.ssm_{role}.weight"]
        assert int(tensor.tensor_type) == 30
        dense = torch.from_numpy(tensor.data.view(np.uint16).copy()).view(
            torch.bfloat16
        )
        dense = layout.weight_to_vllm(dense, dim=0, head_dim=1)
        shard = dense[rank * 12 : (rank + 1) * 12].half().cuda()
        assert bool(torch.isfinite(shard).all())
        floating.append(shard)
        sources.append((shard, 30))
        references.append(shard.float())
        raw_bytes += shard.numel() * shard.element_size()
    # N32/K128 lane-interleaved packets, padded to one N64 tile. Only the
    # twenty-four live columns are written; padding does not touch metadata.
    padded = torch.zeros(64, 5120, dtype=torch.float16)
    padded[:24] = torch.cat(floating).cpu()
    packed = padded.reshape(2, 32, 40, 16, 8).permute(0, 2, 3, 1, 4).contiguous()
    weights.append(packed.cuda())
    stats.append(torch.empty(0, dtype=torch.int32, device="cuda"))
    types.append(1)
    return sources, references, weights, stats, types, raw_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 0, 9, 20])
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    assert 0 <= args.rank < 4
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    assert torch.cuda.get_device_capability() == (7, 0)
    tensors = {t.name: t for t in gguf.GGUFReader(args.model).tensors}
    report = {
        "model": args.model.name,
        "rank": args.rank,
        "cases": [],
        "clock_before": clocks(),
    }
    flush = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for layer in args.layers:
        sources, refs, weights, stats, types, raw_bytes = load_sources(
            tensors, layer, args.rank
        )
        prepared = prepare_gguf_projections(sources, torch.float16, True, 8)
        partials = torch.empty(65, 2, 512, dtype=torch.float32, device="cuda")
        counters = torch.zeros(65, dtype=torch.int32, device="cuda")
        output = torch.empty(8, 4120, dtype=torch.float16, device="cuda")

        def candidate(
            x,
            output=output,
            weights=weights,
            stats=stats,
            types=types,
            partials=partials,
            counters=counters,
        ):
            torch.ops._C.gguf_qkvz_sm70_out(
                output, x, weights, stats, types, partials, counters
            )
            return output

        def canonical(x, prepared=prepared):
            return apply_prepared_gguf_projections(x, prepared)

        checks = []
        for seed in (131, 132, 133):
            torch.manual_seed(seed)
            x = torch.randn(8, 5120, dtype=torch.float16, device="cuda")
            oracle = torch.cat([(x.float() @ w.T).half() for w in refs], dim=-1)
            actual = candidate(x).clone()
            difference = actual.float() - oracle.float()
            relative = float(difference.norm() / oracle.float().norm())
            assert bool(torch.isfinite(actual).all()) and relative < 0.001, (
                layer,
                relative,
            )
            for offset, width in ((4096, 12), (4108, 12)):
                error = difference[:, offset : offset + width].norm()
                assert (
                    float(error / oracle[:, offset : offset + width].float().norm())
                    < 0.001
                )
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
                    "relative_l2": relative,
                    "max_abs": float(difference.abs().max()),
                }
            )
        timings = []
        for label, function in (
            ("canonical_a", canonical),
            ("qkvz_b", candidate),
            ("qkvz_b", candidate),
            ("canonical_a", canonical),
        ):
            time = cold_graph(lambda function=function, x=x: function(x), flush)
            timings.append({"arm": label, "us": time, "clock": clocks()})
        report["cases"].append(
            {
                "layer": layer,
                "source_types": [t for _, t in sources],
                "operator_types": types,
                "m": 8,
                "n": 4120,
                "k": 5120,
                "source_bytes": raw_bytes,
                "loaded_candidate_bytes": sum(
                    w.numel() * w.element_size() + s.numel() * s.element_size()
                    for w, s in zip(weights, stats)
                ),
                "checks": checks,
                "timings": timings,
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report.update(complete=True, clock_after=clocks())
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
