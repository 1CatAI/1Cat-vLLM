# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the packaged single-projection reader with canonical TP4 down.

Hold the shared GPU leases before running. This operator benchmark does not
admit any model shape. L2 eviction is outside the timed graph interval.
"""

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from benchmark_gguf_iq3_gated import clocks, cold_graph

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_native_pair import _SOURCE_PACKERS
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    apply_prepared_gguf_projections,
    prepare_gguf_projections,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    assert torch.cuda.get_device_capability() == (7, 0)
    assert hasattr(torch.ops._C, "gguf_native_linear_sm70_out")
    tensors = {t.name: t for t in gguf.GGUFReader(args.model).tensors}
    layers = args.layers
    if layers is None:
        by_type = {}
        for layer in range(64):
            tensor = tensors[f"blk.{layer}.ffn_down.weight"]
            by_type.setdefault(int(tensor.tensor_type), layer)
        layers = list(by_type.values())
    report = {
        "model": args.model.name,
        "scope": "unadmitted operator; real TP4 down weights; cold-L2 graph ABBA",
        "cases": [],
        "complete": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for layer in layers:
        tensor = tensors[f"blk.{layer}.ffn_down.weight"]
        source_type = int(tensor.tensor_type)
        assert source_type in _SOURCE_PACKERS
        # Row-parallel TP4 splits complete source blocks along K.
        raw = np.ascontiguousarray(tensor.data[:, : tensor.data.shape[1] // 4])
        n, k = 5120, 4352
        assert raw.shape[0] == n
        records = torch.from_numpy(_SOURCE_PACKERS[source_type](raw)).cuda()
        source = torch.from_numpy(raw).cuda()
        projections = prepare_gguf_projections(
            [(source, source_type)], torch.float16, True, 8
        )
        canonical_bytes = sum(
            t.numel() * t.element_size()
            for p in projections
            for t in (p.codes, p.stats)
        )
        official = torch.from_numpy(
            gguf.quants.dequantize(raw, tensor.tensor_type)
        ).cuda()
        assert official.shape == (n, k)
        out = torch.empty(8, n, dtype=torch.float16, device="cuda")

        def native(x, out=out, records=records, source_type=source_type):
            torch.ops._C.gguf_native_linear_sm70_out(out, x, records, source_type)
            return out

        def canonical(x, projections=projections):
            return apply_prepared_gguf_projections(x, projections)

        checks = []
        for seed in (131, 132, 133):
            torch.manual_seed(seed)
            rows = torch.randn(8, k, dtype=torch.float16, device="cuda")
            reference = (rows.float() @ official.T).half()
            for label, value in (
                ("native", native(rows)),
                ("canonical", canonical(rows)),
            ):
                diff = value.float() - reference.float()
                relative = float(diff.norm() / reference.float().norm())
                checks.append(
                    {
                        "seed": seed,
                        "route": label,
                        "relative_l2": relative,
                        "max_abs": float(diff.abs().max()),
                    }
                )
                assert torch.isfinite(value).all() and relative < 0.001, checks[-1]
            expected = native(rows).clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                native(rows)
            for _ in range(3):
                graph.replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
        flush = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        timings = []
        for label, call in (
            ("canonical", canonical),
            ("native", native),
            ("native", native),
            ("canonical", canonical),
        ):
            before = clocks()
            us = cold_graph(lambda call=call, rows=rows: call(rows), flush)
            read_bytes = raw.nbytes if label == "native" else canonical_bytes
            timings.append(
                {
                    "route": label,
                    "median_us": us,
                    "weight_stream_bytes": read_bytes,
                    "effective_weight_gbps": read_bytes / us / 1000,
                    "clock_before": before,
                    "clock_after": clocks(),
                }
            )
        report["cases"].append(
            {
                "layer": layer,
                "source_type": source_type,
                "tensor": tensor.name,
                "m": 8,
                "n": n,
                "k": k,
                "source_bytes": raw.nbytes,
                "canonical_bytes": canonical_bytes,
                "checks": checks,
                "graph_bitwise_equal": True,
                "abba": timings,
            }
        )
        save()
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
