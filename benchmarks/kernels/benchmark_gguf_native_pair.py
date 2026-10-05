# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate installed IQ3_S/IQ4_XS gated pairs on actual TP4 weight slices."""

import argparse
import json
from pathlib import Path

import gguf
import torch
from benchmark_gguf_iq3_gated import clocks, cold_graph

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_iq3_records import (
    signed_index_records,
)
from vllm.model_executor.layers.quantization.gguf_iq4_native import (
    pack_iq4_xs_records,
)
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    prepare_gguf_projections,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[39, 42])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    assert torch.cuda.get_device_capability() == (7, 0)
    assert hasattr(torch.ops._C, "gguf_native_pair_sm70_out")
    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(args.model).tensors}
    report = {"model": args.model.name, "cases": [], "clock_before": clocks()}
    for layer in args.layers:
        names = [f"blk.{layer}.ffn_{role}.weight" for role in ("gate", "up")]
        types = [int(tensors[name].tensor_type) for name in names]
        assert tuple(types) in ((21, 23), (23, 21)), types
        raw = [tensors[name].data[:4352].copy() for name in names]
        sources = [
            (torch.from_numpy(data).cuda(), kind) for data, kind in zip(raw, types)
        ]
        projections = prepare_gguf_projections(sources, torch.float16, True, 8)
        packed = [
            torch.from_numpy(
                signed_index_records(data) if kind == 21 else pack_iq4_xs_records(data)
            ).cuda()
            for data, kind in zip(raw, types)
        ]
        references = [
            torch.from_numpy(
                gguf.quants.dequantize(data, gguf.GGMLQuantizationType(kind))
            ).cuda()
            for data, kind in zip(raw, types)
        ]
        output = torch.empty(8, 4352, dtype=torch.float16, device="cuda")

        def native(rows, output=output, packed=packed, types=types):
            torch.ops._C.gguf_native_pair_sm70_out(output, rows, *packed, *types)
            return output

        def canonical(rows, output=output, projections=projections):
            pair = torch.cat([projection(rows) for projection in projections], dim=-1)
            result = torch.empty_like(output)
            torch.ops._C.silu_and_mul(result, pair)
            return result

        checks = []
        for seed in (131, 132, 133):
            torch.manual_seed(seed)
            rows = torch.randn(8, 5120, dtype=torch.float16, device="cuda")
            gate, up = [(rows.float() @ weight.T).half() for weight in references]
            oracle = (gate.float() / (1 + torch.exp(-gate.float()))).half() * up
            for label, result in (
                ("native", native(rows)),
                ("canonical", canonical(rows)),
            ):
                difference = result.float() - oracle.float()
                relative = float(difference.norm() / oracle.float().norm())
                assert bool(torch.isfinite(result).all()) and relative < 0.001, (
                    layer,
                    label,
                    relative,
                )
                checks.append(
                    {
                        "seed": seed,
                        "route": label,
                        "relative_l2": relative,
                        "max_abs": float(difference.abs().max()),
                    }
                )
        flush = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        payload_bytes = sum(data.nbytes for data in raw)
        timings = []
        for label, call in (
            ("canonical", canonical),
            ("native", native),
            ("native", native),
            ("canonical", canonical),
        ):
            before = clocks()
            elapsed = cold_graph(lambda call=call, rows=rows: call(rows), flush)
            timings.append(
                {
                    "route": label,
                    "median_us": elapsed,
                    "source_payload_gbps": payload_bytes / elapsed / 1000,
                    "clock_before": before,
                    "clock_after": clocks(),
                }
            )
        report["cases"].append(
            {
                "layer": layer,
                "tensors": names,
                "types": types,
                "m": 8,
                "n": 4352,
                "k": 5120,
                "source_bytes": payload_bytes,
                "checks": checks,
                "abba": timings,
            }
        )
    report["clock_after"] = clocks()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
