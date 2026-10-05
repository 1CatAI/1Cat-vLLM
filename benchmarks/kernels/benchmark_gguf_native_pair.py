# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate installed mixed gated pairs on actual TP4 weight slices."""

import argparse
import json
from pathlib import Path

import gguf
import torch
from benchmark_gguf_iq3_gated import clocks, cold_graph

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_native_pair import (
    apply_native_gated_pair,
    prepare_native_gated_pair,
)
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    apply_prepared_gguf_projections,
    prepare_gguf_projections,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[39, 42])
    parser.add_argument(
        "--prototype-iq3-xxs",
        action="store_true",
        help="Test the unadmitted IQ3_XXS reader through the raw operator",
    )
    parser.add_argument("--prototype-q4-k", action="store_true")
    parser.add_argument("--prototype-iq3xxs-iq4", action="store_true")
    parser.add_argument("--prototype-iq2-s", action="store_true")
    parser.add_argument("--prototype-iq2-xs", action="store_true")
    parser.add_argument("--prototype-iq2-xxs", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Skip timing after numerical and graph checks",
    )
    args = parser.parse_args()
    assert (
        sum(
            (
                args.prototype_iq3_xxs,
                args.prototype_q4_k,
                args.prototype_iq3xxs_iq4,
                args.prototype_iq2_s,
                args.prototype_iq2_xs,
                args.prototype_iq2_xxs,
            )
        )
        <= 1
    )
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    assert torch.cuda.get_device_capability() == (7, 0)
    assert hasattr(torch.ops._C, "gguf_native_pair_sm70_out")
    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(args.model).tensors}
    report = {"model": args.model.name, "cases": [], "clock_before": clocks()}
    for layer in args.layers:
        names = [f"blk.{layer}.ffn_{role}.weight" for role in ("gate", "up")]
        types = [int(tensors[name].tensor_type) for name in names]
        allowed = (
            ((17, 16), (16, 22))
            if args.prototype_iq2_xxs
            else ((17, 18), (22, 17))
            if args.prototype_iq2_xs
            else ((22, 21), (21, 22), (22, 18), (18, 22))
            if args.prototype_iq2_s
            else ((18, 23),)
            if args.prototype_iq3xxs_iq4
            else ((12, 23), (23, 12), (12, 21), (21, 12))
            if args.prototype_q4_k
            else ((18, 21), (21, 18))
            if args.prototype_iq3_xxs
            else (
                (21, 23),
                (23, 21),
                (18, 21),
                (21, 18),
                (12, 23),
                (23, 12),
                (12, 21),
                (21, 12),
                (18, 23),
                (22, 21),
                (21, 22),
                (22, 18),
                (18, 22),
                (17, 18),
                (22, 17),
            )
        )
        assert tuple(types) in allowed, types
        raw = [tensors[name].data[:4352].copy() for name in names]
        sources = [
            (torch.from_numpy(data).cuda(), kind) for data, kind in zip(raw, types)
        ]
        projections = prepare_gguf_projections(sources, torch.float16, True, 8)
        layer_module = torch.nn.Module()
        layer_module.prefix = f"model.layers.{layer}.mlp.gate_up_proj"
        layer_module.gguf_tm_projections = torch.nn.ModuleList(projections)
        if (
            args.prototype_iq3_xxs
            or args.prototype_q4_k
            or args.prototype_iq3xxs_iq4
            or args.prototype_iq2_s
            or args.prototype_iq2_xs
            or args.prototype_iq2_xxs
        ):
            from vllm.model_executor.layers.quantization.gguf_iq3_records import (
                signed_index_records,
            )
            from vllm.model_executor.layers.quantization.gguf_iq3_xxs_records import (
                pack_iq3_xxs_records,
            )

            packers = {18: pack_iq3_xxs_records, 21: signed_index_records}
            if args.prototype_iq2_s or args.prototype_iq2_xs or args.prototype_iq2_xxs:
                from vllm.model_executor.layers.quantization.gguf_iq2_s_records import (
                    pack_iq2_s_records,
                )

                packers[22] = pack_iq2_s_records
            if args.prototype_iq2_xs or args.prototype_iq2_xxs:
                from vllm.model_executor.layers.quantization import gguf_iq2_xs_records

                packers[17] = gguf_iq2_xs_records.pack_iq2_xs_records
            if args.prototype_iq2_xxs:
                from vllm.model_executor.layers.quantization import gguf_iq2_xxs_records

                packers[16] = gguf_iq2_xxs_records.pack_iq2_xxs_records
            if args.prototype_q4_k or args.prototype_iq3xxs_iq4:
                from vllm.model_executor.layers.quantization.gguf_iq4_native import (
                    pack_iq4_xs_records,
                )
                from vllm.model_executor.layers.quantization.gguf_q4_k_records import (
                    pack_q4_k_records,
                )

                packers.update({12: pack_q4_k_records, 23: pack_iq4_xs_records})

            layer_module.gguf_native_gated_records = torch.nn.ParameterList(
                torch.nn.Parameter(
                    torch.from_numpy(packers[kind](data)).cuda(),
                    False,
                )
                for data, kind in zip(raw, types)
            )
            layer_module.gguf_native_gated_types = tuple(types)
            admission = {"model_admitted": False, "reason": "prototype_not_calibrated"}
        else:
            admission = prepare_native_gated_pair(
                layer_module, sources, projections, True
            )
            assert admission["reason"] is None, admission
        references = [
            torch.from_numpy(
                gguf.quants.dequantize(data, gguf.GGMLQuantizationType(kind))
            ).cuda()
            for data, kind in zip(raw, types)
        ]

        def native(rows, layer_module=layer_module):
            return apply_native_gated_pair(layer_module, rows)

        def canonical(rows, projections=projections):
            pair = apply_prepared_gguf_projections(rows, projections)
            result = rows.new_empty((rows.shape[0], 4352))
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
        fallbacks = []
        for m in (512, 8, 1, 5, 16, 20, 32, 8):
            sample = torch.randn(m, 5120, dtype=torch.float16, device="cuda")
            result = native(sample)
            if m != 8:
                torch.testing.assert_close(result, canonical(sample), rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replay_output = native(sample)
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(replay_output, result, rtol=0, atol=0)
            fallbacks.append(
                {
                    "m": m,
                    "route": "native" if m == 8 else "canonical",
                    "graph_bitwise_equal": True,
                }
            )
        flush = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        payload_bytes = sum(data.nbytes for data in raw)
        timings = []
        timing_calls = (
            ()
            if args.check_only
            else (
                ("canonical", canonical),
                ("native", native),
                ("native", native),
                ("canonical", canonical),
            )
        )
        for label, call in timing_calls:
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
                "admission": admission,
                "fallbacks": fallbacks,
                "abba": timings,
            }
        )
    report["clock_after"] = clocks()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
