# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the packaged single-projection reader with canonical TP4 down.

Hold the shared GPU leases before running. This operator benchmark does not
admit any model shape. L2 eviction is outside the timed graph interval.
"""

import argparse
import json
from functools import partial
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
    parser.add_argument("--nvfp4-model", type=Path)
    parser.add_argument("--n64", action="store_true")
    parser.add_argument("--canonical-qpn", action="store_true")
    args = parser.parse_args()
    if args.canonical_qpn:
        args.n64 = True
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
        if args.canonical_qpn and source_type not in (10, 12):
            continue
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
        # Recover every operand in one actual K256 block with unit inputs.
        # This also covers empty/unequal warp partitions, independently of
        # random GEMM error norms and the existing pair's K1024 constraint.
        block_size = raw.shape[1] // (k // 256)
        block_n = 64 if args.n64 else 32
        block_records = torch.from_numpy(
            _SOURCE_PACKERS[source_type](
                np.ascontiguousarray(raw[:block_n, :block_size])
            )
        ).cuda()
        basis = torch.zeros(8, 256, dtype=torch.float16, device="cuda")
        block_out = torch.empty(8, block_n, dtype=torch.float16, device="cuda")
        block_partials = torch.empty(
            block_n // 64, 2, 512, dtype=torch.float32, device="cuda"
        )
        block_counters = torch.zeros(block_n // 64, dtype=torch.int32, device="cuda")
        block_projections = None
        block_reference = None
        if args.canonical_qpn:
            block_source = torch.from_numpy(
                np.ascontiguousarray(raw[:block_n, :block_size])
            ).cuda()
            block_projections = prepare_gguf_projections(
                [(block_source, source_type)], torch.float16, True, 8
            )
            block_reference = torch.empty(
                256, block_n, dtype=torch.float16, device="cuda"
            )
            bp = block_projections[0]
            torch.ops._C.gguf_affine_dequantize_sm70_out(
                block_reference,
                bp.codes,
                bp.stats,
                bp.kernel.bits,
                bp.kernel.config.group_size,
            )
        for begin in range(0, 256, 8):
            basis.zero_()
            basis[:, begin : begin + 8].copy_(
                torch.eye(8, dtype=torch.float16, device="cuda")
            )
            if args.canonical_qpn:
                bp = block_projections[0]
                torch.ops._C.gguf_canonical_linear_n64_sm70_out(
                    block_out,
                    basis,
                    bp.codes,
                    bp.stats,
                    block_partials,
                    block_counters,
                    bp.kernel.bits,
                    bp.kernel.config.group_size,
                )
            elif args.n64:
                torch.ops._C.gguf_native_linear_n64_sm70_out(
                    block_out,
                    basis,
                    block_records,
                    block_partials,
                    block_counters,
                    source_type,
                )
            else:
                torch.ops._C.gguf_native_linear_sm70_out(
                    block_out, basis, block_records, source_type
                )
            torch.testing.assert_close(
                block_out,
                block_reference[begin : begin + 8]
                if args.canonical_qpn
                else official[:block_n, begin : begin + 8].T.half(),
                rtol=0,
                atol=0,
            )
        out = torch.empty(8, n, dtype=torch.float16, device="cuda")
        partials = torch.empty(n // 64, 2, 512, dtype=torch.float32, device="cuda")
        counters = torch.zeros(n // 64, dtype=torch.int32, device="cuda")

        def native(
            x,
            out=out,
            records=records,
            source_type=source_type,
            partials=partials,
            counters=counters,
            projections=projections,
        ):
            if args.canonical_qpn:
                p = projections[0]
                torch.ops._C.gguf_canonical_linear_n64_sm70_out(
                    out,
                    x,
                    p.codes,
                    p.stats,
                    partials,
                    counters,
                    p.kernel.bits,
                    p.kernel.config.group_size,
                )
            elif args.n64:
                torch.ops._C.gguf_native_linear_n64_sm70_out(
                    out, x, records, partials, counters, source_type
                )
            else:
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
        if args.n64:
            assert torch.count_nonzero(counters).item() == 0
            assert torch.count_nonzero(block_counters).item() == 0
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
            read_bytes = (
                raw.nbytes
                if label == "native" and not args.canonical_qpn
                else canonical_bytes
            )
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
        nvfp4_reference = None
        if args.nvfp4_model is not None:
            from benchmark_sm70_nvfp4_qpn2 import _load_projection_shards

            from vllm import _sm70_ops as ops

            _, nv_down = _load_projection_shards(args.nvfp4_model, layer, 0, 4)
            nv_codes, nv_scales = ops.nvfp4_qpn2_prepare_sm70(
                nv_down.packed.cuda(), nv_down.scales.cuda()
            )
            nv_out = torch.empty_like(out)
            nv_bytes = sum(t.numel() * t.element_size() for t in (nv_codes, nv_scales))
            nv_timings = []
            nv_call = partial(
                ops.nvfp4_qpn2_gemm_sm70_out,
                nv_out,
                rows,
                nv_codes,
                nv_scales,
                nv_down.inverse_global_scale,
                16,
                2,
            )
            for _ in range(2):
                before = clocks()
                us = cold_graph(
                    nv_call,
                    flush,
                )
                nv_timings.append(
                    {
                        "median_us": us,
                        "effective_weight_gbps": nv_bytes / us / 1000,
                        "clock_before": before,
                        "clock_after": clocks(),
                    }
                )
            nvfp4_reference = {
                "model": args.nvfp4_model.name,
                "layer": layer,
                "m": 8,
                "n": n,
                "k": k,
                "weight_stream_bytes": nv_bytes,
                "operator": "nvfp4_qpn2_gemm_sm70_out",
                "split_k": 16,
                "accumulator_chains": 2,
                "timings": nv_timings,
                "scope": "same-shape weights; no cross-model quality comparison",
            }
        report["cases"].append(
            {
                "layer": layer,
                "source_type": source_type,
                "tensor": tensor.name,
                "m": 8,
                "n": n,
                "k": k,
                "source_bytes": raw.nbytes,
                "n64": args.n64,
                "canonical_qpn": args.canonical_qpn,
                "canonical_bytes": canonical_bytes,
                "checks": checks,
                "graph_bitwise_equal": True,
                "operand_basis_equal": True,
                "operand_basis_reference": "canonical"
                if args.canonical_qpn
                else "official",
                "abba": timings,
                "nvfp4_reference": nvfp4_reference,
            }
        )
        save()
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
