# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real IQ3_S TP4 expert dots, including quantization/routing accounting."""

import argparse
import hashlib
import json
import statistics
import subprocess
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import vllm._C as core
from safetensors import safe_open

import vllm
from vllm import _sm70_ops as sm70
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.model_executor.layers.quantization.sm70_turbomind import unpack_mxfp4_weight
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def graph_time(operation, iterations):
    for _ in range(3):
        operation()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    for _ in range(10):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def nvfp4_bank(model, layer, rank):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    weights, scales = [], []
    with ExitStack() as stack:
        handles = {}

        def load(expert, suffix):
            key = f"{prefix}.{expert}.{suffix}"
            shard = index[key]
            if shard not in handles:
                handles[shard] = stack.enter_context(
                    safe_open(model / shard, framework="pt", device="cpu")
                )
            return handles[shard].get_tensor(key)

        for expert in range(512):
            packed, coefficients = [], []
            for name in ("gate_proj", "up_proj"):
                rows = slice(rank * 160, (rank + 1) * 160)
                packed.append(load(expert, name + ".weight")[rows])
                scale = load(expert, name + ".weight_scale")[rows].float()
                scale *= load(expert, name + ".weight_scale_2").float().reshape(())
                coefficients.append(scale)
            prepared = sm70.nvfp4_sm70_prepare(
                unpack_mxfp4_weight(torch.cat(packed).cuda()),
                torch.cat(coefficients).half().T.contiguous().cuda(),
                16,
                interleave_gated_silu=True,
            )
            weights.append(prepared[0])
            scales.append(prepared[1])
    return torch.stack(weights), torch.stack(scales)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--m", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--nvfp4", type=Path)
    args = parser.parse_args()
    assert "site-packages" in vllm.__file__, vllm.__file__
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    reader = GGUFReader(args.model)
    tensors = [
        next(
            t
            for t in reader.tensors
            if t.name == f"blk.{args.layer}.ffn_{name}_exps.weight"
        )
        for name in ("gate", "up")
    ]
    assert all(int(t.tensor_type) == 21 for t in tensors)
    banks, sources = [], []
    for tensor in tensors:
        raw = [
            RawGGUFProjection.from_rows(rows, 21).tp_slice(args.rank, 4, axis=0)
            for rows in tensor.data
        ]
        banks.append(torch.from_numpy(np.stack([r.data for r in raw])).cuda())
        sources.append([r.data[:, : r.payload_bytes_per_row] for r in raw])
    experts, n, stride = banks[0].shape
    k = 2560
    nv = nvfp4_bank(args.nvfp4, args.layer, args.rank) if args.nvfp4 else None
    result = {
        "version": vllm.__version__,
        "core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "shape": [experts, n, k],
        "layer": args.layer,
        "cases": [],
    }
    for m in args.m:
        torch.manual_seed(20261005 + m)
        x = (torch.randn((m, k), device="cuda") * 0.125).half()
        ids = torch.randn((m, experts), device="cuda").topk(10, dim=1).indices
        sorted_ids, order = ids.flatten().sort(stable=True)
        offsets = torch.searchsorted(
            sorted_ids, torch.arange(experts + 1, device="cuda")
        ).int()
        routed = x[order // 10].contiguous()
        old_gate, old_up = [
            torch.empty((m * 10, n), device="cuda", dtype=torch.float16)
            for _ in range(2)
        ]
        q8 = torch.empty((m, k // 32, 36), device="cuda", dtype=torch.uint8)
        debug = torch.empty((m, 10, 2, n), device="cuda", dtype=torch.float16)
        fused = torch.empty((m, 10, n), device="cuda", dtype=torch.float16)

        def quantize(q8=q8, x=x):
            torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)

        def native(
            old_gate=old_gate,
            old_up=old_up,
            routed=routed,
            banks=banks,
            offsets=offsets,
            sorted_ids=sorted_ids,
        ):
            torch.ops._C.gguf_lattice_raw_grouped_gate_up_sm70_out(
                old_gate, old_up, routed, *banks, offsets, sorted_ids, 21, 10
            )

        def dp4a(fused=fused, q8=q8, ids=ids, banks=banks):
            torch.ops._C.gguf_dp4a_gate_up_sm70_out(fused, q8, ids, *banks, 21, True)

        def combined(quantize=quantize, dp4a=dp4a):
            quantize()
            dp4a()

        quantize()
        native()
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(debug, q8, ids, *banks, 21, False)
        dp4a()
        integer_x = q8[:, :, 4:].view(torch.int8).float()
        d = q8[:, :, :2].contiguous().view(torch.float16).float()
        decoded_x = (integer_x * d).reshape(m, k)
        error_rows = []
        for projection in range(2):
            expected = torch.empty((m, 10, n), dtype=torch.float32, device="cuda")
            for expert in ids.unique().tolist():
                weight = torch.from_numpy(
                    dequantize(sources[projection][expert], 21)
                ).cuda()
                locations = (ids == expert).nonzero()
                expected[locations[:, 0], locations[:, 1]] = (
                    decoded_x[locations[:, 0]] @ weight.T
                )
            actual = debug[:, :, projection]
            diff = actual.float() - expected
            oracle_relative = (diff.norm() / expected.norm()).item()
            torch.testing.assert_close(actual.float(), expected, rtol=0.003, atol=0.003)
            reference_fp16 = torch.empty_like(old_gate)
            reference_fp16[order] = old_gate if projection == 0 else old_up
            fp16 = reference_fp16.reshape(m, 10, n)
            error_rows.append(
                {
                    "official_q8_relative_l2": oracle_relative,
                    "fp16_relative_l2": (
                        (actual.float() - fp16.float()).norm() / fp16.float().norm()
                    ).item(),
                    "max_abs_vs_fp16": (actual - fp16).abs().max().item(),
                }
            )
        expected_fused = torch.nn.functional.silu(debug[:, :, 0]) * debug[:, :, 1]
        torch.testing.assert_close(fused, expected_fused, rtol=0, atol=0)
        operations = {
            "native_gate_up": native,
            "activation_quantize": quantize,
            "dp4a_fused_gate_up": dp4a,
            "dp4a_quantize_and_fused": combined,
        }
        if nv is not None and m <= 16:
            nv_out = torch.empty((m * 10, n), device="cuda", dtype=torch.float16)
            nv_ids = ids.int().contiguous()
            nv_rows = torch.empty((160, 8), dtype=torch.int32, device="cuda")
            nv_experts, nv_sizes = [
                torch.empty(160, dtype=torch.int32, device="cuda") for _ in range(2)
            ]
            nv_total = torch.empty(1, dtype=torch.int32, device="cuda")

            def nvfp4(
                nv_out=nv_out,
                x=x,
                nv=nv,
                nv_ids=nv_ids,
                nv_rows=nv_rows,
                nv_experts=nv_experts,
                nv_sizes=nv_sizes,
                nv_total=nv_total,
            ):
                sm70.nvfp4_grouped_w13_sm70_out(
                    nv_out,
                    x,
                    *nv,
                    nv_ids,
                    nv_rows,
                    nv_experts,
                    nv_sizes,
                    nv_total,
                    4,
                    True,
                )

            operations["nvfp4_plan_and_w13"] = nvfp4
        samples = {name: [] for name in operations}
        clocks = []
        for epoch in range(8):
            for name in list(operations)[:: 1 if epoch % 2 == 0 else -1]:
                samples[name].append(graph_time(operations[name], args.iterations))
            clocks.append(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=clocks.sm,clocks.mem",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).strip()
            )
        median = {name: statistics.median(values) for name, values in samples.items()}
        route_bytes = m * 10 * n * stride * 2
        unique_bytes = ids.unique().numel() * n * stride * 2
        row = {
            "m": m,
            "routes": m * 10,
            "active_experts": ids.unique().numel(),
            "route_weight_bytes": route_bytes,
            "unique_weight_bytes": unique_bytes,
            "epoch_us": samples,
            "median_us": median,
            "dp4a_route_source_gbps": route_bytes / median["dp4a_fused_gate_up"] / 1000,
            "dp4a_unique_source_gbps": unique_bytes
            / median["dp4a_fused_gate_up"]
            / 1000,
            "kernel_count": {
                "dp4a_fused_gate_up": 1,
                "dp4a_quantize_and_fused": 2,
                "native_gate_up": 1,
                "nvfp4_plan_and_w13": 2,
            },
            "errors": error_rows,
            "clocks": clocks,
        }
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
