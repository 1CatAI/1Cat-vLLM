# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure sorted expert projections with real GGUF weights.

The workload is one projection with one assigned expert per row. Sorting and
the complete MoE FFN are outside the measurement. Empty experts are retained.
The AWQ comparison has the same dimensions and assignment, not the same
quantized checkpoint. Run while holding the shared GPU lock.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from benchmark_gguf_turbomind import M_VALUES, elapsed, prepare_awq_comparator

from vllm import _custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_native import (
    native_available,
    pad_weight_tail,
)
from vllm.model_executor.layers.quantization.gguf_transcode import (
    reconstruction_error,
    transcode_affine,
)
from vllm.transformers_utils.gguf_tensor_reader import (
    GGUFReader,
    dequantize,
    quant_type_name,
)


def stack_prepared(prepared):
    weights = torch.stack([p[0] for p in prepared])
    stats = torch.stack([p[1] for p in prepared])
    k_ld, q_ld = prepared[0][2].tolist()
    ptrs = torch.ops._C.awq_moe_build_strided_ptrs(
        weights, stats, k_ld, q_ld, len(prepared)
    )
    return weights, stats, ptrs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf")
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--m", type=int, nargs="+", default=M_VALUES)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not native_available():
        raise RuntimeError("Packaged GGUF reference extension is required")
    reader = GGUFReader(args.gguf)
    tensor = next(t for t in reader.tensors if t.name == args.tensor)
    if tensor.data.ndim != 3 or not 1 <= args.experts <= tensor.data.shape[0]:
        raise ValueError("Expected a stacked expert tensor and valid expert count")
    source = np.array(tensor.data[: args.experts], copy=True)
    weight_type = int(tensor.tensor_type)
    canonical = [transcode_affine(expert, weight_type) for expert in source]
    reference = dequantize(source, weight_type)
    rounding = [reconstruction_error(p, reference[e]) for e, p in enumerate(canonical)]
    prepared = [
        torch.ops._C.gguf_affine_sm70_prepare(
            torch.from_numpy(p.codes).cuda(),
            torch.from_numpy(p.scales).cuda(),
            torch.from_numpy(p.mins).cuda(),
            p.bits,
            p.group_size,
        )
        for p in canonical
    ]
    tm_weights, tm_stats, (wp, sp) = stack_prepared(prepared)
    awq_prepared = [prepare_awq_comparator(p) for p in canonical]
    awq = None
    if all(p is not None for p in awq_prepared):
        awq_weights, awq_stats, awq = stack_prepared(awq_prepared)
    packed = pad_weight_tail(torch.from_numpy(source).cuda(), weight_type)
    dense = torch.from_numpy(reference).half().cuda()
    e, n, k = dense.shape
    projection = canonical[0]
    output = {
        "checkpoint": Path(args.gguf).name,
        "tensor": tensor.name,
        "type": quant_type_name(weight_type),
        "n": n,
        "k": k,
        "experts": e,
        "canonical_bits": projection.bits,
        "canonical_group": projection.group_size,
        "coefficient_rounding": rounding,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "iterations": args.iterations,
        "warmup_ms_per_route": 100,
        "graph": args.cuda_graph,
        "results": [],
    }
    for m in args.m:
        boundaries = [i * m // e for i in range(e + 1)]
        offsets = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
        ids = torch.repeat_interleave(
            torch.arange(e, dtype=torch.int32, device="cuda"),
            torch.diff(offsets).long(),
            output_size=m,
        ).reshape(m, 1)
        torch.manual_seed(20261003 + m)
        x = (torch.randn((m, k), device="cuda") * 0.125).half()
        out = torch.empty((m, n), dtype=torch.float16, device="cuda")

        def tm(out=out, x=x, offsets=offsets):
            torch.ops._C.gguf_affine_grouped_gemm_sm70_out(
                out, x, offsets, wp, sp, projection.bits, e, projection.group_size
            )
            return out

        def awq_call(out=out, x=x, offsets=offsets):
            torch.ops._C.awq_moe_gemm_sm70_out(
                out, x, offsets, *awq, e, k, n, 128, False
            )
            return out

        def cached_dense(out=out, x=x, boundaries=boundaries):
            for expert, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                if start != end:
                    torch.mm(x[start:end], dense[expert].T, out=out[start:end])
            return out

        tm()
        expected = torch.empty((m, n), device="cuda")
        for expert, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            expected[start:end] = x[start:end].float() @ dense[expert].float().T
        error = (out.float() - expected).norm() / expected.norm()
        if not torch.isfinite(out).all() or error.item() > 0.003:
            raise AssertionError(f"M={m}: grouped affine error {error.item()}")
        caps = torch.ops._C_gguf.ggml_dense_upstream_capabilities(
            packed[0], x, weight_type, n
        )
        routes = {
            "turbomind_gguf_grouped": (tm, True),
            "cached_fp16_per_expert_lower_bound": (cached_dense, True),
            "dequant_cublas_grouped": (
                partial(
                    torch.ops._C_gguf.ggml_moe_grouped_dense,
                    x,
                    packed,
                    ids,
                    weight_type,
                    n,
                    1,
                    m,
                ),
                False,  # The reference sorts expert IDs on the CPU.
            ),
        }
        if awq is not None:
            routes["turbomind_awq_group128_grouped"] = (awq_call, True)
        for bit, route in ((4, "mmvq"), (8, "mmq")):
            if caps & bit:
                routes[f"llama_moe_{route}"] = (
                    partial(
                        getattr(torch.ops._C_gguf, f"ggml_moe_{route}"),
                        x,
                        packed,
                        ids,
                        weight_type,
                        n,
                        1,
                        m,
                    ),
                    True,
                )
        row = {
            "m": m,
            "offsets": boundaries,
            "output_relative_l2": error.item(),
            "routes": {},
        }
        for name, (call, graph_safe) in routes.items():
            times = {"eager_us": elapsed(call, args.iterations)}
            if args.cuda_graph and graph_safe:
                times["graph_us"] = elapsed(call, args.iterations, capture=True)
            elif args.cuda_graph:
                times["graph_unavailable_reason"] = "reference_requires_host_sort"
            row["routes"][name] = times
        output["results"].append(row)
        Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    # Retain the owner tensors until all pointer-based measurements complete.
    del tm_weights, tm_stats, prepared, awq_prepared


if __name__ == "__main__":
    main()
