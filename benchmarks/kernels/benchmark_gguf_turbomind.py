# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure canonical GGUF operators against AWQ and explicit llama operators.

Run with the shared GPU flock held. The checkpoint supplies actual projection
bytes and dimensions; the AWQ comparison measures the same shape, not quality
equivalence to a separately quantized checkpoint. No model forward is involved.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import gguf
import numpy as np
import torch

from vllm import _custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_native import (
    native_available,
    pad_weight_tail,
)
from vllm.model_executor.layers.quantization.gguf_transcode import (
    AFFINE_GROUP32_TYPES,
    reconstruction_error,
    transcode_affine_group32,
)
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, quant_type_name

M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192)


def elapsed(call, iterations, capture=False):
    # Let short decode operators reach steady GPU clocks after CPU transcoding.
    # Synchronize batches rather than time Python launch latency as warmup.
    warm_start, warm_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    warm_start.record()
    while True:
        for _ in range(10):
            result = call()
        warm_end.record()
        warm_end.synchronize()
        if warm_start.elapsed_time(warm_end) >= 100:
            break
    for _ in range(3):
        result = call()
    if capture:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                result = call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = call()
        measure = graph.replay
        # Exclude the driver's first replay/upload from steady-state timing.
        # Warming the eager call does not warm this newly instantiated graph.
        for _ in range(3):
            measure()
        torch.accelerator.synchronize()
    else:
        measure = call
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        measure()
    end.record()
    end.synchronize()
    # Keep output alive through replay: upstream ops allocate their result.
    del result
    return start.elapsed_time(end) * 1000 / iterations


def awq_comparator(canonical):
    n, k = canonical.codes.shape
    if k % 128:
        return None
    codes = torch.from_numpy(canonical.codes.copy()).cuda().to(torch.int64)
    if canonical.bits == 8:
        codes = codes >> 4
    shifts = torch.arange(8, device="cuda") * 4
    packed = (codes.T.reshape(k, n // 8, 8) << shifts).sum(-1).int()
    scale = torch.full((k // 128, n), 0.00390625, dtype=torch.float16, device="cuda")
    zero = torch.full(
        (k // 128, n // 8), -2004318072, dtype=torch.int32, device="cuda"
    )  # 0x88888888
    weight, stats, meta = torch.ops._C.awq_sm70_prepare(packed, scale, zero, 128, False)
    k_ld, q_ld = meta.tolist()

    def run(out, x):
        torch.ops._C.awq_gemm_sm70_out(out, x, weight, stats, 128, k_ld, q_ld, False)
        return out

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf")
    parser.add_argument("--tensor", action="append", required=True)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--m", type=int, nargs="+", default=M_VALUES)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not native_available():
        raise RuntimeError("Packaged GGUF reference extension is required")
    reader = GGUFReader(args.gguf)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    output = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "iterations": args.iterations,
        "warmup_ms_per_route": 100,
        "graph": args.cuda_graph,
        "checkpoint": Path(args.gguf).name,
        "tensor_manifest": [
            {
                "name": t.name,
                "shape": t.shape.tolist(),
                "type": quant_type_name(int(t.tensor_type)),
            }
            for t in reader.tensors
        ],
        "results": [],
    }
    destination = Path(args.output)
    for name in args.tensor:
        tensor = tensors[name]
        weight_type = int(tensor.tensor_type)
        if weight_type not in AFFINE_GROUP32_TYPES:
            raise ValueError(f"{name}: canonical affine codec unavailable")
        array = tensor.data[args.expert] if tensor.data.ndim == 3 else tensor.data
        source = np.array(array, copy=True)
        canonical = transcode_affine_group32(source, weight_type)
        reference = gguf.quants.dequantize(
            source, gguf.GGMLQuantizationType(weight_type)
        )
        rounding = reconstruction_error(canonical, reference)
        n, k = canonical.codes.shape
        weight, stats, meta = torch.ops._C.gguf_affine_sm70_prepare(
            torch.from_numpy(canonical.codes).cuda(),
            torch.from_numpy(canonical.scales).cuda(),
            torch.from_numpy(canonical.mins).cuda(),
            canonical.bits,
        )
        k_ld, q_ld = meta.tolist()
        packed = pad_weight_tail(torch.from_numpy(source).cuda(), weight_type)
        dense = torch.from_numpy(reference).half().cuda()
        awq = awq_comparator(canonical)
        for m in args.m:
            torch.manual_seed(20261003 + m)
            x = (torch.randn((m, k), device="cuda") * 0.125).half()
            out = torch.empty((m, n), dtype=torch.float16, device="cuda")

            def tm(
                out=out,
                x=x,
                weight=weight,
                stats=stats,
                bits=canonical.bits,
                k_ld=k_ld,
                q_ld=q_ld,
            ):
                torch.ops._C.gguf_affine_gemm_sm70_out(
                    out, x, weight, stats, bits, k_ld, q_ld
                )
                return out

            tm()
            expected = x.float() @ dense.float().T
            error = (out.float() - expected).norm() / expected.norm()
            if not torch.isfinite(out).all() or error.item() > 0.003:
                raise AssertionError(f"{name}, M={m}: GGUF affine error {error.item()}")
            capabilities = torch.ops._C_gguf.ggml_dense_upstream_capabilities(
                packed, x, weight_type, n
            )
            routes = {
                "turbomind_gguf": tm,
                "dequant_cublas": partial(
                    torch.ops._C_gguf.ggml_dense_blas, packed, x, weight_type, n
                ),
                "cached_fp16_lower_bound": partial(
                    torch.nn.functional.linear, x, dense
                ),
            }
            if awq is not None:
                routes["turbomind_awq_group128"] = partial(awq, out, x)
            for bit, route in ((4, "mmvq"), (8, "mmq")):
                if capabilities & bit:
                    routes[f"llama_{route}"] = partial(
                        getattr(torch.ops._C_gguf, f"ggml_dense_{route}"),
                        packed,
                        x,
                        weight_type,
                        n,
                    )
            row = {
                "tensor": name,
                "type": quant_type_name(weight_type),
                "expert": args.expert if tensor.data.ndim == 3 else None,
                "m": m,
                "n": n,
                "k": k,
                "coefficient_rounding": rounding,
                "output_relative_l2": error.item(),
                "routes": {},
            }
            for route, call in routes.items():
                row["routes"][route] = {
                    "eager_us": elapsed(call, args.iterations),
                }
                if args.cuda_graph:
                    row["routes"][route]["graph_us"] = elapsed(
                        call, args.iterations, capture=True
                    )
            output["results"].append(row)
            destination.write_text(json.dumps(output, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
