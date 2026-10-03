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

import numpy as np
import torch

from vllm import _custom_ops  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_lut_transcode import (
    LUT4_TYPES,
    Lut4GGUFProjection,
    transcode_lut4,
)
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
    quant_size,
    quant_type_name,
)

M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192)


def transcode_projection(data, weight_type):
    if weight_type in LUT4_TYPES:
        return transcode_lut4(data, weight_type)
    return transcode_affine(data, weight_type)


def prepare_projection(projection):
    codes = torch.from_numpy(projection.codes).cuda()
    scales = torch.from_numpy(projection.scales).cuda()
    if isinstance(projection, Lut4GGUFProjection):
        return torch.ops._C.gguf_lut4_sm70_prepare(
            codes, scales, projection.lut_id, projection.group_size
        )
    return torch.ops._C.gguf_affine_sm70_prepare(
        codes,
        scales,
        torch.from_numpy(projection.mins).cuda(),
        projection.bits,
        projection.group_size,
    )


def partition_source(source, weight_type, size, rank, axis):
    canonical = transcode_projection(source, weight_type)
    reference = dequantize(source, weight_type)
    if size == 1:
        if rank != 0:
            raise ValueError("TP rank outside TP size")
        return canonical, reference, source, None
    canonical = canonical.tp_slice(rank, size, axis=axis)
    span = canonical.codes.shape[axis]
    selection = [slice(None), slice(None)]
    selection[axis] = slice(rank * span, (rank + 1) * span)
    reference = np.ascontiguousarray(reference[tuple(selection)])
    if axis == 0:
        return (
            canonical,
            reference,
            np.ascontiguousarray(source[tuple(selection)]),
            None,
        )
    block, byte_size = quant_size(weight_type)
    if span % block:
        return canonical, reference, None, "tp_slice_cuts_original_gguf_block"
    raw = source[
        :, rank * span // block * byte_size : (rank + 1) * span // block * byte_size
    ]
    return canonical, reference, np.ascontiguousarray(raw), None


def canonical_dense_call(projection, out, x, weight, stats, k_ld, q_ld):
    is_lut = isinstance(projection, Lut4GGUFProjection)
    op = (
        torch.ops._C.gguf_lut4_gemm_sm70_out
        if is_lut
        else torch.ops._C.gguf_affine_gemm_sm70_out
    )
    decoder = projection.lut_id if is_lut else projection.bits
    return partial(
        op, out, x, weight, stats, decoder, k_ld, q_ld, projection.group_size
    )


def canonical_grouped_call(projection, out, x, offsets, wp, sp, experts):
    is_lut = isinstance(projection, Lut4GGUFProjection)
    op = (
        torch.ops._C.gguf_lut4_grouped_gemm_sm70_out
        if is_lut
        else torch.ops._C.gguf_affine_grouped_gemm_sm70_out
    )
    decoder = projection.lut_id if is_lut else projection.bits
    return partial(op, out, x, offsets, wp, sp, decoder, experts, projection.group_size)


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
        # Several device invocations per replay keep small-M timings from
        # including gaps between Python graph.replay calls under CPU load.
        # Large outputs retain one invocation to bound graph-pool memory.
        inner = (
            8
            if isinstance(result, torch.Tensor) and result.numel() <= 10_000_000
            else 1
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                result = call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(inner):
                result = call()
        measure = graph.replay
        # Exclude the driver's first replay/upload from steady-state timing.
        # Warming the eager call does not warm this newly instantiated graph.
        for _ in range(3):
            measure()
        torch.accelerator.synchronize()
    else:
        measure = call
        inner = 1
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        measure()
    end.record()
    end.synchronize()
    # Keep output alive through replay: upstream ops allocate their result.
    del result
    return start.elapsed_time(end) * 1000 / (iterations * inner)


def prepare_awq_comparator(canonical):
    n, k = canonical.codes.shape
    if k % 128:
        return None
    codes = torch.from_numpy(canonical.codes.copy()).cuda().to(torch.int64)
    if canonical.bits == 8:
        codes = codes >> 4
    codes = codes & 15
    shifts = torch.arange(8, device="cuda") * 4
    packed = (codes.T.reshape(k, n // 8, 8) << shifts).sum(-1).int()
    scale = torch.full((k // 128, n), 0.00390625, dtype=torch.float16, device="cuda")
    zero = torch.full(
        (k // 128, n // 8), -2004318072, dtype=torch.int32, device="cuda"
    )  # 0x88888888
    return torch.ops._C.awq_sm70_prepare(packed, scale, zero, 128, False)


def awq_comparator(canonical):
    prepared = prepare_awq_comparator(canonical)
    if prepared is None:
        return None
    weight, stats, meta = prepared
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
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument(
        "--tp-axis",
        type=int,
        choices=(0, 1),
        default=0,
        help="Logical [N,K] partition axis",
    )
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
        "graph_inner_invocations": "8 for outputs <= 10000000 elements; otherwise 1",
        "tp": {"size": args.tp_size, "rank": args.tp_rank, "axis": args.tp_axis},
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
        array = tensor.data[args.expert] if tensor.data.ndim == 3 else tensor.data
        source = np.array(array, copy=True)
        canonical, reference, raw_source, native_reason = partition_source(
            source, weight_type, args.tp_size, args.tp_rank, args.tp_axis
        )
        rounding = reconstruction_error(canonical, reference)
        n, k = canonical.codes.shape
        weight, stats, meta = prepare_projection(canonical)
        k_ld, q_ld = meta.tolist()
        packed = (
            pad_weight_tail(torch.from_numpy(raw_source).cuda(), weight_type)
            if raw_source is not None
            else None
        )
        dense = torch.from_numpy(reference).half().cuda()
        awq = awq_comparator(canonical)
        for m in args.m:
            torch.manual_seed(20261003 + m)
            x = (torch.randn((m, k), device="cuda") * 0.125).half()
            out = torch.empty((m, n), dtype=torch.float16, device="cuda")

            call = canonical_dense_call(canonical, out, x, weight, stats, k_ld, q_ld)

            def tm(
                out=out,
                call=call,
            ):
                call()
                return out

            tm()
            expected = x.float() @ dense.float().T
            error = (out.float() - expected).norm() / expected.norm()
            if not torch.isfinite(out).all() or error.item() > 0.003:
                raise AssertionError(
                    f"{name}, M={m}: GGUF canonical error {error.item()}"
                )
            capabilities = (
                torch.ops._C_gguf.ggml_dense_upstream_capabilities(
                    packed, x, weight_type, n
                )
                if packed is not None
                else 0
            )
            routes = {
                "turbomind_gguf": tm,
                "cached_fp16_lower_bound": partial(
                    torch.nn.functional.linear, x, dense
                ),
            }
            if packed is not None:
                routes["dequant_cublas"] = partial(
                    torch.ops._C_gguf.ggml_dense_blas, packed, x, weight_type, n
                )
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
                "native_unavailable_reason": native_reason,
                "tensor": name,
                "type": quant_type_name(weight_type),
                "expert": args.expert if tensor.data.ndim == 3 else None,
                "m": m,
                "n": n,
                "k": k,
                "canonical_bits": canonical.bits,
                "canonical_group": canonical.group_size,
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
