# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare GGUF projection routes with real checkpoint bytes on one GPU.

Hold the shared GPU lock outside this process. These are operator timings,
not model throughput or AWQ/NVFP4 acceptance results. Cached FP16 excludes
dequantization and is reported only as a tensor-core lower bound.
"""

import argparse
import json
from functools import partial

import torch

from vllm.model_executor.layers.quantization.gguf import _fused_mul_mat_gguf
from vllm.model_executor.layers.quantization.gguf_native import (
    native_available,
    native_dense,
    pad_weight_tail,
)
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, quant_type_name


def elapsed_ms(call, iterations):
    for _ in range(3):
        call()
    torch.accelerator.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def graph_elapsed_ms(call, iterations):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    # Keep captured output and its allocation alive while measuring replay.
    assert output is not None
    return elapsed_ms(graph.replay, iterations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()
    if not native_available():
        raise RuntimeError("The packaged vllm._C_gguf extension is required")
    reader = GGUFReader(args.gguf)
    names = {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"}
    selected = [t for t in reader.tensors if t.name in names]
    selected += [next(t for t in reader.tensors if int(t.tensor_type) == 42)]
    results = []
    for tensor in selected:
        # One full expert preserves the format and the real K/N dimensions.
        array = tensor.data[0] if tensor.data.ndim == 3 else tensor.data
        weight_type = int(tensor.tensor_type)
        weight = pad_weight_tail(torch.from_numpy(array.copy()).cuda(), weight_type)
        rows, k = int(tensor.shape[1]), int(tensor.shape[0])
        dense = torch.ops._C_gguf.ggml_dequantize_upstream(
            weight, weight_type, rows, k, torch.float16
        )
        for m in (1, 4, 8, 16, 256):
            torch.manual_seed(20261003 + m)
            x = (torch.randn((m, k), device="cuda") * 0.125).half()
            native_call = partial(native_dense, x, weight, weight_type)
            legacy_call = partial(_fused_mul_mat_gguf, x, weight, weight_type, False)
            dense_call = partial(torch.nn.functional.linear, x, dense)
            output = native_call()
            if output is None:
                raise RuntimeError(f"Native route rejected {tensor.name}, M={m}")
            reference = x.float() @ dense.float().T
            error = (output.float() - reference).norm() / reference.norm()
            result = {
                "tensor": tensor.name,
                "format": quant_type_name(weight_type),
                "m": m,
                "n": rows,
                "k": k,
                "native_ms": elapsed_ms(native_call, args.iterations),
                "cached_fp16_ms": elapsed_ms(dense_call, args.iterations),
                "relative_l2": error.item(),
            }
            if weight_type != 42:
                result["legacy_ms"] = elapsed_ms(
                    legacy_call,
                    args.iterations,
                )
            if args.cuda_graph:
                result["native_graph_ms"] = graph_elapsed_ms(
                    native_call, args.iterations
                )
                if weight_type != 42:
                    result["legacy_graph_ms"] = graph_elapsed_ms(
                        legacy_call,
                        args.iterations,
                    )
            results.append(result)
    with open(args.output, "w") as destination:
        json.dump(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "results": results,
            },
            destination,
            indent=2,
        )


if __name__ == "__main__":
    main()
