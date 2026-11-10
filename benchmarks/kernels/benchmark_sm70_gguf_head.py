# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the real TP vocabulary shard with canonical GGUF and dense FP16."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.layers.quantization.gguf_transcode import transcode_affine
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    GGUFPreparedProjection,
)
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def graph_us(fn):
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    samples = []
    for _ in range(5):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 10)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lm-head-method", action="store_true")
    args = parser.parse_args()
    assert args.tp > 0 and 0 <= args.rank < args.tp
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    reader = GGUFReader(str(args.model))
    tensor = next(t for t in reader.tensors if t.name == "output.weight")
    assert tensor.data.ndim == 2
    assert tensor.data.shape[0] % args.tp == 0
    n = tensor.data.shape[0] // args.tp
    data = np.ascontiguousarray(tensor.data[args.rank * n : (args.rank + 1) * n])
    source_type = int(tensor.tensor_type)
    official = dequantize(data, source_type)
    canonical = transcode_affine(data, source_type)
    restored = canonical.dequantize()
    error = restored - official
    coefficient_error = {
        "max_abs": float(np.max(np.abs(error))),
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(official)),
    }
    del restored, canonical, error
    official_gpu = torch.from_numpy(official).cuda()
    dense = official_gpu.half()
    projection = GGUFPreparedProjection(
        torch.from_numpy(data).cuda(), source_type, torch.float16, True, 8
    )
    assert projection.kernel is not None, projection.admission()
    assert projection.fp16_cache is None, "Vocabulary head must remain packed"
    assert not hasattr(projection, "weight")
    if args.lm_head_method:
        from vllm.model_executor.layers.quantization.gguf import GGUFConfig
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        layer = ParallelLMHead.__new__(ParallelLMHead)
        torch.nn.Module.__init__(layer)
        method = GGUFConfig().get_quant_method(layer, "lm_head")
        method.create_weights(
            layer, official.shape[1], [n], official.shape[1], n, torch.float16
        )
        layer.qweight.materialize(data.shape, device="cuda", dtype=torch.uint8)
        layer.qweight.data.copy_(torch.from_numpy(data).cuda())
        layer.qweight_type.weight_type = source_type
        method.process_weights_after_loading(layer)
        assert method.canonical_lm_head, method.native_admission
        assert len(layer.gguf_tm_projections) == 1
        del projection
        projection = layer.gguf_tm_projections[0]
        project = lambda x: method.apply(layer, x)
    else:
        project = projection
    results = []
    for m in (1, 5, 8, 20):
        torch.manual_seed(20261004 + m)
        x = (torch.randn(m, official.shape[1], device="cuda") * 0.125).half()
        reference = x.float() @ official_gpu.T
        old = torch.nn.functional.linear(x, dense)
        new = project(x)
        old_error = old.float() - reference
        new_error = new.float() - reference
        old_us = graph_us(lambda x=x: torch.nn.functional.linear(x, dense))
        new_us = graph_us(lambda x=x: project(x))
        results.append(
            {
                "m": m,
                "dense_us": old_us,
                "canonical_us": new_us,
                "saved_us": old_us - new_us,
                "dense_relative_l2": (old_error.norm() / reference.norm()).item(),
                "canonical_relative_l2": (new_error.norm() / reference.norm()).item(),
                "canonical_max_abs": new_error.abs().max().item(),
                "top1_matches_official": bool(
                    (new.argmax(-1) == reference.argmax(-1)).all()
                ),
            }
        )
    report = {
        "source_type": source_type,
        "local_shape": list(official.shape),
        "source_bytes": data.nbytes,
        "dense_bytes": dense.numel() * dense.element_size(),
        "canonical_bytes": sum(
            p.numel() * p.element_size() for p in projection.parameters()
        ),
        "coefficient_error": coefficient_error,
        "admission": projection.admission(),
        "cases": results,
        "scope": (
            "single GPU TP vocabulary shard, synthetic activations, "
            "official FP32 dequantization oracle; not model throughput"
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
