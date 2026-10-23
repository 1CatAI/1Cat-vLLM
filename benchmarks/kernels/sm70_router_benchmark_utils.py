# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared real-weight fixtures and alternating graph timing for router benches."""

import statistics

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    _sm70_qwen38_router_topk_kernel,
    vllm_topk_softmax,
)
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def topk(logits, output):
    m = logits.size(0)
    if m > 16:
        vllm_topk_softmax(*output, logits, renormalize=True)
        return
    _sm70_qwen38_router_topk_kernel[(m,)](
        logits,
        *output,
        E=512,
        K=10,
        M=m,
        BLOCK_E=512,
        PACKED_HALF_KEY=m <= 16,
        SELECT_TOP16=m in (5, 10),
        num_warps=8,
    )


def pair(a, b, repeats):
    for _ in range(3):
        a()
        b()
    torch.accelerator.synchronize()
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    for graph, fn in zip(graphs, (a, b)):
        with torch.cuda.graph(graph):
            fn()
    samples = [[], []]
    for group in range(5):
        order = (0, 1, 1, 0) if group % 2 == 0 else (1, 0, 0, 1)
        for arm in order:
            for _ in range(3):
                graphs[arm].replay()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(repeats):
                graphs[arm].replay()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end) / repeats)
    medians = [statistics.median(x) for x in samples]
    return dict(
        control_ms=medians[0],
        candidate_ms=medians[1],
        saved_ms=medians[0] - medians[1],
        samples_ms=samples,
    )


def outputs(m):
    return [
        torch.empty((m, 10), device="cuda", dtype=d)
        for d in (torch.float32, torch.int32, torch.int32)
    ]


def load_weights(model):
    tensors = {}
    for path in sorted(model.parent.glob(model.name.split("-00001-of-")[0] + "*.gguf")):
        reader = GGUFReader(path)
        for tensor in reader.tensors:
            if tensor.name.endswith(".ffn_gate_inp.weight"):
                data = dequantize(tensor.data, int(tensor.tensor_type))
                tensors[int(tensor.name.split(".")[1])] = torch.from_numpy(
                    np.array(data, dtype=np.float16, copy=True).reshape(512, 2560)
                ).cuda()
    assert len(tensors) == 48, sorted(tensors)
    return [tensors[i] for i in range(48)]
