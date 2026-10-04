# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact FP32 router keys, including changed-input CUDA graph replay."""

import pytest
import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    _sm70_qwen38_router_topk_kernel,
)
from vllm.platforms import current_platform


def launch(logits, outputs, partial):
    _sm70_qwen38_router_topk_kernel[(logits.shape[0],)](
        logits,
        *outputs,
        E=512,
        K=10,
        M=logits.shape[0],
        BLOCK_E=512,
        PACKED_HALF_KEY=False,
        SELECT_TOP16=partial,
        num_warps=8,
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability(70), reason="Requires SM70"
)
@pytest.mark.parametrize("rows", [5, 10, 20])
@pytest.mark.parametrize(
    "case", ["random", "ties", "nearby", "signed_zero", "nan", "inf", "negative_inf"]
)
def test_fp32_router_partial_selection(rows, case):
    logits = torch.empty(rows, 512, dtype=torch.float32, device="cuda")

    def outputs():
        return (
            torch.empty(rows, 10, dtype=torch.float32, device="cuda"),
            torch.empty(rows, 10, dtype=torch.int32, device="cuda"),
            torch.empty(rows, 10, dtype=torch.int32, device="cuda"),
        )

    full, partial, official = outputs(), outputs(), outputs()
    logits.normal_()
    launch(logits, full, False)
    launch(logits, partial, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(logits, full, False)
        launch(logits, partial, True)
    for seed in range(3):
        torch.manual_seed(seed)
        logits.normal_()
        if case == "ties":
            logits[:, :16] = 8.0
        elif case == "nearby":
            logits.fill_(1.0)
            # Distinct FP32 keys that collapse to a single FP16 value.
            logits[:, :32] += torch.arange(32, device="cuda") * 2**-23
        elif case == "signed_zero":
            logits.zero_()
            logits[:, 1::2] = -0.0
        elif case == "nan":
            logits[:, seed] = float("nan")
        elif case == "inf":
            logits[:, seed] = float("inf")
        elif case == "negative_inf":
            logits.fill_(-float("inf"))
        graph.replay()
        ops.topk_softmax(*official, logits, True)
        for actual, expected in zip(partial, full):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for actual, expected in zip(partial[1:], official[1:]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(partial[0], official[0], atol=1e-7, rtol=1e-7)
