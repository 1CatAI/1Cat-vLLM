# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator

import pytest
import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
    _attach_sm70_nvfp4_prefetch,
)
from vllm.model_executor.layers.quantization.utils import (  # noqa: F401
    sm70_nvfp4_native,
)


@pytest.mark.parametrize("functionalized", [False, True])
@pytest.mark.parametrize("view", [False, True])
@pytest.mark.parametrize("unsupported", [None, "shape", "dtype", "split", "ungated"])
def test_only_native_gate_weight_is_attached(functionalized, view, unsupported):
    graph = torch.fx.Graph()
    x, residual, weight, codes, scales, out = [
        graph.placeholder(name)
        for name in ["x", "residual", "weight", "codes", "scales", "out"]
    ]
    code_bytes = 272 * 320 * 256 if unsupported != "shape" else 1024
    codes.meta["val"] = torch.empty(
        code_bytes,
        dtype=torch.uint8 if unsupported != "dtype" else torch.float16,
        device="meta",
    )
    norm = graph.call_function(
        torch.ops.vllm.sm70_tp4_all_reduce_gemma_rms_norm.default,
        (x, residual, weight, 1e-6, "tp"),
    )
    normalized = graph.call_function(operator.getitem, (norm, 0))
    value = torch.empty(8, 5120, dtype=torch.float16, device="meta")
    normalized.meta["val"] = value
    if view:
        normalized = graph.call_function(
            torch.ops.aten.view.default, (normalized, [8, 5120])
        )
        normalized.meta["val"] = value
    arguments = dict(
        out=out,
        x=normalized,
        codes=codes,
        scales=scales,
        global_scale=1.0,
        split_k=8 if unsupported != "split" else 16,
        accumulator_chains=2,
        gated_silu=unsupported != "ungated",
    )
    native = torch.ops.vllm.sm70_nvfp4_native_dispatch.default
    if functionalized:
        result = graph.call_function(auto_functionalized, (native,), arguments)
    else:
        result = graph.call_function(native, tuple(arguments.values()))
    graph.output(result)
    expected = int(unsupported is None)
    assert _attach_sm70_nvfp4_prefetch(graph) == expected
    assert norm.kwargs.get("prefetch_codes") is (codes if expected else None)
    assert _attach_sm70_nvfp4_prefetch(graph) == 0
    graph.lint()
