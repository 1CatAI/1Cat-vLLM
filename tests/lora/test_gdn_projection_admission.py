# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.lora.layers.row_parallel_linear import RowParallelLinearWithLoRA
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A16Fp8,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize("wrapped", [False, True])
def test_gdn_direct_output_requires_the_native_row_projection(monkeypatch, wrapped):
    monkeypatch.setattr(
        gdn.current_platform, "is_device_capability", lambda *a, **k: True
    )
    monkeypatch.setattr(gdn, "_sm70_gdn_graph_buffer_copy", lambda *a, **k: None)
    if wrapped:
        projection = object.__new__(RowParallelLinearWithLoRA)
        torch.nn.Module.__init__(projection)
        projection.base_layer = torch.nn.Linear(4, 4, bias=False)
    else:
        projection = object.__new__(RowParallelLinear)
        torch.nn.Module.__init__(projection)
        projection.bias = None
        projection.reduce_results = False
        projection.sm70_fp8_qpn8 = True
        projection.scheme = object.__new__(CompressedTensorsW8A16Fp8)
    output = torch.empty((8, 5120), dtype=torch.float16)
    expected = torch.full_like(output, 0.25)
    calls = []

    def compute(core, gate, tokens, destination):
        calls.append(destination)
        if destination is not None:
            destination.copy_(expected)
            return destination
        return expected

    owner = SimpleNamespace(
        prefix="audit.gdn",
        enable_sm70_dflash2_fused_gdn_verify=True,
        tp_size=4,
        out_proj=projection,
        _compute_output_projection=compute,
    )
    actual = gdn.QwenGatedDeltaNetAttention._output_projection(
        owner,
        torch.empty(0),
        torch.empty(0),
        output,
        8,
    )
    assert len(calls) == 1
    assert calls[0] is (None if wrapped else output)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
