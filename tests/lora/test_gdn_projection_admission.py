# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.lora.layers.row_parallel_linear import RowParallelLinearWithLoRA
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
    CompressedTensorsW8A16Fp8,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("dflash_verify", [True, False])
@pytest.mark.parametrize("quant_kind", ["ct_fp8", "ct_nvfp4", "block_fp8"])
@pytest.mark.parametrize("tp,rows", [(2, 1), (2, 5), (2, 17), (2, 40), (4, 8)])
def test_gdn_direct_output_requires_the_native_row_projection(
    monkeypatch, wrapped, dflash_verify, quant_kind, tp, rows
):
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
        if quant_kind == "ct_fp8":
            projection.sm70_fp8_qpn8 = True
            projection.scheme = object.__new__(CompressedTensorsW8A16Fp8)
        elif quant_kind == "ct_nvfp4":
            scheme = object.__new__(CompressedTensorsW4A4Fp4)
            scheme._fallback_kernel = lambda: SimpleNamespace(
                supports_direct_output=lambda layer: True
            )
            projection.scheme = scheme
        else:
            projection.scheme = None
            method = object.__new__(Fp8LinearMethod)
            method.fp8_linear = SimpleNamespace(
                supports_direct_output=lambda layer: True
            )
            projection.quant_method = method
    output = torch.empty((rows, 5120), dtype=torch.float16)
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
        enable_sm70_dflash2_fused_gdn_verify=dflash_verify,
        tp_size=tp,
        hidden_size=5120,
        out_proj=projection,
        _compute_output_projection=compute,
    )
    actual = gdn.QwenGatedDeltaNetAttention._output_projection(
        owner,
        torch.empty(0),
        torch.empty(0),
        output,
        rows,
    )
    assert len(calls) == 1
    assert calls[0] is (None if wrapped else output)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
