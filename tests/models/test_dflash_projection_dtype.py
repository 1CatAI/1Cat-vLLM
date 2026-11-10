# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearKernel
from vllm.model_executor.kernels.linear.qpn.nvfp4 import TurboMindNvFp4LinearKernel
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    _dflash_context_projection_dtype,
)


@pytest.mark.parametrize("storage_dtype", [torch.uint8, torch.int32])
@pytest.mark.parametrize("params_dtype", [torch.float16, torch.bfloat16])
def test_empty_packed_weights_retain_their_provider_contract(
    storage_dtype, params_dtype
):
    projection = SimpleNamespace(
        weight=torch.empty(0, dtype=storage_dtype),
        quant_method=SimpleNamespace(params_dtype=params_dtype),
    )
    assert _dflash_context_projection_dtype(projection) is params_dtype
    projection.quant_method.kernel = SimpleNamespace(input_dtype=torch.float32)
    assert _dflash_context_projection_dtype(projection) is torch.float32


def test_only_the_selected_nvfp4_provider_requires_fp16():
    assert NvFp4LinearKernel.input_dtype is None
    projection = SimpleNamespace(
        weight=torch.empty(0, dtype=torch.uint8),
        quant_method=SimpleNamespace(
            params_dtype=torch.bfloat16,
            kernel=object.__new__(TurboMindNvFp4LinearKernel),
        ),
    )
    assert _dflash_context_projection_dtype(projection) is torch.float16
    projection.quant_method.kernel = None
    assert _dflash_context_projection_dtype(projection) is torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("squeeze", [False, True])
def test_aux_projection_executes_prepared_sm70_nvfp4(input_dtype, squeeze):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 provider integration")
    if not hasattr(torch.ops._C, "nvfp4_sm70_prepare"):
        pytest.skip("SM70 TurboMind extension is unavailable")

    class Projection(torch.nn.Module):
        input_size = 256

        def forward(self, hidden):
            return sm70_tm.apply_prepared_linear(self, hidden, None)

    fc = Projection().cuda()
    fc.weight = torch.nn.Parameter(
        torch.full((128, 128), 0x22, dtype=torch.uint8, device="cuda"),
        requires_grad=False,
    )
    fc.weight_scale = torch.nn.Parameter(
        torch.ones(128, 16, dtype=torch.float8_e4m3fn, device="cuda"),
        requires_grad=False,
    )
    fc.weight_global_scale = torch.tensor(0.01, device="cuda")
    sm70_tm.prepare_nvfp4_linear(fc)
    TurboMindNvFp4LinearKernel._release_checkpoint_parameters(fc)
    fc.quant_method = SimpleNamespace(
        params_dtype=torch.bfloat16,
        kernel=object.__new__(TurboMindNvFp4LinearKernel),
    )
    assert fc.weight.numel() == 0 and fc.weight.dtype is torch.uint8
    outer = SimpleNamespace(model=SimpleNamespace(use_aux_hidden_state=True, fc=fc))
    hidden = torch.linspace(-1, 1, 256, device="cuda", dtype=input_dtype)
    if not squeeze:
        hidden = hidden.repeat(3, 1)
    actual = DFlashQwen3ForCausalLM.combine_hidden_states(outer, hidden)
    expected = fc(hidden.to(torch.float16).reshape(-1, 256))
    if squeeze:
        expected = expected.squeeze(0)
    assert actual.dtype is torch.float16
    assert torch.isfinite(actual).all()
    # Same prepared provider and FP16 operands: the adapter must add no math.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
