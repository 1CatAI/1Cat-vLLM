# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped (is_bmm) block-FP8 weights on a Marlin- or QPN8-backed Fp8LinearMethod.

Neither kernel serves DeepSeek-V4's grouped wo_a, so on cards that take one of
them for block FP8 the weight is dequantized at load and applied per group.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization import fp8

GROUPS = 4
ROWS_PER_GROUP = 256
K = 384
BLOCK = 128


def _grouped_layer() -> tuple[torch.nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    layer = torch.nn.Module()
    weight = (torch.randn(GROUPS * ROWS_PER_GROUP, K) * 8).to(torch.float8_e4m3fn)
    exponents = torch.randint(-12, -4, (GROUPS * ROWS_PER_GROUP // BLOCK, K // BLOCK))
    scales = torch.pow(2.0, exponents.float()).to(torch.float8_e8m0fnu)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
    layer.orig_dtype = torch.float16
    layer.is_bmm = True
    layer.bmm_batch_size = GROUPS
    full_scales = (
        scales.float().repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    )
    return layer, weight.float() * full_scales


def _method(kernel: str) -> fp8.Fp8LinearMethod:
    method = fp8.Fp8LinearMethod.__new__(fp8.Fp8LinearMethod)
    method.use_marlin = kernel == "marlin"
    method.use_qpn8 = kernel == "qpn8"
    method.block_quant = True
    method.weight_block_size = [BLOCK, BLOCK]
    method.use_sm70_dequant_fallback = False
    return method


@pytest.mark.parametrize("kernel", ["marlin", "qpn8"])
def test_bmm_weight_is_dequantized_at_load(kernel) -> None:
    layer, reference = _grouped_layer()

    _method(kernel).process_weights_after_loading(layer)

    assert layer.dequantized_bmm
    assert layer.weight.dtype == torch.float16
    assert torch.equal(layer.weight, reference.half())


@pytest.mark.parametrize("kernel", ["marlin", "qpn8"])
@pytest.mark.parametrize("num_tokens", [1, 7])
def test_bmm_apply_multiplies_each_group_by_its_rows(kernel, num_tokens) -> None:
    layer, reference = _grouped_layer()
    method = _method(kernel)
    method.process_weights_after_loading(layer)
    x = torch.randn(num_tokens, GROUPS, K).half()

    out = method.apply(layer, x)

    assert out.shape == (num_tokens, GROUPS, ROWS_PER_GROUP)
    for group in range(GROUPS):
        rows = reference[group * ROWS_PER_GROUP : (group + 1) * ROWS_PER_GROUP]
        expected = x[:, group].float() @ rows.half().float().t()
        torch.testing.assert_close(
            out[:, group].float(), expected, rtol=2e-3, atol=2e-2
        )
