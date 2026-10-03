# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped (is_bmm) block-FP8 weights on a Marlin-backed Fp8LinearMethod.

Marlin cannot serve DeepSeek-V4's grouped wo_a, so on cards that take Marlin
for block FP8 (Turing) the weight is dequantized at load and applied per group.
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


def _marlin_method() -> fp8.Fp8LinearMethod:
    method = fp8.Fp8LinearMethod.__new__(fp8.Fp8LinearMethod)
    method.use_marlin = True
    method.block_quant = True
    method.weight_block_size = [BLOCK, BLOCK]
    method.use_sm70_dequant_fallback = False
    return method


def test_marlin_bmm_weight_is_dequantized_at_load() -> None:
    layer, reference = _grouped_layer()

    _marlin_method().process_weights_after_loading(layer)

    assert layer.dequantized_bmm
    assert layer.weight.dtype == torch.float16
    assert torch.equal(layer.weight, reference.half())


@pytest.mark.parametrize("num_tokens", [1, 7])
def test_marlin_bmm_apply_multiplies_each_group_by_its_rows(num_tokens) -> None:
    layer, reference = _grouped_layer()
    method = _marlin_method()
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


def test_late_grouped_turing_metadata_discards_dense_qpn8_selection():
    from vllm.model_executor.kernels.linear.scaled_mm.qpn8_blk import (
        QPN8Fp8BlockScaledMMLinearKernel,
    )

    layer, reference = _grouped_layer()
    method = _marlin_method()
    method.use_marlin = False
    method.use_sm70_fp8_turbomind = False
    method.fp8_linear = QPN8Fp8BlockScaledMMLinearKernel.__new__(
        QPN8Fp8BlockScaledMMLinearKernel
    )
    method.process_weights_after_loading(layer)
    assert layer.dequantized_bmm
    assert torch.equal(layer.weight, reference.half())
    assert not hasattr(method, "fp8_linear")


def test_late_grouped_volta_metadata_reselects_grouped_kernel(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.kernels.linear.scaled_mm.qpn8_blk import (
        QPN8Fp8BlockScaledMMLinearKernel,
    )

    layer, _ = _grouped_layer()
    method = _marlin_method()
    method.use_marlin = False
    method.use_sm70_fp8_turbomind = True
    method.activation_quant_key = None
    method.weight_quant_key = None
    method.input_dtype = method.out_dtype = torch.float16
    method.is_scale_e8m0 = True
    method.fp8_linear = QPN8Fp8BlockScaledMMLinearKernel.__new__(
        QPN8Fp8BlockScaledMMLinearKernel
    )
    seen: list[torch.nn.Module] = []

    def select(**kwargs):
        assert kwargs["is_bmm"]
        return SimpleNamespace(process_weights_after_loading=seen.append)

    monkeypatch.setattr(fp8, "init_sm70_fp8_linear_kernel", select)
    method.process_weights_after_loading(layer)
    assert seen == [layer]
    assert not getattr(layer, "dequantized_bmm", False)
