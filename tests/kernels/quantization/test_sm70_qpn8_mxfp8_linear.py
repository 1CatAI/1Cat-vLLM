# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 linears on Volta and Turing: QPN8 codes against dequantized weights.

M <= 8 runs the packed GEMV, M up to 512 on Turing the tensor-core GEMM, the rest
(and Volta above M=8) the dequantizing cuBLAS route."""

import pytest
import torch
from torch import nn

from vllm.model_executor.kernels.linear.qpn import pre_ampere  # noqa: F401
from vllm.model_executor.layers.quantization import sm70_turbomind as tm
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    dequant_mxfp8_to_bf16,
)


def _mxfp8_weight(n: int, k: int, exponents: tuple[int, int]):
    weight = (torch.randn(n, k, device="cuda") * 8).to(torch.float8_e4m3fn)
    scale = torch.randint(
        exponents[0],
        exponents[1] + 1,
        (n, k // MXFP8_BLOCK_SIZE),
        device="cuda",
        dtype=torch.uint8,
    )
    return weight, scale


@pytest.mark.parametrize("rows", [1, 4, 8, 9, 33, 64, 300, 512, 777])
@pytest.mark.parametrize("n,k", [(4096, 8192), (12288, 4096)])
def test_qpn8_mxfp8_linear_matches_the_dequantized_weight(rows, n, k):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (7, 0),
        (7, 5),
    ):
        pytest.skip("requires SM70 or SM75")
    import vllm._C  # noqa: F401

    torch.manual_seed(1017)
    low, high = tm.MXFP8_QPN8_EXPONENT_RANGE
    weight, scale = _mxfp8_weight(n, k, (low, high))
    reference_weight = dequant_mxfp8_to_bf16(weight, scale).float()
    layer = nn.Module()
    tm.prepare_mxfp8_qpn8_linear(layer, weight, scale)

    x = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.1
    out = tm.apply_prepared_fp8_qpn8_linear(layer, x, None)
    reference = x.float() @ reference_weight.t()

    assert out.shape == (rows, n)
    assert torch.isfinite(out).all()
    error = (out.float() - reference).abs().max()
    assert error <= 2e-3 * reference.abs().max()


@pytest.mark.parametrize("n,k", [(64, 128), (4096, 8192)])
def test_qpn8_mxfp8_dequantization_equals_the_load_time_dequantization(n, k):
    # The prefill multiplies the dequantized workspace; it must hold the same
    # FP16 weights the loader used to materialize once.
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (7, 0),
        (7, 5),
    ):
        pytest.skip("requires SM70 or SM75")
    import vllm._C  # noqa: F401

    from vllm import _sm70_ops

    torch.manual_seed(2043)
    weight, scale = _mxfp8_weight(n, k, tm.MXFP8_QPN8_EXPONENT_RANGE)
    layer = nn.Module()
    tm.prepare_mxfp8_qpn8_linear(layer, weight, scale)
    state = getattr(layer, tm.FP8_QPN8_STATE_ATTR)

    dense = torch.empty(k, n, device="cuda", dtype=torch.float16)
    _sm70_ops.fp8_qpn8_dequantize_sm70_out(dense, state.codes, state.group_scales)

    expected = dequant_mxfp8_to_bf16(weight, scale).to(torch.float16).t()
    assert torch.equal(dense, expected)


def test_qpn8_mxfp8_prepare_rejects_exponents_outside_the_exact_range():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (7, 0),
        (7, 5),
    ):
        pytest.skip("requires SM70 or SM75")
    import vllm._C  # noqa: F401

    low, _ = tm.MXFP8_QPN8_EXPONENT_RANGE
    weight, scale = _mxfp8_weight(64, 128, (low, low))
    scale[0, 0] = low - 1
    with pytest.raises(RuntimeError, match="exact QPN8 range"):
        tm.prepare_mxfp8_qpn8_linear(nn.Module(), weight, scale)


@pytest.mark.parametrize(
    "n,k,reason",
    [
        (4096, 8192, None),
        (48, 128, "N % 32 == 0"),
        # K = 160 is a valid MXFP8 width, but no split-K divides its 10 groups.
        (64, 160, "no QPN8 launch configuration"),
    ],
)
def test_qpn8_mxfp8_kernel_declines_shapes_it_cannot_pack(monkeypatch, n, k, reason):
    """A shape QPN8 cannot pack goes to the next MXFP8 kernel at selection
    instead of failing when the weights load."""
    from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig

    monkeypatch.setattr(pre_ampere, "_qpn8_fp8_available", lambda: (True, None))
    ok, why = pre_ampere.Qpn8Mxfp8LinearKernel.can_implement(
        Mxfp8LinearLayerConfig((n, k))
    )
    assert ok == (reason is None)
    if reason is not None:
        assert reason in why
