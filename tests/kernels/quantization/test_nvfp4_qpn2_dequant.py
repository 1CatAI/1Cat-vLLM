# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The QPN2 dequantization kernel reproduces the checkpoint weight exactly.

Round trip: random NVFP4 codes and e4m3 block scales -> QPN2 prepack
(``nvfp4_qpn2_prepare_sm70``) -> Triton dequantization -> compare with the
direct dequantization of the checkpoint tensors and with the pure-torch
inverse of the prepack. NVFP4 values times an e4m3 scale times the global
scale are exact in fp32 and round once to fp16, so the comparison is exact.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _checkpoint_dequant(
    packed: torch.Tensor, scales: torch.Tensor, global_scale: float
) -> torch.Tensor:
    n, k_half = packed.shape
    low = packed & 0xF
    high = packed >> 4
    nib = torch.stack([low, high], dim=-1).view(n, k_half * 2)
    values = E2M1.to(packed.device)[(nib & 7).long()] * torch.where(
        nib & 8 > 0, -1.0, 1.0
    )
    scale = scales.view(torch.float8_e4m3fn).to(torch.float32)
    values = values.view(n, -1, 16) * scale.unsqueeze(-1) * global_scale
    return values.view(n, k_half * 2).to(torch.float16)


@pytest.mark.parametrize(
    ("n", "k"),
    [(32, 64), (64, 256), (3584, 5120), (8704, 5120), (1536, 5120)],
)
def test_qpn2_dequant_matches_checkpoint(n: int, k: int):
    from vllm import _sm70_ops as sm70_ops
    from vllm.model_executor.layers.quantization.utils.nvfp4_qpn2_dequant import (
        nvfp4_qpn2_dequant,
        nvfp4_qpn2_dequant_reference,
    )

    if not hasattr(torch.ops._C, "nvfp4_qpn2_prepare_sm70"):
        pytest.skip("build without the SM70 QPN2 extension")
    generator = torch.Generator(device="cuda").manual_seed(n * 31 + k)
    packed = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=generator
    )
    # e4m3 scales without the NaN code and without the sign bit, like
    # ModelOpt block scales.
    scales = torch.randint(
        0, 0x7F, (n, k // 16), dtype=torch.uint8, device="cuda", generator=generator
    )
    global_scale = 0.0123
    expected = _checkpoint_dequant(packed, scales, global_scale)

    codes, qpn2_scales = sm70_ops.nvfp4_qpn2_prepare_sm70(
        packed, scales.view(torch.float8_e4m3fn)
    )
    reference = nvfp4_qpn2_dequant_reference(codes, qpn2_scales, global_scale, n, k)
    kernel = nvfp4_qpn2_dequant(codes, qpn2_scales, global_scale, n, k)

    assert torch.equal(reference, expected)
    assert torch.equal(kernel, expected)


def test_qpn2_dense_linear_matches_fp16_matmul():
    from vllm import _sm70_ops as sm70_ops
    from vllm.model_executor.layers.quantization.utils.nvfp4_qpn2_dequant import (
        nvfp4_qpn2_dense_linear,
    )

    if not hasattr(torch.ops._C, "nvfp4_qpn2_prepare_sm70"):
        pytest.skip("build without the SM70 QPN2 extension")
    n, k, m = 3584, 5120, 64
    generator = torch.Generator(device="cuda").manual_seed(7)
    packed = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=generator
    )
    scales = torch.randint(
        0, 0x7F, (n, k // 16), dtype=torch.uint8, device="cuda", generator=generator
    )
    global_scale = 0.0123
    x = torch.randn(m, k, dtype=torch.float16, device="cuda", generator=generator)
    codes, qpn2_scales = sm70_ops.nvfp4_qpn2_prepare_sm70(
        packed, scales.view(torch.float8_e4m3fn)
    )

    expected = torch.nn.functional.linear(
        x, _checkpoint_dequant(packed, scales, global_scale)
    )
    out = nvfp4_qpn2_dense_linear(x, codes, qpn2_scales, global_scale, n, k)
    assert out.shape == (m, n)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("m", [1, 4, 32, 33, 64])
def test_qpn2_dispatch_linear_both_sides_of_the_threshold(m: int):
    from vllm import _sm70_ops as sm70_ops
    from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
    from vllm.model_executor.layers.quantization.utils.nvfp4_qpn2_dequant import (
        nvfp4_qpn2_dispatch_linear,
    )

    if not hasattr(torch.ops._C, "nvfp4_qpn2_prepare_sm70"):
        pytest.skip("build without the SM70 QPN2 extension")
    n, k = 3584, 5120
    generator = torch.Generator(device="cuda").manual_seed(11)
    packed = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=generator
    )
    # Block scales between 0.25 and 1.5 (e4m3 0x28..0x3c): a K=5120 dot
    # product of such rows stays inside fp16, which the GEMM output is.
    scales = torch.randint(
        0x28, 0x3D, (n, k // 16), dtype=torch.uint8, device="cuda", generator=generator
    )
    global_scale = 0.01
    x = torch.randn(m, k, dtype=torch.float16, device="cuda", generator=generator)
    codes, qpn2_scales = sm70_ops.nvfp4_qpn2_prepare_sm70(
        packed, scales.view(torch.float8_e4m3fn)
    )
    split_k, chains = sm70_tm.qpn2_launch_config(k, n)

    expected = torch.nn.functional.linear(
        x, _checkpoint_dequant(packed, scales, global_scale)
    )
    out = nvfp4_qpn2_dispatch_linear(
        x, codes, qpn2_scales, global_scale, n, k, split_k, chains
    )
    assert out.shape == (m, n)
    # The QPN2 kernel accumulates in a different order than cuBLAS; both are
    # fp32 accumulations of exact products, so they agree to fp16 rounding.
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-2)
