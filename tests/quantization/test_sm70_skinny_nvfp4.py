# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _sm70_ops
from vllm.model_executor.kernels.linear.nvfp4 import skinny
from vllm.model_executor.layers.quantization import modelopt


def _unpack_nibble(codes: torch.Tensor, row: int, column: int) -> int:
    packed = int(codes[row, column // 2])
    return (packed >> (4 * (column & 1))) & 0xF


def test_qpn_prepack_matches_fragment_order_reference():
    n, k = 32, 64
    codes = torch.arange(n * (k // 2), dtype=torch.int64)
    codes = codes.remainder(256).to(torch.uint8).view(n, k // 2)
    scales = torch.arange(n * (k // 16), dtype=torch.int64)
    scales = scales.remainder(256).to(torch.uint8).view(n, k // 16)

    actual_codes, actual_scales = skinny.qpn_prepack(codes, scales)

    assert actual_codes is not None
    assert actual_scales is not None
    expected_codes: list[int] = []
    expected_scales: list[int] = []
    korder = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]
    for group in range(k // 16):
        for lane in range(32):
            column = ((lane >> 2) & 3) * 8 + (lane & 3) + (4 if lane & 16 else 0)
            selected = [
                _unpack_nibble(codes, column, group * 16 + offset) for offset in korder
            ]
            expected_codes.extend(
                selected[index] | (selected[index + 1] << 4)
                for index in range(0, 16, 2)
            )
            expected_scales.append(int(scales[column, group]))

    assert actual_codes.tolist() == expected_codes
    assert actual_scales.tolist() == expected_scales


def test_qpn_prepack_rejects_ineligible_shape_without_copy():
    codes = torch.zeros((31, 32), dtype=torch.uint8)
    scales = torch.zeros((31, 4), dtype=torch.uint8)

    qcodes, qscales = skinny.qpn_prepack(codes, scales)

    assert qcodes is None
    assert qscales is None


def _native_buffers(n: int, k: int):
    codes = torch.zeros((n, k // 2), dtype=torch.uint8)
    scales = torch.zeros((n, k // 16), dtype=torch.uint8)
    qcodes = torch.zeros(n * (k // 2), dtype=torch.uint8)
    qscales = torch.zeros(n * (k // 16), dtype=torch.uint8)
    tm_weight = torch.empty(0, dtype=torch.int32)
    tm_scales = torch.empty(0, dtype=torch.float16)
    return codes, scales, qcodes, qscales, tm_weight, tm_scales


def test_bf16_activation_is_explicitly_converted_and_restored(monkeypatch):
    n, k = 32, 128
    buffers = _native_buffers(n, k)
    seen_dtypes = []

    def fake_simt(input, codes, scales, global_scale):
        del codes, scales, global_scale
        seen_dtypes.append(input.dtype)
        return input.sum(dim=1, keepdim=True).expand(-1, n).contiguous()

    monkeypatch.setattr(_sm70_ops, "skinny_nvfp4_gemm_simt", fake_simt)
    x = torch.ones((1, k), dtype=torch.bfloat16)

    out = skinny._skinny_nvfp4_linear_impl(x, *buffers, 1.0, n, k, 16, 0, 0)

    assert seen_dtypes == [torch.float16]
    assert out.dtype == torch.bfloat16
    assert out.shape == (1, n)


def test_large_m_uses_turbomind_fallback(monkeypatch):
    n, k, m = 32, 128, 17
    buffers = _native_buffers(n, k)
    calls = []

    def fake_turbomind(
        out, input, qweight, scales, group_size, k_ld, q_ld, gated_silu=False
    ):
        del qweight, scales, group_size, k_ld, q_ld, gated_silu
        calls.append(input.dtype)
        out.copy_(input.sum(dim=1, keepdim=True).expand(-1, n))

    monkeypatch.setattr(_sm70_ops, "nvfp4_gemm_sm70_out", fake_turbomind)
    x = torch.ones((m, k), dtype=torch.bfloat16)

    out = skinny._skinny_nvfp4_linear_impl(x, *buffers, 1.0, n, k, 16, 0, 0)

    assert calls == [torch.float16]
    assert out.dtype == torch.bfloat16
    assert out.shape == (m, n)


def test_skinny_kernel_rejects_non_sm70():
    supported, reason = skinny.SkinnyNvFp4LinearKernel.is_supported(80)

    assert not supported
    assert reason == "requires exact CUDA capability 7.0"


def test_modelopt_nvfp4_min_capability_is_lowered_only_for_skinny(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_QUANT_BACKEND", raising=False)
    assert modelopt.ModelOptNvFp4Config.get_min_capability() == 75

    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "skinny")
    assert modelopt.ModelOptNvFp4Config.get_min_capability() == 70


def test_modelopt_w4a16_selects_generic_skinny_kernel(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "skinny")
    monkeypatch.setattr(modelopt, "init_nvfp4_linear_kernel", lambda: sentinel)
    config = modelopt.ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )

    method = modelopt.ModelOptNvFp4W4A16LinearMethod(config)

    assert method.kernel is sentinel


def test_modelopt_w4a4_selects_generic_skinny_kernel(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "skinny")
    monkeypatch.setattr(modelopt, "init_nvfp4_linear_kernel", lambda: sentinel)
    config = modelopt.ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )

    method = modelopt.ModelOptNvFp4LinearMethod(config)

    assert method.kernel is sentinel
