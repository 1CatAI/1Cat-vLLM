# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serialized block-FP8 linears with weight_block_size [32, 32] on SM70.

Exact power-of-two scales keep the weight FP8 in the TurboMind group-32
layout; other scales are dequantized to FP16 at load; grouped (is_bmm) layers
and the existing policy switches keep their previous routes; [128, 128] is
untouched.
"""

import os
from types import SimpleNamespace as NS

import pytest
import torch

from vllm import envs
from vllm.config.kernel import KernelConfig
from vllm.model_executor.layers.quantization import fp8
from vllm.model_executor.layers.quantization.utils import sm70_fp8_block32 as b32
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an SM70 GPU",
)

N, K = 1024, 2048


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch, dist_init):
    for name in list(os.environ):
        if name.startswith("VLLM_SM70_"):
            monkeypatch.delenv(name)
    envs.disable_envs_cache()
    matmul = torch.backends.cuda.matmul
    saved = matmul.allow_fp16_reduced_precision_reduction
    matmul.allow_fp16_reduced_precision_reduction = False
    yield
    matmul.allow_fp16_reduced_precision_reduction = saved
    envs.disable_envs_cache()


def _method(monkeypatch, block: list[int]) -> fp8.Fp8LinearMethod:
    engine = NS(model_config=NS(dtype=torch.float16), kernel_config=KernelConfig())
    monkeypatch.setattr(fp8, "get_current_vllm_config", lambda: engine)
    config = fp8.Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=block,
    )
    return fp8.Fp8LinearMethod(config)


def _layer(method, block: int, exponents: torch.Tensor, n: int = N, k: int = K):
    """A loaded layer: random finite E4M3 codes, scales 2^exponents (FP32 as
    Fp8Config stores them), and the exact FP32 dequantized reference."""
    layer = torch.nn.Module()
    with torch.device("cuda"):
        method.create_weights(
            layer, k, [n], k, n, torch.float16, weight_loader=default_weight_loader
        )
    g = torch.Generator(device="cuda").manual_seed(n + k + block)
    codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda", generator=g)
    codes[(codes & 0x7F) == 0x7F] = 0x3C
    layer.weight.data.copy_(codes.view(torch.float8_e4m3fn))
    layer.weight_scale_inv.data.copy_(torch.exp2(exponents.float()))
    full = torch.exp2(exponents.float())
    full = full.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return layer, layer.weight.float() * full


def _exponents(block: int, lo: int = -13, hi: int = -6, n: int = N, k: int = K):
    g = torch.Generator(device="cuda").manual_seed(block)
    return torch.randint(
        lo, hi + 1, (n // block, k // block), device="cuda", generator=g
    )


def _rel_rms(out: torch.Tensor, exact: torch.Tensor) -> float:
    d = out.double() - exact
    return (d.pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()).item()


def _check_apply(method, layer, reference):
    """Every M regime is as accurate as the FP16 dequantized weight with cuBLAS
    (FP16 output + FP16 bias add: ~3e-4 relative RMS)."""
    bias = torch.randn(N, device="cuda").half()
    ref16 = reference.half()
    for m in (1, 2, 3, 5, 40, 300):
        x = torch.randn(2, m, K, device="cuda").half()
        out = method.apply(layer, x, bias)
        assert out.shape == (2, m, N) and out.dtype == torch.float16
        exact = x.double() @ reference.double().t() + bias.double()
        fallback = _rel_rms(torch.nn.functional.linear(x, ref16, bias), exact)
        rel = _rel_rms(out, exact)
        assert rel < 1.5 * fallback + 1e-6 and rel < 5e-4, (m, rel, fallback)


def test_block32_power_of_two_scales_stay_fp8(monkeypatch):
    method = _method(monkeypatch, [32, 32])
    assert method.sm70_fp8_block32_candidate
    layer, reference = _layer(method, 32, _exponents(32))
    method.process_weights_after_loading(layer)
    assert layer.sm70_fp8_block32
    assert layer.weight.dtype == torch.uint8 and layer.weight.numel() == N * K
    assert tuple(layer.weight_scale_inv.shape) == (K // 32, N)
    _check_apply(method, layer, reference)
    # apply() runs exactly the sm70_fp8_block32 dispatch.
    w = b32.Fp8Block32Weight(
        layer.weight,
        layer.weight_scale_inv,
        layer.sm70_fp8_k_ld,
        layer.sm70_fp8_q_ld,
        N,
        K,
    )
    for m in (1, 4, 64, 257):
        x = torch.randn(m, K, device="cuda").half()
        torch.testing.assert_close(
            method.apply(layer, x), b32.fp8_block32_linear(x, w), rtol=0, atol=0
        )
    # A second call is a no-op (reload / repeated processing).
    method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.uint8


def test_block32_e8m0_scale_parameter(monkeypatch):
    method = _method(monkeypatch, [32, 32])
    exps = _exponents(32)
    layer, reference = _layer(method, 32, exps)
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.exp2(exps.float()).to(torch.float8_e8m0fnu), requires_grad=False
    )
    method.process_weights_after_loading(layer)
    assert layer.sm70_fp8_block32
    _check_apply(method, layer, reference)


@pytest.mark.parametrize("bad", ["exponent", "fraction"])
def test_block32_inexact_scales_dequantize_at_load(monkeypatch, bad):
    """Scales FP16 cannot apply exactly take the load-time FP16 dequantization
    (today's dequant_fallback route) instead of the FP8 kernels."""
    method = _method(monkeypatch, [32, 32])
    exps = _exponents(32)
    if bad == "exponent":
        exps[0, 0] = b32.SCALE_EXP_MIN - 4  # e.g. 2^-18
    layer, reference = _layer(method, 32, exps)
    if bad == "fraction":
        layer.weight_scale_inv.data[0, 0] *= 0.75
        reference[:32, :32] *= 0.75
    expected = method._dequantize_block_weight(
        layer.weight.data.clone(), layer.weight_scale_inv.data.clone(), torch.float16
    )
    method.process_weights_after_loading(layer)
    assert not getattr(layer, "sm70_fp8_block32", False)
    assert layer.sm70_fp8_block32_dequantized
    assert torch.equal(layer.weight, expected)
    x = torch.randn(3, K, device="cuda").half()
    torch.testing.assert_close(
        method.apply(layer, x), torch.nn.functional.linear(x, expected), rtol=0, atol=0
    )


def test_block32_nan_codes_fail_loudly(monkeypatch):
    method = _method(monkeypatch, [32, 32])
    layer, _ = _layer(method, 32, _exponents(32))
    layer.weight.data.view(torch.uint8)[5, 7] = 0xFF
    with pytest.raises(ValueError, match="NaN codes"):
        method.process_weights_after_loading(layer)


def test_block32_grouped_bmm_keeps_the_existing_route(monkeypatch):
    method = _method(monkeypatch, [32, 32])
    layer, _ = _layer(method, 32, _exponents(32))
    layer.is_bmm = True
    layer.bmm_batch_size = 2
    layer.orig_dtype = torch.float16
    method.process_weights_after_loading(layer)
    assert not getattr(layer, "sm70_fp8_block32", False)


@pytest.mark.parametrize(
    "env",
    [{"VLLM_SM70_FP8_TURBOMIND": "0"}, {"VLLM_SM70_QUANT_BACKEND": "marlin"}],
)
def test_block32_policy_switches_keep_previous_routes(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    envs.disable_envs_cache()
    method = _method(monkeypatch, [32, 32])
    assert not method.sm70_fp8_block32_candidate
    layer, reference = _layer(method, 32, _exponents(32))
    if env.get("VLLM_SM70_QUANT_BACKEND") == "marlin":
        # The legacy Marlin backend keeps its own selection, unchanged here.
        assert method.use_marlin
        return
    method.process_weights_after_loading(layer)
    assert not getattr(layer, "sm70_fp8_block32", False)
    assert method.use_sm70_dequant_fallback
    assert torch.equal(layer.weight, reference.half())
    _check_apply(method, layer, reference)


def test_block128_route_unchanged(monkeypatch):
    method = _method(monkeypatch, [128, 128])
    assert not method.sm70_fp8_block32_candidate
    assert method.use_sm70_fp8_turbomind
    layer, reference = _layer(method, 128, _exponents(128, -12, -6))
    layer.prefix = "test.block128"
    method.process_weights_after_loading(layer)
    assert layer.sm70_fp8_turbomind
    assert not getattr(layer, "sm70_fp8_block32", False)
    _check_apply(method, layer, reference)
