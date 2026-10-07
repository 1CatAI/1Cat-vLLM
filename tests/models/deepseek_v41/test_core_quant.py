# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2 item 2: DeepseekV41FP8Config -- dense FP8 32x32 -> exact FP16 fallback, wo_a einsum, MoE factory, Engram guard."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config


def _vllm_config() -> VllmConfig:
    """A default VllmConfig with a stand-in model_config: Fp8LinearMethod reads model_config.dtype."""
    from types import SimpleNamespace

    cfg = VllmConfig()
    object.__setattr__(cfg, "model_config", SimpleNamespace(dtype=torch.float16))
    return cfg

QCFG = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0", "expert_dtype": "fp4"}


def _qconfig():
    from vllm.models.deepseek_v41.quant_config import DeepseekV41FP8Config

    return DeepseekV41FP8Config.from_config(dict(QCFG))


def _random_fp8_block(n: int, k: int, lo: int = -13, hi: int = -6, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (n, k), generator=g, dtype=torch.uint8)
    raw[(raw & 0x7F) == 0x7F] = 0x3C          # no NaN encodings
    weight = raw.view(torch.float8_e4m3fn)
    exps = torch.randint(lo, hi + 1, (-(-n // 32), -(-k // 32)), generator=g)
    scale = torch.pow(2.0, exps.float()).to(torch.float8_e8m0fnu)
    return weight, scale


def test_dequant_exact_matches_fp32_product() -> None:
    from vllm.models.deepseek_v41.quant_config import dequantize_fp8_block32_exact

    weight, scale = _random_fp8_block(96, 160)
    out = dequantize_fp8_block32_exact(weight, scale)
    ref = weight.float() * scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    assert out.dtype == torch.float16 and torch.equal(out.float(), ref)


def test_dequant_refuses_inexact_scale() -> None:
    from vllm.models.deepseek_v41.quant_config import dequantize_fp8_block32_exact

    weight, scale = _random_fp8_block(32, 32, lo=-20, hi=-20)
    with pytest.raises(ValueError, match="not exactly representable"):
        dequantize_fp8_block32_exact(weight, scale)


def test_config_validation() -> None:
    from vllm.models.deepseek_v41.quant_config import DeepseekV41FP8Config

    cfg = _qconfig()
    assert cfg.get_name() == "deepseek_v41_fp8" and cfg.get_min_capability() == 70 and cfg.is_scale_e8m0
    with pytest.raises(ValueError, match="weight_block_size"):
        DeepseekV41FP8Config.from_config({**QCFG, "weight_block_size": [128, 128]})
    with pytest.raises(ValueError, match="expert_dtype"):
        DeepseekV41FP8Config.from_config({**QCFG, "expert_dtype": "fp8"})


def _linear(cfg, prefix: str, n: int, k: int):
    from vllm.model_executor.layers.linear import ReplicatedLinear

    return ReplicatedLinear(k, n, bias=False, quant_config=cfg, prefix=prefix, disable_tp=True,
                            params_dtype=torch.float16)


def test_linear_dequant_fallback_and_forward(ds41_dist_single) -> None:
    from vllm.models.deepseek_v41.quant_config import DeepseekV41SM70Fp8LinearMethod

    with set_current_vllm_config(_vllm_config()):
        cfg = _qconfig()
        layer = _linear(cfg, "model.layers.3.attn.wq_a", 64, 96)
    assert isinstance(layer.quant_method, DeepseekV41SM70Fp8LinearMethod)
    assert layer.weight.dtype == torch.float8_e4m3fn and layer.weight_scale_inv.dtype == torch.float8_e8m0fnu
    weight, scale = _random_fp8_block(64, 96, seed=1)
    layer.weight.weight_loader(layer.weight, weight)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scale)
    layer.quant_method.process_weights_after_loading(layer)
    expect = weight.float() * scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    assert layer.weight.dtype == torch.float16 and torch.equal(layer.weight.float(), expect)
    x = torch.randn(5, 96, dtype=torch.float32).to(torch.float16)
    out, _ = layer(x)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), x.float() @ expect.t(), rtol=2e-3, atol=2e-3)


def test_bmm_layer_uses_grouped_einsum(ds41_dist_single) -> None:
    with set_current_vllm_config(_vllm_config()):
        cfg = _qconfig()
        layer = _linear(cfg, "model.layers.3.attn.wo_a", 2 * 64, 32)   # 2 groups of [64, 32]
    layer.is_bmm = True
    layer.bmm_batch_size = 2
    weight, scale = _random_fp8_block(128, 32, seed=2)
    layer.weight.weight_loader(layer.weight, weight)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scale)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.dequantized_bmm
    w = layer.weight.float().view(2, 64, 32)
    x = torch.randn(3, 2, 32).to(torch.float16)
    out = layer.quant_method.apply(layer, x)
    torch.testing.assert_close(out.float(), torch.einsum("tgk,grk->tgr", x.float(), w), rtol=2e-3, atol=2e-3)


def test_engram_linear_refused(ds41_dist_single) -> None:
    with set_current_vllm_config(_vllm_config()):
        cfg = _qconfig()
        with pytest.raises(ValueError, match="Engram"):
            _linear(cfg, "model.layers.1.engram.wkv", 64, 64)


def test_moe_routes_to_l_moe_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.model_executor.layers.fused_moe import FusedMoE

    calls: list[object] = []
    sentinel = object()
    stub = types.ModuleType("vllm.models.deepseek_v41.sm70.moe")

    def make_v41_moe_method(moe_config):
        calls.append(moe_config)
        return sentinel

    stub.make_v41_moe_method = make_v41_moe_method  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.models.deepseek_v41.sm70.moe", stub)
    layer = FusedMoE.__new__(FusedMoE)
    object.__setattr__(layer, "moe_config", "MOE-CONFIG")
    with set_current_vllm_config(_vllm_config()):
        assert _qconfig().get_quant_method(layer, "model.layers.3.ffn.experts") is sentinel
    assert calls == ["MOE-CONFIG"]
