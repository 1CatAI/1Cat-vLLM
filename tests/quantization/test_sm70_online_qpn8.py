# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization import sm70_online_qpn8 as online_qpn8


def test_qwen4_exp_online_qpn8_defaults_off_and_can_be_enabled(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", raising=False)
    envs.disable_envs_cache()
    try:
        assert not envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8
        monkeypatch.setenv("VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", "1")
        envs.disable_envs_cache()
        assert envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8
    finally:
        envs.disable_envs_cache()


def test_qwen3next_shared_gate_fusion_defaults_on_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_QWEN3NEXT_SHARED_GATE_FUSION", raising=False)
    envs.disable_envs_cache()
    try:
        assert envs.VLLM_SM70_QWEN3NEXT_SHARED_GATE_FUSION
        monkeypatch.setenv("VLLM_SM70_QWEN3NEXT_SHARED_GATE_FUSION", "0")
        envs.disable_envs_cache()
        assert not envs.VLLM_SM70_QWEN3NEXT_SHARED_GATE_FUSION
    finally:
        envs.disable_envs_cache()


def test_qpn_sidecars_respect_safe_online_default(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("VLLM_SM70_FP8_QPN8_LIBRARY", "/tmp/qpn8.so")
    monkeypatch.setenv("VLLM_SM70_NVFP4_QPN_M1_LIBRARY", "/tmp/qpn-m1.so")
    monkeypatch.delenv("VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", raising=False)
    monkeypatch.delenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE", raising=False)
    monkeypatch.setattr(torch.ops, "load_library", calls.append)

    online_qpn8.sm70_ops._maybe_load_fp8_qpn8_library()
    online_qpn8.sm70_ops._maybe_load_nvfp4_qpn_m1_library()

    assert calls == ["/tmp/qpn-m1.so"]

    calls.clear()
    monkeypatch.setenv("VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", "1")
    online_qpn8.sm70_ops._maybe_load_fp8_qpn8_library()
    assert calls == ["/tmp/qpn8.so"]


def test_nvfp4_sidecar_loads_for_mtp5_when_m1_is_disabled(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("VLLM_SM70_NVFP4_QPN_M1_LIBRARY", "/tmp/qpn-mtp5.so")
    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_M1_DECODE", "0")
    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE", "1")
    monkeypatch.setattr(torch.ops, "load_library", calls.append)

    online_qpn8.sm70_ops._maybe_load_nvfp4_qpn_m1_library()

    assert calls == ["/tmp/qpn-mtp5.so"]


def test_nvfp4_mtp5_capability_is_not_inferred_from_m1(monkeypatch):
    legacy_sidecar = SimpleNamespace(nvfp4_moe_qpn_m1_sm70_out=object())
    monkeypatch.setattr(torch.ops, "_C_qwen38", legacy_sidecar)
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace())

    assert online_qpn8.sm70_ops.has_nvfp4_qpn_m1_dispatch()
    assert not online_qpn8.sm70_ops.has_nvfp4_qpn_mtp5_dispatch()

    legacy_sidecar.nvfp4_moe_qpn_mtp5_sm70_out = object()
    assert online_qpn8.sm70_ops.has_nvfp4_qpn_mtp5_dispatch()


def test_nvfp4_w2_direct_reduce_capability_is_explicit(monkeypatch):
    legacy_sidecar = SimpleNamespace(nvfp4_moe_qpn_m1_sm70_out=object())
    monkeypatch.setattr(torch.ops, "_C_qwen38", legacy_sidecar)
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace())

    assert not online_qpn8.sm70_ops.has_nvfp4_qwen38_w2_direct_reduce()

    legacy_sidecar.nvfp4_qwen38_w2_direct_reduce_out = object()
    assert online_qpn8.sm70_ops.has_nvfp4_qwen38_w2_direct_reduce()


def test_qwen38_shared_gate_exact_capability_is_explicit(monkeypatch):
    sidecar = SimpleNamespace(qwen38_shared_gate_exact_out=object())
    monkeypatch.setattr(torch.ops, "_C_qwen38", sidecar)
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace())

    assert online_qpn8.sm70_ops.has_qwen38_shared_gate_exact()

    del sidecar.qwen38_shared_gate_exact_out
    assert not online_qpn8.sm70_ops.has_qwen38_shared_gate_exact()


@pytest.mark.parametrize(
    ("prefix", "k", "n", "expected"),
    [
        ("model.layers.0.linear_attn.in_proj_qkvz", 2560, 4096, (16, 1, False)),
        ("model.layers.0.linear_attn.out_proj", 1536, 2560, (12, 2, False)),
        ("model.layers.3.self_attn.qkv_proj", 2560, 3584, (16, 1, False)),
        ("model.layers.3.self_attn.o_proj", 1536, 2560, (12, 2, False)),
        (
            "model.layers.0.attn_hyper_connection.input_mix_weight_up",
            320,
            10240,
            (4, 2, False),
        ),
        (
            "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject",
            10240,
            336,
            (32, 1, False),
        ),
        ("model.layers.0.mlp.shared_expert.gate_up_proj", 2560, 320, None),
    ],
)
def test_online_qpn8_shape_gate(prefix, k, n, expected):
    assert online_qpn8._shape_config(prefix, k, n) == expected


def test_online_qpn8_runtime_contract_is_tp4_no_mtp(monkeypatch):
    text_config = SimpleNamespace(
        model_type="generic_hybrid_moe",
        hidden_size=2560,
        num_hidden_layers=48,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        hc_count=4,
        hc_lowrank=320,
        num_attention_heads=24,
        num_key_value_heads=2,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text_config),
        speculative_config=None,
    )
    monkeypatch.setattr(online_qpn8, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(online_qpn8, "get_tensor_model_parallel_world_size", lambda: 4)
    assert online_qpn8._exact_runtime_contract()

    config.speculative_config = SimpleNamespace(method="mtp")
    assert not online_qpn8._exact_runtime_contract()
    config.speculative_config = None
    monkeypatch.setattr(online_qpn8, "get_tensor_model_parallel_world_size", lambda: 2)
    assert not online_qpn8._exact_runtime_contract()
    monkeypatch.setattr(online_qpn8, "get_tensor_model_parallel_world_size", lambda: 4)
    text_config.hc_count = 2
    assert not online_qpn8._exact_runtime_contract()


def test_online_qpn8_apply_uses_prepared_state(monkeypatch):
    layer = torch.nn.Module()
    layer.register_buffer(
        online_qpn8._CODES_ATTR, torch.empty((16, 32), dtype=torch.uint8)
    )
    layer.register_buffer(
        online_qpn8._SCALES_ATTR, torch.empty((1, 32), dtype=torch.float16)
    )
    layer._sm70_qwen4_exp_online_qpn8 = True
    layer._sm70_qwen4_exp_online_qpn8_k = 16
    layer._sm70_qwen4_exp_online_qpn8_n = 32
    layer._sm70_qwen4_exp_online_qpn8_logical_n = 32
    layer._sm70_qwen4_exp_online_qpn8_split_k = 4
    layer._sm70_qwen4_exp_online_qpn8_nacc = 2
    layer._sm70_qwen4_exp_online_qpn8_prefetch = False
    layer._sm70_qwen4_exp_online_qpn8_workspace_ptr = 123
    calls = []

    def fake_dispatch(*args):
        calls.append(args)
        args[0].fill_(2)

    monkeypatch.setattr(
        online_qpn8.sm70_ops, "fp8_qpn8_dispatch_sm70_out", fake_dispatch
    )
    x = torch.ones((2, 3, 16), dtype=torch.float16)
    bias = torch.ones((32,), dtype=torch.float16)
    out = online_qpn8.maybe_apply_online_qpn8(layer, x, bias)
    assert out is not None and out.shape == (2, 3, 32)
    assert torch.equal(out, torch.full_like(out, 3))
    assert calls[0][1] == 123
    assert calls[0][-1] is False


def test_fused_hc_uses_prepared_partials_without_global_lookup(monkeypatch):
    down = torch.nn.Module()
    up = torch.nn.Module()
    for layer, k, n in ((down, 10240, 384), (up, 320, 10240)):
        setattr(layer, online_qpn8._STATE_ATTR, True)
        layer._sm70_qwen4_exp_online_qpn8_k = k
        layer._sm70_qwen4_exp_online_qpn8_n = n
        layer._sm70_qwen4_exp_online_qpn8_workspace_ptr = 123
        layer.register_buffer(
            online_qpn8._CODES_ATTR, torch.empty((1,), dtype=torch.uint8)
        )
        layer.register_buffer(
            online_qpn8._SCALES_ATTR, torch.empty((1,), dtype=torch.float16)
        )
    partials = torch.empty((32 * 384,), dtype=torch.float32)
    down.register_buffer(
        online_qpn8._HC_PARTIALS_ATTR,
        partials,
        persistent=False,
    )
    calls = []

    def fake_dispatch(*args):
        calls.append(args)
        args[0].fill_(2)
        args[1].fill_(3)

    monkeypatch.setattr(online_qpn8.sm70_ops, "has_fp8_qpn8_hc_dispatch", lambda: True)
    monkeypatch.setattr(
        online_qpn8.sm70_ops, "fp8_qpn8_hc_dispatch_sm70_out", fake_dispatch
    )
    xn = torch.ones((1, 10240), dtype=torch.float16)
    result = online_qpn8.maybe_apply_fused_hc(down, up, xn)
    assert result is not None
    block_out, injection_out = result
    assert torch.equal(block_out, torch.full_like(block_out, 2))
    assert torch.equal(injection_out, torch.full_like(injection_out, 3))
    assert calls[0][5] is partials


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_channel_qpn8_accepts_k16_n32_alignment():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    n, k = 64, 320
    weight = torch.randn((n, k), dtype=torch.float16, device="cuda").mul_(0.02)
    scales = weight.float().abs().amax(dim=1, keepdim=True).div_(448.0)
    qweight = (weight.float() / scales).to(torch.float8_e4m3fn)
    try:
        codes, packed_scales = online_qpn8.sm70_ops.fp8_qpn8_prepare_sm70(
            qweight.contiguous(), scales.contiguous()
        )
    except RuntimeError as exc:
        if "multiples of 128" in str(exc):
            pytest.skip("installed SM70 extension predates channel-QPN8 alignment")
        raise
    x = torch.randn((1, k), dtype=torch.float16, device="cuda").mul_(0.1)
    out = torch.empty((1, n), dtype=torch.float16, device="cuda")
    online_qpn8.sm70_ops.fp8_qpn8_gemm_sm70_out(
        out, x, codes, packed_scales, 4, 2, True, False
    )
    reference = x.float().matmul((qweight.float() * scales).t()).half()
    torch.accelerator.synchronize()
    relative_l2 = (out.float() - reference.float()).norm() / reference.float().norm()
    assert float(relative_l2) < 5e-3
