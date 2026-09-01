# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a4_nvfp4 as nvfp4_scheme,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Fp4,
)


def test_nvfp4_qpn2_is_default_off_with_explicit_on(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_NVFP4_QPN2", raising=False)
    monkeypatch.delenv("VLLM_SM70_NVFP4_QPN2_PREFILL", raising=False)
    monkeypatch.delenv("VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M", raising=False)
    envs.disable_envs_cache()
    try:
        assert not envs.VLLM_SM70_NVFP4_QPN2
        assert not envs.VLLM_SM70_NVFP4_QPN2_PREFILL
        assert envs.VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M == 1024
        monkeypatch.setenv("VLLM_SM70_NVFP4_QPN2", "1")
        monkeypatch.setenv("VLLM_SM70_NVFP4_QPN2_PREFILL", "1")
        monkeypatch.setenv("VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M", "9")
        envs.disable_envs_cache()
        assert envs.VLLM_SM70_NVFP4_QPN2
        assert envs.VLLM_SM70_NVFP4_QPN2_PREFILL
        assert envs.VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M == 9
    finally:
        envs.disable_envs_cache()


def _runtime_config(
    *, method="dflash", draft_tokens=7, selector_top_k=16, tp=4, max_num_seqs=1
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=tp,
            enable_dbo=False,
            ubatch_size=0,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=(
            None
            if method is None
            else SimpleNamespace(
                method=method,
                num_speculative_tokens=draft_tokens,
                draft_model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        dflash_config={"selector_top_k": selector_top_k}
                    )
                ),
            )
        ),
    )


def test_nvfp4_qpn2_dflash2_default_contract(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_NVFP4_QPN2", raising=False)
    monkeypatch.delenv("VLLM_SM70_NVFP4_QPN2_PREFILL", raising=False)
    monkeypatch.setattr(
        nvfp4_scheme, "get_current_vllm_config", lambda: _runtime_config()
    )
    envs.disable_envs_cache()
    try:
        assert nvfp4_scheme._sm70_nvfp4_qpn2_enabled()
        assert nvfp4_scheme._sm70_nvfp4_qpn2_prefill_enabled()

        monkeypatch.setattr(
            nvfp4_scheme,
            "get_current_vllm_config",
            lambda: _runtime_config(method=None),
        )
        assert not nvfp4_scheme._sm70_nvfp4_qpn2_enabled()
        assert not nvfp4_scheme._sm70_nvfp4_qpn2_prefill_enabled()

        monkeypatch.setattr(
            nvfp4_scheme,
            "get_current_vllm_config",
            lambda: _runtime_config(draft_tokens=5),
        )
        assert not nvfp4_scheme._sm70_nvfp4_qpn2_enabled()

        monkeypatch.setattr(
            nvfp4_scheme,
            "get_current_vllm_config",
            lambda: _runtime_config(selector_top_k=0),
        )
        assert not nvfp4_scheme._sm70_nvfp4_qpn2_enabled()

        monkeypatch.setattr(
            nvfp4_scheme,
            "get_current_vllm_config",
            lambda: _runtime_config(tp=2),
        )
        assert not nvfp4_scheme._sm70_nvfp4_qpn2_prefill_enabled()

        monkeypatch.setattr(
            nvfp4_scheme,
            "get_current_vllm_config",
            lambda: _runtime_config(max_num_seqs=256),
        )
        assert nvfp4_scheme._sm70_nvfp4_qpn2_enabled()
        assert nvfp4_scheme._sm70_nvfp4_qpn2_prefill_enabled()
    finally:
        envs.disable_envs_cache()


def test_nvfp4_qpn2_dflash2_explicit_rollback(monkeypatch):
    monkeypatch.setattr(
        nvfp4_scheme, "get_current_vllm_config", lambda: _runtime_config()
    )
    monkeypatch.setenv("VLLM_SM70_NVFP4_QPN2", "0")
    monkeypatch.setenv("VLLM_SM70_NVFP4_QPN2_PREFILL", "0")
    envs.disable_envs_cache()
    try:
        assert not nvfp4_scheme._sm70_nvfp4_qpn2_enabled()
        assert not nvfp4_scheme._sm70_nvfp4_qpn2_prefill_enabled()
    finally:
        envs.disable_envs_cache()


def test_nvfp4_qpn2_shape_gate_is_exact_tp4():
    layer = SimpleNamespace(
        tp_size=4,
        prefix="model.language_model.layers.0.mlp.gate_up_proj",
        input_size_per_partition=5120,
        output_size_per_partition=8704,
        weight=SimpleNamespace(shape=(8704, 2560)),
    )
    assert nvfp4_scheme._is_qpn2_layer(layer)

    layer.tp_size = 2
    assert not nvfp4_scheme._is_qpn2_layer(layer)
    layer.tp_size = 4
    layer.prefix = "model.language_model.layers.0.self_attn.qkv_proj"
    assert not nvfp4_scheme._is_qpn2_layer(layer)
    layer.prefix = "model.language_model.layers.0.mlp.down_proj"
    layer.input_size_per_partition = 4352
    layer.output_size_per_partition = 5120
    layer.weight = SimpleNamespace(shape=(5120, 2176))
    assert nvfp4_scheme._is_qpn2_layer(layer)
    layer.input_size_per_partition = 4096
    assert not nvfp4_scheme._is_qpn2_layer(layer)


def _make_small_layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.prefix = "model.language_model.layers.0.mlp.gate_up_proj"
    layer.tp_size = 4
    layer.input_size_per_partition = 64
    layer.output_size_per_partition = 64
    layer.weight_packed = Parameter(
        torch.zeros((64, 32), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = Parameter(
        torch.ones((64, 4), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_global_scale = Parameter(
        torch.tensor([2.0, 1.0], dtype=torch.float32), requires_grad=False
    )
    layer.input_global_scale = Parameter(
        torch.tensor([4.0, 4.0], dtype=torch.float32), requires_grad=False
    )
    return layer


def test_nvfp4_qpn2_prepare_and_dispatch_contract(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_NVFP4_QPN2", "1")
    monkeypatch.setenv("VLLM_SM70_NVFP4_QPN2_PREFILL", "1")
    monkeypatch.setenv("VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M", "9")
    envs.disable_envs_cache()
    layer = _make_small_layer()
    calls = []

    monkeypatch.setattr(nvfp4_scheme.sm70_tm, "use_turbomind", lambda enabled: True)
    scheme = CompressedTensorsW4A4Fp4()
    monkeypatch.setattr(
        nvfp4_scheme.sm70_tm,
        "should_prepare_turbomind",
        lambda tensor, enabled: enabled,
    )
    monkeypatch.setattr(nvfp4_scheme, "_is_qpn2_layer", lambda layer: True)
    monkeypatch.setattr(nvfp4_scheme, "_missing_qpn2_ops", lambda: [])
    monkeypatch.setattr(nvfp4_scheme, "_missing_qpn2_prefill_ops", lambda: [])
    monkeypatch.setitem(nvfp4_scheme._SM70_NVFP4_QPN2_CONFIGS, (64, 64, False), (8, 2))
    monkeypatch.setitem(nvfp4_scheme._SM70_NVFP4_QPN2_CONFIGS, (64, 64, True), (8, 2))
    monkeypatch.setattr(
        nvfp4_scheme.sm70_ops,
        "nvfp4_qpn2_prepare_sm70",
        lambda weight, scales: (
            torch.empty_like(weight),
            torch.empty(scales.shape, dtype=torch.uint8),
        ),
    )

    def fake_prepare(prepared_layer, *, interleave_gated_silu=False):
        assert not interleave_gated_silu
        state = sm70_tm.SM70TurboMindLinearState(
            weight=torch.empty((1,), dtype=torch.int32),
            scales=torch.empty((1,), dtype=torch.float16),
            group_size=16,
            k_ld=64,
            q_ld=64,
            output_size=64,
            op_kind="nvfp4",
        )
        setattr(prepared_layer, sm70_tm.STATE_ATTR, state)

    monkeypatch.setattr(nvfp4_scheme.sm70_tm, "prepare_nvfp4_linear", fake_prepare)

    def fake_dispatch(*args):
        calls.append(args)
        args[0].fill_(3)

    monkeypatch.setattr(
        nvfp4_scheme.sm70_ops, "nvfp4_qpn2_dispatch_sm70_out", fake_dispatch
    )
    combined_calls = []

    def fake_combined_dispatch(*args):
        combined_calls.append(args)
        args[0].fill_(5 if args[1].shape[0] >= args[-1] else 3)

    monkeypatch.setattr(
        nvfp4_scheme.sm70_ops,
        "nvfp4_qpn2_prefill_dispatch_sm70_out",
        fake_combined_dispatch,
    )

    try:
        scheme.process_weights_after_loading(layer)
        assert layer.sm70_nvfp4_qpn2
        assert layer.sm70_nvfp4_qpn2_gated_silu
        assert layer.sm70_nvfp4_qpn2_global_scale == 0.5
        assert layer.sm70_nvfp4_qpn2_prefill_enabled
        assert layer.weight.numel() == 0
        assert layer.weight_scale.numel() == 0

        x = torch.ones((8, 64), dtype=torch.float16)
        raw = scheme.apply_weights(layer, x)
        fused = scheme.apply_fused_silu_and_mul(layer, x)
        assert raw.shape == (8, 64)
        assert fused is not None and fused.shape == (8, 32)
        assert torch.equal(raw, torch.full_like(raw, 3))
        assert torch.equal(fused, torch.full_like(fused, 3))
        assert not calls
        assert combined_calls[0][-2:] == (False, 9)
        assert combined_calls[1][-2:] == (True, 9)

        large_x = torch.ones((9, 64), dtype=torch.float16)
        large_raw = scheme.apply_weights(layer, large_x)
        large_fused = scheme.apply_fused_silu_and_mul(layer, large_x)
        assert torch.equal(large_raw, torch.full_like(large_raw, 5))
        assert large_fused is not None
        assert torch.equal(large_fused, torch.full_like(large_fused, 5))
        assert combined_calls[2][-2:] == (False, 9)
        assert combined_calls[3][-2:] == (True, 9)
    finally:
        envs.disable_envs_cache()
