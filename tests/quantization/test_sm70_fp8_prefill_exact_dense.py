# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.fp8 import (
    _SM70_FP8_PREFILL_DENSE_MIN_M,
    _SM70_FP8_PREFILL_DENSE_WORKSPACE_BYTES,
    Fp8LinearMethod,
    _get_sm70_fp8_prefill_exact_dense_workspace,
    _is_sm70_fp8_prefill_exact_dense_layer,
    _sm70_fp8_prefill_dense_workspaces,
)


def test_fp8_prefill_exact_dense_is_default_on(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FP8_PREFILL_EXACT_DENSE", raising=False)
    envs.disable_envs_cache()
    try:
        assert envs.VLLM_SM70_FP8_PREFILL_EXACT_DENSE
    finally:
        envs.disable_envs_cache()


def test_fp8_prefill_exact_dense_shape_gate_is_narrow():
    layer = SimpleNamespace(
        tp_size=4,
        prefix="model.language_model.layers.1.mlp.gate_up_proj",
        weight=SimpleNamespace(shape=(5120, 8704)),
    )

    assert _is_sm70_fp8_prefill_exact_dense_layer(layer)

    layer.tp_size = 2
    assert not _is_sm70_fp8_prefill_exact_dense_layer(layer)
    layer.tp_size = 4
    layer.prefix = "model.language_model.layers.1.self_attn.qkv_proj"
    layer.weight = SimpleNamespace(shape=(5120, 3584))
    assert not _is_sm70_fp8_prefill_exact_dense_layer(layer)
    layer.prefix = "model.language_model.layers.1.linear_attn.in_proj_qkvz"
    layer.weight = SimpleNamespace(shape=(5120, 4096))
    assert not _is_sm70_fp8_prefill_exact_dense_layer(layer)
    layer.prefix = "model.language_model.layers.1.mlp.gate_up_proj"
    layer.weight = SimpleNamespace(shape=(5120, 8192))
    assert not _is_sm70_fp8_prefill_exact_dense_layer(layer)


def test_fp8_prefill_exact_dense_workspace_is_bounded():
    assert _SM70_FP8_PREFILL_DENSE_WORKSPACE_BYTES == 85 * 1024**2


def test_fp8_prefill_exact_dense_workspace_is_reused(monkeypatch):
    workspace = torch.empty(1, dtype=torch.float16)
    allocations = []

    def fake_empty(shape, *, dtype, device):
        allocations.append((shape, dtype, device))
        return workspace

    _sm70_fp8_prefill_dense_workspaces.clear()
    monkeypatch.setattr(torch, "empty", fake_empty)
    weight = SimpleNamespace(device=torch.device("cuda:0"))

    try:
        first = _get_sm70_fp8_prefill_exact_dense_workspace(weight)
        second = _get_sm70_fp8_prefill_exact_dense_workspace(weight)

        assert first is workspace
        assert second is workspace
        assert len(allocations) == 1
    finally:
        _sm70_fp8_prefill_dense_workspaces.clear()


def test_fp8_prefill_dispatch_reaches_runtime_op_for_small_and_large_m(monkeypatch):
    calls = []

    def fake_dispatch(
        out,
        dense_weight_ptr,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
        min_prefill_m,
    ):
        assert dense_weight_ptr == 42
        calls.append((input.shape[0], min_prefill_m, gated_silu))
        out.zero_()

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.sm70_ops."
        "fp8_gemm_sm70_prefill_dispatch_out",
        fake_dispatch,
    )
    layer = SimpleNamespace(
        sm70_fp8_turbomind=True,
        sm70_fp8_bmm=False,
        output_size_per_partition=6,
        weight=torch.empty((4, 6), dtype=torch.uint8),
        weight_scale_inv=torch.empty((1, 6), dtype=torch.float16),
        sm70_fp8_k_ld=4,
        sm70_fp8_q_ld=6,
        sm70_fp8_prefill_exact_dense_workspace_ptr=42,
    )
    method = SimpleNamespace()

    for m in (1, _SM70_FP8_PREFILL_DENSE_MIN_M):
        output = Fp8LinearMethod.apply(
            method, layer, torch.empty((m, 4), dtype=torch.float16)
        )
        assert output.shape == (m, 6)

    assert calls == [
        (1, _SM70_FP8_PREFILL_DENSE_MIN_M, False),
        (
            _SM70_FP8_PREFILL_DENSE_MIN_M,
            _SM70_FP8_PREFILL_DENSE_MIN_M,
            False,
        ),
    ]
