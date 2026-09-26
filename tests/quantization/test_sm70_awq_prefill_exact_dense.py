# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization.awq import (
    _SM70_AWQ_PREFILL_DENSE_WORKSPACE_BYTES,
    _awq_exact_f16_weight,
    _get_sm70_awq_prefill_exact_dense_workspace,
    _is_sm70_awq_prefill_exact_dense_layer,
    _sm70_awq_prefill_dense_workspaces,
)


def test_awq_exact_f16_weight_matches_half_fma_rounding():
    qweight = torch.full((8, 2), 0x11111111, dtype=torch.int32)
    qzeros = torch.full((1, 2), 0x33333333, dtype=torch.int32)
    scales = torch.full((1, 16), 0.0001, dtype=torch.float16)

    actual = _awq_exact_f16_weight(qweight, scales, qzeros, group_size=8)
    quant = torch.ones((), dtype=torch.float16)
    zero = torch.full((), 3.0, dtype=torch.float16)
    scale = scales[0, 0]
    bias = -zero * scale
    expected = torch.addcmul(bias, quant, scale)
    naive = (quant - zero) * scale

    assert actual.shape == (8, 16)
    assert actual.is_contiguous()
    assert torch.equal(actual, torch.full_like(actual, expected))
    assert expected != naive


def test_awq_prefill_exact_dense_admits_aligned_local_layouts():
    qweight = SimpleNamespace(shape=(5120, 1088))
    layer = SimpleNamespace(
        tp_size=4,
        prefix="model.language_model.layers.1.mlp.gate_up_proj",
        qweight=qweight,
    )

    assert _is_sm70_awq_prefill_exact_dense_layer(layer)

    layer.tp_size = 2
    assert _is_sm70_awq_prefill_exact_dense_layer(layer)
    layer.tp_size = 4
    layer.prefix = "model.language_model.layers.1.self_attn.qkv_proj"
    assert not _is_sm70_awq_prefill_exact_dense_layer(layer)
    layer.prefix = "model.language_model.layers.1.mlp.gate_up_proj"
    layer.qweight = SimpleNamespace(shape=(5120, 1023))
    assert not _is_sm70_awq_prefill_exact_dense_layer(layer)


def test_awq_prefill_exact_dense_workspace_is_bounded():
    assert _SM70_AWQ_PREFILL_DENSE_WORKSPACE_BYTES == 85 * 1024**2


def test_awq_prefill_exact_dense_workspace_is_reused(monkeypatch):
    workspace = torch.empty(1, dtype=torch.float16)
    allocations = []

    def fake_empty(shape, *, dtype, device):
        allocations.append((shape, dtype, device))
        return workspace

    _sm70_awq_prefill_dense_workspaces.clear()
    monkeypatch.setattr(torch, "empty", fake_empty)
    weight = SimpleNamespace(device=torch.device("cuda:0"), numel=lambda: 5120 * 1088)

    try:
        first = _get_sm70_awq_prefill_exact_dense_workspace(weight)
        second = _get_sm70_awq_prefill_exact_dense_workspace(weight)

        assert first is workspace
        assert second is workspace
        assert len(allocations) == 1
    finally:
        _sm70_awq_prefill_dense_workspaces.clear()
