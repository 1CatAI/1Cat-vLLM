# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    _qwen38_active_grouped_layer_contract,
    _use_qwen38_active_grouped_decode,
    _use_qwen38_indexed_prefill,
)
from vllm.model_executor.warmup import awq_sm70_warmup as warmup

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def default_policy(monkeypatch):
    for name in (
        "VLLM_SM70_AWQ_MOE_BATCHED_DECODE_MAX_TOKENS",
        "VLLM_SM70_AWQ_MOE_BATCHED_SINGLE_TOKEN_DENSE_W13",
        "VLLM_SM70_AWQ_MOE_BATCHED_EXACT_W2",
        "VLLM_SM70_AWQ_MOE_BATCHED_ACTIVE_EXACT_W2",
    ):
        monkeypatch.setenv(name, "0")


def _layer():
    return SimpleNamespace(
        moe_config=SimpleNamespace(tp_size=4),
        sm70_awq_qwen38_active_grouped_decode=True,
        sm70_awq_moe_batched_gemm=True,
        sm70_num_experts=512,
        sm70_hidden_logical_size=2560,
        sm70_w13_k_dim=2560,
        sm70_w13_n_dim=320,
        sm70_w2_k_dim=160,
        sm70_w2_n_dim=2560,
    )


def test_default_and_rollback(monkeypatch):
    name = "VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE"
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name)
    monkeypatch.setenv(name, "0")
    assert not getattr(envs, name)
    layer = _layer()
    layer.sm70_awq_qwen38_active_grouped_decode = False
    assert not _use_qwen38_active_grouped_decode(layer, 4, 10)


@pytest.mark.parametrize("tokens", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 32])
def test_token_gate(tokens):
    assert _use_qwen38_active_grouped_decode(_layer(), tokens, 10) == (2 <= tokens <= 8)


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_group_size_gate(group_size):
    assert _qwen38_active_grouped_layer_contract(_layer(), group_size) == (
        group_size == 32
    )


@pytest.mark.parametrize("tokens", [1, 2, 8, 9, 127, 128])
def test_grouped_decode_and_indexed_prefill_are_disjoint(tokens):
    layer = _layer()
    layer.sm70_awq_qwen38_indexed_prefill = True
    layer.sm70_awq_checkpoint_group_size = 32
    layer.sm70_awq_group_size = 32
    layer.sm70_intermediate_size = 160
    x = torch.empty(tokens, 2560, dtype=torch.float16)
    topk_ids = torch.empty(tokens, 10, dtype=torch.int32)
    grouped = _use_qwen38_active_grouped_decode(layer, tokens, 10)
    indexed = _use_qwen38_indexed_prefill(layer, x, topk_ids)
    assert grouped == (2 <= tokens <= 8)
    assert indexed == (tokens >= 128)
    assert not (grouped and indexed)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("sm70_num_experts", 256),
        ("sm70_hidden_logical_size", 2592),
        ("sm70_w13_k_dim", 2592),
        ("sm70_w13_n_dim", 384),
        ("sm70_w2_k_dim", 192),
        ("sm70_w2_n_dim", 2592),
    ],
)
def test_shape_gate(attribute, value):
    layer = _layer()
    setattr(layer, attribute, value)
    assert not _qwen38_active_grouped_layer_contract(layer, 32)


def test_topology_and_router_gate():
    layer = _layer()
    assert _qwen38_active_grouped_layer_contract(layer, 32)
    layer.moe_config.tp_size = 2
    assert not _qwen38_active_grouped_layer_contract(layer, 32)
    assert not _use_qwen38_active_grouped_decode(layer, 4, 8)
    layer.sm70_awq_moe_batched_gemm = False
    assert not _use_qwen38_active_grouped_decode(layer, 4, 10)


@pytest.mark.parametrize(
    "name",
    [
        "VLLM_SM70_AWQ_MOE_BATCHED_SINGLE_TOKEN_DENSE_W13",
        "VLLM_SM70_AWQ_MOE_BATCHED_EXACT_W2",
        "VLLM_SM70_AWQ_MOE_BATCHED_ACTIVE_EXACT_W2",
    ],
)
def test_explicit_routes_take_precedence(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    assert not _use_qwen38_active_grouped_decode(_layer(), 4, 10)


def test_decode_cap(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_AWQ_MOE_BATCHED_DECODE_MAX_TOKENS", "4")
    assert _use_qwen38_active_grouped_decode(_layer(), 4, 10)
    assert not _use_qwen38_active_grouped_decode(_layer(), 8, 10)


@pytest.mark.parametrize("strict", [False, True])
def test_warmup_reuses_active_op_and_runtime_policy(monkeypatch, strict):
    monkeypatch.setenv(
        "VLLM_SM70_AWQ_MOE_BATCHED_SINGLE_TOKEN_DENSE_W13", str(int(strict))
    )
    layer = _layer()
    layer._awq_moe_buf_top_k = 10
    layer.w13_tm_scales = torch.empty((512, 80, 320), dtype=torch.float16)
    for name in (
        "w13_strided_ptrs_w",
        "w13_strided_ptrs_s",
        "w2_strided_ptrs_w",
        "w2_strided_ptrs_s",
    ):
        setattr(layer, name, torch.empty(1, dtype=torch.uint8))
    dense_calls, active_calls = [], []
    monkeypatch.setattr(
        torch.ops._C, "awq_moe_dense_stage_sm70_out", object(), raising=False
    )
    monkeypatch.setattr(
        warmup.sm70_ops,
        "awq_moe_dense_stage_sm70_out",
        lambda *a: dense_calls.append(a),
    )
    monkeypatch.setattr(
        warmup.sm70_ops,
        "awq_moe_active_dense_stage_sm70_out",
        lambda *a: active_calls.append(a),
    )
    monkeypatch.setattr(warmup, "_silu_and_mul_w13", lambda *a: None)
    assert warmup._warmup_moe_dense_stage_layers([layer], [1, 2, 4, 8, 9]) == 10
    assert [a[7] for a in active_calls] == ([] if strict else [20, 20, 40, 40, 80, 80])
    assert len(dense_calls) == (10 if strict else 4)
    for w13, w2 in zip(active_calls[::2], active_calls[1::2]):
        assert w13[2].tolist() == list(range(w13[7]))
        assert w13[3].numel() == w13[7] + 1
        assert w13[4].numel() == w13[7]
        assert w13[3] is w2[3] and w13[4] is w2[4]
