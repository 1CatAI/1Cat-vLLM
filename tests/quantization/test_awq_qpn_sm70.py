# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.quantization import awq_qpn_sm70 as qpn
from vllm.model_executor.layers.quantization import awq_sm70_moe as moe

pytestmark = pytest.mark.skip_global_cleanup


def layer():
    return SimpleNamespace(
        sm70_awq_moe_batched_gemm=True,
        sm70_awq_moe_w13_interleaved=True,
        sm70_awq_moe_legacy_single_token_compact=True,
        sm70_awq_moe_compact_metadata=True,
        sm70_awq_checkpoint_group_size=32,
        sm70_awq_group_size=32,
        sm70_awq_qwen38_qpn_m1=True,
    )


def test_default_on_unsupported_shape_never_loads(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_AWQ_QWEN38_QPN_M1", raising=False)
    assert envs.VLLM_SM70_AWQ_QWEN38_QPN_M1
    monkeypatch.setattr(
        qpn, "_has_native_op", lambda: pytest.fail("unexpected native lookup")
    )
    assert not qpn.initialize_qpn_m1(object(), False)


def test_explicit_disable_never_loads(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_AWQ_QWEN38_QPN_M1", "0")
    monkeypatch.setattr(
        qpn, "_has_native_op", lambda: pytest.fail("unexpected native lookup")
    )
    assert not qpn.initialize_qpn_m1(layer(), True)


@pytest.mark.parametrize("native_available", [False, True])
def test_default_on_checks_native_capability(monkeypatch, native_available):
    monkeypatch.delenv("VLLM_SM70_AWQ_QWEN38_QPN_M1", raising=False)
    monkeypatch.setattr(qpn, "_has_native_op", lambda: native_available)
    assert qpn.initialize_qpn_m1(layer(), True) == native_available


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("sm70_awq_moe_batched_gemm", False),
        ("sm70_awq_moe_w13_interleaved", False),
        ("sm70_awq_moe_legacy_single_token_compact", False),
        ("sm70_awq_checkpoint_group_size", 128),
        ("sm70_awq_group_size", 64),
    ],
)
def test_implicit_unsupported_layer_falls_back(monkeypatch, attribute, value):
    monkeypatch.delenv("VLLM_SM70_AWQ_QWEN38_QPN_M1", raising=False)
    current = layer()
    setattr(current, attribute, value)
    monkeypatch.setattr(
        qpn, "_has_native_op", lambda: pytest.fail("unexpected native lookup")
    )
    assert not qpn.initialize_qpn_m1(current, True)


@pytest.mark.parametrize("value", ["2", "true", "", "-1"])
def test_bad_flag_rejected(monkeypatch, value):
    monkeypatch.setenv("VLLM_SM70_AWQ_QWEN38_QPN_M1", value)
    with pytest.raises(ValueError):
        qpn.initialize_qpn_m1(layer(), True)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("sm70_awq_moe_batched_gemm", False),
        ("sm70_awq_moe_w13_interleaved", False),
        ("sm70_awq_moe_legacy_single_token_compact", False),
        ("sm70_awq_checkpoint_group_size", 128),
        ("sm70_awq_group_size", 64),
    ],
)
def test_explicit_unsupported_layer_fails(monkeypatch, attribute, value):
    monkeypatch.setenv("VLLM_SM70_AWQ_QWEN38_QPN_M1", "1")
    current = layer()
    setattr(current, attribute, value)
    with pytest.raises(RuntimeError):
        qpn.initialize_qpn_m1(current, True)


def test_explicit_shape_and_native_build(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_AWQ_QWEN38_QPN_M1", "1")
    with pytest.raises(RuntimeError):
        qpn.initialize_qpn_m1(layer(), False)
    monkeypatch.setattr(qpn, "_has_native_op", lambda: False)
    with pytest.raises(RuntimeError, match="native build"):
        qpn.initialize_qpn_m1(layer(), True)
    monkeypatch.setattr(qpn, "_has_native_op", lambda: True)
    assert qpn.initialize_qpn_m1(layer(), True)


@pytest.mark.parametrize("compact", [False, True])
def test_both_existing_metadata_layouts(monkeypatch, compact):
    monkeypatch.setenv("VLLM_SM70_AWQ_QWEN38_QPN_M1", "1")
    monkeypatch.setattr(qpn, "_has_native_op", lambda: True)
    current = layer()
    current.sm70_awq_moe_compact_metadata = compact
    assert qpn.initialize_qpn_m1(current, True)


def test_no_sidecar_or_implicit_build(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_AWQ_QWEN38_QPN_M1", "1")
    monkeypatch.setenv("VLLM_SM70_AWQ_QPN_EXTENSION_PATH", "/old/research.so")
    monkeypatch.setattr(qpn, "_has_native_op", lambda: False)
    monkeypatch.setattr(
        torch.ops, "load_library", lambda _: pytest.fail("unexpected DSO load")
    )
    with pytest.raises(RuntimeError, match="native build"):
        qpn.initialize_qpn_m1(layer(), True)


@pytest.mark.parametrize("tokens", [0, 1, 2, 4, 5, 8, 128, 8192])
def test_only_one_physical_token(tokens):
    x = torch.empty(tokens, 2560, dtype=torch.float16)
    ids = torch.empty(tokens, 10, dtype=torch.int32)
    weights = torch.empty(tokens, 10, dtype=torch.float32)
    assert qpn.use_qpn_m1(layer(), x, weights, ids) == (tokens == 1)


def test_runtime_layout_and_rollback():
    current = layer()
    x = torch.empty(1, 2560, dtype=torch.float16)
    ids = torch.empty(1, 10, dtype=torch.int32)
    weights = torch.empty(1, 10, dtype=torch.float32)
    assert qpn.use_qpn_m1(current, x, weights, ids)
    assert not qpn.use_qpn_m1(current, x.bfloat16(), weights, ids)
    assert not qpn.use_qpn_m1(current, x, weights.half(), ids)
    assert not qpn.use_qpn_m1(current, x, weights, ids.long())
    strided = torch.empty(1, 5120, dtype=torch.float16)[:, ::2]
    assert not qpn.use_qpn_m1(current, strided, weights, ids)
    current.sm70_awq_qwen38_qpn_m1 = False
    assert not qpn.use_qpn_m1(current, x, weights, ids)


@pytest.mark.parametrize("compact", [False, True])
def test_real_moe_branch_calls_native_with_existing_banks(monkeypatch, compact):
    current = layer()
    current.sm70_awq_moe_compact_metadata = compact
    for name in ("w13_tm_weight", "w13_tm_scales", "w2_tm_weight", "w2_tm_scales"):
        setattr(current, name, object())
    x = torch.empty(1, 2560, dtype=torch.float16)
    ids = torch.empty(1, 10, dtype=torch.int32)
    weights = torch.empty(1, 10, dtype=torch.float32)
    out = torch.empty_like(x)
    intermediate = torch.empty(10, 160, dtype=x.dtype)
    seen: list[tuple[Any, ...]] = []

    def native(*args):
        seen.append(args)
        args[0].fill_(7)

    monkeypatch.setattr(moe.sm70_ops, "awq_moe_qpn_m1_sm70_out", native)
    monkeypatch.setattr(
        moe.sm70_ops,
        "awq_moe_single_token_sm70_out",
        lambda *args: pytest.fail("unexpected legacy route"),
    )
    result = moe.AWQSM70MoEMethod._apply_legacy_single_token_compact(
        cast(Any, object()),
        cast(Any, current),
        x,
        weights,
        ids,
        {"intermediate": intermediate},
        10,
        out,
    )
    expected = (
        out,
        intermediate,
        x,
        current.w13_tm_weight,
        current.w13_tm_scales,
        current.w2_tm_weight,
        current.w2_tm_scales,
        ids,
        weights,
    )
    assert len(seen) == 1
    assert all(actual is wanted for actual, wanted in zip(seen[0], expected))
    assert result is out
    assert torch.all(out == 7)
