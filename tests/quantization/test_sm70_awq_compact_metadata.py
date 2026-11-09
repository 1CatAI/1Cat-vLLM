# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm import envs
from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    _is_qwen38_tp4_compact_metadata_shape,
    _resolve_compact_metadata,
)

pytestmark = pytest.mark.skip_global_cleanup


def _layer(
    *,
    experts: int = 512,
    w2_experts: int | None = None,
    w13_shape: tuple[int, int] = (2560, 40),
    w2_shape: tuple[int, int] = (160, 320),
) -> SimpleNamespace:
    if w2_experts is None:
        w2_experts = experts
    return SimpleNamespace(
        w13_qweight=SimpleNamespace(shape=(experts, *w13_shape)),
        w2_qweight=SimpleNamespace(shape=(w2_experts, *w2_shape)),
    )


def test_awq_compact_metadata_is_default_on(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SM70_AWQ_MOE_COMPACT_METADATA", raising=False)
    assert envs.VLLM_SM70_AWQ_MOE_COMPACT_METADATA is True
    monkeypatch.setenv("VLLM_SM70_AWQ_MOE_COMPACT_METADATA", "0")
    assert envs.VLLM_SM70_AWQ_MOE_COMPACT_METADATA is False


def test_awq_compact_metadata_default_falls_back_when_unsupported() -> None:
    resolve = _resolve_compact_metadata
    assert resolve(requested=True, explicit=False, native_available=True, shape_ok=True)
    assert not resolve(
        requested=True, explicit=False, native_available=False, shape_ok=True
    )
    assert not resolve(
        requested=True, explicit=False, native_available=True, shape_ok=False
    )
    assert not resolve(
        requested=False, explicit=True, native_available=True, shape_ok=True
    )


def test_awq_compact_metadata_explicit_request_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="awq_sm70_prepare_compact"):
        _resolve_compact_metadata(
            requested=True, explicit=True, native_available=False, shape_ok=True
        )
    with pytest.raises(RuntimeError, match="Qwen3.8 TP4"):
        _resolve_compact_metadata(
            requested=True, explicit=True, native_available=True, shape_ok=False
        )


def test_awq_compact_metadata_shape_gate_accepts_qwen38_tp4() -> None:
    assert _is_qwen38_tp4_compact_metadata_shape(_layer(), 32)


def test_awq_compact_metadata_shape_gate_rejects_other_contracts() -> None:
    assert not _is_qwen38_tp4_compact_metadata_shape(_layer(experts=256), 32)
    assert not _is_qwen38_tp4_compact_metadata_shape(_layer(w2_experts=256), 32)
    assert not _is_qwen38_tp4_compact_metadata_shape(_layer(), 64)
    assert not _is_qwen38_tp4_compact_metadata_shape(_layer(w13_shape=(2560, 48)), 32)
    assert not _is_qwen38_tp4_compact_metadata_shape(_layer(w2_shape=(192, 320)), 32)
