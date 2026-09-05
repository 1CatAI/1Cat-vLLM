# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm import envs
from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    _is_qwen38_tp4_compact_metadata_shape,
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


def test_awq_compact_metadata_is_default_off() -> None:
    assert envs.VLLM_SM70_AWQ_MOE_COMPACT_METADATA is False


def test_awq_compact_metadata_shape_gate_accepts_qwen38_tp4() -> None:
    assert _is_qwen38_tp4_compact_metadata_shape(_layer(), 32)


def test_awq_compact_metadata_shape_gate_rejects_other_contracts() -> None:
    assert not _is_qwen38_tp4_compact_metadata_shape(_layer(experts=256), 32)
    assert not _is_qwen38_tp4_compact_metadata_shape(_layer(w2_experts=256), 32)
    assert not _is_qwen38_tp4_compact_metadata_shape(_layer(), 64)
    assert not _is_qwen38_tp4_compact_metadata_shape(_layer(w13_shape=(2560, 48)), 32)
    assert not _is_qwen38_tp4_compact_metadata_shape(_layer(w2_shape=(192, 320)), 32)
