# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured MMQ admission excludes decode and unvalidated hardware/layouts."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.gguf import original_moe_prefill_capability


@pytest.fixture(autouse=True)
def operators(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C_gguf", SimpleNamespace(ggml_moe_mmq=object()))


@pytest.mark.parametrize("sm,minimum", [(70, 384), (86, 384), (89, 256)])
@pytest.mark.parametrize("kind,k,n,factor", [(21, 4096, 2048, 1), (23, 2048, 4096, 8)])
def test_measured_prefill_range_preserves_decode(sm, minimum, kind, k, n, factor):
    cap = original_moe_prefill_capability(
        kind,
        k,
        n,
        288,
        torch.float16,
        is_sm70=sm == 70,
        compute_capability=sm,
        top_k=8,
        mmq_supported=True,
    )
    assert cap.reason is None and cap.graph_safe
    assert cap.operator == "ggml_moe_mmq"
    assert cap.supports_m(minimum * factor)
    assert cap.supports_m(768 * factor)
    assert not cap.supports_m(minimum * factor - 1)
    assert not cap.supports_m(768 * factor + 1)
    for decode in (1, 5, 10, 15, 32):
        assert not cap.supports_m(decode * factor)


@pytest.mark.parametrize(
    "change",
    [
        {"compute_capability": 80, "is_sm70": False},
        {"source_type": 22},
        {"k": 4352},
        {"n": 2304},
        {"experts": 256},
        {"top_k": 10},
        {"dtype": torch.bfloat16},
        {"enabled": False},
        {"mmq_supported": False},
    ],
)
def test_unmeasured_or_unsupported_bank_keeps_fallback(change):
    args = dict(
        source_type=21,
        k=4096,
        n=2048,
        experts=288,
        dtype=torch.float16,
        is_sm70=True,
        compute_capability=70,
        top_k=8,
        mmq_supported=True,
    )
    args.update(change)
    assert original_moe_prefill_capability(**args).reason is not None


def test_missing_operator_keeps_fallback(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C_gguf", SimpleNamespace())
    cap = original_moe_prefill_capability(
        21,
        4096,
        2048,
        288,
        torch.float16,
        is_sm70=True,
        compute_capability=70,
        top_k=8,
        mmq_supported=True,
    )
    assert cap.reason == "operator_missing:ggml_moe_mmq"
