# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.gguf import small_grouped_vector_capabilities


@pytest.mark.parametrize("source", [20, 42])
def test_measured_routed_batches_and_negative_c4(monkeypatch, source):
    monkeypatch.setattr(
        torch.ops, "_C", SimpleNamespace(gguf_small_grouped_vec_sm70_out=object())
    )
    caps = small_grouped_vector_capabilities(
        source, 160, 2560, 512, torch.float16, is_sm70=True
    )
    for m in (10, 50):
        assert any(c.reason is None and c.supports_m(m) for c in caps)
    for m in (1, 20, 100, 150, 200, 512):
        assert not any(c.reason is None and c.supports_m(m) for c in caps)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"enabled": False}, "disabled_by_kernel_config"),
        ({"is_sm70": False}, "requires_sm70"),
        ({"dtype": torch.bfloat16}, "requires_fp16_activations"),
        ({"source_type": 21}, "requires_iq4_nl_or_q2_0_canonical_storage"),
        ({"k": 640}, "grouped_vector_shape_has_no_calibration"),
        ({"num_experts": 4}, "grouped_vector_shape_has_no_calibration"),
    ],
)
def test_fallback_reasons(monkeypatch, changes, reason):
    monkeypatch.setattr(
        torch.ops, "_C", SimpleNamespace(gguf_small_grouped_vec_sm70_out=object())
    )
    descriptor = dict(
        source_type=20,
        k=160,
        n=2560,
        num_experts=512,
        dtype=torch.float16,
        is_sm70=True,
    )
    descriptor.update(changes)
    assert {c.reason for c in small_grouped_vector_capabilities(**descriptor)} == {
        reason
    }


def test_missing_packaged_operator_reports_reason(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace())
    caps = small_grouped_vector_capabilities(
        42, 160, 2560, 512, torch.float16, is_sm70=True
    )
    assert {c.reason for c in caps} == {
        "operator_missing:gguf_small_grouped_vec_sm70_out"
    }
