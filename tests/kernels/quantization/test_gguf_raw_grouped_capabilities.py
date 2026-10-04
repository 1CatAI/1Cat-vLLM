# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.gguf import raw_grouped_gate_up_capabilities


@pytest.fixture
def operator(monkeypatch):
    monkeypatch.setattr(
        torch.ops,
        "_C",
        SimpleNamespace(gguf_lattice_raw_grouped_gate_up_sm70_out=object()),
    )


@pytest.mark.parametrize("kind,expected", [(21, [1, 5, 20]), (22, [1, 5])])
def test_only_measured_original_batches_are_admitted(operator, kind, expected):
    caps = raw_grouped_gate_up_capabilities(
        kind, 2560, 160, 512, torch.float16, is_sm70=True
    )
    admitted = [
        m
        for m in (1, 2, 4, 5, 8, 10, 16, 20, 32, 50, 200)
        if any(c.reason is None and c.supports_m(m) for c in caps)
    ]
    assert admitted == expected
    if kind == 22:
        assert caps[-1].reason == "measured_slower_than_canonical_grouped_gemm"


@pytest.mark.parametrize(
    "kind,k,n,experts,dtype,hardware,reason",
    [
        (
            18,
            2560,
            160,
            512,
            torch.float16,
            True,
            "raw_grouped_source_format_unavailable",
        ),
        (21, 2560, 160, 512, torch.float32, True, "requires_fp16_activations"),
        (
            21,
            2560,
            160,
            128,
            torch.float16,
            True,
            "raw_grouped_shape_has_no_calibration",
        ),
        (21, 2560, 160, 512, torch.float16, False, "requires_sm70"),
    ],
)
def test_rejected_descriptors_explain_fallback(
    operator, kind, k, n, experts, dtype, hardware, reason
):
    caps = raw_grouped_gate_up_capabilities(
        kind, k, n, experts, dtype, is_sm70=hardware
    )
    assert all(c.reason == reason for c in caps)


def test_missing_packaged_operator_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace())
    caps = raw_grouped_gate_up_capabilities(
        21, 2560, 160, 512, torch.float16, is_sm70=True
    )
    assert all(c.reason.startswith("operator_missing:") for c in caps)
