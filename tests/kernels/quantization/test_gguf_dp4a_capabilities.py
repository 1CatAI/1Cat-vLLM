# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.gguf import (
    dp4a_expert_capabilities,
    q8_intermediate_expert_capabilities,
)


@pytest.mark.parametrize("supported", [False, True])
def test_routed_q8_requires_packaged_protocol(monkeypatch, supported):
    packet = SimpleNamespace(
        _schemas={"": "lanes_per_row" if supported else "activated"}
    )
    monkeypatch.setattr(
        torch.ops,
        "_C",
        SimpleNamespace(
            gguf_quantize_q8_1_sm70_out=object(),
            gguf_dp4a_gate_up_sm70_out=packet,
            gguf_dp4a_down_unroute_sm70_out=object(),
        ),
    )
    caps = q8_intermediate_expert_capabilities(
        21, 20, 2560, 160, 512, torch.float16, is_sm70=True
    )
    assert all(
        c.reason == (None if supported else "routed_q8_operator_protocol_unavailable")
        for c in caps
    )
    disabled = q8_intermediate_expert_capabilities(
        21, 20, 2560, 160, 512, torch.float16, is_sm70=True, enabled=False
    )
    assert all(c.reason == "disabled_by_kernel_config" for c in disabled)


@pytest.fixture
def packaged(monkeypatch):
    operators = {
        name: object()
        for name in (
            "gguf_quantize_q8_1_sm70_out",
            "gguf_dp4a_gate_up_sm70_out",
            "gguf_dp4a_lut4_gate_up_sm70_out",
            "gguf_dp4a_down_unroute_sm70_out",
        )
    }
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace(**operators))


@pytest.mark.parametrize("gate", [18, 21, 22])
@pytest.mark.parametrize("down", [20, 42])
def test_only_calibrated_original_m_is_admitted(packaged, gate, down):
    caps = dp4a_expert_capabilities(
        gate, down, 2560, 160, 512, torch.float16, is_sm70=True
    )
    admitted = [
        m
        for m in (1, 2, 4, 5, 8, 10, 16, 20, 32, 50, 200)
        if any(c.reason is None and c.supports_m(m) for c in caps)
    ]
    assert admitted == [5, 20]
    assert all(c.graph_safe for c in caps)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"enabled": False}, "disabled_by_kernel_config"),
        ({"is_sm70": False}, "requires_sm70"),
        ({"dtype": torch.bfloat16}, "requires_fp16_activations"),
        ({"source_type": 23}, "dp4a_expert_source_formats_unavailable"),
        ({"down_type": 14}, "dp4a_expert_source_formats_unavailable"),
        ({"n": 320}, "dp4a_expert_shape_has_no_calibration"),
        ({"k": 5120}, "dp4a_expert_shape_has_no_calibration"),
        ({"original_storage_available": False}, "original_expert_bank_not_retained"),
    ],
)
def test_fallback_reasons(packaged, changes, reason):
    parameters = dict(
        source_type=21,
        down_type=42,
        k=2560,
        n=160,
        num_experts=512,
        dtype=torch.float16,
        is_sm70=True,
    )
    parameters.update(changes)
    caps = dp4a_expert_capabilities(**parameters)
    assert all(c.reason == reason for c in caps)


@pytest.mark.parametrize(
    "name",
    [
        "gguf_quantize_q8_1_sm70_out",
        "gguf_dp4a_gate_up_sm70_out",
        "gguf_dp4a_down_unroute_sm70_out",
    ],
)
def test_missing_packaged_component_rejects_entire_route(packaged, monkeypatch, name):
    monkeypatch.delattr(torch.ops._C, name)
    caps = dp4a_expert_capabilities(21, 42, 2560, 160, 512, torch.float16, is_sm70=True)
    assert all(c.reason == f"operator_missing:{name}" for c in caps)


@pytest.mark.parametrize("source", [20, 23])
def test_iq4_requires_qualified_canonical_banks(packaged, monkeypatch, source):
    parameters = dict(
        source_type=source,
        down_type=20,
        k=2560,
        n=160,
        num_experts=512,
        dtype=torch.float16,
        is_sm70=True,
        original_storage_available=False,
    )
    rejected = dp4a_expert_capabilities(**parameters)
    assert all(
        c.reason == "requires_iq4_codebook_zero_group32_banks_without_ep"
        for c in rejected
    )
    caps = dp4a_expert_capabilities(**parameters, canonical_storage_available=True)
    assert [
        m
        for m in (1, 5, 10, 20, 32)
        if any(c.reason is None and c.supports_m(m) for c in caps)
    ] == [5, 20]
    monkeypatch.delattr(torch.ops._C, "gguf_dp4a_lut4_gate_up_sm70_out")
    rejected = dp4a_expert_capabilities(**parameters, canonical_storage_available=True)
    assert all(
        c.reason == "operator_missing:gguf_dp4a_lut4_gate_up_sm70_out" for c in rejected
    )
