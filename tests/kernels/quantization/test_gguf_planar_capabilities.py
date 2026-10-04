# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.kernels.gguf import planar_lattice_capabilities


@pytest.mark.parametrize("kind", [21, 22])
def test_planar_admits_only_measured_faster_expert_m512(monkeypatch, kind):
    monkeypatch.setattr(
        torch.ops, "_C", SimpleNamespace(gguf_lattice_planar_gemm_sm70_out=object())
    )
    caps = planar_lattice_capabilities(kind, 2560, 160, torch.float16, is_sm70=True)
    admitted = [
        m
        for m in (1, 5, 8, 16, 32, 128, 512, 1024)
        if any(c.reason is None and c.supports_m(m) for c in caps)
    ]
    assert admitted == [512]
    dense = planar_lattice_capabilities(21, 2560, 1536, torch.float16, is_sm70=True)
    assert dense[0].reason == "measured_slower_than_canonical_gemm"
    unknown = planar_lattice_capabilities(21, 768, 160, torch.float16, is_sm70=True)
    assert unknown[0].reason == "planar_shape_has_no_calibration"


def test_planar_missing_operator_and_descriptor_reasons(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace())
    kwargs = dict(source_type=21, k=2560, n=160, dtype=torch.float16, is_sm70=True)
    assert planar_lattice_capabilities(**kwargs)[0].reason.startswith(
        "operator_missing:"
    )
    cases: list[tuple[dict[str, Any], str]] = [
        ({"enabled": False}, "disabled_by_kernel_config"),
        ({"is_sm70": False}, "requires_sm70"),
        ({"source_type": 18}, "raw_source_format_unavailable"),
        ({"dtype": torch.float32}, "requires_fp16_activations"),
        ({"k": 160}, "raw_shape_cuts_source_block_or_output_pack"),
        ({"n": 65536}, "planar_descriptor_exceeds_dimension_encoding"),
    ]
    for override, reason in cases:
        assert planar_lattice_capabilities(**(kwargs | override))[0].reason == reason
