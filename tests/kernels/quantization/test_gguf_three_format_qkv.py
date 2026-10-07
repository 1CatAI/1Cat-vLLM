# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured QKV admission and the joint three-decoder graph contract."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels import gguf as kernel


@pytest.mark.parametrize("kinds", [(21, 23, 12), (18, 23, 12), (12, 23, 21)])
def test_measured_three_format_admission(monkeypatch, kinds):
    monkeypatch.setattr(
        torch,
        "ops",
        SimpleNamespace(
            _C=SimpleNamespace(gguf_dmv_three_formats_sm70_supported=lambda: True)
        ),
    )
    caps = kernel.three_format_qkv_capabilities(
        kinds, 5120, (3072, 256, 256), torch.float16
    )
    assert all(c.reason is None and c.graph_safe for c in caps)
    assert all(
        c.supports_m(8) and not c.supports_m(7) and not c.supports_m(9) for c in caps
    )


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"enabled": False}, "three_format_qkv_disabled_by_kernel_config"),
        ({"k": 4096}, "three_format_qkv_shape_or_sources_unmeasured"),
        ({"widths": (3072, 512, 256)}, "three_format_qkv_shape_or_sources_unmeasured"),
        (
            {"source_types": (23, 21, 12)},
            "three_format_qkv_shape_or_sources_unmeasured",
        ),
        ({"dtype": torch.bfloat16}, "three_format_qkv_requires_fp16_activations"),
        ({"compute_capability": 80}, "three_format_qkv_requires_sm70"),
    ],
)
def test_unmeasured_three_format_rejected(monkeypatch, change, reason):
    monkeypatch.setattr(
        torch,
        "ops",
        SimpleNamespace(
            _C=SimpleNamespace(gguf_dmv_three_formats_sm70_supported=lambda: True)
        ),
    )
    args = dict(
        source_types=(21, 23, 12), k=5120, widths=(3072, 256, 256), dtype=torch.float16
    )
    args.update(change)
    assert {c.reason for c in kernel.three_format_qkv_capabilities(**args)} == {reason}


@pytest.mark.parametrize("available", [False, None])
def test_older_artifact_rejected(monkeypatch, available):
    namespace = SimpleNamespace()
    if available is not None:
        namespace.gguf_dmv_three_formats_sm70_supported = lambda: available
    monkeypatch.setattr(torch, "ops", SimpleNamespace(_C=namespace))
    caps = kernel.three_format_qkv_capabilities(
        (21, 23, 12), 5120, (3072, 256, 256), torch.float16
    )
    expected = (
        "native_three_format_support_unavailable"
        if available is False
        else "operator_missing:gguf_dmv_three_formats_sm70_supported"
    )
    assert {c.reason for c in caps} == {expected}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kinds", [(21, 23, 12), (18, 23, 12), (12, 23, 21)])
@pytest.mark.parametrize("split", [1, 2])
def test_three_format_graph_outputs(kinds, split):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    from tests.kernels.quantization.test_gguf_dmv import planes
    from vllm.model_executor.layers.quantization.gguf_dmv import table

    segments = [planes(kind, k=5120) for kind in kinds]
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * 0.125
    result = torch.empty(8, 192, device="cuda", dtype=torch.float16)
    out = list(result.split(64, dim=1))
    ws = torch.empty(8192, device="cuda", dtype=torch.float32)
    counters = torch.zeros(3, device="cuda", dtype=torch.int32)

    def call():
        torch.ops._C.gguf_dmv_sm70_out(
            x,
            [s[1][0] for s in segments],
            [s[1][1] for s in segments],
            [s[1][2] for s in segments],
            out,
            [s[0] for s in segments],
            [64] * 3,
            5120,
            split,
            4,
            ws,
            counters,
            2,
            None,
            table(x.device),
            None,
            None,
            None,
            False,
        )

    call()
    reference = result.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        graph.replay()
        assert torch.equal(reference.view(torch.int16), result.view(torch.int16))
        assert not counters.any()
    expected = torch.cat(
        [(x.double() @ s[2].half().double().T).half() for s in segments], dim=1
    )
    relative = (result.float() - expected.float()).norm() / expected.float().norm()
    assert relative < 0.003
