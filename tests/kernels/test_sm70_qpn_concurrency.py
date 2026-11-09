# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused concurrent-row validation; optionally load a fresh source sidecar."""

import os
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires owned SM70")
    import vllm._C  # noqa: F401

    library = os.getenv("VLLM_SM70_QPN_AUDIT_LIBRARY")
    if library:
        torch.ops.load_library(library)
    fp8 = torch.ops._C_qwen38 if library else torch.ops._C
    nv = torch.ops._qpn2_candidate if library else torch.ops._C
    return SimpleNamespace(
        fp8=fp8,
        prepare=nv.prepare if library else nv.nvfp4_qpn2_prepare_sm70,
        gemm=nv.gemm if library else nv.nvfp4_qpn2_gemm_sm70_out,
        gated=nv.gated if library else nv.nvfp4_qpn2_gated_sm70_out,
    )


def _graph_replay(call, x, actual, reference):
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for scale in (0.1, 0.5, 1.0):
        x.normal_(0, scale)
        actual.fill_(float("nan"))
        graph.replay()
        expected = reference()
        assert torch.isfinite(actual).all()
        error = (actual.double() - expected.double()).norm()
        assert error / expected.double().norm().clamp_min(1e-12) < 0.005


@pytest.mark.parametrize("rows", [9, 16, 17, 24, 31, 32, 33])
@pytest.mark.parametrize("gated", [False, True])
def test_fp8_dispatch_graph_and_fp64(native, monkeypatch, rows, gated):
    for flag in ("M16", "M32_CHUNKED", "M32_NATIVE"):
        monkeypatch.delenv("VLLM_SM70_FP8_QPN8_" + flag, raising=False)
    torch.manual_seed(187)
    n, k = (8704, 5120) if gated else (5120, 1536)
    weight = (torch.randn(n, k, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    scales = torch.full((n, 1), 0.125, device="cuda")
    packed, packed_scales = native.fp8.fp8_qpn8_prepare_sm70(weight, scales)
    dense = (weight.half() * scales.half()).double()
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    actual = torch.empty(rows, n // 2 if gated else n, device="cuda", dtype=x.dtype)
    workspace = torch.empty(k, n, device="cuda", dtype=x.dtype)
    split = 8 if gated else 12

    def call():
        native.fp8.fp8_qpn8_dispatch_sm70_out(
            actual,
            workspace.data_ptr(),
            x,
            packed,
            packed_scales,
            split,
            2,
            False,
            gated,
        )

    def reference():
        raw = x.double() @ dense.T
        if gated:
            gate, up = raw.half().double().chunk(2, dim=-1)
            return gate.sigmoid() * gate * up
        return raw

    _graph_replay(call, x, actual, reference)


@pytest.mark.parametrize("rows", [9, 16, 17, 32])
@pytest.mark.parametrize("gated", [False, True])
def test_nvfp4_preserves_rollback_and_fp64(native, monkeypatch, rows, gated):
    torch.manual_seed(781)
    n, k = 256, 512
    weight = torch.randint(256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.full((n, k // 16), 0.5, device="cuda").to(torch.float8_e4m3fn)
    packed, packed_scales = native.prepare(weight, scales)
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.float64,
    )
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).reshape(n, k)
    dense = table[codes.long()] * scales.double().repeat_interleave(16, dim=1)
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    actual = torch.empty(rows, n // 2 if gated else n, device="cuda", dtype=x.dtype)
    op = native.gated if gated else native.gemm

    def call():
        op(actual, x, packed, packed_scales, 0.125, 8, 2)

    def reference():
        raw = x.double() @ dense.T * 0.125
        if gated:
            gate, up = raw.half().double().chunk(2, dim=-1)
            return (gate.sigmoid() * gate).half().double() * up
        return raw

    monkeypatch.delenv("VLLM_SM70_NVFP4_QPN2_M16_NATIVE", raising=False)
    _graph_replay(call, x, actual, reference)
    baseline = actual.clone()
    monkeypatch.setenv("VLLM_SM70_NVFP4_QPN2_M16_NATIVE", "0")
    call()
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)


def test_fp8_native_plan_mismatch_uses_supported_fallback(native, monkeypatch):
    # A split-8 caller was valid before M32 became a default; keep it valid.
    n, k, rows = 4096, 5120, 24
    weight = torch.zeros(n, k, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.ones(n, 1, device="cuda")
    packed, packed_scales = native.fp8.fp8_qpn8_prepare_sm70(weight, scales)
    x = torch.ones(rows, k, device="cuda", dtype=torch.float16)
    output = torch.full((rows, n), float("nan"), device="cuda", dtype=x.dtype)
    workspace = torch.empty(k, n, device="cuda", dtype=x.dtype)
    for flag in ("M16", "M32_CHUNKED", "M32_NATIVE"):
        monkeypatch.delenv("VLLM_SM70_FP8_QPN8_" + flag, raising=False)
    for split in (8, 32):
        native.fp8.fp8_qpn8_dispatch_sm70_out(
            output,
            workspace.data_ptr(),
            x,
            packed,
            packed_scales,
            split,
            2,
            False,
            False,
        )
        assert torch.equal(output, torch.zeros_like(output))
