# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from vllm.config.vllm import (
    _SM70_BATCH_GEMM_DEFAULTS,
    _apply_sm70_batch_gemm_defaults,
)
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a16_fp8,  # noqa: F401
)


def test_batched_layout_policy_does_not_require_a_service_contract(monkeypatch):
    def fail_config_lookup():
        raise AssertionError("batch layouts must not require a model/service whitelist")

    monkeypatch.setattr(sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr("vllm.config.get_current_vllm_config", fail_config_lookup)
    for name in _SM70_BATCH_GEMM_DEFAULTS:
        monkeypatch.delenv(name, raising=False)

    assert set(_apply_sm70_batch_gemm_defaults(is_sm70=True)) == set(
        _SM70_BATCH_GEMM_DEFAULTS
    )

    assert sm70_tm.use_batched_gemm_layouts()
    monkeypatch.setenv("VLLM_SM70_BATCH_GEMM_LAYOUTS", "0")
    assert not sm70_tm.use_batched_gemm_layouts()


@pytest.mark.parametrize("overridden_name", _SM70_BATCH_GEMM_DEFAULTS)
def test_batch_defaults_preserve_explicit_overrides(monkeypatch, overridden_name):
    for name in _SM70_BATCH_GEMM_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(overridden_name, "0")

    applied = _apply_sm70_batch_gemm_defaults(is_sm70=True)

    assert overridden_name not in applied
    assert os.environ[overridden_name] == "0"
    assert set(applied) == set(_SM70_BATCH_GEMM_DEFAULTS) - {overridden_name}
    for name in applied:
        assert os.environ[name] == _SM70_BATCH_GEMM_DEFAULTS[name]
    assert _apply_sm70_batch_gemm_defaults(is_sm70=True) == ()


def test_batch_defaults_and_layouts_reject_non_sm70(monkeypatch):
    for name in _SM70_BATCH_GEMM_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    assert _apply_sm70_batch_gemm_defaults(is_sm70=False) == ()
    assert all(name not in os.environ for name in _SM70_BATCH_GEMM_DEFAULTS)

    monkeypatch.setenv("VLLM_SM70_BATCH_GEMM_LAYOUTS", "1")
    monkeypatch.setattr(sm70_tm, "is_exact_sm70_cuda_platform", lambda: False)
    assert not sm70_tm.use_batched_gemm_layouts()


@pytest.mark.parametrize(
    ("m", "k", "n", "split_k", "gated_silu", "tm_prescaled"),
    [
        (8, 1536, 5120, 12, False, False),
        (32, 5120, 3584, 16, False, False),
        (48, 1536, 5120, 12, False, True),
        (64, 1536, 5120, 12, False, False),
        (64, 1536, 5120, 12, False, True),
        (64, 5120, 8704, 8, True, False),
    ],
)
def test_batch_fp8_dispatch_matches_qpn8_and_replays(
    m, k, n, split_k, gated_silu, tm_prescaled
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA required")
    torch.manual_seed(20260924)
    weight = torch.randn((n, k), device="cuda").mul_(0.25).to(torch.float8_e4m3fn)
    scales = torch.full((n, 1), 0.125, device="cuda", dtype=torch.float32)
    codes, q_scales = torch.ops._C.fp8_qpn8_prepare_sm70(weight, scales)
    tm_weight, tm_scales, meta = torch.ops._C.fp8_sm70_prepare(
        weight, scales, 128, gated_silu
    )
    if tm_prescaled:
        tm_scales = tm_scales.mul(256)
        assert bool(torch.isfinite(tm_scales).all().item())
    x = torch.randn((m, k), device="cuda", dtype=torch.float16).mul_(0.1)
    expected = torch.empty(
        (m, n // 2 if gated_silu else n), device="cuda", dtype=torch.float16
    )
    actual = torch.empty_like(expected)
    workspace = torch.empty((k, n), device="cuda", dtype=torch.float16)

    def reference():
        torch.ops._C.fp8_qpn8_dispatch_sm70_out(
            expected,
            workspace.data_ptr(),
            x,
            codes,
            q_scales,
            split_k,
            2,
            False,
            gated_silu,
        )

    def candidate():
        torch.ops.vllm.sm70_ct_fp8_qpn8_batch_dispatch(
            actual,
            x,
            codes,
            q_scales,
            tm_weight,
            tm_scales,
            int(meta[0]),
            int(meta[1]),
            split_k,
            2,
            False,
            gated_silu,
            tm_prescaled,
        )

    candidate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    for _ in range(3):
        x.normal_(0, 0.1)
        reference()
        graph.replay()
        torch.accelerator.synchronize()
        if m <= 32:
            assert torch.equal(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
