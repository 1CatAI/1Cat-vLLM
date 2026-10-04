# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MTP shared expert retains all three FP16 rounding boundaries."""

import pytest
import torch

import vllm.envs as envs
from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as batch


@pytest.mark.parametrize("rows", [1, 2, 5, 10, 17])
@pytest.mark.parametrize("enabled", [False, True])
def test_changed_input_graph(rows, enabled, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires SM70")
    if not hasattr(torch.ops._C, "qwen38_shared_up_batch_sm70_out"):
        pytest.skip("Requires source build with shared expert batch kernels")
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    assert not torch.backends.cuda.matmul.allow_fp16_accumulation
    monkeypatch.setenv("VLLM_SM70_MTP_SHARED_BATCH", str(int(enabled)))
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    envs.disable_envs_cache()
    torch.manual_seed(20260927)
    weight = torch.randn(320, 2560, device="cuda", dtype=torch.float16) * 0.03
    packed = weight.view(10, 32, 160, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
    logits = torch.randn(rows, 1, device="cuda", dtype=torch.float16)
    raw = x.new_empty((rows, 320))
    expected = x.new_empty((rows, 160))

    def run():
        torch.mm(x, weight.t(), out=raw)
        torch.ops._C.silu_and_mul(expected, raw)
        actual = batch._qwen38_sm70_shared_up(x, weight, packed)
        gated = batch._qwen38_sm70_shared_gate_mul(logits, x)
        return actual, gated, torch.sigmoid(logits) * x

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, gated, expected_gate = run()
    try:
        for scale in (0.0, 0.001, 0.03, 0.1, 1.0, 3.0, 30.0):
            x.normal_(0, scale)
            logits.normal_(0, scale)
            graph.replay()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            assert torch.equal(gated.view(torch.int16), expected_gate.view(torch.int16))
    finally:
        envs.disable_envs_cache()


@pytest.mark.parametrize("rows", [5, 10])
def test_fp32_partials_changed_graph_with_exact_linear_oracle(rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires SM70")
    if not hasattr(torch.ops._C, "qwen38_shared_up_batch_fp32_sm70_out"):
        pytest.skip("Requires rebuilt FP32 shared operator")
    coefficients = (torch.arange(320, device="cuda") % 16 - 8).half()
    weight = coefficients[:, None].expand(320, 2560).contiguous() / 4096
    packed = weight.view(10, 32, 160, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
    x = torch.ones(rows, 2560, device="cuda", dtype=torch.float16)
    partials = torch.empty(8, rows, 320, device="cuda", dtype=torch.float32)
    actual = torch.empty(rows, 160, device="cuda", dtype=torch.float16)
    expected = torch.empty_like(actual)
    oracle = torch.empty(rows, 320, device="cuda", dtype=torch.float16)
    for _ in range(3):
        torch.ops._C.qwen38_shared_up_batch_fp32_sm70_out(actual, partials, x, packed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.qwen38_shared_up_batch_fp32_sm70_out(actual, partials, x, packed)
    for factor in (0, 1, 2, -1):
        x.fill_(factor)
        # The projection is exactly representable for these integer inputs;
        # equality here tests layout/graph writes, not model bitwise admission.
        oracle.copy_((coefficients * (0.625 * factor)).expand(rows, 320))
        torch.ops._C.silu_and_mul(expected, oracle)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
