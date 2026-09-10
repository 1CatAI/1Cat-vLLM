# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.minimax_h3.ops import (
    fused_qk_norm_rope,
    qk_norm_rope_reference,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length,heads", [(1, 2), (17, 3), (129, 14)])
@pytest.mark.parametrize("rotary", [96, 128])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
@torch.inference_mode()
def test_qk_norm_rope_rounding_and_strided_qkv(length, heads, rotary, weight_dtype):
    torch.manual_seed(42)
    qkv = torch.randn(length, heads * 128 * 3, device="cuda", dtype=torch.float16)
    q, k, _ = [x.view(length, heads, 128) for x in qkv.chunk(3, -1)]
    q[0, 0] = 0
    k[0, 0, :4] = torch.tensor([65504, -65504, 1e-5, -1e-5], device="cuda")
    weights = [torch.randn(128, device="cuda", dtype=weight_dtype) for _ in range(2)]
    angle = torch.randn(length, rotary // 2, device="cuda")
    rope = torch.cat((angle.cos(), angle.sin()), -1)
    expected = qk_norm_rope_reference(q, k, *weights, rope, 1e-5)
    actual = fused_qk_norm_rope(q, k, *weights, rope, 1e-5)
    for got, reference in zip(actual, expected):
        torch.testing.assert_close(got, reference, atol=0, rtol=0)
        assert got.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_qk_norm_rope_graph_replays_changed_input():
    q = torch.randn(17, 3, 128, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    weight = torch.ones(128, device="cuda")
    rope = torch.randn(17, 96, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused_qk_norm_rope(q, k, weight, weight, rope, 1e-5)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused_qk_norm_rope(q, k, weight, weight, rope, 1e-5)
    q.normal_()
    k.normal_()
    graph.replay()
    expected = qk_norm_rope_reference(q, k, weight, weight, rope, 1e-5)
    for got, reference in zip(actual, expected):
        torch.testing.assert_close(got, reference, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qk_norm_rope_preserves_autograd_reference():
    with torch.enable_grad():
        q = torch.randn(
            3, 2, 128, device="cuda", dtype=torch.float16, requires_grad=True
        )
        k = torch.randn_like(q, requires_grad=True)
        weight = torch.ones(128, device="cuda", requires_grad=True)
        rope = torch.randn(3, 96, device="cuda")
        actual = fused_qk_norm_rope(q, k, weight, weight, rope, 1e-5)
        assert all(value.requires_grad for value in actual)
        sum(value.float().square().mean() for value in actual).backward()
        for value in (q, k, weight):
            assert value.grad is not None and torch.isfinite(value.grad).all()


# Additional dependency-branch amplitude and long-shape coverage.
def inputs(tokens, heads, amplitude, weight_dtype):
    torch.manual_seed(42)
    # Split projection views have a non-contiguous token stride and offsets.
    storage = torch.randn(tokens, 3, heads, 128, device="cuda", dtype=torch.float16)
    storage *= amplitude
    q, k, _ = storage.unbind(1)
    q[0].zero_()
    weights = torch.randn(2, 128, device="cuda", dtype=weight_dtype)
    angles = torch.randn(tokens, 48, device="cuda")
    rope = torch.cat((angles.cos(), angles.sin()), -1).half()
    return q, k, *weights, rope, 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens,heads", [(1, 1), (65, 3), (129, 14), (12323, 14)])
@pytest.mark.parametrize(
    "amplitude,weight_dtype", [(1, torch.float16), (2000, torch.float32)]
)
@torch.inference_mode()
def test_rounding_and_partial_rotation(tokens, heads, amplitude, weight_dtype):
    args = inputs(tokens, heads, amplitude, weight_dtype)
    expected = qk_norm_rope_reference(*args)
    actual = fused_qk_norm_rope(*args)
    for result, reference in zip(actual, expected):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_graph_replay_reads_changed_inputs_and_rope():
    args = inputs(65, 3, 1000, torch.float16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused_qk_norm_rope(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused_qk_norm_rope(*args)
    for _ in range(2):
        args[0].normal_()
        args[1].normal_()
        args[4].neg_()
        graph.replay()
        expected = qk_norm_rope_reference(*args)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_other_rotary_width_keeps_reference_path():
    args = list(inputs(65, 3, 1, torch.float16))
    args[4] = args[4][:, :64]
    actual = fused_qk_norm_rope(*args)
    expected = qk_norm_rope_reference(*args)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
