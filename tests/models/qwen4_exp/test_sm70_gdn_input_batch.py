# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN batch input: same half boundaries and unchanged M1/prefill fallback."""

import pytest
import torch

import vllm.envs as envs
from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
    Qwen38SM70FP16LinearMethod,
    _can_use_packed_gdn_input,
    _pack_gdn_input_weight,
    _qwen38_sm70_fp16_gdn_input,
    _qwen38_sm70_fp16_gdn_input_fake,
)


def test_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_QWEN38_GDN_INPUT_BATCH", raising=False)
    assert not envs.VLLM_SM70_QWEN38_GDN_INPUT_BATCH


@pytest.mark.parametrize("n", (24, 4096))
def test_packing_preserves_half_bits(n):
    raw = torch.randint(-(2**15), 2**15, (n, 2560), dtype=torch.int16)
    packed = _pack_gdn_input_weight(raw.view(torch.float16))
    restored = packed.permute(0, 3, 1, 2, 4).contiguous().view(-1, 2560)
    assert torch.equal(restored[:n].view(torch.int16), raw)
    if n == 24:
        assert not torch.count_nonzero(restored[n:].view(torch.int16))


def test_bad_pack_shape_rejected():
    with pytest.raises(ValueError):
        _pack_gdn_input_weight(torch.empty(25, 2560, dtype=torch.float16))
    with pytest.raises(ValueError):
        _pack_gdn_input_weight(torch.empty(24, 2560))


def test_cpu_and_missing_pack_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QWEN38_GDN_INPUT_BATCH", "1")
    assert not _can_use_packed_gdn_input(torch.empty(8, 2560), None, None)


@pytest.mark.parametrize("m", (1, 2, 16, 17, 8192))
def test_fake_shapes(m):
    x = torch.empty(m, 2560, device="meta", dtype=torch.float16)
    outputs = _qwen38_sm70_fp16_gdn_input_fake(x, x, x, x, x)
    assert [tuple(o.shape) for o in outputs] == [(m, n) for n in (2560, 1536, 12, 12)]


@pytest.fixture(scope="module")
def cuda_weights():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    if not hasattr(torch.ops._C, "qwen38_gdn_input_batch_sm70_out"):
        pytest.fail("Rebuild native SM70 op before running GPU tests")
    torch.manual_seed(20926)
    q = torch.randn(4096, 2560, device="cuda", dtype=torch.float16) * 0.02
    b = torch.randn(24, 2560, device="cuda", dtype=torch.float16) * 0.02
    return q, b, _pack_gdn_input_weight(q), _pack_gdn_input_weight(b)


@pytest.mark.parametrize("m", range(2, 17))
def test_dynamic_graph_bitwise(cuda_weights, monkeypatch, m):
    monkeypatch.setenv("VLLM_SM70_QWEN38_GDN_INPUT_BATCH", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    q, b, pq, pb = cuda_weights
    x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
    expected = _qwen38_sm70_fp16_gdn_input(x, q, b)
    _qwen38_sm70_fp16_gdn_input(x, q, b, pq, pb)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = torch.ops.vllm.qwen38_sm70_fp16_gdn_input(x, q, b, pq, pb)
    for scale in (0.0, 0.001, 0.1, 1.0, 3.0):
        x.normal_(0, scale)
        for output in actual:
            output.fill_(float("nan"))
        graph.replay()
        expected = _qwen38_sm70_fp16_gdn_input(x, q, b)
        for a, e in zip(actual, expected):
            assert torch.equal(a.view(torch.int16), e.view(torch.int16))


@pytest.mark.parametrize("m", (1, 17))
def test_unsupported_batch_falls_back(cuda_weights, monkeypatch, m):
    monkeypatch.setenv("VLLM_SM70_QWEN38_GDN_INPUT_BATCH", "1")
    q, b, pq, pb = cuda_weights
    x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
    assert not _can_use_packed_gdn_input(x, pq, pb)
    for actual, expected in zip(
        _qwen38_sm70_fp16_gdn_input(x, q, b, pq, pb),
        _qwen38_sm70_fp16_gdn_input(x, q, b),
    ):
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("which", ("x", "q", "b"))
def test_unaligned_storage_falls_back(cuda_weights, monkeypatch, which):
    monkeypatch.setenv("VLLM_SM70_QWEN38_GDN_INPUT_BATCH", "1")
    q, b, pq, pb = cuda_weights
    x = torch.randn(2, 2560, device="cuda", dtype=torch.float16)
    tensors = dict(x=x, q=pq, b=pb)
    original = tensors[which]
    storage = original.new_empty(original.numel() + 1)
    tensors[which] = storage[1:].view_as(original).copy_(original)
    x, pq, pb = (tensors[k] for k in ("x", "q", "b"))
    assert not _can_use_packed_gdn_input(x, pq, pb)
    for actual, expected in zip(
        _qwen38_sm70_fp16_gdn_input(x, q, b, pq, pb),
        _qwen38_sm70_fp16_gdn_input(x, q, b),
    ):
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("m", (2, 7, 9, 16))
def test_native_output_canaries(cuda_weights, m):
    q, b, pq, pb = cuda_weights
    x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
    storage = [x.new_full((m * n + 16,), 17) for n in (2560, 1536, 12, 12)]
    outputs = [s[8:-8].view(m, n) for s, n in zip(storage, (2560, 1536, 12, 12))]
    torch.ops._C.qwen38_gdn_input_batch_sm70_out(*outputs, x, pq, pb)
    for s in storage:
        assert torch.all(s[:8] == 17) and torch.all(s[-8:] == 17)
    for actual, expected in zip(outputs, _qwen38_sm70_fp16_gdn_input(x, q, b)):
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_native_rejects_wrong_output_geometry(cuda_weights):
    _, _, pq, pb = cuda_weights
    x = torch.zeros(2, 2560, device="cuda", dtype=torch.float16)
    outputs = [x.new_empty((2, n)) for n in (2560, 1536, 12, 13)]
    with pytest.raises(RuntimeError, match="output geometry"):
        torch.ops._C.qwen38_gdn_input_batch_sm70_out(*outputs, x, pq, pb)


def test_loader_prepares_nonpersistent_and_reloadable_bits(cuda_weights):
    _, b, _, _ = cuda_weights
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(b.clone(), requires_grad=False)
    )
    method = Qwen38SM70FP16LinearMethod()
    method.process_weights_after_loading(layer)
    assert not hasattr(layer, "_sm70_qwen38_gdn_packed")
    layer._sm70_qwen38_prepare_gdn_batch = True
    for scale in (1.0, 0.5):
        layer.weight.data.copy_(b * scale)
        method.process_weights_after_loading(layer)
        expected = _pack_gdn_input_weight(layer.weight)
        actual = layer._sm70_qwen38_gdn_packed
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert "_sm70_qwen38_gdn_packed" not in layer.state_dict()


def test_disabled_and_batch_invariant_fall_back(cuda_weights, monkeypatch):
    q, b, pq, pb = cuda_weights
    x = torch.randn(2, 2560, device="cuda", dtype=torch.float16)
    for enabled, invariant in (("0", "0"), ("1", "1")):
        monkeypatch.setenv("VLLM_SM70_QWEN38_GDN_INPUT_BATCH", enabled)
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", invariant)
        assert not _can_use_packed_gdn_input(x, pq, pb)
        for actual, expected in zip(
            _qwen38_sm70_fp16_gdn_input(x, q, b, pq, pb),
            _qwen38_sm70_fp16_gdn_input(x, q, b),
        ):
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
