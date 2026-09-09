# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.minimax_h3.attention import (
    Attention,
    AttentionMetadata,
    attention_backend,
    chunked_attention_reference,
)
from vllm.model_executor.models.minimax_h3.cuda_ops import flashattn_extension

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires SM70 GPU"
)


def test_flashattn_rejects_batch_head_grid_overflow():
    q = torch.empty(1, 1, 65536, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="index limits"):
        flashattn_extension().forward(q, q, q, 128**-0.5)


@pytest.mark.parametrize(
    "length", [1, 31, 32, 33, 63, 64, 65, 96, 97, 127, 128, 129, 12323]
)
@pytest.mark.parametrize("key_tile", [64, 128])
def test_flashattn_d128_mha_tails_and_online_rescaling(length, key_tile):
    torch.manual_seed(42)
    q, k, v = [
        torch.randn(2, length, 2, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    # A later key tile increases the maximum. This checks the distinct PV
    # accumulator mapping, including channels 64..127, across two batches.
    k[:, length // 2 :] *= 4
    rows = torch.linspace(0, length - 1, min(length, 65), device="cuda").long()
    expected = chunked_attention_reference(q[:, rows], k, v, scale=128**-0.5)
    actual = flashattn_extension().forward(q, k, v, 128**-0.5, key_tile)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[:, rows], expected, atol=0.002, rtol=0.03)


@pytest.mark.parametrize("layout", ["offset", "strided"])
@pytest.mark.parametrize("key_tile", [64, 128])
def test_flashattn_d128_storage_and_different_q_k_lengths(layout, key_tile):
    torch.manual_seed(42)
    tensors = []
    for length, offset in [(33, 1), (65, 3), (65, 5)]:
        if layout == "offset":
            value = torch.randn(
                2 * length * 2 * 128 + offset, device="cuda", dtype=torch.float16
            )[offset:].reshape(2, length, 2, 128)
        else:
            value = torch.randn(
                2, length * 2, 2, 128, device="cuda", dtype=torch.float16
            )[:, ::2]
        tensors.append(value)
    q, k, v = tensors
    expected = chunked_attention_reference(q, k, v, scale=128**-0.5)
    actual = flashattn_extension().forward(q, k, v, 128**-0.5, key_tile)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.03)


def test_flashattn_dispatch_slices_poisoned_padding(monkeypatch):
    from vllm.model_executor.models.minimax_h3 import cuda_ops

    native = flashattn_extension()
    calls = []

    class TracedNative:
        def forward(self, q, k, v, scale):
            calls.append(tuple(q.shape))
            return native.forward(q, k, v, scale)

    monkeypatch.setattr(cuda_ops, "flashattn_extension", lambda: TracedNative())
    token = attention_backend.set("FLASH_ATTN_V100")
    try:
        module = Attention(
            num_heads=14, num_kv_heads=14, head_size=128, softmax_scale=128**-0.5
        )
    finally:
        attention_backend.reset(token)
    q, k, v = [
        torch.randn(1, 257, 14, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    for tensor in (q, k, v):
        tensor[:, 243:] = float("nan")
    expected = chunked_attention_reference(
        q[:, :243], k[:, :243], v[:, :243], scale=128**-0.5
    )
    actual = module(q, k, v, AttentionMetadata(extra={"valid_kv_length": 243}))
    assert calls == [(1, 243, 14, 128)]
    torch.testing.assert_close(actual[:, :243], expected, atol=0.002, rtol=0.03)
    assert torch.count_nonzero(actual[:, 243:]) == 0


@pytest.mark.parametrize("key_tile", [64, 128])
def test_flashattn_graph_replay_uses_new_values(key_tile):
    ops = flashattn_extension()
    q, k, v = [
        torch.randn(1, 129, 2, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            ops.forward(q, k, v, 128**-0.5, key_tile)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ops.forward(q, k, v, 128**-0.5, key_tile)
    for _ in range(2):
        q.normal_()
        k.normal_()
        v.normal_()
        graph.replay()
        expected = chunked_attention_reference(q, k, v, scale=128**-0.5)
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.03)


def test_flashattn_fixed_shape_graph_replay():
    ops = flashattn_extension()
    q, k, v = [
        torch.randn(1, 12323, 14, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            ops.forward(q, k, v, 128**-0.5)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ops.forward(q, k, v, 128**-0.5)
    rows = torch.tensor([0, 31, 32, 63, 64, 127, 128, 12322], device="cuda")
    for _ in range(2):
        v.normal_()
        graph.replay()
        expected = chunked_attention_reference(q[:, rows], k, v, scale=128**-0.5)
        torch.testing.assert_close(actual[:, rows], expected, atol=0.002, rtol=0.03)


def test_flashattn_rejects_changed_head_dimension_and_scale():
    ops = flashattn_extension()
    q = torch.randn(1, 32, 2, 256, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="D128 MHA"):
        ops.forward(q, q, q, 256**-0.5)
    q = q[..., :128].contiguous()
    with pytest.raises(RuntimeError, match="finite and positive"):
        ops.forward(q, q, q, float("nan"))
    with pytest.raises(RuntimeError, match="key tile"):
        ops.forward(q, q, q, 128**-0.5, 32)
