# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP64 attention and exact E4M3 storage tests for the v37 prefill route."""

import pytest
import torch


@pytest.fixture
def ops():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    from vllm.v1.attention.backends.flash_attn_v100 import (
        _get_sm70_d256_gqa_architecture_op,
    )

    _get_sm70_d256_gqa_architecture_op()
    ns = torch.ops._vllm_fa2_C
    if not hasattr(ns, "sm70_d256_gqa_v37_fwd"):
        pytest.skip("rebuild FA2 with v37")
    return ns


@pytest.mark.parametrize(
    "q_len,kv_len",
    [
        (64, 96),
        (64, 128),
        (8192, 8224),
        (64, 3136),
        (384, 8192),
        (1600, 3136),
        (1600, 65536),
        (1664, 4864),
        (1600, 131072),
        (1600, 262144),
        (8000, 128000),
        (8192, 262144),
    ],
)
def test_v37_against_fp64(ops, q_len, kv_len):
    torch.manual_seed(20260907)
    q = torch.randn((1, q_len, 6, 256), dtype=torch.float16, device="cuda")
    k = torch.randn((1, kv_len, 1, 256), dtype=torch.float16, device="cuda")
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    ops.sm70_d256_gqa_v37_fwd(q, k, v, out, 0.0625, True)
    rows = torch.tensor(sorted({0, 1, q_len // 2, q_len - 1}), device="cuda")
    scores = q[0, rows].permute(1, 0, 2).double() @ k[0, :, 0].double().T * 0.0625
    scores.masked_fill_(
        torch.arange(kv_len, device="cuda")[None, None]
        > (kv_len - q_len + rows)[None, :, None],
        -torch.inf,
    )
    ref = (scores.softmax(-1) @ v[0, :, 0].double()).permute(1, 0, 2)
    assert bool(torch.isfinite(out).all())
    relative_l2 = (out[0, rows].double() - ref).norm() / ref.norm()
    assert float(relative_l2) < 0.001


def test_v37_leading_query_padding_preserves_causal_offset(ops):
    torch.manual_seed(20260907)
    length, count, padded = 3136, 1568, 1600
    q = torch.randn((1, padded, 6, 256), dtype=torch.float16, device="cuda")
    q[:, : padded - count].zero_()
    k = torch.randn((1, length, 1, 256), dtype=torch.float16, device="cuda")
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    ops.sm70_d256_gqa_v37_fwd(q, k, v, out, 0.0625, True)
    query = q[0, padded - count].double()
    # First real query must see length-count+1 keys, not all of the tail.
    ref = (query @ k[0, : length - count + 1, 0].double().T * 0.0625).softmax(-1)
    ref = ref @ v[0, : length - count + 1, 0].double()
    assert float((out[0, padded - count].double() - ref).norm() / ref.norm()) < 0.001


def test_v37_rejects_misaligned_contiguous_query(ops):
    raw = torch.empty(8000 * 6 * 256 + 1, dtype=torch.float16, device="cuda")
    q = raw[1:].view(1, 8000, 6, 256)
    k = torch.empty((1, 16000, 1, 256), dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        ops.sm70_d256_gqa_v37_fwd(q, k, k, torch.empty_like(q), 0.0625, True)


def test_v37_rejects_partial_k32_tile(ops):
    q = torch.empty((1, 1664, 6, 256), dtype=torch.float16, device="cuda")
    k = torch.empty((1, 4848, 1, 256), dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="32-token KV step"):
        ops.sm70_d256_gqa_v37_fwd(q, k, k, torch.empty_like(q), 0.0625, True)


def test_v37_shared_workspace_orders_different_streams(ops):
    torch.manual_seed(20260907)
    inputs = []
    for q_len, kv_len in ((384, 8192), (1600, 3136)):
        q = torch.randn((1, q_len, 6, 256), dtype=torch.float16, device="cuda")
        k = torch.randn((1, kv_len, 1, 256), dtype=torch.float16, device="cuda")
        v = torch.randn_like(k)
        ref, out = torch.empty_like(q), torch.empty_like(q)
        ops.sm70_d256_gqa_v37_fwd(q, k, v, ref, 0.0625, True)
        inputs.append((q, k, v, ref, out))
    torch.accelerator.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for _ in range(3):
        for stream, (q, k, v, ref, out) in zip(streams, inputs):
            with torch.cuda.stream(stream):
                ops.sm70_d256_gqa_v37_fwd(q, k, v, out, 0.0625, True)
    torch.accelerator.synchronize()
    for q, k, v, ref, out in inputs:
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("page", [800, 1568, 1616])
@pytest.mark.parametrize("scale", [1.0, 0.003, 3.14159])
def test_e4m3_bridge_all_codes_and_graph(ops, page, scale):
    cache = torch.empty((3, 2, page, 1, 256), dtype=torch.uint8, device="cuda")
    k, v = cache.unbind(1)
    codes = torch.arange(256, device="cuda", dtype=torch.int32).byte()
    k.copy_(codes)
    v.copy_(codes.flip(0))
    table = torch.tensor([[2, 0, 1]], device="cuda", dtype=torch.int32)
    seq = torch.tensor([3 * page - 5], device="cuda", dtype=torch.int32)
    output = torch.empty(
        ((3 * page + 31) // 32, 2, 32, 1, 256), dtype=torch.float16, device="cuda"
    )
    ko, vo = output.unbind(1)

    def call():
        ops.sm70_v37_e4m3_bridge(k, v, table, seq, ko, vo, scale, scale)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for length in (3 * page - 5, 0, 1, page + 17):
        seq.fill_(length)
        output.fill_(float("nan"))
        graph.replay()
        for src, dst in ((k, ko), (v, vo)):
            expected = (
                src[table[0].long()].flatten(0, 1).view(torch.float8_e4m3fn).float()
                * scale
            ).half()
            actual = dst.flatten(0, 1)
            torch.testing.assert_close(
                actual[:length], expected[:length], rtol=0, atol=0, equal_nan=True
            )
            zero = expected[:length] == 0
            assert torch.equal(
                torch.signbit(actual[:length])[zero],
                torch.signbit(expected[:length])[zero],
            )
            padded = (length + 15) // 16 * 16
            assert bool((actual[length:padded] == 0).all())
            assert bool(torch.isnan(actual[padded:]).all())
