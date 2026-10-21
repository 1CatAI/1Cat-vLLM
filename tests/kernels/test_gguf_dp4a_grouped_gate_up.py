# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped gate/up (IQ3_S, IQ4_NL, IQ4_XS) equals the per-route kernel bitwise."""

import pytest
import torch

import vllm._custom_ops  # noqa: F401

K, N, EXPERTS, TOP_K = 2560, 128, 64, 10


def _iq3s_bank(generator):
    blocks = torch.randint(
        0, 256, (EXPERTS, N, K // 256, 110), dtype=torch.uint8, generator=generator
    )
    scale = (
        torch.rand(EXPERTS, N, K // 256, generator=generator) * 0.02 + 0.001
    ).half()
    blocks[..., :2] = scale.view(torch.uint8).view(EXPERTS, N, K // 256, 2)
    return blocks.view(EXPERTS, N, K // 256 * 110).cuda()


def _routes(m, shared, generator):
    if shared:
        pool = torch.randperm(EXPERTS, generator=generator)[:shared]
        return pool[torch.randint(0, shared, (m, TOP_K), generator=generator)]
    return torch.stack(
        [torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(m)]
    )


def _available():
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (7, 0)
        and hasattr(torch.ops._C, "gguf_dp4a_gate_up_grouped_sm70_out")
    )


@pytest.mark.parametrize(
    "m,shared", [(1, 0), (5, 0), (5, 12), (20, 40), (20, 3), (20, 64)]
)
@pytest.mark.parametrize("bank_aware", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("per_cta", [1, 3, 8, 16])
def test_matches_per_route_kernel(m, shared, bank_aware, index_dtype, per_cta):
    if not _available():
        pytest.skip("Requires SM70 and the grouped gate/up kernel")
    generator = torch.Generator().manual_seed(17 * m + shared)
    gate, up = _iq3s_bank(generator), _iq3s_bank(generator)
    x = torch.randn(m, K, generator=generator).half().cuda()
    q8 = torch.empty((m, K // 32, 36), dtype=torch.uint8, device="cuda")
    torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
    ids = _routes(m, shared, generator).to(index_dtype).cuda()
    expected = torch.empty((m, TOP_K, N // 32, 36), dtype=torch.uint8, device="cuda")
    actual = torch.full_like(expected, 0xAB)
    torch.ops._C.gguf_dp4a_gate_up_sm70_out(
        expected, q8, ids, gate, up, 21, True, 16, bank_aware
    )
    torch.ops._C.gguf_dp4a_gate_up_grouped_sm70_out(
        actual, q8, ids, gate, up, 21, bank_aware, per_cta
    )
    assert torch.equal(actual, expected)


def test_graph_replay_with_changed_routes():
    if not _available():
        pytest.skip("Requires SM70 and the grouped gate/up kernel")
    generator = torch.Generator().manual_seed(5)
    gate, up = _iq3s_bank(generator), _iq3s_bank(generator)
    m = 20
    q8 = torch.empty((m, K // 32, 36), dtype=torch.uint8, device="cuda")
    ids = torch.zeros((m, TOP_K), dtype=torch.int32, device="cuda")
    out = torch.empty((m, TOP_K, N // 32, 36), dtype=torch.uint8, device="cuda")
    expected = torch.empty_like(out)
    run = lambda: torch.ops._C.gguf_dp4a_gate_up_grouped_sm70_out(  # noqa: E731
        out, q8, ids, gate, up, 21, False, 2
    )
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for shared in (0, 1, 7, 30):
        x = torch.randn(m, K, generator=generator).half().cuda()
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        ids.copy_(_routes(m, shared, generator))
        out.fill_(0)
        graph.replay()
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(
            expected, q8, ids, gate, up, 21, True, 16, False
        )
        assert torch.equal(out, expected)


def _lut4_bank(source_type, generator):
    if source_type == 20:
        blocks = torch.randint(
            0, 256, (EXPERTS, N, K // 32, 18), dtype=torch.uint8, generator=generator
        )
        scale = (torch.rand(EXPERTS, N, K // 32, generator=generator) * 0.02).half()
        blocks[..., :2] = scale.view(torch.uint8).view(EXPERTS, N, K // 32, 2)
        return blocks.view(EXPERTS, N, K // 32 * 18).cuda()
    blocks = torch.randint(
        0, 256, (EXPERTS, N, K // 256, 136), dtype=torch.uint8, generator=generator
    )
    scale = (torch.rand(EXPERTS, N, K // 256, generator=generator) * 0.002).half()
    blocks[..., :2] = scale.view(torch.uint8).view(EXPERTS, N, K // 256, 2)
    return blocks.view(EXPERTS, N, K // 256 * 136).cuda()


@pytest.mark.parametrize("source_type", [20, 23])
@pytest.mark.parametrize("m,shared", [(1, 0), (5, 12), (10, 25), (20, 3), (20, 64)])
@pytest.mark.parametrize("per_cta", [1, 4, 8, 16])
def test_lut4_matches_per_route_kernel(source_type, m, shared, per_cta):
    if not _available():
        pytest.skip("Requires SM70 and the grouped gate/up kernel")
    generator = torch.Generator().manual_seed(7 * m + shared + source_type)
    gate = _lut4_bank(source_type, generator)
    up = _lut4_bank(source_type, generator)
    x = torch.randn(m, K, generator=generator).half().cuda()
    q8 = torch.empty((m, K // 32, 36), dtype=torch.uint8, device="cuda")
    torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
    ids = _routes(m, shared, generator).int().cuda()
    expected = torch.empty((m, TOP_K, N // 32, 36), dtype=torch.uint8, device="cuda")
    actual = torch.full_like(expected, 0xAB)
    torch.ops._C.gguf_dp4a_gate_up_sm70_out(
        expected, q8, ids, gate, up, source_type, True, 4, False
    )
    torch.ops._C.gguf_dp4a_gate_up_grouped_sm70_out(
        actual, q8, ids, gate, up, source_type, False, per_cta
    )
    assert torch.equal(actual, expected)
