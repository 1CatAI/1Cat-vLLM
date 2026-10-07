# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1, 5, 20])
@pytest.mark.parametrize("page", [256, 816, 1568])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_host_gather_collisions_rejection_and_graph(rows, page, dtype):
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    blocks, width = 32, 2051
    state = HostQSAKV(
        blocks, page, 256, device, hot_tokens=64, rows=rows, width=width, dtype=dtype
    )
    count = blocks * page
    key = torch.randn(count, 1, 256, device=device, dtype=torch.float16)
    value = torch.randn_like(key) * 3
    slots = torch.arange(count, device=device, dtype=torch.int64)
    state.write(key, value, slots)
    torch.accelerator.synchronize()
    for kind, tensor in enumerate((key, value)):
        original = tensor.cpu().float().reshape(blocks, page, 256)
        if dtype == torch.float16:
            assert torch.equal(state.host[:, kind, :, 0], original.half())
            continue
        scales = original.double().abs().amax(-1).div(448).float().clamp_min(2.0**-126)
        expected = (original / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
        assert torch.equal(state.host[:, kind, :, 0], expected.view(torch.uint8))
        assert torch.equal(state.host_scales[:, kind].view(blocks, page), scales)

    table = torch.stack((torch.arange(8), torch.arange(24, 32)))
    table = table.to(device=device, dtype=torch.int32)
    requests = torch.arange(rows, device=device, dtype=torch.int32) % 2
    indices = torch.arange(width, device=device, dtype=torch.int32).repeat(rows, 1)
    indices[:, 2048:] = torch.tensor([-1, 2047, 2050], device=device)
    positions = torch.full((rows,), 2047, device=device, dtype=torch.int64)
    lengths = torch.full((2,), 2048, device=device, dtype=torch.int32)

    def check(gather=True):
        if gather:
            actual_k, actual_v, remap = state.gather(
                indices, table, requests, positions, lengths
            )
        else:
            actual_k, actual_v = state.staging.unbind(1)
            remap = state.remapped
        torch.accelerator.synchronize()
        cpu_indices = indices.cpu().long()
        valid = (
            (cpu_indices >= 0)
            & (cpu_indices <= positions.cpu()[:, None])
            & (cpu_indices < lengths.cpu()[requests.cpu().long(), None])
        )
        safe = cpu_indices.clamp(0, 2047)
        physical = table.cpu()[requests.cpu().long()[:, None], safe // page].long()
        for kind, actual in enumerate((actual_k, actual_v)):
            codes = state.host[physical, kind, safe % page, 0]
            if dtype == torch.uint8:
                scales = state.host_scales[physical * page + safe % page, kind]
                reference = codes.view(torch.float8_e4m3fn).float() * scales.unsqueeze(
                    -1
                )
            else:
                reference = codes
            reference = reference.masked_fill(~valid.unsqueeze(-1), 0).half()
            assert torch.equal(actual[:, :width, 0].cpu(), reference)
        expected = torch.arange(width).expand(rows, -1).masked_fill(~valid, -1)
        assert torch.equal(remap.cpu(), expected)
        assert torch.count_nonzero(state.locks).item() == 0

    check()
    check()
    # Rejected speculative positions are rewritten, including previously hot pages.
    overwrite = slots[:8]
    state.write(-key[:8], -value[:8], overwrite)
    check()
    state.write(key[:8], value[:8], overwrite)
    check()
    # Fixed buffers remain valid across capture/replay and changed selections.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        state.gather(indices, table, requests, positions, lengths)
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        state.write(key[:8], value[:8], overwrite)
        state.gather(indices, table, requests, positions, lengths)
    indices[:, :4] = torch.tensor([12, 13, 14, 15], device=device)
    for _ in range(3):
        graph.replay()
    check(gather=False)
    check()


def test_host_cache_hot_pages_reused():
    device = torch.device("cuda:0")
    state = HostQSAKV(1, 256, 256, device, hot_tokens=1024, rows=5, width=128)
    key = torch.randn(256, 1, 256, dtype=torch.float16, device=device)
    state.write(key, key, torch.arange(256, device=device))
    indices = torch.arange(128, dtype=torch.int32, device=device).repeat(5, 1)
    table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    requests = torch.zeros(5, dtype=torch.int32, device=device)
    positions = torch.full((5,), 255, dtype=torch.int64, device=device)
    lengths = torch.full((1,), 256, dtype=torch.int32, device=device)
    for _ in range(2):
        state.gather(indices, table, requests, positions, lengths)
    previous = state.stats.clone()
    state.gather(indices, table, requests, positions, lengths)
    delta = (state.stats - previous).cpu()
    assert delta[0] > delta[1]


@pytest.mark.parametrize("kwargs", [{"page_size": 7}, {"hot_tokens": 7}, {"dim": 128}])
def test_invalid_host_geometry(kwargs):
    args = dict(blocks=1, page_size=256, dim=256, device=torch.device("cuda:0"))
    with pytest.raises(ValueError):
        HostQSAKV(**(args | kwargs))


@pytest.mark.parametrize("visible", [1, 3, 128, 129, 2053])
def test_host_attention_causal_boundary_and_padding(visible):
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    torch.manual_seed(23)
    device = torch.device("cuda:0")
    state = HostQSAKV(9, 256, 256, device, hot_tokens=4096, rows=5)
    key = torch.randn(2304, 1, 256, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    state.write(key, value, torch.arange(2304, device=device))
    complete = min(512, visible // 4) * 4
    selected = list(range(complete)) + list(range((visible // 4) * 4, visible))
    indices = torch.full((5, 2051), -1, dtype=torch.int32, device=device)
    indices[:, : len(selected)] = torch.tensor(selected, device=device)
    table = torch.arange(9, dtype=torch.int32, device=device).view(1, -1)
    requests = torch.zeros(5, dtype=torch.int32, device=device)
    positions = torch.full((5,), visible - 1, dtype=torch.int64, device=device)
    lengths = torch.full((1,), visible, dtype=torch.int32, device=device)
    query = torch.randn(5, 6, 256, dtype=torch.float16, device=device)
    output = torch.empty_like(query)
    k, v, remapped = state.gather(indices, table, requests, positions, lengths)
    qsa_sparse_paged_attention(
        query,
        k,
        v,
        remapped,
        state.table,
        state.requests,
        output,
        query_positions=state.positions,
        sequence_lengths=state.lengths,
    )
    scores = (
        torch.einsum("mhd,mkd->mhk", query.float(), k[:, : len(selected), 0].float())
        / 16
    )
    reference = torch.einsum(
        "mhk,mkd->mhd", scores.softmax(-1), v[:, : len(selected), 0].float()
    )
    assert torch.isfinite(output).all()
    assert ((output.float() - reference).norm() / reference.norm()).item() < 0.002
    assert torch.equal(state.lengths, torch.full_like(state.lengths, len(selected)))


def test_host_attention_empty_padded_rows():
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    device = torch.device("cuda:0")
    state = HostQSAKV(1, 816, 256, device, hot_tokens=64, rows=5)
    indices = torch.full((5, 2051), -1, dtype=torch.int32, device=device)
    table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    requests = torch.tensor([-1, 0, 0, -1, 0], dtype=torch.int32, device=device)
    positions = torch.full((5,), -1, dtype=torch.int64, device=device)
    lengths = torch.zeros(1, dtype=torch.int32, device=device)
    query = torch.randn(5, 6, 256, dtype=torch.float16, device=device)
    output = torch.empty_like(query)
    k, v, remapped = state.gather(indices, table, requests, positions, lengths)
    qsa_sparse_paged_attention(
        query,
        k,
        v,
        remapped,
        state.table,
        state.requests,
        output,
        query_positions=state.positions,
        sequence_lengths=state.lengths,
    )
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output).item() == 0


@pytest.mark.parametrize("rows", [1, 5, 20, 32])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_direct_host_attention_matches_staging_on_misses_and_replay(rows, dtype):
    from vllm.models.qwen4_exp.nvidia.ops.host_kv_attention import host_qsa_attention
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    torch.manual_seed(19)
    device = torch.device("cuda:0")
    state = HostQSAKV(4, 816, 256, device, hot_tokens=64, rows=rows, dtype=dtype)
    direct_state = HostQSAKV(
        4, 816, 256, device, hot_tokens=64, rows=rows, dtype=dtype, staging=False
    )
    key = torch.randn(3264, 1, 256, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    state.write(key, value, torch.arange(3264, device=device))
    direct_state.write(key, value, torch.arange(3264, device=device))
    assert direct_state.staging.nbytes == 2048
    indices = torch.arange(2051, device=device, dtype=torch.int32).repeat(rows, 1)
    table = torch.arange(4, dtype=torch.int32, device=device).view(1, -1)
    requests = torch.zeros(rows, dtype=torch.int32, device=device)
    if rows > 1:
        requests[-1] = -1
    positions = torch.full((rows,), 2049, dtype=torch.int64, device=device)
    lengths = torch.full((1,), 2050, dtype=torch.int32, device=device)
    query = torch.randn(rows, 6, 256, dtype=torch.float16, device=device)
    reference, actual = torch.empty_like(query), torch.empty_like(query)
    gate = torch.randn_like(query)

    def run():
        k, v, remap = state.gather(indices, table, requests, positions, lengths)
        qsa_sparse_paged_attention(
            query,
            k,
            v,
            remap,
            state.table,
            state.requests,
            reference,
            query_positions=state.positions,
            sequence_lengths=state.lengths,
            output_gate=gate,
        )
        host_qsa_attention(
            query,
            direct_state,
            indices,
            table,
            requests,
            positions,
            lengths,
            actual,
            gate,
        )

    for _ in range(2):
        run()
    torch.accelerator.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        state.write(key[:8], value[:8], torch.arange(8, device=device))
        direct_state.write(key[:8], value[:8], torch.arange(8, device=device))
        run()
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
