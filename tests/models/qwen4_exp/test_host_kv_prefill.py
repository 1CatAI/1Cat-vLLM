# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.host_kv_prefill import (
    host_qsa_prefill,
    prefill_staging_reason,
    stage_prefill_history,
)


def test_prefill_admission_preserves_decode_and_workspace_bound():
    state = SimpleNamespace(
        rows=32, page_size=816, dim=256, staging=torch.empty(32 * 2 * 2052 * 256)
    )
    assert prefill_staging_reason(state, torch.empty(1, 42), 16384) is None
    assert prefill_staging_reason(state, torch.empty(4, 42), 16384) == (
        "request_history_exceeds_shared_miss_workspace"
    )
    for rows in (1, 5, 20, 32, 128, 511):
        assert prefill_staging_reason(state, torch.empty(1, 42), rows) == (
            "query_rows_below_prefill_band"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_history_staging_matches_official_decode_and_preserves_page_aliases(dtype):
    from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV

    torch.manual_seed(37)
    device = torch.device("cuda:0")
    state = HostQSAKV(5, 816, 256, device, hot_tokens=64, dtype=dtype)
    key = torch.randn(5 * 816, 1, 256, dtype=torch.float16, device=device)
    value = torch.randn_like(key)
    slots = torch.arange(key.shape[0], device=device)
    state.write(key, value, slots)
    # Noncontiguous row stride, shared physical pages and an invalid final page.
    table = torch.tensor(
        [[0, 2, 4, 99], [0, 3, -1, 99]], device=device, dtype=torch.int32
    )[:, :3]
    lengths = torch.tensor([2050, 2000], device=device, dtype=torch.int32)

    def check():
        cache, mapped = stage_prefill_history(state, table, lengths)
        torch.accelerator.synchronize()
        expected = torch.zeros_like(cache, device="cpu")
        cpu_table, cpu_lengths = table.cpu(), lengths.cpu()
        for request in range(2):
            for page in range(3):
                physical = int(cpu_table[request, page])
                count = max(0, min(816, int(cpu_lengths[request]) - page * 816))
                if physical < 0:
                    continue
                source = state.host[physical, :, :count].float()
                if dtype == torch.uint8:
                    source = (
                        state.host[physical, :, :count]
                        .view(torch.float8_e4m3fn)
                        .float()
                    )
                    scales = state.host_scales[
                        physical * 816 : physical * 816 + count
                    ].T
                    source *= scales[:, :, None, None]
                expected[request * 3 + page, :, :count] = source.half()
        torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            mapped.cpu(),
            torch.tensor([[0, 1, 2], [3, 4, -1]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        assert cache.untyped_storage().data_ptr() == (
            state.staging.untyped_storage().data_ptr()
        )

    check()
    state.write(key[:7].neg(), value[:7].mul(0.5), slots[:7])
    check()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
@pytest.mark.parametrize("rows", [512, 527, 544])
@pytest.mark.parametrize("context", [257, 2050])
def test_batched_prefill_matches_32_row_reader_with_causal_tails_and_rewrites(
    dtype, rows, context
):
    from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV
    from vllm.models.qwen4_exp.nvidia.ops.host_kv_attention import host_qsa_attention
    from vllm.models.qwen4_exp.nvidia.ops.qsa import expand_qsa_block_indices_cuda

    torch.manual_seed(41)
    device = torch.device("cuda:0")
    state = HostQSAKV(6, 816, 256, device, hot_tokens=64, dtype=dtype)
    key = torch.randn(6 * 816, 1, 256, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    slots = torch.arange(key.shape[0], device=device)
    state.write(key, value, slots)
    table = torch.tensor([[0, 2, 4], [0, 3, -1]], device=device, dtype=torch.int32)
    requests = torch.arange(rows, device=device, dtype=torch.int32) % 2
    positions = context - 5 + torch.arange(rows, device=device) % 5
    lengths = torch.full((2,), context, device=device, dtype=torch.int32)
    compressed = torch.arange(512, device=device, dtype=torch.int32).repeat(rows, 1)
    indices = expand_qsa_block_indices_cuda(
        compressed, positions, lengths, requests, 4, 2048
    )
    query = torch.randn(rows, 6, 256, dtype=torch.float16, device=device)
    gate = torch.randn_like(query)
    expected, actual = torch.empty_like(query), torch.empty_like(query)

    def run():
        for start in range(0, rows, 32):
            stop = min(start + 32, rows)
            host_qsa_attention(
                query[start:stop],
                state,
                indices[start:stop],
                table,
                requests[start:stop],
                positions[start:stop],
                lengths,
                expected[start:stop],
                gate[start:stop],
            )
        host_qsa_prefill(
            query, state, indices, table, requests, positions, lengths, actual, gate
        )

    run()
    torch.accelerator.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        state.write(key[:7].neg(), value[:7].mul(0.5), slots[:7])
        run()
    for _ in range(2):
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
