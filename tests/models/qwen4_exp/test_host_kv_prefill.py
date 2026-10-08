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


@pytest.mark.parametrize("skip_topk", [False, True])
def test_grouped_prefill_rejects_reused_mtp_tail(monkeypatch, skip_topk):
    from vllm.models.qwen4_exp.nvidia.ops import host_kv_prefill
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention

    calls = []
    monkeypatch.setattr(host_kv_prefill, "prefill_staging_reason", lambda *args: None)
    monkeypatch.setattr(
        host_kv_prefill,
        "host_qsa_prefill",
        lambda *args, **kwargs: calls.append(kwargs["grouped_page4"]),
    )
    owner = SimpleNamespace(
        host_kv=object(),
        host_kv_prefill_enabled=True,
        host_kv_prefill_grouped=True,
        indexer=SimpleNamespace(skip_topk=skip_topk),
    )
    query = torch.empty(512, 6, 256, dtype=torch.float16)
    Qwen4ExpQSAAttention.host_kv_forward(
        owner, query, None, None, None, None, None, None, None
    )
    assert calls == [not skip_topk]


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
@pytest.mark.parametrize("grouped_page4", [False, True])
def test_batched_prefill_matches_32_row_reader_with_causal_tails_and_rewrites(
    dtype, rows, context, grouped_page4, monkeypatch
):
    from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV
    from vllm.models.qwen4_exp.nvidia.ops.host_kv_attention import host_qsa_attention
    from vllm.models.qwen4_exp.nvidia.ops.qsa import expand_qsa_block_indices_cuda

    native_calls = []
    if grouped_page4:
        if torch.cuda.get_device_capability() != (7, 0):
            pytest.skip("Requires SM70 grouped attention")
        from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda

        from vllm.models.qwen4_exp.nvidia.ops import qsa

        if not qsa._qsa_grouped_page4_supported(flash_attn_v100_cuda, "auto"):
            pytest.skip("Requires the Flash-V100 grouped ABI")
        original = qsa._qsa_sparse_paged_attention_sm70_grouped_page4

        def native(*args, **kwargs):
            native_calls.append(args[0].shape[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(
            qsa, "_qsa_sparse_paged_attention_sm70_grouped_page4", native
        )

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
    positions[:8] = torch.tensor([-1, 0, 1, 2, 3, 14, 31, 63], device=device)
    requests[8:16] = -1
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
            query,
            state,
            indices,
            table,
            requests,
            positions,
            lengths,
            actual,
            gate,
            grouped_page4=grouped_page4,
        )

    def compare():
        if not grouped_page4:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert torch.isfinite(actual).all()
            error = actual.float() - expected.float()
            relative = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(
                expected.float()
            )
            assert relative < 1e-3
            assert error.abs().max() < 4e-3

    run()
    torch.accelerator.synchronize()
    if grouped_page4:
        assert native_calls == [rows // 8 * 8]
    compare()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        state.write(key[:7].neg(), value[:7].mul(0.5), slots[:7])
        run()
    for _ in range(2):
        graph.replay()
        torch.accelerator.synchronize()
        compare()
