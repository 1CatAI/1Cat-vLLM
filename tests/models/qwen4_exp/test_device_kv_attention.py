# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.device_kv_attention import (
    _direct_history_triton,
    initialize_device_history_attention,
)
from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV
from vllm.models.qwen4_exp.nvidia.ops.host_kv_attention import host_qsa_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_draft_owner_never_admits_native_history():
    state = HostQSAKV(
        1,
        256,
        256,
        torch.device("cuda:0"),
        width=64,
        hot_tokens=64,
        device_reference=True,
        direct_device=True,
        is_speculative_draft=True,
    )
    assert state.device_history_workspace is None
    assert state.device_history_reason == "speculative_draft_unqualified"
    # Reconfiguration outside a draft config must retain the owner's policy.
    initialize_device_history_attention(state, True)
    assert state.device_history_workspace is None
    assert state.device_history_reason == "speculative_draft_unqualified"


def decoded_history(state):
    if not state.fp8:
        return state.history.squeeze(3)
    values = state.history.view(torch.float8_e4m3fn).float().squeeze(3)
    scales = state.scales.view(state.blocks, state.page_size, 2).transpose(1, 2)
    return (values * scales[..., None]).half()


@pytest.mark.parametrize("rows", [32, 33, 40, 64, 65])
def test_batched_verifier_does_not_stage_prefill_history(rows, monkeypatch):
    """Batch size must not put decode into a CPU-planned prefill route."""
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention

    torch.manual_seed(rows)
    device = torch.device("cuda:0")
    state = HostQSAKV(
        8,
        256,
        256,
        device,
        width=65,
        hot_tokens=128,
        rows=8,
        direct_device=True,
        hot_fp8=True,
    )
    layer = SimpleNamespace(
        host_kv=state,
        host_kv_prefill_stage_tokens=131072,
        host_kv_decode_query_len=5,
        host_kv_is_draft=False,
    )
    key = torch.randn(2048, 1, 256, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    slots = torch.arange(2048, device=device)
    state.write(key, value, slots)
    query = torch.randn(rows, 6, 256, device=device, dtype=torch.float16)
    gate = torch.randn_like(query)
    indices = torch.arange(65, device=device, dtype=torch.int32).repeat(rows, 1)
    table = torch.arange(8, device=device, dtype=torch.int32).view(8, 1)
    requests = torch.arange(rows, device=device, dtype=torch.int32) % 8
    positions = torch.full((rows,), 100, device=device, dtype=torch.int64)
    lengths = torch.full((8,), 200, device=device, dtype=torch.int32)
    actual, reference = torch.empty_like(query), torch.empty_like(query)
    metadata = SimpleNamespace(max_query_len=5)

    def forbidden(*args, **kwargs):
        raise AssertionError("decode attempted a host prefill staging plan")

    monkeypatch.setattr(state, "stage_plan", forbidden)

    def run():
        Qwen4ExpQSAAttention.host_kv_forward(
            layer,
            query,
            indices,
            table,
            requests,
            positions,
            lengths,
            actual,
            gate,
            metadata,
        )

    def check():
        for start in range(0, rows, state.resolve_rows):
            stop = min(start + state.resolve_rows, rows)
            host_qsa_attention(
                query[start:stop],
                state,
                indices[start:stop],
                table,
                requests[start:stop],
                positions[start:stop],
                lengths,
                reference[start:stop],
                gate[start:stop],
            )
        torch.accelerator.synchronize()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, rtol=0.003, atol=0.0002)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Reorder request pages, change selections/causal lengths and rewrite KV.
    # Captured replay must observe live metadata, rather than a captured plan.
    for length in (63, 177):
        table.copy_(table.flip(0))
        lengths.fill_(length)
        positions.fill_(length - 1)
        indices.add_(3)
        query.mul_(0.75)
        state.write(key * 0.5, value * 0.5, slots)
        graph.replay()
        check()


@pytest.mark.parametrize("rows", [1, 5, 20])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
@pytest.mark.parametrize("strided", [False, True])
def test_direct_history_matches_protected_arithmetic_and_replay(rows, dtype, strided):
    """Check the actual FP16 probability boundary, not only an FP32 oracle."""
    torch.manual_seed(104)
    state = HostQSAKV(
        16,
        256,
        256,
        torch.device("cuda:0"),
        width=2051,
        rows=rows,
        hot_tokens=64,
        device_reference=True,
        direct_device=True,
        dtype=dtype,
    )
    key = torch.randn(4096, 1, 256, dtype=torch.float16, device="cuda")
    value = torch.randn_like(key)
    slots = torch.arange(4096, device="cuda")
    state.write(key, value, slots)
    table = torch.arange(16, dtype=torch.int32, device="cuda").repeat(2, 1)
    requests = torch.arange(rows, dtype=torch.int32, device="cuda") % 2
    positions = torch.full((rows,), 3790, dtype=torch.int64, device="cuda")
    lengths = torch.full((2,), 4096, dtype=torch.int32, device="cuda")
    # Compact complete page4 groups, followed by the open causal group.
    groups = torch.randperm(947, device="cuda")[:512].int().sort().values
    selected = (groups[:, None] * 4 + torch.arange(4, device="cuda")).flatten()
    indices = torch.cat([selected, torch.tensor([3788, 3789, 3790], device="cuda")])
    indices = indices.int().repeat(rows, 1)
    if strided:
        packed = torch.randn(rows, 6, 512, dtype=torch.float16, device="cuda")
        q, gate = packed.tensor_split(2, dim=-1)
    else:
        q = torch.randn(rows, 6, 256, dtype=torch.float16, device="cuda")
        gate = torch.randn_like(q)
    direct, protected, native = (torch.empty_like(q) for _ in range(3))
    workspace = state.device_history_workspace
    assert workspace is not None, state.device_history_reason

    def captured():
        state.write(key[:7], value[:7], slots[:7])
        _direct_history_triton(
            q,
            state,
            indices,
            table,
            requests,
            positions,
            lengths,
            direct,
            gate,
            workspace,
        )
        host_qsa_attention(
            q, state, indices, table, requests, positions, lengths, native, gate
        )
        state.device_history_workspace = None
        try:
            host_qsa_attention(
                q, state, indices, table, requests, positions, lengths, protected, gate
            )
        finally:
            state.device_history_workspace = workspace

    def check():
        torch.testing.assert_close(direct, protected, rtol=0, atol=0)
        decoded = decoded_history(state).double()
        expected = torch.zeros_like(q)
        for row in range(rows):
            tokens = indices[row]
            tokens = tokens[(tokens >= 0) & (tokens <= positions[row])]
            page = table[requests[row], tokens // state.page_size].long()
            offset = tokens % state.page_size
            k = decoded[page, 0, offset]
            v = decoded[page, 1, offset]
            probabilities = (q[row].double() @ k.T / 16).softmax(-1)
            attention = (probabilities @ v).half()
            expected[row] = (attention.float() * gate[row].float().sigmoid()).half()
        torch.testing.assert_close(native, expected, rtol=0.003, atol=0.0002)

    captured()
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured()
    torch.cuda.current_stream().wait_stream(stream)
    for context, amplitude in [(63, 8.0), (3790, 0.125)]:
        positions.fill_(context)
        # The short selection is compact and padded with invalid tokens.
        if context == 63:
            indices.fill_(-1)
            indices[:, :64] = torch.arange(64, device="cuda", dtype=torch.int32)
        else:
            indices[:, :2048] = selected
            indices[:, 2048:] = torch.tensor([3788, 3789, 3790], device="cuda")
        q.mul_(amplitude)
        value[:7].mul_(-0.75)
        graph.replay()
        torch.accelerator.synchronize()
        check()


@pytest.mark.parametrize("scale", [1.0, 0.001, 256.0])
def test_all_e4m3_codes_single_key_exact_values(scale):
    state = HostQSAKV(
        1,
        256,
        256,
        torch.device("cuda:0"),
        width=1,
        rows=1,
        hot_tokens=64,
        device_reference=True,
        direct_device=True,
    )
    assert state.device_history_workspace is not None, state.device_history_reason
    state.history[0, 0, 0].zero_()
    state.history[0, 1, 0, 0] = torch.arange(
        256, device="cuda", dtype=torch.int32
    ).byte()
    state.scales.fill_(scale)
    q = torch.zeros(1, 6, 256, dtype=torch.float16, device="cuda")
    out = torch.empty_like(q)
    meta = (
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([0], dtype=torch.int64, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
    )
    host_qsa_attention(q, state, *meta, out)
    expected = decoded_history(state)[0, 1, 0]
    torch.testing.assert_close(
        out, expected.expand_as(q), rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.parametrize("rows", [1, 5, 20])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_fp32_attention_oracle_aliases_masks_writes_and_graph(rows, dtype):
    torch.manual_seed(53)
    state = HostQSAKV(
        2,
        256,
        256,
        torch.device("cuda:0"),
        width=129,
        rows=rows,
        hot_tokens=64,
        device_reference=True,
        direct_device=True,
        dtype=dtype,
    )
    assert state.device_history_workspace is not None, state.device_history_reason
    key = torch.randn(512, 1, 256, dtype=torch.float16, device="cuda")
    value = torch.randn_like(key)
    state.write(key, value, torch.arange(512, device="cuda"))
    table = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32, device="cuda")
    requests = torch.arange(rows, device="cuda", dtype=torch.int32) % 2
    positions = torch.full((rows,), 380, dtype=torch.int64, device="cuda")
    lengths = torch.full((2,), 400, dtype=torch.int32, device="cuda")
    indices = (torch.arange(129, device="cuda", dtype=torch.int32) * 3).repeat(rows, 1)
    if rows > 1:
        requests[1] = -1
    if rows > 2:
        positions[2] = -1
    q = torch.randn(rows, 6, 256, device="cuda", dtype=torch.float16)
    gate = torch.randn_like(q)
    out = torch.empty_like(q)

    def check():
        decoded = decoded_history(state)
        expected = torch.zeros_like(q)
        for row in range(rows):
            req = int(requests[row])
            if req < 0:
                continue
            tokens = indices[row]
            tokens = tokens[
                (tokens >= 0) & (tokens <= positions[row]) & (tokens < lengths[req])
            ]
            if tokens.numel() == 0:
                continue
            page = table[req, tokens // 256].long()
            offsets = tokens % 256
            k, v = decoded[page, 0, offsets].float(), decoded[page, 1, offsets].float()
            probabilities = (q[row].float() @ k.T / 16).softmax(-1)
            attention = (probabilities @ v).half()
            expected[row] = (attention.float() * gate[row].float().sigmoid()).half()
        torch.testing.assert_close(out, expected, rtol=0.003, atol=0.0002)

    slots = torch.arange(7, device="cuda")

    def captured():
        state.write(key[:7], value[:7], slots)
        host_qsa_attention(
            q, state, indices, table, requests, positions, lengths, out, gate
        )

    captured()
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured()
    torch.cuda.current_stream().wait_stream(stream)
    for factor in [0.5, -1.0]:
        q.mul_(factor)
        value[:7].mul_(factor)
        graph.replay()
        torch.accelerator.synchronize()
        check()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "rows,width,capacity",
    [
        (1, 1, 48),
        (5, 15, 48),
        (5, 16, 48),
        (5, 17, 48),
        (5, 65, 128),
        (20, 2051, 128),
        (21, 2051, 128),
        (32, 4096, 128),
        (5, 2051, 32768),
    ],
)
def test_encoded_hot_storage_operands_and_changed_graph(rows, width, capacity, native):
    """Raw storage is exact; native attention keeps its existing FP16 gate."""
    torch.manual_seed(73)
    device = torch.device("cuda:0")
    states = [
        HostQSAKV(
            2,
            816,
            256,
            device,
            width=width,
            rows=rows,
            hot_tokens=capacity,
            direct_device=native,
            hot_fp8=encoded,
        )
        for encoded in (False, True)
    ]
    key = torch.randn(1632, 1, 256, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    slots = torch.arange(1632, device=device)
    for state in states:
        state.write(key, value, slots)
    indices = torch.arange(width, device=device, dtype=torch.int32).repeat(rows, 1)
    if width > 1:
        indices[:, -1] = -1
    table = torch.tensor([[0, 1], [0, -1]], device=device, dtype=torch.int32)
    requests = torch.arange(rows, device=device, dtype=torch.int32) % 2
    if rows > 1:
        requests[-1] = -1
    positions = torch.full((rows,), 1540, device=device, dtype=torch.int64)
    lengths = torch.tensor([1600, 1500], device=device, dtype=torch.int32)
    query = torch.randn(rows, 6, 256, device=device, dtype=torch.float16)
    gate = torch.randn_like(query)
    outputs = [torch.empty_like(query) for _ in states]

    def run():
        for state, output in zip(states, outputs):
            state.write(key[:2], value[:2], slots[:2])
            host_qsa_attention(
                query, state, indices, table, requests, positions, lengths, output, gate
            )

    def check():
        assert all(torch.isfinite(output).all() for output in outputs)
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0.003, atol=0.0002)
        assert torch.equal(states[0].host, states[1].host)
        assert torch.equal(states[0].host_scales, states[1].host_scales)
        for state in states:
            pages = state.tags.flatten().cpu()
            live = pages >= 0
            if not live.any():
                continue
            slots_live = torch.arange(pages.numel())[live]
            offsets = torch.arange(4)
            tokens = (pages[live, None] * 4 + offsets).flatten().long()
            cached_tokens = (slots_live[:, None] * 4 + offsets).flatten().long()
            codes = state.host[tokens // 816, :, tokens % 816, 0]
            scales = state.host_scales[tokens]
            cache = state.hot_values[cached_tokens.to(device)].cpu()
            if state.hot_fp8:
                assert torch.equal(cache, codes)
                cached_scales = state.hot_scales[cached_tokens.to(device)].cpu()
                assert torch.equal(cached_scales, scales)
            else:
                decoded = (
                    codes.view(torch.float8_e4m3fn).float() * scales[..., None]
                ).half()
                assert torch.equal(cache, decoded)
            assert not (state.page_slots == -2).any()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    value[:2].mul_(-0.125)
    query.mul_(0.25)
    positions.fill_(63)
    for _ in range(3):
        graph.replay()
        torch.accelerator.synchronize()
        check()


def test_encoded_hot_fallback_without_new_native_op(monkeypatch):
    importlib.import_module("vllm._sm70_qsa_device_C")
    original = torch.ops.vllm_sm70_qsa_device
    monkeypatch.setattr(
        torch.ops,
        "vllm_sm70_qsa_device",
        SimpleNamespace(run_hot=original.run_hot),
    )
    state = HostQSAKV(
        1,
        64,
        256,
        torch.device("cuda:0"),
        hot_tokens=64,
        rows=1,
        width=64,
        hot_fp8=True,
        direct_device=True,
    )
    assert state.device_history_workspace is None
    assert state.device_history_reason == "native_encoded_hot_cache_unavailable"
    key = torch.zeros(64, 1, 256, device="cuda", dtype=torch.float16)
    state.write(key, torch.full_like(key, 3.25), torch.arange(64, device="cuda"))
    query = torch.zeros(1, 6, 256, device="cuda", dtype=torch.float16)
    output = torch.empty_like(query)
    host_qsa_attention(
        query,
        state,
        torch.arange(64, device="cuda", dtype=torch.int32)[None],
        torch.zeros(1, 1, device="cuda", dtype=torch.int32),
        torch.zeros(1, device="cuda", dtype=torch.int32),
        torch.tensor([63], device="cuda", dtype=torch.int64),
        torch.tensor([64], device="cuda", dtype=torch.int32),
        output,
    )
    torch.testing.assert_close(output, torch.full_like(output, 3.25), rtol=0, atol=0)


@pytest.mark.parametrize("encoded", [False, True])
def test_native_hot_rejects_misaligned_storage(encoded):
    importlib.import_module("vllm._sm70_qsa_device_C")
    state = HostQSAKV(
        1,
        64,
        256,
        torch.device("cuda:0"),
        width=16,
        rows=1,
        hot_tokens=16,
        hot_fp8=encoded,
        direct_device=True,
    )
    query = torch.zeros(1, 6, 256, device="cuda", dtype=torch.float16)
    bad = torch.empty(16 * 512 + 1, device="cuda", dtype=state.hot_values.dtype)[1:]
    bad = bad.view(16, 2, 256)
    args = (
        query,
        state.history,
        state.scales,
        torch.arange(16, device="cuda", dtype=torch.int32)[None],
        torch.zeros(1, 1, device="cuda", dtype=torch.int32),
        torch.zeros(1, device="cuda", dtype=torch.int32),
        torch.tensor([15], device="cuda", dtype=torch.int64),
        torch.tensor([16], device="cuda", dtype=torch.int32),
        torch.empty_like(query),
        None,
        *state.device_history_workspace,
        bad,
        torch.zeros(1, 16, device="cuda", dtype=torch.int32),
    )
    op = (
        torch.ops.vllm_sm70_qsa_device.run_hot_fp8
        if encoded
        else torch.ops.vllm_sm70_qsa_device.run_hot
    )
    if encoded:
        args += (state.hot_scales,)
    with pytest.raises(RuntimeError, match="aligned vector loads"):
        op(*args)
