# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV
from vllm.models.qwen4_exp.nvidia.ops.host_kv_attention import host_qsa_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def decoded_history(state):
    if not state.fp8:
        return state.history.squeeze(3)
    values = state.history.view(torch.float8_e4m3fn).float().squeeze(3)
    scales = state.scales.view(state.blocks, state.page_size, 2).transpose(1, 2)
    return (values * scales[..., None]).half()


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
