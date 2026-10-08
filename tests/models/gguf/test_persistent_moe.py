# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident scheduling, expert reuse and graph-epoch numerical contracts."""

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_turbomind_moe import GGUFExpertBank


@pytest.fixture(scope="module", autouse=True)
def native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires SM70")
    pytest.importorskip("vllm._sm70_gguf_persistent_C")


def blocks(rng, experts, rows, k, size, block, scale):
    raw = rng.integers(0, 256, (experts, rows, k // block, size), dtype=np.uint8)
    raw[..., :2] = np.float16(scale).reshape(1).view(np.uint8)
    return raw.reshape(experts, rows, -1)


@pytest.mark.parametrize("source_type", [18, 21, 22])
@pytest.mark.parametrize("down_type", [20, 42])
@pytest.mark.parametrize("shared_routes", [False, True])
def test_replay_changes_routes_and_inputs(source_type, down_type, shared_routes):
    rng = np.random.default_rng(source_type * 100 + down_type)
    experts, n, k, topk = 64, 160, 2560, 10
    size = {18: 98, 21: 110, 22: 82}[source_type]
    stride = ((k // 256 * size + 7) // 8) * 8
    banks = []
    for _ in range(2):
        raw = blocks(rng, experts, n, k, size, 256, 0.0002)
        padded = np.zeros((experts, n, stride), dtype=np.uint8)
        padded[..., : raw.shape[-1]] = raw
        banks.append(torch.from_numpy(padded).cuda())
    gate, up = banks
    down = blocks(rng, experts, k, n * 4, 18, 64, 0.01)
    if down_type == 20:
        down = blocks(rng, experts, k, n * 4, 18, 32, 0.001)
    bank = GGUFExpertBank(down_type, experts, torch.device("cuda"), torch.float16)
    for expert, rows in enumerate(down):
        bank.add(expert, torch.from_numpy(rows.copy()), 0, 4, axis=1)
    bank.finalize()
    torch.manual_seed(source_type * 100 + down_type)
    x = torch.empty((5, k), device="cuda", dtype=torch.float16)
    ids = torch.empty((5, topk), device="cuda", dtype=torch.int32)
    probabilities = torch.empty((5, topk), device="cuda", dtype=torch.float32)
    q8 = torch.empty((5, k // 32, 36), device="cuda", dtype=torch.uint8)
    hidden, expected_hidden = [
        torch.empty((5, topk, n // 32, 36), device="cuda", dtype=torch.uint8)
        for _ in range(2)
    ]
    out, expected = [torch.empty_like(x) for _ in range(2)]
    routes = torch.empty((5, topk, k), device="cuda", dtype=torch.float16)
    ready = torch.zeros(5 * topk * (n // 32), device="cuda", dtype=torch.int64)
    epochs = torch.zeros(k // 32, device="cuda", dtype=torch.int64)

    def refresh():
        x.copy_((torch.randn_like(x.float()) * 0.15).half())
        scores = torch.randn((1 if shared_routes else 5, experts), device="cuda")
        ids.copy_(scores.topk(topk, dim=1).indices.expand(5, -1).int())
        probabilities.copy_(torch.softmax(torch.randn_like(probabilities), dim=1))

    def control():
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(
            expected_hidden, q8, ids, gate, up, source_type, True
        )
        torch.ops._C.gguf_dp4a_down_unroute_sm70_out(
            expected,
            expected_hidden,
            ids,
            probabilities,
            bank.weight_ptrs,
            bank.stat_ptrs,
            down_type,
            experts,
        )

    def candidate():
        torch.ops.sm70_gguf_persistent.run(
            out,
            x,
            ids,
            probabilities,
            gate,
            up,
            bank.weight_ptrs,
            bank.stat_ptrs,
            hidden,
            routes,
            ready,
            epochs,
            source_type,
            down_type,
        )

    refresh()
    control()
    candidate()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
        candidate()
    for _ in range(8):
        refresh()
        hidden.fill_(127)
        routes.fill_(float("nan"))
        ready.fill_(-1)
        control()
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(hidden, expected_hidden, rtol=0, atol=0)
        torch.testing.assert_close(out, expected, rtol=0.002, atol=0.002)
    # Empty routing still writes every output and advances the epoch. Resuming
    # active routing on the same captured graph must not reuse old readiness.
    ids.fill_(-1)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.count_nonzero(out).item() == 0
    refresh()
    control()
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(out, expected, rtol=0.002, atol=0.002)
    with pytest.raises(RuntimeError, match="requires M5"):
        torch.ops.sm70_gguf_persistent.run(
            out,
            x.repeat(4, 1),
            ids,
            probabilities,
            gate,
            up,
            bank.weight_ptrs,
            bank.stat_ptrs,
            hidden,
            routes,
            ready,
            epochs,
            source_type,
            down_type,
        )
