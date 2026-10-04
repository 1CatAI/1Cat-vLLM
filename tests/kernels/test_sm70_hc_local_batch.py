# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix, hc_silu
from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import _pack_hc_batch_weight
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability((7, 0)), reason="CUDA SM70 required"
)


@pytest.mark.parametrize("m", [2, 5, 10, 16])
def test_local_hc_fp32_arithmetic_and_disjoint_tp_packets(m):
    assert ops.supports_sm70_qwen38_hc_local(), "packaged HC local operators missing"
    torch.manual_seed(20261004 + m)
    x = (torch.randn(m, 10240, device="cuda") * 0.125).half()
    down = (torch.randn(336, 10240, device="cuda") * 0.005).half()
    up = (torch.randn(10240, 320, device="cuda") * 0.005).half()
    down[324:] = 0
    projected = (x.float() @ down.float().T).half()
    reference_lora = hc_silu(projected[:, :320], 4)
    reference_gates = (reference_lora.float() @ up.float().T).half()
    reference_block = hc_gate_mix(x, reference_gates, 4)
    packets = []
    packs = []
    for rank in range(4):
        packed_down = _pack_hc_batch_weight(down, "down", rank)
        packed_up = _pack_hc_batch_weight(up, "up", rank)
        partials = torch.empty(20, m, 96, dtype=torch.float32, device="cuda")
        packet = torch.empty(m, 336, dtype=torch.float16, device="cuda")
        ops.sm70_qwen38_hc_down_local(x, packed_down, partials, packet, rank)
        outside = torch.ones(336, dtype=torch.bool, device="cuda")
        outside[rank * 80 : (rank + 1) * 80] = False
        if rank == 3:
            outside[320:324] = False
        assert torch.count_nonzero(packet[:, outside]) == 0
        packets.append(packet)
        packs.append(packed_up)
    gathered = torch.stack(packets).float().sum(0).half()
    torch.testing.assert_close(
        gathered[:, :320], reference_lora, rtol=0.003, atol=0.003
    )
    torch.testing.assert_close(
        gathered[:, 320:324], projected[:, 320:324], rtol=0.003, atol=0.003
    )
    lora = gathered[:, :320].contiguous()
    blocks = []
    for rank, packed in enumerate(packs):
        block = torch.empty(m, 2560, dtype=torch.float16, device="cuda")
        ops.sm70_qwen38_hc_up_local(lora, packed, x, block, rank)
        assert torch.count_nonzero(block[:, : rank * 640]) == 0
        assert torch.count_nonzero(block[:, (rank + 1) * 640 :]) == 0
        blocks.append(block)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ops.sm70_qwen38_hc_up_local(lora, packed, x, block, rank)
        previous = block.clone()
        graph.replay()
        torch.testing.assert_close(block, previous, rtol=0, atol=0)
    actual = torch.stack(blocks).float().sum(0).half()
    torch.testing.assert_close(actual, reference_block, rtol=0.003, atol=0.003)
    assert ops.supports_sm70_qwen38_hc_replicated()
    packed_down = _pack_hc_batch_weight(down, "down", None)
    packed_up = _pack_hc_batch_weight(up, "up", None)
    partials = torch.empty(20, m, 352, dtype=torch.float32, device="cuda")
    lora = torch.empty(m, 320, dtype=torch.float16, device="cuda")
    block = torch.empty(m, 2560, dtype=torch.float16, device="cuda")
    injection = torch.empty(m, 4, dtype=torch.float16, device="cuda")
    ops.sm70_qwen38_hc_replicated(
        x, packed_down, packed_up, partials, lora, block, injection
    )
    torch.testing.assert_close(block, reference_block, rtol=0.003, atol=0.003)
    torch.testing.assert_close(injection, projected[:, 320:324], rtol=0.003, atol=0.003)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.sm70_qwen38_hc_replicated(
            x, packed_down, packed_up, partials, lora, block, injection
        )
    previous = block.clone()
    graph.replay()
    torch.testing.assert_close(block, previous, rtol=0, atol=0)


@torch.inference_mode()
def test_replicated_hc_m20_fp32_reference_and_changed_input_graph():
    m = 20
    torch.manual_seed(20261005)
    x = (torch.randn(m, 10240, device="cuda") * 0.125).half()
    down = (torch.randn(336, 10240, device="cuda") * 0.005).half()
    up = (torch.randn(10240, 320, device="cuda") * 0.005).half()
    down[324:] = 0
    packed_down = _pack_hc_batch_weight(down, "down", None)
    packed_up = _pack_hc_batch_weight(up, "up", None)
    partials = torch.empty(20, m, 352, dtype=torch.float32, device="cuda")
    lora = torch.empty(m, 320, dtype=torch.float16, device="cuda")
    block = torch.empty(m, 2560, dtype=torch.float16, device="cuda")
    injection = torch.empty(m, 4, dtype=torch.float16, device="cuda")

    def run():
        ops.sm70_qwen38_hc_replicated(
            x, packed_down, packed_up, partials, lora, block, injection
        )

    def check():
        projected = (x.float() @ down.float().T).half()
        expected_lora = hc_silu(projected[:, :320], 4)
        gates = (expected_lora.float() @ up.float().T).half()
        expected = hc_gate_mix(x, gates, 4)
        torch.testing.assert_close(block, expected, rtol=0.003, atol=0.003)
        torch.testing.assert_close(
            injection, projected[:, 320:324], rtol=0.003, atol=0.003
        )
        assert torch.isfinite(block).all()

    for _ in range(3):
        run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.mul_(0.5)
    graph.replay()
    check()
    previous = block.clone()
    graph.replay()
    torch.testing.assert_close(block, previous, rtol=0, atol=0)
