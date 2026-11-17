# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact storage/cast contract, separate from full-model FP16 quality gates."""

import os
import subprocess
import sys

import pytest
import torch

from vllm.model_executor.kernels.ple.bf16_result import decode_bf16_result_fake


def test_meta_geometry_and_dtype():
    packet = torch.empty((7, 5120), dtype=torch.uint8, device="meta")
    output = decode_bf16_result_fake(packet)
    assert output.shape == (7, 2560)
    assert output.dtype == torch.float16
    for bad in (
        torch.empty((7, 5119), dtype=torch.uint8, device="meta"),
        packet[:, ::2],
        packet.to(torch.int16),
        packet.reshape(-1),
    ):
        with pytest.raises(ValueError, match="uint8 pairs"):
            decode_bf16_result_fake(bad)


def test_interpreted_conversion_all_bf16_patterns():
    # A fresh process prevents TRITON_INTERPRET from affecting other tests.
    # This executes the kernel's integer/float operations on the CPU. It does
    # not qualify CUDA execution, graph replay, or runtime performance.
    program = r"""
import numpy as np
import torch
from vllm.model_executor.kernels.ple.bf16_result import _decode_bf16_bytes
bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
bf16 = bits.view(torch.bfloat16)
packet = bf16.view(torch.uint8)
expected = bf16.to(torch.float16)
actual = torch.empty_like(expected)
with np.errstate(over='ignore', invalid='ignore', under='ignore'):
    _decode_bf16_bytes[(64,)](packet, actual, 65536, BLOCK=1024)
nan = torch.isnan(expected)
assert torch.equal(torch.isnan(actual), nan)
assert torch.equal(actual.view(torch.int16)[~nan], expected.view(torch.int16)[~nan])
print('65536 BF16 encodings passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", TRITON_INTERPRET="1"),
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "65536 BF16 encodings passed" in result.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA execution")
def test_gpu_all_bf16_patterns_and_graph_replay():
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    bf16 = bits.view(torch.bfloat16)
    packet = bf16.view(torch.uint8).reshape(1, -1).cuda()
    # All finite conversions, including FP16 subnormals and signed zero, have
    # an exact storage contract. NaN payload bits need not be preserved.
    for _ in range(3):
        torch.ops.vllm.ple_decode_bf16_result(packet)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = torch.ops.vllm.ple_decode_bf16_result(packet)
    for shift in (0, 137, 32768):
        selected = bf16.roll(shift)
        packet.copy_(selected.view(torch.uint8).reshape_as(packet))
        graph.replay()
        actual = result.cpu().reshape(-1)
        expected = selected.to(torch.float16)
        nan = torch.isnan(expected)
        assert torch.equal(torch.isnan(actual), nan)
        assert torch.equal(
            actual.view(torch.int16)[~nan], expected.view(torch.int16)[~nan]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA execution")
@pytest.mark.parametrize("m", [0, 1, 4, 17])
def test_gpu_tail_and_compiled_result(m):
    values = (torch.arange(m * 2560).reshape(m, 2560) % 1001 - 500).float() / 1024
    bf16 = values.to(torch.bfloat16)
    packet = bf16.view(torch.uint8).cuda()

    def consume(data):
        return torch.ops.vllm.ple_decode_bf16_result(data) * 0.5

    compiled = torch.compile(consume, fullgraph=True)
    expected = bf16.to(torch.float16) * 0.5
    for shift in (0, 1):
        packet.copy_(bf16.roll(shift, 1).view(torch.uint8))
        actual = compiled(packet).cpu()
        assert torch.equal(actual, expected.roll(shift, 1))
