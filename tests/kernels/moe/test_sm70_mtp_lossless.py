# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
)
from vllm.models.qwen4_exp.nvidia.mtp_lossless_experts import pack_lossless_fp16
from vllm.triton_utils import tl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
CONFIG = dict(
    BLOCK_SIZE_M=2,
    BLOCK_SIZE_N=128,
    BLOCK_SIZE_K=64,
    GROUP_SIZE_M=1,
    SPLIT_K=1,
    num_warps=4,
    num_stages=3,
)


@pytest.fixture(scope="module")
def banks():
    torch.manual_seed(1167)
    result = []
    for n, k in ((320, 2560), (2560, 160)):
        # Include all subnormal magnitudes and both signs in realistic BF16-origin
        # banks. Distinct expert addresses exercise the packed row/expert stride.
        source = (torch.randn(n, k) * 0.02).bfloat16().half()
        source.flatten()[:2048].copy_(
            torch.cat((torch.arange(1024), torch.arange(1024) | 0x8000))
            .to(torch.uint16)
            .view(torch.float16)
        )
        packed = pack_lossless_fp16(source)
        result.append(
            (
                source.unsqueeze(0).expand(512, -1, -1).to("cuda").contiguous(),
                packed.unsqueeze(0).expand(512, -1, -1).to("cuda").contiguous(),
            )
        )
    return result


@pytest.mark.parametrize("m", [1, 5, 20, 512])
@pytest.mark.parametrize("down", [False, True])
def test_u13_projection_matches_sequential_fp16_fma_and_graph(banks, m, down):
    native, packed = banks[int(down)]
    n, k = native.shape[-2:]
    x = torch.randn(m * 10 if down else m, k, device="cuda", dtype=torch.float16)
    ids = torch.randint(512, (m * 10,), device="cuda", dtype=torch.int32)
    weights = torch.randn(m, 10, device="cuda")
    padded = torch.tensor([m * 20], device="cuda", dtype=torch.int32)
    storage = x.new_full((m * 10 * n + 16,), 19)
    actual = storage[8:-8].view(m, 10, n)
    expected = torch.empty_like(actual)

    def candidate():
        torch.ops._C.sm70_mtp_moe_u13_out(actual, x, packed, ids, weights, padded, down)

    def reference():
        invoke_fused_moe_triton_kernel(
            x,
            native,
            expected,
            None,
            None,
            weights,
            None,
            ids,
            padded,
            down,
            1 if down else 10,
            CONFIG,
            tl.float16,
            False,
            False,
            False,
            False,
            False,
        )

    candidate()
    reference()
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    for scale in (0.001, 1.0, 3.0):
        x.normal_(0, scale)
        ids.random_(0, 512)
        ids[0] = -1
        weights.normal_()
        graph.replay()
        reference()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert torch.all(storage[:8] == 19) and torch.all(storage[-8:] == 19)
    padded.fill_(m * 20 - 2)
    actual.fill_(19)
    expected.fill_(19)
    graph.replay()
    reference()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("m", [1, 5, 20])
def test_complete_lossless_moe_retains_fp16_epilogues(banks, m):
    (w13, p13), (w2, p2) = banks
    x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
    ids = torch.randint(512, (m, 10), device="cuda", dtype=torch.int32)
    weights = torch.randn(m, 10, device="cuda").softmax(-1)
    padded = torch.tensor([m * 20], device="cuda", dtype=torch.int32)
    gate_up = x.new_empty((m, 10, 320))
    invoke_fused_moe_triton_kernel(
        x,
        w13,
        gate_up,
        None,
        None,
        weights,
        None,
        ids.flatten(),
        padded,
        False,
        10,
        CONFIG,
        tl.float16,
        False,
        False,
        False,
        False,
        False,
    )
    hidden = x.new_empty((m * 10, 160))
    torch.ops._C.silu_and_mul(hidden, gate_up.view(m * 10, 320))
    down = x.new_empty((m, 10, 2560))
    invoke_fused_moe_triton_kernel(
        hidden,
        w2,
        down,
        None,
        None,
        weights,
        None,
        ids.flatten(),
        padded,
        True,
        1,
        CONFIG,
        tl.float16,
        False,
        False,
        False,
        False,
        False,
    )
    expected = torch.empty_like(x)
    ops.moe_sum(down, expected)
    candidate = torch.ops.vllm.sm70_mtp_lossless_moe(x, p13, p2, ids, weights)
    assert torch.equal(candidate.view(torch.int16), expected.view(torch.int16))
