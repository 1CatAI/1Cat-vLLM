# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repeated expert decode must retain the independent integer-dot oracle."""

import numpy as np
import pytest
import torch
from test_gguf_lattice_transcode import source
from test_gguf_v41_dp4a import gate_reference, integer_weights, quantize_reference

from vllm import _custom_ops  # noqa: F401
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(70), reason="SM70 required"
)


@pytest.mark.parametrize("kind", [16, 17, 29])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("same_expert", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_expert_reuse_preserves_groups_and_half_rounding(
    kind, dtype, same_expert, adaptive
):
    scale, n, k, experts = 0.00137, 576, 5120, 6
    data = [
        source(kind, scale, n=experts * n, k=k, seed=41 + part).reshape(experts, n, -1)
        for part in range(2)
    ]
    weights = [torch.from_numpy(value).cuda() for value in data]
    decoded = [integer_weights(value, kind, scale, n, k) for value in data]
    ids = (
        (torch.arange(6, device="cuda")[:, None] + torch.arange(6, device="cuda"))
        .remainder(6)
        .to(dtype)
    )
    if same_expert:
        ids.zero_()
    x = (torch.randn(6, k, device="cuda") * 0.25).half()
    q8 = torch.empty(6, 160, 36, device="cuda", dtype=torch.uint8)
    output = torch.empty(6, 6, 2, n, device="cuda", dtype=torch.half)

    def run():
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        operator = (
            torch.ops._C.gguf_v41_dp4a_gate_adaptive_sm70_out
            if adaptive
            else torch.ops._C.gguf_v41_dp4a_gate_reuse_sm70_out
        )
        operator(output, q8, ids, *weights, kind)

    def check():
        q, d, _ = quantize_reference(x)
        repeated_q = np.repeat(q.cpu().numpy()[:, None], 6, axis=1)
        repeated_d = np.repeat(d.cpu().numpy()[:, None], 6, axis=1)
        for part in range(2):
            expected = gate_reference(
                decoded[part][ids.cpu().numpy()], repeated_q, repeated_d, scale
            )
            np.testing.assert_array_equal(output[:, :, part].cpu().numpy(), expected)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.mul_(-0.75)
    ids.add_(1).remainder_(6)
    graph.replay()
    check()


@pytest.mark.parametrize("kind", [16, 17, 29])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_adaptive_branch_changes_during_graph_replay(kind, dtype):
    weights = [
        torch.from_numpy(source(kind, 0.00137, n=36 * 576, k=5120, seed=41 + p))
        .cuda()
        .view(36, 576, -1)
        for p in range(2)
    ]
    ids = torch.arange(36, device="cuda", dtype=dtype).view(6, 6)
    x = (torch.randn(6, 5120, device="cuda") * 0.25).half()
    q8 = torch.empty(6, 160, 36, device="cuda", dtype=torch.uint8)
    actual = torch.empty(6, 6, 2, 576, device="cuda", dtype=torch.half)
    expected = torch.empty_like(actual)

    def run():
        torch.ops._C.gguf_v41_dp4a_gate_adaptive_sm70_out(
            actual, q8, ids, *weights, kind
        )

    torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for same in (False, True, False):
        x.mul_(-0.75)
        if same:
            ids.zero_()
        else:
            ids.copy_(torch.arange(36, device="cuda", dtype=dtype).view(6, 6))
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        graph.replay()
        torch.ops._C.gguf_v41_dp4a_gate_up_sm70_out(
            expected, q8, ids, *weights, kind, False, 16, False
        )
        torch.testing.assert_close(
            actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )
