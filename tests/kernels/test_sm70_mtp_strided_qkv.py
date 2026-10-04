# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update_mixed_qkv,
)


@pytest.mark.parametrize("requests", [1, 4])
@pytest.mark.parametrize("feature_stride", [1, 2])
def test_strided_qkv_output_and_fp32_state_graph(requests, feature_stride):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    torch.manual_seed(31)
    tokens = requests * 5
    storage = torch.randn(
        tokens, 4096 * feature_stride, device="cuda", dtype=torch.float16
    )
    view = storage[:, : 2560 * feature_stride : feature_stride]
    contiguous = view.clone()
    a = torch.randn(tokens, 12, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    a_log = torch.randn(12, device="cuda", dtype=torch.float32)
    bias = torch.randn(12, device="cuda", dtype=torch.float16)
    seed = (
        torch.randn(tokens + 3, 12, 128, 128, device="cuda", dtype=torch.float32) * 0.02
    )
    states = [seed.clone(), seed.clone()]
    indices = torch.arange(tokens, device="cuda", dtype=torch.int32).view(requests, 5)
    accepted = torch.ones(requests, device="cuda", dtype=torch.int32)
    cu = torch.arange(requests + 1, device="cuda", dtype=torch.int32) * 5

    def run(x, state):
        return fused_sigmoid_gating_delta_rule_update_mixed_qkv(
            A_log=a_log,
            a=a,
            b=b,
            dt_bias=bias,
            mixed_qkv=x,
            num_q_heads=4,
            num_v_heads=12,
            head_k_dim=128,
            head_v_dim=128,
            initial_state=state,
            cu_seqlens=cu,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True,
        )[0]

    for _ in range(3):
        run(view, states[1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(view, states[1])
    for replay in range(5):
        storage.normal_(0, 0.1 * (replay + 1))
        contiguous.copy_(view)
        a.normal_()
        b.normal_()
        accepted.fill_(replay + 1)
        for state in states:
            state.copy_(seed)
        expected = run(contiguous, states[0])
        graph.replay()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(states[0].view(torch.int32), states[1].view(torch.int32))
