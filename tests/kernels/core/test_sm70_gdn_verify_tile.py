# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.gdn_schedule import GdnScheduleConfig
from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating


@pytest.mark.parametrize("accepted", [1, 4, 8])
@pytest.mark.parametrize("projection_stride", [2560, 4096, 4120])
@pytest.mark.parametrize(
    "tokens,requests", [(t, n) for t in (1, 3, 5, 7, 8) for n in (1, 3, 8)]
)
@torch.inference_mode()
def test_verify_tile_preserves_output_and_every_state_snapshot(
    accepted, projection_stride, tokens, requests
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA device required")
    torch.manual_seed(20261004 + accepted)
    rows = tokens * requests
    projection = torch.randn(
        rows, projection_stride, device="cuda", dtype=torch.float16
    )
    mixed = projection[:, :2560]
    a = torch.randn(rows, 12, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    decay = -torch.rand(rows, 12, device="cuda")
    beta = torch.sigmoid(b.float())
    a_log = torch.randn(12, device="cuda")
    dt_bias = torch.randn(12, device="cuda", dtype=torch.float16)
    indices = torch.randperm(rows + 3, device="cuda", dtype=torch.int64)[:rows]
    indices = indices.int().reshape(requests, tokens)
    accepted_tensor = torch.full(
        (requests,), min(accepted, tokens), device="cuda", dtype=torch.int32
    )
    cu = torch.arange(0, rows + 1, tokens, device="cuda", dtype=torch.int32)
    initial = torch.randn(rows + 3, 12, 128, 128, device="cuda")

    schedule = GdnScheduleConfig()
    schedule.resolve()

    def apply(state, output):
        fused_sigmoid_gating.fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
            a_log,
            a,
            b,
            dt_bias,
            mixed,
            4,
            12,
            128,
            128,
            output,
            initial_state=state,
            cu_seqlens=cu,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted_tensor,
            use_qk_l2norm_in_kernel=True,
            precomputed_g=decay,
            precomputed_beta=beta,
            match_recurrent_schedule=True,
            match_recurrent_numerics=True,
            schedule=schedule,
        )

    graphs, states, outputs = [], [], []
    for legacy in (True, False):
        # The existing legacy override suppresses automatic tile admission.
        schedule.recurrent_override = legacy
        state = initial.clone()
        output = torch.empty(rows, 1, 12, 128, device="cuda", dtype=torch.float16)
        apply(state, output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            apply(state, output)
        graphs.append(graph)
        states.append(state)
        outputs.append(output)
    for cycle in range(4):
        projection.normal_().mul_(0.125 * (cycle + 1))
        decay.copy_(-torch.rand_like(decay))
        beta.copy_(torch.sigmoid(torch.randn_like(beta)))
        for graph, state in zip(graphs, states):
            # Each stateful replay starts from the same pre-forward snapshots.
            state.copy_(initial)
            graph.replay()
        assert torch.equal(outputs[0], outputs[1])
        assert torch.equal(states[0], states[1])
