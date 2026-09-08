# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    fused_gdn_gating,
)


@pytest.mark.parametrize("tp_size", [2, 4])
def test_packed_entry_preserves_fp32_beta_and_strided_state(tp_size: int):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("The packed DFlash2 verifier requires SM70")
    torch.manual_seed(20260911)
    q_heads, v_heads, dim, tokens = 16 // tp_size, 48 // tp_size, 128, 8
    mixed = (
        torch.randn(
            (tokens, (2 * q_heads + v_heads) * dim),
            device="cuda",
            dtype=torch.float16,
        )
        * 0.1
    )
    a = torch.randn((tokens, v_heads), device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    a_log = torch.randn(v_heads, device="cuda", dtype=torch.float32)
    bias = torch.randn(v_heads, device="cuda", dtype=torch.float16)
    # Keep a gap between pool slots to exercise the native state stride.
    control_storage = (
        torch.randn((12, 2, v_heads, dim, dim), device="cuda", dtype=torch.float32)
        * 0.02
    )
    candidate_storage = control_storage.clone()
    control_state, candidate_state = control_storage[:, 1], candidate_storage[:, 1]
    indices = torch.tensor([[3, 9, 4, 8, 2, 6, 1, 5]], device="cuda", dtype=torch.int32)
    retired = torch.tensor([0, 7, 10, 11], device="cuda")
    accepted = torch.ones(1, device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    expected = torch.empty((tokens, v_heads, dim), device="cuda", dtype=torch.float16)
    actual = torch.empty_like(expected)
    layer = SimpleNamespace(
        A_log=a_log,
        dt_bias=bias,
        num_k_heads=16,
        num_v_heads=48,
        tp_size=tp_size,
        head_k_dim=dim,
        head_v_dim=dim,
    )

    def control():
        q, k, v = torch.split(
            mixed, [q_heads * dim, q_heads * dim, v_heads * dim], dim=-1
        )
        g, beta = fused_gdn_gating(a_log, a, b, bias, beta_dtype=torch.float32)
        output, _ = fused_recurrent_gated_delta_rule(
            q=q.contiguous().view(1, tokens, q_heads, dim),
            k=k.contiguous().view(1, tokens, q_heads, dim),
            v=v.contiguous().view(1, tokens, v_heads, dim),
            g=g,
            beta=beta,
            initial_state=control_state,
            inplace_final_state=True,
            cu_seqlens=cu,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True,
        )
        expected.copy_(output.squeeze(0))

    def candidate():
        QwenGatedDeltaNetAttention._forward_dflash2_packed_gdn_verify(
            layer,
            mixed_qkv=mixed,
            a=a,
            b=b,
            core_attn_out=actual,
            ssm_state=candidate_state,
            spec_query_start_loc=cu,
            spec_state_indices_tensor=indices,
            spec_state_slot_selectors=accepted,
            num_spec_decodes=1,
        )

    control()
    candidate()
    graphs = []
    for run in (control, candidate):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graphs.append(graph)
    for selector in range(1, 9):
        accepted.fill_(selector)
        initial = torch.randn_like(control_storage) * 0.02
        control_storage.copy_(initial)
        candidate_storage.copy_(initial)
        for _ in range(2):
            mixed.copy_(torch.randn_like(mixed) * 0.1)
            a.copy_(torch.randn_like(a))
            b.copy_(torch.randn_like(b))
            for graph in graphs:
                graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            assert torch.equal(
                candidate_storage.view(torch.int32), control_storage.view(torch.int32)
            )
            assert torch.equal(candidate_storage[:, 0], initial[:, 0])
            assert torch.equal(
                candidate_storage.index_select(0, retired),
                initial.index_select(0, retired),
            )
