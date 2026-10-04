# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fused_gdn_gating
from vllm.model_executor.layers.mamba.gdn.sm70_preprocess import conv_gate_zero
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_device_capability(70), reason="SM70 verifier"
)
@pytest.mark.parametrize(
    "tokens,slot,accepted,length,row_stride,state_length",
    [
        (8, 0, 1, 8, 2560, 10),
        (8, 0, 8, 8, 4096, 10),
        (8, 0, 4, 4, 2560, 10),
        (8, 0, 1, 0, 4096, 10),
        (8, -1, 4, 8, 2560, 10),
        (8, -1, 8, 4, 4096, 10),
        (5, 0, 1, 5, 2560, 7),
        (5, 0, 5, 5, 4096, 7),
        (5, 0, 4, 3, 2560, 10),
        (5, 0, 1, 0, 4096, 7),
        (5, -1, 4, 5, 2560, 7),
        (5, -1, 5, 3, 4096, 7),
    ],
)
def test_conv_gate_zero(tokens, slot, accepted, length, row_stride, state_length):
    torch.manual_seed(17)
    raw = torch.randn(tokens, row_stride, device="cuda", dtype=torch.float16) * 0.1
    reference = raw.clone()[:, :2560]
    candidate = raw.clone()[:, :2560]
    state = torch.randn(1, state_length, 2560, device="cuda", dtype=torch.float16) * 0.1
    reference_state = state.clone().transpose(-1, -2)
    candidate_state = state.clone().transpose(-1, -2)
    weight = torch.randn(2560, 4, device="cuda", dtype=torch.float16) * 0.1
    a = torch.randn(tokens, 12, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    a_log = torch.randn(12, device="cuda")
    bias = torch.randn(12, device="cuda")
    slots = torch.tensor([slot], device="cuda", dtype=torch.int32)
    counts = torch.tensor([accepted], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    out = torch.full(
        (tokens, 12, 128), float("nan"), device="cuda", dtype=torch.float16
    )
    causal_conv1d_update(
        reference,
        reference_state,
        weight,
        None,
        "silu",
        conv_state_indices=slots,
        num_accepted_tokens=counts,
        query_start_loc=cu,
        max_query_len=tokens,
        validate_data=False,
    )
    g, beta = fused_gdn_gating(a_log, a, b, bias, beta_dtype=torch.float32)
    _, candidate_g, candidate_beta = conv_gate_zero(
        candidate,
        candidate_state,
        weight,
        slots,
        counts,
        cu,
        a_log,
        a,
        b,
        bias,
        out,
    )
    assert torch.equal(reference.view(torch.int16), candidate.view(torch.int16))
    assert torch.equal(
        reference_state.view(torch.int16), candidate_state.view(torch.int16)
    )
    assert torch.equal(g.view(torch.int32), candidate_g.view(torch.int32))
    assert torch.equal(beta.view(torch.int32), candidate_beta.view(torch.int32))
    assert torch.equal(out.view(torch.int16), torch.zeros_like(out).view(torch.int16))


@pytest.mark.skipif(
    not current_platform.is_device_capability(70), reason="SM70 verifier"
)
def test_single_request_in_padded_metadata():
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    qkv = torch.empty(8, 2560, device="cuda", dtype=torch.float16)
    a = torch.empty(8, 12, device="cuda", dtype=torch.float16)
    b = torch.empty_like(a)
    state = torch.empty(1, 10, 2560, device="cuda", dtype=torch.float16).transpose(
        -1, -2
    )
    layer = SimpleNamespace(
        enable_sm70_dflash2_fused_gdn_verify=True,
        tp_size=4,
        num_k_heads=16,
        num_v_heads=48,
        head_k_dim=128,
        head_v_dim=128,
        gqa_interleaved_layout=False,
        activation="silu",
        conv1d=SimpleNamespace(
            weight=torch.empty(2560, 1, 4, device="cuda", dtype=torch.float16),
            bias=None,
        ),
    )
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=1,
        num_actual_tokens=8,
        spec_sequence_masks=torch.empty(8, device="cuda", dtype=torch.bool),
        ddtree_parent_ids=None,
        spec_query_start_loc=torch.empty(9, device="cuda", dtype=torch.int32),
        spec_state_indices_tensor=torch.empty(8, 8, device="cuda", dtype=torch.int32),
        spec_state_slot_selectors=torch.empty(8, device="cuda", dtype=torch.int32),
    )
    can_use = QwenGatedDeltaNetAttention._can_use_sm70_gdn_preprocess
    assert can_use(layer, qkv, a, b, state, metadata)
    metadata.num_spec_decodes = 2
    assert not can_use(layer, qkv, a, b, state, metadata)
    metadata.num_spec_decodes = 1
    metadata.num_actual_tokens = 4
    assert not can_use(layer, qkv, a, b, state, metadata)


@pytest.mark.skipif(
    not current_platform.is_device_capability(70), reason="SM70 verifier"
)
@pytest.mark.parametrize("tokens", [5, 8])
def test_conv_gate_zero_changed_metadata_graph(tokens):
    torch.manual_seed(23)
    raw = torch.randn(tokens, 4096, device="cuda", dtype=torch.float16) * 0.1
    original = torch.randn(1, 12, 2560, device="cuda", dtype=torch.float16) * 0.1
    reference, candidate = raw.clone()[:, :2560], raw.clone()[:, :2560]
    reference_state = original.clone().transpose(-1, -2)
    candidate_state = original.clone().transpose(-1, -2)
    weight = torch.randn(2560, 4, device="cuda", dtype=torch.float16) * 0.1
    a = torch.randn(tokens, 12, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    a_log, bias = torch.randn(12, device="cuda"), torch.randn(12, device="cuda")
    slots = torch.zeros(1, device="cuda", dtype=torch.int32)
    counts = torch.ones(1, device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    out = torch.empty(tokens, 12, 128, device="cuda", dtype=torch.float16)

    def run():
        return conv_gate_zero(
            candidate,
            candidate_state,
            weight,
            slots,
            counts,
            cu,
            a_log,
            a,
            b,
            bias,
            out,
        )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, candidate_g, candidate_beta = run()
    a.add_(0.1)
    b.mul_(0.5)
    counts.fill_(tokens)
    cu[1].fill_(tokens - 1)
    for slot in (0, -1):
        slots.fill_(slot)
        reference.copy_(raw[:, :2560])
        candidate.copy_(raw[:, :2560])
        reference_state.copy_(original.transpose(-1, -2))
        candidate_state.copy_(original.transpose(-1, -2))
        out.fill_(float("nan"))
        causal_conv1d_update(
            reference,
            reference_state,
            weight,
            None,
            "silu",
            conv_state_indices=slots,
            num_accepted_tokens=counts,
            query_start_loc=cu,
            max_query_len=tokens,
            validate_data=False,
        )
        g, beta = fused_gdn_gating(a_log, a, b, bias, beta_dtype=torch.float32)
        graph.replay()
        assert torch.equal(reference.view(torch.int16), candidate.view(torch.int16))
        assert torch.equal(
            reference_state.view(torch.int16), candidate_state.view(torch.int16)
        )
        assert torch.equal(g.view(torch.int32), candidate_g.view(torch.int32))
        assert torch.equal(beta.view(torch.int32), candidate_beta.view(torch.int32))
        assert torch.count_nonzero(out) == 0
