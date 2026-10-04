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
    "slot,accepted,length,row_stride",
    [
        (0, 1, 8, 2560),
        (0, 8, 8, 4096),
        (0, 4, 4, 2560),
        (0, 1, 0, 4096),
        (-1, 4, 8, 2560),
        (-1, 8, 4, 4096),
    ],
)
def test_conv_gate_zero(slot, accepted, length, row_stride):
    torch.manual_seed(17)
    raw = torch.randn(8, row_stride, device="cuda", dtype=torch.float16) * 0.1
    reference = raw.clone()[:, :2560]
    candidate = raw.clone()[:, :2560]
    state = torch.randn(1, 10, 2560, device="cuda", dtype=torch.float16) * 0.1
    reference_state = state.clone().transpose(-1, -2)
    candidate_state = state.clone().transpose(-1, -2)
    weight = torch.randn(2560, 4, device="cuda", dtype=torch.float16) * 0.1
    a = torch.randn(8, 12, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    a_log = torch.randn(12, device="cuda")
    bias = torch.randn(12, device="cuda")
    slots = torch.tensor([slot], device="cuda", dtype=torch.int32)
    counts = torch.tensor([accepted], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    out = torch.full((8, 12, 128), float("nan"), device="cuda", dtype=torch.float16)
    causal_conv1d_update(
        reference,
        reference_state,
        weight,
        None,
        "silu",
        conv_state_indices=slots,
        num_accepted_tokens=counts,
        query_start_loc=cu,
        max_query_len=8,
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
