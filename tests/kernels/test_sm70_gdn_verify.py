# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update_mixed_qkv,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0)
    or not hasattr(torch.ops._C, "sm70_gdn_verify_out"),
    reason="requires SM70 and sm70_gdn_verify_out",
)

H, HV, K = 4, 8, 128  # TP4 shard of Flash-Next GDN


@pytest.mark.parametrize("tokens", [1, 5])
@pytest.mark.parametrize("accepted", [1, 3])
def test_cuda_verify_matches_triton(tokens, accepted):
    torch.manual_seed(tokens * 7 + accepted)
    dev = "cuda"
    slots = 8
    qkv = (torch.randn(tokens, 2 * H * K + HV * K, device=dev) * 0.5).half()
    a = torch.randn(tokens, HV, device=dev).half()
    b = torch.randn(tokens, HV, device=dev).half()
    A_log = torch.randn(HV, device=dev) * 0.5
    dt_bias = torch.randn(HV, device=dev) * 0.5
    state = torch.randn(slots + 1, HV, K, K, device=dev) * 0.1
    cu = torch.tensor([0, tokens], device=dev, dtype=torch.int32)
    idx = torch.arange(1, tokens + 1, device=dev, dtype=torch.int32).view(1, tokens)
    nacc = torch.tensor([min(accepted, tokens)], device=dev, dtype=torch.int32)
    ref_state = state.clone()
    ref, _ = fused_sigmoid_gating_delta_rule_update_mixed_qkv(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        mixed_qkv=qkv,
        num_q_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=K,
        initial_state=ref_state,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=idx,
        num_accepted_tokens=nacc,
        use_qk_l2norm_in_kernel=True,
    )
    out = torch.empty(tokens, HV, K, device=dev, dtype=torch.half)
    new_state = state.clone()
    torch.ops._C.sm70_gdn_verify_out(
        qkv, a, b, A_log, dt_bias, new_state, out, cu, idx, nacc, H, HV, K**-0.5,
        1, None, None, None, tokens,
    )
    torch.testing.assert_close(out.float(), ref.reshape(tokens, HV, K).float(),
                               atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(new_state, ref_state, atol=1e-5, rtol=1e-4)
