# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)


def _fused_recurrent_reference_from_mixed_qkv(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    num_q_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = mixed_qkv.shape[0]
    HV, V, K = state.shape[-3:]

    q, k, v = torch.split(
        mixed_qkv,
        [num_q_heads * K, num_q_heads * K, HV * V],
        dim=-1,
    )
    q = q.view(B, num_q_heads, K).unsqueeze(1).contiguous()
    k = k.view(B, num_q_heads, K).unsqueeze(1).contiguous()
    v = v.view(B, HV, V).unsqueeze(1).contiguous()

    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x
    )
    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(1)
    beta = torch.sigmoid(b.float()).to(a.dtype).unsqueeze(1)

    return fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided_mixed_qkv", [False, True])
def test_fused_recurrent_packed_decode_matches_reference(
    dtype: torch.dtype, strided_mixed_qkv: bool
):
    torch.manual_seed(0)

    # Small but representative GDN config (Qwen3Next defaults are K=128, V=128).
    B = 32
    H = 4
    HV = 8  # grouped value attention: HV must be divisible by H
    K = 128
    V = 128
    qkv_dim = 2 * (H * K) + (HV * V)

    device = torch.device("cuda")

    if strided_mixed_qkv:
        # Simulate a packed view into a larger projection buffer:
        # mixed_qkv.stride(0) > mixed_qkv.shape[1]
        proj = torch.randn((B, qkv_dim + 64), device=device, dtype=dtype)
        mixed_qkv = proj[:, :qkv_dim]
    else:
        mixed_qkv = torch.randn((B, qkv_dim), device=device, dtype=dtype)

    a = torch.randn((B, HV), device=device, dtype=dtype)
    b = torch.randn((B, HV), device=device, dtype=dtype)
    A_log = torch.randn((HV,), device=device, dtype=dtype)
    dt_bias = torch.randn((HV,), device=device, dtype=dtype)

    # Continuous batching indices (include PAD_SLOT_ID=-1 cases).
    ssm_state_indices = torch.arange(B, device=device, dtype=torch.int32)
    ssm_state_indices[-3:] = -1

    state0 = torch.randn((B, HV, V, K), device=device, dtype=dtype)
    state_ref = state0.clone()
    state_packed = state0.clone()

    out_packed = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

    # Reference path: materialize contiguous Q/K/V + explicit gating.
    out_ref, state_ref = _fused_recurrent_reference_from_mixed_qkv(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        state=state_ref,
        state_indices=ssm_state_indices,
        num_q_heads=H,
    )

    # Packed path: fused gating + recurrent directly from packed mixed_qkv.
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    torch.testing.assert_close(out_packed, out_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(state_packed, state_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_fused_recurrent_packed_decode_cuda_graph_replay_uses_runtime_indices():
    torch.manual_seed(1)

    # Mirror Qwen GDN decode in miniature: fixed FULL-graph batch shape, live
    # state slot 0, and PAD_SLOT_ID=-1 rows in the captured tensor.
    B = 4
    H = 2
    HV = 4
    K = 128
    V = 64
    num_state_slots = 8
    dtype = torch.float16
    device = torch.device("cuda")
    qkv_dim = 2 * (H * K) + (HV * V)

    A_log = torch.randn((HV,), device=device, dtype=dtype)
    dt_bias = torch.randn((HV,), device=device, dtype=dtype)
    mixed_qkv_static = torch.empty((B, qkv_dim), device=device, dtype=dtype)
    a_static = torch.empty((B, HV), device=device, dtype=dtype)
    b_static = torch.empty((B, HV), device=device, dtype=dtype)
    state_static = torch.empty(
        (num_state_slots, HV, V, K), device=device, dtype=dtype
    )
    indices_static = torch.empty((B,), device=device, dtype=torch.int32)
    out_static = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

    # Warm Triton outside capture so the graph records only kernel launches.
    warm_mixed_qkv = torch.randn((B, qkv_dim), device=device, dtype=dtype)
    warm_a = torch.randn((B, HV), device=device, dtype=dtype)
    warm_b = torch.randn((B, HV), device=device, dtype=dtype)
    warm_state = torch.randn(
        (num_state_slots, HV, V, K), device=device, dtype=dtype
    )
    warm_indices = torch.tensor([0, 1, -1, -1], device=device, dtype=torch.int32)
    warm_out = torch.empty_like(out_static)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=warm_mixed_qkv,
        a=warm_a,
        b=warm_b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=warm_state,
        out=warm_out,
        ssm_state_indices=warm_indices,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    capture_mixed_qkv = torch.randn((B, qkv_dim), device=device, dtype=dtype)
    capture_a = torch.randn((B, HV), device=device, dtype=dtype)
    capture_b = torch.randn((B, HV), device=device, dtype=dtype)
    capture_state = torch.randn(
        (num_state_slots, HV, V, K), device=device, dtype=dtype
    )
    capture_indices = torch.tensor([6, -1, 7, -1], device=device, dtype=torch.int32)
    mixed_qkv_static.copy_(capture_mixed_qkv)
    a_static.copy_(capture_a)
    b_static.copy_(capture_b)
    state_static.copy_(capture_state)
    indices_static.copy_(capture_indices)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_static,
            a=a_static,
            b=b_static,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=K**-0.5,
            initial_state=state_static,
            out=out_static,
            ssm_state_indices=indices_static,
            use_qk_l2norm_in_kernel=True,
        )

    replay_indices = (
        [0, 1, -1, -1],
        [3, 0, 2, -1],
        [-1, 4, 1, 5],
    )
    for case_id, indices in enumerate(replay_indices):
        mixed_qkv = torch.randn((B, qkv_dim), device=device, dtype=dtype)
        a = torch.randn((B, HV), device=device, dtype=dtype)
        b = torch.randn((B, HV), device=device, dtype=dtype)
        state = torch.randn((num_state_slots, HV, V, K), device=device, dtype=dtype)
        state_indices = torch.tensor(indices, device=device, dtype=torch.int32)

        ref_state = state.clone()
        out_ref, state_ref = _fused_recurrent_reference_from_mixed_qkv(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            state=ref_state,
            state_indices=state_indices,
            num_q_heads=H,
        )

        mixed_qkv_static.copy_(mixed_qkv)
        a_static.copy_(a)
        b_static.copy_(b)
        state_static.copy_(state)
        indices_static.copy_(state_indices)
        out_static.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()

        live_rows = state_indices >= 0
        torch.testing.assert_close(
            out_static[live_rows],
            out_ref[live_rows],
            rtol=1e-2,
            atol=2e-2,
            msg=f"case {case_id} live output mismatch",
        )
        torch.testing.assert_close(
            out_static[~live_rows],
            torch.zeros_like(out_static[~live_rows]),
            rtol=0.0,
            atol=0.0,
            msg=f"case {case_id} padded output mismatch",
        )
        torch.testing.assert_close(
            state_static,
            state_ref,
            rtol=1e-2,
            atol=2e-2,
            msg=f"case {case_id} state mismatch",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_gdn_decode_chain_cuda_graph_replay_uses_runtime_indices():
    torch.manual_seed(2)

    # Miniature Qwen GDN decode chain:
    # causal_conv1d_update(mixed_qkv cache) -> packed recurrent decode.
    # The graph shape is fixed, but state slots are changed before each replay.
    B = 4
    H = 2
    HV = 4
    K = 128
    V = 64
    conv_width = 4
    num_state_slots = 8
    dtype = torch.float16
    device = torch.device("cuda")
    qkv_dim = 2 * (H * K) + (HV * V)

    conv_weight = torch.randn((qkv_dim, conv_width), device=device, dtype=dtype)
    conv_bias = torch.randn((qkv_dim,), device=device, dtype=dtype)
    A_log = torch.randn((HV,), device=device, dtype=dtype)
    dt_bias = torch.randn((HV,), device=device, dtype=dtype)

    x_static = torch.empty((B, qkv_dim), device=device, dtype=dtype)
    a_static = torch.empty((B, HV), device=device, dtype=dtype)
    b_static = torch.empty((B, HV), device=device, dtype=dtype)
    conv_state_static = torch.empty(
        (num_state_slots, qkv_dim, conv_width - 1), device=device, dtype=dtype
    )
    recurrent_state_static = torch.empty(
        (num_state_slots, HV, V, K), device=device, dtype=dtype
    )
    indices_static = torch.empty((B,), device=device, dtype=torch.int32)
    out_static = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

    # Warm both Triton kernels before capture.
    warm_x = torch.randn((B, qkv_dim), device=device, dtype=dtype)
    warm_a = torch.randn((B, HV), device=device, dtype=dtype)
    warm_b = torch.randn((B, HV), device=device, dtype=dtype)
    warm_conv_state = torch.randn(
        (num_state_slots, qkv_dim, conv_width - 1), device=device, dtype=dtype
    )
    warm_recurrent_state = torch.randn(
        (num_state_slots, HV, V, K), device=device, dtype=dtype
    )
    warm_indices = torch.tensor([0, 1, -1, -1], device=device, dtype=torch.int32)
    warm_out = torch.empty_like(out_static)
    warm_mixed_qkv = causal_conv1d_update(
        warm_x,
        warm_conv_state,
        conv_weight,
        conv_bias,
        activation="silu",
        conv_state_indices=warm_indices,
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=warm_mixed_qkv,
        a=warm_a,
        b=warm_b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=warm_recurrent_state,
        out=warm_out,
        ssm_state_indices=warm_indices,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    capture_x = torch.randn((B, qkv_dim), device=device, dtype=dtype)
    capture_a = torch.randn((B, HV), device=device, dtype=dtype)
    capture_b = torch.randn((B, HV), device=device, dtype=dtype)
    capture_conv_state = torch.randn(
        (num_state_slots, qkv_dim, conv_width - 1), device=device, dtype=dtype
    )
    capture_recurrent_state = torch.randn(
        (num_state_slots, HV, V, K), device=device, dtype=dtype
    )
    capture_indices = torch.tensor([6, -1, 7, -1], device=device, dtype=torch.int32)
    x_static.copy_(capture_x)
    a_static.copy_(capture_a)
    b_static.copy_(capture_b)
    conv_state_static.copy_(capture_conv_state)
    recurrent_state_static.copy_(capture_recurrent_state)
    indices_static.copy_(capture_indices)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mixed_qkv_static = causal_conv1d_update(
            x_static,
            conv_state_static,
            conv_weight,
            conv_bias,
            activation="silu",
            conv_state_indices=indices_static,
        )
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_static,
            a=a_static,
            b=b_static,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=K**-0.5,
            initial_state=recurrent_state_static,
            out=out_static,
            ssm_state_indices=indices_static,
            use_qk_l2norm_in_kernel=True,
        )

    replay_indices = (
        [0, 1, -1, -1],
        [3, 0, 2, -1],
        [-1, 4, 1, 5],
    )
    for case_id, indices in enumerate(replay_indices):
        x = torch.randn((B, qkv_dim), device=device, dtype=dtype)
        a = torch.randn((B, HV), device=device, dtype=dtype)
        b = torch.randn((B, HV), device=device, dtype=dtype)
        conv_state = torch.randn(
            (num_state_slots, qkv_dim, conv_width - 1), device=device, dtype=dtype
        )
        recurrent_state = torch.randn(
            (num_state_slots, HV, V, K), device=device, dtype=dtype
        )
        state_indices = torch.tensor(indices, device=device, dtype=torch.int32)

        ref_conv_state = conv_state.clone()
        ref_recurrent_state = recurrent_state.clone()
        mixed_qkv_ref = causal_conv1d_update(
            x.clone(),
            ref_conv_state,
            conv_weight,
            conv_bias,
            activation="silu",
            conv_state_indices=state_indices,
        )
        out_ref = torch.empty_like(out_static)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_ref,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=K**-0.5,
            initial_state=ref_recurrent_state,
            out=out_ref,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )

        x_static.copy_(x)
        a_static.copy_(a)
        b_static.copy_(b)
        conv_state_static.copy_(conv_state)
        recurrent_state_static.copy_(recurrent_state)
        indices_static.copy_(state_indices)
        out_static.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()

        live_rows = state_indices >= 0
        torch.testing.assert_close(
            out_static[live_rows],
            out_ref[live_rows],
            rtol=1e-2,
            atol=2e-2,
            msg=f"case {case_id} live output mismatch",
        )
        torch.testing.assert_close(
            out_static[~live_rows],
            torch.zeros_like(out_static[~live_rows]),
            rtol=0.0,
            atol=0.0,
            msg=f"case {case_id} padded output mismatch",
        )
        torch.testing.assert_close(
            conv_state_static,
            ref_conv_state,
            rtol=1e-2,
            atol=2e-2,
            msg=f"case {case_id} conv state mismatch",
        )
        torch.testing.assert_close(
            recurrent_state_static,
            ref_recurrent_state,
            rtol=1e-2,
            atol=2e-2,
            msg=f"case {case_id} recurrent state mismatch",
        )
