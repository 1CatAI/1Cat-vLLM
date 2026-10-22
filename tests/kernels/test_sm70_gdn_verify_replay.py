# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-request SM70 GDN verification and deferred-commit replay."""

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

H, HV, K, T = 4, 8, 128, 5  # TP4 shard of Flash-Next GDN, MTP4 rows
TMAX = 8
SEQ = TMAX * (H * K + HV * K + HV * 2)


def _inputs(requests, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = (
        torch.randn(requests * T, 2 * H * K + HV * K, device="cuda", generator=g) * 0.5
    ).half()
    a = torch.randn(requests * T, HV, device="cuda", generator=g).half()
    b = torch.randn(requests * T, HV, device="cuda", generator=g).half()
    return qkv, a, b


def _verify(state, qkv, a, b, A_log, dt_bias, cu, idx, nacc, **replay):
    out = torch.empty(qkv.shape[0], HV, K, device="cuda", dtype=torch.half)
    torch.ops._C.sm70_gdn_verify_out(
        qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        out,
        cu,
        idx,
        nacc,
        H,
        HV,
        K**-0.5,
        1,
        replay.get("rbuf"),
        replay.get("rn"),
        replay.get("ctr"),
        T,
        replay.get("rmap"),
        replay.get("ckt"),
        replay.get("cks"),
    )
    return out


@pytest.mark.parametrize("requests", [2, 4, 8])
def test_multi_request_verify_matches_triton(requests):
    torch.manual_seed(requests)
    A_log = torch.randn(HV, device="cuda") * 0.5
    dt_bias = torch.randn(HV, device="cuda") * 0.5
    qkv, a, b = _inputs(requests, requests)
    slots = requests * T + 1
    state = torch.randn(slots, HV, K, K, device="cuda") * 0.1
    cu = torch.arange(0, requests * T + 1, T, device="cuda", dtype=torch.int32)
    idx = (1 + torch.arange(requests * T, device="cuda", dtype=torch.int32)).view(
        requests, T
    )
    nacc = torch.randint(1, T + 1, (requests,), device="cuda", dtype=torch.int32)
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
    new_state = state.clone()
    out = _verify(new_state, qkv, a, b, A_log, dt_bias, cu, idx, nacc)
    torch.testing.assert_close(
        out.float(), ref.reshape(-1, HV, K).float(), atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(new_state, ref_state, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("requests", [1, 3, 8])
def test_replay_matches_materialized_states(requests):
    """Deferred commit + replay reproduces per-position state materialization,
    including an in-round checkpoint and a commit-only final call."""
    torch.manual_seed(100 + requests)
    A_log = torch.randn(HV, device="cuda") * 0.5
    dt_bias = torch.randn(HV, device="cuda") * 0.5
    base = torch.randn(requests, HV, K, K, device="cuda") * 0.1
    rounds = 6
    accepted = torch.randint(1, T + 1, (rounds, requests), dtype=torch.int32)
    cu = torch.arange(0, requests * T + 1, T, device="cuda", dtype=torch.int32)

    # Materialized: each request owns T spec slots; the next round reads the
    # accepted slot.
    mat = torch.zeros(requests * T, HV, K, K, device="cuda")
    for r in range(requests):
        mat[r * T] = base[r]
    idx_m = torch.arange(requests * T, device="cuda", dtype=torch.int32).view(
        requests, T
    )
    nacc_m = torch.ones(requests, device="cuda", dtype=torch.int32)

    # Replay: one committed slot per request (scattered); persistent replay
    # slots permuted against batch rows, per-slot buffer sides.
    slots = 2 * requests + 3
    ckpt = slots - 1
    rep = torch.zeros(slots, HV, K, K, device="cuda")
    committed = torch.randperm(slots - 2)[:requests].to(torch.int32) + 1
    for r in range(requests):
        rep[committed[r]] = base[r]
    idx_r = committed.to("cuda").view(requests, 1).contiguous()
    slot_of = torch.randperm(requests).to(torch.int32)
    side = torch.zeros(requests, dtype=torch.int32)
    rbuf = torch.zeros(2 * requests * SEQ, device="cuda")
    rn = torch.zeros(requests, device="cuda", dtype=torch.int32)

    for step in range(rounds):
        qkv, a, b = _inputs(requests, 1000 * requests + step)
        out_m = _verify(mat, qkv, a, b, A_log, dt_bias, cu, idx_m, nacc_m)
        side ^= 1  # every request verifies this round
        rmap = (2 * slot_of + side).cuda()
        ckt = torch.full((requests,), -1, device="cuda", dtype=torch.int32)
        cks = torch.full((requests,), -1, device="cuda", dtype=torch.int32)
        if step == 3:
            ckt[-1] = 2  # last request, after its current token 2
            cks[-1] = ckpt
        out_r = _verify(
            rep,
            qkv,
            a,
            b,
            A_log,
            dt_bias,
            cu,
            idx_r,
            None,
            rbuf=rbuf,
            rn=rn,
            rmap=rmap,
            ckt=ckt,
            cks=cks,
        )
        torch.testing.assert_close(out_r, out_m, atol=0, rtol=0)
        if step == 3:
            want = mat[(requests - 1) * T + 2]
            torch.testing.assert_close(rep[ckpt], want, atol=0, rtol=0)
        nacc_m = accepted[step].cuda()
        rn = accepted[step].cuda()
    # Commit-only call (no current tokens, no side flip) settles the round.
    empty = torch.zeros(requests + 1, device="cuda", dtype=torch.int32)
    rmap = (2 * slot_of + (side ^ 1)).cuda()
    _verify(
        rep,
        qkv[:0],
        a[:0],
        b[:0],
        A_log,
        dt_bias,
        empty,
        idx_r,
        None,
        rbuf=rbuf,
        rn=rn,
        rmap=rmap,
    )
    for r in range(requests):
        want = mat[r * T + accepted[-1, r].item() - 1]
        torch.testing.assert_close(rep[committed[r]], want, atol=0, rtol=0)
