# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise tests of the exact SM70 Qwen GDN verify fusion units."""

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import sm70_gdn_verify_fused as fused

QD, VD, NH, D, T_SPEC, W, SLOTS = 512, 1536, 12, 128, 5, 4, 64
CONV_DIM = 2 * QD + VD
STATE_LEN = W - 1 + T_SPEC - 1


def _sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


requires_sm70 = pytest.mark.skipif(not _sm70(), reason="SM70 GPU required")


def test_parse_units():
    assert fused.parse_units(None) == frozenset()
    assert fused.parse_units("") == frozenset()
    assert fused.parse_units("all") == frozenset(fused.UNITS)
    assert fused.parse_units(" u4, U1 ") == frozenset({"u1", "u4"})
    with pytest.raises(ValueError):
        fused.parse_units("u5")


@requires_sm70
@pytest.mark.parametrize("tokens", [1, 5, 40])
def test_u4_rmsnorm_gated_bitwise_equals_forward_native(tokens):
    # The fused kernel replays the association order of ATen's mean reduction
    # in torch 2.10 (ATen/native/cuda/Reduce.cuh: vec4 per lane, then
    # shfl_down with decreasing offsets). Another torch version may reduce in
    # another order, so the test fails loudly instead of skipping.
    if not torch.__version__.startswith("2.10."):
        pytest.fail(
            "gdn_rmsnorm_sigmoid_gate_exact mirrors torch 2.10 Reduce.cuh; "
            f"re-verify the reduction order for torch {torch.__version__}"
        )
    from vllm.model_executor.layers.layernorm import RMSNormGated

    g = torch.Generator(device="cpu").manual_seed(tokens)
    x = (torch.randn(tokens, NH, D, generator=g) * 0.3).half().cuda()
    qkvz = (torch.randn(tokens, CONV_DIM + VD, generator=g) * 2.0).half().cuda()
    z = qkvz[:, CONV_DIM:]
    w = (1.0 + torch.randn(D, generator=g) * 0.1).half().cuda()
    ref = RMSNormGated.forward_static(
        x.reshape(-1, D),
        z.reshape(tokens, NH, D).reshape(-1, D),
        w,
        1e-6,
        torch.float16,
        group_size=None,
        norm_before_gate=True,
        activation="sigmoid",
    )
    out = fused.gdn_rmsnorm_sigmoid_gate_exact(x, z, w, 1e-6)
    assert torch.equal(out, ref)


def _spec_case(n_seqs: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = n_seqs * T_SPEC
    idx = torch.randperm(SLOTS, generator=g)[: n_seqs * T_SPEC].view(n_seqs, T_SPEC)
    return {
        "m": m,
        "n": n_seqs,
        "qkvz": (torch.randn(m, CONV_DIM + VD, generator=g) * 2.0).half().cuda(),
        "ba": (torch.randn(m, 2 * NH, generator=g) * 2.0).half().cuda(),
        "idx": idx.to(torch.int32).cuda(),
        "sel": torch.randint(1, T_SPEC + 1, (n_seqs,), generator=g)
        .to(torch.int32)
        .cuda(),
        "qsl": (torch.arange(n_seqs + 1) * T_SPEC).to(torch.int32).cuda(),
        "conv_sd": (torch.randn(SLOTS, STATE_LEN, CONV_DIM, generator=g) * 0.5)
        .half()
        .cuda(),
        "ssm": (torch.randn(SLOTS, NH, D, D, generator=g) * 0.05).cuda(),
        "conv_w": (torch.randn(CONV_DIM, W, generator=g) * 0.3).half().cuda(),
        "A_log": torch.randn(NH, generator=g).cuda(),
        "dt_bias": torch.randn(NH, generator=g).half().cuda(),
    }


@requires_sm70
@pytest.mark.parametrize("n_seqs", [1, 3, 8])
def test_u2_packed_conv_bitwise(n_seqs):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    c = _spec_case(n_seqs, 10 + n_seqs)
    ref_x, ref_state = c["qkvz"].clone(), c["conv_sd"].clone()
    out_x, out_state = c["qkvz"].clone(), c["conv_sd"].clone()
    kw = dict(
        conv_state_indices=c["idx"][:, 0],
        num_accepted_tokens=c["sel"],
        query_start_loc=c["qsl"],
        max_query_len=T_SPEC,
    )
    mq = causal_conv1d_update(
        ref_x[:, :CONV_DIM],
        ref_state.transpose(-1, -2),
        c["conv_w"],
        None,
        "silu",
        validate_data=False,
        **kw,
    )
    q, k, v = torch.split(mq, [QD, QD, VD], dim=-1)
    ref = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)])
    packed = torch.empty(c["m"] * CONV_DIM, dtype=torch.float16, device="cuda")
    fused.causal_conv1d_update_packed(
        out_x[:, :CONV_DIM],
        out_state.transpose(-1, -2),
        c["conv_w"],
        None,
        "silu",
        out=packed,
        q_dim=QD,
        k_dim=QD,
        v_dim=VD,
        **kw,
    )
    assert torch.equal(packed, ref)
    assert torch.equal(out_state, ref_state)


@requires_sm70
@pytest.mark.parametrize("n_seqs", [1, 3, 8])
@pytest.mark.parametrize("bv_cap", ["", "8"])
def test_u3_gated_recurrent_bitwise(n_seqs, bv_cap, monkeypatch):
    monkeypatch.setattr(fused, "_GDN_BV48", bv_cap)
    from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fused_gdn_gating,
    )

    c = _spec_case(n_seqs, 20 + n_seqs)
    m = c["m"]
    flat = c["qkvz"][:, :CONV_DIM].contiguous()
    q, k, v = torch.split(flat, [QD, QD, VD], dim=-1)
    packed = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)])
    q = packed[: m * QD].view(1, m, -1, D)
    k = packed[m * QD : 2 * m * QD].view(1, m, -1, D)
    v = packed[2 * m * QD :].view(1, m, -1, D)
    b_view, a_view = c["ba"][:, :NH], c["ba"][:, NH:]
    ref_ssm, out_ssm = c["ssm"].clone(), c["ssm"].clone()
    g, beta = fused_gdn_gating(
        c["A_log"],
        a_view.contiguous(),
        b_view.contiguous(),
        c["dt_bias"],
        beta_dtype=torch.float32,
    )
    o, _ = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=ref_ssm,
        inplace_final_state=True,
        cu_seqlens=c["qsl"],
        ssm_state_indices=c["idx"],
        num_accepted_tokens=c["sel"],
        use_qk_l2norm_in_kernel=True,
    )
    out = torch.zeros(m, NH, D, dtype=torch.float16, device="cuda")
    fused.fused_recurrent_gdn_verify(
        q=q,
        k=k,
        v=v,
        b=b_view,
        a=a_view,
        A_log=c["A_log"],
        dt_bias=c["dt_bias"],
        initial_state=out_ssm,
        out=out,
        cu_seqlens=c["qsl"],
        ssm_state_indices=c["idx"],
        num_accepted_tokens=c["sel"],
    )
    assert torch.equal(out, o.squeeze(0))
    assert torch.equal(out_ssm, ref_ssm)
