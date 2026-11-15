# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 Wave 5 kernel parity and CUDAGraph replay tests."""

import importlib

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.device_kv_attention import (
    initialize_device_history_attention,
)
from vllm.models.qwen4_exp.nvidia.ops.host_kv import HostQSAKV
from vllm.models.qwen4_exp.nvidia.ops.host_kv_attention import host_qsa_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


def _gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    rrms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((xf * rrms) * (1.0 + w.float())).half()


def _neox_rope(
    x: torch.Tensor, cos_sin: torch.Tensor, pos: torch.Tensor
) -> torch.Tensor:
    rot = cos_sin.shape[1]
    half_rot = rot // 2
    cs = cos_sin[pos.long()].float()
    cos = cs[:, None, :half_rot]
    sin = cs[:, None, half_rot:]
    xr = x[..., :rot].float()
    a, b = xr[..., :half_rot], xr[..., half_rot:]
    rotated = torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1).half()
    return torch.cat([rotated, x[..., rot:]], dim=-1)


@pytest.mark.parametrize("k_down", [160, 320, 640])
@pytest.mark.parametrize("rows", [1, 5, 10, 20, 40])
def test_shared_down_gated_matches_reference_and_cudagraph(rows: int, k_down: int):
    if not hasattr(torch.ops._C, "qwen38_shared_down_gated_sm70_out"):
        pytest.skip("qwen38_shared_down_gated_sm70_out unavailable")
    torch.manual_seed(1000 + rows * 7 + k_down)
    dev = torch.device("cuda")
    hidden_x = torch.randn(rows, 2560, device=dev, dtype=torch.float16) * 0.5
    act_out = torch.randn(rows, k_down, device=dev, dtype=torch.float16) * 0.5
    down_w = torch.randn(2560, k_down, device=dev, dtype=torch.float16) * 0.05
    gate_w = torch.randn(1, 2560, device=dev, dtype=torch.float16) * 0.05
    out = torch.empty(rows, 2560, device=dev, dtype=torch.float16)

    def ref_fn():
        down = (act_out.float() @ down_w.float().T).half()
        gate_logit = (hidden_x.float() @ gate_w.float().T).half().float()
        gate = torch.sigmoid(gate_logit)
        return (down.float() * gate).half()

    torch.ops._C.qwen38_shared_down_gated_sm70_out(
        out, hidden_x, act_out, down_w, gate_w
    )
    torch.testing.assert_close(out, ref_fn(), rtol=2e-3, atol=2e-3)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.ops._C.qwen38_shared_down_gated_sm70_out(
            out, hidden_x, act_out, down_w, gate_w
        )
    hidden_x.mul_(-0.75)
    act_out.mul_(0.5)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref_fn(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("rows", [1, 4, 5, 10, 20, 40])
def test_router_top10_batch_matches_reference(rows: int):
    pytest.importorskip("vllm._sm70_router_C")
    if not hasattr(torch.ops.vllm_sm70_router, "top10_batch"):
        pytest.skip("top10_batch unavailable")
    torch.manual_seed(2000 + rows)
    dev = torch.device("cuda")
    # Guarantee distinct FP16 logits per row so top-10 ordering is unambiguous.
    perm = torch.stack([torch.randperm(512, device=dev) for _ in range(rows)])
    logits_fp16 = ((perm.float() - 256.0) * 0.015625).half().contiguous()
    weights = torch.empty(rows, 10, device=dev, dtype=torch.float32)
    ids = torch.empty(rows, 10, device=dev, dtype=torch.int32)
    source = torch.empty(rows, 10, device=dev, dtype=torch.int32)
    torch.ops.vllm_sm70_router.top10_batch(weights, ids, source, logits_fp16)

    logits = logits_fp16.float()
    top_l, top_i = torch.topk(logits, 10, dim=-1)
    raw_w = torch.exp(top_l - top_l[:, :1])
    ref_w = raw_w / raw_w.sum(dim=-1, keepdim=True)
    ref_src = (
        torch.arange(10, device=dev, dtype=torch.int32)[None, :] * rows
        + torch.arange(rows, device=dev, dtype=torch.int32)[:, None]
    )
    assert torch.equal(ids, top_i.int())
    assert torch.equal(source, ref_src)
    torch.testing.assert_close(weights, ref_w, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("hq", [6, 12, 24])
@pytest.mark.parametrize("fp8_mode", ["fp16", "fp8_hot_fp16", "fp8_hot_fp8"])
@pytest.mark.parametrize("rows", [1, 5, 15])
def test_qsa_prep_write_matches_norm_rope_and_host_write(
    rows: int, hq: int, fp8_mode: str
):
    importlib.import_module("vllm._sm70_qsa_device_C")
    if not hasattr(torch.ops.vllm_sm70_qsa_device, "prep_write"):
        pytest.skip("prep_write unavailable")
    torch.manual_seed(3000 + rows * 31 + hq)
    dev = torch.device("cuda")
    eps = 1e-6
    rot = 64
    is_fp8 = fp8_mode != "fp16"
    is_hot_fp8 = fp8_mode == "fp8_hot_fp8"
    dtype = torch.uint8 if is_fp8 else torch.float16

    state_a = HostQSAKV(
        4,
        64,
        256,
        dev,
        hot_tokens=64,
        rows=rows,
        width=64,
        dtype=dtype,
        hot_fp8=is_hot_fp8,
        direct_device=True,
    )
    state_b = HostQSAKV(
        4,
        64,
        256,
        dev,
        hot_tokens=64,
        rows=rows,
        width=64,
        dtype=dtype,
        hot_fp8=is_hot_fp8,
        direct_device=True,
    )
    # Pre-populate hot cache so both hot-hit updates and cold writes are tested.
    warm_k = torch.randn(32, 1, 256, device=dev, dtype=torch.float16)
    warm_v = torch.randn(32, 1, 256, device=dev, dtype=torch.float16)
    warm_slots = torch.arange(32, device=dev, dtype=torch.int32)
    for st in (state_a, state_b):
        st.hot_values.zero_()
        if st.hot_scales is not None:
            st.hot_scales.zero_()
        st.write(warm_k, warm_v, warm_slots)
        st.resolve_hot(
            torch.arange(64, device=dev, dtype=torch.int32)[None],
            torch.zeros(1, 1, device=dev, dtype=torch.int32),
            torch.zeros(1, device=dev, dtype=torch.int32),
            torch.tensor([31], device=dev, dtype=torch.int64),
            torch.tensor([32], device=dev, dtype=torch.int32),
        )

    qkv = torch.randn(rows, hq * 512 + 512, device=dev, dtype=torch.float16)
    pos = torch.randint(0, 200, (rows,), device=dev, dtype=torch.int64)
    slots = torch.arange(16, 16 + rows, device=dev, dtype=torch.int32)
    if rows > 1:
        slots[0] = -1
    cos_sin = torch.randn(256, rot, device=dev, dtype=torch.float32)
    qw = torch.randn(256, device=dev, dtype=torch.float16) * 0.1
    kw = torch.randn(256, device=dev, dtype=torch.float16) * 0.1

    q_out = torch.empty(rows, hq, 256, device=dev, dtype=torch.float16)
    k_out = torch.empty(rows, 1, 256, device=dev, dtype=torch.float16)
    torch.ops.vllm_sm70_qsa_device.prep_write(
        qkv,
        pos,
        cos_sin,
        qw,
        kw,
        eps,
        q_out,
        k_out,
        slots,
        state_a.history,
        state_a.scales,
        state_a.tags,
        state_a.page_slots,
        state_a.hot_values,
        state_a.hot_scales if is_hot_fp8 else None,
        state_a.epoch,
    )

    q_raw = qkv[:, : hq * 512].view(rows, hq, 512)[..., :256]
    k_raw = qkv[:, hq * 512 : hq * 512 + 256].view(rows, 1, 256)
    v_raw = qkv[:, hq * 512 + 256 : hq * 512 + 512].view(rows, 1, 256)
    q_ref = _neox_rope(_gemma_rmsnorm(q_raw, qw, eps), cos_sin, pos)
    k_ref = _neox_rope(_gemma_rmsnorm(k_raw, kw, eps), cos_sin, pos)
    state_b.write(k_ref, v_raw, slots)
    torch.cuda.synchronize()

    torch.testing.assert_close(q_out, q_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(k_out, k_ref, rtol=2e-3, atol=2e-3)
    assert int(state_a.epoch.item()) == int(state_b.epoch.item())
    if not is_fp8:
        torch.testing.assert_close(
            state_a.host.float(), state_b.host.float(), rtol=3e-3, atol=3e-3
        )
    else:
        torch.testing.assert_close(
            state_a.host_scales, state_b.host_scales, rtol=3e-3, atol=3e-5
        )
    idx_check = torch.arange(32, 64, device=dev, dtype=torch.int32).view(1, 32)
    idx_pad = torch.cat([idx_check, torch.zeros_like(idx_check)], dim=1)
    res_a = state_a.resolve_hot(
        idx_pad,
        torch.zeros(1, 1, device=dev, dtype=torch.int32),
        torch.zeros(1, device=dev, dtype=torch.int32),
        torch.tensor([31], device=dev, dtype=torch.int64),
        torch.tensor([32], device=dev, dtype=torch.int32),
    )[0, :16].clone()
    res_b = state_b.resolve_hot(
        idx_pad,
        torch.zeros(1, 1, device=dev, dtype=torch.int32),
        torch.zeros(1, device=dev, dtype=torch.int32),
        torch.tensor([31], device=dev, dtype=torch.int64),
        torch.tensor([32], device=dev, dtype=torch.int32),
    )[0, :16].clone()
    # Check hot-cache logical values for slots 0..31 (updated in-place on hit).
    # Clone resolved views because HostQSAKV instances with identical geometry
    # share the per-device resolve workspace.
    idx_warm = torch.arange(64, device=dev, dtype=torch.int32)[None]
    ra = state_a.resolve_hot(
        idx_warm,
        torch.zeros(1, 1, device=dev, dtype=torch.int32),
        torch.zeros(1, device=dev, dtype=torch.int32),
        torch.tensor([31], device=dev, dtype=torch.int64),
        torch.tensor([32], device=dev, dtype=torch.int32),
    )[0, :32].clone()
    rb = state_b.resolve_hot(
        idx_warm,
        torch.zeros(1, 1, device=dev, dtype=torch.int32),
        torch.zeros(1, device=dev, dtype=torch.int32),
        torch.tensor([31], device=dev, dtype=torch.int64),
        torch.tensor([32], device=dev, dtype=torch.int32),
    )[0, :32].clone()
    assert (ra >= 0).all() and (rb >= 0).all()
    assert (res_a < 0).all() and (res_b < 0).all()
    if is_hot_fp8:

        def _decode_e4m3(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
            b = codes.to(torch.int32)
            sign = torch.where((b & 0x80) != 0, -1.0, 1.0)
            exp = (b >> 3) & 0xF
            mant = (b & 0x7).to(torch.float32) * 0.125
            val = torch.where(
                exp == 0,
                mant * (2.0**-6),
                (1.0 + mant) * torch.pow(2.0, (exp - 7).to(torch.float32)),
            )
            return sign * val * scales[..., None]

        hot_a = _decode_e4m3(state_a.hot_values[ra], state_a.hot_scales[ra])
        hot_b = _decode_e4m3(state_b.hot_values[rb], state_b.hot_scales[rb])
        torch.testing.assert_close(hot_a, hot_b, rtol=7e-2, atol=5e-3)
    else:
        r_tol = 7e-2 if is_fp8 else 3e-3
        a_tol = 5e-3 if is_fp8 else 3e-3
        torch.testing.assert_close(
            state_a.hot_values[ra].float(),
            state_b.hot_values[rb].float(),
            rtol=r_tol,
            atol=a_tol,
        )


def test_draft_host_backed_admits_native_hot_while_device_ref_rejects():
    dev = torch.device("cuda")
    host_draft = HostQSAKV(
        2,
        64,
        256,
        dev,
        width=64,
        rows=4,
        hot_tokens=64,
        dtype=torch.float16,
        device_reference=False,
        direct_device=True,
        is_speculative_draft=True,
    )
    initialize_device_history_attention(host_draft, True)
    assert host_draft.device_history_reason is None
    assert host_draft.device_history_hot is True
    assert host_draft.device_history_workspace is not None

    k = torch.randn(64, 1, 256, device=dev, dtype=torch.float16)
    v = torch.randn(64, 1, 256, device=dev, dtype=torch.float16)
    host_draft.write(k, v, torch.arange(64, device=dev))
    q = torch.randn(2, 6, 256, device=dev, dtype=torch.float16)
    gate = torch.randn(2, 6, 512, device=dev, dtype=torch.float16)[..., 256:]
    indices = torch.arange(64, device=dev, dtype=torch.int32).repeat(2, 1)
    table = torch.tensor([[0, 1]], device=dev, dtype=torch.int32)
    reqs = torch.zeros(2, device=dev, dtype=torch.int32)
    pos = torch.tensor([63, 63], device=dev, dtype=torch.int64)
    lens = torch.tensor([64], device=dev, dtype=torch.int32)
    out_native = torch.empty_like(q)
    out_triton = torch.empty_like(q)
    host_qsa_attention(q, host_draft, indices, table, reqs, pos, lens, out_native, gate)
    ws = host_draft.device_history_workspace
    host_draft.device_history_workspace = None
    try:
        host_qsa_attention(
            q, host_draft, indices, table, reqs, pos, lens, out_triton, gate
        )
    finally:
        host_draft.device_history_workspace = ws
    torch.testing.assert_close(out_native, out_triton, rtol=3e-3, atol=3e-4)


@pytest.mark.parametrize("is_bytes", [False, True])
@pytest.mark.parametrize("rows", [1, 5, 15])
def test_ple_local_kv_and_gate_norm_match_eager(rows: int, is_bytes: bool):
    if not hasattr(torch.ops._C, "qwen38_ple_kv_proj_local_sm70_out"):
        pytest.skip("PLE local KV + gate norm ops unavailable")
    torch.manual_seed(4000 + rows + int(is_bytes))
    dev = torch.device("cuda")
    eps = 1e-6
    if is_bytes:
        emb = torch.randint(0, 120, (rows, 1024), device=dev, dtype=torch.uint8)
        scale = torch.tensor([0.03125], device=dev, dtype=torch.float16)
        emb_fp16 = (emb.view(torch.float8_e4m3fn).float() * scale.float()).half()
    else:
        emb = torch.randn(rows, 1024, device=dev, dtype=torch.float16) * 0.2
        scale = torch.empty(0, device=dev, dtype=torch.float32)
        emb_fp16 = emb
    key_w = torch.randn(10240, 1024, device=dev, dtype=torch.float16) * 0.03
    val_w = torch.randn(2560, 1024, device=dev, dtype=torch.float16) * 0.03
    hidden = torch.randn(rows, 10240, device=dev, dtype=torch.float16) * 0.5
    nkw = torch.randn(10240, device=dev, dtype=torch.float16) * 0.1
    nqw = torch.randn(10240, device=dev, dtype=torch.float16) * 0.1
    ncw = torch.randn(10240, device=dev, dtype=torch.float16) * 0.1

    key_out = torch.empty(rows, 10240, device=dev, dtype=torch.float16)
    val_out = torch.empty(rows, 2560, device=dev, dtype=torch.float16)
    torch.ops._C.qwen38_ple_kv_proj_local_sm70_out(
        key_out, val_out, emb, scale, key_w, val_w
    )
    gv_out = torch.empty(rows, 10240, device=dev, dtype=torch.float16)
    norm_out = torch.empty(rows, 10240, device=dev, dtype=torch.float16)
    torch.ops._C.qwen38_ple_gate_norm_sm70_out(
        gv_out, norm_out, hidden, key_out, val_out, 0, hidden, nkw, nqw, ncw, eps
    )

    k_ref = (emb_fp16.float() @ key_w.float().T).half()
    v_ref = (emb_fp16.float() @ val_w.float().T).half()
    torch.testing.assert_close(key_out, k_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(val_out, v_ref, rtol=2e-3, atol=2e-3)

    kn = _gemma_rmsnorm(k_ref.view(rows, 4, 2560), nkw.view(4, 2560), eps)
    qn = _gemma_rmsnorm(hidden.view(rows, 4, 2560), nqw.view(4, 2560), eps)
    dot = (kn * qn).sum(dim=-1, keepdim=True) / (2560**0.5)
    min_c = torch.tensor(1e-6, device=dev, dtype=torch.float16)
    gate = torch.sigmoid(
        torch.sign(dot) * torch.sqrt(torch.clamp(dot.abs(), min=min_c))
    )
    gv_ref = (gate * v_ref.unsqueeze(1)).reshape(rows, 10240)
    norm_ref = _gemma_rmsnorm(gv_ref.view(rows, 4, 2560), ncw.view(4, 2560), eps).view(
        rows, 10240
    )
    torch.testing.assert_close(gv_out, gv_ref, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(norm_out, norm_ref, rtol=5e-3, atol=5e-3)
