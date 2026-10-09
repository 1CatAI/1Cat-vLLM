# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2 item 4: common/hc.py vs a direct torch transcription of the official reference
(inference/model.py Block.hc_mixes / hc_pre / hc_post, RMSNorm; kernel.py hc_split_sinkhorn_kernel as a scalar
loop). Gate (PORT_DESIGN §4.5): HC mixes / Sinkhorn / pre / post rel error <= 1e-6 (FP32 both sides)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.models.deepseek_v41.common import hc as hc_mod
from vllm.models.deepseek_v41.common.hc import (
    hc_collapse, hc_expand, hc_mixes, hc_post, hc_pre, rmsnorm_to_act, sinkhorn_split)

HCM, D, EPS_NORM, EPS_HC, ITERS = 4, 5120, 1e-20, 1e-6, 20


# ---------------- reference twin (transcribed, not imported) ----------------
def ref_sinkhorn_kernel(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
    """kernel.py:406-457, one program per row, scalar FP32 math in the kernel's order."""
    n = mixes.shape[0]
    pre = torch.empty(n, HCM, dtype=torch.float32)
    post = torch.empty(n, HCM, dtype=torch.float32)
    comb = torch.empty(n, HCM, HCM, dtype=torch.float32)
    sig = lambda v: 1.0 / (1.0 + torch.exp(-v))  # noqa: E731
    for i in range(n):
        m = mixes[i]
        for j in range(HCM):
            pre[i, j] = sig(m[j] * hc_scale[0] + hc_base[j]) + EPS_HC
            post[i, j] = 2 * sig(m[j + HCM] * hc_scale[1] + hc_base[j + HCM])
        c = torch.empty(HCM, HCM, dtype=torch.float32)
        for j in range(HCM):
            for k in range(HCM):
                c[j, k] = m[j * HCM + k + 2 * HCM] * hc_scale[2] + hc_base[j * HCM + k + 2 * HCM]
        row_max = c.max(dim=1).values
        c = torch.exp(c - row_max[:, None])
        c = c / c.sum(dim=1)[:, None] + EPS_HC
        c = c / (c.sum(dim=0)[None, :] + EPS_HC)
        for _ in range(ITERS - 1):
            c = c / (c.sum(dim=1)[:, None] + EPS_HC)
            c = c / (c.sum(dim=0)[None, :] + EPS_HC)
        comb[i] = c
    return pre, post, comb


def ref_hc_mixes(x, hc_fn, hc_scale, hc_base):          # model.py Block.hc_mixes, x [b,s,hc,d]
    x = x.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS_NORM)
    mixes = F.linear(x, hc_fn) * rsqrt
    b, s, _ = mixes.shape
    pre, post, comb = ref_sinkhorn_kernel(mixes.view(-1, 24), hc_scale, hc_base)
    return pre.view(b, s, HCM), post.view(b, s, HCM), comb.view(b, s, HCM, HCM)


def ref_hc_pre(x, pre_mix):                              # model.py Block.hc_pre
    y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
    return y.to(x.dtype)


def ref_hc_post(x, residual, post, comb):                # model.py Block.hc_post (x kept FP32: §4.3 deviation)
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.to(torch.bfloat16)


def ref_rmsnorm(x, weight):                              # model.py RMSNorm (weight bf16 in the reference)
    dtype = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + EPS_NORM)
    return weight * x                                    # FP32 here; the port rounds to FP16


def _params(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    fn = torch.randn(24, HCM * D, generator=g) * 0.02
    scale = torch.tensor([0.9, 1.3, 2.0])
    base = torch.randn(24, generator=g) * 0.5
    return fn, scale, base


def _stream(t: int, seed: int = 1, amp: float = 3.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(t, HCM, D, generator=g) * amp).to(torch.bfloat16)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def test_sinkhorn_matches_kernel_transcription() -> None:
    g = torch.Generator().manual_seed(5)
    mixes = torch.randn(64, 24, generator=g) * 3
    _, scale, base = _params()
    ours = sinkhorn_split(mixes, scale, base)
    ref = ref_sinkhorn_kernel(mixes, scale, base)
    for a, b in zip(ours, ref):
        assert _rel(a, b) <= 1e-6
    comb = ours[2]   # doubly stochastic after 20 iterations (up to eps)
    torch.testing.assert_close(comb.sum(-2), torch.ones(64, HCM), atol=1e-4, rtol=0)


def test_hc_mixes_matches_reference() -> None:
    fn, scale, base = _params()
    stream = _stream(9)
    pre, post, comb = hc_mixes(stream, fn, scale, base)
    rpre, rpost, rcomb = ref_hc_mixes(stream.unsqueeze(0), fn, scale, base)
    assert pre.dtype == post.dtype == comb.dtype == torch.float32
    assert pre.shape == (9, 4) and post.shape == (9, 4) and comb.shape == (9, 4, 4)
    assert _rel(pre, rpre[0]) <= 1e-6 and _rel(post, rpost[0]) <= 1e-6 and _rel(comb, rcomb[0]) <= 1e-6


def test_hc_pre_bitwise() -> None:
    stream = _stream(17)
    pre = torch.rand(17, 4) + 0.1
    out = hc_pre(stream, pre)
    assert out.dtype == torch.bfloat16 and out.shape == (17, D)
    assert torch.equal(out, ref_hc_pre(stream.unsqueeze(0), pre.unsqueeze(0))[0])


def test_hc_post_matches_reference() -> None:
    _, scale, base = _params()
    stream = _stream(11)
    g = torch.Generator().manual_seed(3)
    _, post, comb = sinkhorn_split(torch.randn(11, 24, generator=g), scale, base)
    x = torch.randn(11, D, generator=g) * 7
    out = hc_post(x, stream, post, comb)
    ref = ref_hc_post(x.unsqueeze(0), stream.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0))[0]
    assert out.dtype == torch.bfloat16 and out.shape == (11, 4, D)
    exact = (post.unsqueeze(-1) * x.unsqueeze(-2)).double() + torch.einsum(
        "tjc,tjd->tcd", comb.double(), stream.double())
    # both round the same FP32 value up to FP32 summation order: <= 1 bf16 ulp apart, rarely different
    diff = (out.float() - ref.float()).abs()
    ulp = torch.ldexp(torch.ones_like(exact), (torch.frexp(exact.abs().clamp_min(1e-30))[1] - 8).to(torch.int32))
    assert bool((diff <= ulp.float() + 0).all())
    assert float((diff > 0).float().mean()) < 1e-3
    assert _rel(out, exact) <= 4e-3        # bf16 output rounding (8 significant bits)


def test_first_comb_index_contracts() -> None:
    # comb = e_{0,2}: copy 2 of the output receives residual copy 0 (not the other way round)
    stream = torch.zeros(1, 4, D, dtype=torch.bfloat16)
    stream[0, 0] = 1.0
    comb = torch.zeros(1, 4, 4)
    comb[0, 0, 2] = 1.0
    out = hc_post(torch.zeros(1, D), stream, torch.zeros(1, 4), comb)
    assert torch.equal(out[0, 2], torch.ones(D, dtype=torch.bfloat16)) and float(out[0, [0, 1, 3]].abs().sum()) == 0


def test_rmsnorm_to_act() -> None:
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(6, D, generator=g) * 50).to(torch.bfloat16)
    x[2] = 0
    weight = (torch.rand(D, generator=g) + 0.5).to(torch.bfloat16).float()
    out = rmsnorm_to_act(x, weight)
    assert out.dtype == torch.float16
    assert torch.equal(out, ref_rmsnorm(x, weight).to(torch.float16))
    assert float(out[2].abs().sum()) == 0 and bool(torch.isfinite(out).all())


def test_zero_padding_rows_are_finite() -> None:
    fn, scale, base = _params()
    stream = torch.zeros(4, HCM, D, dtype=torch.bfloat16)
    pre, post, comb = hc_mixes(stream, fn, scale, base)
    assert all(bool(torch.isfinite(t).all()) for t in (pre, post, comb))
    out = hc_post(torch.zeros(4, D), stream, post, comb)
    assert float(out.float().abs().sum()) == 0
    assert float(hc_collapse(stream, pre, torch.ones(D)).float().abs().sum()) == 0


def test_chunking_is_invisible(monkeypatch: pytest.MonkeyPatch) -> None:
    fn, scale, base = _params()
    stream = _stream(37)
    full = hc_mixes(stream, fn, scale, base)
    pre_full = hc_pre(stream, full[0])
    post_full = hc_post(stream[:, 0].float(), stream, full[1], full[2])
    monkeypatch.setattr(hc_mod, "_CHUNK", 8)
    chunked = hc_mixes(stream, fn, scale, base)
    for a, b in zip(full, chunked):     # FP32 GEMM blocking depends on M: same values up to FP32 noise
        assert _rel(a, b) <= 1e-6
    assert torch.equal(hc_pre(stream, full[0]), pre_full)
    assert torch.equal(hc_post(stream[:, 0].float(), stream, full[1], full[2]), post_full)


def test_expand_identity_pre_mix() -> None:
    emb = torch.randn(3, D).to(torch.float16)
    stream, pre = hc_expand(emb)
    assert stream.dtype == torch.bfloat16 and stream.shape == (3, 4, D)
    assert torch.equal(stream[:, 0], stream[:, 3]) and torch.equal(pre, torch.tensor([[1.0, 0, 0, 0]] * 3))
    assert torch.equal(hc_pre(stream, pre), emb.to(torch.bfloat16))


def test_input_validation() -> None:
    fn, scale, base = _params()
    with pytest.raises(TypeError):
        hc_mixes(torch.zeros(2, 4, D, dtype=torch.float16), fn, scale, base)
    with pytest.raises(ValueError):
        hc_mixes(torch.zeros(2, 4, D, dtype=torch.bfloat16), fn.half(), scale, base)


@pytest.mark.sm70
def test_gpu_matches_cpu_reference() -> None:
    dev = torch.device("cuda")
    fn, scale, base = _params()
    stream = _stream(33)
    pre, post, comb = hc_mixes(stream.to(dev), fn.to(dev), scale.to(dev), base.to(dev))
    # Gate against FP64 math: with K = 20480 the CPU FP32 twin itself is ~1.4e-6 from FP64 on comb (measured
    # 2026-10-07; GPU 3.1e-7), so an FP32-vs-FP32 1e-6 gate would test GEMM summation order, not the port.
    x64 = stream.flatten(1).double()
    mix64 = (x64 @ fn.double().t()) * torch.rsqrt(x64.square().mean(-1, keepdim=True) + EPS_NORM)
    rpre, rpost, rcomb = sinkhorn_split(mix64, scale.double(), base.double())
    assert _rel(pre.cpu(), rpre) <= 1e-6 and _rel(post.cpu(), rpost) <= 1e-6
    assert _rel(comb.cpu(), rcomb) <= 1e-6
    x = torch.randn(33, D) * 5
    out = hc_post(x.to(dev), stream.to(dev), post, comb).cpu()
    ref = ref_hc_post(x.unsqueeze(0), stream.unsqueeze(0), post.cpu().unsqueeze(0), comb.cpu().unsqueeze(0))[0]
    assert float(((out.float() - ref.float()).abs() > 0).float().mean()) < 1e-3
    assert torch.equal(hc_pre(stream.to(dev), pre).cpu(), ref_hc_pre(stream.unsqueeze(0), pre.cpu().unsqueeze(0))[0])
