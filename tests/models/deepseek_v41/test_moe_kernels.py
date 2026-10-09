# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-MOE: Triton kernels of the V4.1 MoE (routing tables, FP32 SwiGLU, FP32 combine, decode GEMVs) against
their torch twins. Bitwise where the twin defines the exact operation order; FP32-accumulation-order tolerance
where a GEMV reduction order differs from cuBLAS."""

from __future__ import annotations

import pytest
import torch

from vllm.models.deepseek_v41.sm70 import gemv as v41_gemv
from vllm.models.deepseek_v41.sm70 import moe_kernels as mk

pytestmark = pytest.mark.sm70


def _ids(num_tokens: int, top_k: int, num_experts: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(num_tokens, num_experts, device="cuda", generator=g)
    return scores.topk(top_k, dim=-1)[1].to(torch.int32).contiguous()


def _segments(perm, gids, goff) -> dict[int, list[int]]:
    """expert -> sorted slot list described by one partition's tables."""
    out: dict[int, list[int]] = {}
    goff = goff.tolist()
    for g, e in enumerate(gids.tolist()):
        beg, end = goff[g], goff[g + 1]
        if end > beg:
            out[e] = sorted(perm[beg:end].tolist())
    return out


@pytest.mark.parametrize("num_tokens,top_k,num_experts", [(1, 6, 384), (2, 6, 384), (8, 6, 384), (8, 3, 128),
                                                          (3, 3, 128), (10, 6, 384)])
@pytest.mark.parametrize("spill", [False, True])
@pytest.mark.parametrize("remap", [False, True])
def test_route_prep_matches_reference(num_tokens, top_k, num_experts, spill, remap):
    ids = _ids(num_tokens, top_k, num_experts, seed=num_tokens * 31 + top_k)
    if top_k == 6 and num_tokens >= 2:
        ids[1] = ids[0]  # repeated experts -> multi-row groups
    phys = None
    if remap:
        phys = torch.randperm(num_experts, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
        phys = phys.to(torch.int32)
    n_res = num_experts * 2 // 3 if spill else num_experts
    perm, tables = mk.route_prep(ids, phys, n_res, spill)
    rperm, rtables = mk.route_prep_reference(ids, phys, n_res, spill)
    torch.testing.assert_close(perm, rperm, rtol=0, atol=0)
    assert len(tables) == len(rtables) == (2 if spill else 1)
    for (gids, goff), (rgids, rgoff) in zip(tables, rtables):
        torch.testing.assert_close(gids, rgids, rtol=0, atol=0)
        torch.testing.assert_close(goff, rgoff, rtol=0, atol=0)
    # semantic check: every slot appears exactly once, in the partition that owns its expert
    flat = ids.flatten().long()
    key = phys.long()[flat] if phys is not None else flat
    seen: dict[int, list[int]] = {}
    for p, (gids, goff) in enumerate(tables):
        for e, slots in _segments(perm, gids, goff).items():
            phys_e = e + (n_res if p == 1 else 0)
            assert phys_e not in seen
            seen[phys_e] = slots
    expect: dict[int, list[int]] = {}
    for s, k in enumerate(key.tolist()):
        expect.setdefault(k, []).append(s)
    assert seen == expect


def test_route_prep_large_uses_torch_tables():
    ids = _ids(512, 6, 384, seed=3)
    perm, tables = mk.route_prep(ids, None, 384, False)
    rperm, rtables = mk.route_prep_reference(ids, None, 384, False)
    assert torch.equal(perm, rperm) and torch.equal(tables[0][0], rtables[0][0])


@pytest.mark.parametrize("rows,inter", [(1, 576), (6, 576), (48, 576), (300, 1152), (7, 288)])
def test_swiglu_fp32_bitwise(rows, inter):
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, 2 * inter, device="cuda", generator=g) * 6).half()
    x[0, :8] = torch.tensor([30.0, -30.0, 10.0, -10.0, 9.99, 0.0, 65000.0, -65000.0], device="cuda").half()
    out = mk.swiglu_fp32(x, 10.0)
    ref = mk.swiglu_fp32_reference(x, 10.0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("num_tokens,top_k,hidden", [(1, 6, 5120), (8, 6, 5120), (77, 6, 5120), (5, 3, 5120)])
@pytest.mark.parametrize("shared_dtype", [None, torch.float32, torch.float16])
def test_combine_fp32_bitwise(num_tokens, top_k, hidden, shared_dtype):
    g = torch.Generator(device="cuda").manual_seed(num_tokens + top_k)
    y = torch.randn(num_tokens * top_k, hidden, device="cuda", generator=g).half()
    w = torch.rand(num_tokens, top_k, device="cuda", generator=g)
    shared = None if shared_dtype is None else torch.randn(num_tokens, hidden, device="cuda", generator=g).to(
        shared_dtype)
    out = mk.combine_fp32(y, w, shared)
    torch.testing.assert_close(out, mk.combine_reference(y, w, shared), rtol=0, atol=0)


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("n,k", [(384, 5120), (128, 5120), (1024, 5120), (32, 5120), (4096, 1280)])
def test_gemv_matches_fp32(m, n, k):
    g = torch.Generator(device="cuda").manual_seed(m * n + k)
    x = torch.randn(m, k, device="cuda", generator=g).half()
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.05).half()
    ref = x.float() @ w.float().t()
    out = v41_gemv.gemv(x, w)
    assert out.dtype == torch.float32
    assert _rel_rms(out, ref) < 1e-6
    out16 = v41_gemv.gemv(x, w, out_dtype=torch.float16, alpha=0.5)
    assert out16.dtype == torch.float16
    assert (out16.float() - (ref * 0.5).half().float()).abs().max().item() <= 2e-3 * ref.abs().max().item()


@pytest.mark.parametrize("m", [1, 2, 8])
def test_gate_up_swiglu_matches_unfused(m):
    g = torch.Generator(device="cuda").manual_seed(m)
    inter, k = 576, 5120
    x = torch.randn(m, k, device="cuda", generator=g).half()
    w13 = (torch.randn(2 * inter, k, device="cuda", generator=g) * 0.02).half()
    gate_up = (x.float() @ w13.float().t()).half()
    ref = mk.swiglu_fp32_reference(gate_up, 10.0)
    out = v41_gemv.gate_up_swiglu(x, w13, 10.0)
    # identical up to FP32 accumulation order deciding an FP16 rounding of gate/up
    diff = (out.float() - ref.float()).abs()
    assert (diff > 0).float().mean().item() < 0.01
    assert _rel_rms(out, ref) < 2e-4


@pytest.mark.parametrize("m,top_k", [(1, 6), (4, 6), (8, 6), (2, 3)])
@pytest.mark.parametrize("with_routed", [False, True])
def test_down_combine_matches_reference(m, top_k, with_routed):
    g = torch.Generator(device="cuda").manual_seed(m * 10 + top_k)
    n, k = 5120, 576
    act = torch.randn(m, k, device="cuda", generator=g).half()
    w2 = (torch.randn(n, k, device="cuda", generator=g) * 0.02).half()
    shared = act.float() @ w2.float().t()
    y = torch.randn(m * top_k, n, device="cuda", generator=g).half() if with_routed else None
    tw = torch.rand(m, top_k, device="cuda", generator=g) if with_routed else None
    out = v41_gemv.down_combine(act, w2, y, tw)
    ref = mk.combine_reference(y, tw, shared) if with_routed else shared
    assert _rel_rms(out, ref) < 1e-6
    if with_routed:
        # routed part alone is bitwise: zero shared weights
        out0 = v41_gemv.down_combine(act, torch.zeros_like(w2), y, tw)
        torch.testing.assert_close(out0, mk.combine_reference(y, tw, None), rtol=0, atol=0)


def test_gemv_rejects_large_m():
    x = torch.zeros(9, 5120, device="cuda", dtype=torch.float16)
    w = torch.zeros(384, 5120, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="at most 8 rows"):
        v41_gemv.gemv(x, w)


@pytest.mark.parametrize("spill", [False, True])
def test_route_prep_large_is_graph_capturable(spill):
    ids = _ids(32, 6, 384, seed=5)  # 192 slots: the torch path
    phys = torch.randperm(384, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)).to(torch.int32)
    eager = mk.route_prep(ids, phys, 300, spill)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mk.route_prep(ids, phys, 300, spill)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mk.route_prep(ids, phys, 300, spill)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured[0], eager[0])
    for (g, o), (ge, oe) in zip(captured[1], eager[1]):
        assert torch.equal(g, ge) and torch.equal(o, oe)
