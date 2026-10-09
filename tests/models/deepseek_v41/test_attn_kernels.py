# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L-ATTN kernel unit tests: sm70 (Triton / cuBLAS) paths vs the torch twins and the reference formulas."""

from __future__ import annotations

import pytest
import torch

from vllm.models.deepseek_v41.common import candidate_blocks as cb
from vllm.models.deepseek_v41.common import qat
from vllm.models.deepseek_v41.common.contracts import CAND_ALL
from vllm.models.deepseek_v41.common.rope import V41RopeParams, apply_rope_torch, compute_cos_sin_cache
from vllm.models.deepseek_v41.sm70 import indexer_kernels as ik
from vllm.models.deepseek_v41.sm70 import sparse_kernels as sk
from vllm.models.deepseek_v41.sm70.q_rope_kv_insert import q_rope_kv_insert

from .test_attn_harness import ref_select_candidate_blocks, rel_rms

pytestmark = pytest.mark.sm70
DEV = "cuda"
YARN = V41RopeParams(64, 160000.0, True, 16.0, 65536, 32.0, 1.0)


def _cs(n: int = 70000) -> torch.Tensor:
    return compute_cos_sin_cache(YARN, n, DEV)


def _flip_fraction(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.view(torch.int16) != b.view(torch.int16)).float().mean())


def test_q_rope_kv_insert_sm70_matches_torch() -> None:
    torch.manual_seed(0)
    T, H = 300, 16
    cs = _cs()
    pos = torch.randint(0, 70000, (T,), device=DEV)
    q = torch.randn(T, H, 512, device=DEV).half()
    kv = torch.randn(T, 512, device=DEV) * torch.rand(T, 1, device=DEV) * 4
    slots = torch.randperm(1000, device=DEV)[:T]
    slots[::7] = -1
    rows_a = torch.full((1000, 512), float("nan"), device=DEV).half()
    rows_b = rows_a.clone()
    qa, qb = q.clone(), q.clone()
    q_rope_kv_insert(qa, kv, pos, cs, rows_a, slots, impl="sm70")
    q_rope_kv_insert(qb, kv, pos, cs, rows_b, slots, impl="torch")
    torch.testing.assert_close(qa.float(), qb.float(), rtol=1e-3, atol=1e-3)
    assert torch.equal(qa[..., :448], q[..., :448])
    keep = slots >= 0
    a, b = rows_a[slots[keep]], rows_b[slots[keep]]
    assert not torch.isnan(a).any()
    # identical up to FMA-contraction ties in the rope; a flip moves one element by one QAT step
    assert _flip_fraction(a, b) < 1e-4
    ref = qat.fp8_block32_qdq(apply_rope_torch(kv, pos, cs), out_dtype=torch.float16, impl="torch")
    assert _flip_fraction(b, ref[keep]) == 0.0
    untouched = torch.ones(1000, dtype=torch.bool, device=DEV)
    untouched[slots[keep]] = False
    assert torch.isnan(rows_a[untouched]).all()


def test_ckv_store_and_export() -> None:
    torch.manual_seed(1)
    N = 257
    cs = _cs()
    lat = (torch.randn(N, 512, device=DEV) * 2).half()
    lpos = torch.randint(0, 60000, (N,), device=DEV)
    slots = torch.randperm(600, device=DEV)[:N]
    out = {}
    for impl in ("sm70", "torch"):
        rows = torch.zeros(600, 512, device=DEV).half()
        exp = torch.zeros(N, 512, device=DEV).half()
        sk.ckv_rope_qat_store(lat, lpos, cs, rows, slots, exp, impl=impl)
        assert torch.equal(rows[slots], exp)
        out[impl] = exp
    assert _flip_fraction(out["sm70"], out["torch"]) < 1e-4
    ref = qat.fp4_e4m3_qdq(apply_rope_torch(lat.float(), lpos, cs), out_dtype=torch.float16, impl="torch")
    assert _flip_fraction(out["torch"], ref) == 0.0


def test_index_q_and_k() -> None:
    torch.manual_seed(2)
    cs = _cs()
    T = 129
    pos = torch.randint(0, 60000, (T,), device=DEV)
    q = torch.randn(T, 32, 128, device=DEV).half()
    a = ik.index_q_rope_qat(q, pos, cs, impl="sm70")
    b = ik.index_q_rope_qat(q, pos, cs, impl="torch")
    assert _flip_fraction(a, b) < 1e-4
    k = torch.randn(T, 128, device=DEV)
    for impl in ("sm70", "torch"):
        rows = torch.zeros(300, 128, device=DEV).half()
        exp = torch.zeros(T, 128, device=DEV).half()
        slots = torch.arange(T, device=DEV) * 2
        ik.index_k_rope_qat_store(k, pos, cs, rows, slots, exp, impl=impl)
        assert torch.equal(rows[slots], exp)


def test_index_scores_and_topk() -> None:
    torch.manual_seed(3)
    Q, n = 37, 3000
    q = torch.randn(Q, 32, 128, device=DEV).half()
    w = torch.randn(Q, 32, device=DEV)
    keys = torch.randn(n, 128, device=DEV).half()
    out = torch.empty(Q, n, device=DEV)
    ik.index_scores(q, w, keys, out, key_tile=1024)
    ref = (torch.einsum("qhd,nd->qhn", q.float(), keys.float()).relu() * w[..., None]).sum(1)
    assert rel_rms(out, ref) < 1e-5
    ends = torch.randint(1, n + 1, (Q,), device=DEV)
    ends[0] = 3
    s = out.masked_fill(torch.arange(n, device=DEV)[None, :] >= ends[:, None], float("-inf"))
    got = ik.topk_sorted(s, ends, 512)
    kk = min(512, n)
    exp = s.topk(kk, -1).indices.sort(-1).values
    exp = torch.where(exp < ends[:, None], exp, -1)
    assert torch.equal(got[:, :kk].long(), exp)
    assert got[0, :3].tolist() == [0, 1, 2] and (got[0, 3:] == -1).all()


@pytest.mark.parametrize("width", [16384, 16385, 20000, 25000, 70])
def test_candidate_blocks_match_reference(width: int) -> None:
    torch.manual_seed(width)
    rows = 64
    logits = torch.randn(rows, width, device=DEV)
    logits[:, ::5] = torch.round(logits[:, ::5] * 4) / 4          # ties
    ends = torch.randint(1, width + 1, (rows,), device=DEV)
    ends[0], ends[1], ends[2] = width, 16384, 16385
    ends = ends.clamp(max=width)
    masked = logits.masked_fill(torch.arange(width, device=DEV)[None, :] >= ends[:, None], float("-inf"))
    ref_keep = ref_select_candidate_blocks(masked, ends[:, None])
    out = torch.empty(rows, 2048, dtype=torch.int32, device=DEV)
    cb.select_candidate_blocks(masked.clone(), ends, 2048, 8, out)
    for r in range(rows):
        nb = (int(ends[r]) + 7) // 8
        if nb <= 2048:
            assert out[r, 0].item() == CAND_ALL
            got = torch.zeros(width, dtype=torch.bool, device=DEV)
            got[: int(ends[r])] = True
        else:
            ids = out[r][out[r] >= 0].long()
            assert torch.equal(ids, ids.sort().values) and ids.numel() == 2048
            got = torch.zeros((width + 7) // 8, dtype=torch.bool, device=DEV)
            got[ids] = True
            got = got.repeat_interleave(8)[:width]
        # the reference keeps exactly the reachable positions of the kept blocks (others are -inf anyway)
        reach = torch.arange(width, device=DEV) < ends[r]
        assert torch.equal(got & reach, ref_keep[r] & reach), f"row {r} end {int(ends[r])}"
    # masking: positions outside the candidates and beyond ends -> -inf; CAND_ALL rows causal only
    m = logits.clone()
    cb.apply_candidate_mask(m, ends, out, 8)
    for r in range(rows):
        reach = torch.arange(width, device=DEV) < ends[r]
        keep = torch.isfinite(m[r])
        if out[r, 0].item() == CAND_ALL:
            assert torch.equal(keep, reach)
        else:
            assert torch.equal(keep, ref_keep[r] & reach)


def test_candidate_torch_twin_equals_reference() -> None:
    torch.manual_seed(9)
    logits = torch.randn(8, 20000, device=DEV)
    ends = torch.tensor([20000, 17000, 16384, 100, 16385, 19999, 8, 1], device=DEV)
    masked = logits.masked_fill(torch.arange(20000, device=DEV)[None, :] >= ends[:, None], float("-inf"))
    assert torch.equal(cb.select_candidate_blocks_torch(logits, ends, 2048, 8),
                       ref_select_candidate_blocks(masked, ends[:, None]))


@pytest.mark.parametrize("impl", ["torch", "sm70"])
def test_sparse_attention_vs_dense_reference(impl: str) -> None:
    torch.manual_seed(4)
    T, H, D = 70, 16, 512
    q = (torch.randn(T, H, D, device=DEV) * 0.05).half()
    rows1 = torch.randn(500, D, device=DEV).half()
    rows2 = torch.randn(800, D, device=DEV).half()
    i1 = torch.randint(0, 500, (T, 128), device=DEV)
    i2 = torch.randint(0, 800, (T, 512), device=DEV)
    i1[:, 100:] = -1
    i2[:, ::3] = -1
    i1[5], i2[5] = -1, -1                       # all-invalid row -> exactly zero
    sink = torch.randn(H, device=DEV)
    out = torch.empty(T, H, D, device=DEV).half()
    sk.sparse_attention(q, [(rows1, i1), (rows2, i2)], sink, D ** -0.5, out, impl=impl, tile=16)
    k = torch.cat([rows1[i1.clamp(min=0)], rows2[i2.clamp(min=0)]], 1).float()
    v = torch.cat([i1 >= 0, i2 >= 0], 1)
    s = torch.einsum("thd,tkd->thk", q.float(), k) * D ** -0.5
    s = s.masked_fill(~v[:, None, :], float("-inf"))
    logits = torch.cat([s, sink[None, :, None].expand(T, H, 1)], -1)
    p = torch.softmax(logits, -1)[..., :-1]
    ref = torch.einsum("thk,tkd->thd", torch.nan_to_num(p), k)
    ref[5] = 0
    assert torch.equal(out[5], torch.zeros_like(out[5]))
    # torch: FP32 math, error = FP16 output rounding only (<= 2^-12 relative per element)
    assert rel_rms(out, ref) < (2e-3 if impl == "sm70" else 3e-4)
