# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared dense, paged and split-KV3 core with independently strided scales."""

import pytest
import torch

from tests.kernels.attention.test_sm70_kv_engine_bindings import CODECS, cache

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)

ALGORITHMS = ("dense", "paged", "splitkv3")


@pytest.fixture
def native():
    from vllm.v1.attention.backends.flash_v100 import ops

    assert ops.get_sm70_splitd_d256_ops() is not None
    namespace = torch.ops._vllm_fa2_C
    assert namespace.sm70_d256_splitd_late_scale_abi() == 1
    return namespace


def inputs(codec, batch, heads, q_len, kv_len):
    from flash_attn_v100 import paged_kv_to_fp16

    torch.manual_seed(944)
    page = 32
    pages = kv_len // page
    _, view = cache(codec, pages=batch * pages + 2, page=page, heads=heads)
    table = (torch.randperm(batch * pages, device="cuda") + 2).int().view(batch, pages)
    lengths = torch.full((batch,), kv_len, device="cuda", dtype=torch.int32)
    k = torch.empty(
        (batch * pages, page, heads, 256), device="cuda", dtype=torch.float16
    )
    v = torch.empty_like(k)
    scales = torch.empty((batch * pages, page, heads, 4), device="cuda")
    ks, vs = scales[..., 0], scales[..., 2]
    paged_kv_to_fp16(
        view.key,
        view.value,
        table,
        lengths,
        k,
        v,
        0.5,
        2.0,
        kv_cache_dtype=codec.name,
        **view.scale_kwargs,
        key_scale_out=ks,
        value_scale_out=vs,
    )
    q = torch.randn((batch, q_len, heads * 6, 256), device="cuda", dtype=torch.float16)
    return q, k, v, ks, vs


def execute(native, algorithm, q, k, v, ks, vs, out):
    batch, q_len, heads = q.shape[:3]
    kv_len = k.shape[0] * k.shape[1] // batch
    if algorithm == "paged":
        # Reorder physical pages and both planes, then use a nontrivial table.
        permutation = torch.randperm(k.shape[0], device="cuda")
        inverse = torch.argsort(permutation).int().view(batch, -1)
        return native.sm70_d256_splitd_n32_paged_fwd(
            q,
            k[permutation],
            v[permutation],
            inverse,
            out,
            kv_len,
            0.0625,
            True,
            ks[permutation],
            vs[permutation],
        )
    k = k.view(batch, kv_len, k.shape[2], 256)
    v = v.view_as(k)
    ks = ks.view(batch, kv_len, k.shape[2])
    vs = vs.view_as(ks)
    if algorithm == "dense":
        return native.sm70_d256_splitd_n32_dense_fwd(q, k, v, out, 0.0625, True, ks, vs)
    partial = torch.empty((3, batch, q_len, heads, 256), device=q.device)
    maximum = torch.empty((3, batch, q_len, heads), device=q.device)
    denominator = torch.empty_like(maximum)
    return native.sm70_d256_splitd_n32_dense_splitkv3_fwd(
        q,
        k,
        v,
        partial,
        maximum,
        denominator,
        out,
        0.0625,
        True,
        ks,
        vs,
    )


def check_attention(q, k, v, ks, vs, out):
    batch, q_len, heads = q.shape[:3]
    kv_len = k.shape[0] * k.shape[1] // batch
    kv_heads = k.shape[2]
    k = k.reshape(batch, kv_len, kv_heads, 256)
    v = v.reshape_as(k)
    ks = ks.reshape(batch, kv_len, kv_heads)
    vs = vs.reshape_as(ks)
    rows = (
        torch.arange(q_len, device=q.device)
        if q_len <= 384
        else torch.tensor([0, 1, q_len // 2, q_len - 1], device=q.device)
    )
    for b in range(batch):
        for h in range(kv_heads):
            qh = q[b, rows, h * 6 : (h + 1) * 6].permute(1, 0, 2).double()
            scores = (qh @ k[b, :, h].double().T) * (ks[b, :, h].double() * 0.0625)
            scores.masked_fill_(
                torch.arange(kv_len, device=q.device)[None, None]
                > (kv_len - q_len + rows)[None, :, None],
                -torch.inf,
            )
            expected = (
                (scores.softmax(-1) * vs[b, :, h].double()) @ v[b, :, h].double()
            ).permute(1, 0, 2)
            torch.testing.assert_close(
                out[b, rows, h * 6 : (h + 1) * 6].double(),
                expected,
                atol=0.004,
                rtol=0.015,
            )
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize(
    "batch,heads,q_len,kv_len", [(1, 1, 64, 96), (2, 2, 384, 3136), (1, 1, 1600, 8192)]
)
def test_shared_splitd_late_scales(
    native, codec, algorithm, batch, heads, q_len, kv_len
):
    q, k, v, ks, vs = inputs(codec, batch, heads, q_len, kv_len)
    out = torch.full_like(q, float("nan"))
    execute(native, algorithm, q, k, v, ks, vs, out)
    check_attention(q, k, v, ks, vs, out)


def test_shared_dense_graph_uses_changed_scale_values(native):
    from vllm.v1.attention.kv_codecs import INT8

    q, k, v, ks, vs = inputs(INT8, 1, 1, 64, 96)
    out = torch.empty_like(q)
    execute(native, "dense", q, k, v, ks, vs, out)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute(native, "dense", q, k, v, ks, vs, out)
    for factor in (0.5, 2.0):
        ks.mul_(factor)
        vs.mul_(1 / factor)
        graph.replay()
        check_attention(q, k, v, ks, vs, out)
