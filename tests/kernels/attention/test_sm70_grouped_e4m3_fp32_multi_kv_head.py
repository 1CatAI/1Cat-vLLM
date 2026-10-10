# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-KV E4M3 and request-major SM70 verifier coverage.

The current E4M3 native entry handles all KV heads in one call. Its backend
admission helper accepts GQA layouts where each KV head owns six query heads;
the native implementation then preserves each head's result. Legacy E5M2
verification is covered through the current DFlash2 admission and call path.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops import sm70_e4m3_grouped as ops

GQA_GROUP_SIZE = 6


def _native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    module = pytest.importorskip("flash_attn_v100")
    if not module.flash_attn_grouped_e4m3_fp32_available():
        pytest.skip("rebuild native E4M3 FP32 entry")
    return module.flash_attn_grouped_e4m3_fp32_paged


class _Instance:
    def __init__(self, op):
        self.flash_attn_grouped_e4m3_fp32_paged = op
        self.kv_cache_dtype = "fp8_e4m3"
        self.use_smallq_decode_xqa = True

    def _flash_v100_window_size(self, causal=True):
        return (-1, -1)


def _case(rows, page, length, num_kv_heads, seed):
    torch.manual_seed(seed)
    pages = (length + page - 1) // page
    capacity = pages * page
    q = torch.randn(
        (rows, GQA_GROUP_SIZE * num_kv_heads, 256), device="cuda", dtype=torch.float16
    )
    raw = torch.randn(
        (2, capacity, num_kv_heads, 256), device="cuda", dtype=torch.float16
    )
    encoded = raw.to(torch.float8_e4m3fn).view(torch.uint8)
    backing = torch.empty(
        (pages, 2, page, num_kv_heads, 256), device="cuda", dtype=torch.uint8
    )
    k, v = backing.unbind(1)
    order = torch.randperm(pages, device="cuda")
    k[order] = encoded[0].reshape_as(k)
    v[order] = encoded[1].reshape_as(v)
    parent_table = order.int()[None].contiguous()
    parent_seq = torch.tensor([length], device="cuda", dtype=torch.int32)
    lengths = torch.arange(
        length - rows + 1, length + 1, device="cuda", dtype=torch.int32
    )
    table = parent_table.expand(rows, -1).contiguous()
    metadata = SimpleNamespace(
        block_table=parent_table, seq_lens=parent_seq, causal=True
    )
    return q, k, v, encoded, table, lengths, metadata


def _reference_single_head(op, q, k, v, table, lengths, head, ks, vs):
    heads = slice(head * GQA_GROUP_SIZE, (head + 1) * GQA_GROUP_SIZE)
    q_h = q[:, heads, :].contiguous()
    k_h = k[:, :, head : head + 1, :].contiguous()
    v_h = v[:, :, head : head + 1, :].contiguous()
    out_h = torch.empty_like(q_h)
    op(
        q_h,
        k_h,
        v_h,
        table[:1],
        lengths,
        out=out_h,
        softmax_scale=0.0625,
        k_scale=ks,
        v_scale=vs,
    )
    return out_h


@pytest.mark.parametrize(
    "rows,page,length",
    [
        (5, 1616, 8197),
        (8, 1648, 8197),
        (5, 1616, 65536),
        (8, 1648, 131072),
        (2, 848, 1700),
    ],
)
def test_multi_kv_head_matches_contiguous_single_head_and_oracle(
    monkeypatch, rows, page, length
):
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    op = _native()
    num_kv_heads = 2
    q, k, v, encoded, table, lengths, metadata = _case(
        rows, page, length, num_kv_heads, 20260914
    )
    ks, vs = 0.5, 1.25
    out = torch.empty_like(q)
    instance = _Instance(op)
    assert ops.grouped_e4m3_fp32_allowed(
        instance,
        q,
        k,
        v,
        table,
        lengths,
        metadata,
        out=out,
        partition_size_hint=None,
    )
    # This is the exact upstream backend dispatch: the gate consumes the
    # expanded per-query table, while the operator receives request metadata.
    op(
        q,
        k,
        v,
        metadata.block_table,
        lengths,
        out=out,
        softmax_scale=0.0625,
        k_scale=ks,
        v_scale=vs,
    )
    for head in range(num_kv_heads):
        expected = _reference_single_head(op, q, k, v, table, lengths, head, ks, vs)
        heads = slice(head * GQA_GROUP_SIZE, (head + 1) * GQA_GROUP_SIZE)
        assert torch.equal(out[:, heads, :], expected)
        # FP64 oracle on the same quantized KV, as in the single-head tests.
        rk = encoded[0, :length, head].view(torch.float8_e4m3fn).double() * ks
        rv = encoded[1, :length, head].view(torch.float8_e4m3fn).double() * vs
        score = q[:, heads, :].transpose(0, 1).double() @ rk.T * 0.0625
        mask = torch.arange(length, device="cuda")[None] >= lengths[:, None]
        score.masked_fill_(mask[None], -torch.inf)
        oracle = (score.softmax(-1) @ rv).transpose(0, 1)
        relative_l2 = (out[:, heads, :].double() - oracle).norm() / oracle.norm()
        assert float(relative_l2) < 0.001


def test_multi_kv_head_is_graph_replayable(monkeypatch):
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    op = _native()
    rows, page, length = 5, 1616, 4000
    q, k, v, _, table, lengths, metadata = _case(rows, page, length, 2, 20260915)
    out = torch.empty_like(q)
    instance = _Instance(op)

    def call():
        assert ops.grouped_e4m3_fp32_allowed(
            instance,
            q,
            k,
            v,
            table,
            lengths,
            metadata,
            out=out,
            partition_size_hint=None,
        )
        op(
            q,
            k,
            v,
            metadata.block_table,
            lengths,
            out=out,
            softmax_scale=0.0625,
            k_scale=1.0,
            v_scale=1.0,
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(2):
        q.normal_()
        expected = torch.cat(
            [
                _reference_single_head(op, q, k, v, table, lengths, head, 1.0, 1.0)
                for head in range(2)
            ],
            dim=1,
        )
        graph.replay()
        assert torch.equal(out, expected)


def test_upstream_route_admits_multi_kv_and_rejects_bad_shapes(monkeypatch):
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    op = _native()
    rows, page, length = 5, 1616, 4000
    q, k, v, _, table, lengths, metadata = _case(rows, page, length, 2, 20260916)
    instance = _Instance(op)
    out = torch.empty_like(q)
    assert ops.grouped_e4m3_fp32_allowed(
        instance,
        q,
        k,
        v,
        table,
        lengths,
        metadata,
        out=out,
        partition_size_hint=None,
    )
    # The upstream route has no multi-KV opt-in switch. Its public gate still
    # rejects unsupported operator settings and malformed GQA layouts.
    instance.use_smallq_decode_xqa = False
    assert not ops.grouped_e4m3_fp32_allowed(
        instance,
        q,
        k,
        v,
        table,
        lengths,
        metadata,
        out=out,
        partition_size_hint=None,
    )
    instance.use_smallq_decode_xqa = True
    wrong_heads = q[:, :-1]
    assert not ops.grouped_e4m3_fp32_allowed(
        instance,
        wrong_heads,
        k,
        v,
        table,
        lengths,
        metadata,
        out=out[:, :-1],
        partition_size_hint=None,
    )


# ---------------------------------------------------------------------------
# Legacy E5M2 one-pass verifier (DFlash2 q8) on per-KV-head strided views.
# ---------------------------------------------------------------------------


def _native_e5m2():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    module = pytest.importorskip("flash_attn_v100")
    return module.flash_attn_grouped_verify_paged


@pytest.mark.parametrize("page,length", [(1648, 8197), (1648, 65536), (3296, 131072)])
def test_e5m2_one_pass_verifier_per_kv_head_matches_contiguous(page, length):
    op = _native_e5m2()
    torch.manual_seed(20260917)
    rows, num_kv_heads = 8, 2
    pages = (length + page - 1) // page
    capacity = pages * page
    q = torch.randn(
        (rows, GQA_GROUP_SIZE * num_kv_heads, 256), device="cuda", dtype=torch.float16
    )
    raw = torch.randn(
        (2, capacity, num_kv_heads, 256), device="cuda", dtype=torch.float16
    )
    encoded = raw.to(torch.float8_e5m2).view(torch.uint8)
    backing = torch.empty(
        (pages, 2, page, num_kv_heads, 256), device="cuda", dtype=torch.uint8
    )
    k, v = backing.unbind(1)
    order = torch.randperm(pages, device="cuda")
    k[order] = encoded[0].reshape_as(k)
    v[order] = encoded[1].reshape_as(v)
    table = order.int()[None].contiguous()
    seq_lens = torch.tensor([length], device="cuda", dtype=torch.int32)
    for head in range(num_kv_heads):
        heads = slice(head * GQA_GROUP_SIZE, (head + 1) * GQA_GROUP_SIZE)
        q_h = q[:, heads, :].contiguous()
        # Deliberately retain the per-head page-stride view here: E5M2 still
        # uses the legacy verifier and does not use the E4M3 grouping helper.
        k_h = k[:, :, head : head + 1, :]
        v_h = v[:, :, head : head + 1, :]
        # The Python wrapper makes a non-contiguous ``out`` contiguous and
        # returns that tensor; a strided output view would not be updated.
        out_h = torch.empty_like(q_h)
        out_h = op(
            q_h,
            k_h,
            v_h,
            table,
            seq_lens,
            softmax_scale=0.0625,
            out=out_h,
            kv_cache_dtype="fp8_e5m2",
            k_scale=0.5,
            v_scale=1.25,
            one_pass=True,
        )
        expected = torch.empty_like(q_h)
        expected = op(
            q_h,
            k_h.contiguous(),
            v_h.contiguous(),
            table,
            seq_lens,
            softmax_scale=0.0625,
            out=expected,
            kv_cache_dtype="fp8_e5m2",
            k_scale=0.5,
            v_scale=1.25,
            one_pass=True,
        )
        assert torch.equal(out_h, expected)
        # FP64 oracle on the same quantized KV (causal rows ending at length).
        rk = encoded[0, :length, head].view(torch.float8_e5m2).double() * 0.5
        rv = encoded[1, :length, head].view(torch.float8_e5m2).double() * 1.25
        lengths = torch.arange(length - rows + 1, length + 1, device="cuda")
        score = q_h.transpose(0, 1).double() @ rk.T * 0.0625
        mask = torch.arange(length, device="cuda")[None] >= lengths[:, None]
        score.masked_fill_(mask[None], -torch.inf)
        oracle = (score.softmax(-1) @ rv).transpose(0, 1)
        relative_l2 = (out_h.double() - oracle).norm() / oracle.norm()
        assert float(relative_l2) < 0.01


# ---------------------------------------------------------------------------
# Multi-request verify batch on a rank holding a single KV head (TP4 layout).
# The current backend admits one request with q8/q16, or request-major q8
# batches when the native ABI supports them. These tests use its per-request
# route, comparing each result to an independent native invocation.
# ---------------------------------------------------------------------------


def _build_impl(kv_dtype, *, e5m2_op=None, e4m3_op=None):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    impl = FlashAttnV100Impl(
        num_heads=6,
        head_size=256,
        scale=0.0625,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_dtype,
    )
    if e5m2_op is not None:
        impl.use_dflash2_grouped_verify = True
        impl.dflash2_grouped_verify_max_query_tokens = 16
        impl.dflash2_grouped_verify_min_model_len = 32768
        impl.flash_attn_grouped_verify_paged = e5m2_op
    if e4m3_op is not None:
        impl.use_smallq_decode_xqa = True
        impl.flash_attn_grouped_e4m3_fp32_paged = e4m3_op
    return impl


def _single_head_multi_request_case(q_per_req, lengths_req, page, fp8_dtype, seed):
    """Disjoint physical pages and a distinct length per request."""
    torch.manual_seed(seed)
    num_reqs = len(lengths_req)
    ppr = max((length + page - 1) // page for length in lengths_req)
    capacity = ppr * page
    total_pages = ppr * num_reqs
    query = torch.randn(
        (q_per_req * num_reqs, GQA_GROUP_SIZE, 256), device="cuda", dtype=torch.float16
    )
    backing = torch.empty(
        (total_pages, 2, page, 1, 256), device="cuda", dtype=torch.uint8
    )
    k, v = backing.unbind(1)
    parent_rows = []
    encoded = []
    for i in range(num_reqs):
        raw = torch.randn((2, capacity, 1, 256), device="cuda", dtype=torch.float16)
        enc = raw.to(fp8_dtype).view(torch.uint8)
        encoded.append(enc)
        perm = torch.randperm(ppr, device="cuda") + i * ppr
        k[perm] = enc[0].reshape(ppr, page, 1, 256)
        v[perm] = enc[1].reshape(ppr, page, 1, 256)
        parent_rows.append(perm.int())
    parent_table = torch.stack(parent_rows).contiguous()
    parent_seq = torch.tensor(lengths_req, device="cuda", dtype=torch.int32)
    block_table = parent_table.repeat_interleave(q_per_req, dim=0).contiguous()
    seq_lens = torch.cat(
        [
            torch.arange(
                length - q_per_req + 1, length + 1, device="cuda", dtype=torch.int32
            )
            for length in lengths_req
        ]
    )
    metadata = SimpleNamespace(
        block_table=parent_table,
        seq_lens=parent_seq,
        num_reqs=num_reqs,
        max_query_len=q_per_req,
        is_dflash_selector_target=True,
        max_model_len=262144,
        causal=True,
    )
    return SimpleNamespace(
        query=query,
        k=k,
        v=v,
        encoded=encoded,
        parent_table=parent_table,
        parent_seq=parent_seq,
        block_table=block_table,
        seq_lens=seq_lens,
        metadata=metadata,
        q_per_req=q_per_req,
        num_reqs=num_reqs,
    )


@pytest.mark.parametrize("q_per_req", [8, 16])
def test_dflash2_grouped_verify_per_request_e5m2(q_per_req):
    op = _native_e5m2()
    ks, vs = 0.5, 1.25
    c = _single_head_multi_request_case(
        q_per_req, [8197, 5003], 1648, torch.float8_e5m2, 20260918
    )
    impl = _build_impl("fp8_e5m2", e5m2_op=op)
    layer = SimpleNamespace(_k_scale_float=ks, _v_scale_float=vs)
    out = torch.empty_like(c.query)
    batched = False
    if q_per_req == 8:
        # Upstream's request-major E5M2 path is Q8 per request. Use it when
        # the installed native ABI advertises the batch contract; older
        # extensions remain covered by the supported single-request route.
        module = pytest.importorskip("flash_attn_v100")
        get_abi = getattr(
            module, "flash_attn_grouped_verify_request_major_abi_version", None
        )
        abi_version = 0 if get_abi is None else int(get_abi())
        if abi_version >= 1:
            impl.use_dflash2_batched_grouped_verify = True
            impl.dflash2_grouped_verify_request_major_abi_version = abi_version
            assert impl._dflash2_grouped_verify_allowed(
                c.query,
                c.k,
                c.v,
                c.metadata,
                num_query_tokens=c.query.shape[0],
            )
            impl._call_dflash2_grouped_verify(
                layer, c.query, c.k, c.v, c.metadata, out=out
            )
            batched = True
    for i in range(c.num_reqs):
        rows = slice(i * c.q_per_req, (i + 1) * c.q_per_req)
        q_i = c.query[rows].contiguous()
        metadata_i = SimpleNamespace(
            block_table=c.parent_table[i : i + 1],
            seq_lens=c.parent_seq[i : i + 1],
            is_dflash_selector_target=True,
            max_model_len=262144,
            max_query_len=c.q_per_req,
            num_reqs=1,
            causal=True,
        )
        if not batched:
            # Exercise the current single-request Q8/Q16 DFlash2 path.
            assert impl._dflash2_grouped_verify_allowed(
                q_i,
                c.k,
                c.v,
                metadata_i,
                num_query_tokens=c.q_per_req,
            )
            impl._call_dflash2_grouped_verify(
                layer,
                q_i,
                c.k,
                c.v,
                metadata_i,
                out=out[rows],
            )
        expected = torch.empty_like(q_i)
        op(
            q_i,
            c.k,
            c.v,
            c.parent_table[i : i + 1],
            c.parent_seq[i : i + 1],
            softmax_scale=0.0625,
            out=expected,
            kv_cache_dtype="fp8_e5m2",
            k_scale=ks,
            v_scale=vs,
            one_pass=True,
        )
        assert torch.equal(out[rows], expected)
        # FP64 oracle on this request's quantized KV (causal, ending at length).
        length = int(c.parent_seq[i])
        rk = c.encoded[i][0, :length, 0].view(torch.float8_e5m2).double() * ks
        rv = c.encoded[i][1, :length, 0].view(torch.float8_e5m2).double() * vs
        ends = torch.arange(length - c.q_per_req + 1, length + 1, device="cuda")
        score = q_i.transpose(0, 1).double() @ rk.T * 0.0625
        mask = torch.arange(length, device="cuda")[None] >= ends[:, None]
        score.masked_fill_(mask[None], -torch.inf)
        oracle = (score.softmax(-1) @ rv).transpose(0, 1)
        relative_l2 = (out[rows].double() - oracle).norm() / oracle.norm()
        assert float(relative_l2) < 0.01


def test_e4m3_grouped_fp32_per_request_upstream_route(monkeypatch):
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    op = _native()
    ks, vs = 0.5, 1.25
    c = _single_head_multi_request_case(
        4, [8197, 5003], 1648, torch.float8_e4m3fn, 20260919
    )
    impl = _build_impl("fp8_e4m3", e4m3_op=op)
    out = torch.empty_like(c.query)
    for i in range(c.num_reqs):
        rows = slice(i * c.q_per_req, (i + 1) * c.q_per_req)
        q_i = c.query[rows].contiguous()
        length_rows = c.seq_lens[rows]
        metadata_i = SimpleNamespace(
            block_table=c.parent_table[i : i + 1],
            seq_lens=c.parent_seq[i : i + 1],
            causal=True,
        )
        out_i = out[rows]
        assert ops.grouped_e4m3_fp32_allowed(
            impl,
            q_i,
            c.k,
            c.v,
            c.block_table[rows],
            length_rows,
            metadata_i,
            out=out_i,
            partition_size_hint=None,
        )
        # E4M3 uses the request's block-table row and its per-token lengths.
        op(
            q_i,
            c.k,
            c.v,
            c.parent_table[i : i + 1],
            length_rows,
            out=out_i,
            softmax_scale=0.0625,
            k_scale=ks,
            v_scale=vs,
        )
        expected = torch.empty_like(q_i)
        op(
            q_i,
            c.k,
            c.v,
            c.parent_table[i : i + 1],
            length_rows,
            out=expected,
            softmax_scale=0.0625,
            k_scale=ks,
            v_scale=vs,
        )
        assert torch.equal(out_i, expected)
        length = int(c.parent_seq[i])
        rk = c.encoded[i][0, :length, 0].view(torch.float8_e4m3fn).double() * ks
        rv = c.encoded[i][1, :length, 0].view(torch.float8_e4m3fn).double() * vs
        score = q_i.transpose(0, 1).double() @ rk.T * 0.0625
        mask = torch.arange(length, device="cuda")[None] >= length_rows[:, None]
        score.masked_fill_(mask[None], -torch.inf)
        oracle = (score.softmax(-1) @ rv).transpose(0, 1)
        relative_l2 = (out_i.double() - oracle).norm() / oracle.norm()
        assert float(relative_l2) < 0.001
