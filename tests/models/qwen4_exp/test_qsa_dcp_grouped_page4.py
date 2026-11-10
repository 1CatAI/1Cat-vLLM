# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.config.sm70_sparse import Sm70SparseConfig
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    qsa_dcp_sparse_paged_attention_sm70_grouped_page4,
    qsa_sparse_paged_attention,
)
from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import (
    qsa_dcp_local_selection_width,
    qsa_localize_dcp_indices,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skip_global_cleanup

TOPK, RATIO, WORLD, INTERLEAVE = 2048, 4, 2, 1
WIDTH = TOPK + RATIO - 1


@pytest.mark.parametrize("enabled", [False, True])
def test_forward_uses_layer_owned_admission(monkeypatch, enabled):
    import vllm.distributed
    from vllm.models.qwen4_exp.nvidia import qsa as model_qsa
    from vllm.models.qwen4_exp.nvidia.ops import qsa_dcp

    rows, heads = 8, 6
    query = torch.ones((rows, heads, 256), dtype=torch.float16)
    output = torch.empty_like(query)
    layer = torch.nn.Module()
    layer._sm70_qsa_dcp_grouped_page4 = enabled
    layer.dcp_local_indices_buffer = torch.empty((rows, WIDTH), dtype=torch.int32)
    layer.dcp_partial_output_buffer = torch.empty((rows, 2 * heads, 256))
    layer.dcp_partial_lse_buffer = torch.empty((rows, 2 * heads))
    layer.cp_kv_cache_interleave_size = 1
    layer.indexer = SimpleNamespace(token_topk=TOPK, compress_ratio=RATIO)
    layer._k_scale_float = layer._v_scale_float = 1.0
    impl = object.__new__(model_qsa.Qwen4ExpQSAFlashAttentionImpl)
    impl.dcp_world_size, impl.dcp_rank = 2, 0
    impl.kv_cache_dtype = "auto"
    impl.dcp_combine = lambda partial, *_args, **_kw: partial[:, :heads]
    monkeypatch.setattr(
        vllm.distributed,
        "get_dcp_group",
        lambda: SimpleNamespace(
            all_gather=lambda q, dim: torch.cat([q, q], dim=dim),
        ),
    )
    monkeypatch.setattr(
        qsa_dcp,
        "qsa_localize_dcp_indices",
        lambda indices, out, **_kw: out.copy_(indices),
    )
    calls = []

    def grouped(q, k, v, indices, table, request, out, lse, *_args):
        calls.append("grouped")
        out.fill_(1)
        lse.zero_()
        return q.shape[0]

    def fallback(q, k, v, indices, table, request, out, **kwargs):
        calls.append("fallback")
        out.fill_(2)
        kwargs["lse"].zero_()

    monkeypatch.setattr(
        qsa_ops, "qsa_dcp_sparse_paged_attention_sm70_grouped_page4", grouped
    )
    monkeypatch.setattr(qsa_ops, "qsa_sparse_paged_attention", fallback)
    monkeypatch.setattr(qsa_ops, "_qsa_output_gate", lambda out, gate: out.mul_(gate))
    cache = torch.empty((2, 16, 1, 256), dtype=torch.float16)
    impl._forward_qsa_dcp(
        layer,
        query,
        cache,
        cache,
        torch.zeros((rows, WIDTH), dtype=torch.int32),
        torch.zeros((1, 2), dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        output,
        torch.ones_like(output),
    )
    assert calls == ["grouped" if enabled else "fallback"]
    assert torch.all(output == (1 if enabled else 2))


@pytest.fixture(autouse=True)
def explicit_grouped_dcp_policy(monkeypatch):
    policy = Sm70SparseConfig(
        qsa_dcp_grouped_page4=True,
        qsa_xqa_page4=True,
        qsa_grouped_page4=True,
        qsa_grouped_pad_fix=True,
    )
    monkeypatch.setattr(qsa_ops, "sparse_policy", lambda: policy)
    return policy


def test_dcp_partial_rounding_requires_explicit_policy():
    default = Sm70SparseConfig()
    enabled = Sm70SparseConfig(qsa_dcp_grouped_page4=True)
    default.active = enabled.active = True
    assert not default.qsa_dcp_grouped_page4
    assert default.compute_hash() != enabled.compute_hash()
    default.qualify("indexer")
    enabled.qualify("indexer")
    assert default.compute_hash() == enabled.compute_hash()


@pytest.mark.parametrize("heads", [6, 12, 18])
@pytest.mark.parametrize("cache", ["auto", "fp8_e4m3"])
def test_startup_capability_admits_native_head_groups(monkeypatch, heads, cache):
    monkeypatch.setattr(current_platform, "is_device_capability", lambda _: True)
    monkeypatch.setattr(qsa_ops, "_qsa_grouped_page4_supported", lambda *_: True)
    monkeypatch.setitem(
        sys.modules,
        "flash_attn_v100.flash_attn_interface",
        SimpleNamespace(flash_attn_v100_cuda=object()),
    )
    assert (
        qsa_ops.qsa_dcp_grouped_page4_config_reason(
            sharded=True,
            query_heads=heads,
            kv_heads=1,
            head_dim=256,
            dtype=torch.float16,
            kv_cache_dtype=cache,
            selection_width=2051,
            max_rows=8192,
        )
        is None
    )


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"sharded": False}, "cache_not_dcp_sharded"),
        ({"query_heads": 8}, "requires_six_head_groups_one_kv_head_and_dim256"),
        ({"kv_heads": 2}, "requires_six_head_groups_one_kv_head_and_dim256"),
        ({"head_dim": 128}, "requires_six_head_groups_one_kv_head_and_dim256"),
        ({"dtype": torch.bfloat16}, "requires_fp16_queries_and_fp16_or_e4m3_cache"),
        ({"selection_width": 1027}, "requires_grouped_page4_selection_width2051"),
        ({"max_rows": 255}, "prefill_budget_below_grouped_route_threshold"),
    ],
)
def test_startup_capability_reports_rejection(monkeypatch, change, reason):
    monkeypatch.setattr(current_platform, "is_device_capability", lambda _: True)
    config = dict(
        sharded=True,
        query_heads=12,
        kv_heads=1,
        head_dim=256,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        selection_width=2051,
        max_rows=8192,
    )
    assert qsa_ops.qsa_dcp_grouped_page4_config_reason(**(config | change)) == reason


def test_small_batches_stay_on_triton():
    # Decode-sized batches never reach the grouped route, on any platform.
    rows = qsa_ops._SM70_QSA_DCP_GROUPED_PAGE4_MIN_ROWS - 1
    q = torch.zeros((rows, 12, 256), dtype=torch.float16)
    written = qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
        q,
        torch.zeros((4, 16, 1, 256), dtype=torch.uint8),
        torch.zeros((4, 16, 1, 256), dtype=torch.uint8),
        torch.full((rows, WIDTH), -1, dtype=torch.int32),
        torch.zeros((1, 4), dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.zeros(q.shape, dtype=torch.float32),
        torch.zeros(q.shape[:2], dtype=torch.float32),
        "fp8_e4m3",
        1.0,
        1.0,
    )
    assert written == 0


def _selection(positions, generator):
    """Model-format global selection: score-ordered blocks plus the open tail."""
    rows = []
    for position in positions.tolist():
        visible = (position + 1) // RATIO
        blocks = torch.randperm(visible, generator=generator)[: TOPK // RATIO]
        tokens = (blocks[:, None] * RATIO + torch.arange(RATIO)).flatten().tolist()
        tail_start = visible * RATIO
        tokens += list(range(tail_start, position + 1))
        rows.append(tokens + [-1] * (WIDTH - len(tokens)))
    return torch.tensor(rows, dtype=torch.int32)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_device_capability(70),
    reason="SM70 grouped page4 route",
)
@pytest.mark.parametrize("heads,graph_replay", [(12, False), (6, True), (18, True)])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("e4m3", [False, True])
def test_grouped_route_matches_triton_lse_route(e4m3, packed, heads, graph_replay):
    flash = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    kv_dtype = "fp8_e4m3" if e4m3 else "auto"
    if not qsa_ops._qsa_grouped_page4_supported(flash.flash_attn_v100_cuda, kv_dtype):
        pytest.skip("Flash-V100 build has no grouped page4 route for this cache")
    generator = torch.Generator().manual_seed(816)
    page, pages, dim = 16, 96, 256
    # Request 0 selects 512 of its visible blocks; request 1 is short enough
    # to select every visible position. 257 rows exercise the real 256-row
    # admission boundary and leave one row for the existing fallback.
    positions = torch.cat([torch.arange(2400, 2530), torch.arange(100, 227)])
    token_to_req = torch.tensor([0] * 130 + [1] * 127, dtype=torch.int32)
    indices = _selection(positions, generator)
    indices[3] = -1  # A row without any selected position.
    indices, token_to_req = indices.cuda(), token_to_req.cuda()
    q = (torch.randn((257, heads, dim), generator=generator) * 0.3).half().cuda()
    # Interleaved K/V blocks, as the model's cache ABI stores them. A sharded
    # DCP cache packs one page of each member layer into every physical block.
    members = 2 if packed else 1
    cache = torch.randn((2 * pages + 3, members, 2, page, 1, dim), generator=generator)
    cache *= 0.5
    k_scale, v_scale = (0.25, 0.5) if e4m3 else (1.0, 1.0)
    if e4m3:
        cache[:, :, 0] /= k_scale
        cache[:, :, 1] /= v_scale
        cache = cache.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        cache = cache.half()
    k_cache, v_cache = cache.cuda()[:, members - 1].unbind(1)
    assert k_cache.stride(0) == members * 2 * page * dim
    block_table = (
        torch.randperm(2 * pages + 3, generator=generator)[: 2 * pages]
        .view(2, pages)
        .to(torch.int32)
        .cuda()
    )
    local_width = qsa_dcp_local_selection_width(TOPK, RATIO, WORLD, INTERLEAVE, WIDTH)

    for rank in range(WORLD):
        local = torch.empty_like(indices)
        qsa_localize_dcp_indices(
            indices,
            local,
            dcp_world_size=WORLD,
            dcp_rank=rank,
            interleave_size=INTERLEAVE,
            local_block_size=page,
        )
        expected_out = torch.empty(q.shape, dtype=torch.float32, device="cuda")
        expected_lse = torch.empty(q.shape[:2], dtype=torch.float32, device="cuda")
        qsa_sparse_paged_attention(
            q,
            k_cache,
            v_cache,
            local[:, :local_width],
            block_table,
            token_to_req,
            expected_out,
            kv_cache_dtype=kv_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            lse=expected_lse,
        )
        out = torch.full_like(expected_out, torch.nan)
        lse = torch.full_like(expected_lse, torch.nan)
        for unsupported in ("fp8_e5m2", "int8_per_token_head"):
            assert (
                qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
                    q,
                    k_cache,
                    v_cache,
                    local,
                    block_table,
                    token_to_req,
                    out,
                    lse,
                    unsupported,
                    k_scale,
                    v_scale,
                )
                == 0
            )
            assert torch.isnan(out).all() and torch.isnan(lse).all()
        assert (
            qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
                q,
                k_cache,
                v_cache,
                local,
                block_table,
                token_to_req,
                out,
                lse.half(),
                kv_dtype,
                k_scale,
                v_scale,
            )
            == 0
        )
        assert torch.isnan(out).all() and torch.isnan(lse).all()
        rows = qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
            q,
            k_cache,
            v_cache,
            local,
            block_table,
            token_to_req,
            out,
            lse,
            kv_dtype,
            k_scale,
            v_scale,
        )
        if graph_replay:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
                    q,
                    k_cache,
                    v_cache,
                    local,
                    block_table,
                    token_to_req,
                    out,
                    lse,
                    kv_dtype,
                    k_scale,
                    v_scale,
                )
            q.mul_(0.9)
            local[0].fill_(-1)
            graph.replay()
            qsa_sparse_paged_attention(
                q,
                k_cache,
                v_cache,
                local[:, :local_width],
                block_table,
                token_to_req,
                expected_out,
                kv_cache_dtype=kv_dtype,
                k_scale=k_scale,
                v_scale=v_scale,
                lse=expected_lse,
            )
        assert rows == 256
        assert torch.isnan(out[rows:]).all() and torch.isnan(lse[rows:]).all()
        assert torch.isneginf(lse[3]).all() and not out[3].any()
        torch.testing.assert_close(lse[:rows], expected_lse[:rows], rtol=0, atol=1e-3)
        torch.testing.assert_close(
            out[:rows], expected_out[:rows], rtol=1e-2, atol=3e-3
        )
