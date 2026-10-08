# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative route coverage and differential legacy XQA admission."""

import ast
import itertools
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm.envs as envs
from vllm.v1.attention.backends.flash_v100 import routing as r
from vllm.v1.attention.kv_codecs import BF16, FP8_E4M3, FP8_E5M2, FP16, KV_CODECS

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    monkeypatch.setenv("VLLM_FLASH_V100_E4M3_BATCH_XQA", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_DECODE_XQA_Q4_MIN_SEQ_LEN", "32768")
    monkeypatch.setenv("VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN", "16384")
    monkeypatch.setenv("VLLM_FLASH_V100_SMALLQ_DECODE_XQA_MIN_SEQ_LEN", "4096")
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


@pytest.fixture(scope="module")
def legacy():
    data = json.loads(
        (Path(__file__).parent / "fixtures/flash_v100_xqa_legacy.json").read_text()
    )
    namespace = {
        "_routing": r,
        "os": r.os,
        "torch": r.torch,
        "TritonAttentionMetadata": r.TritonAttentionMetadata,
        "FP8_E4M3": FP8_E4M3,
        "FP8_E5M2": FP8_E5M2,
    }
    exec(data["verify"], namespace)
    return data, namespace["_smallq_decode_xqa_allowed"]


def context(stage, codec=FP16, ratio=6, rows=1, length=32768):
    query = SimpleNamespace(shape=(rows, ratio, 256), ndim=3, is_cuda=False)
    metadata = SimpleNamespace(
        seq_lens=SimpleNamespace(shape=(rows,)),
        flash_v100_decode_max_seq_len_hint=length,
        flash_v100_decode_workspace_seq_capacity_hint=length,
        flash_v100_static_decode_seq_hint=None,
        flash_v100_cudagraph_capture=False,
    )
    return r.RouteContext(
        stage=stage,
        codec=codec,
        shape=r.RouteShape(rows, ratio, 1, 256, 848),
        query=query,
        metadata=metadata,
        seq_rows=rows if stage != "mixed_decode" else None,
        max_seq_len_hint=length,
        workspace_seq_capacity_hint=length,
    )


def legacy_allowed(ctx, legacy):
    data, verify = legacy
    shape = ctx.shape
    k = SimpleNamespace(shape=(1, shape.page_size, shape.heads_kv, shape.head_dim))
    native_codec = ctx.codec if ctx.codec in (FP16, FP8_E4M3, FP8_E5M2) else None
    instance = SimpleNamespace(
        use_decode_xqa=ctx.enabled,
        use_smallq_decode_xqa=ctx.enabled,
        flash_attn_decode_paged_xqa=object() if ctx.available else None,
        _xqa_kv_codec=lambda *args: native_codec,
    )
    if ctx.stage == "verify":
        return verify(
            instance,
            ctx.query,
            k,
            k,
            SimpleNamespace(shape=(ctx.seq_rows,)),
            ctx.metadata,
            window_size=ctx.window_size,
            max_seq_len_hint=ctx.max_seq_len_hint,
            workspace_seq_capacity_hint=ctx.workspace_seq_capacity_hint,
            partition_size_hint=ctx.partition_size_hint,
        )
    namespace = dict(
        _routing=r,
        envs=envs,
        FP8_E4M3=FP8_E4M3,
        FP8_E5M2=FP8_E5M2,
        self=instance,
        query=ctx.query,
        key_cache=k,
        attn_metadata=ctx.metadata,
        window_size=ctx.window_size,
        q_per_kv=shape.gqa,
        xqa_codec=native_codec,
        fp8_e4m3_kv=ctx.codec is FP8_E4M3,
        fp8_e5m2_kv=ctx.codec is FP8_E5M2,
        num_rows=shape.rows,
        num_heads=shape.heads_q,
        max_seq_len_hint=ctx.max_seq_len_hint,
    )
    return eval(data["uniform" if ctx.stage == "decode" else "mixed"], namespace)


_XQA = {
    "decode": "decode_xqa_paged",
    "verify": "prefill_smallq_decode_xqa",
    "mixed_decode": "prefill_prefix_decode_rows_xqa",
}


@pytest.mark.parametrize("stage", _XQA)
@pytest.mark.parametrize("codec", (FP16, FP8_E4M3, FP8_E5M2, BF16, None))
def test_xqa_boundary_matrix_matches_frozen_legacy(stage, codec, legacy):
    for ratio, rows, length in itertools.product(
        (4, 6, 8), (0, 1, 2, 32), (0, 4095, 4096, 16383, 16384, 32767, 32768)
    ):
        ctx = context(stage, codec, ratio, rows, length)
        result = r.select_route(ctx, (_XQA[stage],)) is not None
        assert result == legacy_allowed(ctx, legacy), ctx


@pytest.mark.parametrize("stage", _XQA)
def test_availability_window_partition_and_graph_hints_match_legacy(stage, legacy):
    for codec, enabled, available, window, partition, capture in itertools.product(
        (FP16, FP8_E4M3, FP8_E5M2),
        (False, True),
        (False, True),
        ((-1, -1), (127, 0)),
        (None, 256),
        (False, True),
    ):
        ctx = context(stage, codec, rows=2, length=1024)
        ctx.metadata.flash_v100_cudagraph_capture = capture
        ctx.metadata.flash_v100_decode_workspace_seq_capacity_hint = 65536
        ctx = replace(
            ctx,
            enabled=enabled,
            available=available,
            window_size=window,
            partition_size_hint=partition,
            workspace_seq_capacity_hint=65536,
        )
        assert (r.select_route(ctx, (_XQA[stage],)) is not None) == legacy_allowed(
            ctx, legacy
        ), ctx


@pytest.mark.parametrize("spec", tuple(r.ROUTE_SPECS.values()), ids=lambda s: s.name)
@pytest.mark.parametrize("codec", KV_CODECS, ids=lambda c: c.name)
def test_generated_path_codec_shape_matrix(spec, codec):
    chunk = (
        spec.chunk_sizes[0]
        if spec.chunk_sizes
        else max(spec.min_chunk, spec.chunk_alignment)
    )
    chunk = (
        (chunk + spec.chunk_alignment - 1)
        // spec.chunk_alignment
        * spec.chunk_alignment
    )
    shape = r.RouteShape(
        1,
        spec.gqa_ratios[0] if spec.gqa_ratios else 6,
        1,
        spec.head_dims[0] if spec.head_dims else 256,
        spec.page_sizes[0] if spec.page_sizes else 16 * spec.page_alignment,
        chunk,
    )
    assert (spec.shape_reason(codec, shape) is None) == (codec in spec.codecs)
    if codec not in spec.codecs:
        return
    if spec.head_dims:
        assert spec.shape_reason(codec, replace(shape, head_dim=0)) == "head_dim"
    if spec.gqa_ratios:
        assert spec.shape_reason(codec, replace(shape, heads_kv=0)) == "gqa"
    if spec.page_alignment > 1:
        assert spec.shape_reason(codec, replace(shape, page_size=1)) == "page_alignment"
    if spec.page_sizes:
        assert (
            spec.shape_reason(
                codec, replace(shape, page_size=shape.page_size + spec.page_alignment)
            )
            == "page_size"
        )
    if spec.chunk_alignment > 1:
        assert (
            spec.shape_reason(codec, replace(shape, chunk_size=shape.chunk_size + 1))
            == "chunk_alignment"
        )
    if spec.chunk_sizes:
        assert spec.shape_reason(codec, replace(shape, chunk_size=1)) == "chunk_size"
    if spec.min_chunk:
        assert spec.shape_reason(codec, replace(shape, chunk_size=0)) == "min_chunk"
    if spec.max_chunk is not None:
        assert (
            spec.shape_reason(codec, replace(shape, chunk_size=spec.max_chunk + 1))
            == "max_chunk"
        )


def test_every_literal_accounting_site_uses_a_declared_spec():
    package = Path(r.__file__).parent
    hits = []
    for path in package.rglob("*.py"):
        name = str(path.relative_to(package))
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_record_route"
            ):
                continue
            arg = node.args[0]
            assert not isinstance(arg, ast.Constant), (name, node.lineno)
            if isinstance(arg, ast.Attribute) and arg.attr == "name":
                assert isinstance(arg.value, ast.Subscript)
                route = ast.literal_eval(arg.value.slice)
                assert route in r.ROUTE_SPECS
                hits.append(route)
    assert len(set(hits)) == 44


@pytest.mark.parametrize(
    "name",
    (
        "decode_xqa_e4m3_dynamic_page848",
        "decode_xqa_p256_page784",
        "fp8_kv_decode",
        "fp8_kv_decode_scalar_paged",
        "fp8_kv_prefill_prefix",
        "prefill_prefix_contig_splitd_d256",
        "prefill_prefix_gather_splitd_d256",
        "prefill_prefix_paged_splitd_d256",
        "flashinfer_sm70_fixed_entry",
        "flashinfer_sm70_splitkv3_fast_visible",
        "dflash_draft_triton_fallback",
    ),
)
def test_dynamic_names_are_registered(name):
    assert r.route_spec(name)


def test_undeclared_routes_and_implicit_fallbacks_are_rejected():
    with pytest.raises(ValueError, match="Undeclared"):
        r.route_spec("decode_xqa_typo")
    with pytest.raises(ValueError, match="not a declared fallback"):
        r.select_route(
            context("decode", BF16), ("decode_xqa_paged",), fallback="decode_xqa_paged"
        )
    assert r.select_route(
        context("decode", BF16), ("decode_xqa_paged",), fallback="decode_scalar_paged"
    ).fallback


def test_fallback_accounting_preserves_legacy_route_summary(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(r, "logger", logger)
    monkeypatch.setattr(r, "_fallback_counts", {})
    monkeypatch.setattr(r, "_route_counts", {})
    monkeypatch.setattr(r, "_route_summary_registered", True)
    monkeypatch.setattr(r, "_route_summary_enabled", lambda: False)
    r._record_route("decode_scalar_paged")
    assert r._fallback_counts == {"decode_scalar_paged": 1}
    assert r._route_counts == {}
    monkeypatch.setattr(r, "_route_summary_enabled", lambda: True)
    r._record_route("decode_scalar_paged")
    assert r._route_counts == {"decode_scalar_paged": 1}
    assert r._fallback_counts == {"decode_scalar_paged": 2}
    r._log_route_summary()
    logger.info.assert_called_once_with(
        "FLASH_ATTN_V100 route summary: %s", '{"decode_scalar_paged": 1}'
    )
    assert not r._fallback_counts


@pytest.mark.parametrize("stage", _XQA)
def test_disabled_e4m3_batch_matches_legacy(stage, legacy, monkeypatch):
    monkeypatch.setenv("VLLM_FLASH_V100_E4M3_BATCH_XQA", "0")
    envs.disable_envs_cache()
    for rows in (0, 1, 2, 32):
        ctx = context(stage, FP8_E4M3, rows=rows)
        assert (r.select_route(ctx, (_XQA[stage],)) is not None) == legacy_allowed(
            ctx, legacy
        )


def test_native_tree_correction_is_not_reported_as_fallback():
    assert not r.ROUTE_SPECS["prefill_ddtree_triton"].fallback
