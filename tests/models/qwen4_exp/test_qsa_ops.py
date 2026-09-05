# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    _qsa_indexer_cublas_shape_supported,
    _qsa_sparse_launch_profile,
    _qsa_xqa_page4_shape_supported,
    _sm70_qsa_lexicographic_topk_op,
    _use_sm70_qsa_lexicographic_topk,
    _use_sm70_qsa_two_warp_partial,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_pre_ampere_qsa_prefill_uses_narrow_tiles_and_four_warps():
    assert _qsa_sparse_launch_profile(511, 8, True) == (16, 4, 4)
    assert _qsa_sparse_launch_profile(512, 8, True) == (16, 4, 4)
    assert _qsa_sparse_launch_profile(8192, 8, True) == (16, 1, 4)


def test_ampere_qsa_prefill_keeps_gb300_profile():
    assert _qsa_sparse_launch_profile(512, 8, False) == (64, 4, 2)
    assert _qsa_sparse_launch_profile(8192, 8, False) == (64, 1, 2)


@pytest.mark.parametrize("rows", (1, 2, 4, 8, 16))
def test_sm70_qsa_two_warp_partial_accepts_small_decode_family(monkeypatch, rows):
    monkeypatch.setattr(
        qsa_ops.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )
    assert _use_sm70_qsa_two_warp_partial(rows, 6, 256)
    assert not _use_sm70_qsa_two_warp_partial(rows, 4, 256)
    assert not _use_sm70_qsa_two_warp_partial(rows, 6, 128)


def test_sm70_qsa_two_warp_partial_rejects_large_batch_and_other_arch(
    monkeypatch,
):
    monkeypatch.setattr(
        qsa_ops.current_platform,
        "is_device_capability",
        lambda capability: False,
    )
    assert not _use_sm70_qsa_two_warp_partial(8, 6, 256)
    monkeypatch.setattr(
        qsa_ops.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )
    assert not _use_sm70_qsa_two_warp_partial(0, 6, 256)
    assert not _use_sm70_qsa_two_warp_partial(17, 6, 256)


def test_qsa_indexer_cublas_accepts_only_exact_single_request_shape():
    query = torch.empty(8, 4, 128, dtype=torch.float16)
    cache = torch.empty(2, 400, 1, 128, dtype=torch.float16)
    page_table = torch.empty(1, 2, dtype=torch.int32)

    assert _qsa_indexer_cublas_shape_supported(query, cache, page_table)
    assert not _qsa_indexer_cublas_shape_supported(
        query.to(torch.bfloat16), cache, page_table
    )
    assert not _qsa_indexer_cublas_shape_supported(
        query, cache, page_table.expand(2, -1)
    )
    assert not _qsa_indexer_cublas_shape_supported(query[:, :3], cache, page_table)


def test_qsa_xqa_page4_accepts_only_exact_sm70_prefill_shape():
    query = torch.empty(8, 6, 256, dtype=torch.float16)
    key_cache = torch.empty(2, 400, 1, 256, dtype=torch.float16)
    value_cache = torch.empty_like(key_cache)
    indices = torch.empty(8, 2051, dtype=torch.int32)
    page_table = torch.empty(1, 2, dtype=torch.int32)
    token_to_request = torch.zeros(8, dtype=torch.int32)
    query_positions = torch.arange(8, dtype=torch.int64)
    sequence_lengths = torch.full((1,), 8, dtype=torch.int32)

    assert _qsa_xqa_page4_shape_supported(
        query,
        key_cache,
        value_cache,
        indices,
        page_table,
        token_to_request,
        query_positions,
        sequence_lengths,
    )
    strided_query = torch.empty(8, 6, 257, dtype=torch.float16)[..., :256]
    assert _qsa_xqa_page4_shape_supported(
        strided_query,
        key_cache,
        value_cache,
        indices,
        page_table,
        token_to_request,
        query_positions,
        sequence_lengths,
    )
    interleaved_cache = torch.empty(2, 2, 400, 1, 256, dtype=torch.float16)
    interleaved_key_cache, interleaved_value_cache = interleaved_cache.unbind(1)
    assert _qsa_xqa_page4_shape_supported(
        query,
        interleaved_key_cache,
        interleaved_value_cache,
        indices,
        page_table,
        token_to_request,
        query_positions,
        sequence_lengths,
    )
    assert not _qsa_xqa_page4_shape_supported(
        query.to(torch.bfloat16),
        key_cache,
        value_cache,
        indices,
        page_table,
        token_to_request,
        query_positions,
        sequence_lengths,
    )
    assert not _qsa_xqa_page4_shape_supported(
        query,
        key_cache[:, :398],
        value_cache[:, :398],
        indices,
        page_table,
        token_to_request,
        query_positions,
        sequence_lengths,
    )
    assert not _qsa_xqa_page4_shape_supported(
        query,
        key_cache,
        value_cache,
        indices,
        page_table,
        token_to_request,
        query_positions.to(torch.int32),
        sequence_lengths,
    )


def test_qsa_xqa_page4_route_uses_configured_boundary(monkeypatch):
    rows = 8
    query = torch.empty(rows, 6, 256, dtype=torch.float16)
    key_cache = torch.empty(2, 400, 1, 256, dtype=torch.float16)
    value_cache = torch.empty_like(key_cache)
    indices = torch.empty(rows, 2051, dtype=torch.int32)
    page_table = torch.empty(1, 2, dtype=torch.int32)
    token_to_request = torch.zeros(rows, dtype=torch.int32)
    query_positions = torch.arange(rows, dtype=torch.int64)
    sequence_lengths = torch.full((1,), rows, dtype=torch.int32)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4", True)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4_MIN_ROWS", rows)
    monkeypatch.setattr(
        qsa_ops.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )

    args = (
        key_cache,
        value_cache,
        indices,
        page_table,
        token_to_request,
        query_positions,
        sequence_lengths,
    )
    assert qsa_ops._use_sm70_qsa_xqa_page4(query, *args)
    assert not qsa_ops._use_sm70_qsa_xqa_page4(
        query[:-1],
        key_cache,
        value_cache,
        indices[:-1],
        page_table,
        token_to_request[:-1],
        query_positions[:-1],
        sequence_lengths,
    )
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4", False)
    assert not qsa_ops._use_sm70_qsa_xqa_page4(query, *args)


def test_qsa_grouped_page4_modern_abi_forwards_quantized_kv_metadata():
    calls = []

    def forward(*args):
        calls.append(args)

    extension = SimpleNamespace(
        grouped_sparse_page4_abi_version=lambda: 2,
        grouped_sparse_page4_plan_fwd=lambda *args: None,
        grouped_sparse_page4_fwd=forward,
    )
    tensors = [torch.empty(0) for _ in range(8)]

    assert qsa_ops._qsa_grouped_page4_supported(extension, "auto")
    assert qsa_ops._qsa_grouped_page4_supported(extension, "fp8_e4m3")
    qsa_ops._qsa_grouped_page4_forward(
        extension,
        *tensors,
        0.0625,
        "fp8_e4m3",
        0.125,
        0.25,
    )

    assert len(calls) == 1
    assert len(calls[0]) == 12
    assert calls[0][-3:] == ("fp8_e4m3", 0.125, 0.25)


def test_qsa_grouped_page4_legacy_abi_is_fp16_only():
    calls = []

    def forward(*args):
        calls.append(args)

    forward.__doc__ = "grouped_sparse_page4_fwd(" + ", ".join(
        f"arg{index}: object" for index in range(9)
    )
    extension = SimpleNamespace(
        grouped_sparse_page4_plan_fwd=lambda *args: None,
        grouped_sparse_page4_fwd=forward,
    )
    tensors = [torch.empty(0) for _ in range(8)]

    assert qsa_ops._qsa_grouped_page4_abi_version(extension) == 1
    assert qsa_ops._qsa_grouped_page4_supported(extension, "auto")
    assert not qsa_ops._qsa_grouped_page4_supported(extension, "fp8_e4m3")
    qsa_ops._qsa_grouped_page4_forward(
        extension,
        *tensors,
        0.0625,
        "auto",
        1.0,
        1.0,
    )

    assert len(calls) == 1
    assert len(calls[0]) == 9


def test_qsa_e4m3_page4_routes_large_mixed_batch_below_prefill_boundary(
    monkeypatch,
):
    rows = 49
    query = torch.empty(rows, 6, 256, dtype=torch.float16)
    key_cache = torch.empty(2, 400, 1, 256, dtype=torch.uint8)
    value_cache = torch.empty_like(key_cache)
    indices = torch.empty(rows, 2051, dtype=torch.int32)
    page_table = torch.empty(4, 2, dtype=torch.int32)
    token_to_request = torch.zeros(rows, dtype=torch.int32)
    query_positions = torch.arange(rows, dtype=torch.int64)
    sequence_lengths = torch.full((4,), rows, dtype=torch.int32)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4", True)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4_MIN_ROWS", 4096)
    monkeypatch.setattr(
        qsa_ops.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )

    assert qsa_ops._use_sm70_qsa_xqa_page4(
        query,
        key_cache,
        value_cache,
        indices,
        page_table,
        token_to_request,
        query_positions,
        sequence_lengths,
    )


@pytest.mark.parametrize(
    ("rows", "kv_cache_dtype"),
    [(49, "fp8_e4m3"), (65, "auto")],
)
def test_qsa_xqa_page4_splits_non_grouped_large_batch(
    monkeypatch,
    rows,
    kv_cache_dtype,
):
    query = torch.empty(rows, 6, 256, dtype=torch.float16)
    flash_cuda = SimpleNamespace(
        decode_paged_xqa_fwd=object(),
        grouped_sparse_page4_abi_version=lambda: 2,
        grouped_sparse_page4_plan_fwd=lambda *args: None,
        grouped_sparse_page4_fwd=lambda *args: None,
    )
    flash_interface = ModuleType("flash_attn_v100.flash_attn_interface")
    cast(Any, flash_interface).flash_attn_v100_cuda = flash_cuda
    flash_package = ModuleType("flash_attn_v100")
    cast(Any, flash_package).flash_attn_interface = flash_interface
    monkeypatch.setitem(sys.modules, "flash_attn_v100", flash_package)
    monkeypatch.setitem(
        sys.modules,
        "flash_attn_v100.flash_attn_interface",
        flash_interface,
    )
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_GROUPED_PAGE4", True)

    calls = []

    def fake_grouped(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        out,
        *args,
    ):
        calls.append(
            (
                "grouped",
                q.shape[0],
                logical_indices.shape[0],
                token_to_req.shape[0],
                query_positions.shape[0],
                out.shape[0],
            )
        )
        return out

    def fake_xqa_batch(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        out,
        *args,
    ):
        calls.append(
            (
                "xqa",
                q.shape[0],
                logical_indices.shape[0],
                token_to_req.shape[0],
                query_positions.shape[0],
                out.shape[0],
            )
        )
        return out

    monkeypatch.setattr(
        qsa_ops,
        "_qsa_sparse_paged_attention_sm70_grouped_page4",
        fake_grouped,
    )
    monkeypatch.setattr(
        qsa_ops,
        "_qsa_sparse_paged_attention_sm70_xqa_page4_batch",
        fake_xqa_batch,
    )

    cache = torch.empty(0)
    logical_indices = torch.empty(rows, 2051, dtype=torch.int32)
    block_table = torch.empty(4, 1, dtype=torch.int32)
    token_to_req = torch.empty(rows, dtype=torch.int32)
    query_positions = torch.empty(rows, dtype=torch.int64)
    sequence_lengths = torch.empty(4, dtype=torch.int32)
    out = torch.empty_like(query)
    result = qsa_ops._qsa_sparse_paged_attention_sm70_xqa_page4(
        query,
        cache,
        cache,
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        out,
        kv_cache_dtype,
        0.05,
        0.05,
    )

    assert result is out
    grouped_rows = rows // 8 * 8
    assert calls == [
        (
            "grouped",
            grouped_rows,
            grouped_rows,
            grouped_rows,
            grouped_rows,
            grouped_rows,
        ),
        ("xqa", *(rows - grouped_rows,) * 5),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_xqa_page4_table_rejects_stale_or_invalid_tail_metadata():
    indices = torch.full((1, 2051), -1, dtype=torch.int32, device="cuda")
    indices[:, :2048] = torch.arange(2048, dtype=torch.int32, device="cuda")
    block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32, device="cuda")
    token_to_request = torch.zeros(1, dtype=torch.int32, device="cuda")
    query_positions = torch.tensor([3000], dtype=torch.int64, device="cuda")
    sequence_lengths = torch.tensor([2049], dtype=torch.int32, device="cuda")

    _, xqa_lengths = qsa_ops._qsa_xqa_page4_block_table(
        indices,
        block_table,
        token_to_request,
        query_positions,
        sequence_lengths,
        num_cache_blocks=3,
        page_size=784,
    )
    assert xqa_lengths.item() == 2048

    invalid_request = torch.full_like(token_to_request, -1)
    _, invalid_lengths = qsa_ops._qsa_xqa_page4_block_table(
        indices,
        block_table,
        invalid_request,
        query_positions,
        sequence_lengths,
        num_cache_blocks=3,
        page_size=784,
    )
    assert invalid_lengths.item() == 0

    indices[:, 2048] = 2048
    query_positions.fill_(2048)
    physical_pages, tail_lengths = qsa_ops._qsa_xqa_page4_block_table(
        indices,
        block_table,
        token_to_request,
        query_positions,
        sequence_lengths,
        num_cache_blocks=3,
        page_size=784,
    )
    assert tail_lengths.item() == 2049
    assert physical_pages[0, 512].item() == 316


def test_qsa_indexer_cublas_does_not_capture_decode_rows(monkeypatch):
    cache = torch.empty(2, 400, 1, 128, dtype=torch.float16)
    page_table = torch.empty(1, 2, dtype=torch.int32)
    monkeypatch.setattr(qsa_ops, "_SM70_INDEXER_CUBLAS", True)
    monkeypatch.setattr(qsa_ops, "_SM70_INDEXER_CUBLAS_MIN_ROWS", 256)
    monkeypatch.setattr(
        qsa_ops.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )

    assert qsa_ops._use_sm70_qsa_indexer_cublas(
        torch.empty(256, 4, 128, dtype=torch.float16), cache, page_table
    )
    assert not qsa_ops._use_sm70_qsa_indexer_cublas(
        torch.empty(255, 4, 128, dtype=torch.float16), cache, page_table
    )


def test_qsa_indexer_cublas_requires_enough_score_work(monkeypatch):
    monkeypatch.setattr(
        qsa_ops,
        "_SM70_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS",
        1024**2,
    )

    assert not qsa_ops._qsa_indexer_cublas_work_supported(1024, 512)
    assert qsa_ops._qsa_indexer_cublas_work_supported(2048, 512)


def test_qsa_lexicographic_topk_is_limited_to_sm70_qsa_shape(monkeypatch):
    monkeypatch.setattr(
        qsa_ops.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )

    assert _use_sm70_qsa_lexicographic_topk(512)
    assert not _use_sm70_qsa_lexicographic_topk(1024)

    monkeypatch.setattr(
        qsa_ops.current_platform,
        "is_device_capability",
        lambda capability: False,
    )
    assert not _use_sm70_qsa_lexicographic_topk(512)


def test_qsa_lexicographic_topk_prefers_validation_sidecar(monkeypatch):
    sidecar = object()
    wheel = object()
    monkeypatch.setattr(
        qsa_ops.torch,
        "ops",
        SimpleNamespace(
            _C_qsa_sm70=SimpleNamespace(qsa_lexicographic_topk=sidecar),
            _C=SimpleNamespace(qsa_lexicographic_topk=wheel),
        ),
    )

    assert _sm70_qsa_lexicographic_topk_op() is sidecar

    qsa_ops.torch.ops._C_qsa_sm70 = SimpleNamespace()
    assert _sm70_qsa_lexicographic_topk_op() is wheel
