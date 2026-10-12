# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared QSA mask/page/scale contract; not model quality qualification.

The ordinary attention bound remains atol=0.004, rtol=0.015. Native execution
is checked separately from dispatch and output comparisons.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.models.qwen4_exp.qsa_policy_utils import set_qsa_option
from vllm.models.qwen4_exp.nvidia.ops import qsa as ops
from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import FP8_E4M3, FP16, INT8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="QSA codec qualification on SM70",
)
CODECS = (FP16, FP8_E4M3, INT8)


def cache(codec, page=2048, padding=0):
    # Padded physical pages reserve a real final page as well. The borrowed
    # carrier describes capacity; the public view supplies every byte offset.
    width = 256 + (4 if codec.per_token_head else 0)
    pages = 2 * ((2051 + page - 1) // page) + 2
    carrier = torch.empty(
        (pages, 2, page, 1, width), device="cuda", dtype=codec.storage_dtype
    )
    if padding:
        stride = carrier.stride(0) + padding
        storage = torch.empty(pages * stride, device="cuda", dtype=carrier.dtype)
        carrier = storage.as_strided(carrier.shape, (stride, *carrier.stride()[1:]))
    view = KVCacheView.bind(carrier, codec, 256)
    for data, scales in ((view.key, view.key_scale), (view.value, view.value_scale)):
        values = torch.randn(data.shape, device="cuda", dtype=torch.float16) * 0.3
        if codec.per_token_head:
            values *= torch.linspace(0.03, 3.0, page, device="cuda")[
                None, :, None, None
            ]
            raw, scale = codec.encode(values)
            data.copy_(raw)
            scales.copy_(scale)
        elif codec.quantized:
            data.copy_(
                (values / 0.01).to(codec.element_dtype).view(codec.storage_dtype)
            )
        else:
            data.copy_(values)
    return carrier, view


def reference(query, view, table, requests, indices, positions, lengths):
    outputs = []
    page = view.key.shape[1]
    for row in range(query.shape[0]):
        req = int(requests[row])
        tokens = indices[row].long()
        tokens = tokens[
            (tokens >= 0) & (tokens <= positions[row]) & (tokens < lengths[req])
        ]
        if not tokens.numel():
            outputs.append(torch.zeros_like(query[row], dtype=torch.float64))
            continue
        block, offset = table[req, tokens // page].long(), tokens % page
        decoded = []
        for data, scales in (
            (view.key, view.key_scale),
            (view.value, view.value_scale),
        ):
            raw = data[block, offset, 0]
            if view.codec.element_dtype is not None:
                raw = raw.view(view.codec.element_dtype)
            values = raw.double()
            values *= (
                scales[block, offset, 0, None].double()
                if scales is not None
                else (0.01 if view.codec.quantized else 1.0)
            )
            decoded.append(values)
        keys, values = decoded
        logits = query[row].double() @ keys.T * (256**-0.5)
        outputs.append(logits.softmax(-1) @ values)
    return torch.stack(outputs)


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("native,rows", [(False, 3), (True, 1), (True, 8), (True, 17)])
@pytest.mark.parametrize("padding", [0, 1024])
@pytest.mark.parametrize("page", [64, 2048])
def test_qsa_shared_codecs_pages_tail_and_changed_graph_inputs(
    monkeypatch, codec, native, rows, padding, page
):
    torch.manual_seed(20261012)
    set_qsa_option(monkeypatch, "qsa_xqa_page4", native)
    set_qsa_option(monkeypatch, "qsa_xqa_page4_min_rows", 1)
    set_qsa_option(monkeypatch, "qsa_grouped_page4", native)
    _, view = cache(codec, page=page, padding=padding)
    query = torch.randn((rows, 6, 256), device="cuda", dtype=torch.float16) * 0.2
    indices = torch.arange(2051, device="cuda", dtype=torch.int32).repeat(rows, 1)
    # Non-contiguous physical pages and multiple requests share one cache.
    blocks = (2051 + page - 1) // page
    table = (
        torch.randperm(2 * blocks + 2, device="cuda")[: 2 * blocks]
        .reshape(2, blocks)
        .int()
    )
    requests = torch.arange(rows, device="cuda", dtype=torch.int32) % 2
    positions = torch.full((rows,), 2050, device="cuda", dtype=torch.int64)
    lengths = torch.full((2,), 2051, device="cuda", dtype=torch.int32)
    output = torch.empty_like(query)

    def launch():
        return ops.qsa_sparse_paged_attention(
            query,
            view.key,
            view.value,
            indices,
            table,
            requests,
            out=output,
            query_positions=positions,
            sequence_lengths=lengths,
            kv_cache_dtype=codec.name,
            selection_group_size=4,
            k_scale=0.01,
            v_scale=0.01,
            **view.scale_kwargs,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for boundary in (2050, 2047, 2048):
        positions.fill_(boundary)
        table.copy_(table.flip(1))
        query.mul_(0.99)
        graph.replay()
        torch.accelerator.synchronize()
        expected = reference(query, view, table, requests, indices, positions, lengths)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.double(), expected, atol=0.004, rtol=0.015)


def test_qsa_int8_native_execution_and_main_writer(monkeypatch):
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionImpl

    native = pytest.importorskip(
        "flash_attn_v100.flash_attn_interface"
    ).flash_attn_v100_cuda
    assert native.decode_paged_xqa_kv_view_abi >= 1
    assert native.grouped_sparse_page4_kv_abi >= 1
    carrier, view = cache(INT8)
    impl = object.__new__(Qwen4ExpQSAFlashAttentionImpl)
    impl.kv_cache_dtype, impl.head_size = "int8", 256
    values = torch.randn((8, 1, 256), device="cuda", dtype=torch.float16)
    slots = torch.tensor(
        [3 * 2048 + i for i in range(8)], device="cuda", dtype=torch.int64
    )
    scale = torch.ones((), device="cuda")
    impl.do_kv_cache_update(
        SimpleNamespace(_k_scale=scale, _v_scale=scale), values, values, carrier, slots
    )
    raw, encoded_scale = INT8.encode(values)
    assert torch.equal(view.key[3, :8], raw)
    assert torch.equal(view.key_scale[3, :8], encoded_scale)
    assert torch.equal(view.value[3, :8], raw)
    assert torch.equal(view.value_scale[3, :8], encoded_scale)
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )

    triton_carrier = torch.zeros_like(carrier)
    triton_view = KVCacheView.bind(triton_carrier, INT8, 256)
    triton_reshape_and_cache_flash_per_token_head_quant(
        values,
        values,
        triton_view.key,
        triton_view.value,
        triton_view.key_scale,
        triton_view.value_scale,
        slots,
        round_int8_even=True,
    )
    for native_data, triton_data in (
        (view.key, triton_view.key),
        (view.value, triton_view.value),
        (view.key_scale, triton_view.key_scale),
        (view.value_scale, triton_view.value_scale),
    ):
        assert torch.equal(native_data[3, :8], triton_data[3, :8])

    query = torch.randn((8, 6, 256), device="cuda", dtype=torch.float16)
    indices = torch.arange(2051, device="cuda", dtype=torch.int32).repeat(8, 1)
    table = torch.tensor([[3, 4]], device="cuda", dtype=torch.int32)
    requests = torch.zeros(8, device="cuda", dtype=torch.int32)
    positions = torch.full((8,), 2050, device="cuda", dtype=torch.int64)
    lengths = torch.full((1,), 2051, device="cuda", dtype=torch.int32)
    set_qsa_option(monkeypatch, "qsa_xqa_page4", True)
    set_qsa_option(monkeypatch, "qsa_xqa_page4_min_rows", 1)
    for grouped in (False, True):
        set_qsa_option(monkeypatch, "qsa_grouped_page4", grouped)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            ops.qsa_sparse_paged_attention(
                query,
                view.key,
                view.value,
                indices,
                table,
                requests,
                query_positions=positions,
                sequence_lengths=lengths,
                kv_cache_dtype="int8",
                selection_group_size=4,
                **view.scale_kwargs,
            )
            torch.accelerator.synchronize()
        names = [
            event.name
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        family = "grouped_verify_e5m2_partial_kernel" if grouped else "xqa"
        assert any(
            family in name and ("3," in name or "3>" in name) for name in names
        ), names


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("algorithm", ["triton", "grouped", "xqa_incompatible"])
def test_qsa_sparse_mask_empty_rows_and_unselected_nonfinite_pages(
    monkeypatch, codec, algorithm
):
    torch.manual_seed(20261012)
    _, view = cache(codec)
    if codec.per_token_head:
        view.key_scale[0].fill_(torch.nan)
        view.value_scale[0].fill_(torch.nan)
    elif codec.quantized:
        view.key[0].fill_(0x7F)
        view.value[0].fill_(0x7F)
    else:
        view.key[0].fill_(torch.nan)
        view.value[0].fill_(torch.nan)
    query = torch.randn((8, 6, 256), device="cuda", dtype=torch.float16) * 0.2
    indices = torch.arange(2051, device="cuda", dtype=torch.int32).repeat(8, 1)
    indices[:, ::7] = -1
    indices[0].fill_(-1)
    indices[3].fill_(-1)
    table = torch.tensor([[3, 4]], device="cuda", dtype=torch.int32)
    requests = torch.zeros(8, device="cuda", dtype=torch.int32)
    positions = torch.full((8,), 2050, device="cuda", dtype=torch.int64)
    lengths = torch.full((1,), 2051, device="cuda", dtype=torch.int32)
    set_qsa_option(monkeypatch, "qsa_xqa_page4", algorithm != "triton")
    set_qsa_option(monkeypatch, "qsa_xqa_page4_min_rows", 1)
    set_qsa_option(monkeypatch, "qsa_grouped_page4", algorithm == "grouped")
    if algorithm == "xqa_incompatible":

        def reject_xqa(*args, **kwargs):
            pytest.fail("Arbitrary token masks entered complete-page XQA")

        monkeypatch.setattr(
            ops, "_qsa_sparse_paged_attention_sm70_xqa_page4_batch", reject_xqa
        )
    output = ops.qsa_sparse_paged_attention(
        query,
        view.key,
        view.value,
        indices,
        table,
        requests,
        query_positions=positions,
        sequence_lengths=lengths,
        kv_cache_dtype=codec.name,
        k_scale=0.01,
        v_scale=0.01,
        **view.scale_kwargs,
    )
    expected = reference(query, view, table, requests, indices, positions, lengths)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[[0, 3]]) == 0
    torch.testing.assert_close(output.double(), expected, atol=0.004, rtol=0.015)


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("rows", [1, 8])
def test_qsa_xqa_complete_page_selections_count_only_valid_pages(
    monkeypatch, codec, rows
):
    torch.manual_seed(20261012)
    _, view = cache(codec)
    query = torch.randn((rows, 6, 256), device="cuda", dtype=torch.float16) * 0.2
    indices = torch.arange(2051, device="cuda", dtype=torch.int32).repeat(rows, 1)
    indices[:, 28:32] = -1
    indices[0].fill_(-1)
    table = torch.tensor([[3, 4]], device="cuda", dtype=torch.int32)
    requests = torch.zeros(rows, device="cuda", dtype=torch.int32)
    positions = torch.full((rows,), 2050, device="cuda", dtype=torch.int64)
    lengths = torch.full((1,), 2051, device="cuda", dtype=torch.int32)
    set_qsa_option(monkeypatch, "qsa_xqa_page4", True)
    set_qsa_option(monkeypatch, "qsa_xqa_page4_min_rows", 1)
    set_qsa_option(monkeypatch, "qsa_grouped_page4", False)
    output = ops.qsa_sparse_paged_attention(
        query,
        view.key,
        view.value,
        indices,
        table,
        requests,
        query_positions=positions,
        sequence_lengths=lengths,
        kv_cache_dtype=codec.name,
        selection_group_size=4,
        k_scale=0.01,
        v_scale=0.01,
        **view.scale_kwargs,
    )
    expected = reference(query, view, table, requests, indices, positions, lengths)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[0]) == 0
    torch.testing.assert_close(output.double(), expected, atol=0.004, rtol=0.015)
