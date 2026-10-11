# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared grouped algorithm: codec bindings, causal rows and graph state.

The arithmetic bound is frozen before candidate execution: the existing
grouped FP32 relative-L2 bound of 1.02 times the FP16 output rounding floor
plus 2e-6. This compares the same quantized cache, not model task quality.
"""

import pytest
import torch

from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3,
    FP8_E5M2,
    INT8,
    INT8_PER_TOKEN_HEAD,
    KV_CODECS,
)

package = pytest.importorskip("flash_attn_v100")
operation = getattr(package, "flash_attn_grouped_fp32_paged", None)
codecs = tuple(
    codec
    for codec in KV_CODECS
    if codec.name in getattr(operation, "kv_cache_codecs", ())
)


@pytest.fixture(autouse=True)
def sm70():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")


def test_required_bindings_are_in_the_installed_artifact():
    assert operation is not None, "Rebuild Flash-V100 for grouped codec ABI 1"
    assert {"auto", "fp8_e4m3", "fp8_e5m2", "int8", "int8_per_token_head"} <= {
        codec.name for codec in codecs
    }


def _view(codec, pages, page, heads):
    carrier = torch.zeros(
        (pages, 2, page, heads, 260 if codec.per_token_head else 256),
        dtype=codec.storage_dtype,
        device="cuda",
    )
    return KVCacheView.bind(carrier, codec, 256)


def _reference(q, view, table, lengths):
    results = []
    per_request = q.shape[0] // table.shape[0]
    for row, length in enumerate(lengths.cpu().tolist()):
        if length == 0:
            results.append(torch.zeros_like(q[row], device="cpu", dtype=torch.float64))
            continue
        tokens = torch.arange(length, device="cuda")
        pages = table[row // per_request, tokens // view.key.shape[1]].long()
        offsets = tokens % view.key.shape[1]
        values = []
        for payload, scales in (
            (view.key, view.key_scale),
            (view.value, view.value_scale),
        ):
            raw = payload[pages, offsets].cpu()
            if view.codec.element_dtype is not None:
                raw = raw.view(view.codec.element_dtype)
            decoded = raw.double()
            if scales is not None:
                decoded *= scales[pages, offsets].cpu().double()[..., None]
            values.append(decoded.repeat_interleave(6, 1))
        k, v = values
        logits = torch.einsum("hd,thd->ht", q[row].cpu().double(), k) * 0.0625
        results.append(torch.einsum("ht,thd->hd", logits.softmax(-1), v))
    return torch.stack(results)


def _check(actual, expected):
    assert torch.isfinite(actual).all()
    actual = actual.cpu().double()
    zero = expected.abs().sum(dim=(1, 2)) == 0
    assert torch.equal(actual[zero], expected[zero])
    norm = expected.norm()
    if norm > 0:
        error = (actual - expected).norm() / norm
        floor = (expected.half().double() - expected).norm() / norm
        assert error <= 1.02 * floor + 2e-6, (float(error), float(floor))


@pytest.mark.parametrize("codec", codecs, ids=lambda codec: codec.name)
@pytest.mark.parametrize(
    "rows,batch,heads,page,length",
    [
        (2, 1, 1, 16, 17),
        (5, 1, 2, 16, 255),
        (8, 1, 2, 32, 257),
        (8, 2, 1, 32, 1025),
        (8, 16, 1, 16, 129),
        (8, 1, 1, 16, 4097),
    ],
)
def test_grouped_codecs_share_masks_pages_and_graph_replay(
    codec, rows, batch, heads, page, length
):
    generator = torch.Generator().manual_seed(20261011)
    blocks = (length + page - 1) // page
    view = _view(codec, batch * blocks + 2, page, heads)
    order = (torch.randperm(batch * blocks, generator=generator) + 2).reshape(batch, -1)
    table = order.cuda().int()
    tokens = torch.arange(length, device="cuda")
    positions = (
        table[:, tokens // page].long().flatten(),
        (tokens % page).repeat(batch),
    )
    for payload, scales in (
        (view.key, view.key_scale),
        (view.value, view.value_scale),
    ):
        values = torch.randn(batch * length, heads, 256, generator=generator).half()
        if scales is not None:
            # Independently varying K/V scales, including very small rows.
            values *= torch.logspace(-3, 1, values.shape[0])[:, None, None]
            encoded, scale = codec.encode(values)
            scales[positions] = scale.cuda()
        elif codec.element_dtype is not None:
            encoded = values.to(codec.element_dtype).view(codec.storage_dtype)
        else:
            encoded = values
        payload[positions] = encoded.cuda()
    q = torch.randn(batch * rows, heads * 6, 256, generator=generator).half().cuda()
    lengths = torch.arange(length - rows + 1, length + 1, dtype=torch.int32).repeat(
        batch
    )
    lengths = lengths.cuda()
    initial = lengths.clone()
    out = torch.empty_like(q)
    kwargs = dict(
        out=out,
        softmax_scale=0.0625,
        kv_cache_dtype=codec.name,
        k_scale_cache=view.key_scale,
        v_scale_cache=view.value_scale,
    )

    def run():
        assert operation is not None
        operation(q, view.key, view.value, table, lengths, **kwargs)

    run()
    _check(out, _reference(q, view, table, lengths))
    if length == 257:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as trace:
            run()
            torch.accelerator.synchronize()
        kernels = [
            event.name
            for event in trace.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert any("grouped_verify_e5m2_partial_kernel" in name for name in kernels)
        assert not any("decode_partition_kernel" in name for name in kernels)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for state in ("zero", "reorder", "restore"):
        lengths.copy_(initial)
        if state == "zero":
            lengths[::rows] = 0
            lengths[-1] = 0
        elif state == "reorder":
            table.copy_(order.flip(0))
            q.mul_(0.5)
            if view.key_scale is not None:
                view.key_scale.mul_(0.875)
                view.value_scale.mul_(1.125)
        else:
            table.copy_(order)
        graph.replay()
        _check(out, _reference(q, view, table, lengths))


@pytest.mark.parametrize("codec", [INT8, INT8_PER_TOKEN_HEAD], ids=lambda c: c.name)
@pytest.mark.parametrize("invalid_key", [False, True])
def test_invalid_scales_respect_causal_rows_requests_and_kv_heads(codec, invalid_key):
    view = _view(codec, 4, 16, 2)
    view.key.fill_(1)
    view.value.fill_(1)
    view.key_scale.fill_(0.125)
    view.value_scale.fill_(0.25)
    scale = view.key_scale if invalid_key else view.value_scale
    scale[2, 3, 1] = float("nan")
    q = torch.ones(16, 12, 256, device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    table = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor(
        [1, 2, 3, 4, 8, 16, 0, 17] * 2, device="cuda", dtype=torch.int32
    )
    assert operation is not None
    operation(
        q,
        view.key,
        view.value,
        table,
        lengths,
        out=out,
        softmax_scale=0.0625,
        kv_cache_dtype=codec.name,
        k_scale_cache=view.key_scale,
        v_scale_cache=view.value_scale,
    )
    expected_invalid = torch.zeros(16, 12, device="cuda", dtype=torch.bool)
    expected_invalid[8:, 6:] = (lengths[8:] > 3)[:, None]
    assert torch.equal(torch.isnan(out).all(-1), expected_invalid)
    expected = torch.full_like(out, 0.25)
    expected[lengths == 0] = 0
    assert torch.equal(out[~expected_invalid], expected[~expected_invalid])


@pytest.mark.parametrize("codec", [FP8_E4M3, FP8_E5M2], ids=lambda c: c.name)
def test_eight_byte_token_and_head_strides_use_the_qualified_reader(codec):
    assert operation is not None
    generator = torch.Generator().manual_seed(20261011)
    shape = (5, 16, 2, 256)
    strides = (16 * 2 * 264 + 8, 2 * 264, 264, 1)
    caches = []
    for _ in range(2):
        raw = torch.randn(shape, generator=generator).to(codec.element_dtype)
        cache = torch.empty_strided(shape, strides, device="cuda", dtype=torch.uint8)
        cache.copy_(raw.view(torch.uint8))
        caches.append(cache)
    q = torch.randn(5, 12, 256, generator=generator).half().cuda()
    table = torch.tensor([[4, 0, 2, 1, 3]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([65, 0, 67, 68, 69], device="cuda", dtype=torch.int32)
    reference, actual = torch.empty_like(q), torch.empty_like(q)
    kwargs = dict(softmax_scale=0.0625, kv_cache_dtype=codec.name)
    operation(
        q,
        *(cache.contiguous() for cache in caches),
        table,
        lengths,
        out=reference,
        **kwargs,
    )
    operation(q, *caches, table, lengths, out=actual, **kwargs)
    assert torch.equal(actual, reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation(q, *caches, table, lengths, out=actual, **kwargs)
    lengths[1] = 66
    operation(
        q,
        *(cache.contiguous() for cache in caches),
        table,
        lengths,
        out=reference,
        **kwargs,
    )
    graph.replay()
    assert torch.equal(actual, reference)


def test_bound_grouped_owners_keep_different_codecs_and_captures_independent():
    from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda
    from flash_attn_v100.runtime import AttentionRuntime, PythonPolicy

    assert operation is not None
    policy = PythonPolicy(
        dynamic_partitions=False,
        staged_pv=False,
        share_workspace=True,
        partition_size=None,
        partition_error=None,
        scalar_fast=False,
        batch_xqa=False,
        padded_smem=False,
        dual_cta=False,
    )
    owners = [
        AttentionRuntime(policy, [None] * flash_attn_v100_cuda.policy_fields)
        for _ in range(2)
    ]
    views = [_view(codec, 2, 16, 1) for codec in (INT8, FP8_E4M3)]
    views[0].key.fill_(1)
    views[0].value.fill_(1)
    views[0].key_scale.fill_(0.125)
    views[0].value_scale.fill_(0.25)
    for payload in (views[1].key, views[1].value):
        payload.fill_(
            torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8).item()
        )
    q = torch.ones(5, 6, 256, device="cuda", dtype=torch.float16)
    table = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 1, 15, 16, 17], device="cuda", dtype=torch.int32)
    outputs = [torch.empty_like(q) for _ in owners]
    graphs = [torch.cuda.CUDAGraph() for _ in owners]
    for owner, view, output, graph in zip(owners, views, outputs, graphs):
        bound = owner.bind(operation)
        kwargs = dict(
            out=output,
            softmax_scale=0.0625,
            kv_cache_dtype=view.codec.name,
            k_scale_cache=view.key_scale,
            v_scale_cache=view.value_scale,
        )
        bound(q, view.key, view.value, table, lengths, **kwargs)
        with torch.cuda.graph(graph):
            bound(q, view.key, view.value, table, lengths, **kwargs)
    assert owners[0].grouped_verify_workspace_cache is not (
        owners[1].grouped_verify_workspace_cache
    )
    lengths.copy_(torch.tensor([17, 0, 2, 15, 1], device="cuda", dtype=torch.int32))
    views[0].value_scale.mul_(2)
    for graph in graphs:
        graph.replay()
    for actual, value in zip(outputs, (0.5, 1.0)):
        expected = torch.full_like(actual, value)
        expected[lengths == 0] = 0
        assert torch.equal(actual, expected)
    graphs[0].reset()
    owners[0].close()
    views[1].value.fill_(
        torch.tensor(2.0).to(torch.float8_e4m3fn).view(torch.uint8).item()
    )
    graphs[1].replay()
    expected = torch.full_like(outputs[1], 2.0)
    expected[lengths == 0] = 0
    assert torch.equal(outputs[1], expected)
    graphs[1].reset()
    owners[1].close()


@pytest.mark.parametrize("codec", codecs, ids=lambda c: c.name)
def test_grouped_capacity_boundary_against_fp64(codec):
    """The declared 266240-token envelope, its neighbor and a padded row."""
    assert operation is not None
    torch.manual_seed(20261011)
    capacity, page, rows = 266240, 16, 8
    blocks = capacity // page
    view = _view(codec, blocks + 2, page, 1)
    order = torch.randperm(blocks, device="cuda") + 2
    decoded = []
    for payload, scales in (
        (view.key, view.key_scale),
        (view.value, view.value_scale),
    ):
        values = torch.randn(capacity, 1, 256, device="cuda", dtype=torch.float16)
        if scales is not None:
            encoded, scale = codec.encode(values)
            scales[order] = scale.reshape(blocks, page, 1)
            reference = encoded.double() * scale.double()[..., None]
        elif codec.element_dtype is not None:
            encoded = values.to(codec.element_dtype)
            reference = encoded.double()
            encoded = encoded.view(codec.storage_dtype)
        else:
            encoded, reference = values, values.double()
        payload[order] = encoded.reshape(blocks, page, 1, 256)
        decoded.append(reference[:, 0])
    q = torch.randn(rows, 6, 256, device="cuda", dtype=torch.float16)
    table = order.int()[None]
    lengths = torch.arange(
        capacity - rows + 1, capacity + 1, device="cuda", dtype=torch.int32
    )
    lengths[0] = 0
    out = torch.empty_like(q)
    operation(
        q,
        view.key,
        view.value,
        table,
        lengths,
        out=out,
        softmax_scale=0.0625,
        kv_cache_dtype=codec.name,
        k_scale_cache=view.key_scale,
        v_scale_cache=view.value_scale,
    )
    key, value = decoded
    score = q.transpose(0, 1).double() @ key.T * 0.0625
    mask = torch.arange(capacity, device="cuda")[None] >= lengths[:, None]
    score.masked_fill_(mask[None], -torch.inf)
    expected = (score.softmax(-1).nan_to_num(0.0) @ value).transpose(0, 1)
    _check(out, expected.cpu())
